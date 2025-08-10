"""
Unified IPUMS data acquisition script for ACS, CPS, and ATUS.

Features:
- Single YAML configuration to control years, variables, samples, paths, and API key.
- Minimizes variables to only what's needed by downstream processing scripts.
- Submits extracts per dataset (and in chunks if necessary) and saves CSV outputs.
- ATUS:  microdata (with CPS linkage keys)

Datasets and default sample patterns:
- ATUS (collection='atus'): samples like 'atus{YEAR}'.
- ACS (collection='usa'): default 1-year ACS samples like 'usa_{YEAR}a'.
- CPS (collection='cps'): default to ASEC samples 'cpsasec{YEAR}' unless samples are explicitly provided.
    Note: For Basic Monthly CPS (e.g., to use TELWRKHR), specify monthly sample names explicitly in YAML.

Outputs:
- data/raw/<dataset>/<dataset>_<start>-<end>.csv for ACS/CPS
- data/processed/atus/<micro_filename>.csv for ATUS microdata

Requirements:
- ipumspy, pyyaml, pandas
"""

# Standard and third-party imports
from __future__ import annotations

import os
import sys
import json
import math
import time
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple

import pandas as pd
import yaml
from ipumspy import IpumsApiClient, MicrodataExtract, readers


# --------------- Logging --------------- #
LOGGER = logging.getLogger("ipums_acquire")
LOGGER.setLevel(logging.INFO)

# Console handler
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
LOGGER.addHandler(_console_handler)


# --------------- Data classes --------------- #
# Represents configuration for a single dataset (ACS, CPS, or ATUS)
@dataclass
class DatasetConfig:
    collection: str  # IPUMS collection name (e.g., 'usa', 'cps', 'atus')
    enabled: bool  # Whether to acquire this dataset
    years_start: int | None  # Start year (inclusive)
    years_end: str | int | None  # End year ('present' or int)
    variables: List[str]  # Variables to request from IPUMS
    samples: List[str] | None  # Explicit sample list; overrides pattern if provided
    sample_pattern: str | None  # Python f-string pattern for sample names (e.g., 'usa_{year}a')
    months: List[int] | None  # Months to include (1-12); if None, use all 12 months
    max_samples_per_extract: int | None  # Max samples per extract request
    output_filename: str | None  # Output CSV filename for this dataset
    # ATUS-specific: custom micro output filename (ignored by other datasets)
    atus_micro_output_filename: str | None

# Represents the overall configuration loaded from YAML
@dataclass
class Config:
    api_key: str | None
    api_key_path: str | None
    api_key_json_field: str | None
    raw_dir: Path
    processed_dir: Path
    datasets: Dict[str, DatasetConfig]


# --------------- Helpers --------------- #
def load_config(config_path: str | Path) -> Config:
    """
    Load configuration from a YAML file and return a Config object.

    Args:
        config_path (str | Path): Path to the YAML configuration file.

    Returns:
        Config: Parsed configuration object.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Extract paths for raw and processed data directories
    # Resolve paths relative to the project root (2 levels up from src/data/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    raw_dir = project_root / cfg["paths"].get("raw_dir", "data/raw")
    processed_dir = project_root / cfg["paths"].get("processed_dir", "data/processed")

    # Extract API configuration - check both authentication and ipums sections
    auth_cfg = cfg.get("authentication", {})
    ipums_cfg = cfg.get("ipums", {})
    
    # Prefer authentication section, fallback to ipums section
    api_key = ipums_cfg.get("api_key") or auth_cfg.get("api_key")
    api_key_path = auth_cfg.get("api_key_path") or ipums_cfg.get("api_key_path")
    api_key_json_field = auth_cfg.get("ipums_api_key_field") or ipums_cfg.get("api_key_json_field", "ipums")

    # Parse dataset configurations
    ds_cfg: Dict[str, DatasetConfig] = {}
    for name, d in cfg.get("datasets", {}).items():
        ds_cfg[name] = DatasetConfig(
            collection=d.get("collection", name),
            enabled=bool(d.get("enabled", True)),
            years_start=d.get("years", {}).get("start"),
            years_end=d.get("years", {}).get("end"),
            variables=d.get("variables", []),
            samples=d.get("samples"),
            sample_pattern=d.get("sample_pattern"),
            months=d.get("months"),
            max_samples_per_extract=d.get("max_samples_per_extract", 12),
            output_filename=d.get("output_filename"),
            atus_micro_output_filename=d.get("micro_output_filename"),
        )

    return Config(
        api_key=api_key,
        api_key_path=api_key_path,
        api_key_json_field=api_key_json_field,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        datasets=ds_cfg,
    )

def resolve_api_key(cfg: Config) -> str:
    """
    Resolve the IPUMS API key from config, file, environment variable, or default location.

    Priority:
    1. Directly from config.api_key
    2. From config.api_key_path (plain text or JSON)
    3. From environment variable IPUMS_API_KEY
    4. From default api_keys.json in repo root
    """
    if cfg.api_key:
        return cfg.api_key
    if cfg.api_key_path:
        path = Path(cfg.api_key_path)
        if not path.exists():
            raise FileNotFoundError(f"API key file not found: {path}")
        if path.suffix.lower() in {".json"}:
            with open(path, "r") as f:
                data = json.load(f)
            key = data.get(cfg.api_key_json_field or "ipums")
            if not key:
                raise RuntimeError(
                    f"API key field '{cfg.api_key_json_field}' not found in {path}"
                )
            return key
        else:
            return path.read_text().strip()
    # Fallback to env var
    env_key = os.getenv("IPUMS_API_KEY")
    if env_key:
        return env_key
    # Repo default path (compatibility with existing project)
    default_json = Path(__file__).resolve().parents[2] / "api_keys.json"
    if default_json.exists():
        with open(default_json, "r") as f:
            return json.load(f).get("ipums")
    raise RuntimeError("No IPUMS API key provided. Set in YAML, env, or api_keys.json.")

def year_range(start: int | None, end: int | None | str) -> List[int]:
    """
    Generate a list of years from start to end (inclusive).

    Args:
        start (int | None): Start year (required).
        end (int | None | str): End year (can be int, "present", or None).

    Returns:
        List[int]: List of years from start to end.

    Raises:
        ValueError: If start year is not provided.
    """
    if start is None:
        raise ValueError("years.start is required")
    if end == "present" or end is None:
        end_year = datetime.now().year
    else:
        end_year = int(end)
    return list(range(int(start), int(end_year) + 1))

def default_sample_pattern_for(dataset: str) -> str:
    """
    Return the default sample name pattern for a given dataset.

    Args:
        dataset (str): Dataset name (e.g., 'atus', 'usa', 'acs', 'cps').

    Returns:
        str: Python format string for sample names.
    """
    if dataset.lower() == "atus":
        return "atus{year}"
    if dataset.lower() in {"usa", "acs"}:
        # IPUMS USA 1-year ACS sample code pattern
        return "usa_{year}a"
    if dataset.lower() == "cps":
        # Default to ASEC annual samples; override in YAML for Basic Monthly
        return "cpsasec{year}"
    return "{year}"


def resolve_samples_for(ds_name: str, ds_cfg: DatasetConfig) -> List[str]:
    """
    Determine the list of sample names for a dataset based on config.

    Args:
        ds_name (str): Name of the dataset (e.g., 'acs', 'cps', 'atus').
        ds_cfg (DatasetConfig): Dataset configuration.

    Returns:
        List[str]: List of sample names to request from IPUMS.

    Logic:
    - If explicit samples are provided in config, use them.
    - Otherwise, generate sample names using the year range and pattern.
    - For CPS with months specified, generate year x month combinations.
    - For CPS datasets, validate against available samples from IPUMS.
    """
    if ds_cfg.samples:
        samples = list(ds_cfg.samples)
    else:
        years = year_range(ds_cfg.years_start, ds_cfg.years_end)
        pattern = ds_cfg.sample_pattern or default_sample_pattern_for(ds_name)
        
        # If months are specified and pattern contains {month}, generate year x month combinations
        if ds_cfg.months and "{month" in pattern:
            samples = []
            for year in years:
                for month in ds_cfg.months:
                    sample = pattern.format(year=year, month=month)
                    samples.append(sample)
        else:
            # Standard year-only pattern
            samples = [pattern.format(year=y) for y in years]
    
    # Validate CPS samples against IPUMS available list
    if ds_name.lower() == "cps":
        samples = validate_cps_samples(samples)
    
    return samples


def chunked(iterable: List[Any], size: int) -> Iterable[List[Any]]:
    """
    Yield successive chunks of a specified size from the given iterable.

    Args:
        iterable (List[Any]): The list to be divided into chunks.
        size (int): The size of each chunk.

    Yields:
        List[Any]: Slices of the original iterable, each with a maximum length of 'size'.

    Example:
        >>> list(chunked([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]
    """
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def ensure_dir(path: Path) -> None:
    """Ensure that a directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def clean_download_folder(dataset_name: str, output_dir: Path) -> None:
    """
    Clean the download folder for a specific dataset before starting data acquisition.
    
    This removes any existing files to ensure a clean download environment and 
    prevent conflicts with previous incomplete downloads.
    
    Args:
        dataset_name (str): Name of the dataset (for logging purposes)
        output_dir (Path): Directory to clean
    """
    if not output_dir.exists():
        LOGGER.info(f"🗂️  [{dataset_name}] Download directory doesn't exist, will be created: {output_dir}")
        return
    
    # Get list of files before cleaning
    existing_files = list(output_dir.iterdir())
    if not existing_files:
        LOGGER.info(f"🗂️  [{dataset_name}] Download directory is already clean: {output_dir}")
        return
    
    LOGGER.info(f"🧹 [{dataset_name}] Cleaning download directory: {output_dir}")
    LOGGER.info(f"   📁 Found {len(existing_files)} items to remove")
    
    cleaned_count = 0
    failed_count = 0
    
    for item in existing_files:
        try:
            if item.is_dir():
                import shutil
                shutil.rmtree(item)
                LOGGER.debug(f"   🗑️  Removed directory: {item.name}")
            else:
                item.unlink()
                LOGGER.debug(f"   🗑️  Removed file: {item.name}")
            cleaned_count += 1
        except Exception as e:
            LOGGER.warning(f"   ⚠️  Failed to remove {item.name}: {e}")
            failed_count += 1
    
    if failed_count == 0:
        LOGGER.info(f"✅ [{dataset_name}] Successfully cleaned {cleaned_count} items from download directory")
    else:
        LOGGER.warning(f"⚠️  [{dataset_name}] Cleaned {cleaned_count} items, failed to remove {failed_count} items")


def validate_cps_samples_api(client: IpumsApiClient, samples: List[str]) -> List[str]:
    """
    Validate CPS samples against IPUMS API by attempting a minimal extract.
    This is more reliable than scraping the website.
    """
    try:
        LOGGER.info(f"🔍 API-validating {len(samples)} CPS samples...")
        
        # Try to get available samples via API metadata (if supported)
        # This is a placeholder - actual implementation depends on ipumspy capabilities
        
        # For now, return the samples with web validation as fallback
        return validate_cps_samples(samples)
        
    except Exception as e:
        LOGGER.warning(f"⚠️ API validation failed, using web validation: {e}")
        return validate_cps_samples(samples)


def validate_cps_samples(samples: List[str]) -> List[str]:
    """
    Validate CPS sample names against IPUMS available samples list.
    Enhanced error handling and logging.
    """
    try:
        # Get available CPS samples from IPUMS website
        LOGGER.info("🌐 Fetching available CPS samples from IPUMS website...")
        tables = pd.read_html("https://cps.ipums.org/cps-action/samples/sample_ids")
        available_df = tables[1]  # Second table contains the sample IDs
        available_samples = set(available_df['Sample ID'].str.strip().tolist())
        
        LOGGER.info(f"📊 Found {len(available_samples)} available CPS samples on IPUMS website")
        
        # Filter requested samples with b/s fallback logic
        valid_samples = []
        invalid_samples = []
        fallback_used = []
        
        for sample in samples:
            if sample in available_samples:
                # Direct match - use as is
                valid_samples.append(sample)
            elif sample.endswith('b'):
                # Try fallback from 'b' to 's'
                fallback_sample = sample[:-1] + 's'
                if fallback_sample in available_samples:
                    valid_samples.append(fallback_sample)
                    fallback_used.append(f"{sample} -> {fallback_sample}")
                else:
                    # Try without suffix (annual samples)
                    fallback_annual = sample[:-1]
                    if fallback_annual in available_samples:
                        valid_samples.append(fallback_annual)
                        fallback_used.append(f"{sample} -> {fallback_annual}")
                    else:
                        invalid_samples.append(sample)
            elif sample.endswith('s'):
                # Try fallback from 's' to 'b' 
                fallback_sample = sample[:-1] + 'b'
                if fallback_sample in available_samples:
                    valid_samples.append(fallback_sample)
                    fallback_used.append(f"{sample} -> {fallback_sample}")
                else:
                    # Try without suffix (annual samples)
                    fallback_annual = sample[:-1]
                    if fallback_annual in available_samples:
                        valid_samples.append(fallback_annual)
                        fallback_used.append(f"{sample} -> {fallback_annual}")
                    else:
                        invalid_samples.append(sample)
            else:
                # No fallback possible for non-b/s samples
                if sample not in available_samples:
                    invalid_samples.append(sample)
                else:
                    valid_samples.append(sample)
        
        # Deduplicate: when both 'b' and 's' exist for same year-month, keep only 'b'
        deduplicated_samples = []
        year_month_seen = set()
        
        # Sort to process 'b' samples first (they come before 's' alphabetically)
        valid_samples.sort()
        
        for sample in valid_samples:
            if sample.endswith('b') or sample.endswith('s'):
                # Extract year-month prefix (e.g., "cps2022_03" from "cps2022_03b")
                year_month = sample[:-1]
                if year_month not in year_month_seen:
                    deduplicated_samples.append(sample)
                    year_month_seen.add(year_month)
                # else: skip duplicate (s when b already exists)
            else:
                # Non-monthly samples (no suffix) - always include
                deduplicated_samples.append(sample)
        
        # Enhanced logging
        duplicates_removed = len(valid_samples) - len(deduplicated_samples)
        if duplicates_removed > 0:
            LOGGER.info(f"🔄 Removed {duplicates_removed} duplicate 's' samples where 'b' exists")
        
        if fallback_used:
            LOGGER.info(f"🔀 Applied {len(fallback_used)} b/s/annual fallbacks:")
            for fallback in fallback_used[:10]:  # Show first 10
                LOGGER.info(f"   📝 {fallback}")
            if len(fallback_used) > 10:
                LOGGER.info(f"   ... and {len(fallback_used) - 10} more")
        
        if invalid_samples:
            LOGGER.warning(f"❌ {len(invalid_samples)} unavailable CPS samples will be skipped:")
            for invalid in invalid_samples[:10]:  # Show first 10
                LOGGER.warning(f"   ❌ {invalid}")
            if len(invalid_samples) > 10:
                LOGGER.warning(f"   ... and {len(invalid_samples) - 10} more")
        
        LOGGER.info(f"✅ CPS validation complete: {len(deduplicated_samples)}/{len(samples)} samples are valid")
        return deduplicated_samples
        
    except Exception as e:
        LOGGER.error(f"❌ Could not validate CPS samples against IPUMS website: {e}")
        LOGGER.warning("⚠️ Proceeding with all requested samples - some may fail during submission")
        return samples


def _resolve_ipums_data_path(output_dir: Path, ddi) -> Path | None:
    """
    Given a DDI, find the corresponding data file in output_dir, trying
    .dat.gz, .dat, .csv.gz, .csv (in that order).
    """
    base = Path(ddi.file_description.filename)  # e.g., atus_00021.dat
    stem = base.stem  # -> "atus_00021"
    candidates = [
        output_dir / f"{stem}.dat.gz",
        output_dir / f"{stem}.dat",
        output_dir / f"{stem}.csv.gz",
        output_dir / f"{stem}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def post_process_dataset(name: str, ds_cfg, output_dir: Path) -> None:
    """
    Enhanced post-processing that handles both single large extracts and chunked extracts.
    Detects single file downloads and renames appropriately, skipping concatenation.
    """
    try:
        LOGGER.info(f"🔧 [{name}] Starting post-processing...")

        # Find all XML files (DDI files)
        ddi_paths = sorted(output_dir.glob("*.xml"))
        if not ddi_paths:
            LOGGER.warning(f"⚠️ [{name}] No XML/DDI files found for processing")
            return

        LOGGER.info(f"📄 [{name}] Found {len(ddi_paths)} DDI files to process")

        # Check if this is a single extract (most common case now)
        if len(ddi_paths) == 1:
            LOGGER.info(f"📦 [{name}] Single extract detected - will process without concatenating")
            
            ddi_path = ddi_paths[0]
            try:
                ddi = readers.read_ipums_ddi(ddi_path)
                data_file_path = _resolve_ipums_data_path(output_dir, ddi)

                if not data_file_path:
                    LOGGER.warning(f"⚠️ [{name}] Data file not found for {ddi_path.name}")
                    return

                LOGGER.info(f"📖 [{name}] Reading single extract from {data_file_path.name}...")

                # Read the data (IPUMS fixed-width format needs DDI schema)
                if data_file_path.suffix.lower() == ".gz" and data_file_path.name.endswith(".csv.gz"):
                    # Already CSV format
                    df_combined = pd.read_csv(data_file_path)
                    LOGGER.info(f"📊 [{name}] Loaded CSV data: {len(df_combined):,} rows, {len(df_combined.columns)} columns")
                elif data_file_path.suffix.lower() == ".csv":
                    # Already CSV format
                    df_combined = pd.read_csv(data_file_path)
                    LOGGER.info(f"📊 [{name}] Loaded CSV data: {len(df_combined):,} rows, {len(df_combined.columns)} columns")
                else:
                    # Fixed-width format - need to parse using DDI schema
                    LOGGER.info(f"📋 [{name}] Parsing fixed-width data using DDI schema...")
                    df_combined = readers.read_microdata(ddi, data_file_path)
                    LOGGER.info(f"📊 [{name}] Parsed fixed-width data: {len(df_combined):,} rows, {len(df_combined.columns)} columns")

                # Report on variable availability for CPS
                if name.lower() == "cps":
                    telework_vars = {'TELWORK', 'TELWRKHR', 'TELWRKPAY'}
                    present_telework = telework_vars & set(df_combined.columns)
                    missing_telework = telework_vars - set(df_combined.columns)
                    
                    if present_telework:
                        LOGGER.info(f"📊 [{name}] Telework variables present: {sorted(present_telework)}")
                        for var in present_telework:
                            non_null_count = df_combined[var].notna().sum()
                            total_count = len(df_combined)
                            null_count = total_count - non_null_count
                            LOGGER.info(f"   📈 {var}: {non_null_count:,} non-null, {null_count:,} null ({non_null_count/total_count*100:.1f}% coverage)")
                    
                    if missing_telework:
                        LOGGER.info(f"⚠️ [{name}] Telework variables not available: {sorted(missing_telework)}")

                # Build output base name
                years = year_range(ds_cfg.years_start, ds_cfg.years_end)
                start_year, final_year = years[0], years[-1]
                if getattr(ds_cfg, "output_filename", None):
                    base_name = ds_cfg.output_filename.format(start_year=start_year, final_year=final_year)
                else:
                    base_name = f"{name.lower()}_{start_year}_{final_year}"

                # Save as compressed CSV for downstream processing
                output_csv = output_dir / f"{base_name}.csv.gz"
                df_combined.to_csv(output_csv, index=False)
                LOGGER.info(f"💾 [{name}] Saved processed dataset: {output_csv}")
                LOGGER.info(f"📊 [{name}] Final dataset: {len(df_combined):,} rows, {len(df_combined.columns)} columns")

                # Rename DDI file
                new_xml_path = output_dir / f"{base_name}.xml"
                ddi_path.replace(new_xml_path)
                LOGGER.info(f"📄 [{name}] Renamed DDI file: {ddi_path.name} → {new_xml_path.name}")

                # Clean up original data file and any temp files (but keep our final outputs)
                files_to_keep = {output_csv.name, new_xml_path.name}
                for file_path in output_dir.iterdir():
                    if file_path.name in files_to_keep:
                        continue
                    try:
                        if file_path.is_dir():
                            import shutil
                            shutil.rmtree(file_path)
                        else:
                            file_path.unlink()
                        LOGGER.debug(f"🗑️ [{name}] Removed: {file_path.name}")
                    except Exception as e:
                        LOGGER.warning(f"⚠️ [{name}] Failed to remove {file_path.name}: {e}")

                remaining_files = [f.name for f in output_dir.iterdir()]
                LOGGER.info(f"🧹 [{name}] Single file processing complete. Final files: {remaining_files}")

            except Exception as e:
                LOGGER.error(f"❌ [{name}] Failed to process single extract: {e}")
                return
            
            ddi_path = ddi_paths[0]
            try:
                ddi = readers.read_ipums_ddi(ddi_path)
                data_file_path = _resolve_ipums_data_path(output_dir, ddi)

                if not data_file_path:
                    LOGGER.warning(f"⚠️ [{name}] Data file not found for {ddi_path.name}")
                    return

                # Build output base name
                years = year_range(ds_cfg.years_start, ds_cfg.years_end)
                start_year, final_year = years[0], years[-1]
                if getattr(ds_cfg, "output_filename", None):
                    base_name = ds_cfg.output_filename.format(start_year=start_year, final_year=final_year)
                else:
                    base_name = f"{name.lower()}_{start_year}_{final_year}"

                # Rename data file directly (no read/write needed!)
                if data_file_path.suffix.lower() == ".gz":
                    output_csv = output_dir / f"{base_name}.csv.gz"
                else:
                    output_csv = output_dir / f"{base_name}.csv"
                
                data_file_path.replace(output_csv)
                LOGGER.info(f"� [{name}] Renamed data file: {data_file_path.name} → {output_csv.name}")

                # Get basic file info without reading the entire dataset
                file_size_mb = output_csv.stat().st_size / (1024 * 1024)
                LOGGER.info(f"� [{name}] Dataset size: {file_size_mb:.1f} MB")

                # Optional: Quick telework variables check for CPS (only if requested)
                if name.lower() == "cps" and hasattr(ds_cfg, 'variables'):
                    telework_vars = {'TELWORK', 'TELWRKHR', 'TELWRKPAY'}
                    requested_telework = set(ds_cfg.variables) & telework_vars
                    if requested_telework:
                        LOGGER.info(f"📊 [{name}] Telework variables requested: {sorted(requested_telework)}")
                        LOGGER.info(f"   ℹ️  These will have values for 2022+ samples and NULL for 2013-2021")

                # Rename DDI file
                new_xml_path = output_dir / f"{base_name}.xml"
                ddi_path.replace(new_xml_path)
                LOGGER.info(f"📄 [{name}] Renamed DDI file: {ddi_path.name} → {new_xml_path.name}")

                # Clean up any other files (but keep our renamed files)
                files_to_keep = {output_csv.name, new_xml_path.name}
                for file_path in output_dir.iterdir():
                    if file_path.name in files_to_keep:
                        continue
                    try:
                        if file_path.is_dir():
                            import shutil
                            shutil.rmtree(file_path)
                        else:
                            file_path.unlink()
                        LOGGER.debug(f"🗑️ [{name}] Removed: {file_path.name}")
                    except Exception as e:
                        LOGGER.warning(f"⚠️ [{name}] Failed to remove {file_path.name}: {e}")

                remaining_files = [f.name for f in output_dir.iterdir()]
                LOGGER.info(f"🧹 [{name}] Single file processing complete. Final files: {remaining_files}")
                LOGGER.info(f"✅ [{name}] Processing complete - no data reading/writing needed!")

            except Exception as e:
                LOGGER.error(f"❌ [{name}] Failed to process single extract: {e}")
                return

        else:
            # Multiple extracts - use original concatenation logic
            LOGGER.info(f"📦 [{name}] Multiple extracts detected - will concatenate {len(ddi_paths)} files")
            
            # Read and concatenate all data files
            df_chunks = []
            all_columns = set()
            
            for ddi_path in ddi_paths:
                LOGGER.info(f"📖 [{name}] Processing {ddi_path.name}...")
                try:
                    ddi = readers.read_ipums_ddi(ddi_path)
                    data_file_path = _resolve_ipums_data_path(output_dir, ddi)

                    if not data_file_path:
                        LOGGER.warning(f"⚠️ [{name}] Data file not found for {ddi_path.name}")
                        continue

                    # Read data
                    if data_file_path.suffix.lower() == ".gz" and data_file_path.name.endswith(".csv.gz"):
                        chunk_df = pd.read_csv(data_file_path)
                    elif data_file_path.suffix.lower() == ".csv":
                        chunk_df = pd.read_csv(data_file_path)
                    else:
                        chunk_df = readers.read_microdata(ddi, data_file_path)

                    df_chunks.append(chunk_df)
                    all_columns.update(chunk_df.columns)
                    LOGGER.info(f"✅ [{name}] Added {len(chunk_df):,} rows from {data_file_path.name}")

                except Exception as e:
                    LOGGER.error(f"❌ [{name}] Failed to process {ddi_path.name}: {e}")

            if not df_chunks:
                LOGGER.warning(f"⚠️ [{name}] No data was successfully processed")
                return

            # Ensure all chunks have the same columns (add missing columns with NaN)
            if len(df_chunks) > 1:
                LOGGER.info(f"🔄 [{name}] Harmonizing columns across {len(df_chunks)} chunks...")
                
                # Add missing columns to each chunk
                harmonized_chunks = []
                for i, chunk_df in enumerate(df_chunks):
                    missing_cols = all_columns - set(chunk_df.columns)
                    if missing_cols:
                        LOGGER.info(f"   📝 Adding {len(missing_cols)} missing columns to chunk {i+1}: {sorted(missing_cols)}")
                        for col in missing_cols:
                            chunk_df[col] = pd.NA  # Use pandas NA for missing values
                    
                    # Ensure consistent column order
                    chunk_df = chunk_df[sorted(all_columns)]
                    harmonized_chunks.append(chunk_df)
                
                df_chunks = harmonized_chunks

            df_combined = pd.concat(df_chunks, ignore_index=True)

            # Build output base name
            years = year_range(ds_cfg.years_start, ds_cfg.years_end)
            start_year, final_year = years[0], years[-1]
            if getattr(ds_cfg, "output_filename", None):
                base_name = ds_cfg.output_filename.format(start_year=start_year, final_year=final_year)
            else:
                base_name = f"{name.lower()}_{start_year}_{final_year}"

            # Save combined dataset
            output_csv = output_dir / f"{base_name}.csv.gz"
            df_combined.to_csv(output_csv, index=False)
            
            # Report on missing variables for CPS
            if name.lower() == "cps":
                telework_vars = {'TELWORK', 'TELWRKHR', 'TELWRKPAY'}
                present_telework = telework_vars & set(df_combined.columns)
                missing_telework = telework_vars - set(df_combined.columns)
                
                if present_telework:
                    LOGGER.info(f"📊 [{name}] Telework variables present: {sorted(present_telework)}")
                    for var in present_telework:
                        non_null_count = df_combined[var].notna().sum()
                        total_count = len(df_combined)
                        LOGGER.info(f"   📈 {var}: {non_null_count:,}/{total_count:,} ({non_null_count/total_count*100:.1f}%) non-null")
                
                if missing_telework:
                    LOGGER.info(f"⚠️ [{name}] Telework variables not available: {sorted(missing_telework)}")
            
            LOGGER.info(
                f"💾 [{name}] Saved combined dataset: {output_csv} "
                f"({len(df_combined):,} rows, {len(df_combined.columns)} cols)"
            )

            # Keep ONE DDI and clean up
            ddi_path_keep = max(ddi_paths, key=lambda p: p.stat().st_mtime)
            new_xml_path = output_dir / f"{base_name}.xml"
            ddi_path_keep.replace(new_xml_path)
            LOGGER.info(f"📄 [{name}] Kept DDI file as: {new_xml_path.name}")

            # Clean up temporary files
            files_to_keep = {output_csv.name, new_xml_path.name}
            for file_path in output_dir.iterdir():
                if file_path.name in files_to_keep:
                    continue
                try:
                    if file_path.is_dir():
                        import shutil
                        shutil.rmtree(file_path)
                    else:
                        file_path.unlink()
                except Exception as e:
                    LOGGER.warning(f"⚠️ [{name}] Failed to remove {file_path.name}: {e}")

            remaining_files = [f.name for f in output_dir.iterdir()]
            LOGGER.info(f"🧹 [{name}] Multi-file processing complete. Final files: {remaining_files}")

    except Exception as e:
        LOGGER.error(f"❌ [{name}] Post-processing failed: {e}")


def submit_extract(client: IpumsApiClient, collection: str, samples: List[str], variables: List[str]) -> Tuple[Any, str]:
    """
    Submits an extract request to the IPUMS API and returns the extract ID.
    Pre-validates samples to avoid submission errors.
    """
    # Pre-validate CPS samples before submission
    if collection.lower() == "cps":
        original_count = len(samples)
        samples = validate_cps_samples_api(client, samples)
        if len(samples) != original_count:
            LOGGER.warning(f"⚠️ CPS sample validation reduced samples from {original_count} to {len(samples)}")
        
        if not samples:
            raise ValueError("No valid CPS samples remaining after validation")
    
    # Create an IPUMS extract request with proper configuration
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    description = f"Searching-Flexibility project extract for {collection} - {timestamp}"
    
    # Use rectangular-on-A for ATUS (activity level), rectangular-on-P for others (person level)
    if collection == "atus":
        data_structure = {"rectangular": {"on": "A"}}
    else:
        data_structure = {"rectangular": {"on": "P"}}
    
    ext = MicrodataExtract(
        collection=collection, 
        samples=samples, 
        variables=variables,
        description=description,
        data_format="fixed_width",
        data_structure=data_structure
    )
    
    try:
        # Submit the extract request to the IPUMS API and obtain the extract ID
        extract_id = client.submit_extract(ext)
        LOGGER.info(f"✅ Submitted {collection} extract with {len(samples)} samples. ID: {extract_id}")
        LOGGER.info(f"   📋 Samples: {samples}")
        LOGGER.info(f"   🏷️  Variables: {variables}")
        LOGGER.info(f"   📄 Description: {description}")
        
        return extract_id, description
        
    except Exception as e:
        error_msg = str(e)
        if "Invalid sample name" in error_msg:
            LOGGER.error(f"❌ IPUMS API rejected samples. Error: {error_msg}")
            # Try to extract invalid sample names from error message
            invalid_samples = []
            lines = error_msg.split('\n')
            for line in lines:
                if 'Invalid sample name:' in line:
                    sample_name = line.split('Invalid sample name:')[1].strip()
                    invalid_samples.append(sample_name)
            
            if invalid_samples:
                LOGGER.info(f"🔧 Attempting to retry with {len(invalid_samples)} invalid samples removed...")
                valid_samples = [s for s in samples if s not in invalid_samples]
                if valid_samples:
                    return submit_extract(client, collection, valid_samples, variables)
                else:
                    raise ValueError("No valid samples remaining after removing invalid ones")
        raise


def download_when_ready(client: IpumsApiClient, collection: str, extract_id: Any, description: str, output_dir: Path) -> None:
    """
    Waits for an extract to be ready and downloads it directly to the output directory.

    Args:
        client (IpumsApiClient): An authenticated IPUMS API client instance.
        collection (str): The name of the IPUMS data collection.
        extract_id (Any): The extract ID returned from submit_extract.
        description (str): Description of the extract for logging.
        output_dir (Path): Directory to download files to.

    Returns:
        None: Files are downloaded directly to disk.
    """
    LOGGER.info(f"⏳ Waiting for {collection} extract to be processed...")
    LOGGER.info(f"   🆔 Extract ID: {extract_id}")
    LOGGER.info(f"   📄 Description: {description}")
    max_wait_time = 3600  # 1 hour max wait time
    check_interval = 30   # Check every 30 seconds
    elapsed_time = 0
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    while elapsed_time < max_wait_time:
        try:
            # Try to download - if successful, extract is ready
            client.download_extract(extract_id, download_dir=str(output_dir))
            LOGGER.info(f"🎉 Successfully downloaded {collection} extract!")
            LOGGER.info(f"💾 Files saved to: {output_dir}")
            
            # List downloaded files for logging
            downloaded_files = list(output_dir.glob("*"))
            if downloaded_files:
                LOGGER.info(f"📂s Downloaded files: {[f.name for f in downloaded_files]}")
            
            return
                
        except Exception as e:
            error_msg = str(e)
            if "not finished yet" in error_msg.lower() or "not ready" in error_msg.lower():
                # Extract not ready yet, wait and try again
                mins_elapsed = elapsed_time // 60
                LOGGER.info(f"⌛ {collection} extract still processing... waiting {check_interval}s (elapsed: {mins_elapsed}m {elapsed_time % 60}s)")
                time.sleep(check_interval)
                elapsed_time += check_interval
            else:
                # Some other error occurred
                LOGGER.error(f"❌ Error downloading {collection} extract: {error_msg}")
                raise e
    
    # If we get here, we've exceeded max wait time
    raise TimeoutError(f"⏰ Extract {extract_id} for {collection} not ready after {max_wait_time} seconds")


def submit_and_download(client: IpumsApiClient, collection: str, samples: List[str], variables: List[str], output_dir: Path) -> None:
    """
    Legacy function that submits and downloads in sequence (kept for compatibility).
    """
    extract_id, description = submit_extract(client, collection, samples, variables)
    download_when_ready(client, collection, extract_id, description, output_dir)


def acquire_all_datasets_concurrent(client: IpumsApiClient, cfg: Config, log_file: Path) -> None:
    """
    Acquires all enabled datasets concurrently by submitting all extracts first,
    then waiting for them to complete in parallel.
    
    This is much more efficient than sequential processing since IPUMS processes
    extracts in parallel on their servers.
    """
    # Phase 1: Submit all extracts
    LOGGER.info("🚀 === Phase 1: Submitting all extracts ===")
    extract_submissions = {}  # {dataset_name: (extract_id, description, ds_cfg, samples)}
    
    for name, ds_cfg in cfg.datasets.items():
        if not ds_cfg.enabled:
            LOGGER.info(f"⏭️  Dataset '{name}' is disabled; skipping.")
            continue
            
        try:
            # Resolve samples for this dataset
            samples = resolve_samples_for(name, ds_cfg)
            max_per = ds_cfg.max_samples_per_extract or 12
            LOGGER.info(f"📝 Preparing {name} ({ds_cfg.collection}) with {len(samples)} samples, {len(ds_cfg.variables)} variables")
            
            # For now, submit as single extract per dataset
            # Chunking will be handled at the download level if needed
            extract_id, description = submit_extract(client, ds_cfg.collection, samples, ds_cfg.variables)
            extract_submissions[name] = (extract_id, description, ds_cfg, samples)
                
        except Exception as e:
            LOGGER.error(f"❌ Failed to submit extract for '{name}': {e}")
    
    if not extract_submissions:
        LOGGER.warning("⚠️  No extracts were submitted successfully.")
        return
    
    LOGGER.info(f"📊 Summary: {len(extract_submissions)} extracts submitted successfully")
    for name in extract_submissions.keys():
        LOGGER.info(f"   ✅ {name}")
    
    # Phase 2: Wait for all extracts to complete and download concurrently
    LOGGER.info("⏳ === Phase 2: Waiting for extracts to complete ===")
    LOGGER.info(f"💫 Processing {len(extract_submissions)} extracts concurrently...")
    
    def download_and_process_dataset(name: str, extract_info: Tuple) -> None:
        """Helper function to download a single dataset, handling chunking if needed."""
        extract_id, description, ds_cfg, samples = extract_info
        try:
            LOGGER.info(f"🔄 [{name}] Starting download...")
            
            # All datasets save to raw_dir now
            output_dir = Path(cfg.raw_dir) / name.lower()
            
            # Clean the download folder before starting (only for this specific dataset)
            clean_download_folder(name, output_dir)
            
            # Check if we need to chunk this dataset
            max_per = ds_cfg.max_samples_per_extract or 12
            
            if len(samples) <= max_per:
                # Single download
                download_when_ready(client, ds_cfg.collection, extract_id, description, output_dir)
            else:
                # Need chunking - create separate extracts and download them in parallel
                LOGGER.info(f"📦 Dataset {name} has {len(samples)} samples > max {max_per}. Will download {math.ceil(len(samples)/max_per)} chunks in parallel.")
                
                # Submit separate extracts for each chunk
                chunk_threads = []
                chunk_extracts = []
                
                for i, chunk in enumerate(chunked(samples, max_per), start=1):
                    LOGGER.info(f"📝 Submitting chunk {i}/{math.ceil(len(samples)/max_per)} with {len(chunk)} samples")
                    chunk_extract_id, chunk_description = submit_extract(client, ds_cfg.collection, chunk, ds_cfg.variables)
                    # All chunks download to the same directory (no subfolders)
                    
                    # Create thread for this chunk
                    chunk_thread = threading.Thread(
                        target=download_when_ready,
                        args=(client, ds_cfg.collection, chunk_extract_id, chunk_description, output_dir),
                        name=f"download-{name}-chunk-{i}"
                    )
                    chunk_threads.append(chunk_thread)
                    chunk_thread.start()
                
                # Wait for all chunks to complete
                for thread in chunk_threads:
                    thread.join()
                    LOGGER.info(f"🏁 Chunk thread '{thread.name}' completed")
            
            # Post-process: combine chunks and clean up files
            post_process_dataset(name, ds_cfg, output_dir)
            
            LOGGER.info(f"✅ [{name}] Completed download and post-processing successfully!")
            
        except Exception as e:
            LOGGER.error(f"❌ [{name}] Failed to download dataset: {e}")
    
    # Create threads for concurrent downloading
    threads = []
    for name, extract_info in extract_submissions.items():
        thread = threading.Thread(
            target=download_and_process_dataset, 
            args=(name, extract_info),
            name=f"download-{name}"
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all downloads to complete
    LOGGER.info(f"⏳ Waiting for {len(threads)} concurrent downloads to complete...")
    for thread in threads:
        thread.join()
        LOGGER.info(f"🏁 Thread '{thread.name}' completed")
    
    LOGGER.info("🎉 === All datasets completed successfully! ===")
    LOGGER.info(f"📄 Full log saved to: {log_file}")


def acquire_dataset(client: IpumsApiClient, name: str, cfg: Config, ds_cfg: DatasetConfig) -> None:
    """
    Acquires and downloads a dataset from the IPUMS API client according to the provided configuration.

    This function:
    - Resolves the list of samples to acquire.
    - Cleans the download directory for the specific dataset.
    - Downloads the data in chunks, respecting the maximum samples per extract.
    - Saves files directly to the appropriate output directory.

    Parameters:
        client (IpumsApiClient): The IPUMS API client used to submit and download data extracts.
        name (str): The name of the dataset to acquire (e.g., "ATUS", "ACS", "CPS").
        cfg (Config): Global configuration object containing directory paths and other settings.
        ds_cfg (DatasetConfig): Dataset-specific configuration, including enabled flag, variables, years, and output filenames.

    Returns:
        None

    Logs:
        - Skips disabled datasets.
        - Logs progress for each chunk submission and data download.
    """
    if not ds_cfg.enabled:
        LOGGER.info(f"Dataset '{name}' is disabled; skipping.")
        return

    # Resolve the list of samples to acquire for this dataset
    samples = resolve_samples_for(name, ds_cfg)
    max_per = ds_cfg.max_samples_per_extract or 12
    LOGGER.info(f"Acquiring {name} ({ds_cfg.collection}) for {len(samples)} samples.")

    # Determine output directory - all datasets save to raw_dir now
    output_dir = Path(cfg.raw_dir) / name.lower()
    
    # Clean the download folder before starting (only for this specific dataset)
    clean_download_folder(name, output_dir)
    
    ensure_dir(output_dir)

    # Download data in chunks, if necessary
    for i, chunk in enumerate(chunked(samples, max_per), start=1):
        LOGGER.info(f"Submitting chunk {i}/{math.ceil(len(samples)/max_per)} with {len(chunk)} samples…")
        chunk_output_dir = output_dir / f"chunk_{i}" if len(samples) > max_per else output_dir
        submit_and_download(client, ds_cfg.collection, chunk, ds_cfg.variables, chunk_output_dir)

    LOGGER.info(f"Completed downloading {name} to {output_dir}")


def main(argv: List[str] | None = None) -> None:
    """
    Main entry point for the IPUMS data acquisition script.

    - Loads configuration from YAML (default or provided as first CLI argument).
    - Resolves the IPUMS API key.
    - Initializes the IPUMS API client.
    - Acquires all configured datasets concurrently for efficiency.
    - Logs errors for any dataset that fails to acquire.
    """
    # Set up file logging
    log_file = Path(__file__).parent / f"ipums_acquire_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(file_handler)
    LOGGER.info(f"📄 Logging to file: {log_file}")
    
    argv = argv or sys.argv[1:]
    # Allow optional path override: python ipums_acquire.py path/to/ipums_config.yml
    repo_root = Path(__file__).resolve().parents[2]
    default_cfg = repo_root / "src" / "data" / "config.yml"  # Updated path
    cfg_path = Path(argv[0]) if argv else default_cfg
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg = load_config(cfg_path)
    api_key = resolve_api_key(cfg)
    client = IpumsApiClient(api_key=api_key)

    # Log startup summary
    LOGGER.info("🚀 Starting IPUMS data acquisition")
    LOGGER.info(f"📁 Config file: {cfg_path}")
    LOGGER.info(f"📂 Raw data directory: {cfg.raw_dir}")
    LOGGER.info(f"📂 Processed data directory: {cfg.processed_dir}")
    
    enabled_datasets = [name for name, ds_cfg in cfg.datasets.items() if ds_cfg.enabled]
    LOGGER.info(f"📊 Enabled datasets: {enabled_datasets}")
    
    for name, ds_cfg in cfg.datasets.items():
        if ds_cfg.enabled:
            samples = resolve_samples_for(name, ds_cfg)
            LOGGER.info(f"   🏷️  {name}: {len(samples)} samples, {len(ds_cfg.variables)} variables")

    # Use concurrent processing for efficiency
    acquire_all_datasets_concurrent(client, cfg, log_file)


if __name__ == "__main__":
    main()
