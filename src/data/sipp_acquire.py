"""
SIPP downloader (config-driven, Polars)

- Reads settings from src/data/config.yml (top-level 'sipp' section + paths.raw_dir)
- Connects to Census FTP (anonymous), downloads PU (primary) and RW (replicate weights)
- Handles both pu{year}_csv.zip / pu{year}.csv.gz naming patterns
- Uses schema JSON for dtypes; special-case 2021 PU schema to use 2022's
- Selects columns via YAML variables; RW auto-selects REPWGT* + keys
- Merges on ['SSUID','PNUM','MONTHCODE']
- Saves CSV (gz) and optional Parquet
- Parallel per-year downloads via ThreadPoolExecutor
"""
from __future__ import annotations

import io
import os
import sys
import json
import gzip
import zipfile
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
from ftplib import FTP
from tqdm import tqdm
from types import SimpleNamespace

# Optional imports from sibling module for bridge construction
try:
    # When executed as a module (e.g., python -m src.data.sipp_acquire)
    from .ipums_process import build_bridge_lambdas, diagnose_bridge  # type: ignore
except Exception:
    try:
        # When executed as a script from repo root (PYTHONPATH not set)
        import sys as _sys
        from pathlib import Path as _Path
        _repo_root = _Path(__file__).resolve().parents[2]
        if str(_repo_root) not in _sys.path:
            _sys.path.insert(0, str(_repo_root))
        from src.data.ipums_process import build_bridge_lambdas, diagnose_bridge  # type: ignore
    except Exception:
        build_bridge_lambdas = None  # type: ignore
        diagnose_bridge = None  # type: ignore


# ---------------- Logging ---------------- #
LOGGER = logging.getLogger("sipp_acquire")
LOGGER.setLevel(logging.INFO)
_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
LOGGER.addHandler(_console)


# ---------------- Config ---------------- #
@dataclass
class SippConfig:
    enabled: bool
    ftp_host: str
    ftp_base_path: str
    years_start: int | None
    years_end: int | str | None
    years_list: List[int] | None
    variables: List[str]
    max_workers: int
    save_parquet: bool
    output_basename: str  # e.g., "sipp_{year}"
    # merge keys (usually SSUID, PNUM, MONTHCODE)
    merge_keys: List[str]

@dataclass
class RootConfig:
    raw_dir: Path
    sipp: SippConfig


def load_yaml_config(cfg_path: Path) -> RootConfig:
    import yaml

    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # repo_root/src/data/ -> repo_root
    raw_dir = project_root / raw["paths"].get("raw_dir", "data/raw")

    sipp_raw = raw.get("sipp", {})
    years_list = sipp_raw.get("years", {}).get("list")
    years_start = sipp_raw.get("years", {}).get("start")
    years_end = sipp_raw.get("years", {}).get("end")

    # Expand pattern variables like EJB{i}_JOBID for i=1..7
    def _expand_i_vars(vars_list: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for v in vars_list:
            if "{i}" in v:
                for i in range(1, 8):
                    vv = v.replace("{i}", str(i))
                    if vv not in seen:
                        out.append(vv)
                        seen.add(vv)
            else:
                if v not in seen:
                    out.append(v)
                    seen.add(v)
        return out

    scfg = SippConfig(
        enabled=bool(sipp_raw.get("enabled", True)),
        ftp_host=sipp_raw.get("ftp", {}).get("host", "ftp2.census.gov"),
        ftp_base_path=sipp_raw.get("ftp", {}).get("base_path", "programs-surveys/sipp/data/datasets"),
        years_start=years_start,
        years_end=years_end,
        years_list=years_list,
        variables=_expand_i_vars(sipp_raw.get("variables", [])),
        max_workers=int(sipp_raw.get("max_workers", 4)),
        save_parquet=bool(sipp_raw.get("save_parquet", False)),
        output_basename=sipp_raw.get("output_basename", "sipp_{year}"),
        merge_keys=sipp_raw.get("merge_keys", ["SSUID", "PNUM", "MONTHCODE"]),
    )
    return RootConfig(raw_dir=raw_dir, sipp=scfg)


def year_range(start: int | None, end: int | str | None) -> List[int]:
    if start is None and end is None:
        raise ValueError("Provide years.list or years.start/end in YAML.")
    if isinstance(end, str) and end.lower() == "present":
        end_year = datetime.now().year
    else:
        end_year = int(end) if end is not None else int(start)
    return list(range(int(start), end_year + 1))


# ---------------- FTP helpers ---------------- #
def ftp_connect(host: str) -> FTP:
    ftp = FTP(host, timeout=180)
    ftp.login()
    return ftp


def ftp_cwd(ftp: FTP, path: str) -> None:
    ftp.cwd(path)


def ftp_bytes(ftp: FTP, filename: str, desc: str) -> io.BytesIO:
    """Download a file to memory with progress bar (size if available)."""
    buf = io.BytesIO()
    try:
        ftp.voidcmd('TYPE I')
        size = ftp.size(filename)
    except Exception:
        size = None

    if size:
        pbar = tqdm(total=size, unit="B", unit_scale=True, desc=f"Downloading {desc}", leave=False)
    else:
        pbar = tqdm(unit="B", unit_scale=True, desc=f"Downloading {desc}", leave=False)

    def _cb(chunk: bytes):
        buf.write(chunk)
        pbar.update(len(chunk))

    ftp.retrbinary(f"RETR {filename}", _cb)
    pbar.close()
    buf.seek(0)
    return buf


def detect_data_file(ftp: FTP, year: int, prefix: str) -> Tuple[str, bool]:
    """
    Returns (filename, is_zip). Tries {prefix}{year}_csv.zip then {prefix}{year}.csv.gz
    """
    zip_name = f"{prefix}{year}_csv.zip"
    gz_name = f"{prefix}{year}.csv.gz"
    try:
        ftp.voidcmd('TYPE I')
        ftp.size(zip_name)
        return zip_name, True
    except Exception:
        try:
            ftp.voidcmd('TYPE I')
            ftp.size(gz_name)
            return gz_name, False
        except Exception:
            # As last resort, try to list and match
            names = []
            ftp.retrlines("NLST", names.append)
            if zip_name in names:
                return zip_name, True
            if gz_name in names:
                return gz_name, False
            raise FileNotFoundError(f"No data file found for {prefix}{year} (.zip or .csv.gz)")


# ---------------- Schema and reading ---------------- #
def _map_json_dtype_to_polars(t: Any) -> pl.DataType:
    ts = str(t).lower()
    if "int" in ts:
        return pl.Int64
    if "float" in ts or "double" in ts or "number" in ts:
        return pl.Float64
    if "bool" in ts:
        return pl.Boolean
    # default
    return pl.Utf8

def _parse_schema_json(raw_json: str) -> Tuple[List[str], Dict[str, pl.DataType]]:
    """
    Return (names, dtypes_map) from schema JSON (list of {name, type/dtype/...})
    """
    data = json.loads(raw_json)
    names: List[str] = []
    dtypes: Dict[str, pl.DataType] = {}
    # schema is typically a list of dicts; be robust
    for col in data:
        if not isinstance(col, dict):
            continue
        name = col.get("name") or col.get("varname") or col.get("column") or None
        if not name:
            continue
        jtype = col.get("dtype") or col.get("type") or col.get("dataType") or "string"
        names.append(name)
        dtypes[name] = _map_json_dtype_to_polars(jtype)
    return names, dtypes

def load_schema(ftp: FTP, year: int, kind: str) -> Tuple[List[str], Dict[str, pl.DataType]]:
    """
    kind in {'pu','rw'}
    Returns (all_names, dtypes_map[name]->pl.DataType)
    """
    # Use 2022 schema only for 2021 PU (corrupted 2021 PU schema on FTP)
    schema_year = 2022 if (year == 2021 and kind.lower() == "pu") else year
    schema_name = f"{kind}{schema_year}_schema.json"

    # If falling back to another year, temporarily change directory to that year's folder
    if schema_year != year:
        cur_dir = ftp.pwd()
        try:
            parent = cur_dir.rstrip("/").rsplit("/", 1)[0]
            alt_dir = f"{parent}/{schema_year}"
            LOGGER.warning(f"{kind.upper()} {year}: schema fallback to {schema_year}; switching directory {cur_dir} -> {alt_dir}")
            ftp_cwd(ftp, alt_dir)
            raw = ftp_bytes(ftp, schema_name, desc=schema_name)
        finally:
            ftp_cwd(ftp, cur_dir)
    else:
        raw = ftp_bytes(ftp, schema_name, desc=schema_name)

    names, dtypes = _parse_schema_json(raw.read().decode("utf-8"))
    return names, dtypes

def decompress_to_csv_bytes(file_bytes: io.BytesIO, is_zip: bool) -> io.BytesIO:
    if is_zip:
        with zipfile.ZipFile(file_bytes, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise ValueError("No .csv found inside zip")
            with zf.open(csv_names[0], "r") as f:
                return io.BytesIO(f.read())
    else:
        with gzip.GzipFile(fileobj=file_bytes, mode="rb") as f:
            return io.BytesIO(f.read())

def read_sipp_csv(csv_bytes: io.BytesIO, all_names: List[str], dtypes_map: Dict[str, pl.DataType], usecols: List[str]) -> pl.DataFrame:
    """
    Reads SIPP CSV with '|' separator using Polars, selecting usecols.
    """
    cols = [c for c in usecols if c in all_names]
    if not cols:
        raise ValueError("None of the requested variables exist in schema.")
    # Restrict dtypes to requested columns; let others infer if missing
    dtype_subset = {c: dtypes_map.get(c, pl.Utf8) for c in cols}
    csv_bytes.seek(0)
    df = pl.read_csv(
        csv_bytes,
        separator="|",
        has_header=True,
        columns=cols,
        dtypes=dtype_subset,
        infer_schema_length=0,  # no need to scan extra rows; we set dtypes for requested cols
    )
    return df


# ---------------- Year processing ---------------- #
def download_year(
                    ftp_host: str,
                    base_path: str,
                    year: int,
                    pu_vars: List[str],
                    merge_keys: List[str],
                    replicate_weights: bool
                                            ) -> Dict[str, Any]:
    
    LOGGER.info(f"➡️  Year {year}: start")
    ftp = ftp_connect(ftp_host)
    try:
        ftp_cwd(ftp, f"{base_path}/{year}")

        # Load schemas
        pu_names, pu_dtypes = load_schema(ftp, year, "pu")
        rw_names, rw_dtypes = load_schema(ftp, year, "rw")

        # Validate pu_vars against pu schema
        pu_vars_not_in_schema = [v for v in pu_vars if v not in set(pu_names)]
        if pu_vars_not_in_schema:
            LOGGER.warning(
                f"Year {year}: {len(pu_vars_not_in_schema)} requested PU variables not found in schema: "
                f"{pu_vars_not_in_schema[:10]}{'...' if len(pu_vars_not_in_schema) > 10 else ''}"
            )
        else:
            LOGGER.info(f"Year {year}: All requested PU variables found in schema.")

        # Download primary
        pu_filename, pu_is_zip = detect_data_file(ftp, year, "pu")
        pu_bytes = ftp_bytes(ftp, pu_filename, desc=pu_filename)
        pu_csv = decompress_to_csv_bytes(pu_bytes, pu_is_zip)

        # PU usecols = merge_keys + requested in config
        requested_pu = list(dict.fromkeys(merge_keys + pu_vars))
        pu_usecols = [c for c in requested_pu if c in pu_names]
        missing_pu = sorted(set(requested_pu) - set(pu_usecols))
        if missing_pu:
            LOGGER.warning(f"Year {year}: {len(missing_pu)} PU vars not in schema: {missing_pu[:10]}{'...' if len(missing_pu)>10 else ''}")

        df_pu = read_sipp_csv(pu_csv, pu_names, pu_dtypes, pu_usecols)

        # Download replicate weights
        if replicate_weights:
            rw_filename, rw_is_zip = detect_data_file(ftp, year, "rw")
            rw_bytes = ftp_bytes(ftp, rw_filename, desc=rw_filename)
            rw_csv = decompress_to_csv_bytes(rw_bytes, rw_is_zip)

            # RW columns: merge keys + any REPWGT*
            rw_rep_cols = [c for c in rw_names if c.startswith("REPWGT")]
            rw_usecols = list(dict.fromkeys(merge_keys + rw_rep_cols))
            df_rw = read_sipp_csv(rw_csv, rw_names, rw_dtypes, rw_usecols)

            # Merge
            for k in merge_keys:
                if k not in df_pu.columns or k not in df_rw.columns:
                    raise KeyError(f"Merge key missing: {k}")

            df = df_pu.join(df_rw, on=merge_keys, how="left")
        else:
            df = df_pu
            rw_rep_cols = []

        # Add a year column to the dataframe
        df = df.with_columns(pl.lit(year).alias("year"))

        # Return the DataFrame and metadata
        return {
            "year": int(year),
            "df": df,
            "rows": int(df.height),
            "cols": int(df.width),
            "pu_vars_requested": pu_vars,
            "pu_vars_used": pu_usecols,
            "missing_pu_vars": missing_pu,
            "rw_cols": rw_rep_cols if replicate_weights else [],
            "timestamp": datetime.now().isoformat(),
        }
    finally:
        ftp.quit()


# ---------------- Concatenation ---------------- #
def concatenate_final_dataframe(
    results: Dict[int, Dict[str, Any]],
    out_dir: Path,
    filename_pattern: str = "sipp_{start}_{end}.csv.gz",
    write_parquet: bool = False,
) -> Optional[Path]:
    """
    Concatenate per-year DataFrames held in `results[year]['df']` into a single CSV.gz file.
    - Uses Polars vertical concatenation (diagonal_relaxed) to allow differing columns across years.
    - Returns the path to the combined CSV, or None if no DataFrames found.
    """
    if not results:
        LOGGER.warning("No results to concatenate.")
        return None

    years_done = sorted(y for y, r in results.items() if isinstance(r.get("df"), pl.DataFrame))
    if not years_done:
        LOGGER.warning("No DataFrames present in results; nothing to concatenate.")
        return None

    LOGGER.info(f"🔗 Concatenating final DataFrame for years: {years_done}")
    dfs = [results[y]["df"] for y in years_done]
    try:
        combined = pl.concat(dfs, how="diagonal_relaxed", rechunk=True)
    except Exception as e:
        LOGGER.error(f"Polars concat failed: {e}")
        raise

    start_y, end_y = min(years_done), max(years_done)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / filename_pattern.format(start=start_y, end=end_y)

    # Write CSV.gz
    # Write in binary mode; Polars write_csv emits bytes
    with gzip.open(combined_path, "wb") as f:
        combined.write_csv(f)
    LOGGER.info(f"📦 Single combined file ready: {combined_path.name} (rows={combined.height:,}, cols={combined.width})")

    # Optional Parquet
    if write_parquet:
        pq_path = combined_path.with_suffix("").with_suffix(".parquet")
        try:
            combined.write_parquet(pq_path)
            LOGGER.info(f"💾 Also wrote Parquet: {pq_path.name}")
        except Exception as e:
            LOGGER.warning(f"Parquet save failed (install pyarrow?): {e}")

    return combined_path


# ---------------- Main ---------------- #
def main(argv: Optional[List[str]] = None) -> int:
    # File logging
    log_path = Path(__file__).parent / f"sipp_acquire_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(fh)
    LOGGER.info(f"📄 Logging to: {log_path}")

    argv = argv or sys.argv[1:]
    repo_root = Path(__file__).resolve().parents[2]
    default_cfg = repo_root / "src" / "data" / "config.yml"
    cfg_path = Path(argv[0]) if argv else default_cfg
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    root_cfg = load_yaml_config(cfg_path)
    scfg = root_cfg.sipp

    if not scfg.enabled:
        LOGGER.info("SIPP is disabled in config. Exiting.")
        return 0

    # Output directory under raw_dir/sipp
    out_dir = Path(root_cfg.raw_dir) / "sipp"

    # Create output dir if not exist
    out_dir.mkdir(parents=True, exist_ok=True)

    # Years to process
    if scfg.years_list:
        years = [int(y) for y in scfg.years_list]
    else:
        years = year_range(scfg.years_start, scfg.years_end)

    LOGGER.info(f"🚀 SIPP download starting for years: {years}")
    LOGGER.info(f"📂 Output: {out_dir}")
    LOGGER.info(f"🌐 FTP: {scfg.ftp_host} / {scfg.ftp_base_path}")
    LOGGER.info(f"🧾 PU variables ({len(scfg.variables)}): {scfg.variables[:8]}{'...' if len(scfg.variables)>8 else ''}")

    results: Dict[int, Dict[str, Any]] = {}
    failures: List[int] = []

    with ThreadPoolExecutor(max_workers=scfg.max_workers) as ex:
        futures = {
            ex.submit(
                download_year,  # renamed from process_year
                scfg.ftp_host,
                scfg.ftp_base_path,
                year,
                scfg.variables,
                scfg.merge_keys,
                replicate_weights=False,  #! We will set this to True when computing Standard Errors
            ): year for year in years
        }
        for fut in as_completed(futures):
            y = futures[fut]
            try:
                results[y] = fut.result()
            except Exception as e:
                failures.append(y)
                LOGGER.error(f"❌ Year {y} failed: {e}")

    summary = {
        "successful_years": sorted(list(results.keys())),
        "failed_years": sorted(failures),
        "total_rows": int(sum(r["rows"] for r in results.values())),
        "timestamp": datetime.now().isoformat(),
    }
    (out_dir / "sipp_download_summary.json").write_text(json.dumps(summary, indent=2))

    # Concatenate DataFrames into a single CSV.gz
    combined_path: Optional[Path] = None
    try:
        combined_path = concatenate_final_dataframe(
            results=results,
            out_dir=out_dir,
            filename_pattern="sipp_{start}_{end}.csv.gz",
            write_parquet=False,
        )
    except Exception as e:
        LOGGER.error(f"Failed to concatenate final DataFrame: {e}")

    # 5) Optional bridge lambdas for 2022–2025 (diagnostics after build)
    # This is best-effort and won't fail the downloader.
    produce_bridge = True
    if produce_bridge and build_bridge_lambdas is not None and diagnose_bridge is not None:
        try:
            # Locate processed empirical inputs
            repo_root = Path(__file__).resolve().parents[2]
            processed_dir = repo_root / "data" / "processed"
            empirical_dir = processed_dir / "empirical"

            cps_path_dta = empirical_dir / "cps_mi_ready_2013_2025.dta"
            atus_path_dta = empirical_dir / "atus_cell_measures_2013_2025.dta"

            if not cps_path_dta.exists() or not atus_path_dta.exists():
                missing = [str(p) for p in [cps_path_dta, atus_path_dta] if not p.exists()]
                LOGGER.warning(
                    "Bridge build skipped: required inputs not found: %s",
                    ", ".join(missing),
                )
            else:
                # Read .dta via pandas then convert to Polars (Polars doesn't read Stata natively)
                try:
                    import pandas as pd  # type: ignore
                except Exception as _e:
                    pd = None  # type: ignore
                cps_mi: Optional[pl.DataFrame] = None
                atus_cell: Optional[pl.DataFrame] = None
                if 'pd' in locals() and pd is not None:
                    try:
                        cps_mi = pl.from_pandas(pd.read_stata(cps_path_dta))  # type: ignore
                        atus_cell = pl.from_pandas(pd.read_stata(atus_path_dta))  # type: ignore
                    except Exception as e:
                        LOGGER.error(f"Failed reading DTA files for bridge: {e}")
                else:
                    LOGGER.warning("pandas not available; cannot read .dta to build bridge.")

                if cps_mi is not None and atus_cell is not None:
                    try:
                        bridge = build_bridge_lambdas(
                            cps_mi=cps_mi,
                            atus_cell=atus_cell,
                            eps=1e-6,
                            # clip=(0.5, 2.0),
                            # clip=(0.33, 3.0),
                            clip=(0.25, 4.0),
                            pool_years=[2022, 2023, 2024],
                        )
                        # Minimal cfg shim for diagnostics (paths only)
                        cfg_shim = SimpleNamespace(
                            processed_dir=processed_dir,
                            raw_dir=repo_root / "data" / "raw",
                            aux_dir=repo_root / "data" / "aux",
                        )
                        if bridge is not None:
                            try:
                                diagnose_bridge(bridge, cfg_shim, logger=LOGGER)
                            except Exception as e:
                                LOGGER.warning(f"diagnose_bridge failed: {e}")
                            # Optional: save bridge for reuse
                            try:
                                out_bridge = empirical_dir / "bridge_lambda_2022_2025.dta"
                                empirical_dir.mkdir(parents=True, exist_ok=True)
                                if 'pd' in locals() and pd is not None:
                                    pd_df = bridge.to_pandas()  # type: ignore
                                    pd_df.to_stata(out_bridge, write_index=False)  # type: ignore
                                    LOGGER.info(f"Saved bridge to {out_bridge}")
                                else:
                                    # Fallback CSV
                                    out_csv = out_bridge.with_suffix(".csv")
                                    bridge.write_csv(str(out_csv))  # type: ignore
                                    LOGGER.info(f"Saved bridge CSV to {out_csv}")
                            except Exception as e:
                                LOGGER.warning(f"Saving bridge failed: {e}")
                    except Exception as e:
                        LOGGER.warning(f"Bridge build failed: {e}")
        except Exception as e:
            LOGGER.debug(f"Bridge step encountered an error: {e}")
    else:
        LOGGER.info("Bridge step not executed (flag off or imports unavailable).")

    LOGGER.info(f"🎉 Done. Success: {len(results)} | Failed: {len(failures)}")
    return 0 if not failures else 1

if __name__ == "__main__":
    sys.exit(main())