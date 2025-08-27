"""
Unified processor for ACS, ATUS, and CPS using config-driven crosswalks and mappings.

Steps:
1) ATUS: aggregate micro to CPSIDP+YEAR for pre-2022 telework linkage.
2) ACS: apply filters, industry cleanup, WFH flag; compute SOC groupings; assign teleworkability; PUMA->CBSA and state mappings.
3) CPS: keep all rows pre-2022; left-join ATUS linkage; set telework for pre-2022 from ATUS; map industry; map OCC->SOC; compute SOC groupings; assign teleworkability.

All paths, crosswalks, and mapping specs come from src/empirical/config.yml.
"""

# ---------------- Configuration loading ---------------- #
# Core imports, config loading, and utility functions for path resolution and config parsing.

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field, asdict

import polars as pl
import yaml
import logging
import time
import numpy as np
from datetime import date
from pandas_datareader import data as web

# ANSI color codes for enhanced logging
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Text colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    
    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'

class ColorFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages"""
    
    def __init__(self):
        super().__init__()
        
    def format(self, record):
        # Color mapping for log levels
        level_colors = {
            'DEBUG': Colors.DIM + Colors.WHITE,
            'INFO': Colors.GREEN,
            'WARNING': Colors.YELLOW,
            'ERROR': Colors.BRIGHT_RED,
            'CRITICAL': Colors.BG_RED + Colors.WHITE
        }
        
        # Format timestamp in dark grey
        timestamp = f"{Colors.DIM}{Colors.BLACK}{self.formatTime(record)}{Colors.RESET}"
        
        # Color the log level
        level_color = level_colors.get(record.levelname, Colors.WHITE)
        level_text = f"{level_color}{record.levelname}{Colors.RESET}"
        
        # Enhanced message formatting with smart highlighting
        message = self.enhance_message(record.getMessage())
        
        return f"{timestamp} - {level_text} - {message}"
    
    def enhance_message(self, message: str) -> str:
        """Add smart highlighting to important parts of log messages, and shorten repo paths."""
        import re, os
        # 1) Shorten absolute repo paths to repo-relative (e.g., data/raw/...)
        def _shorten_abs_repo_paths(msg: str) -> str:
            # Match absolute paths ending with common data/code extensions (supports .csv.gz)
            pattern = r'(/[^/\s]+(?:/[^/\s]+)*\.(?:csv(?:\.gz)?|dta|json|xlsx?|ya?ml|parquet|feather|gz))'
            def _repl(m: re.Match) -> str:
                p = m.group(1)
                try:
                    root = str(REPO_ROOT)
                    if p.startswith(root):
                        rel = os.path.relpath(p, root).replace(os.sep, '/')
                        return rel  # drop leading slash to be repo-relative
                except Exception:
                    pass
                return p
            return re.sub(pattern, _repl, msg)

        message = _shorten_abs_repo_paths(message)

        # 2) Highlight numbers with commas (like row counts)
        message = re.sub(r'(\d{1,3}(?:,\d{3})+)', f'{Colors.BRIGHT_CYAN}\\1{Colors.RESET}', message)
        
        # Highlight large numbers without commas
        message = re.sub(r'(\d{4,})', f'{Colors.BRIGHT_CYAN}\\1{Colors.RESET}', message)
        
        # Highlight percentages
        message = re.sub(r'(\d+\.?\d*%)', f'{Colors.BRIGHT_YELLOW}\\1{Colors.RESET}', message)
        
        # 3) Highlight file paths (absolute or repo-relative)
        path_pattern = r'((?:/|)(?:[^/\s]+/)*[^/\s]+\.(?:csv(?:\.gz)?|dta|json|xlsx?|ya?ml|parquet|feather|gz))'
        message = re.sub(path_pattern, f'{Colors.CYAN}\\1{Colors.RESET}', message)
        
        # Highlight important keywords
        keywords = {
            'completed': Colors.BRIGHT_GREEN,
            'failed': Colors.BRIGHT_RED,
            'error': Colors.BRIGHT_RED,
            'success': Colors.BRIGHT_GREEN,
            'warning': Colors.YELLOW,
            'missing': Colors.YELLOW,
            'found': Colors.GREEN,
            'saved': Colors.GREEN,
            'loaded': Colors.GREEN,
            'created': Colors.GREEN,
            'processed': Colors.GREEN,
            'filtered': Colors.BLUE,
            'joined': Colors.BLUE,
            'mapped': Colors.BLUE,
            'aggregated': Colors.BLUE,
        }
        
        for keyword, color in keywords.items():
            pattern = re.compile(rf'\b({keyword})\b', re.IGNORECASE)
            message = pattern.sub(f'{color}\\1{Colors.RESET}', message)
        
        # Highlight data processing steps
        if 'Reweighting:' in message:
            message = message.replace('Reweighting:', f'{Colors.BOLD}{Colors.MAGENTA}Reweighting:{Colors.RESET}')
        
        if 'Sanity:' in message:
            message = message.replace('Sanity:', f'{Colors.BOLD}{Colors.CYAN}Sanity:{Colors.RESET}')
        
        if 'Bridge:' in message:
            message = message.replace('Bridge:', f'{Colors.BOLD}{Colors.BLUE}Bridge:{Colors.RESET}')
        
        if 'CPS:' in message:
            message = message.replace('CPS:', f'{Colors.BOLD}{Colors.GREEN}CPS:{Colors.RESET}')
        
        if 'ATUS:' in message:
            message = message.replace('ATUS:', f'{Colors.BOLD}{Colors.YELLOW}ATUS:{Colors.RESET}')
        
        # Highlight targets vs actual values
        if 'target' in message.lower():
            message = re.sub(r'(target[^\d]*)([\d.]+%?)', f'\\1{Colors.BRIGHT_BLUE}\\2{Colors.RESET}', message)
        
        return message

# Set up colored logging
def setup_colored_logging():
    """Configure colored logging for the module"""
    # Remove existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create console handler with color formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter())
    
    logging.root.addHandler(console_handler)
    logging.root.setLevel(logging.INFO)
    
    return logging.getLogger(__name__)

# Initialize colored logging
logger = setup_colored_logging()
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
# Default config search paths
_CFG_SEARCH = [
    REPO_ROOT / "src" / "data" / "config.yml",
    REPO_ROOT / "src" / "empirical" / "config.yml",
    REPO_ROOT / "config.yml",
]


# ---------------- Configuration object model ---------------- #

@dataclass
class DatasetConfig:
    collection: str
    enabled: bool = True
    years_start: Optional[int] = None
    years_end: Optional[int] = None
    variables: list[str] = field(default_factory=list)
    samples: Optional[Any] = None
    sample_pattern: Optional[str] = None
    max_samples_per_extract: int = 12
    output_filename: Optional[str] = None
    # Specific to ATUS micro output used for CPS merge
    atus_micro_output_filename: Optional[str] = None


@dataclass
class Config:
    # API auth (optional)
    api_key: Optional[str]
    api_key_path: Optional[str]
    api_key_json_field: Optional[str]
    # Paths
    raw_dir: Path
    processed_dir: Path
    aux_dir: Path
    # Datasets
    datasets: Dict[str, DatasetConfig] = field(default_factory=dict)
    # Extra sections from YAML we still reference directly
    crosswalks: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Path] = field(default_factory=dict)
    project_root: Path = field(default=REPO_ROOT)

    def get_output_path(self, key: str, default: Path) -> Path:
        p = self.outputs.get(key)
        return p if p is not None else default


def _coerce_to_path(p: Any, base: Path) -> Path:
    if isinstance(p, Path):
        return p
    if p is None:
        return base
    return (base / str(p)).resolve() if not str(p).startswith("/") else Path(str(p)).resolve()


def load_config(config_path: Optional[str | Path] = None) -> Config:
    """
    Load configuration from YAML and return a Config object.

    If config_path is None, searches a few common locations in the repo.
    """
    cfg_path: Optional[Path] = None
    if config_path is not None:
        cfg_path = Path(config_path)
    else:
        for p in _CFG_SEARCH:
            if p.exists():
                cfg_path = p
                break
    if cfg_path is None:
        raise FileNotFoundError("config.yml not found in expected locations.")

    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # Paths with sensible defaults relative to project root
    paths = raw.get("paths", {})
    raw_dir = _coerce_to_path(paths.get("raw_dir", "data/raw"), REPO_ROOT)
    processed_dir = _coerce_to_path(paths.get("processed_dir", "data/processed"), REPO_ROOT)
    aux_dir = _coerce_to_path(paths.get("aux_dir", "data/aux"), REPO_ROOT)

    # Outputs (if present) -> coerce to absolute Paths
    outputs: Dict[str, Path] = {}
    for k, v in (paths.get("outputs", {}) or {}).items():
        try:
            outputs[k] = _coerce_to_path(v, REPO_ROOT)
        except Exception:
            # Skip bad entries; caller can use defaults
            continue

    # API config: support both authentication and ipums sections
    auth_cfg = raw.get("authentication", {}) or {}
    ipums_cfg = raw.get("ipums", {}) or {}
    api_key = ipums_cfg.get("api_key") or auth_cfg.get("api_key")
    api_key_path = auth_cfg.get("api_key_path") or ipums_cfg.get("api_key_path")
    api_key_json_field = auth_cfg.get("ipums_api_key_field") or ipums_cfg.get("api_key_json_field", "ipums")

    # Datasets parsing
    ds_cfg: Dict[str, DatasetConfig] = {}
    for name, d in (raw.get("datasets", {}) or {}).items():
        ds_cfg[name] = DatasetConfig(
            collection=d.get("collection", name),
            enabled=bool(d.get("enabled", True)),
            years_start=(d.get("years", {}) or {}).get("start"),
            years_end=(d.get("years", {}) or {}).get("end"),
            variables=d.get("variables", []) or [],
            samples=d.get("samples"),
            sample_pattern=d.get("sample_pattern"),
            max_samples_per_extract=int(d.get("max_samples_per_extract", 12)),
            output_filename=d.get("output_filename"),
            atus_micro_output_filename=d.get("micro_output_filename"),
        )

    return Config(
        api_key=api_key,
        api_key_path=api_key_path,
        api_key_json_field=api_key_json_field,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        aux_dir=aux_dir,
        datasets=ds_cfg,
        crosswalks=raw.get("crosswalks", {}) or {},
        outputs=outputs,
        project_root=REPO_ROOT,
    )


def resolve_tokens(s: str, cfg: Any) -> str:
    """Resolve ${a.b.c} tokens in a string using the config dict."""
    if not isinstance(s, str):
        return s
    # Build a lookup mapping whether cfg is dict or Config
    def _to_lookup(cfg_any: Any) -> Dict[str, Any]:
        if isinstance(cfg_any, Config):
            base = asdict(cfg_any)
            # Convert Paths to strings for token replacement
            def _pathify(x: Any) -> Any:
                if isinstance(x, Path):
                    return str(x)
                if isinstance(x, dict):
                    return {k: _pathify(v) for k, v in x.items()}
                if isinstance(x, list):
                    return [_pathify(v) for v in x]
                return x
            base = _pathify(base)
            # Provide a 'paths' section similar to YAML for compatibility
            base.setdefault("paths", {})
            base["paths"].update({
                "raw_dir": str(cfg_any.raw_dir),
                "processed_dir": str(cfg_any.processed_dir),
                "aux_dir": str(cfg_any.aux_dir),
                "outputs": {k: str(v) for k, v in cfg_any.outputs.items()},
            })
            base["crosswalks"] = base.get("crosswalks", {})
            return base
        elif isinstance(cfg_any, dict):
            return cfg_any
        else:
            return {}

    lookup = _to_lookup(cfg)
    out = s
    while "${" in out:
        start = out.find("${")
        end = out.find("}", start)
        if end == -1:
            break
        token = out[start + 2 : end]
        # Resolve dotted path in cfg
        cur: Any = lookup
        for part in token.split('.'):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                cur = None
                break
        if cur is None:
            # Drop token if unresolved
            out = out[:start] + out[end + 1 :]
        else:
            out = out[:start] + str(cur) + out[end + 1 :]
    return out


def map_with_crosswalk(
                        df: pl.DataFrame, # The input DataFrame
                        cw_df: pl.DataFrame, # The crosswalk DataFrame
                        source_col: str, # The source column to map
                        cw_key: str, # The key column in the crosswalk
                        cw_value: str, # The value column in the crosswalk
                        out_col: str | None = None # The output column name
                    ) -> pl.DataFrame:
    """
    Maps values in a DataFrame column to new values using a crosswalk DataFrame via a left join.

    Args:
        df (pl.DataFrame): The input DataFrame containing the source column to map.
        cw_df (pl.DataFrame): The crosswalk DataFrame containing mapping keys and values.
        source_col (str): The name of the column in `df` to map.
        cw_key (str): The name of the key column in `cw_df` to join on.
        cw_value (str): The name of the value column in `cw_df` to map to.
        out_col (str | None, optional): The name of the output column. If None, overwrites `source_col`.

    Returns:
        pl.DataFrame: A new DataFrame with the mapped column. If a mapping is not found, retains the original value.
    """
    # Prepare columns for join by casting to string and stripping whitespace
    out_col = out_col or source_col
    left = df.with_columns(pl.col(source_col).cast(pl.Utf8).str.strip_chars().alias("__key"))
    right = cw_df.select([
        pl.col(cw_key).cast(pl.Utf8).str.strip_chars().alias("__key"),
        pl.col(cw_value).alias("__val"),
    ])
    # Perform left join on the prepared key
    joined = left.join(right, on="__key", how="left")
    # If mapping found, use mapped value; else retain original
    joined = joined.with_columns(
        pl.when(pl.col("__val").is_not_null()).then(pl.col("__val")).otherwise(pl.col(source_col)).alias(out_col)
    )
    # Drop temporary columns used for joining
    return joined.drop(["__key", "__val"], strict=False)


def add_soc_groupings(df: pl.DataFrame, occ_col: str, prefix: str | None = None) -> pl.DataFrame:
    """
    Create SOC groupings using only occ_col text:
    - Normalize: replace X/x -> 0, keep first 7 chars (XX-XXXX).
    - detailed: last digit != 0 (e.g., 11-1111)
    - broad: last digit == 0 and second-to-last != 0 (e.g., 11-1110)
    - minor: last three digits == 000 and 4th-to-last != 0 (e.g., 11-1000)
    Produces {prefix}_detailed, {prefix}_broad, {prefix}_minor.
    """
    prefix = prefix or occ_col

    # Normalize to 7-char SOC and digits-only helper, treating "0" as null
    out = (
        df.with_columns([
            pl.col(occ_col).cast(pl.Utf8).str.strip_chars()
                .str.replace_all(r"(?i)X", "0")
                .str.slice(0, 7)
                .pipe(lambda x: pl.when(x == "0").then(None).otherwise(x))
                .alias("__occ"),
        ])
    )

    # Masks
    detailed = (pl.col("__occ").str.slice(-1) != "0")
    broad = (pl.col("__occ").str.slice(-1) == "0") & (pl.col("__occ").str.slice(-2, 1) != "0")
    minor = (pl.col("__occ").str.slice(-3) == "000") & (pl.col("__occ").str.slice(-4, 1) != "0")

    # Outputs
    out = out.with_columns([
        pl.when(detailed).then(pl.col("__occ")).otherwise(None).alias(f"{prefix}_detailed"),
        pl.when(broad).then(pl.col("__occ"))
         .when(detailed).then(pl.col("__occ").str.slice(0, 6) + pl.lit("0"))
         .otherwise(None).alias(f"{prefix}_broad"),
        pl.when(minor).then(pl.col("__occ"))
         .when(detailed).then(pl.col("__occ").str.slice(0, 4) + pl.lit("000"))
         .when(detailed).then(pl.col("__occ").str.slice(0, 4) + pl.lit("000"))
         .otherwise(None).alias(f"{prefix}_minor"),
    ])

    return out.drop(["__occ", "__digits"], strict=False)


def assign_teleworkability(
                            df: pl.DataFrame,
                            tw_path: Path,
                            occ_col: str,
                            map_cfg: dict,
                            out_prefix: str = "TELEWORKABLE_OCSSOC"
                        ) -> pl.DataFrame:
    
    # 1) Load teleworkability file
    if not tw_path.exists():
        return df
    tw = pl.read_csv(str(tw_path))

    # 2) Use config for mapping columns
    cw_key = map_cfg.get("crosswalk_key", "OCC_CODE")
    cw_value = map_cfg.get("target_column", "TELEWORKABLE")

    # 3) Normalize OCC codes in teleworkability crosswalk to 7-char SOC and X->0
    tw = tw.select([
        pl.col(cw_key).cast(pl.Utf8).str.strip_chars()
            .str.slice(0, 7)
            .str.replace_all(r"(?i)X", "0")
            .alias("OCC"),
        pl.col(cw_value).cast(pl.Float64).alias("TELEWORKABLE"),
    ])

    # 4) Ensure single row per detailed OCC (average if duplicates)
    tw_det = (
        tw.group_by("OCC").agg(pl.col("TELEWORKABLE").mean().alias(f"{out_prefix}_detailed"))
    )

    # 5) Build broad/minor keys and aggregate
    # Broad: 15-1121 -> 15-1120
    tw_broad = (
        tw_det.with_columns((pl.col("OCC").str.slice(0, 6) + pl.lit("0")).alias("OCC_broad"))
            .group_by("OCC_broad")
            .agg(pl.col(f"{out_prefix}_detailed").mean().alias(f"{out_prefix}_broad"))
    )
    # Minor: 15-1121 -> 15-1100
    tw_minor = (
        tw_det.with_columns((pl.col("OCC").str.slice(0, 4) + pl.lit("000")).alias("OCC_minor"))
            .group_by("OCC_minor")
            .agg(pl.col(f"{out_prefix}_detailed").mean().alias(f"{out_prefix}_minor"))
    )

    # 6) Determine base occ prefix (handle if caller passed *_detailed/_broad/_minor)
    if occ_col.endswith("_detailed"):
        base = occ_col[: -len("_detailed")]
    elif occ_col.endswith("_broad"):
        base = occ_col[: -len("_broad")]
    elif occ_col.endswith("_minor"):
        base = occ_col[: -len("_minor")]
    else:
        base = occ_col

    # 7) Normalize left join keys to same format (7-char SOC, X->0)
    out = df
    # Detailed
    det_col = f"{base}_detailed"
    if det_col in out.columns:
        out = out.with_columns(
            pl.col(det_col).cast(pl.Utf8).str.strip_chars()
                .str.slice(0, 7)
                .str.replace_all(r"(?i)X", "0")
                .alias("__det")
        )
        out = out.join(tw_det, left_on="__det", right_on="OCC", how="left").drop(["OCC"], strict=False)

    # Broad
    broad_col = f"{base}_broad"
    if broad_col in out.columns:
        out = out.with_columns(
            pl.col(broad_col).cast(pl.Utf8).str.strip_chars()
                .str.slice(0, 7)
                .str.replace_all(r"(?i)X", "0")
                .alias("__broad")
        )
        out = out.join(tw_broad, left_on="__broad", right_on="OCC_broad", how="left").drop(["OCC_broad"], strict=False)

    # Minor
    minor_col = f"{base}_minor"
    if minor_col in out.columns:
        out = out.with_columns(
            pl.col(minor_col).cast(pl.Utf8).str.strip_chars()
                .str.slice(0, 7)
                .str.replace_all(r"(?i)X", "0")
                .alias("__minor")
        )
        out = out.join(tw_minor, left_on="__minor", right_on="OCC_minor", how="left").drop(["OCC_minor"], strict=False)

    # 8) Clean up temp columns
    return out.drop(["__det", "__broad", "__minor"], strict=False)



def reweight_poststrat(
    data: pl.DataFrame,
    weight_col: str = "WTFINL",
    response_col: str = "WAGE",
    min_wage: float = 0,
    cell_vars: list | None = None,
    cap_factor: float | None = 5.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create adjusted weights to account for item non-response in a response variable via post-stratification.

    Returns
    -------
    target_sample : pl.DataFrame
        Universe after non-response agnostic filters (keeps original weight).
    final_sample_reweighted : pl.DataFrame
        Subset with valid wage information and an adjusted weight column 'WTFINL_ADJ'.
    """
    t0 = time.time()
    if cell_vars is None:
        cell_vars = ["YEAR", "SEX", "EDUC", "RACE", "AGE_GROUP"]

    # Ensure needed columns exist; drop missing ones (except AGE_GROUP which we derive)
    missing_cols = [c for c in cell_vars if c not in data.columns and c != "AGE_GROUP"]
    if missing_cols:
        logging.warning(f"Reweighting: dropping missing stratification columns: {missing_cols}")
        cell_vars = [c for c in cell_vars if (c in data.columns) or (c == "AGE_GROUP")]
        if not any(c != "AGE_GROUP" for c in cell_vars):
            logging.warning("Reweighting skipped: no stratification columns available.")
            return data, data.filter(pl.lit(False))

    # Create AGE_GROUP if not present
    if "AGE_GROUP" not in data.columns:
        data = data.with_columns(
            pl.when(pl.col("AGE") < 18).then(pl.lit(None))
            .when(pl.col("AGE") <= 24).then(pl.lit('18-24'))
            .when(pl.col("AGE") <= 34).then(pl.lit('25-34'))
            .when(pl.col("AGE") <= 44).then(pl.lit('35-44'))
            .when(pl.col("AGE") <= 54).then(pl.lit('45-54'))
            .when(pl.col("AGE") <= 64).then(pl.lit('55-64'))
            .otherwise(pl.lit('65+')).alias('AGE_GROUP')
        )

    # Target sample: keep non-missing weights
    target_sample = data.filter(pl.col(weight_col).is_not_null())
    logging.info(f"Reweighting: target sample rows = {target_sample.height:,}")

    # Final sample (requires response variable present, e.g., wage)
    final_sample = target_sample.filter(
        pl.col(response_col).is_not_null() & (pl.col(response_col) >= min_wage)
    )
    logging.info(f"Reweighting: final (response-present) sample rows = {final_sample.height:,}")

    if final_sample.height == 0:
        logging.warning("Reweighting skipped: no observations with valid wage.")
        return target_sample, final_sample

    # Aggregate population totals
    target_totals = (
        target_sample.group_by(cell_vars)
        .agg(pl.col(weight_col).sum().alias("TARGET_POP"))
    )
    observed_totals = (
        final_sample.group_by(cell_vars)
        .agg(pl.col(weight_col).sum().alias("OBSERVED_POP"))
    )

    adjustment = target_totals.join(observed_totals, on=cell_vars, how="left")
    adjustment = adjustment.with_columns(
        pl.when((pl.col("OBSERVED_POP").is_null()) | (pl.col("OBSERVED_POP") <= 0))
        .then(pl.lit(1.0))
        .otherwise(pl.col("TARGET_POP") / pl.col("OBSERVED_POP"))
        .alias("ADJ_FACTOR_RAW")
    )

    # Optional capping to avoid extreme inflation
    if cap_factor is not None:
        adjustment = adjustment.with_columns(
            pl.when(pl.col("ADJ_FACTOR_RAW") > cap_factor)
            .then(pl.lit(cap_factor))
            .otherwise(pl.col("ADJ_FACTOR_RAW"))
            .alias("ADJ_FACTOR")
        )
        n_capped = adjustment.filter(pl.col("ADJ_FACTOR_RAW") > cap_factor).height
        if n_capped > 0:
            logging.info(f"Reweighting: capped {n_capped} cell factors at {cap_factor}.")
    else:
        adjustment = adjustment.rename({"ADJ_FACTOR_RAW": "ADJ_FACTOR"})

    # Merge factors into final_sample
    final_sample = final_sample.join(
        adjustment.select(cell_vars + ["ADJ_FACTOR"]),
        on=cell_vars,
        how="left"
    )

    # Apply factor
    final_sample = final_sample.with_columns(
        (pl.col(weight_col) * pl.col("ADJ_FACTOR")).alias(f"{weight_col}_ADJ")
    )

    # Year-level calibration (ratio adjust per year to match target total exactly)
    year_tot_target = target_sample.group_by("YEAR").agg(pl.col(weight_col).sum().alias("TARGET_YEAR_SUM"))
    year_tot_adj = final_sample.group_by("YEAR").agg(pl.col(f"{weight_col}_ADJ").sum().alias("ADJ_YEAR_SUM"))
    year_scalers = year_tot_target.join(year_tot_adj, on="YEAR", how="inner").with_columns(
        pl.when(pl.col("ADJ_YEAR_SUM") <= 0)
        .then(pl.lit(1.0))
        .otherwise(pl.col("TARGET_YEAR_SUM") / pl.col("ADJ_YEAR_SUM"))
        .alias("YEAR_SCALE")
    )
    final_sample = final_sample.join(year_scalers.select(["YEAR", "YEAR_SCALE"]), on="YEAR", how="left")
    final_sample = final_sample.with_columns(
        (pl.col(f"{weight_col}_ADJ") * pl.col("YEAR_SCALE")).alias(f"{weight_col}_ADJ")
    ).drop("YEAR_SCALE")

    # Diagnostics
    tot_target = target_sample[weight_col].sum()
    tot_adj = final_sample[f"{weight_col}_ADJ"].sum()
    diff_pct = (tot_adj - tot_target) / tot_target if tot_target else 0
    logging.info(f"Reweighting: total target weight = {tot_target:,.2f}; adjusted sample weight = {tot_adj:,.2f}; diff={diff_pct:.4%}")

    logging.info(f"Reweighting completed in {time.time() - t0:.2f} seconds.")
    return target_sample, final_sample


def fetch_cpi_series() -> pl.DataFrame:
    """Fetch monthly CPIAUCSL from FRED via pandas-datareader and return YEAR, MONTH, CPI as Polars.

    Requires pandas-datareader. If unavailable or request fails, returns an empty DataFrame.
    """
    if web is None:
        logging.error("pandas-datareader is not installed. Install with: pip install pandas-datareader")
        return pl.DataFrame({"YEAR": pl.Series([], dtype=pl.Int64), "MONTH": pl.Series([], dtype=pl.Int64), "CPI": pl.Series([], dtype=pl.Float64)})
    try:
        start = date(1990, 1, 1)
        end = date.today()
        s = web.DataReader("CPIAUCSL", "fred", start, end)  # monthly CPI (seasonally adjusted)
        s = s.dropna()
        s = s.reset_index()  # columns: DATE, CPIAUCSL
        s["YEAR"] = s["DATE"].dt.year
        s["MONTH"] = s["DATE"].dt.month
        s = s.rename(columns={"CPIAUCSL": "CPI"})
        out_pd = s[["YEAR", "MONTH", "CPI"]].copy()
        return pl.from_pandas(out_pd).with_columns([
            pl.col("YEAR").cast(pl.Int64),
            pl.col("MONTH").cast(pl.Int64),
            pl.col("CPI").cast(pl.Float64),
        ])
    except Exception as e:
        logging.error(f"Failed to fetch CPIAUCSL from FRED: {e}")
        return pl.DataFrame({"YEAR": pl.Series([], dtype=pl.Int64), "MONTH": pl.Series([], dtype=pl.Int64), "CPI": pl.Series([], dtype=pl.Float64)})


def add_topcode_and_real_wages(data: pl.DataFrame) -> pl.DataFrame:
    """Identify top-coded wages, adjust, and deflate to 2019 real terms."""
    required_cols = ["YEAR", "MONTH", "WAGE", "HOURWAGE2", "EARNWEEK2"]
    for c in required_cols:
        if c not in data.columns:
            logging.warning(f"add_topcode_and_real_wages: missing column {c}; skipping real wage construction")
            return data

    # Ensure numeric types
    data = data.with_columns([
        pl.col("YEAR").cast(pl.Int64),
        pl.col("MONTH").cast(pl.Int64),
        pl.col("WAGE").cast(pl.Float64),
        pl.col("HOURWAGE2").cast(pl.Float64),
        pl.col("EARNWEEK2").cast(pl.Float64),
    ])

    # Preserve raw wage
    data = data.with_columns(pl.col("WAGE").alias("WAGE_RAW"))

    hour_topcode_dict = {
        (2023, 4, 4): 65.24, (2023, 4, 8): 63.34,
        (2023, 5, 4): 66.73, (2023, 5, 8): 66.03,
        (2023, 6, 4): 60.51, (2023, 6, 8): 59.44,
        (2023, 7, 4): 65.80, (2023, 7, 8): 63.34,
        (2023, 8, 4): 65.89, (2023, 8, 8): 64.22,
        (2023, 9, 4): 66.05, (2023, 9, 8): 67.06,
        (2023,10, 4): 69.44, (2023,10, 8): 66.86,
        (2023,11, 4): 69.84, (2023,11, 8): 64.49,
        (2023,12, 4): 68.39, (2023,12, 8): 61.43,
        (2024, 1, 4): 69.68, (2024, 1, 8): 66.92,
        (2024, 2, 4): 67.71, (2024, 2, 8): 68.93,
        (2024, 3, 4): 68.24, (2024, 3, 8): 65.20,
    }

    # Ensure MISH exists
    if "MISH" not in data.columns:
        logging.warning("MISH column not found; hour wage top-code identification will rely on generic rules only.")
        data = data.with_columns(pl.lit(None).cast(pl.Int64).alias("MISH"))
    else:
        data = data.with_columns(pl.col("MISH").cast(pl.Int64))

    # Explicit listed thresholds column
    data = data.with_columns(
        pl.struct(["YEAR", "MONTH", "MISH"]).map_elements(
            lambda s: hour_topcode_dict.get((s["YEAR"], s["MONTH"], s["MISH"]), None),
            return_dtype=pl.Float64
        ).alias("HOUR_TOPCODE_THRESHOLD")
    )

    # Monthly max and counts
    monthly_max = (
        data.group_by(["YEAR", "MONTH"]).agg([
            pl.col("WAGE").max().alias("MONTH_MAX_WAGE"),
            pl.len().alias("MONTH_COUNT"),
        ])
    )
    data = data.join(monthly_max, on=["YEAR", "MONTH"], how="left")

    # Fixed EARNWEEK2 threshold through Mar 2024
    data = data.with_columns([
        pl.when(((pl.col("YEAR") < 2024) | ((pl.col("YEAR") == 2024) & (pl.col("MONTH") <= 3))) & (pl.col("EARNWEEK2") >= 2884.61))
        .then(1).otherwise(0).alias("TOP_EARNWEEK_FLAG_FIXED"),
        pl.when(pl.col("HOUR_TOPCODE_THRESHOLD").is_not_null() & (pl.col("HOURWAGE2") == pl.col("HOUR_TOPCODE_THRESHOLD")))
        .then(1).otherwise(0).alias("TOP_HOUR_FLAG_LISTED"),
    ])

    max_counts = (
        data.filter(pl.col("WAGE") == pl.col("MONTH_MAX_WAGE"))
        .group_by(["YEAR", "MONTH"]).agg(pl.len().alias("MAX_WAGE_COUNT"))
    )
    data = data.join(max_counts, on=["YEAR", "MONTH"], how="left")
    data = data.with_columns([
        pl.when(((pl.col("YEAR") > 2024) | ((pl.col("YEAR") == 2024) & (pl.col("MONTH") >= 4))) &
                (pl.col("WAGE") == pl.col("MONTH_MAX_WAGE")) & (pl.col("MAX_WAGE_COUNT") >= 5))
        .then(1).otherwise(0).alias("TOP_DYNAMIC_FLAG")
    ])

    data = data.with_columns(
        ((pl.col("TOP_EARNWEEK_FLAG_FIXED") == 1) | (pl.col("TOP_HOUR_FLAG_LISTED") == 1) | (pl.col("TOP_DYNAMIC_FLAG") == 1))
        .cast(pl.Int64).alias("WAGE_TOPCODED_FLAG")
    )

    # Wage source flag
    data = data.with_columns(
        pl.when(pl.col("HOURWAGE2") != 999.99).then(2)
        .when(pl.col("EARNWEEK2").is_not_null()).then(3)
        .otherwise(0).alias("SOURCE_WAGE_FLAG")
    )

    # Adjust top-coded wage by 1.3
    data = data.with_columns(
        pl.when(pl.col("WAGE_TOPCODED_FLAG") == 1).then(pl.col("WAGE") * 1.3).otherwise(pl.col("WAGE")).alias("WAGE")
    )

    # CPI deflation to 2019 base
    cpi = fetch_cpi_series()
    if cpi.height > 0:
        base_2019 = cpi.filter(pl.col("YEAR") == 2019).get_column("CPI").mean()
        if base_2019 is None or (isinstance(base_2019, float) and np.isnan(base_2019)):
            logging.warning("Base year 2019 CPI missing; skipping deflation")
        else:
            cpi = cpi.with_columns((pl.col("CPI") / base_2019).alias("CPI_INDEX"))
            data = data.join(cpi.select(["YEAR", "MONTH", "CPI", "CPI_INDEX"]), on=["YEAR", "MONTH"], how="left")
            data = data.with_columns([
                (pl.col("WAGE") / pl.col("CPI_INDEX")).alias("WAGE_REAL"),
                pl.when(pl.col("WAGE").is_not_null() & (pl.col("WAGE") > 0))
                .then((pl.col("WAGE") / pl.col("CPI_INDEX")).log())
                .otherwise(None).alias("LOG_WAGE_REAL"),
            ])
    else:
        logging.warning("CPI series empty; falling back to nominal wages for real/log wage variables.")
        data = data.with_columns([
            pl.col("WAGE").alias("WAGE_REAL"),
            pl.when(pl.col("WAGE").is_not_null() & (pl.col("WAGE") > 0))
             .then(pl.col("WAGE").log())
             .otherwise(None).alias("LOG_WAGE_REAL"),
        ])

    # Diagnostics
    try:
        top_share = data.filter(pl.col("WAGE_TOPCODED_FLAG") == 1).height / data.height if data.height else 0
        logging.info(f"Top-coded wage share (unweighted) = {top_share:.4%}")
    except Exception:
        pass

    return data

def atus_pipeline(cfg: Config) -> Path:
    raw_dir = cfg.raw_dir  / "atus"
    atus_ds = cfg.datasets.get("atus")
    atus_micro_name = f"atus_{atus_ds.years_start or 'unknown'}_{atus_ds.years_end or 'present'}.csv.gz"
    atus_micro_path = raw_dir / atus_micro_name
    if not atus_micro_path.exists():
        raise FileNotFoundError(f"ATUS micro not found: {atus_micro_path}")
    df = pl.read_csv(str(atus_micro_path))
    logger.info(f"ATUS: Loaded {len(df):,} rows from {atus_micro_path}")

    # Ensure core columns exist; DATE/HOLIDAY are optional with safe fallbacks
    for c in ["YEAR", "CPSIDP", "WHERE", "ACTIVITY", "DURATION"]:
        if c not in df.columns:
            raise RuntimeError(f"Missing required ATUS column: {c}")

    # Define location and work codes (IPUMS WHERE)
    home_where = [101]         # Respondent's home or yard
    workplace_where = [102]    # Respondent's workplace
    where_unknown = [89, 99, 9997, 9998, 9999]  # unspecified / NA

    # Drop any records with unknown location
    df = df.filter(~pl.col("WHERE").is_in(where_unknown))

    # Drop any records where diary day was a holiday if available; treat non-integer as integer
    if "HOLIDAY" in df.columns:
        df = df.with_columns(pl.col("HOLIDAY").cast(pl.Int64, strict=False))
        df = df.filter(~pl.col("HOLIDAY").is_in([1]))
    else:
        logger.warning("ATUS: HOLIDAY column not found; not excluding holidays.")

    # 1) Parse YYYYMMDD to extract proper Date, Month, and weekday
    if "DATE" in df.columns:
        df = df.with_columns([
            pl.when(pl.col("DATE").cast(pl.Utf8).str.len_chars() >= 8)
              .then(pl.col("DATE").cast(pl.Utf8).str.slice(0,8).str.strptime(pl.Date, "%Y%m%d", strict=False))
              .otherwise(None).alias("DATE_PARSED")
        ])
        df = df.with_columns([
            pl.when(pl.col("DATE_PARSED").is_not_null()).then(pl.col("DATE_PARSED").dt.month()).otherwise(pl.lit(6)).alias("MONTH"),
            pl.when(pl.col("DATE_PARSED").is_not_null()).then(pl.col("DATE_PARSED").dt.weekday()).otherwise(pl.lit(2)).alias("DAY_OF_WEEK")
        ])
        # Keep original DATE column, drop the temporary parsed one
        df = df.drop("DATE_PARSED")
    else:
        logger.warning("ATUS: DATE column not found; assuming June (MONTH=6) and weekdays.")
       

    # Convert MONTH to Int32 to match CPS schema
    df = df.with_columns(pl.col("MONTH").cast(pl.Int32))

    # Identify main‐job work (501xx) and home location
    is_work = (pl.col("ACTIVITY") // 100 == 501)
    is_home = pl.col("WHERE").is_in(home_where)

    # Compute minutes by type
    df = df.with_columns([
        pl.when(is_work & is_home)
          .then(pl.col("DURATION"))
          .otherwise(0)
          .alias("remote_minutes"),
        pl.when(is_work)
          .then(pl.col("DURATION"))
          .otherwise(0)
          .alias("work_minutes"),
    ])

    # Create a weight column by coalescing WT06 and WT20, and drop WT06 and WT20
    df = df.with_columns(
        pl.coalesce(pl.col("WT06"), pl.col("WT20")).alias("ATUS_WT")
    ).drop("WT06", "WT20", strict=False)

    # Aggregate to person‐year and derive flags
    link = (
        df.group_by(["YEAR", "MONTH", "CPSIDP"]).agg([
            pl.col("remote_minutes").sum().alias("remote_minutes"),
            pl.col("work_minutes").sum().alias("work_minutes"),
            pl.col("ATUS_WT").mean().alias("ATUS_WT")
        ])
        .with_columns([
            (pl.col("work_minutes") > 0).alias("worked"),
            (pl.col("remote_minutes") > 0).alias("any_remote"),
            ((pl.col("remote_minutes") == pl.col("work_minutes")) & (pl.col("work_minutes") > 0)).alias("full_remote"),
            ((pl.col("remote_minutes") == 0) & (pl.col("work_minutes") > 0)).alias("full_inperson"),
            ((pl.col("remote_minutes") > 0) & (pl.col("remote_minutes") < pl.col("work_minutes"))).alias("hybrid"),
        ])
        .filter(pl.col("worked"))
    )
    logger.info(f"ATUS: Aggregated to {len(link):,} person-year records")

    out_link = cfg.get_output_path("atus_link", cfg.processed_dir / "atus_link.csv")
    out_link.parent.mkdir(parents=True, exist_ok=True)
    link.write_csv(str(out_link))
    logger.info(f"ATUS: Saved linkage file to {out_link}")
    return out_link


def acs_pipeline(cfg: Config) -> Path:
    raw_dir = cfg.raw_dir / "acs"
    acs_ds = cfg.datasets.get("acs")
    acs_file = acs_ds.output_filename if (acs_ds and acs_ds.output_filename) else "acs_2013_present_minimal.csv"
    raw_path = raw_dir / acs_file
    if not raw_path.exists():
        raise FileNotFoundError(f"ACS raw not found: {raw_path}")

    df = pl.read_csv(str(raw_path))
    logger.info(f"ACS: Loaded {len(df):,} rows from {raw_path}")

    # Filters
    if "UHRSWORK" in df.columns:
        df = df.filter(pl.col("UHRSWORK") >= 35)
        logger.info(f"ACS: After filtering for full-time workers (>=35 hrs): {len(df):,} rows")
    if set(["INCWAGE", "UHRSWORK"]).issubset(df.columns):
        df = df.with_columns((pl.col("INCWAGE") / (pl.col("UHRSWORK").clip_min(1) * 52)).alias("WAGE"))
    if "WAGE" in df.columns:
        df = df.filter(pl.col("WAGE") >= 5)
        logger.info(f"ACS: After filtering for wage >= $5/hour: {len(df):,} rows")
    if "CLASSWKR" in df.columns:
        df = df.filter(pl.col("CLASSWKR").cast(pl.Utf8) == "2")
        logger.info(f"ACS: After filtering for private sector workers: {len(df):,} rows")
    if "INDNAICS" in df.columns:
        df = df.with_columns(pl.col("INDNAICS").cast(pl.Utf8).str.strip_chars())
        df = df.filter(pl.col("INDNAICS") != "0")
        df = df.filter(~pl.col("INDNAICS").is_in(["928110P1","928110P2","928110P3","928110P4","928110P5","928110P6","928110P7"]))
        df = df.filter(pl.col("INDNAICS") != "999920")
        logger.info(f"ACS: After filtering for valid industries: {len(df):,} rows")

    # WFH flag
    if "TRANWORK" in df.columns:
        df = df.with_columns((pl.col("TRANWORK").cast(pl.Utf8) == "80").cast(pl.Int8).alias("WFH"))
        logger.info(f"ACS: Added WFH flag")

    # SOC groupings and teleworkability (ACS OCCSOC present)
    cw_cfg = cfg.crosswalks
    if "OCCSOC" in df.columns:
        df = df.with_columns(pl.col("OCCSOC").cast(pl.Utf8).str.strip_chars())
        df = add_soc_groupings(df, "OCCSOC", prefix="OCCSOC")
        tw_path = Path(resolve_tokens(cw_cfg["teleworkability"]["path"], cfg))
        tw_map_cfg = cw_cfg["teleworkability"]["map"]
        df = assign_teleworkability(df, tw_path, "OCCSOC", tw_map_cfg, out_prefix="TELEWORKABLE_OCSSOC")
        logger.info(f"ACS: Added SOC groupings and teleworkability")

    # PUMA->CBSA and state mappings
    if set(["STATEFIP", "PUMA"]).issubset(df.columns):
        df = df.with_columns([
            pl.col("STATEFIP").cast(pl.Utf8).str.pad_start(2, "0"),
            pl.col("PUMA").cast(pl.Utf8).str.pad_start(5, "0"),
        ])
        df = df.with_columns((pl.col("STATEFIP") + pl.lit("-") + pl.col("PUMA")).alias("state_puma"))
        puma_cfg = cw_cfg["puma_to_cbsa"]["map"]
        puma_path = Path(resolve_tokens(cw_cfg["puma_to_cbsa"]["path"], cfg))
        puma_cw = pl.read_csv(str(puma_path))
        df = map_with_crosswalk(df, puma_cw, puma_cfg["source_column"], puma_cfg["crosswalk_key"], puma_cfg["target_column"], out_col="cbsa20")
        logger.info(f"ACS: Mapped PUMA to CBSA")

    # State fips-> name/div
    state_cfg = cw_cfg["state_censusdiv"]
    state_path = Path(resolve_tokens(state_cfg["path"], cfg))
    state_cw = pl.read_csv(str(state_path))
    has_statefip = "STATEFIP" in df.columns
    df = df.with_columns(
        (pl.col("STATEFIP") if has_statefip else pl.col("state_fips")).cast(pl.Utf8).str.pad_start(2, "0").alias("state_fips")
    )
    f2d = state_cfg["map"]["fips_to_div"]
    df = map_with_crosswalk(df, state_cw, "state_fips", f2d["crosswalk_key"], f2d["target_column"], out_col="censusdiv")
    f2n = state_cfg["map"]["fips_to_name"]
    df = map_with_crosswalk(df, state_cw, "state_fips", f2n["crosswalk_key"], f2n["target_column"], out_col="state_name")
    logger.info(f"ACS: Mapped state FIPS to names and divisions")

    out_path = cfg.get_output_path("acs", (cfg.processed_dir / "acs" / "acs_processed.csv").resolve())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(str(out_path))
    logger.info(f"ACS: Final dataset has {len(df):,} rows, saved to {out_path}")
    return out_path


def cps_pipeline(cfg: Config) -> Path:
    raw_dir = cfg.raw_dir / "cps"
    cps_ds = cfg.datasets.get("cps")
    cps_micro_name = f"cps_{cps_ds.years_start or 'unknown'}_{cps_ds.years_end or 'present'}.csv.gz"
    cps_micro_path = raw_dir / cps_micro_name

    if not cps_micro_path.exists():
        raise FileNotFoundError(f"CPS raw not found: {cps_micro_path}")

    cps = pl.read_csv(str(cps_micro_path))
    logger.info(f"CPS: Loaded {len(cps):,} rows from {cps_micro_path}")

    for col in ["YEAR", "MONTH"]:
        if col in cps.columns:
            cps = cps.with_columns(pl.col(col).cast(pl.Int32))


    # Industry mapping (CPS IND -> NAICS definitive)
    cw_cfg = cfg.crosswalks
    ind_cfg = cw_cfg["cps_industry"]
    ind_path = Path(resolve_tokens(ind_cfg["path"], cfg))
    ind_cw = pl.read_csv(str(ind_path))
    cps = map_with_crosswalk(cps, ind_cw, ind_cfg["map"]["source_column"], ind_cfg["map"]["crosswalk_key"], ind_cfg["map"]["target_column"], out_col=ind_cfg.get("output_column", "IND_MAPPED"))
    logger.info(f"CPS: Mapped industry codes")
    
    # Ensure a common alias for downstream: if only IND_MAPPED exists, create INDNAICS alias
    if ("INDNAICS" not in cps.columns) and ("IND_MAPPED" in cps.columns):
        cps = cps.with_columns(pl.col("IND_MAPPED").alias("INDNAICS"))

    # OCC -> SOC mapping (assumes one-to-one mapping in the crosswalk)
    occsoc_cfg = cw_cfg["cps_occ_to_soc"]
    occsoc_path = Path(resolve_tokens(occsoc_cfg["path"], cfg))
    occsoc_cw = pl.read_csv(str(occsoc_path))

    # Normalize OCC and map to SOC via left join
    occ_src = occsoc_cfg.get("map", {}).get("source_column", "OCC")
    cps = cps.with_columns(pl.col(occ_src).cast(pl.Utf8).str.strip_chars().alias(occ_src))
    cps = map_with_crosswalk(
        cps,
        occsoc_cw,
        source_col=occ_src,
        cw_key=occsoc_cfg["map"]["crosswalk_key"],
        cw_value=occsoc_cfg["map"]["target_column"],
        out_col="OCCSOC",
    )

    # Ensure OCCSOC codes are exactly 7 characters
    cps = cps.with_columns(
        pl.when(pl.col("OCCSOC").is_not_null())
          .then(pl.col("OCCSOC").cast(pl.Utf8).str.slice(0, 7))
          .otherwise(pl.col("OCCSOC"))
          .alias("OCCSOC")
    )

    # Replace the character "X" in OCCSOC with "0"
    cps = cps.with_columns(
            pl.col("OCCSOC")
                .cast(pl.Utf8)
                .str.replace_all(r"(?i)X", "0")
                .alias("OCCSOC")
    )

    mapped_count = cps.filter(pl.col("OCCSOC").is_not_null()).height
    total_count = len(cps)
    count_zero = cps.filter(pl.col("OCCSOC") == "0").height
    logger.info(f"CPS: Successfully mapped {mapped_count:,}/{total_count:,} OCC codes to SOC ({mapped_count/total_count:.1%}) with {count_zero:,} codes mapped to '0' ({count_zero/total_count:.1%})")

    cps = add_soc_groupings(cps, "OCCSOC", prefix="OCCSOC")
    tw_path = Path(resolve_tokens(cw_cfg["teleworkability"]["path"], cfg))
    tw_map_cfg = cw_cfg["teleworkability"]["map"]
    cps = assign_teleworkability(cps, tw_path, "OCCSOC", tw_map_cfg, out_prefix="TELEWORKABLE_OCSSOC")
    logger.info(f"CPS: Added SOC groupings and teleworkability")

    
    # Ensure cell components exist before building cell_id
    cps = _ensure_cell_components(cps)

    # Add cell_id creation early for ATUS join compatibility
    cps = _derive_cell_components(cps)
    logger.info(f"CPS: Added cell_id components early for ATUS compatibility")


    # Telework variables: After 2022 use TELWRKPAY/TELWRKHR/UHRSWORKT logic
    has_post_cols = all(col in cps.columns for col in ["TELWRKHR", "TELWRKPAY", "UHRSWORKT"])
    if has_post_cols:
        # Cast to numeric types once
        cps = cps.with_columns([
            pl.col("TELWRKHR").cast(pl.Float64),
            pl.col("TELWRKPAY").cast(pl.Int64),
            pl.col("UHRSWORKT").cast(pl.Float64),
        ])
        cps = cps.with_columns(
            pl.when(pl.col("TELWRKPAY") == 0)  # NIU
             .then(pl.lit(None))
            .when(pl.col("TELWRKPAY") == 2)   # No telework
             .then(0.0)
            .when(pl.col("TELWRKPAY") == 1)   # Compute share
             .then(
                pl.when(
                    pl.col("UHRSWORKT").is_in([0.0, 997.0, 999.0]) | (pl.col("TELWRKHR") == 999.0)
                ).then(pl.lit(None))
                 .otherwise(pl.col("TELWRKHR") / pl.col("UHRSWORKT"))
             )
            .otherwise(pl.lit(None))
            .alias("ALPHA")
        )
    else:
        cps = cps.with_columns(pl.lit(None).cast(pl.Float64).alias("ALPHA"))

    # Final ALPHA clamp to [0,1]
    cps = cps.with_columns(
        pl.when(pl.col("ALPHA").is_not_null())
        .then(pl.col("ALPHA").clip(0.0, 1.0))
        .otherwise(pl.col("ALPHA"))
        .alias("ALPHA")
    )

    # Dummies and WFH from final ALPHA
    cps = cps.with_columns([
        (pl.col("ALPHA").is_not_null() & (pl.col("ALPHA") == 0)).cast(pl.Int64).alias("FULL_INPERSON"),
        (pl.col("ALPHA").is_not_null() & (pl.col("ALPHA") == 1)).cast(pl.Int64).alias("FULL_REMOTE"),
        (pl.col("ALPHA").is_not_null() & (pl.col("ALPHA") > 0) & (pl.col("ALPHA") < 1)).cast(pl.Int64).alias("HYBRID"),
        (pl.col("ALPHA").is_not_null() & (pl.col("ALPHA") > 0)).cast(pl.Int64).alias("WFH"),
    ])

    # Construct WAGE from HOURWAGE2/EARNWEEK2/UHRSWORKT - NO FILTERING, preserve nulls
    has_wage_cols = all(c in cps.columns for c in ["HOURWAGE2", "EARNWEEK2", "UHRSWORKT"])
    if has_wage_cols:
        cps = cps.with_columns([
            pl.col("HOURWAGE2").cast(pl.Float64, strict=False).alias("HOURWAGE2"),
            pl.col("EARNWEEK2").cast(pl.Float64, strict=False).alias("EARNWEEK2"),
            pl.col("UHRSWORKT").cast(pl.Float64, strict=False).alias("UHRSWORKT"),
        ])

        # Build WAGE - allow nulls, no filtering
        cps = cps.with_columns([
            pl.when((pl.col("HOURWAGE2").is_not_null()) & (pl.col("HOURWAGE2") != 999.99))
            .then(pl.col("HOURWAGE2"))
            .when(
                (pl.col("EARNWEEK2").is_not_null()) & (~pl.col("EARNWEEK2").is_in([9999.99, 999999.99])) &
                (pl.col("UHRSWORKT").is_not_null()) & (~pl.col("UHRSWORKT").is_in([999, 997]))
            )
            .then(pl.col("EARNWEEK2") / pl.col("UHRSWORKT"))
            .otherwise(None)
            .alias("WAGE")
        ])
        logger.info("CPS: Created WAGE from HOURWAGE2 or EARNWEEK2/UHRSWORKT (preserving nulls)")
    else:
        logger.info("CPS: Skipping wage construction; required columns not present.")

    # Save full processed dataset with all original records
    out_path = cfg.get_output_path("cps", (cfg.processed_dir / "cps" / "cps_processed.csv").resolve())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cps = cps.drop(["__rid"], strict=False)  # Clean any temp columns
    cps.write_csv(str(out_path))
    logger.info(f"CPS: Full dataset with {len(cps):,} rows saved to {out_path}")

    # Controls filter function
    def apply_controls_filter(df: pl.DataFrame) -> pl.DataFrame:
        df2 = df
        if "CLASSWKR" in df2.columns:
            df2 = df2.with_columns(pl.col("CLASSWKR").cast(pl.Utf8, strict=False).str.strip_chars().alias("CLASSWKR"))
            df2 = df2.filter(pl.col("CLASSWKR").is_not_null() & (~pl.col("CLASSWKR").is_in(["00", "99"])))
        if "AGE" in df2.columns:
            df2 = df2.with_columns(pl.col("AGE").cast(pl.Int32, strict=False).alias("AGE"))
            df2 = df2.filter(pl.col("AGE").is_not_null() & (pl.col("AGE") != 999))
        if "SEX" in df2.columns:
            df2 = df2.with_columns(pl.col("SEX").cast(pl.Int32, strict=False).alias("SEX"))
            df2 = df2.filter(pl.col("SEX").is_not_null() & (pl.col("SEX") != 9))
        if "RACE" in df2.columns:
            df2 = df2.with_columns(pl.col("RACE").cast(pl.Int32, strict=False).alias("RACE"))
            df2 = df2.filter(pl.col("RACE").is_not_null() & (pl.col("RACE") != 999))
        if "HISPAN" in df2.columns:
            df2 = df2.with_columns(pl.col("HISPAN").cast(pl.Int32, strict=False).alias("HISPAN"))
            df2 = df2.filter(pl.col("HISPAN").is_not_null() & (~pl.col("HISPAN").is_in([901, 999])))
        return df2

    # ALPHA + controls subset
    alpha_present = cps.filter(pl.col("ALPHA").is_not_null())
    before_ctrl = len(alpha_present)
    alpha_present = apply_controls_filter(alpha_present)
    logger.info(f"CPS: ALPHA-present subset: {before_ctrl:,} -> {len(alpha_present):,} after controls filter")

    # Reweighting for ALPHA subset
    if "WTFINL" in alpha_present.columns:
        target_univ, resp_sub_adj = reweight_poststrat(alpha_present, weight_col="WTFINL", response_col="WAGE", min_wage=0)
        key_cols = [c for c in ["YEAR", "MONTH", "CPSIDP"] if c in alpha_present.columns]
        if key_cols:
            alpha_present = alpha_present.join(
                resp_sub_adj.select(key_cols + ["WTFINL_ADJ"]),
                on=key_cols,
                how="left"
            )
        else:
            alpha_present = alpha_present.with_row_index(name="__rid")
            resp_sub_adj = resp_sub_adj.with_row_index(name="__rid")
            alpha_present = alpha_present.join(resp_sub_adj.select(["__rid", "WTFINL_ADJ"]), on="__rid", how="left").drop("__rid")

        alpha_present = alpha_present.with_columns(pl.col("WTFINL").alias("WTFINL_OLD"))
        alpha_present = alpha_present.with_columns(
            pl.when(pl.col("WTFINL_ADJ").is_not_null()).then(pl.col("WTFINL_ADJ")).otherwise(pl.col("WTFINL")).alias("WTFINL")
        ).drop(["WTFINL_ADJ"], strict=False)

    alpha_present = add_topcode_and_real_wages(alpha_present)
    alpha_present_path = (cfg.processed_dir / "cps" / "cps_alpha_present_reweighted.csv").resolve()
    alpha_present_path.parent.mkdir(parents=True, exist_ok=True)
    alpha_present.write_csv(str(alpha_present_path))
    logger.info(f"CPS: ALPHA-present + controls subset saved to {alpha_present_path}")

    # ALPHA + WAGE + controls subset
    alpha_wage_present = alpha_present.filter(pl.col("WAGE").is_not_null())
    alpha_wage_present = add_topcode_and_real_wages(alpha_wage_present)
    alpha_wage_present_path = (cfg.processed_dir / "cps" / "cps_alpha_wage_present_reweighted.csv").resolve()
    alpha_wage_present.write_csv(str(alpha_wage_present_path))
    logger.info(f"CPS: ALPHA + WAGE + controls subset ({len(alpha_wage_present):,} rows) saved to {alpha_wage_present_path}")

    return out_path

def _ensure_cell_components(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ensures the presence of key derived columns in a Polars DataFrame based on existing raw columns.
    This function checks for the existence of several harmonized or derived columns commonly used in labor force and demographic analysis.
    If a derived column is missing but its source column is present, the function computes and adds the derived column to the DataFrame.
    Derived columns and their logic:
    - 'occ2_harmonized': Extracts the 2-digit SOC major group from 'OCCSOC'.
    - 'ind_broad': Maps the 2-digit NAICS code from 'INDNAICS' to a broad industry sector.
    - 'ftpt': Categorizes 'UHRSWORKT' as 'fulltime' (>=35 hours) or 'parttime'.
    - 'edu3': Bins 'EDUC' into 'lt_ba' (less than bachelor's), 'ba' (bachelor's), or 'adv' (advanced).
    - 'age4': Bins 'AGE' into four groups: '25-34', '35-44', '45-54', '55-64'.
    - 'sex': Maps 'SEX' (1/2) to 'male'/'female'.
    - 'state': Pads 'STATEFIP' to two digits as a string.
    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing raw columns.
    Returns
    -------
    pl.DataFrame
        DataFrame with the required derived columns added if they were missing.
    """
    
    # occ2_harmonized from OCCSOC (2-digit SOC major group)
    if "occ2_harmonized" not in df.columns and "OCCSOC" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("OCCSOC").is_not_null())
                .then(pl.col("OCCSOC").cast(pl.Utf8).str.slice(0, 2))
                .otherwise(pl.lit(None))
                .alias("occ2_harmonized")
        )

    # ind_broad from INDNAICS (2-digit NAICS -> sector)
    if "ind_broad" not in df.columns and "INDNAICS" in df.columns:
        ind2 = pl.col("INDNAICS").cast(pl.Utf8).str.strip_chars().str.slice(0, 2)
        df = df.with_columns(
            pl.when(ind2 == "11").then(pl.lit("agriculture_forestry_fishing"))
                .when(ind2 == "21").then(pl.lit("mining_oil_gas"))
                .when(ind2 == "22").then(pl.lit("utilities"))
                .when(ind2 == "23").then(pl.lit("construction"))
                .when(ind2.is_in(["31", "32", "33"])).then(pl.lit("manufacturing"))
                .when(ind2 == "42").then(pl.lit("wholesale_trade"))
                .when(ind2.is_in(["44", "45"])).then(pl.lit("retail_trade"))
                .when(ind2.is_in(["48", "49"])).then(pl.lit("transportation_warehousing"))
                .when(ind2 == "51").then(pl.lit("information"))
                .when(ind2 == "52").then(pl.lit("finance_insurance"))
                .when(ind2 == "53").then(pl.lit("real_estate_rental"))
                .when(ind2 == "54").then(pl.lit("professional_scientific_technical"))
                .when(ind2 == "55").then(pl.lit("management_companies"))
                .when(ind2 == "56").then(pl.lit("administrative_support_waste"))
                .when(ind2 == "61").then(pl.lit("educational_services"))
                .when(ind2 == "62").then(pl.lit("health_care_social_assistance"))
                .when(ind2 == "71").then(pl.lit("arts_entertainment_recreation"))
                .when(ind2 == "72").then(pl.lit("accommodation_food_services"))
                .when(ind2 == "81").then(pl.lit("other_services"))
                .when(ind2 == "92").then(pl.lit("public_administration"))
                .otherwise(pl.lit(None))
                .alias("ind_broad")
        )

    # ftpt from UHRSWORKT (>=35 fulltime)
    if "ftpt" not in df.columns and "UHRSWORKT" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("UHRSWORKT").cast(pl.Float64) >= 35)
                .then(pl.lit("fulltime"))
                .otherwise(pl.lit("parttime"))
                .alias("ftpt")
        )

    # edu3 from EDUC (coarse bins; adjust if you have a detailed codebook)
    if "edu3" not in df.columns and "EDUC" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("EDUC") < 111).then(pl.lit("lt_ba"))     # HS or some college
                .when(pl.col("EDUC") < 123).then(pl.lit("ba"))         # Bachelor's
                .otherwise(pl.lit("adv"))                              # Advanced
                .alias("edu3")
        )

    # age4 from AGE (25–34, 35–44, 45–54, 55–64)
    if "age4" not in df.columns and "AGE" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("AGE") < 25).then(pl.lit(None))
                .when(pl.col("AGE") <= 34).then(pl.lit("25-34"))
                .when(pl.col("AGE") <= 44).then(pl.lit("35-44"))
                .when(pl.col("AGE") <= 54).then(pl.lit("45-54"))
                .when(pl.col("AGE") <= 64).then(pl.lit("55-64"))
                .otherwise(pl.lit(None))
                .alias("age4")
        )

    # sex from SEX (1 male, 2 female)
    if "sex" not in df.columns and "SEX" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("SEX") == 1).then(pl.lit("male"))
                .when(pl.col("SEX") == 2).then(pl.lit("female"))
                .otherwise(pl.lit(None))
                .alias("sex")
        )

    # state from STATEFIP
    if "state" not in df.columns and "STATEFIP" in df.columns:
        df = df.with_columns(
            pl.col("STATEFIP").cast(pl.Utf8).str.pad_start(2, "0").alias("state")
        )
    return df



def _derive_cell_components(df: pl.DataFrame) -> pl.DataFrame:
    """Derive cell_id, psi, and high_psi_flag from demographic variables"""
    
    # Create cell_id from key demographic variables
    # Format: "occ2|ind_broad|ftpt|edu3|age4|sex|state"
    # cell_vars = ["occ2_harmonized", "ind_broad", "ftpt", "edu3", "age4", "sex", "state"]
    # cell_vars = ["occ2_harmonized", "ind_broad", "ftpt", "edu3", "age4", "sex"]
    # cell_vars = ["occ2_harmonized", "ftpt", "edu3", "age4", "sex"]
    cell_vars = ["occ2_harmonized", "ftpt", "edu3", "sex"]
    
    # Ensure all required variables exist, convert to string
    df_with_strings = df
    for var in cell_vars:
        if var in df.columns:
            df_with_strings = df_with_strings.with_columns(
                pl.col(var).cast(pl.Utf8).alias(var)
            )
        else:
            # Add placeholder if missing
            df_with_strings = df_with_strings.with_columns(
                pl.lit("unknown").alias(var)
            )
    
    # Create cell_id by concatenating with "|"
    df_with_strings = df_with_strings.with_columns(
        pl.concat_str([pl.col(var) for var in cell_vars], separator="|").alias("cell_id")
    )
    
    # Create psi and high_psi_flag (placeholder logic - adjust as needed)
    if "TELEWORKABLE_OCSSOC_3D" in df.columns:
        df_with_strings = df_with_strings.with_columns([
            pl.col("TELEWORKABLE_OCSSOC_3D").cast(pl.Float64).alias("psi"),
            (pl.col("TELEWORKABLE_OCSSOC_3D").cast(pl.Float64) > 0.5).cast(pl.Int8).alias("high_psi_flag")
        ])
    else:
        df_with_strings = df_with_strings.with_columns([
            pl.lit(0.5).alias("psi"),
            pl.lit(0).cast(pl.Int8).alias("high_psi_flag")
        ])
    
    return df_with_strings


def _winsorize_by_year(df: pl.DataFrame, col: str, lower_pct: float, upper_pct: float) -> pl.DataFrame:
    """Winsorize a column by year at specified percentiles"""
    
    # Calculate percentiles by year
    percentiles = (
        df.filter(pl.col(col).is_not_null())
        .group_by("YEAR")
        .agg([
            pl.col(col).quantile(lower_pct).alias("lower_bound"),
            pl.col(col).quantile(upper_pct).alias("upper_bound")
        ])
    )
    
    # Join back to main dataframe and winsorize
    df_winsorized = (
        df.join(percentiles, on="YEAR", how="left")
        .with_columns([
            pl.when(pl.col(col).is_not_null())
            .then(
                pl.col(col)
                .clip(pl.col("lower_bound"), pl.col("upper_bound"))
            )
            .otherwise(pl.col(col))
            .alias(col)
        ])
        .drop(["lower_bound", "upper_bound"])
    )
    
    return df_winsorized


def _export_stata(df: pl.DataFrame, file_path: Path) -> None:
    """Export a Polars DataFrame to Stata .dta format via pandas"""
    
    # Convert to pandas for Stata export
    df_pandas = df.to_pandas()
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to Stata format
    df_pandas.to_stata(str(file_path), write_index=False, version=119)
    
    logger.info(f"Exported {len(df):,} rows to {file_path}")


def diagnose_bridge(bridge: pl.DataFrame, cfg: Config, logger=None) -> None:
    """Comprehensive diagnostics for bridge lambda results."""
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        # 1) Non-finite λ counts
        bad = bridge.select([
            (~pl.col("lambda_remote").is_finite()).sum().alias("bad_remote"),
            (~pl.col("lambda_hybrid").is_finite()).sum().alias("bad_hybrid"),
            (~pl.col("lambda_inperson").is_finite()).sum().alias("bad_inperson"),
        ]).to_dicts()[0]
        logger.info(f"Bridge: λ non-finite counts — remote={bad['bad_remote']}, hybrid={bad['bad_hybrid']}, inperson={bad['bad_inperson']} (target=0)")

        # 2) Overall medians and central spread
        finite = bridge.filter(
            (pl.col("cps_w") > 0)
            & pl.all_horizontal([
                pl.col("lambda_remote").is_finite(),
                pl.col("lambda_hybrid").is_finite(),
                pl.col("lambda_inperson").is_finite(),
                pl.col("worker_share_remote").is_finite(),
                pl.col("worker_share_hybrid").is_finite(),
                pl.col("worker_share_inperson").is_finite(),
                pl.col("worker_share_remote_from_atus").is_finite(),
                pl.col("worker_share_hybrid_from_atus").is_finite(),
                pl.col("worker_share_inperson_from_atus").is_finite(),
            ])
        )

        summ = finite.select([
            pl.col("lambda_remote").quantile(0.1).alias("r_p10"),
            pl.col("lambda_remote").median().alias("r_p50"),
            pl.col("lambda_remote").quantile(0.9).alias("r_p90"),
            pl.col("lambda_hybrid").quantile(0.1).alias("h_p10"),
            pl.col("lambda_hybrid").median().alias("h_p50"),
            pl.col("lambda_hybrid").quantile(0.9).alias("h_p90"),
            pl.col("lambda_inperson").quantile(0.1).alias("i_p10"),
            pl.col("lambda_inperson").median().alias("i_p50"),
            pl.col("lambda_inperson").quantile(0.9).alias("i_p90"),
        ]).to_dicts()[0]
        logger.info(
            "Bridge: λ percentiles — "
            f"\n\t\t\t\tremote [p10={summ['r_p10']:.3f}, p50={summ['r_p50']:.3f}, p90={summ['r_p90']:.3f}] | "
            f"\n\t\t\t\thybrid [p10={summ['h_p10']:.3f}, p50={summ['h_p50']:.3f}, p90={summ['h_p90']:.3f}] | "
            f"\n\t\t\t\tinperson [p10={summ['i_p10']:.3f}, p50={summ['i_p50']:.3f}, p90={summ['i_p90']:.3f}] "
            "\n\t\t\t\t\t(expect medians ≈ 1; in-person usually closest to 1)"
        )

        # 3) Medians by occupation (spot centering/drift)
        by_occ = (
            finite.group_by("occ2_harmonized")
                .agg([
                    pl.col("lambda_remote").median().alias("med_r"),
                    pl.col("lambda_hybrid").median().alias("med_h"),
                    pl.col("lambda_inperson").median().alias("med_i"),
                ])
                .sort("occ2_harmonized")
        )
        med_occ = by_occ.select([
            pl.col("med_r").median().alias("occ_med_r"),
            pl.col("med_h").median().alias("occ_med_h"),
            pl.col("med_i").median().alias("occ_med_i"),
        ]).to_dicts()[0]
        logger.info(f"Bridge: median of occ-level medians — remote={med_occ['occ_med_r']:.3f}, hybrid={med_occ['occ_med_h']:.3f}, inperson={med_occ['occ_med_i']:.3f}")

        # 4) How well ATUS-implied worker shares match CPS (weighted)
        def share_metrics(k: str) -> dict:
            a = f"worker_share_{k}"
            p = f"worker_share_{k}_from_atus"
            w = "cps_w"
            # avoid division by zero in MAPE
            eps = 1e-8
            m = finite.select([
                (pl.col(w)).sum().alias("W"),
                (pl.col(w) * (pl.col(p) - pl.col(a)).abs()).sum().alias("WMAE_num"),
                (pl.col(w) * ((pl.col(p) - pl.col(a))**2)).sum().alias("WRMSE_num"),
                (pl.col(w) * (pl.col(p) / (pl.col(a) + eps) - 1).abs()).sum().alias("WMAPE_num"),
                # unweighted Pearson correlation (quick signal)
                pl.corr(pl.col(p), pl.col(a)).alias("corr"),
            ]).to_dicts()[0]
            W = m["W"] if m["W"] and m["W"] > 0 else 1.0
            return {
                "WMAE": m["WMAE_num"] / W,
                "WRMSE": (m["WRMSE_num"] / W) ** 0.5,
                "WMAPE": m["WMAPE_num"] / W,
                "corr": m["corr"],
            }

        for k in ["remote", "hybrid", "inperson"]:
            mk = share_metrics(k)
            logger.info(
                f"Bridge fit ({k}): WMAE={mk['WMAE']:.4f}, WMAPE={mk['WMAPE']:.3f}, WRMSE={mk['WRMSE']:.4f}, corr={mk['corr']:.3f}"
            )

        # 5) Worst cells (so you can inspect outliers)
        worst = (
            finite
            .with_columns([
                (pl.col("worker_share_remote_from_atus") - pl.col("worker_share_remote")).abs().alias("ae_remote"),
                (pl.col("worker_share_hybrid_from_atus") - pl.col("worker_share_hybrid")).abs().alias("ae_hybrid"),
                (pl.col("worker_share_inperson_from_atus") - pl.col("worker_share_inperson")).abs().alias("ae_inperson"),
            ])
            .with_columns([
                (pl.col("ae_remote") + pl.col("ae_hybrid") + pl.col("ae_inperson")).alias("ae_sum"),
            ])
            .sort(["ae_sum"], descending=True)
            .head(10)
        )
        logger.info("Bridge: top 10 cell×years by total absolute share error (remote+hybrid+inperson):")
        for r in worst.select(["YEAR", "cell_id", "occ2_harmonized", "ae_sum"]).to_dicts():
            logger.info(f"  YEAR={r['YEAR']}, cell_id={r['cell_id']}, occ2={r['occ2_harmonized']}, abs_err_sum={r['ae_sum']:.4f}")

        # 6) (Optional) save a small CSV of diagnostics
        # worst.write_csv((cfg.processed_dir / "empirical" / "bridge_outliers_top10.csv"))

    except Exception as e:
        logger.warning(f"Bridge diagnostics failed: {e}")

def build_bridge_lambdas(
    cps_mi: pl.DataFrame,
    atus_cell: pl.DataFrame,
    eps: float = 1e-6,
    clip: Tuple[float, float] = (0.5, 2.0),
    pool_years: List[int] = [2022, 2023, 2024, 2025],
    min_eff: int = 15,
) -> pl.DataFrame:
    """
    Bridge day- to worker-shares with λ at key level (occ2 × edu3 × sex), 2022+.

    Ladder for day shares (per cell×year):
        yearly (if n_eff_diaries ≥ min_eff) → pooled-by-cell over pool_years →
        pooled-by-occ2 over pool_years → global pooled fallback.
    """

    # ---------- Meta ----------
    occ_meta = cps_mi.select(["YEAR", "cell_id", "occ2_harmonized"]).unique()
    key_map  = cps_mi.select(["cell_id", "occ2_harmonized", "edu3", "sex"]).unique()

    # ---------- CPS worker shares at cell×year (observed; 2022+) ----------
    cps_obs = cps_mi.filter(pl.col("YEAR") >= 2022)
    worker = (
        cps_obs
        .group_by(["YEAR", "cell_id"])
        .agg([
            ((pl.col("FULL_REMOTE")   * pl.col("cps_weight")).sum() / pl.col("cps_weight").sum()).alias("worker_share_remote"),
            ((pl.col("HYBRID")        * pl.col("cps_weight")).sum() / pl.col("cps_weight").sum()).alias("worker_share_hybrid"),
            ((pl.col("FULL_INPERSON") * pl.col("cps_weight")).sum() / pl.col("cps_weight").sum()).alias("worker_share_inperson"),
            pl.col("cps_weight").sum().alias("cps_w"),
        ])
        .with_columns((pl.col("worker_share_remote")+pl.col("worker_share_hybrid")+pl.col("worker_share_inperson")).alias("_sw"))
        .with_columns([
            (pl.col("worker_share_remote")   / pl.col("_sw")).alias("worker_share_remote"),
            (pl.col("worker_share_hybrid")   / pl.col("_sw")).alias("worker_share_hybrid"),
            (pl.col("worker_share_inperson") / pl.col("_sw")).alias("worker_share_inperson"),
        ])
        .drop("_sw")
    )

    # ---------- ATUS day shares & fallbacks ----------
    # yearly (2022+)
    day_y = (
        atus_cell
        .filter(pl.col("YEAR") >= 2022)
        .select(["YEAR", "cell_id", "share_remote_day", "share_hybrid_day", "share_inperson_day"])
    )

    # pooled by cell across pool_years
    day_pool_cell = (
        atus_cell
        .filter(pl.col("YEAR").is_in(pool_years))
        .group_by("cell_id")
        .agg([
            pl.col("share_remote_day").mean().alias("share_remote_day_pool_cell"),
            pl.col("share_hybrid_day").mean().alias("share_hybrid_day_pool_cell"),
            pl.col("share_inperson_day").mean().alias("share_inperson_day_pool_cell"),
        ])
    )

    # pooled by occ2 across pool_years
    day_occ2 = (
        atus_cell
        .filter(pl.col("YEAR").is_in(pool_years))
        .join(occ_meta.select(["YEAR","cell_id","occ2_harmonized"]).unique(), on=["YEAR","cell_id"], how="left")
        .group_by("occ2_harmonized")
        .agg([
            pl.col("share_remote_day").mean().alias("share_remote_day_pool_occ2"),
            pl.col("share_hybrid_day").mean().alias("share_hybrid_day_pool_occ2"),
            pl.col("share_inperson_day").mean().alias("share_inperson_day_pool_occ2"),
        ])
    )

    # global pooled (last resort)
    gvals = (
        atus_cell
        .filter(pl.col("YEAR").is_in(pool_years))
        .select([
            pl.col("share_remote_day").mean().alias("g_remote"),
            pl.col("share_hybrid_day").mean().alias("g_hybrid"),
            pl.col("share_inperson_day").mean().alias("g_inperson"),
        ])
        .to_dicts()[0]
    )

    # effective diaries for gating yearly vs pooled
    neff = atus_cell.select(["YEAR","cell_id","n_eff_diaries"]).unique()

    # combine ladder at cell×year
    df = (
        worker
        .join(day_y, on=["YEAR","cell_id"], how="left")
        .join(day_pool_cell, on="cell_id", how="left")
        .join(occ_meta.select(["YEAR","cell_id","occ2_harmonized"]).unique(), on=["YEAR","cell_id"], how="left")
        .join(day_occ2, on="occ2_harmonized", how="left")
        .join(neff, on=["YEAR","cell_id"], how="left")
        .with_columns([
            pl.when(pl.col("n_eff_diaries") >= min_eff).then(pl.col("share_remote_day")).otherwise(pl.col("share_remote_day_pool_cell")).alias("share_remote_day_eff"),
            pl.when(pl.col("n_eff_diaries") >= min_eff).then(pl.col("share_hybrid_day")).otherwise(pl.col("share_hybrid_day_pool_cell")).alias("share_hybrid_day_eff"),
            pl.when(pl.col("n_eff_diaries") >= min_eff).then(pl.col("share_inperson_day")).otherwise(pl.col("share_inperson_day_pool_cell")).alias("share_inperson_day_eff"),
        ])
        .with_columns([
            pl.coalesce([pl.col("share_remote_day_eff"),   pl.col("share_remote_day_pool_occ2"),   pl.lit(gvals["g_remote"])]).alias("share_remote_day"),
            pl.coalesce([pl.col("share_hybrid_day_eff"),   pl.col("share_hybrid_day_pool_occ2"),   pl.lit(gvals["g_hybrid"])]).alias("share_hybrid_day"),
            pl.coalesce([pl.col("share_inperson_day_eff"), pl.col("share_inperson_day_pool_occ2"), pl.lit(gvals["g_inperson"])]).alias("share_inperson_day"),
        ])
        .drop([
            "share_remote_day_eff","share_hybrid_day_eff","share_inperson_day_eff",
            "share_remote_day_pool_cell","share_hybrid_day_pool_cell","share_inperson_day_pool_cell",
            "share_remote_day_pool_occ2","share_hybrid_day_pool_occ2","share_inperson_day_pool_occ2",
        ])
        .with_columns((pl.col("share_remote_day")+pl.col("share_hybrid_day")+pl.col("share_inperson_day")).alias("_ss"))
        .with_columns([
            (pl.col("share_remote_day")  / pl.col("_ss")).alias("share_remote_day"),
            (pl.col("share_hybrid_day")  / pl.col("_ss")).alias("share_hybrid_day"),
            (pl.col("share_inperson_day")/ pl.col("_ss")).alias("share_inperson_day"),
        ])
        .drop("_ss")
    )

    # ---------- Key-level aggregation (OCC2×EDU3×SEX) ----------
    # ensure W_worked exists (fallback to 1.0 if not carried over)
    if "W_worked" not in atus_cell.columns:
        atus_cell = atus_cell.with_columns(pl.lit(1.0).alias("W_worked"))

    # key×year CPS worker shares
    wk_key = (
        cps_mi.filter(pl.col("YEAR") >= 2022)
              .join(key_map, on="cell_id", how="left")
              .group_by(["YEAR","occ2_harmonized","edu3","sex"])
              .agg([
                  ((pl.col("FULL_REMOTE")   * pl.col("cps_weight")).sum() / pl.col("cps_weight").sum()).alias("worker_share_remote"),
                  ((pl.col("HYBRID")        * pl.col("cps_weight")).sum() / pl.col("cps_weight").sum()).alias("worker_share_hybrid"),
                  ((pl.col("FULL_INPERSON") * pl.col("cps_weight")).sum() / pl.col("cps_weight").sum()).alias("worker_share_inperson"),
                  pl.col("cps_weight").sum().alias("cps_w_key"),
              ])
              .with_columns((pl.col("worker_share_remote")+pl.col("worker_share_hybrid")+pl.col("worker_share_inperson")).alias("_sw"))
              .with_columns([
                  (pl.col("worker_share_remote")  / pl.col("_sw")).alias("worker_share_remote"),
                  (pl.col("worker_share_hybrid")  / pl.col("_sw")).alias("worker_share_hybrid"),
                  (pl.col("worker_share_inperson")/ pl.col("_sw")).alias("worker_share_inperson"),
              ])
              .drop("_sw")
    )

    # key×year ATUS day shares (weighted by worked mass across cells)
    day_key_y = (
        atus_cell
        .join(key_map, on="cell_id", how="left")
        .filter(pl.col("YEAR") >= 2022)
        .group_by(["YEAR","occ2_harmonized","edu3","sex"])
        .agg([
            (pl.col("share_remote_day")   * pl.col("W_worked")).sum() / pl.col("W_worked").sum().alias("share_remote_day"),
            (pl.col("share_hybrid_day")   * pl.col("W_worked")).sum() / pl.col("W_worked").sum().alias("share_hybrid_day"),
            (pl.col("share_inperson_day") * pl.col("W_worked")).sum() / pl.col("W_worked").sum().alias("share_inperson_day"),
            pl.col("W_worked").sum().alias("W_worked_key"),
        ])
    )

    # pooled by key across pool_years — use *_pool_key names
    day_key_pool = (
        atus_cell
        .join(key_map, on="cell_id", how="left")
        .filter(pl.col("YEAR").is_in(pool_years))
        .group_by(["occ2_harmonized","edu3","sex"])
        .agg([
            (pl.col("share_remote_day")   * pl.col("W_worked")).sum() / pl.col("W_worked").sum()
                .alias("share_remote_day"),
            (pl.col("share_hybrid_day")   * pl.col("W_worked")).sum() / pl.col("W_worked").sum()
                .alias("share_hybrid_day"),
            (pl.col("share_inperson_day") * pl.col("W_worked")).sum() / pl.col("W_worked").sum()
                .alias("share_inperson_day"),
        ])
    )

    # join and coalesce yearly with pooled-key
    key_df = (
        wk_key
        .join(day_key_y,   on=["YEAR","occ2_harmonized","edu3","sex"], how="left")
        .join(
            day_key_pool.select([
                "occ2_harmonized","edu3","sex",
                "share_remote_day","share_hybrid_day","share_inperson_day",
            ]),
            on=["occ2_harmonized","edu3","sex"], how="left"
        )
        .with_columns([
            pl.coalesce([pl.col("share_remote_day"),   pl.col("share_remote_day_right")]).alias("share_remote_day"),
            pl.coalesce([pl.col("share_hybrid_day"),   pl.col("share_hybrid_day_right")]).alias("share_hybrid_day"),
            pl.coalesce([pl.col("share_inperson_day"), pl.col("share_inperson_day_right")]).alias("share_inperson_day"),
        ])
        .drop(["share_remote_day_right","share_hybrid_day_right","share_inperson_day_right"])
    )

    # λ at key×year; shrink across years within key; fill; clip
    for k in ["remote","hybrid","inperson"]:
        wk, dk, lk = f"worker_share_{k}", f"share_{k}_day", f"lambda_{k}"
        key_df = key_df.with_columns(
            pl.when(pl.col(dk).is_null() | (pl.col(dk) < eps))
              .then(pl.when(pl.col(wk).is_null() | (pl.col(wk) < eps)).then(pl.lit(1.0)).otherwise(pl.lit(None)))
              .otherwise(pl.col(wk)/pl.col(dk))
              .alias(lk)
        )
    key_df = key_df.with_columns([
        pl.col("lambda_remote").cast(pl.Float64),
        pl.col("lambda_hybrid").cast(pl.Float64),
        pl.col("lambda_inperson").cast(pl.Float64),
    ])

    key_meds = (
        key_df.group_by(["occ2_harmonized","edu3","sex"])
              .agg([
                  pl.col("lambda_remote").median().alias("med_lambda_remote_key"),
                  pl.col("lambda_hybrid").median().alias("med_lambda_hybrid_key"),
                  pl.col("lambda_inperson").median().alias("med_lambda_inperson_key"),
              ])
    )
    key_df = key_df.join(key_meds, on=["occ2_harmonized","edu3","sex"], how="left")
    for k in ["remote","hybrid","inperson"]:
        key_df = key_df.with_columns(
            pl.coalesce([pl.col(f"lambda_{k}"), pl.col(f"med_lambda_{k}_key"), pl.lit(1.0)]).alias(f"lambda_{k}")
        ).drop(f"med_lambda_{k}_key")

    lo, hi = clip
    for k in ["remote","hybrid","inperson"]:
        key_df = key_df.with_columns(pl.col(f"lambda_{k}").clip(lo, hi).alias(f"lambda_{k}"))

    # ---------- Apply key λ to each cell×year ----------
    df = df.drop(["lambda_remote","lambda_hybrid","lambda_inperson"], strict=False)
    df = (
        df.join(key_map, on="cell_id", how="left")
          .join(
              key_df.select(["YEAR","occ2_harmonized","edu3","sex",
                             "lambda_remote","lambda_hybrid","lambda_inperson"]),
              on=["YEAR","occ2_harmonized","edu3","sex"], how="left"
          )
    )
    for k in ["remote","hybrid","inperson"]:
        df = df.with_columns(pl.col(f"lambda_{k}").clip(lo, hi).fill_null(1.0).alias(f"lambda_{k}"))

    df = df.with_columns([
        (pl.col("lambda_remote")   * pl.col("share_remote_day")   +
         pl.col("lambda_hybrid")   * pl.col("share_hybrid_day")   +
         pl.col("lambda_inperson") * pl.col("share_inperson_day")).alias("_den")
    ]).with_columns([
        (pl.col("lambda_remote")   * pl.col("share_remote_day")   / pl.col("_den")).alias("worker_share_remote_from_atus"),
        (pl.col("lambda_hybrid")   * pl.col("share_hybrid_day")   / pl.col("_den")).alias("worker_share_hybrid_from_atus"),
        (pl.col("lambda_inperson") * pl.col("share_inperson_day") / pl.col("_den")).alias("worker_share_inperson_from_atus"),
    ]).drop("_den")

    # ---------- Output ----------
    bridge = df.select([
        "YEAR","cell_id","occ2_harmonized","cps_w",
        "worker_share_remote","worker_share_hybrid","worker_share_inperson",
        "share_remote_day","share_hybrid_day","share_inperson_day",
        "lambda_remote","lambda_hybrid","lambda_inperson",
        "worker_share_remote_from_atus","worker_share_hybrid_from_atus","worker_share_inperson_from_atus",
    ])

    
    return bridge

def preestimation_procedure(cfg: Config,
                            random_seed: int = 123,
                            bootstrap_reps: int = 500,
                            write_schema: bool = True,
                            produce_bridge: bool = True) -> dict[str, Path]:
    """Build Stata-ready artifacts with moved age/hours filters and improved logging."""
    np.random.seed(random_seed)

    # 1) Load Inputs
    cps_path = cfg.get_output_path("cps", (cfg.processed_dir / "cps" / "cps_processed.csv").resolve())
    atus_link_path = cfg.get_output_path("atus_link", (cfg.processed_dir / "atus_link.csv").resolve())
    if not cps_path.exists():
        raise FileNotFoundError(f"Missing CPS processed: {cps_path}")
    if not atus_link_path.exists():
        raise FileNotFoundError(f"Missing ATUS link: {atus_link_path}")

    cps = pl.read_csv(str(cps_path), schema_overrides={"INDNAICS": pl.String})
    atus_link = pl.read_csv(str(atus_link_path))

    key = ["CPSIDP"]
    # Filer cps to have only one obervation per key
    cps_for_merge = cps.filter(~pl.col("cell_id").is_null()).unique(subset = key, keep = 'first')
    atus_join = atus_link.join(cps_for_merge[key + ["cell_id"]], on=key, how="left") 
    atus_join = atus_join.filter(pl.col("cell_id").is_not_null())
    n_atus_initial = len(atus_link)
    n_atus_matched = len(atus_join)
    logger.info(f"ATUS-CPS Merge: Starting with {n_atus_initial:,} ATUS person entries.")
    coverage = (n_atus_matched / n_atus_initial) if n_atus_initial > 0 else 0
    logger.info(f"ATUS-CPS Merge: {n_atus_matched:,} entries successfully matched to a CPS cell_id.")
    logger.info(f"ATUS-CPS Merge: Coverage rate = {coverage:.2%}")
    # Attach cell ids via CPS join and then log coverage

    atus_join = atus_join.with_columns(
        pl.when(pl.col("work_minutes") > 0)
        .then(pl.col("remote_minutes") / pl.col("work_minutes"))
        .otherwise(None)
        .alias("alpha_atus")
    )

    # 2) CPS Processing
    # Apply age/hours filters first to the main CPS dataframe
    logger.info(f"CPS: Before age/hours filters: {len(cps):,} rows")
    for c in ["AGE", "UHRSWORKT"]:
        if c not in cps.columns:
            raise RuntimeError(f"CPS missing required column {c}")
    
    cps = cps.filter((pl.col("AGE") >= 25) & (pl.col("AGE") <= 64))
    cps = cps.filter((pl.col("UHRSWORKT") >= 20) & (pl.col("UHRSWORKT") <= 84))
    logger.info(f"CPS: After age/hours filters: {len(cps):,} rows")

    # Weights: recompute wage nonresponse post-stratification
    strat_cols = [c for c in ["YEAR", "SEX", "EDUC", "RACE"] if c in cps.columns]
    if "AGE" in cps.columns:
        cps = cps.with_columns(
            pl.when(pl.col("AGE") < 18).then(pl.lit(None))
                .when(pl.col("AGE") <= 24).then(pl.lit('18-24'))
                .when(pl.col("AGE") <= 34).then(pl.lit('25-34'))
                .when(pl.col("AGE") <= 44).then(pl.lit('35-44'))
                .when(pl.col("AGE") <= 54).then(pl.lit('45-54'))
                .when(pl.col("AGE") <= 64).then(pl.lit('55-64'))
                .otherwise(pl.lit('65+')).alias('AGE_GROUP')
        )
        strat_cols.append("AGE_GROUP")

    if ("WTFINL" in cps.columns) and ("WAGE" in cps.columns) and strat_cols:
        target_univ, wage_present = reweight_poststrat(cps.select([*(set(strat_cols)|{"WTFINL","WAGE","YEAR"})]), weight_col="WTFINL", response_col="WAGE", min_wage=0, cell_vars=strat_cols)
        cps = cps.with_row_index(name="__rid")
        wage_present = wage_present.with_row_index(name="__rid")
        cps = cps.join(wage_present.select(["__rid", "WTFINL_ADJ"]), on="__rid", how="left").drop("__rid")
        cps = cps.with_columns(
            pl.coalesce([pl.col("WTFINL_ADJ"), pl.col("WTFINL")]).alias("cps_weight")
        ).drop(["WTFINL_ADJ"], strict=False)
    elif "WTFINL" in cps.columns:
        cps = cps.with_columns(pl.col("WTFINL").alias("cps_weight"))
    else:
        cps = cps.with_columns(pl.lit(1.0).alias("cps_weight"))

    # Wages & Deflation
    has_real = ("WAGE_REAL" in cps.columns) and ("LOG_WAGE_REAL" in cps.columns)
    if not has_real:
        cps = add_topcode_and_real_wages(cps)
    cps = cps.with_columns(pl.col("LOG_WAGE_REAL").alias("logw"))
    cps = _winsorize_by_year(cps, "logw", 0.01, 0.99)

    # Derive cells and psi/high_psi_flag
    if "cell_id" not in cps.columns:
        logger.warning("CPS: cell_id missing; re-deriving from scratch")
        cps = _derive_cell_components(cps)
    
    if "psi" not in cps.columns:
        if "TELEWORKABLE_OCSSOC_DETAILED" in cps.columns:
            cps = cps.with_columns(
                pl.col("TELEWORKABLE_OCSSOC_DETAILED").cast(pl.Float64).alias("psi")
            )
        else:
            cps = cps.with_columns(pl.lit(0.5).alias("psi"))

    # Observed Telework (2022–2025)
    if "ALPHA" not in cps.columns and {"TELWRKHR","UHRSWORKT"}.issubset(cps.columns):
        cps = cps.with_columns((pl.col("TELWRKHR") / pl.col("UHRSWORKT")).alias("ALPHA"))

    # Thresholds to not rely on exact results of real division
    thr_hi, thr_lo = 0.995, 0.005
    flags = {
        "FULL_REMOTE": (pl.col("ALPHA") >= thr_hi),
        "FULL_INPERSON": (pl.col("ALPHA") <= thr_lo),
        "HYBRID": (pl.col("ALPHA") > thr_lo) & (pl.col("ALPHA") < thr_hi),
    }
    for d, expr in flags.items():
        if d not in cps.columns:
            cps = cps.with_columns(expr.cast(pl.Int8).alias(d))


    # Collapse to cell×year with weights
    wt_col = "ATUS_WT" if "ATUS_WT" in atus_join.columns else None
    def _wmean(col: str) -> pl.Expr:
        weight = pl.col(wt_col) if wt_col else 1.0
        return (pl.col(col) * weight).sum() / weight.sum()

    def _wmean_conditional(val_col: str, cond_col: str) -> pl.Expr:
        weight = pl.col(wt_col) if wt_col else 1.0
        condition = pl.col(cond_col)
        numerator = (pl.when(condition).then(pl.col(val_col)).otherwise(None) * weight).sum()
        denominator = (pl.when(condition).then(weight).otherwise(None)).sum()
        return numerator / denominator

    atus_cell = (
        atus_join
        .with_columns([
            (pl.col("work_minutes") > 0).alias("worked"),
            (pl.col("remote_minutes") > 0).alias("any_remote"),
            ((pl.col("remote_minutes") == pl.col("work_minutes")) & (pl.col("work_minutes") > 0)).alias("full_remote"),
            ((pl.col("remote_minutes") == 0) & (pl.col("work_minutes") > 0)).alias("full_inperson"),
            ((pl.col("remote_minutes") > 0) & (pl.col("remote_minutes") < pl.col("work_minutes"))).alias("hybrid"),
            # intensity
            pl.when(pl.col("work_minutes") > 0)
            .then(pl.col("remote_minutes") / pl.col("work_minutes"))
            .otherwise(None)
            .alias("alpha_atus"),
            # unify weights (use whatever you have: ATUS_WT/WT06/WT20)
            pl.coalesce([pl.col("ATUS_WT")])
            .cast(pl.Float64)
            .alias("ATUS_WEIGHT"),
        ])
        .group_by(["YEAR", "cell_id"])
        .agg([
            # weighted day shares (workday-conditional means)
            _wmean("any_remote").alias("p_any_remote_uncond"),
            _wmean_conditional("any_remote", "worked").alias("p_RH_workday"),
            _wmean_conditional("alpha_atus", "worked").alias("mean_alpha_workday"),
            _wmean_conditional("full_remote", "worked").alias("share_remote_day"),
            _wmean_conditional("hybrid", "worked").alias("share_hybrid_day"),
            _wmean_conditional("full_inperson", "worked").alias("share_inperson_day"),

            # counts and weight mass
            pl.len().alias("N_diaries"),
            pl.when(pl.col("worked")).then(pl.col("ATUS_WEIGHT")).otherwise(0.0).sum().alias("W_worked"),
            pl.when(pl.col("worked")).then(pl.col("ATUS_WEIGHT").pow(2)).otherwise(0.0).sum().alias("W2_worked"),
        ])
        # Kish effective n using worked weights (guard denominator)
        .with_columns(
            (pl.col("W_worked").pow(2) /
            pl.when(pl.col("W2_worked") > 0).then(pl.col("W2_worked")).otherwise(None)
            ).alias("n_eff_diaries")
        )
        .drop(["W2_worked"])
        # Renormalize shares to sum to 1 (defensive)
        .with_columns((pl.col("share_remote_day")+pl.col("share_hybrid_day")+pl.col("share_inperson_day")).alias("_ss"))
        .with_columns([
            (pl.col("share_remote_day")  / pl.col("_ss")).alias("share_remote_day"),
            (pl.col("share_hybrid_day")  / pl.col("_ss")).alias("share_hybrid_day"),
            (pl.col("share_inperson_day")/ pl.col("_ss")).alias("share_inperson_day"),
        ])
        .drop("_ss")
    )


    # Log distribution of N_diaries across cells
    try:
        n_diaries_stats = atus_cell.select([
            pl.col("N_diaries").min().alias("min_diaries"),
            pl.col("N_diaries").quantile(0.25).alias("p25_diaries"),
            pl.col("N_diaries").median().alias("median_diaries"),
            pl.col("N_diaries").quantile(0.75).alias("p75_diaries"),
            pl.col("N_diaries").max().alias("max_diaries"),
            pl.col("N_diaries").mean().alias("mean_diaries"),
            pl.len().alias("total_cells")
        ]).to_dicts()[0]
        
        logger.info(f"ATUS N_diaries distribution across {n_diaries_stats['total_cells']:,} cell×year combinations:")
        logger.info(f"  Min: {n_diaries_stats['min_diaries']:,}")
        logger.info(f"  P25: {n_diaries_stats['p25_diaries']:,.1f}")
        logger.info(f"  Median: {n_diaries_stats['median_diaries']:,.1f}")
        logger.info(f"  Mean: {n_diaries_stats['mean_diaries']:,.1f}")
        logger.info(f"  P75: {n_diaries_stats['p75_diaries']:,.1f}")
        logger.info(f"  Max: {n_diaries_stats['max_diaries']:,}")
        
        # Log cells with very few diaries
        low_diary_cells = atus_cell.filter(pl.col("N_diaries") <= 5).height
        share_low = low_diary_cells / n_diaries_stats['total_cells'] if n_diaries_stats['total_cells'] > 0 else 0
        logger.info(f"  Cells with ≤5 diaries: {low_diary_cells:,} ({share_low:.1%})")
        
    except Exception as e:
        logger.warning(f"Failed to compute N_diaries distribution: {e}")

    # Add placeholder SEs and metadata to ATUS measures
    atus_cell = atus_cell.with_columns([
        pl.lit(None, dtype=pl.Float64).alias(f"se_{col}")
        for col in ["p_RH", "p_RH_workday", "mean_alpha_workday", "share_remote_day", "share_hybrid_day", "share_inperson_day"]
    ]).with_columns([
        pl.col("N_diaries").cast(pl.Float64).alias("n_eff_diaries"),
        pl.lit(0, dtype=pl.Int8).alias("pooled_flag"),
        pl.lit("501xx").alias("workcode_spec"),
        pl.lit("WT06/WT20 coalesce").alias("weight_spec"),
    ])

    # 4) Prepare final CPS microdata outputs
    keep_cols = [
        "YEAR", "MONTH", "cps_weight", "logw", "WAGE", "UHRSWORKT",
        # "occ2_harmonized", "ind_broad", "ftpt", "edu3", "age4", "sex", "state",
        "occ2_harmonized", "ftpt", "edu3", "sex",
        "cell_id", "psi",
        "ALPHA", "FULL_REMOTE", "HYBRID", "FULL_INPERSON"
    ]
    cps_mi = cps.select([c for c in keep_cols if c in cps.columns])
    
    # Join ATUS measures into CPS microdata
    cps_mi = cps_mi.join(
        atus_cell.select(["YEAR", "cell_id", "p_RH_workday"]),
        on=["YEAR", "cell_id"], how="left"
    )
    
    # Compute weights for RH/IP projection
    cps_mi = cps_mi.with_columns([
        (pl.col("cps_weight") * pl.col("p_RH_workday")).alias("w_RH"),
        (pl.col("cps_weight") * (1 - pl.col("p_RH_workday"))).alias("w_IP"),
    ])
    
    # Create observed telework subset
    cps_obs = cps_mi.filter(pl.col("YEAR") >= 2022)

    # 5) Optional bridge lambdas for 2022–2025
    bridge = None
    if produce_bridge:
        bridge = build_bridge_lambdas(
            cps_mi=cps_mi,
            atus_cell=atus_cell,
            eps=1e-6,
            # clip=(0.5, 2.0),
            # clip=(0.33, 3.0),
            clip=(0.25, 4.0),
            pool_years=[2022, 2023, 2024],
        )
        
        # Call the diagnostics function right after building bridge
        if  bridge is not None:
            diagnose_bridge(bridge, cfg, logger)

    # 6) Export and Sanity Checks
    out_dir = (cfg.processed_dir / "empirical").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {
        "cps_mi_ready": out_dir / "cps_mi_ready.dta",
        "cps_observed_telework": out_dir / "cps_observed_telework.dta",
        "atus_cell_measures": out_dir / "atus_cell_measures.dta",
    }
    if bridge is not None:
        out_paths["bridge_lambda"] = out_dir / "bridge_lambda.dta"

    _export_stata(cps_mi, out_paths["cps_mi_ready"])
    _export_stata(cps_obs, out_paths["cps_observed_telework"])
    _export_stata(atus_cell, out_paths["atus_cell_measures"])
    if bridge is not None:
        _export_stata(bridge, out_paths["bridge_lambda"])

    if write_schema:
        for k, p in out_paths.items():
            # ... (schema writing logic) ...
            pass

    logger.info("Pre-estimation artifacts written:")
    for k, p in out_paths.items():
        logger.info(f" - {k}: {p}")

    # Final, consolidated sanity checks
    try:
        # Pre-2022 coverage of cell_id and p_RH_workday by weight
        pre = cps_mi.filter(pl.col("YEAR") < 2022)
        denom = pre.select(pl.col("cps_weight").sum()).item()
        if denom and denom > 0:
            numer = pre.filter(pl.col("cell_id").is_not_null() & pl.col("p_RH_workday").is_not_null()) \
                       .select(pl.col("cps_weight").sum()).item()
            share = (numer / denom) if numer else 0.0
            logger.info(f"Sanity: pre-2022 weight share with non-missing cell_id & p_RH_workday = {share:.3%} (target ≥ 90%)")
        else:
            logger.info("Sanity: pre-2022 coverage unavailable (no relevant data)")

        # ind_broad availability
        ind_nonnull = cps_mi.select((pl.col("ind_broad").is_not_null()).mean()).item()
        logger.info(f"Sanity: share of CPS rows with ind_broad non-missing = {ind_nonnull:.3%}")

        # Lambda centering check
        if produce_bridge and bridge is not None and len(bridge) > 0:
            med_remote = bridge.select(pl.col("lambda_remote").drop_nulls().median()).item()
            med_hybrid = bridge.select(pl.col("lambda_hybrid").drop_nulls().median()).item()
            med_inpers = bridge.select(pl.col("lambda_inperson").drop_nulls().median()).item()
            logger.info(f"Sanity: Overall median λ — remote={med_remote:.3f}, hybrid={med_hybrid:.3f}, inperson={med_inpers:.3f} (expect ~1)")
    except Exception as e:
        logger.warning(f"Sanity checks failed: {e}")

    return out_paths

def main() -> None:
    cfg = load_config()
    # Run base pipelines if needed
    # atus_link = atus_pipeline(cfg)
    # print(f"ATUS link written: {atus_link}")
    # cps_out = cps_pipeline(cfg)
    # print(f"CPS processed written: {cps_out}")
    # Build Stata-ready artifacts
    pre_paths = preestimation_procedure(cfg)
    # print("Pre-estimation artifacts:")
    # for k, p in pre_paths.items():
        # print(f"  {k}: {p}")

if __name__ == "__main__":
    main()