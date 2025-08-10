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
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict

import polars as pl
import pandas as pd  # used only for reading Excel (SOC aggregator)
import yaml
import logging
import time
import numpy as np
from datetime import date
try:
    from pandas_datareader import data as web
except Exception:
    web = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
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


def load_soc_aggregator(path: Path) -> pl.DataFrame:
    """Load SOC structure and normalize to expected column names.

    Tries a couple of header offsets and column name variants, then returns a
    DataFrame with columns: Detailed Occupation, Broad Group, Minor Group, Major Group.
    """
    tried: list[pd.DataFrame] = []
    for skip in (7, 0, 1, 2):
        try:
            df = pd.read_excel(path, skiprows=skip)
            tried.append(df)
        except Exception:
            continue

    if not tried:
        raise RuntimeError(f"Unable to read SOC aggregator: {path}")

    # Choose the first with the most columns
    agg = max(tried, key=lambda d: d.shape[1])

    # Normalize columns to expected names when possible
    colmap = {}
    lower = {c.lower(): c for c in agg.columns}
    # Common variants
    det = lower.get("detailed occupation") or lower.get("detailed occupation code") or lower.get("detailed") or lower.get("detailed soc")
    bro = lower.get("broad group") or lower.get("broad occupation") or lower.get("broad soc")
    minor = lower.get("minor group") or lower.get("minor occupation") or lower.get("minor soc")
    major = lower.get("major group") or lower.get("major occupation") or lower.get("major soc")
    if det:
        colmap[det] = "Detailed Occupation"
    if bro:
        colmap[bro] = "Broad Group"
    if minor:
        colmap[minor] = "Minor Group"
    if major:
        colmap[major] = "Major Group"
    agg = agg.rename(columns=colmap)

    expected = {"Detailed Occupation", "Broad Group", "Minor Group", "Major Group"}
    missing = expected.difference(agg.columns)
    if missing:
        raise RuntimeError(f"SOC aggregator missing columns: {sorted(missing)}")
    # Convert to Polars for downstream processing
    return pl.from_pandas(agg)


def normalize_soc_agg(agg: pl.DataFrame) -> pl.DataFrame:
    """Forward-fill SOC hierarchy so each row carries Major/Minor/Broad with Detailed."""
    cols = ["Major Group", "Minor Group", "Broad Group", "Detailed Occupation"]
    df = agg.select([pl.col(c).cast(pl.Utf8).str.strip_chars().alias(c) for c in cols])
    df = df.with_row_count("_order").sort("_order")
    df = df.with_columns([
        pl.col("Major Group").forward_fill(),
        pl.col("Minor Group").forward_fill(),
        pl.col("Broad Group").forward_fill(),
        pl.col("Detailed Occupation"),
    ])
    # Normalize codes to XX-XXXX length where present
    df = df.with_columns([
        pl.when(pl.col(c).is_not_null()).then(pl.col(c).str.slice(0, 7)).otherwise(pl.col(c)).alias(c)
        for c in cols
    ])
    return df.drop("_order")


def add_soc_groupings(df: pl.DataFrame, occ_col: str, agg: pl.DataFrame, prefix: str = "OCCSOC") -> pl.DataFrame:
    """
    Add SOC groupings based on the original modify_occupation_codes logic.
    This assumes the occ_col already contains SOC codes (from OCC->SOC mapping).
    """
    # Use normalized aggregator with forward-filled hierarchy
    agg_f = normalize_soc_agg(agg)

    # Membership lists
    detailed_list = agg_f.get_column("Detailed Occupation").unique().to_list()
    broad_list    = agg_f.get_column("Broad Group").unique().to_list()
    minor_list    = agg_f.get_column("Minor Group").unique().to_list()
    major_list    = agg_f.get_column("Major Group").unique().to_list()

    # Dictionaries from filled data
    det_rows = agg_f.filter(pl.col("Detailed Occupation").is_not_null())
    soc_dict_broad = dict(zip(det_rows.get_column("Detailed Occupation").to_list(),
                              det_rows.get_column("Broad Group").to_list()))
    soc_dict_minor = dict(zip(det_rows.get_column("Detailed Occupation").to_list(),
                              det_rows.get_column("Minor Group").to_list()))
    b2m_df = agg_f.select(["Broad Group", "Minor Group"]).unique()
    soc_dict_broad_to_minor = dict(zip(b2m_df.get_column("Broad Group").to_list(),
                                       b2m_df.get_column("Minor Group").to_list()))

    # Classify level of occ_col
    df = df.with_columns([
        pl.when(pl.col(occ_col).is_in(detailed_list)).then(pl.lit("detailed"))
        .when(pl.col(occ_col).is_in(broad_list)).then(pl.lit("broad"))
        .when(pl.col(occ_col).is_in(minor_list)).then(pl.lit("minor"))
        .when(pl.col(occ_col).is_in(major_list)).then(pl.lit("major"))
        .otherwise(pl.lit("none"))
        .alias(f"{prefix}_group")
    ])

    # Detailed
    df = df.with_columns([
        pl.when(pl.col(f"{prefix}_group") == "detailed").then(pl.col(occ_col)).otherwise(pl.lit(None)).alias(f"{prefix}_detailed")
    ])

    # Broad: keep if broad, map if detailed
    df = df.with_columns([
        pl.when(pl.col(f"{prefix}_group") == "broad").then(pl.col(occ_col))
        .when(pl.col(f"{prefix}_group") == "detailed")
        .then(pl.col(occ_col).cast(pl.Utf8).replace_strict(soc_dict_broad, default=None))
        .otherwise(pl.lit(None))
        .alias(f"{prefix}_broad")
    ])

    # Minor: keep if minor, map from broad/detailed
    df = df.with_columns([
        pl.when(pl.col(f"{prefix}_group") == "minor").then(pl.col(occ_col))
        .when(pl.col(f"{prefix}_group") == "broad")
        .then(pl.col(occ_col).cast(pl.Utf8).replace_strict(soc_dict_broad_to_minor, default=None))
        .when(pl.col(f"{prefix}_group") == "detailed")
        .then(pl.col(occ_col).cast(pl.Utf8).replace_strict(soc_dict_minor, default=None))
        .otherwise(pl.lit(None))
        .alias(f"{prefix}_minor")
    ])

    return df.drop([f"{prefix}_group"], strict=False)


def assign_teleworkability(df: pl.DataFrame, agg: pl.DataFrame, tw_path: Path, occ_det_col: str, out_prefix: str = "TELEWORKABLE_OCSSOC") -> pl.DataFrame:
    if not tw_path.exists():
        return df
    tw = pl.read_csv(str(tw_path))
    # Normalize expected columns
    cols_lower = {c.lower(): c for c in tw.columns}
    occ_col = cols_lower.get("occ_code") or cols_lower.get("occsoc") or "OCC_CODE"
    tel_col = cols_lower.get("teleworkable") or "TELEWORKABLE"
    tw = tw.rename({occ_col: "OCC_CODE", tel_col: "TELEWORKABLE"}).select([
        pl.col("OCC_CODE").cast(pl.Utf8), pl.col("TELEWORKABLE").cast(pl.Float64)
    ])

    # Detailed level join
    left = df.with_columns(pl.col(occ_det_col).cast(pl.Utf8))
    left = left.join(tw, left_on=occ_det_col, right_on="OCC_CODE", how="left")
    left = left.rename({"TELEWORKABLE": f"TELEWORKABLE_{occ_det_col.upper()}"}).drop(["OCC_CODE"], strict=False)

    # Group averages: map OCC_CODE -> Broad/Minor via agg
    occ_to_group = agg.select([
        pl.col("Detailed Occupation").cast(pl.Utf8).alias("OCC_CODE"),
        pl.col("Broad Group").cast(pl.Utf8),
        pl.col("Minor Group").cast(pl.Utf8),
    ])
    tw_cl = tw.join(occ_to_group, on="OCC_CODE", how="left")
    tw_broad = tw_cl.group_by("Broad Group", maintain_order=False).agg(pl.col("TELEWORKABLE").mean().alias("TW_BROAD"))
    tw_minor = tw_cl.group_by("Minor Group", maintain_order=False).agg(pl.col("TELEWORKABLE").mean().alias("TW_MINOR"))

    base = occ_det_col.replace("_detailed", "")
    broad_col = base + "_broad"
    minor_col = base + "_minor"

    out = left.join(
        tw_broad.rename({"Broad Group": broad_col, "TW_BROAD": f"TELEWORKABLE_{base.upper()}_broad"}),
        on=broad_col,
        how="left",
    ).join(
        tw_minor.rename({"Minor Group": minor_col, "TW_MINOR": f"TELEWORKABLE_{base.upper()}_minor"}),
        on=minor_col,
        how="left",
    )
    return out


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
            lambda s: hour_topcode_dict.get((s["YEAR"], s["MONTH"], s["MISH"]), None)
        ).alias("HOUR_TOPCODE_THRESHOLD")
    )

    # Monthly max and counts
    monthly_max = (
        data.group_by(["YEAR", "MONTH"]).agg([
            pl.col("WAGE").max().alias("MONTH_MAX_WAGE"),
            pl.count().alias("MONTH_COUNT"),
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
        .group_by(["YEAR", "MONTH"]).agg(pl.count().alias("MAX_WAGE_COUNT"))
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
        logging.warning("CPI series empty; real wage variables not created.")

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

    # Ensure columns exist per fetch logic
    for c in ["YEAR", "CPSIDP", "WHERE", "ACTIVITY", "DURATION"]:
        if c not in df.columns:
            raise RuntimeError(f"Missing required ATUS column: {c}")

    # Define location and work codes (IPUMS WHERE)
    home_where = [101]         # Respondent's home or yard
    workplace_where = [102]    # Respondent's workplace
    where_unknown = [89, 99, 9997, 9998, 9999]  # unspecified / NA

    # Drop any records with unknown location
    df = df.filter(~pl.col("WHERE").is_in(where_unknown))

    # Drop Any Records where Diary Day was a Holliday
    df = df.filter(~pl.col("HOLIDAY").is_in([1]))

    # 1) Parse YYYYMMDD to a proper Date
    df = df.with_columns(
        pl.col("DATE").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=True).alias("DATE")
    )

    # 2) Derive weekday (Mon=0 ... Sun=6)
    df = df.with_columns(
        pl.col("DATE").dt.weekday().alias("DAY_OF_WEEK")
    )

    # 3) Keep only weekdays (Mon–Fri)
    df = df.filter(pl.col("DAY_OF_WEEK") < 5)

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

    # Aggregate to person‐year and derive flags
    link = (
        df.group_by(["YEAR", "CPSIDP"]).agg([
            pl.col("remote_minutes").sum().alias("remote_minutes"),
            pl.col("work_minutes").sum().alias("work_minutes"),
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
    soc_path = Path(resolve_tokens(cw_cfg["soc_aggregator"]["path"], cfg))
    agg = load_soc_aggregator(soc_path)
    if "OCCSOC" in df.columns:
        df = df.with_columns(pl.col("OCCSOC").cast(pl.Utf8).str.strip_chars())
        df = add_soc_groupings(df, "OCCSOC", agg, prefix="OCCSOC")
        tw_path = Path(resolve_tokens(cw_cfg["teleworkability"]["path"], cfg))
        df = assign_teleworkability(df, agg, tw_path, "OCCSOC_detailed", out_prefix="TELEWORKABLE_OCSSOC")
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

    atus_link_path = cfg.get_output_path("atus_link", cfg.processed_dir / "atus_link.csv").resolve()
    if not atus_link_path.exists():
        raise FileNotFoundError(f"ATUS link not found: {atus_link_path}. Run ATUS pipeline first.")

    cps = pl.read_csv(str(cps_micro_path))
    atus = pl.read_csv(str(atus_link_path))
    logger.info(f"CPS: Loaded {len(cps):,} rows from {cps_micro_path}")
    logger.info(f"CPS: Loaded {len(atus):,} ATUS linkage records")

    for col in ["YEAR", "MONTH"]:
        if col in cps.columns:
            cps = cps.with_columns(pl.col(col).cast(pl.Int32))
    if "YEAR" in atus.columns:
        atus = atus.with_columns(pl.col("YEAR").cast(pl.Int32))



    # Industry mapping (CPS IND -> NAICS definitive)
    cw_cfg = cfg.crosswalks
    ind_cfg = cw_cfg["cps_industry"]
    ind_path = Path(resolve_tokens(ind_cfg["path"], cfg))
    ind_cw = pl.read_csv(str(ind_path))
    cps = map_with_crosswalk(cps, ind_cw, ind_cfg["map"]["source_column"], ind_cfg["map"]["crosswalk_key"], ind_cfg["map"]["target_column"], out_col=ind_cfg.get("output_column", "IND_MAPPED"))
    logger.info(f"CPS: Mapped industry codes")

    # OCC -> SOC mapping with aggregation for multiple mappings
    soc_path = Path(resolve_tokens(cw_cfg["soc_aggregator"]["path"], cfg))
    agg = load_soc_aggregator(soc_path)
    occsoc_cfg = cw_cfg["cps_occ_to_soc"]
    occsoc_path = Path(resolve_tokens(occsoc_cfg["path"], cfg))
    occsoc_cw = pl.read_csv(str(occsoc_path))
    
    # Process OCC->SOC mapping to handle multiple mappings by aggregating to common prefix levels
    cps_to_soc_mappings = occsoc_cw.group_by(
        pl.col(occsoc_cfg["map"]["crosswalk_key"]).cast(pl.Utf8)  # Ensure OCC is a string
    ).agg(pl.col(occsoc_cfg["map"]["target_column"]).alias("SOC"))
    
    # Create dictionary to handle multiple mappings like in original code
    cps_to_soc_dict = {}
    multiple_mappings_count = 0
    
    for row in cps_to_soc_mappings.iter_rows(named=True):
        cps_code = str(row[occsoc_cfg["map"]["crosswalk_key"]])
        soc_codes = row["SOC"]
        
        # If only one SOC code, use it directly
        if len(soc_codes) == 1:
            cps_to_soc_dict[cps_code] = str(soc_codes[0])
            continue
            
        # For multiple mappings, find common prefix at appropriate level
        multiple_mappings_count += 1
        
        # Try detailed level first (full code)
        if all(code == soc_codes[0] for code in soc_codes):
            cps_to_soc_dict[cps_code] = str(soc_codes[0])
            continue
            
        # Try broad level (xx-xxxx)
        broad_prefixes = [code[:6] for code in soc_codes if len(str(code)) >= 6]
        if len(set(broad_prefixes)) == 1 and len(broad_prefixes) == len(soc_codes):
            aggregated_code = broad_prefixes[0] + "00"
            cps_to_soc_dict[cps_code] = aggregated_code
            continue
            
        # Try minor level (xx-xx)
        minor_prefixes = [code[:4] for code in soc_codes if len(str(code)) >= 4]
        if len(set(minor_prefixes)) == 1 and len(minor_prefixes) == len(soc_codes):
            aggregated_code = minor_prefixes[0] + "00"
            cps_to_soc_dict[cps_code] = aggregated_code
            continue
            
        # Default to major level (xx)
        major_prefixes = [code[:2] for code in soc_codes if len(str(code)) >= 2]
        if len(set(major_prefixes)) == 1 and len(major_prefixes) == len(soc_codes):
            aggregated_code = major_prefixes[0] + "-0000"
            cps_to_soc_dict[cps_code] = aggregated_code
            continue
            
        # If no common prefix, assign null no clear occupational mapping
        cps_to_soc_dict[cps_code] = None
    
    logger.info(f"CPS: Created OCC->SOC crosswalk with {len(cps_to_soc_dict)} mappings")
    logger.info(f"CPS: Found {multiple_mappings_count} OCC codes with multiple SOC mappings")
    
    # Apply the crosswalk to map OCC codes to SOC codes
    cps = cps.with_columns(pl.col("OCC").cast(pl.Utf8).str.strip_chars())
    
    # Map OCC to SOC using the processed dictionary
    cps = cps.with_columns(
        pl.col("OCC").replace_strict(cps_to_soc_dict, default=None).alias("OCCSOC")
    )

    # Patch: Ensure OCCSOC codes are exactly 7 characters long (XX-XXXX)
    cps = cps.with_columns(
        pl.when(pl.col("OCCSOC").is_not_null())
        .then(pl.col("OCCSOC").str.slice(0, 7))  # Trim to 7 characters
        .otherwise(pl.col("OCCSOC"))
        .alias("OCCSOC")
    )
    
    # Log mapping results
    mapped_count = cps.filter(pl.col("OCCSOC").is_not_null()).height
    total_count = len(cps)
    logger.info(f"CPS: Successfully mapped {mapped_count:,}/{total_count:,} OCC codes to SOC ({mapped_count/total_count:.1%})")

    cps = add_soc_groupings(cps, "OCCSOC", agg, prefix="OCCSOC")
    tw_path = Path(resolve_tokens(cw_cfg["teleworkability"]["path"], cfg))
    cps = assign_teleworkability(cps, agg, tw_path, "OCCSOC_detailed", out_prefix="TELEWORKABLE_OCSSOC")
    logger.info(f"CPS: Added SOC groupings and teleworkability")



    # Keep all rows pre-2022; left join ATUS on CPSIDP+YEAR
    cps = cps.join( 
            atus.select(["YEAR", "CPSIDP", "remote_minutes", "work_minutes", "any_remote"]),
            on=["YEAR", "CPSIDP"],
            how="left"
        )
    logger.info(f"CPS: After joining with ATUS data: {len(cps):,} rows")
    # Telework variables:
    # After 2022: use TELWRKPAY/TELWRKHR/UHRSWORKT logic exactly as specified.
    # For pre-2022 (and any rows where ALPHA remains null, including early 2022 months), fill using minutes ratio.
    has_post_cols = all(col in cps.columns for col in ["TELWRKHR", "TELWRKPAY", "UHRSWORKT"])
    if has_post_cols:
        # Cast TELWRKHR to float
        cps = cps.with_columns( pl.col("TELWRKHR").cast(pl.Float64) )
        cps = cps.with_columns(
                pl.when(pl.col("TELWRKPAY") == "0")                           # NIU → not in universe
                        .then(pl.lit(None))                                      # leave ALPHA_TEL null
                    .when(pl.col("TELWRKPAY") == "2")                           # Pay flag=2 → no telework
                        .then(0.0)                                               # ALPHA_TEL = 0
                    .when(pl.col("TELWRKPAY") == "1")                           # Pay flag=1 → compute share
                        .then(
                            pl.when(pl.col("UHRSWORKT").is_in([0, 997, 999])        # invalid total hours
                            | (pl.col("TELWRKHR") == 999)       )              # or invalid remote hours
                                .then(pl.lit(None))                                 # drop to null
                            .otherwise(pl.col("TELWRKHR") / pl.col("UHRSWORKT")))
                    .otherwise(pl.lit(None))                                  # any other code → null
                .alias("_ALPHA_TEL")
            )
    else:
        cps = cps.with_columns(pl.lit(None).cast(pl.Float64).alias("_ALPHA_TEL"))

    # Fallback ALPHA from ATUS minutes for rows with null ALPHA (pre-2022 and early 2022)
    has_minutes = all(col in cps.columns for col in ["remote_minutes", "work_minutes"])
    if has_minutes:
        cps = cps.with_columns([
            pl.col("remote_minutes").cast(pl.Float64).alias("remote_minutes"),
            pl.col("work_minutes").cast(pl.Float64).alias("work_minutes"),
        ])
        # 2) Minutes‐based fallback (pre‐2022 or missing hours info):
        cps = cps.with_columns(
            pl.when((pl.col("work_minutes") > 0) & pl.col("remote_minutes").is_not_null())
            .then(pl.col("remote_minutes") / pl.col("work_minutes"))
            .otherwise(pl.lit(None))
            .alias("_ALPHA_MIN")
        )
    else:
        cps = cps.with_columns(pl.lit(None).cast(pl.Float64).alias("_ALPHA_MIN"))

    # 3) Final ALPHA: prefer hours‐based, then minutes‐based, then clamp to [0,1]
    cps = cps.with_columns(pl.coalesce([pl.col("_ALPHA_TEL"), pl.col("_ALPHA_MIN")]).alias("ALPHA")
            ).with_columns(pl.when(pl.col("ALPHA") > 1).then(1.0)
            .when(pl.col("ALPHA") < 0).then(0.0)
            .otherwise(pl.col("ALPHA"))
            .alias("ALPHA")
        )

    # Dummies and WFH from final ALPHA
    cps = cps.with_columns([
        (pl.col("ALPHA").is_not_null() & (pl.col("ALPHA") == 0)).cast(pl.Int64).alias("FULL_INPERSON"),
        (pl.col("ALPHA").is_not_null() & (pl.col("ALPHA") == 1)).cast(pl.Int64).alias("FULL_REMOTE"),
        (pl.col("ALPHA").is_not_null() & (pl.col("ALPHA") > 0) & (pl.col("ALPHA") < 1)).cast(pl.Int64).alias("HYBRID"),
        (pl.col("ALPHA").is_not_null() & (pl.col("ALPHA") > 0)).cast(pl.Int64).alias("WFH"),
    ]).drop(["_ALPHA_TEL", "_ALPHA_MIN"], strict=False)

    # Optional yearly stats if columns present - CALCULATE BEFORE ANY WAGE FILTERING
    if all(c in cps.columns for c in ["YEAR", "WFH", "FULL_INPERSON", "FULL_REMOTE", "HYBRID"]):
        # Only do this simple average when TELEWORK information is present
        cps_filt = cps.filter(
            pl.col("ALPHA").is_not_null()
        )
        # Compute weighted averages using WTFINL as weight column
        stats = (
            cps_filt.group_by("YEAR")
            .agg([
            (pl.col("WFH") * pl.col("WTFINL")).sum() / pl.col("WTFINL").sum().alias("WFH"),
            (pl.col("FULL_INPERSON") * pl.col("WTFINL")).sum() / pl.col("WTFINL").sum().alias("FULL_INPERSON"),
            (pl.col("FULL_REMOTE") * pl.col("WTFINL")).sum() / pl.col("WTFINL").sum().alias("FULL_REMOTE"),
            (pl.col("HYBRID") * pl.col("WTFINL")).sum() / pl.col("WTFINL").sum().alias("HYBRID"),
            pl.len().alias("N")
            ])
            .sort("YEAR")
        )
        print("\nYearly statistics for WFH dummies (BEFORE wage filtering):")
        for row in stats.iter_rows(named=True):
            print(f"Year {row['YEAR']}: WFH={row['WFH']:.3f}, Full In-Person={row['FULL_INPERSON']:.3f}, Full Remote={row['FULL_REMOTE']:.3f}, Hybrid={row['HYBRID']:.3f}, N={row['N']:,}")
    logger.info("WFH, ALPHA, FULL_INPERSON, FULL_REMOTE, HYBRID variables created.")

    # ------------------------------------------------------------
    # Construct WAGE from HOURWAGE2/EARNWEEK2/UHRSWORKT (ignore legacy HOURWAGE/EARNWEEK)
    # ------------------------------------------------------------
    has_wage_cols = all(c in cps.columns for c in ["HOURWAGE2", "EARNWEEK2", "UHRSWORKT"])
    if has_wage_cols:
        # Ensure numeric types for operations
        cps = cps.with_columns([
            pl.col("HOURWAGE2").cast(pl.Float64, strict=False).alias("HOURWAGE2"),
            pl.col("EARNWEEK2").cast(pl.Float64, strict=False).alias("EARNWEEK2"),
            pl.col("UHRSWORKT").cast(pl.Float64, strict=False).alias("UHRSWORKT"),
        ])

        initial_n = len(cps)
        # Keep rows with usable wage info according to provided logic
        cps = cps.filter(
            (pl.col("HOURWAGE2") != 999.99)
            | (
                (~pl.col("EARNWEEK2").is_in([9999.99, 999999.99]))
                & (~pl.col("UHRSWORKT").is_in([999, 997]))
            )
        )
        logger.info(
            f"CPS: Dropped rows with invalid wage information (from {initial_n:,} to {len(cps):,})."
        )

        # Build WAGE from HOURWAGE2 else EARNWEEK2/UHRSWORKT
        cps = cps.with_columns([
            pl.when(pl.col("HOURWAGE2") != 999.99)
            .then(pl.col("HOURWAGE2"))
            .when(
                ((pl.col("UHRSWORKT") == 999) | (pl.col("UHRSWORKT") == 997))
                & (~pl.col("EARNWEEK2").is_in([9999.99, 999999.99]))
            )
            .then(pl.col("EARNWEEK2") / pl.col("UHRSWORKT"))
            .otherwise(None)
            .alias("WAGE_TEMP2")
        ])

        cps = cps.with_columns(
            pl.col("WAGE_TEMP2").alias("WAGE")
        )
        logger.info("CPS: Created WAGE from HOURWAGE2 or EARNWEEK2/UHRSWORKT.")

        # Log non-null counts for wage-related columns
        for col in ["HOURWAGE2", "EARNWEEK2", "UHRSWORKT", "WAGE"]:
            non_null = cps.filter(~pl.col(col).is_null()).height
            logger.info(f"CPS: Column {col} has {non_null:,} non-null values.")

        cps = cps.drop(["WAGE_TEMP2"], strict=False)
    else:
        logger.info("CPS: Skipping wage construction; required columns not present.")

    out_path = cfg.get_output_path("cps", (cfg.processed_dir / "cps" / "cps_processed.csv").resolve())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cps.write_csv(str(out_path))
    logger.info(f"CPS: Final dataset has {len(cps):,} rows, saved to {out_path}")

    # ------------------------------------------------------------
    # Additional exports requested:
    # 1) CPS subset where ALPHA is not null (there is info about remote work)
    # 2) Same subset but dropping rows where WAGE is null
    #    Also drop rows where required controls are invalid or missing.
    # ------------------------------------------------------------
    # Controls filters
    def apply_controls_filter(df: pl.DataFrame) -> pl.DataFrame:
        # Prepare typed columns safely
        df2 = df
        # CLASSWKR filtering: drop 00/99/null
        if "CLASSWKR" in df2.columns:
            df2 = df2.with_columns(pl.col("CLASSWKR").cast(pl.Utf8, strict=False).str.strip_chars().alias("CLASSWKR"))
            df2 = df2.filter(pl.col("CLASSWKR").is_not_null() & (~pl.col("CLASSWKR").is_in(["00", "99"])))
        # AGE != 999 and not null
        if "AGE" in df2.columns:
            df2 = df2.with_columns(pl.col("AGE").cast(pl.Int32, strict=False).alias("AGE"))
            df2 = df2.filter(pl.col("AGE").is_not_null() & (pl.col("AGE") != 999))
        # SEX != 9 and not null
        if "SEX" in df2.columns:
            df2 = df2.with_columns(pl.col("SEX").cast(pl.Int32, strict=False).alias("SEX"))
            df2 = df2.filter(pl.col("SEX").is_not_null() & (pl.col("SEX") != 9))
        # RACE != 999 and not null
        if "RACE" in df2.columns:
            df2 = df2.with_columns(pl.col("RACE").cast(pl.Int32, strict=False).alias("RACE"))
            df2 = df2.filter(pl.col("RACE").is_not_null() & (pl.col("RACE") != 999))
        # HISPAN not in {901, 999} and not null
        if "HISPAN" in df2.columns:
            df2 = df2.with_columns(pl.col("HISPAN").cast(pl.Int32, strict=False).alias("HISPAN"))
            df2 = df2.filter(pl.col("HISPAN").is_not_null() & (~pl.col("HISPAN").is_in([901, 999])))
        return df2

    # 1) ALPHA present subset -> apply reweighting for wage nonresponse and attach adjusted weights where available
    alpha_present = cps.filter(pl.col("ALPHA").is_not_null())
    before_ctrl = len(alpha_present)
    alpha_present = apply_controls_filter(alpha_present)
    logger.info(
        f"CPS: ALPHA-present subset size: {before_ctrl:,} -> {len(alpha_present):,} after controls filter"
    )

    # Reweighting: compute adjusted weights using response availability (here response_col='WAGE')
    if "WTFINL" in alpha_present.columns:
        target_univ, resp_sub_adj = reweight_poststrat(alpha_present, weight_col="WTFINL", response_col="WAGE", min_wage=0)
        # Merge adjusted weights back into ALPHA-present universe (resp_sub_adj has WTFINL_ADJ for rows with response)
        # Use a stable key: if CPSIDP exists use that + YEAR + MONTH else fallback to row count index
        key_cols = [c for c in ["YEAR", "MONTH", "CPSIDP"] if c in alpha_present.columns]
        if key_cols:
            alpha_present = alpha_present.join(
                resp_sub_adj.select(key_cols + ["WTFINL_ADJ"]),
                on=key_cols,
                how="left"
            )
        else:
            alpha_present = alpha_present.with_row_count("__rid")
            resp_sub_adj = resp_sub_adj.with_row_count("__rid")
            alpha_present = alpha_present.join(resp_sub_adj.select(["__rid", "WTFINL_ADJ"]), on="__rid", how="left").drop("__rid")

        # Replace weight column with adjusted weight, keep old as <WEIGHT>_OLD
        alpha_present = alpha_present.with_columns(pl.col("WTFINL").alias("WTFINL_OLD"))
        alpha_present = alpha_present.with_columns(
            pl.when(pl.col("WTFINL_ADJ").is_not_null()).then(pl.col("WTFINL_ADJ")).otherwise(pl.col("WTFINL")).alias("WTFINL")
        ).drop(["WTFINL_ADJ"], strict=False)
    else:
        logger.warning("CPS: WTFINL not found; skipping reweighting for ALPHA-present subset.")

    # Apply topcoding and real wage adjustment
    alpha_present = add_topcode_and_real_wages(alpha_present)

    alpha_present_path = (cfg.processed_dir / "cps" / "cps_alpha_present_reweighted.csv").resolve()
    alpha_present_path.parent.mkdir(parents=True, exist_ok=True)
    alpha_present.write_csv(str(alpha_present_path))
    logger.info(f"CPS: Wrote ALPHA-present (with adjusted weights column WTFINL_ADJ where applicable) to {alpha_present_path}")

    # 2) ALPHA present and WAGE present -> this already has wage; ensure WTFINL_ADJ present; if missing, set WTFINL_ADJ = WTFINL
    alpha_wage_present = alpha_present
    if "WAGE" in alpha_wage_present.columns:
        alpha_wage_present = alpha_wage_present.filter(pl.col("WAGE").is_not_null())
    # Ensure weights: keep old, and use adjusted if available else original
    if "WTFINL" in alpha_wage_present.columns:
        if "WTFINL_OLD" not in alpha_wage_present.columns:
            alpha_wage_present = alpha_wage_present.with_columns(pl.col("WTFINL").alias("WTFINL_OLD"))
        # Already replaced in alpha_present; just ensure column exists
        # No-op replacement to keep consistent pipeline
        alpha_wage_present = alpha_wage_present.with_columns(pl.col("WTFINL").alias("WTFINL"))
    logger.info(f"CPS: ALPHA-present + WAGE-present subset size: {len(alpha_wage_present):,}")
    # Apply topcoding and real wage adjustment
    alpha_wage_present = add_topcode_and_real_wages(alpha_wage_present)

    alpha_wage_present_path = (cfg.processed_dir / "cps" / "cps_alpha_wage_present_reweighted.csv").resolve()
    alpha_wage_present.write_csv(str(alpha_wage_present_path))
    logger.info(f"CPS: Wrote ALPHA-present + WAGE-present (reweighted) subset to {alpha_wage_present_path}")
    return out_path


def main() -> None:
    cfg = load_config()
    atus_link = atus_pipeline(cfg)
    print(f"ATUS link written: {atus_link}")
    # acs_out = acs_pipeline(cfg)
    # print(f"ACS processed written: {acs_out}")
    cps_out = cps_pipeline(cfg)
    print(f"CPS processed written: {cps_out}")


if __name__ == "__main__":
    main()
