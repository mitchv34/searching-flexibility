#!/usr/bin/env python3
"""
SIPP post-processing (Polars)

- Loads combined SIPP CSV from data/raw/sipp (created by sipp_acquire.py).
- Uses main-job picker over up to 7 jobs to build person-month telework metrics.
- Aggregates to person-year with two weights:
  A: cross-sectional person-month weight (WPFINWGT)
  B: "worked-mass" weight (WPFINWGT × days × hours)
- Saves outputs under data/processed/sipp.

Requires: polars >= 0.20
"""
from __future__ import annotations

import sys
import gzip
import glob
import logging
from pathlib import Path
from typing import Iterable, List
import re

import polars as pl
import yaml
from typing import Optional, Any

# ------- config you might tweak -------
JOB_MAX = 7  # SIPP allows up to 7 jobs
WEIGHT_COL = "WPFINWGT"  # person-month base weight (cross-sectional)
CAL_YEAR = "year"     # calendar year column
MAIN_JOB_COL = "EMAIN_JOB"

# Epsilon for full remote full inperson
EPSILON = 0.0

# Optional: “employed any week” constraint using RWKESR1..5 recodes
EMPLOYED_WEEK_CODES: set[int] = set()  # e.g. {1,2,3,4}
# --------------------------------------


def _exists(df: pl.DataFrame, cols: Iterable[str]) -> list[str]:
    have = set(df.columns)
    return [c for c in cols if c in have]


def build_person_month(sipp: pl.DataFrame) -> pl.DataFrame:
    # sanity: require calendar year
    if CAL_YEAR not in sipp.columns:
        raise ValueError(
            f"Missing {CAL_YEAR}. Add RHCALYR to your SIPP variable list; "
            "it’s needed to form person–year aggregates."
        )

    df = sipp

    # ------ select job info for the main job (per person-month) ------
    # exprs = [
    #     pick_job_expr(df, "EJB{i}_DYSWKD",  "dyswkd"),
    #     pick_job_expr(df, "EJB{i}_DYSWKDH", "dyswkdh"),
    #     pick_job_expr(df, "TJB{i}_MWKHRS",  "mwhours"),
    #     pick_job_expr(df, "TJB{i}_OCC",     "occ"),
    #     pick_job_expr(df, "TJB{i}_IND",     "ind"),
    #     pick_job_expr(df, "EJB{i}_CLWRK",   "clwrk"),
    #     pick_job_expr(df, "EJB{i}_JBORSE",  "jborse"),
    # ]
    # # Ensure EMAIN_JOB is int64 if present
    # if MAIN_JOB_COL in df.columns:
    #     df = df.with_columns(pl.col(MAIN_JOB_COL).cast(pl.Int64))

    # pm = (
    #     df.with_columns(
    #         *exprs,
    #         pl.col(CAL_YEAR).alias("year"),
    #     )
    # )
    #! I'm not going to pick the job for now let's use only first job

    pm = (
        df.with_columns(
            pl.col("EJB1_DYSWKD").alias("dyswkd"),  # Number of days worked per week at job 1
            pl.col("EJB1_DYSWKDH").alias("dyswkdh"),# Number of days worked only at home for job 1
            pl.col("TJB1_MWKHRS").alias("mwhours"), # Average number of hours worked per week at job 1
            pl.col("TJB1_OCC").alias("occ"),        # Occupation code for job 1
            pl.col("TJB1_IND").alias("ind"),        # Industry code for job 1
            pl.col("EJB1_CLWRK").alias("clwrk"),    # Class of worker for job 1
            pl.col("EJB1_JBORSE").alias("jborse"),  # Type of work arrangement (1=Employer, 2=Self-employed, 3=Other work)
            pl.col(CAL_YEAR).alias("year"),
        )
    )

    # Preserve raw SIPP codes explicitly from TJB1_OCC / TJB1_IND
    pm = pm.with_columns([
        pl.col("occ").alias("occ_raw"),
        pl.col("ind").alias("ind_raw"),
    ])

    # --- Apply crosswalks (explicitly from TJB1_OCC and TJB1_IND) ---
    # Expect config.yml to define:
    # crosswalks:
    #   sipp_occ: { path: ".../occ_xwalk.csv", key: "<crosswalk_key_col>", value: "<target_col>" }
    #   sipp_ind: { path: ".../ind_xwalk.csv", key: "<crosswalk_key_col>", value: "<target_col>" }
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / "src" / "data" / "config.yml"
    with open(cfg_path, "r") as _f:
        _cfg = yaml.safe_load(_f) or {}
    _xw = (_cfg.get("crosswalks") or {})
    _occ_cfg = _xw.get("cps_occ_to_soc")
    _ind_cfg = _xw.get("cps_industry")
    if not _occ_cfg or not _ind_cfg:
        raise RuntimeError("Missing crosswalks.cps_occ_to_soc or crosswalks.cps_industry in config.yml.")

    # Load crosswalk CSVs (resolve ${...} tokens, then absolutize)
    occ_path_str = _resolve_tokens(str(_occ_cfg["path"]), _cfg)
    ind_path_str = _resolve_tokens(str(_ind_cfg["path"]), _cfg)
    if "${" in occ_path_str or "${" in ind_path_str:
        raise ValueError(f"Unresolved tokens in crosswalk paths: occ='{occ_path_str}', ind='{ind_path_str}'")

    occ_xw_path = Path(occ_path_str) if Path(occ_path_str).is_absolute() else (repo_root / occ_path_str).resolve()
    ind_xw_path = Path(ind_path_str) if Path(ind_path_str).is_absolute() else (repo_root / ind_path_str).resolve()
    if not occ_xw_path.exists() or not ind_xw_path.exists():
        raise FileNotFoundError(f"Crosswalk not found. occ: {occ_xw_path}, ind: {ind_xw_path}")

    occ_xw = pl.read_csv(str(occ_xw_path))
    ind_xw = pl.read_csv(str(ind_xw_path))

    # Keys and targets from config
    occ_key, occ_val = _occ_cfg["map"]["crosswalk_key"], _occ_cfg["map"]["target_column"]
    ind_key, ind_val = _ind_cfg["map"]["crosswalk_key"], _ind_cfg["map"]["target_column"]

    
    # Join and map OCC
    pm = (
        pm.join(
            occ_xw.select([pl.col(occ_key), pl.col(occ_val).alias("__occ_mapped")]),
            left_on="occ_raw",
            right_on=occ_key,
            how="left",
        )
        .with_columns(pl.coalesce([pl.col("__occ_mapped"), pl.col("occ_raw")]).alias("occ"))
        .drop(["__occ_mapped"])
    )

    # Join and map IND
    pm = (
        pm.join(
            ind_xw.select([pl.col(ind_key), pl.col(ind_val).alias("__ind_mapped")]),
            left_on="ind_raw",
            right_on=ind_key,
            how="left",
        )
        .with_columns(pl.coalesce([pl.col("__ind_mapped"), pl.col("ind_raw")]).alias("ind"))
        .drop(["__ind_mapped"])
    )

    # Explicitly cast to Float64 and use new column names to avoid type issues
    pm = pm.with_columns([
        pl.col("dyswkd").cast(pl.Float64).alias("dyswkd"),
        pl.col("dyswkdh").cast(pl.Float64).alias("dyswkdh"),
        pl.col("mwhours").cast(pl.Float64).alias("mwhours"),
        pl.col(WEIGHT_COL).cast(pl.Float64).alias("w_base"),
    ])

    # Drop occ_raw and ind_raw
    pm = pm.drop(["occ_raw", "ind_raw"])

    # clean & guards
    pm = pm.with_columns([
        pl.when((pl.col("dyswkd") >= 0) & (pl.col("dyswkdh") >= 0))
            .then(pl.min_horizontal(pl.col("dyswkdh"), pl.col("dyswkd")))
            .otherwise(None)
            .alias("dyswkdh_clamped"),
    ])


    pm = pm.with_columns([
        pl.when(pl.col("dyswkd").is_not_null() & pl.col("dyswkdh_clamped").is_not_null() & (pl.col("dyswkd") > 0))
            .then(pl.col("dyswkdh_clamped") / pl.col("dyswkd"))
            .otherwise(None)
            .alias("p_remote"),
    ])

    # Compute average hours worked per month (mean of TJB{i}_JOBHRS1 across jobs)
    jobhrs_cols = [f"TJB{i}_JOBHRS1" for i in range(1, JOB_MAX + 1) if f"TJB{i}_JOBHRS1" in df.columns]
    if jobhrs_cols:
        pm = pm.with_columns(
            pl.mean_horizontal(*[pl.col(c).cast(pl.Float64) for c in jobhrs_cols]).alias("mean_jobhrs1")
        )
    else:
        pm = pm.with_columns(pl.lit(None).alias("mean_jobhrs1"))

    # --- optional “employed any week” mask (off by default) ---

    # Optional: weekly recodes for “employed any week” filtering
    # wk_cols = _exists(df, [f"RWKESR{k}" for k in range(1, 6)])
    # if wk_cols and EMPLOYED_WEEK_CODES:
    #     any_emp = pl.any_horizontal(*[pl.col(c).is_in(list(EMPLOYED_WEEK_CODES)) for c in wk_cols])
    # elif wk_cols:
    #     # conservative default: treat codes {1,2,3,4} as "employed some/any" if that matches your recode doc
    #     any_emp = pl.any_horizontal(*[(pl.col(c) <= 4) for c in wk_cols])
    # else:
    #     any_emp = pl.lit(True)
    any_emp = pl.col("dyswkd").fill_null(0) > 0
    pm = pm.with_columns(any_emp.alias("employed_any_week"))

    # valid telework month = has main job days & in employment universe
    pm = pm.with_columns(((pl.col("dyswkd") > 0) & pl.col("employed_any_week")).alias("valid_month"))

    # weights
    # Weeks-per-month factor (≈ 4.345)
    WEEKS_PER_MONTH = 365.25 / 12 / 7

    pm = pm.with_columns([
        # Cross-sectional person-month weight
        pl.col("w_base").alias("wA"),

        # Worked-mass weight for day-based telework stats:
        # wB = w_base * (days worked per week) * weeks per month
        (
            pl.col("w_base")
            * pl.max_horizontal(pl.col("dyswkd").fill_null(0), pl.lit(0))
            * pl.lit(WEEKS_PER_MONTH)
        ).alias("wB"),

        # OPTIONAL: hour-exposure variant (commented out)
        # (
        #     pl.col("w_base")
        #     * pl.max_horizontal(pl.col("mwhours").fill_null(0), pl.lit(0))
        #     * pl.lit(WEEKS_PER_MONTH)
        # ).alias("wB_hours"),
    ])

# If you want the effective (masked) versions right away:
    pm = pm.with_columns([
        pl.when(pl.col("valid_month")).then(pl.col("wA")).otherwise(0.0).alias("wA_eff"),
        pl.when(pl.col("valid_month")).then(pl.col("wB")).otherwise(0.0).alias("wB_eff"),
        # pl.when(pl.col("valid_month")).then(pl.col("wB_hours")).otherwise(0.0).alias("wB_hours_eff"),  # optional
    ])

    return pm

def person_year(
                pm: pl.DataFrame,
                weight: str = "B",
                episode_keys: tuple[str, ...] = ("occ", "ind"),
                fill_missing_keys: bool = False,
                ) -> pl.DataFrame:
    """
    Collapse SIPP person-months to person–year–episode rows.
    An episode is defined by `episode_keys` (default: occ × ind).
    Weights are exposure-like (your w*_eff), so job changers are partitioned,
    not double-counted.

    First aggregates monthly remote work fractions to annual level, then 
    classifies workers as:
    - Full remote: p_remote_day_year >= (1 - EPSILON)
    - Hybrid: EPSILON < p_remote_day_year < (1 - EPSILON)  
    - Full in-person: p_remote_day_year <= EPSILON

    Returns columns:
        SSUID, PNUM, year, *episode_keys,
        months_obs, months_emp, wsum_pm,
        p_remote_day_year,
        s_full_remote_year, s_hybrid_year, s_inperson_year,
        mean_jobhrs1_year,
        wtype_used
    """
    if weight not in ("A", "B"):
        raise ValueError("weight must be 'A' or 'B'")
    w_eff = f"w{weight}_eff"

    # Guard: ensure valid_month is numeric for sums
    pm = pm.with_columns(pl.col("valid_month").cast(pl.Int64).alias("valid_month_int"))

    # Optionally park missing occ/ind into a catch-all code (prevents mass loss)
    if fill_missing_keys:
        fill_map = {k: -9 for k in episode_keys}
        pm = pm.with_columns(
            [pl.col(k).fill_null(fill_map[k]).alias(k) for k in episode_keys if k in pm.columns]
        )

    group_keys = ["SSUID", "PNUM", "year", *episode_keys]

    # Build numerators/denominator first, then ratios

    year_agg = (
        pm.group_by(group_keys).agg([
                pl.len().alias("months_obs"),
                pl.col("valid_month_int").sum().alias("months_emp"),
                pl.col(w_eff).sum().alias("wsum_pm"),
                (pl.col(w_eff) * pl.col("p_remote")).sum().alias("num_p_remote_day"),
                # Weighted average of mean_jobhrs1
                (pl.col(w_eff) * pl.col("mean_jobhrs1")).sum().alias("num_mean_jobhrs1"),
                # Take first TAGE, EEDUC and ESEX for each person-year
                pl.col("TAGE").first().alias("TAGE"),
                pl.col("ESEX").first().alias("ESEX"),
                pl.col("EEDUC").first().alias("EEDUC"),
                ]).with_columns([
                    # Calculate annual fraction of remote work
                    pl.when(pl.col("wsum_pm") > 0)
                        .then(pl.col("num_p_remote_day") / pl.col("wsum_pm"))
                        .otherwise(None)
                        .alias("p_remote_day_year"),
                    pl.when(pl.col("wsum_pm") > 0)
                        .then(pl.col("num_mean_jobhrs1") / pl.col("wsum_pm"))
                        .otherwise(None)
                        .alias("mean_jobhrs1_year"),
                    pl.lit(weight).alias("wtype_used"),
                ]).with_columns([
                    # Worker type classification based on annual remote fraction
                    pl.when(pl.col("p_remote_day_year") >= (1.0 - EPSILON)).then(1.0)
                        .otherwise(0.0).alias("s_full_remote_year"),
                    pl.when((pl.col("p_remote_day_year") > EPSILON) & 
                            (pl.col("p_remote_day_year") < (1.0 - EPSILON))).then(1.0)
                        .otherwise(0.0).alias("s_hybrid_year"),
                    pl.when(pl.col("p_remote_day_year") <= EPSILON).then(1.0)
                        .otherwise(0.0).alias("s_inperson_year"),
                ]).drop([
                    "num_p_remote_day",
                    "num_mean_jobhrs1",
                    ])
        )

        # Compute yearly stats #! For diagnostics only
    yearly_stats = (
            year_agg.group_by("year")
            .agg([
                pl.len().alias("n_workers"),
                pl.col("wsum_pm").sum().alias("total_weight"),
                (pl.col("wsum_pm") * pl.col("s_full_remote_year")).sum().alias("weight_full_remote"),
                (pl.col("wsum_pm") * pl.col("s_hybrid_year")).sum().alias("weight_hybrid"),
                (pl.col("wsum_pm") * pl.col("s_inperson_year")).sum().alias("weight_inperson"),
                (pl.col("wsum_pm") * pl.col("p_remote_day_year")).sum().alias("weight_p_remote_sum"),
            ])
            .with_columns([
                (pl.col("weight_full_remote") / pl.col("total_weight")).alias("share_full_remote"),
                (pl.col("weight_hybrid") / pl.col("total_weight")).alias("share_hybrid"),
                (pl.col("weight_inperson") / pl.col("total_weight")).alias("share_inperson"),
                (pl.col("weight_p_remote_sum") / pl.col("total_weight")).alias("avg_p_remote"),
            ])
            .select([
                "year", "n_workers", "share_full_remote", "share_hybrid", 
                "share_inperson", "avg_p_remote"
            ])
            .sort("year")
        )

    # Display the results
    print(yearly_stats)
        
    return year_agg


def map_sipp_eeduc_to_cps_educ(df: pl.DataFrame) -> pl.DataFrame:
    """
    Map SIPP EEDUC codes to CPS-compatible EDUC codes and create edu3 categories.
    
    SIPP EEDUC codes:
    31-38: Less than high school
    39: High school graduate 
    40-42: Some college/Associates
    43: Bachelor's degree
    44-46: Advanced degrees (Master's, Professional, Doctorate)
    
    Maps to edu3 categories:
    - "lt_ba": Less than bachelor's (EEDUC 31-42)
    - "ba": Bachelor's degree (EEDUC 43)  
    - "adv": Advanced degrees (EEDUC 44-46)
    """
    if "EEDUC" not in df.columns:
        return df.with_columns(pl.lit(None).alias("edu3"))
    
    # Map SIPP EEDUC to CPS-compatible EDUC codes (roughly)
    # SIPP 31-38 (< HS) -> CPS ~002-073 (< HS)
    # SIPP 39 (HS grad) -> CPS ~073 (HS grad)
    # SIPP 40-41 (some college) -> CPS ~081-110 (some college)
    # SIPP 42 (Associates) -> CPS ~111 (Associates)
    # SIPP 43 (Bachelor's) -> CPS ~111 (Bachelor's)
    # SIPP 44-46 (Advanced) -> CPS ~123+ (Advanced)
    
    df = df.with_columns([
        # Create CPS-compatible EDUC code
        pl.when(pl.col("EEDUC") < 39).then(pl.lit(73))  # Less than HS -> HS level
        .when(pl.col("EEDUC") == 39).then(pl.lit(73))   # HS graduate
        .when(pl.col("EEDUC").is_in([40, 41])).then(pl.lit(110))  # Some college
        .when(pl.col("EEDUC") == 42).then(pl.lit(111))  # Associates
        .when(pl.col("EEDUC") == 43).then(pl.lit(111))  # Bachelor's
        .when(pl.col("EEDUC").is_in([44, 45, 46])).then(pl.lit(125))  # Advanced
        .otherwise(pl.col("EEDUC"))  # Keep original if outside expected range
        .alias("EDUC"),
        
        # Create edu3 categories
        pl.when(pl.col("EEDUC") < 43).then(pl.lit("lt_ba"))     # Less than bachelor's
        .when(pl.col("EEDUC") == 43).then(pl.lit("ba"))         # Bachelor's
        .when(pl.col("EEDUC").is_in([44, 45, 46])).then(pl.lit("adv"))  # Advanced
        .otherwise(pl.lit(None))
        .alias("edu3")
    ])
    
    return df


def _ensure_cell_components(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ensures the presence of key derived columns in a Polars DataFrame based on existing raw columns.
    This function checks for the existence of several harmonized or derived columns commonly used 
    in labor force and demographic analysis. If a derived column is missing but its source column 
    is present, the function computes and adds the derived column to the DataFrame.
    
    For SIPP data, maps EEDUC to CPS-compatible education categories.
    """
    # Map SIPP education codes to CPS-compatible format
    if "EEDUC" in df.columns and "edu3" not in df.columns:
        df = map_sipp_eeduc_to_cps_educ(df)
    
    # Map ESEX to sex (SIPP uses 1=male, 2=female like CPS)
    if "sex" not in df.columns and "ESEX" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("ESEX") == 1).then(pl.lit("male"))
            .when(pl.col("ESEX") == 2).then(pl.lit("female"))
            .otherwise(pl.lit(None))
            .alias("sex")
        )
    
    # Map TAGE to age4 categories  
    if "age4" not in df.columns and "TAGE" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("TAGE") < 25).then(None)
            .when(pl.col("TAGE") <= 34).then(pl.lit("25-34"))
            .when(pl.col("TAGE") <= 44).then(pl.lit("35-44"))
            .when(pl.col("TAGE") <= 54).then(pl.lit("45-54"))
            .when(pl.col("TAGE") <= 64).then(pl.lit("55-64"))
            .otherwise(None)
            .alias("age4")
        )
    
    return df


def check_weight_conservation(
    pm: pl.DataFrame,
    py_ep: pl.DataFrame,
    weight: str = "B",
    episode_keys: tuple[str, ...] = ("occ", "ind"),
    atol: float = 1e-6,
) -> pl.DataFrame:
    """
    Verifies that the sum of episode weights equals the person–year mass.

    Returns a small DataFrame of any person–years that fail the check.
    """
    w_eff = f"w{weight}_eff"
    base = (
        pm.group_by(["SSUID", "PNUM", "year"])
          .agg(pl.col(w_eff).sum().alias("wsum_pm_total"))
    )
    ep = (
        py_ep.group_by(["SSUID", "PNUM", "year"])
             .agg(pl.col("wsum_pm").sum().alias("wsum_pm_ep"))
    )
    chk = (
        base.join(ep, on=["SSUID", "PNUM", "year"], how="left")
            .with_columns((pl.col("wsum_pm_total") - pl.col("wsum_pm_ep")).alias("diff"))
            .filter(pl.col("diff").abs() > atol)
    )
    return chk


def qc_person_year(py: pl.DataFrame, atol: float = 1e-6) -> pl.DataFrame:
    """
    Return rows with potential issues:
    - p_remote_day_year outside [0, 1]
    - component shares sum > 1 + atol (should be exactly 1)
    - months_emp > months_obs
    """
    tmp = (
        py.with_columns(
            (
                pl.col("s_full_remote_year").fill_null(0.0)
                + pl.col("s_hybrid_year").fill_null(0.0)
                + pl.col("s_inperson_year").fill_null(0.0)
            ).alias("__sum_s")
        )
        .with_columns([
            ((pl.col("p_remote_day_year") < 0) | (pl.col("p_remote_day_year") > 1)).alias("__bad_p"),
            ((pl.col("__sum_s") - 1.0).abs() > atol).alias("__bad_sum"),  # Should sum to exactly 1
            (pl.col("months_emp") > pl.col("months_obs")).alias("__bad_months"),
        ])
    )
    return (
        tmp.filter(pl.col("__bad_p") | pl.col("__bad_sum") | pl.col("__bad_months"))
           .drop(["__sum_s", "__bad_p", "__bad_sum", "__bad_months"])
    )


def _expand_i_vars(patterns: List[str]) -> List[str]:
    out, seen = [], set()
    for p in patterns:
        if "{i}" in p:
            for i in range(1, JOB_MAX + 1):
                v = p.replace("{i}", str(i))
                if v not in seen:
                    out.append(v); seen.add(v)
        else:
            if p not in seen:
                out.append(p); seen.add(p)
    return out


def _load_paths_from_yaml(cfg_path: Path) -> tuple[Path, Path]:
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    paths = raw.get("paths", {}) or {}
    repo_root = Path(__file__).resolve().parents[2]
    def to_abs(p: str | None, default: str) -> Path:
        if not p:
            p = default
        return (repo_root / p).resolve() if not str(p).startswith("/") else Path(p).resolve()
    return to_abs(paths.get("raw_dir"), "data/raw"), to_abs(paths.get("processed_dir"), "data/processed")


def _find_sipp_raw_file(raw_dir: Path) -> Path:
    """Prefer combined file sipp_{start}_{end}.csv.gz; fallback to latest per-year."""
    sdir = raw_dir / "sipp"
    pats = [
        str(sdir / "sipp_*_*.csv.gz"),  # combined
        str(sdir / "sipp_*.csv.gz"),    # per-year
    ]
    candidates = []
    for pat in pats:
        candidates.extend(sorted(glob.glob(pat)))
        if candidates:
            break
    if not candidates:
        raise FileNotFoundError(f"No SIPP raw CSV.gz found under {sdir}")
    # Choose the largest (most complete)
    candidates = sorted(candidates, key=lambda p: Path(p).stat().st_size, reverse=True)
    return Path(candidates[0])


def _resolve_tokens(s: str, cfg: dict) -> str:
    # Replace ${a.b.c} with cfg["a"]["b"]["c"]
    def get_path(d, path):
        cur = d
        for k in path.split("."):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return None
        return cur
    def repl(m: re.Match) -> str:
        keypath = m.group(1)
        val = get_path(cfg, keypath)
        return str(val) if val is not None else m.group(0)
    return re.sub(r"\$\{([^}]+)\}", repl, s)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    log = logging.getLogger("sipp_process")

    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / "src" / "data" / "config.yml"
    raw_dir, processed_dir = _load_paths_from_yaml(cfg_path)

    in_path = _find_sipp_raw_file(raw_dir)
    out_dir = (processed_dir / "sipp").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Columns to read (minimize IO), fall back to auto if missing
    base_cols = [
                    "SPANEL", "SSUID", "PNUM", # Person Identifiers
                    CAL_YEAR, # Calendar Year
                    MAIN_JOB_COL, # Main Job Column
                    WEIGHT_COL, # Weight Column
                    "TAGE", "ESEX", "EEDUC" # Demographics
                    ]
    job_patterns = [
        "EJB{i}_DYSWKD", "EJB{i}_DYSWKDH",
        "TJB{i}_MWKHRS", "TJB{i}_OCC", "TJB{i}_IND",
        # optional:
        "EJB{i}_CLWRK", "EJB{i}_JBORSE",
    ]
    weekly_cols = [f"RWKESR{k}" for k in range(1, 6)]

    usecols = set(base_cols + weekly_cols + _expand_i_vars(job_patterns))

    log.info(f"Loading SIPP raw: {in_path.name}")
    # Read header first to check available columns
    # Read only the header row using Polars
    header = pl.read_csv(str(in_path), n_rows=0).columns
    missing_cols = [c for c in usecols if c not in header]
    if missing_cols:
        log.warning(f"Missing columns in input file and will be skipped: {missing_cols}")

    # Only read columns that exist in the file
    read_cols = [c for c in usecols if c in header]
    df = pl.read_csv(
        str(in_path),
        columns=read_cols,
        ignore_errors=True,
    )

    # Warn on critical columns
    must_have = {"SPANEL", "SSUID", "PNUM", CAL_YEAR, WEIGHT_COL}
    missing_crit = [c for c in must_have if c not in df.columns]
    if missing_crit:
        raise RuntimeError(f"Missing required columns: {missing_crit}. Update SIPP variables in config and re-run acquisition.")

    pm = build_person_month(df)
    log.info(f"Person-month rows: {pm.height:,}")

    # Add CPS-compatible education mapping to person-month data
    pm = _ensure_cell_components(pm)
    log.info("Added CPS-compatible education categories to person-month data")

    pyA = person_year(pm, weight="A")
    pyB = person_year(pm, weight="B")
    log.info(f"Person-year (A) rows: {pyA.height:,}; (B) rows: {pyB.height:,}")
    
    # Add CPS-compatible education mapping and other cell components
    pyA = _ensure_cell_components(pyA)
    pyB = _ensure_cell_components(pyB)
    log.info("Added CPS-compatible education categories and demographic mappings")
    
    # Check weight conservation
    bad_A = check_weight_conservation(pm, pyA, weight="A")
    if bad_A.height:
        print("WARNING: weight conservation failures (A):\n", bad_A.head(10))

    bad_B = check_weight_conservation(pm, pyB, weight="B")
    if bad_B.height:
        print("WARNING: weight conservation failures (B):\n", bad_B.head(10))

    # Quick QC
    qcA = qc_person_year(pyA)
    qcB = qc_person_year(pyB)
    if qcA.height or qcB.height:
        log.warning(f"QC: Found {qcA.height + qcB.height} rows with potential share issues (showing up to 20 per set).")

    # Save
    pm_path = out_dir / "sipp_pm.csv.gz"
    pyA_path = out_dir / "sipp_py_A.csv.gz"
    pyB_path = out_dir / "sipp_py_B.csv.gz"

    with gzip.open(pm_path, "wb") as f:
        pm.write_csv(f)
    with gzip.open(pyA_path, "wb") as f:
        pyA.write_csv(f)
    with gzip.open(pyB_path, "wb") as f:
        pyB.write_csv(f)

    log.info(f"Saved: {pm_path}")
    log.info(f"Saved: {pyA_path}")
    log.info(f"Saved: {pyB_path}")

    # Preestimation moved to preestimation_from_sipp.py
    return 0


if __name__ == "__main__":
    sys.exit(main())