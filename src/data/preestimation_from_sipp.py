from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import polars as pl
from dataclasses import dataclass

# Reuse your existing tools
try:
    # allow running as module within src/data
    from .ipums_process import (
        reweight_poststrat,
        add_topcode_and_real_wages,
        _winsorize_by_year,
        _ensure_cell_components,   # <-- needed by SIPP aggregation
        _derive_cell_components,
        _export_stata,
    )
except Exception:
    # allow running as script: put repo_root/src on sys.path and import data.ipums_process
    import sys as _sys
    from pathlib import Path as _Path
    _repo_root = _Path(__file__).resolve().parents[2]
    _src_path = str(_repo_root / "src")
    if _src_path not in _sys.path:
        _sys.path.insert(0, _src_path)
    from data.ipums_process import (
        reweight_poststrat,
        add_topcode_and_real_wages,
        _winsorize_by_year,
        _ensure_cell_components,   # <-- needed by SIPP aggregation
        _derive_cell_components,
        _export_stata,
    )

logger = logging.getLogger("preestimation_from_sipp")

# ----------------------- SIPP→CPS Bridge (bespoke) -----------------------

_EPS = 1e-8
_MAE_EPS = 1e-4  # for WMAPE denominator

@dataclass
class BridgeFitResult:
    lambdas: pl.DataFrame          # [POOL_KEY..., YEAR, lambda_remote, lambda_hybrid, lambda_inperson]
    bridged_cells: pl.DataFrame    # [YEAR, cell_id, pR_hat, pH_hat, pI_hat, pR_cps, pH_cps, pI_cps, W_tot]
    metrics: Dict[str, float]      # WMAE/WMAPE/WRMSE + correlations


def _make_cps_cell_targets(cps_mi: pl.DataFrame) -> pl.DataFrame:
    """Build CPS targets per (YEAR, cell_id): weighted shares for R/H/I."""
    need = {"YEAR","cell_id","cps_weight","FULL_REMOTE","HYBRID","FULL_INPERSON"}
    missing = [c for c in need if c not in cps_mi.columns]
    if missing:
        raise RuntimeError(f"CPS micro missing: {missing}")

    t = (cps_mi
         .with_columns(pl.col("cps_weight").cast(pl.Float64).alias("W"))
         .group_by(["YEAR","cell_id"])
         .agg([
             pl.col("W").sum().alias("W_tot"),
             (pl.col("W") * pl.col("FULL_REMOTE").cast(pl.Float64)).sum().alias("num_R"),
             (pl.col("W") * pl.col("HYBRID").cast(pl.Float64)).sum().alias("num_H"),
             (pl.col("W") * pl.col("FULL_INPERSON").cast(pl.Float64)).sum().alias("num_I"),
         ])
         .with_columns([
             (pl.col("num_R") / (pl.col("W_tot")+_EPS)).alias("pR_cps"),
             (pl.col("num_H") / (pl.col("W_tot")+_EPS)).alias("pH_cps"),
             (pl.col("num_I") / (pl.col("W_tot")+_EPS)).alias("pI_cps"),
         ])
         .select(["YEAR","cell_id","W_tot","pR_cps","pH_cps","pI_cps"])
    )
    return t


def _prepare_bridge_design(sipp_cell: pl.DataFrame, cps_cell: pl.DataFrame) -> pl.DataFrame:
    """Inner-join SIPP cell shares with CPS targets; keep positive weights."""
    need = {"YEAR","cell_id","fraction_full_remote","fraction_hybrid","fraction_full_inperson"}
    missing = [c for c in need if c not in sipp_cell.columns]
    if missing:
        raise RuntimeError(f"SIPP cells missing: {missing}")

    s = (sipp_cell
         .select([
             "YEAR","cell_id",
             pl.col("fraction_full_remote").cast(pl.Float64).alias("sR"),
             pl.col("fraction_hybrid").cast(pl.Float64).alias("sH"),
             pl.col("fraction_full_inperson").cast(pl.Float64).alias("sI"),
         ])
    )
    df = (s.join(cps_cell, on=["YEAR","cell_id"], how="inner")
            .filter(pl.col("W_tot") > 0)
            .with_columns((pl.col("sR")+pl.col("sH")+pl.col("sI")).alias("__s_sum"))
            .with_columns([
                (pl.col("sR")/pl.col("__s_sum")).alias("sR"),
                (pl.col("sH")/pl.col("__s_sum")).alias("sH"),
                (pl.col("sI")/pl.col("__s_sum")).alias("sI"),
            ])
            .drop("__s_sum")
    )
    return df


def _fit_lambdas_by_pool(df: pl.DataFrame, pool_keys: list[str]) -> pl.DataFrame:
    """Fit log-lambdas per pool (YEAR by default); θI=0 for ID; minimize weighted SSE."""
    def _fit_one(pdf: pl.DataFrame) -> Dict[str, float]:
        import numpy as np
        W  = pdf["W_tot"].to_numpy(dtype=float)
        sR = pdf["sR"].to_numpy(dtype=float); sH = pdf["sH"].to_numpy(dtype=float); sI = pdf["sI"].to_numpy(dtype=float)
        yR = pdf["pR_cps"].to_numpy(dtype=float); yH = pdf["pH_cps"].to_numpy(dtype=float); yI = pdf["pI_cps"].to_numpy(dtype=float)

        theta = np.array([0.0, 0.0])  # [θR, θH]; θI=0

        def _pred(th):
            aR = np.log(sR + _EPS) + th[0]
            aH = np.log(sH + _EPS) + th[1]
            aI = np.log(sI + _EPS) + 0.0
            m = np.maximum.reduce([aR,aH,aI])
            eR = np.exp(aR - m); eH = np.exp(aH - m); eI = np.exp(aI - m)
            Z = eR + eH + eI + _EPS
            return eR/Z, eH/Z, eI/Z

        lr = 0.5
        for _ in range(400):
            pR, pH, pI = _pred(theta)
            eR = (pR - yR); eH = (pH - yH)
            gR = (W * (2*eR * (pR*(1-pR)) + 2*eH * (-pR*pH))).sum()
            gH = (W * (2*eH * (pH*(1-pH)) + 2*eR * (-pR*pH))).sum()
            g = np.array([gR, gH])

            def loss(th):
                R,H,I = _pred(th)
                return float((W * ((R-yR)**2 + (H-yH)**2 + (I-yI)**2)).sum())

            theta_new = theta - lr * g
            if loss(theta_new) > loss(theta):
                lr *= 0.5
            else:
                theta = theta_new
                lr = min(lr*1.05, 1.0)
            if np.linalg.norm(g) < 1e-7:
                break

        return {"lambda_remote": float(np.exp(theta[0])),
                "lambda_hybrid": float(np.exp(theta[1])),
                "lambda_inperson": 1.0}

    out = []
    for key_vals, g in df.group_by(pool_keys, maintain_order=True):
        row = {}
        if isinstance(key_vals, tuple):
            for i,k in enumerate(pool_keys):
                row[k] = key_vals[i]
        else:
            row[pool_keys[0]] = key_vals
        row.update(_fit_one(g))
        out.append(row)
    return pl.DataFrame(out)


def _apply_lambdas(df: pl.DataFrame, lambdas: pl.DataFrame, pool_keys: list[str]) -> pl.DataFrame:
    z = (df.join(lambdas, on=pool_keys, how="left")
           .with_columns([
               (pl.col("lambda_remote")   * pl.col("sR")).alias("__nR"),
               (pl.col("lambda_hybrid")   * pl.col("sH")).alias("__nH"),
               (pl.col("lambda_inperson") * pl.col("sI")).alias("__nI"),
           ])
           .with_columns((pl.col("__nR")+pl.col("__nH")+pl.col("__nI")).alias("__den"))
           .with_columns([
               (pl.col("__nR")/(pl.col("__den")+_EPS)).alias("pR_hat"),
               (pl.col("__nH")/(pl.col("__den")+_EPS)).alias("pH_hat"),
               (pl.col("__nI")/(pl.col("__den")+_EPS)).alias("pI_hat"),
           ])
           .drop(["__nR","__nH","__nI","__den"])
    )
    return z


def _diagnose_fit(df_hat: pl.DataFrame) -> Dict[str, float]:
    W = df_hat["W_tot"].to_numpy()
    pR = df_hat["pR_hat"].to_numpy(); pH = df_hat["pH_hat"].to_numpy(); pI = df_hat["pI_hat"].to_numpy()
    yR = df_hat["pR_cps"].to_numpy(); yH = df_hat["pH_cps"].to_numpy(); yI = df_hat["pI_cps"].to_numpy()
    import numpy as np

    def _wstats(pred, true):
        ae  = np.abs(pred-true)
        se  = (pred-true)**2
        wmae  = float((W*ae).sum() / (W.sum()+_EPS))
        wmape = float((W*(ae/np.maximum(true,_MAE_EPS))).sum() / (W.sum()+_EPS))
        wrmse = float(np.sqrt((W*se).sum() / (W.sum()+_EPS)))
        def _wcorr(a,b):
            wa = (W*a).sum()/W.sum(); wb = (W*b).sum()/W.sum()
            ac = a-wa; bc = b-wb
            va = (W*(ac**2)).sum()/W.sum(); vb = (W*(bc**2)).sum()/W.sum()
            if va < 1e-12 or vb < 1e-12: return float("nan")
            cov = (W*(ac*bc)).sum()/W.sum()
            return float(cov/np.sqrt(va*vb))
        return wmae, wmape, wrmse, _wcorr(pred,true)

    mR = _wstats(pR,yR); mH = _wstats(pH,yH); mI = _wstats(pI,yI)
    return {
        "WMAE_remote": mR[0], "WMAPE_remote": mR[1], "WRMSE_remote": mR[2], "corr_remote": mR[3],
        "WMAE_hybrid": mH[0], "WMAPE_hybrid": mH[1], "WRMSE_hybrid": mH[2], "corr_hybrid": mH[3],
        "WMAE_inperson": mI[0], "WMAPE_inperson": mI[1], "WRMSE_inperson": mI[2], "corr_inperson": mI[3],
    }


def build_sipp_cps_bridge(
    cps_mi: pl.DataFrame,
    sipp_cell: pl.DataFrame,
    pool_keys: list[str] = ["YEAR"],
) -> BridgeFitResult:
    cps_cell = _make_cps_cell_targets(cps_mi)
    df = _prepare_bridge_design(sipp_cell, cps_cell)
    lambdas = _fit_lambdas_by_pool(df, pool_keys)
    bridged = _apply_lambdas(df, lambdas, pool_keys)
    metrics = _diagnose_fit(bridged)
    return BridgeFitResult(lambdas=lambdas, bridged_cells=bridged, metrics=metrics)

# ----------------------- CPS prep -----------------------

def _ensure_cps_weights_and_vars(cps: pl.DataFrame) -> pl.DataFrame:
    # Age/hours filters
    for c in ["AGE", "UHRSWORKT"]:
        if c not in cps.columns:
            raise RuntimeError(f"CPS missing required column {c}")
    logger.info(f"CPS: Before age/hours filters: {len(cps):,} rows")
    cps = cps.filter((pl.col("AGE") >= 25) & (pl.col("AGE") <= 64))
    cps = cps.filter((pl.col("UHRSWORKT") >= 20) & (pl.col("UHRSWORKT") <= 84))
    logger.info(f"CPS: After age/hours filters: {len(cps):,} rows")

    # Post-stratification of WTFINL to wage-respondents
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
        target_univ, wage_present = reweight_poststrat(
            cps.select([*(set(strat_cols)|{"WTFINL","WAGE","YEAR"})]),
            weight_col="WTFINL",
            response_col="WAGE",
            min_wage=0,
            cell_vars=strat_cols
        )
        cps = cps.with_row_index(name="__rid")
        wage_present = wage_present.with_row_index(name="__rid")
        cps = (cps.join(wage_present.select(["__rid", "WTFINL_ADJ"]), on="__rid", how="left")
                  .drop("__rid")
                  .with_columns(pl.coalesce([pl.col("WTFINL_ADJ"), pl.col("WTFINL")]).alias("cps_weight"))
                  .drop(["WTFINL_ADJ"], strict=False))
    elif "WTFINL" in cps.columns:
        cps = cps.with_columns(pl.col("WTFINL").alias("cps_weight"))
    else:
        cps = cps.with_columns(pl.lit(1.0).alias("cps_weight"))

    # Real wages & winsorization
    has_real = ("WAGE_REAL" in cps.columns) and ("LOG_WAGE_REAL" in cps.columns)
    if not has_real:
        cps = add_topcode_and_real_wages(cps)
    cps = cps.with_columns(pl.col("LOG_WAGE_REAL").alias("logw"))
    cps = _winsorize_by_year(cps, "logw", 0.01, 0.99)

    # Cells & psi
    if "cell_id" not in cps.columns:
        logger.warning("CPS: cell_id missing; deriving via _derive_cell_components()")
        cps = _derive_cell_components(cps)

    if "psi" not in cps.columns:
        if "TELEWORKABLE_OCSSOC_DETAILED" in cps.columns:
            cps = cps.with_columns(pl.col("TELEWORKABLE_OCSSOC_DETAILED").cast(pl.Float64).alias("psi"))
        else:
            cps = cps.with_columns(pl.lit(0.5).alias("psi"))

    # Observed telework ALPHA + flags (2022+)
    if "ALPHA" not in cps.columns and {"TELWRKHR","UHRSWORKT"}.issubset(cps.columns):
        cps = cps.with_columns((pl.col("TELWRKHR") / pl.col("UHRSWORKT")).alias("ALPHA"))
    thr_hi, thr_lo = 0.995, 0.005
    flags = {
        "FULL_REMOTE": (pl.col("ALPHA") >= thr_hi),
        "FULL_INPERSON": (pl.col("ALPHA") <= thr_lo),
        "HYBRID": (pl.col("ALPHA") > thr_lo) & (pl.col("ALPHA") < thr_hi),
    }
    for d, expr in flags.items():
        if d not in cps.columns:
            cps = cps.with_columns(expr.cast(pl.Int8).alias(d))

    return cps

# ----------------------- SIPP cell measures -----------------------

def _sipp_cell_measures_from_year_episodes(
    sipp_py_ep: pl.DataFrame,
    weight_col: str = "wsum_pm",   # from person_year()
    year_col: str = "year",
) -> pl.DataFrame:
    """
    Aggregate SIPP person–year–episode rows to cell×year measures.
    Requires: year, occ/ind, wsum_pm, p_remote_day_year, s_*_year.
    """
    # Make names CPS-like for cell derivation
    ren = {}
    if year_col in sipp_py_ep.columns and "YEAR" not in sipp_py_ep.columns:
        ren[year_col] = "YEAR"
    if "occ" in sipp_py_ep.columns and "OCC" not in sipp_py_ep.columns:
        ren["occ"] = "OCC"
    if "ind" in sipp_py_ep.columns and "INDNAICS" not in sipp_py_ep.columns:
        ren["ind"] = "INDNAICS"
    if "ESEX" in sipp_py_ep.columns and "SEX" not in sipp_py_ep.columns:
        ren["ESEX"] = "SEX"
    if "EEDUC" in sipp_py_ep.columns and "EDUC" not in sipp_py_ep.columns:
        ren["EEDUC"] = "EDUC"
    if "TAGE" in sipp_py_ep.columns and "AGE" not in sipp_py_ep.columns:
        ren["TAGE"] = "AGE"

    sipp = sipp_py_ep.rename(ren)

    # Ensure cell components & cell_id
    sipp = _ensure_cell_components(sipp)
    if "cell_id" not in sipp.columns:
        sipp = _derive_cell_components(sipp)

    # Guards
    need = {
        "YEAR","cell_id", weight_col,
        "p_remote_day_year",
        "s_full_remote_year","s_hybrid_year","s_inperson_year",
    }
    miss = [c for c in need if c not in sipp.columns]
    if miss:
        raise RuntimeError(f"SIPP year-episodes missing required columns: {miss}")

    # Unified weight & defensive share clean-up
    sipp = (sipp
            .with_columns([
                pl.col(weight_col).cast(pl.Float64).alias("W"),
                (pl.col("s_full_remote_year")+pl.col("s_hybrid_year")+pl.col("s_inperson_year")).alias("__S"),
            ])
            .with_columns([
                (pl.col("s_full_remote_year")/pl.when(pl.col("__S")>0).then(pl.col("__S")).otherwise(1.0)).alias("sR"),
                (pl.col("s_hybrid_year")     /pl.when(pl.col("__S")>0).then(pl.col("__S")).otherwise(1.0)).alias("sH"),
                (pl.col("s_inperson_year")   /pl.when(pl.col("__S")>0).then(pl.col("__S")).otherwise(1.0)).alias("sI"),
            ])
            .drop(["__S"])
    )

    def wmean(expr: pl.Expr) -> pl.Expr:
        return (expr * pl.col("W")).sum() / pl.col("W").sum()

    cell = (
        sipp.group_by(["YEAR","cell_id"])
            .agg([
                pl.len().alias("N_ep"),
                pl.col("W").sum().alias("W_tot"),
                (pl.col("W").pow(2).sum()).alias("__W2"),
                wmean(pl.col("sR")).alias("fraction_full_remote"),
                wmean(pl.col("sH")).alias("fraction_hybrid"),
                wmean(pl.col("sI")).alias("fraction_full_inperson"),
                wmean(pl.col("p_remote_day_year")).alias("avg_fraction_remote_days"),
            ])
            .with_columns([
                (pl.col("fraction_full_remote")+pl.col("fraction_hybrid")).alias("p_any_remote"),
                pl.when(pl.col("__W2")>0)
                  .then((pl.col("W_tot")**2)/pl.col("__W2"))
                  .otherwise(None)
                  .alias("n_eff"),
            ])
            .drop(["__W2"])
    )
    return cell

# ----------------------- Driver -----------------------

def preestimation_from_sipp_year_avgs(
    cfg,
    sipp_py_path: Path | None = None,
    random_seed: int = 123,
    write_schema: bool = True,
    produce_bridge: bool = True,
) -> Dict[str, Path]:
    """
    Build Stata-ready artifacts using SIPP cell×year averages instead of ATUS.
    Exports:
        - cps_mi_ready.dta
        - cps_observed_telework.dta (YEAR>=2022)
        - sipp_cell_measures.dta
        - bridge_lambda.dta (optional; bespoke SIPP→CPS)
    """
    np.random.seed(random_seed)

    # ---- 1) Load Inputs ----
    cps_path = cfg.get_output_path("cps", (cfg.processed_dir / "cps" / "cps_processed.csv").resolve())
    if not cps_path.exists():
        raise FileNotFoundError(f"Missing CPS processed: {cps_path}")

    if sipp_py_path is None:
        sipp_py_path = (cfg.processed_dir / "sipp" / "sipp_py_B.csv.gz").resolve()
    if not sipp_py_path.exists():
        raise FileNotFoundError(f"Missing SIPP person–year episodes: {sipp_py_path}")

    cps = pl.read_csv(str(cps_path), schema_overrides={"INDNAICS": pl.String})
    # SIPP person-year robust read: ensure 'ind' and 'occ' are strings if present
    try:
        sipp_py = pl.read_csv(
            str(sipp_py_path),
            infer_schema_length=10000,
            schema_overrides={"ind": pl.String, "occ": pl.String},
        )
    except Exception:
        hdr = pl.read_csv(str(sipp_py_path), n_rows=0).columns
        overrides = {k: pl.String for k in ("ind", "occ") if k in hdr}
        sipp_py = pl.read_csv(
            str(sipp_py_path),
            infer_schema_length=10000,
            schema_overrides=overrides if overrides else None,
        )

    # ---- 2) CPS processing ----
    cps = _ensure_cps_weights_and_vars(cps)

    # ---- 3) SIPP → cell×year measures ----
    sipp_cell = _sipp_cell_measures_from_year_episodes(sipp_py)

    # ---- 4) Join SIPP measures into CPS micro and compute RH/IP weights ----
    keep_cols = [
        "YEAR", "MONTH", "cps_weight", "logw", "WAGE", "UHRSWORKT",
        "occ2_harmonized", "ftpt", "edu3", "sex",
        "cell_id", "psi",
        "ALPHA", "FULL_REMOTE", "HYBRID", "FULL_INPERSON",
    ]
    cps_mi = cps.select([c for c in keep_cols if c in cps.columns])

    cps_mi = cps_mi.join(
        sipp_cell.select([
            "YEAR",
            "cell_id",
            "p_any_remote",
            "avg_fraction_remote_days",
            "fraction_full_remote",
            "fraction_hybrid",
            "fraction_full_inperson",
        ]),
        on=["YEAR", "cell_id"],
        how="left",
    )

    cps_mi = cps_mi.with_columns([
        (pl.col("cps_weight") * pl.col("p_any_remote").fill_null(0.0)).alias("w_RH"),
        (pl.col("cps_weight") * (1 - pl.col("p_any_remote").fill_null(0.0))).alias("w_IP"),
    ])

    # Observed telework subset (for 2022+ validation or bridge targets)
    cps_obs = cps_mi.filter(pl.col("YEAR") >= 2022)

    # ---- 5) Bridge lambdas (bespoke SIPP→CPS) ----
    bridge = None
    if produce_bridge:
        fit = build_sipp_cps_bridge(
            cps_mi=cps_mi.select([
                "YEAR","cell_id","cps_weight","FULL_REMOTE","HYBRID","FULL_INPERSON"
            ]),
            sipp_cell=sipp_cell.select([
                "YEAR","cell_id","fraction_full_remote","fraction_hybrid","fraction_full_inperson"
            ]),
            pool_keys=["YEAR"],
        )
        bridge = fit.lambdas.with_columns([pl.lit("bespoke").alias("bridge_kind")])
        logger.info(f"Bridge metrics: {fit.metrics}")

    # ---- 6) Export ----
    out_dir = (cfg.processed_dir / "empirical").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: Dict[str, Path] = {
        "cps_mi_ready": out_dir / "cps_mi_ready.dta",
        "cps_observed_telework": out_dir / "cps_observed_telework.dta",
        "sipp_cell_measures": out_dir / "sipp_cell_measures.dta",
    }
    if bridge is not None:
        out_paths["bridge_lambda"] = out_dir / "bridge_lambda.dta"

    _export_stata(cps_mi, out_paths["cps_mi_ready"])
    _export_stata(cps_obs, out_paths["cps_observed_telework"])
    _export_stata(sipp_cell, out_paths["sipp_cell_measures"])
    if bridge is not None:
        _export_stata(bridge, out_paths["bridge_lambda"])

    if write_schema:
        # (optional) write dicts/schemas alongside .dta the same way you currently do
        pass

    # ---- 7) Sanity logs ----
    try:
        pre = cps_mi.filter(pl.col("YEAR") < 2022)
        denom = pre.select(pl.col("cps_weight").sum()).item()
        if denom and denom > 0:
            numer = pre.filter(pl.col("cell_id").is_not_null() & pl.col("p_any_remote").is_not_null()) \
                       .select(pl.col("cps_weight").sum()).item()
            share = (numer / denom) if numer else 0.0
            logger.info(f"Sanity: pre-2022 weight share with non-missing cell_id & p_any_remote (from SIPP) = {share:.3%}")
        else:
            logger.info("Sanity: pre-2022 coverage unavailable (no relevant data)")
    except Exception as e:
        logger.warning(f"Sanity checks failed: {e}")

    logger.info("Pre-estimation artifacts (SIPP-based) written:")
    for k, p in out_paths.items():
        logger.info(f" - {k}: {p}")

    return out_paths


if __name__ == "__main__":
    import argparse
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Run SIPP-based preestimation pipeline.")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (YAML, optional)")
    parser.add_argument("--sipp-py", type=str, default=None, help="Path to SIPP person-year episodes (optional)")
    parser.add_argument("--no-bridge", action="store_true", help="Disable bridge computation/diagnostics")
    parser.add_argument("--random-seed", type=int, default=123, help="Random seed")
    args = parser.parse_args()

    # Load config (minimal shim if not provided)
    repo_root = Path(__file__).resolve().parents[2]
    processed_dir = repo_root / "data" / "processed"
    cfg = None
    if args.config:
        try:
            sys.path.insert(0, str(repo_root / "src"))
            from data.ipums_process import load_config  # type: ignore
            cfg = load_config(args.config)
        except Exception as e:
            print(f"Failed to load config from {args.config}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        class CfgShim:
            def __init__(self, processed_dir: Path):
                self.processed_dir = processed_dir
            def get_output_path(self, key, default):
                return default
        cfg = CfgShim(processed_dir.resolve())

    sipp_py_path = Path(args.sipp_py) if args.sipp_py else None
    out = preestimation_from_sipp_year_avgs(
        cfg,
        sipp_py_path=sipp_py_path,
        random_seed=args.random_seed,
        produce_bridge=not args.no_bridge,
    )
    print("Preestimation outputs:")
    for k, v in out.items():
        print(f"  {k}: {v}")
