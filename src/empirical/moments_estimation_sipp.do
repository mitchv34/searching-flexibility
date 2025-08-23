*******************************************************
* Moments pipeline: CPS (post-2022 observed) + SIPP→CPS (pre & post imputed)
* Requires:
*   - data/processed/empirical/cps_mi_ready.dta
*   - data/processed/empirical/bridge_lambda.dta
*   - data/processed/empirical/sipp_cell_measures.dta
* Writes:
*   - moments_post2022_observed_by_year_sipp.csv
*   - moments_post2022_imputed_by_year_sipp.csv
*   - moments_pre2022_by_year_sipp.csv
*   - moments_combined_pre_post_sipp.csv
*   - moments_post2022_obs_vs_imp_sipp.csv
*   - moments_validation_sipp_vs_atus.csv
*******************************************************

version 17.0
clear all
set more off

* ---------- Logging ----------
local script_dir "/project/high_tech_ind/searching-flexibility/src/empirical"
local logfile "`script_dir'/moments_estimation_sipp.log"
capture log close _all
log using "`logfile'", replace text
display as text "=== MOMENTS ESTIMATION PIPELINE (SIPP-BASED) - LOG STARTED ==="
display as text "Log file: `logfile'"
display as text "Start time: " as result c(current_time) " " c(current_date)

* ---------- Paths ----------
local root "/project/high_tech_ind/searching-flexibility"
display as text "=== SETTING UP PATHS ==="
display as text "Project root: " as result "`root'"

local cps_dta      "`root'/data/processed/empirical/cps_mi_ready.dta"
local bridge_dta   "`root'/data/processed/empirical/bridge_lambda.dta"
local sipp_dta     "`root'/data/processed/empirical/sipp_cell_measures.dta"

local outdir       "`root'/data/processed/empirical"
local out_post_obs "`outdir'/moments_post2022_observed_by_year_sipp.csv"
local out_post_imp "`outdir'/moments_post2022_imputed_by_year_sipp.csv"
local out_pre      "`outdir'/moments_pre2022_by_year_sipp.csv"
cap mkdir "`outdir'"

display as text "=== FILE PATHS SUMMARY ==="
display as text "CPS:     " as result "`cps_dta'"
display as text "Bridge:  " as result "`bridge_dta'"
display as text "SIPP:    " as result "`sipp_dta'"
display as text "Out post observed: " as result "`out_post_obs'"
display as text "Out post imputed:  " as result "`out_post_imp'"
display as text "Out pre imputed:   " as result "`out_pre'"

* ---------- 1) Load CPS micro ----------
display as text "=== STEP 1: LOADING CPS MICRO DATA ==="
capture confirm file "`cps_dta'"
if _rc {
    display as error "ERROR: Missing CPS micro file: `cps_dta'"
    log close
    exit 198
}
use "`cps_dta'", clear
rename *, lower
compress
display as text "Data loaded successfully!"
display as text "Observations: " as result %12.0fc c(N)
display as text "Variables: " as result %3.0f c(k)

* Analysis weight
capture confirm variable cps_weight
if _rc {
    display as text "cps_weight not found, using analysis_weight = 1"
    gen double analysis_weight = 1
}
else {
    gen double analysis_weight = cps_weight
}

* Sanity: 2022+ dummy flags should be 0/1 and sum to ~1
preserve
keep if year>=2022
capture assert inlist(full_remote,0,1) if full_remote<.
if _rc display as error "WARN: full_remote not in {0,1} for some 2022+ rows"
capture assert inlist(hybrid,0,1) if hybrid<.
if _rc display as error "WARN: hybrid not in {0,1} for some 2022+ rows"
capture assert inlist(full_inperson,0,1) if full_inperson<.
if _rc display as error "WARN: full_inperson not in {0,1} for some 2022+ rows"
gen double _sum3 = full_remote + hybrid + full_inperson
sum _sum3
capture assert inrange(_sum3,0.999,1.001) if _sum3<.
if _rc display as error "WARN: remote+hybrid+inperson != 1 for some 2022+ rows"
restore

* Save full CPS for later mass computations
tempfile cps_full
save "`cps_full'", replace

* ---------- 2) Post-2022 observed (CPS) ----------
display as text "=== STEP 2: POST-2022 OBSERVED (CPS) ==="

* Raw counts by year (unweighted)
preserve
keep year
keep if year>=2022
bys year: gen long n_unw = _N
bys year: keep if _n==1
tempfile Nunw
save `Nunw', replace
restore

* Renorm
preserve
keep year analysis_weight full_remote hybrid full_inperson
keep if year>=2022

* Weighted masses in each mode
gen double wr = analysis_weight * full_remote
gen double wh = analysis_weight * hybrid
gen double wi = analysis_weight * full_inperson

collapse (sum) wr wh wi W = analysis_weight (count) n_unw = year, by(year)

* Denominator is only the mass classified into the 3 modes
gen double denom = wr + wh + wi

* Shares renormalized within the 3-way split
gen share_full_remote   = wr / denom
gen share_hybrid        = wh / denom
gen share_full_inperson = wi / denom

* Coverage of the three-mode classification (should be ~1 if exhaustive)
gen coverage = denom / W

order year n_unw W coverage share_full_remote share_hybrid share_full_inperson

export delimited using "`outdir'/moments_post2022_observed_by_year_RENORM_sipp.csv", replace
display as text "Wrote renormalized observed: `outdir'/moments_post2022_observed_by_year_RENORM_sipp.csv"
restore

* Weighted shares by year
preserve
keep year analysis_weight full_remote hybrid full_inperson
keep if year>=2022
collapse (sum) wsum = analysis_weight ///
        (mean) share_full_remote   = full_remote ///
               share_hybrid        = hybrid      ///
               share_full_inperson = full_inperson ///
        [pw = analysis_weight], by(year)

merge 1:1 year using `Nunw', nogen
order year n_unw wsum share_full_remote share_hybrid share_full_inperson
export delimited using "`out_post_obs'", replace
display as text "Wrote post-2022 observed: `out_post_obs'"
restore

* ---------- 3) Key map (cell_id -> occ2×edu3×sex) ----------
display as text "=== STEP 3: KEY MAP FROM CPS ==="
preserve
keep cell_id occ2_harmonized edu3 sex
keep if !missing(cell_id)
duplicates drop
tempfile keymap
save "`keymap'", replace
display as text "Key map saved."
restore

* ---------- 4) Key-level λ medians from bridge (2022–2025) ----------
display as text "=== STEP 4: KEY-LEVEL λ FROM BRIDGE (2022+) ==="
capture confirm file "`bridge_dta'"
if _rc {
    display as error "ERROR: Missing bridge file: `bridge_dta'"
    log close
    exit 198
}
use "`bridge_dta'", clear
rename *, lower
merge m:1 cell_id using "`keymap'", keep(3) nogen
foreach v in lambda_remote lambda_hybrid lambda_inperson {
    capture confirm variable `v'
    if _rc {
        display as error "ERROR: Bridge missing variable `v'"
        log close
        exit 459
    }
    capture confirm numeric variable `v'
    if _rc destring `v', replace force
}
collapse (median) lambda_remote lambda_hybrid lambda_inperson, by(occ2_harmonized edu3 sex)
tempfile lambdakey
save "`lambdakey'", replace
display as text "Key-level λ medians saved."

* ---------- 5) CPS cell masses by year (all years) ----------
display as text "=== STEP 5: CPS CELL MASSES (ALL YEARS) ==="
use "`cps_full'", clear
keep year cell_id analysis_weight occ2_harmonized edu3 sex
keep if !missing(year) & !missing(cell_id)
collapse (sum) w_cell = analysis_weight, by(year cell_id occ2_harmonized edu3 sex)
tempfile cpsmass
save "`cpsmass'", replace
display as text "CPS masses saved."

* ---------- Helper: function-like block to compute SIPP→CPS by year range ----------
program drop _all
program define _compute_sipp_to_cps, rclass
    * expects: using sipp_dta (lowercased), cpsmass tempfile, lambdakey tempfile
    * args: range flag (0=pre2022, 1=post2022), outfile path, sipp file path, cpsmass file, lambdakey file
    syntax , RANGE(integer) OUTFILE(string) SIPPDTA(string) CPSMASS(string) LAMBDAKEY(string)

    use "`sippdta'", clear
    rename *, lower

    * Normalize shares (SIPP already has normalized shares, but double-check)
    gen double _ss = share_remote_day + share_hybrid_day + share_inperson_day
    replace share_remote_day   = share_remote_day   / _ss if _ss>0 & _ss<.
    replace share_hybrid_day   = share_hybrid_day   / _ss if _ss>0 & _ss<.
    replace share_inperson_day = share_inperson_day / _ss if _ss>0 & _ss<.
    drop _ss

    * Filter by range
    if `range'==0 {
        keep if year<2022
    }
    else {
        keep if year>=2022
    }
    if c(N)==0 {
        display as error "ERROR: SIPP has no rows for requested range."
        exit 459
    }

    * Merge CPS masses (year×cell_id)
    merge 1:1 year cell_id using "`cpsmass'", keep(3) nogen

    * Merge key-level λ
    merge m:1 occ2_harmonized edu3 sex using "`lambdakey'", keep(3) nogen

    * Implied worker shares (same formula as ATUS→CPS)
    gen double den = lambda_remote*share_remote_day + lambda_hybrid*share_hybrid_day + lambda_inperson*share_inperson_day
    replace den = . if den<=0

    foreach k in remote hybrid inperson {
        gen double worker_share_`k'_from_sipp = (lambda_`k' * share_`k'_day) / den
    }
    drop den

    * Aggregate to year with CPS masses
    gen double w_remote_mass   = w_cell * worker_share_remote_from_sipp
    gen double w_hybrid_mass   = w_cell * worker_share_hybrid_from_sipp
    gen double w_inperson_mass = w_cell * worker_share_inperson_from_sipp

    collapse (sum) wtot = w_cell ///
             (sum) w_remote   = w_remote_mass ///
             (sum) w_hybrid   = w_hybrid_mass ///
             (sum) w_inperson = w_inperson_mass, by(year)

    gen share_full_remote   = w_remote   / wtot
    gen share_hybrid        = w_hybrid   / wtot
    gen share_full_inperson = w_inperson / wtot

    * Sanity
    capture assert inrange(share_full_remote,0,1) & inrange(share_hybrid,0,1) & inrange(share_full_inperson,0,1)
    if _rc display as error "WARN: shares out of [0,1]"
    capture assert inrange(share_full_remote + share_hybrid + share_full_inperson, 0.999, 1.001)
    if _rc display as error "WARN: shares don't sum to ~1"

    order year share_full_remote share_hybrid share_full_inperson wtot
    export delimited using "`outfile'", replace
    display as text "Wrote: `outfile'"
end

* ---------- 6) SIPP→CPS imputed: PRE-2022 ----------
display as text "=== STEP 6: SIPP→CPS IMPUTED (PRE-2022) ==="
_compute_sipp_to_cps, range(0) outfile("`out_pre'") sippdta("`sipp_dta'") cpsmass("`cpsmass'") lambdakey("`lambdakey'")

* ---------- 7) SIPP→CPS imputed: POST-2022 ----------
display as text "=== STEP 7: SIPP→CPS IMPUTED (POST-2022) ==="
_compute_sipp_to_cps, range(1) outfile("`out_post_imp'") sippdta("`sipp_dta'") cpsmass("`cpsmass'") lambdakey("`lambdakey'")

* ---------- 8) Comparison tables ----------
display as text "=== STEP 8: VALIDATION & COMPARISON ==="
tempfile post_obs post_imp pre_imp

* Load post observed
import delimited "`out_post_obs'", clear varn(1)
rename *, lower
gen source = "observed_2022plus"
save "`post_obs'", replace

* Load post imputed
import delimited "`out_post_imp'", clear varn(1)
rename *, lower
gen source = "imputed_2022plus_sipp"
save "`post_imp'", replace

* Load pre imputed
import delimited "`out_pre'", clear varn(1)
rename *, lower
gen source = "imputed_pre2022_sipp"
save "`pre_imp'", replace

* Combined pre + post (long)
use "`post_obs'", clear
append using "`post_imp'"
append using "`pre_imp'"
order source year share_full_remote share_hybrid share_full_inperson wtot
sort source year
export delimited using "`outdir'/moments_combined_pre_post_sipp.csv", replace
display as text "Wrote combined: `outdir'/moments_combined_pre_post_sipp.csv'"

* Side-by-side comparison for 2022–2025: observed vs imputed
preserve
use "`post_obs'", clear
rename share_full_remote   obs_remote
rename share_hybrid        obs_hybrid
rename share_full_inperson obs_inperson
merge 1:1 year using "`post_imp'", nogen keep(3)
rename share_full_remote   imp_remote_sipp
rename share_hybrid        imp_hybrid_sipp
rename share_full_inperson imp_inperson_sipp

gen d_remote_sipp   = imp_remote_sipp   - obs_remote
gen d_hybrid_sipp   = imp_hybrid_sipp   - obs_hybrid
gen d_inperson_sipp = imp_inperson_sipp - obs_inperson

order year obs_remote imp_remote_sipp d_remote_sipp obs_hybrid imp_hybrid_sipp d_hybrid_sipp obs_inperson imp_inperson_sipp d_inperson_sipp
export delimited using "`outdir'/moments_post2022_obs_vs_imp_sipp.csv", replace
display as text "Wrote comparison (post only): `outdir'/moments_post2022_obs_vs_imp_sipp.csv'"
restore

* ---------- 9) SIPP vs ATUS validation (if ATUS results exist) ----------
display as text "=== STEP 9: SIPP vs ATUS VALIDATION ==="
local atus_post_imp "`outdir'/moments_post2022_imputed_by_year.csv"
local atus_pre      "`outdir'/moments_pre2022_by_year.csv"

capture confirm file "`atus_post_imp'"
local atus_post_exists = !_rc
capture confirm file "`atus_pre'"
local atus_pre_exists = !_rc

if `atus_post_exists' & `atus_pre_exists' {
    display as text "Found ATUS results, creating SIPP vs ATUS comparison"
    
    * Load SIPP results
    import delimited "`out_post_imp'", clear varn(1)
    rename *, lower
    rename share_full_remote   sipp_remote_post
    rename share_hybrid        sipp_hybrid_post
    rename share_full_inperson sipp_inperson_post
    tempfile sipp_post
    save "`sipp_post'", replace
    
    import delimited "`out_pre'", clear varn(1)
    rename *, lower
    rename share_full_remote   sipp_remote_pre
    rename share_hybrid        sipp_hybrid_pre
    rename share_full_inperson sipp_inperson_pre
    tempfile sipp_pre
    save "`sipp_pre'", replace
    
    * Load ATUS results
    import delimited "`atus_post_imp'", clear varn(1)
    rename *, lower
    rename share_full_remote   atus_remote_post
    rename share_hybrid        atus_hybrid_post
    rename share_full_inperson atus_inperson_post
    merge 1:1 year using "`sipp_post'", nogen keep(3)
    tempfile comp_post
    save "`comp_post'", replace
    
    import delimited "`atus_pre'", clear varn(1)
    rename *, lower
    rename share_full_remote   atus_remote_pre
    rename share_hybrid        atus_hybrid_pre
    rename share_full_inperson atus_inperson_pre
    merge 1:1 year using "`sipp_pre'", nogen keep(3)
    
    * Append post-2022 data
    append using "`comp_post'"
    
    * Calculate differences
    gen d_remote_sipp_atus   = sipp_remote_post   - atus_remote_post   if year>=2022
    gen d_hybrid_sipp_atus   = sipp_hybrid_post   - atus_hybrid_post   if year>=2022
    gen d_inperson_sipp_atus = sipp_inperson_post - atus_inperson_post if year>=2022
    
    replace d_remote_sipp_atus   = sipp_remote_pre   - atus_remote_pre   if year<2022
    replace d_hybrid_sipp_atus   = sipp_hybrid_pre   - atus_hybrid_pre   if year<2022
    replace d_inperson_sipp_atus = sipp_inperson_pre - atus_inperson_pre if year<2022
    
    * Create period indicator
    gen period = "pre2022" if year<2022
    replace period = "post2022" if year>=2022
    
    sort period year
    order period year atus_* sipp_* d_*
    export delimited using "`outdir'/moments_validation_sipp_vs_atus.csv", replace
    display as text "Wrote SIPP vs ATUS validation: `outdir'/moments_validation_sipp_vs_atus.csv'"
    
    * Summary statistics of differences
    preserve
    keep if !missing(d_remote_sipp_atus)
    sum d_remote_sipp_atus d_hybrid_sipp_atus d_inperson_sipp_atus
    display as text "=== SUMMARY: SIPP vs ATUS DIFFERENCES ==="
    display as text "Mean absolute difference (remote):   " %6.4f r(mean)
    display as text "Mean absolute difference (hybrid):   " %6.4f r(mean)
    display as text "Mean absolute difference (inperson): " %6.4f r(mean)
    restore
}
else {
    display as text "ATUS results not found, skipping SIPP vs ATUS comparison"
    display as text "Missing files:"
    if !`atus_post_exists' display as text "  - `atus_post_imp'"
    if !`atus_pre_exists'  display as text "  - `atus_pre'"
}

display as text "=== PIPELINE COMPLETED SUCCESSFULLY ==="
display as text "End time: " as result c(current_time) " " c(current_date)
log close
