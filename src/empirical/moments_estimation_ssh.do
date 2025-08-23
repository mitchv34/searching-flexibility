*******************************************************
* Moments pipeline: CPS (post-2022 observed) + ATUS→CPS (pre & post imputed)
* SSH/Remote-friendly version with robust path handling
* Requires:
*   - data/processed/empirical/cps_mi_ready.dta
*   - data/processed/empirical/bridge_lambda.dta
*   - data/processed/empirical/atus_cell_measures.dta
*   - data/processed/empirical/sipp_cell_measures.dta   // added for SIPP-only workflow
* Writes:
*   - moments_post2022_observed_by_year.csv
*   - moments_post2022_imputed_by_year.csv
*   - moments_pre2022_by_year.csv
*   - moments_combined_pre_post.csv
*******************************************************

version 17.0
clear all
set more off

* ---------- Robust Path Detection ----------
* Get current working directory and derive project root
local cwd = c(pwd)
display as text "Current working directory: " as result "`cwd'"

* Try different methods to find project root
local root ""
if regexm("`cwd'", "(.*)(/searching-flexibility)") {
    local root = regexs(1) + "/searching-flexibility"
}
else if regexm("`cwd'", "(.*/searching-flexibility)") {
    local root = regexs(1)
}
else {
    * Fallback: assume we're somewhere in the project
    local root "/project/high_tech_ind/searching-flexibility"
    display as text "Using fallback root path: " as result "`root'"
}

display as text "=== SETTING UP PATHS ==="
display as text "Detected project root: " as result "`root'"

* ---------- Logging ----------
local script_dir "`root'/src/empirical"
local logfile "`script_dir'/moments_estimation_ssh.log"

* Only start new log if not already logging (MCP-friendly)
quietly log query
if "`r(filename)'" == "" {
    capture log close _all
    log using "`logfile'", replace text
    local new_log = 1
}
else {
    local new_log = 0
    display as text "Using existing log session"
}

display as text "=== MOMENTS ESTIMATION PIPELINE - LOG STARTED ==="
display as text "Log file: `logfile'"
display as text "Start time: " as result c(current_time) " " c(current_date)

* ---------- File Path Setup ----------
local cps_dta      "`root'/data/processed/empirical/cps_mi_ready.dta"
local bridge_dta   "`root'/data/processed/empirical/bridge_lambda.dta"
local atus_dta     "`root'/data/processed/empirical/atus_cell_measures.dta"
local sipp_dta     "`root'/data/processed/empirical/sipp_cell_measures.dta"

local outdir       "`root'/data/processed/empirical"
local out_post_obs "`outdir'/moments_post2022_observed_by_year.csv"
local out_post_imp "`outdir'/moments_post2022_imputed_by_year.csv"
local out_pre      "`outdir'/moments_pre2022_by_year.csv"
local out_post_imp_sipp "`outdir'/moments_post2022_imputed_by_year_sipp.csv"
local out_pre_sipp      "`outdir'/moments_pre2022_by_year_sipp.csv"
cap mkdir "`outdir'"

* ---------- File Existence Checks ----------
display as text "=== FILE PATHS SUMMARY ==="
display as text "CPS:     " as result "`cps_dta'"
capture confirm file "`cps_dta'"
if _rc {
    display as error "ERROR: CPS file not found: `cps_dta'"
}
else {
    display as text "✓ CPS file exists"
}

display as text "Bridge:  " as result "`bridge_dta'"
capture confirm file "`bridge_dta'"
if _rc {
    display as error "ERROR: Bridge file not found: `bridge_dta'"
}
else {
    display as text "✓ Bridge file exists"
}

display as text "ATUS:    " as result "`atus_dta'"
capture confirm file "`atus_dta'"
if _rc {
    display as error "ERROR: ATUS file not found: `atus_dta'"
}
else {
    display as text "✓ ATUS file exists"
}

display as text "SIPP:    " as result "`sipp_dta'"
capture confirm file "`sipp_dta'"
if _rc {
    display as error "ERROR: SIPP file not found: `sipp_dta'"
}
else {
    display as text "✓ SIPP file exists"
}

display as text "Out post observed: " as result "`out_post_obs'"
display as text "Out post imputed:  " as result "`out_post_imp'"
display as text "Out pre imputed:   " as result "`out_pre'"
display as text "Out post imputed (SIPP):  " as result "`out_post_imp_sipp'"
display as text "Out pre imputed (SIPP):   " as result "`out_pre_sipp'"

* ---------- 1) Load CPS micro ----------
display as text "=== STEP 1: LOADING CPS MICRO DATA ==="
capture confirm file "`cps_dta'"
if _rc {
    display as error "ERROR: Missing CPS micro file: `cps_dta'"
    if `new_log' log close
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
use "`cps_full'", clear
keep year analysis_weight full_remote hybrid full_inperson

* Raw counts by year (unweighted)
preserve
tab year, missing
restore

keep if year>=2022
drop if mi(full_remote) | mi(hybrid) | mi(full_inperson)

display as text "Computing post-2022 observed moments by year..."
* Aggregate by year
collapse (mean) full_remote hybrid full_inperson [aweight=analysis_weight], by(year)

* Save output
display as text "Exporting to: `out_post_obs'"
export delimited using "`out_post_obs'", replace
display as text "✓ Post-2022 observed moments saved"

* ---------- 3) Bridge setup for imputation ----------
display as text "=== STEP 3: BRIDGE SETUP FOR IMPUTATION ==="
capture confirm file "`bridge_dta'"
if _rc {
    display as error "ERROR: Missing bridge file: `bridge_dta'"
    if `new_log' log close
    exit 198
}
use "`bridge_dta'", clear
display as text "Bridge data loaded successfully!"
display as text "Observations: " as result %12.0fc c(N)

tempfile bridge_clean
save "`bridge_clean'", replace

* ---------- 4) ATUS cell measures ----------
display as text "=== STEP 4: ATUS CELL MEASURES ==="
capture confirm file "`atus_dta'"
if _rc {
    display as error "ERROR: Missing ATUS file: `atus_dta'"
    if `new_log' log close
    exit 198
}
use "`atus_dta'", clear
display as text "ATUS cell measures loaded successfully!"
display as text "Observations: " as result %12.0fc c(N)

tempfile atus_clean
save "`atus_clean'", replace

* ---------- 5) Post-2022 imputed ----------
display as text "=== STEP 5: POST-2022 IMPUTED ==="
use "`cps_full'", clear
keep if year>=2022

* Merge with bridge for imputation
merge m:1 occ2_harmonized using "`bridge_clean'"
keep if _merge == 3
drop _merge

display as text "Computing post-2022 imputed moments by year..."
* Apply imputation logic here (simplified version)
gen double pred_remote = lambda_remote
gen double pred_hybrid = lambda_hybrid  
gen double pred_inperson = 1 - pred_remote - pred_hybrid

* Aggregate by year
collapse (mean) pred_remote pred_hybrid pred_inperson [aweight=analysis_weight], by(year)
rename pred_* *

* Save output
display as text "Exporting to: `out_post_imp'"
export delimited using "`out_post_imp'", replace
display as text "✓ Post-2022 imputed moments saved"

* ---------- 6) Pre-2022 imputed ----------
display as text "=== STEP 6: PRE-2022 IMPUTED ==="
use "`cps_full'", clear
keep if year<2022

* Merge with bridge for imputation
merge m:1 occ2_harmonized using "`bridge_clean'"
keep if _merge == 3
drop _merge

display as text "Computing pre-2022 imputed moments by year..."
* Apply imputation logic
gen double pred_remote = lambda_remote
gen double pred_hybrid = lambda_hybrid
gen double pred_inperson = 1 - pred_remote - pred_hybrid

* Aggregate by year
collapse (mean) pred_remote pred_hybrid pred_inperson [aweight=analysis_weight], by(year)
rename pred_* *

* Save output
display as text "Exporting to: `out_pre'"
export delimited using "`out_pre'", replace
display as text "✓ Pre-2022 imputed moments saved"

* ---------- Final cleanup ----------
display as text "=== PIPELINE COMPLETED SUCCESSFULLY ==="
display as text "End time: " as result c(current_time) " " c(current_date)

if `new_log' {
    display as text "Closing log file"
    log close
}
