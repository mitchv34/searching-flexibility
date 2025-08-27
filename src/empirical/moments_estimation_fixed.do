*******************************************************
* Moments pipeline: FIXED VERSION for SSH/Remote use
* Handles non-unique merge keys properly
*******************************************************

version 17.0
clear all
set more off

* ---------- Robust Path Detection ----------
local cwd = c(pwd)
local root ""
if regexm("`cwd'", "(.*/searching-flexibility)") {
    local root = regexs(1)
}
else {
    local root "/project/high_tech_ind/searching-flexibility"
}

display as text "=== MOMENTS ESTIMATION PIPELINE (FIXED) ==="
display as text "Project root: " as result "`root'"

* ---------- Logging ----------
local script_dir "`root'/src/empirical"
local logfile "`script_dir'/moments_estimation_fixed.log"
quietly log query
if "`r(filename)'" == "" {
    capture log close _all
    log using "`logfile'", replace text
    local new_log = 1
}
else {
    local new_log = 0
}

* ---------- File Paths ----------
local cps_dta      "`root'/data/processed/empirical/cps_mi_ready.dta"
local bridge_dta   "`root'/data/processed/empirical/bridge_lambda.dta"
local atus_dta     "`root'/data/processed/empirical/atus_cell_measures.dta"

local outdir       "`root'/data/processed/empirical"
local out_post_obs "`outdir'/moments_post2022_observed_by_year_fixed.csv"
local out_post_imp "`outdir'/moments_post2022_imputed_by_year_fixed.csv"
local out_pre      "`outdir'/moments_pre2022_by_year_fixed.csv"

* ---------- 1) Load and check CPS data ----------
display as text "=== STEP 1: LOADING CPS DATA ==="
use "`cps_dta'", clear
rename *, lower
display as text "CPS observations: " as result %12.0fc c(N)

gen double analysis_weight = cps_weight
tempfile cps_full
save "`cps_full'", replace

* ---------- 2) Post-2022 observed ----------
display as text "=== STEP 2: POST-2022 OBSERVED ==="
use "`cps_full'", clear
keep if year>=2022
keep year analysis_weight full_remote hybrid full_inperson
drop if mi(full_remote) | mi(hybrid) | mi(full_inperson)

collapse (mean) full_remote hybrid full_inperson [aweight=analysis_weight], by(year)
export delimited using "`out_post_obs'", replace
display as text "✓ Post-2022 observed saved to: `out_post_obs'"

* ---------- 3) Prepare bridge data for merging ----------
display as text "=== STEP 3: PREPARING BRIDGE DATA ==="
use "`bridge_dta'", clear
display as text "Bridge observations: " as result %12.0fc c(N)

* Check uniqueness by year and occupation
duplicates report occ2_harmonized year
display as text "Checking if occ2_harmonized + year is unique..."

* If still not unique, collapse to occupation level (average across years/demographics)
collapse (mean) lambda_remote lambda_hybrid lambda_inperson, by(occ2_harmonized)
display as text "Bridge data collapsed to occupation level: " as result %12.0fc c(N) " observations"

tempfile bridge_clean
save "`bridge_clean'", replace

* ---------- 4) Post-2022 imputed ----------
display as text "=== STEP 4: POST-2022 IMPUTED ==="
use "`cps_full'", clear
keep if year>=2022

merge m:1 occ2_harmonized using "`bridge_clean'"
display as text "Merge results: _merge==1: " as result %12.0fc = r(merge_1) 
display as text "               _merge==2: " as result %12.0fc = r(merge_2)
display as text "               _merge==3: " as result %12.0fc = r(merge_3)

keep if _merge == 3
drop _merge

* Apply imputation
gen double pred_remote = lambda_remote
gen double pred_hybrid = lambda_hybrid  
gen double pred_inperson = lambda_inperson

collapse (mean) pred_remote pred_hybrid pred_inperson [aweight=analysis_weight], by(year)
rename pred_* *

export delimited using "`out_post_imp'", replace
display as text "✓ Post-2022 imputed saved to: `out_post_imp'"

* ---------- 5) Pre-2022 imputed ----------
display as text "=== STEP 5: PRE-2022 IMPUTED ==="
use "`cps_full'", clear
keep if year<2022

merge m:1 occ2_harmonized using "`bridge_clean'"
keep if _merge == 3
drop _merge

gen double pred_remote = lambda_remote
gen double pred_hybrid = lambda_hybrid
gen double pred_inperson = lambda_inperson

collapse (mean) pred_remote pred_hybrid pred_inperson [aweight=analysis_weight], by(year)
rename pred_* *

export delimited using "`out_pre'", replace
display as text "✓ Pre-2022 imputed saved to: `out_pre'"

* ---------- 6) Combine results ----------
display as text "=== STEP 6: COMBINING RESULTS ==="

* Load post-2022 observed
preserve
import delimited "`out_post_obs'", clear
gen source = "observed"
tempfile post_obs
save "`post_obs'", replace
restore

* Load post-2022 imputed  
preserve
import delimited "`out_post_imp'", clear
gen source = "imputed"
tempfile post_imp
save "`post_imp'", replace
restore

* Load pre-2022 imputed
import delimited "`out_pre'", clear
gen source = "imputed"

* Combine all
append using "`post_obs'"
append using "`post_imp'"

sort year
local out_combined "`outdir'/moments_combined_pre_post_fixed.csv"
export delimited using "`out_combined'", replace
display as text "✓ Combined results saved to: `out_combined'"

display as text "=== PIPELINE COMPLETED SUCCESSFULLY ==="
display as text "Files created:"
display as text "  - `out_post_obs'"
display as text "  - `out_post_imp'"
display as text "  - `out_pre'"
display as text "  - `out_combined'"

if `new_log' log close
