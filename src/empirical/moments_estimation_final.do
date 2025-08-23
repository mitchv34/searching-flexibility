*******************************************************
* Moments pipeline: FINAL FIXED VERSION for SSH/Remote use
* Properly handles bridge data with correct variable names
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

display as text "=== MOMENTS ESTIMATION PIPELINE (FINAL) ==="
display as text "Project root: " as result "`root'"
display as text "Start time: " as result c(current_time) " " c(current_date)

* ---------- Logging ----------
local script_dir "`root'/src/empirical"
local logfile "`script_dir'/moments_estimation_final.log"
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

local outdir       "`root'/data/processed/empirical"
local out_post_obs "`outdir'/moments_post2022_observed_by_year_final.csv"
local out_post_imp "`outdir'/moments_post2022_imputed_by_year_final.csv"
local out_pre      "`outdir'/moments_pre2022_by_year_final.csv"
local out_combined "`outdir'/moments_combined_pre_post_final.csv"

* ---------- 1) Load CPS data ----------
display as text "=== STEP 1: LOADING CPS DATA ==="
use "`cps_dta'", clear
rename *, lower  // This will make YEAR->year if needed
display as text "CPS observations: " as result %12.0fc c(N)
display as text "CPS years range: " as result %4.0f = r(min) " to " %4.0f = r(max)
quietly summarize year

gen double analysis_weight = cps_weight
tempfile cps_full
save "`cps_full'", replace

* ---------- 2) Post-2022 observed ----------
display as text "=== STEP 2: POST-2022 OBSERVED ==="
use "`cps_full'", clear
keep if year>=2022
keep year analysis_weight full_remote hybrid full_inperson
drop if mi(full_remote) | mi(hybrid) | mi(full_inperson)

display as text "Post-2022 observations: " as result %12.0fc c(N)
collapse (mean) full_remote hybrid full_inperson [aweight=analysis_weight], by(year)
list

export delimited using "`out_post_obs'", replace
display as text "✓ Post-2022 observed saved"

* ---------- 3) Prepare bridge data ----------
display as text "=== STEP 3: PREPARING BRIDGE DATA ==="
use "`bridge_dta'", clear
rename *, lower  // This will make YEAR->year
display as text "Bridge observations: " as result %12.0fc c(N)

* Since the bridge has multiple years and cells, collapse to occupation level
* Take mean of lambda values across all years and demographic cells
display as text "Collapsing bridge data by occupation..."
collapse (mean) lambda_remote lambda_hybrid lambda_inperson, by(occ2_harmonized)
display as text "Bridge data after collapse: " as result %12.0fc c(N) " unique occupations"

* Verify lambdas are reasonable
summarize lambda_*
tempfile bridge_clean
save "`bridge_clean'", replace

* ---------- 4) Post-2022 imputed ----------
display as text "=== STEP 4: POST-2022 IMPUTED ==="
use "`cps_full'", clear
keep if year>=2022

display as text "Merging CPS with bridge data..."
merge m:1 occ2_harmonized using "`bridge_clean'"
tab _merge
keep if _merge == 3
drop _merge

* Apply imputation using lambda values
gen double pred_remote = lambda_remote
gen double pred_hybrid = lambda_hybrid  
gen double pred_inperson = lambda_inperson

* Check that predicted values are reasonable
summarize pred_*

collapse (mean) pred_remote pred_hybrid pred_inperson [aweight=analysis_weight], by(year)
rename pred_* *
list

export delimited using "`out_post_imp'", replace
display as text "✓ Post-2022 imputed saved"

* ---------- 5) Pre-2022 imputed ----------
display as text "=== STEP 5: PRE-2022 IMPUTED ==="
use "`cps_full'", clear
keep if year<2022

display as text "Pre-2022 observations before merge: " as result %12.0fc c(N)
merge m:1 occ2_harmonized using "`bridge_clean'"
tab _merge
keep if _merge == 3
drop _merge
display as text "Pre-2022 observations after merge: " as result %12.0fc c(N)

gen double pred_remote = lambda_remote
gen double pred_hybrid = lambda_hybrid
gen double pred_inperson = lambda_inperson

collapse (mean) pred_remote pred_hybrid pred_inperson [aweight=analysis_weight], by(year)
rename pred_* *
list

export delimited using "`out_pre'", replace
display as text "✓ Pre-2022 imputed saved"

* ---------- 6) Create combined dataset ----------
display as text "=== STEP 6: CREATING COMBINED DATASET ==="

* Post-2022 observed
import delimited "`out_post_obs'", clear
gen source = "observed"
gen period = "post2022"
tempfile temp1
save "`temp1'", replace

* Post-2022 imputed
import delimited "`out_post_imp'", clear
gen source = "imputed"
gen period = "post2022"
append using "`temp1'"
tempfile temp2
save "`temp2'", replace

* Pre-2022 imputed
import delimited "`out_pre'", clear
gen source = "imputed"
gen period = "pre2022"
append using "`temp2'"

* Sort and save
sort year source
list

export delimited using "`out_combined'", replace
display as text "✓ Combined dataset saved"

* ---------- Summary ----------
display as text "=== PIPELINE COMPLETED SUCCESSFULLY ==="
display as text "Output files created:"
display as text "  1. `out_post_obs'"
display as text "  2. `out_post_imp'"
display as text "  3. `out_pre'"
display as text "  4. `out_combined'"
display as text "End time: " as result c(current_time) " " c(current_date)

if `new_log' log close
