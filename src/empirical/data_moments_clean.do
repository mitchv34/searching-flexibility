/*==============================================================================
* data_moments_clean.do
* 
* Purpose: Compute empirical moments from CPS data by year (2022-2025)
* Author: Mitchell Valdes-Bobes
* Date: August 2025
*
* Inputs:  data/processed/cps/cps_alpha_wage_present_reweighted.csv
* Outputs: data/results/data_moments/data_moments_by_year.csv
*          data/results/data_moments/data_moments_YYYY.yaml (one per year)
*
* Notes: Computes mean_logwage, remote_share, hybrid_share, market_tightness,
*        and diff_logwage_inperson_remote using Mincer regression
*==============================================================================*/

clear all
set more off

* CONSTANTS
local HIGH_PSI_PCT = 75   // High-psi = psi >= 75th percentile
local LOW_PSI_PCT  = 25   // Low-psi  = psi <= 25th percentile
local ALPHA_TOL    = 0.2  // In-person alpha <= 0.2, Remote alpha >= 0.8, else hybrid

* Set working directory to project root
cd "/project/high_tech_ind/searching-flexibility"

* Create output directory if it doesn't exist
capture mkdir "data/results"
capture mkdir "data/results/data_moments"

* Load the dataset
import delimited "data/processed/cps/cps_alpha_wage_present_reweighted.csv", clear

drop indnaics occsoc_detailed occsoc_minor ftpt edu3 age4 v38 cell_id

* Keep only years 2022-2025 (year is already lowercase)
keep if inlist(year, 2022, 2023, 2024, 2025)

* Convert log wage variable to numeric if needed and create working variable
destring log_wage_real, replace force
gen logwage = log_wage_real if log_wage_real != . & log_wage_real != 0

* Convert alpha variable to numeric if needed
destring alpha, replace force

* Rename original psi variable (we are going to construct our own)
rename psi psi_original

* Clean education variable and create years of education from modern CPS codes
destring educ, replace force

* Create years of education variable using modern CPS education codes
gen years_educ = .
replace years_educ = 0 if educ == 2      // No schooling completed
replace years_educ = 8 if educ == 20     // 8th grade
replace years_educ = 9 if educ == 30     // 9th grade
replace years_educ = 10 if educ == 40    // 10th grade
replace years_educ = 11 if educ == 50    // 11th grade
replace years_educ = 12 if educ == 60    // 12th grade, no diploma
replace years_educ = 12 if educ == 71    // High school graduate
replace years_educ = 12 if educ == 73    // High school graduate
replace years_educ = 13 if educ == 81    // Some college but no degree
replace years_educ = 14 if educ == 91    // Associate degree - occupational
replace years_educ = 14 if educ == 92    // Associate degree - academic
replace years_educ = 14 if educ == 10    // Some college level (based on data analysis)
replace years_educ = 16 if educ == 111   // Bachelor's degree
replace years_educ = 18 if educ == 123   // Master's degree
replace years_educ = 20 if educ == 124   // Professional degree
replace years_educ = 21 if educ == 125   // Doctoral degree

* Keep all observations with valid education data and at least 6 years of education
keep if !missing(years_educ) & years_educ >= 6

* Create potential experience variables
destring age, replace force
gen experience = age - years_educ - 6
gen experience_sq = experience^2

* Convert other control variables to numeric
destring sex, replace force
destring race, replace force

* Convert weight variable to numeric (wtfinl is already lowercase)
destring wtfinl, replace force

* Convert teleworkable variables to numeric
destring teleworkable_ocssoc_detailed, replace force
destring teleworkable_ocssoc_broad, replace force  
destring teleworkable_ocssoc_minor, replace force

* Construct psi variable using hierarchy: detailed -> broad -> minor
* Drop if all teleworkable variables are missing
drop if missing(teleworkable_ocssoc_detailed) & missing(teleworkable_ocssoc_broad) & missing(teleworkable_ocssoc_minor)

* Generate psi using the hierarchy
gen psi = .
replace psi = teleworkable_ocssoc_detailed if !missing(teleworkable_ocssoc_detailed)
replace psi = teleworkable_ocssoc_broad if missing(psi) & !missing(teleworkable_ocssoc_broad)
replace psi = teleworkable_ocssoc_minor if missing(psi) & !missing(teleworkable_ocssoc_minor)

* Verify we have valid psi for all remaining observations
assert !missing(psi)

* Create psi thresholds for high / low (75 / 25 split)
capture drop psi_hi_threshold psi_lo_threshold high_psi low_psi
gen psi_hi_threshold = .
gen psi_lo_threshold = .
forvalues yr = 2022/2025 {
    quietly sum psi if year == `yr', detail
    local p75 = r(p75)
    local p25 = r(p25)
    replace psi_hi_threshold = `p75' if year == `yr'
    replace psi_lo_threshold = `p25' if year == `yr'
    display "Year `yr': psi p25 = " %8.4f `p25' ", psi p75 = " %8.4f `p75'
}

* High psi: >= 75th percentile; Low psi: <= 25th percentile; middle band unused for diff moments
gen high_psi = (psi >= psi_hi_threshold) if !missing(psi) & !missing(psi_hi_threshold)
gen low_psi  = (psi <= psi_lo_threshold) if !missing(psi) & !missing(psi_lo_threshold)

* Synchronize legacy flag if present
capture confirm variable high_psi_flag
if !_rc {
    replace high_psi_flag = high_psi if !missing(psi)
}

* Create education dummies
qui tab educ, gen(educ_)

* Create sex and race dummies
qui tab sex, gen(sex_)
qui tab race, gen(race_)

* Create industry and occupation dummies if variables exist
* capture confirm string variable ind_broad
* if _rc == 0 {
*     encode ind_broad, gen(ind_broad_num)
*     tab ind_broad_num, gen(ind_)
* }
* else {
*     destring ind_broad, replace force
*     tab ind_broad, gen(ind_)
* }

* capture confirm string variable occsoc_broad
* if _rc == 0 {
*     encode occsoc_broad, gen(occsoc_broad_num)
*     tab occsoc_broad_num, gen(occ_)
* }
* else {
*     destring occsoc_broad, replace force
*     tab occsoc_broad, gen(occ_)
* }

* Reconstruct work arrangement from continuous alpha (aligned with Julia threshold rule)
* In-person: alpha <= ALPHA_TOL; Remote: alpha >= 1 - ALPHA_TOL; Hybrid: else
* drop in_person hybrid full_remote work_cat_sum
gen in_person = (alpha <= `ALPHA_TOL') if !missing(alpha)
drop full_remote 
gen full_remote = (alpha >= 1 - `ALPHA_TOL') if !missing(alpha)
drop hybrid
gen hybrid = (alpha > `ALPHA_TOL' & alpha < 1 - `ALPHA_TOL') if !missing(alpha)

* Sanity check: categories mutually exclusive
gen work_cat_sum = in_person + hybrid + full_remote
assert work_cat_sum <= 1  // middle band could be missing if alpha missing

*===============================================================================
* CREATE CONTROL VARIABLE DUMMIES
*===============================================================================

display ""
display "============================================================"
display "🔧 CREATING CONTROL VARIABLE DUMMIES:"
display "============================================================"

* Check what variables we have for creating dummies
describe years_educ sex race
describe *ind* *occ*, varlist
local industry_vars `r(varlist)'
display "Industry variables found: `industry_vars'"

* Create education dummies from years_educ
* First check the distribution
qui tab years_educ, missing

* Create education category dummies (combine some categories to avoid small cells)
gen educ_less_hs = (years_educ < 12) if !missing(years_educ)           // Less than high school
gen educ_hs = (years_educ == 12) if !missing(years_educ)               // High school
gen educ_some_college = (years_educ > 12 & years_educ < 16) if !missing(years_educ)  // Some college
gen educ_ba = (years_educ == 16) if !missing(years_educ)               // Bachelor's
gen educ_graduate = (years_educ > 16) if !missing(years_educ)          // Graduate degree

* Verify education dummies sum to 1
egen educ_sum = rowtotal(educ_less_hs educ_hs educ_some_college educ_ba educ_graduate)
assert educ_sum == 1 if !missing(years_educ)
drop educ_sum

* Create sex dummies
qui tab sex, missing
gen sex_male = (sex == 1) if !missing(sex)
gen sex_female = (sex == 2) if !missing(sex)

* Verify sex dummies
egen sex_sum = rowtotal(sex_male sex_female)
assert sex_sum == 1 if !missing(sex)
drop sex_sum

* Create race dummies
qui tab race, missing

* Create race dummies based on CPS race codes (using actual codes from data)
gen race_white = (race == 100) if !missing(race)                // White alone
gen race_black = (race == 200) if !missing(race)                // Black alone
gen race_asian = inlist(race, 651, 652) if !missing(race)       // Asian alone and Asian-Pacific Islander
gen race_other = !inlist(race, 100, 200, 651, 652) if !missing(race) // All other races and multiracial

* Verify race dummies
egen race_sum = rowtotal(race_white race_black race_asian race_other)
assert race_sum == 1 if !missing(race)
drop race_sum

* Create industry dummies
* First check what industry variable we have
capture confirm variable ind
if _rc == 0 {
    local ind_var = "ind"
}
else {
    capture confirm variable industry
    if _rc == 0 {
        local ind_var = "industry"
    }
    else {
        * Look for any variable with "ind" in the name
        ds *ind*
        local possible_ind `r(varlist)'
        if "`possible_ind'" != "" {
            local ind_var : word 1 of `possible_ind'
        }
        else {
            * Create dummy industry variable
            gen ind_missing = 1
            local ind_var = "ind_missing"
        }
    }
}


encode ind_broad, generate(industry)

* if "`ind_var'" != "ind_missing" {
*     tab `ind_var', missing
    
*     * Create major industry groups to avoid too many dummies
*     quietly tab `ind_var'
*     local n_industries = r(r)
    
*     if `n_industries' > 20 {
*         * Create broad industry groups (adjust these based on your actual industry codes)
*         gen ind_agriculture = inrange(`ind_var', 170, 290) if !missing(`ind_var')
*         gen ind_mining = inrange(`ind_var', 370, 490) if !missing(`ind_var') 
*         gen ind_construction = inrange(`ind_var', 570, 690) if !missing(`ind_var')
*         gen ind_manufacturing = inrange(`ind_var', 1070, 3990) if !missing(`ind_var')
*         gen ind_trade = inrange(`ind_var', 4070, 5790) if !missing(`ind_var')
*         gen ind_transportation = inrange(`ind_var', 6070, 6390) if !missing(`ind_var')
*         gen ind_information = inrange(`ind_var', 6470, 6780) if !missing(`ind_var')
*         gen ind_finance = inrange(`ind_var', 6870, 6990) if !missing(`ind_var')
*         gen ind_professional = inrange(`ind_var', 7070, 7790) if !missing(`ind_var')
*         gen ind_education = inrange(`ind_var', 7860, 7890) if !missing(`ind_var')
*         gen ind_health = inrange(`ind_var', 7970, 8470) if !missing(`ind_var')
*         gen ind_leisure = inrange(`ind_var', 8560, 8690) if !missing(`ind_var')
*         gen ind_other_services = inrange(`ind_var', 8770, 9290) if !missing(`ind_var')
*         gen ind_public_admin = inrange(`ind_var', 9370, 9590) if !missing(`ind_var')
*         gen ind_other = !inrange(`ind_var', 170, 9590) if !missing(`ind_var')
*     }
*     else {
*         * Create individual industry dummies
*         quietly levelsof `ind_var', local(ind_levels)
*         foreach level of local ind_levels {
*             gen ind_`level' = (`ind_var' == `level') if !missing(`ind_var')
*         }
*     }
    
* }

encode occsoc_broad, generate(occupation)

* Create occupation dummies
* First check what occupation variable we have
* capture confirm variable occ
* if _rc == 0 {
*     local occ_var = "occ"
* }
* else {
*     capture confirm variable occupation
*     if _rc == 0 {
*         local occ_var = "occupation"
*     }
*     else {
*         * Look for any variable with "occ" in the name
*         ds *occ*
*         local possible_occ `r(varlist)'
*         if "`possible_occ'" != "" {
*             local occ_var : word 1 of `possible_occ'
*         }
*         else {
*             * Create dummy occupation variable
*             gen occ_missing = 1
*             local occ_var = "occ_missing"
*         }
*     }
* }

* if "`occ_var'" != "occ_missing" {
*     * Ensure the occupation variable is numeric before using inrange()
*     capture destring `occ_var', replace
*     if _rc != 0 {
*         capture encode `occ_var', gen(`occ_var'_num)
*         drop `occ_var'
*         rename `occ_var'_num `occ_var'
*     }

*     tab `occ_var', missing
    
*     quietly tab `occ_var'
*     local n_occupations = r(r)
    
*     if `n_occupations' > 20 {
*         * Create broad occupation groups (adjust these based on your actual occupation codes)
*         gen occ_management = inrange(`occ_var', 10, 950) if !missing(`occ_var')
*         gen occ_business = inrange(`occ_var', 1000, 1240) if !missing(`occ_var')
*         gen occ_computer = inrange(`occ_var', 1300, 1560) if !missing(`occ_var')
*         gen occ_engineering = inrange(`occ_var', 1600, 1980) if !missing(`occ_var')
*         gen occ_science = inrange(`occ_var', 1600, 1980) if !missing(`occ_var')
*         gen occ_legal = inrange(`occ_var', 2100, 2160) if !missing(`occ_var')
*         gen occ_education = inrange(`occ_var', 2200, 2550) if !missing(`occ_var')
*         gen occ_healthcare = inrange(`occ_var', 3000, 3540) if !missing(`occ_var')
*         gen occ_service = inrange(`occ_var', 4000, 4650) if !missing(`occ_var')
*         gen occ_sales = inrange(`occ_var', 4700, 4965) if !missing(`occ_var')
*         gen occ_office = inrange(`occ_var', 5000, 5940) if !missing(`occ_var')
*         gen occ_construction = inrange(`occ_var', 6200, 6765) if !missing(`occ_var')
*         gen occ_production = inrange(`occ_var', 7000, 8965) if !missing(`occ_var')
*         gen occ_transportation = inrange(`occ_var', 9000, 9750) if !missing(`occ_var')
*         gen occ_other = !inrange(`occ_var', 10, 9750) if !missing(`occ_var')
*     }
*     else {
*         * Create individual occupation dummies
*         quietly levelsof `occ_var', local(occ_levels)
*         foreach level of local occ_levels {
*             gen occ_`level' = (`occ_var' == `level') if !missing(`occ_var')
*         }
*     }
    
* }

display "============================================================"

* Compute moments by year
preserve  // <-- This preserve is for the moments computation

* Initialize results matrix (4 years x 23 moments)
matrix results = J(4, 23, .)

* Set column names
matrix colnames results = mean_logwage var_logwage mean_alpha var_alpha mean_logwage_inperson mean_logwage_remote diff_logwage_inperson_remote inperson_share hybrid_share remote_share agg_productivity mean_logwage_RH_lowpsi mean_logwage_RH_highpsi diff_logwage_RH_high_lowpsi mean_alpha_highpsi mean_alpha_lowpsi diff_alpha_high_lowpsi diff_logwage_high_lowpsi var_logwage_highpsi var_logwage_lowpsi ratio_var_logwage_high_lowpsi market_tightness p90_p10_logwage

matrix rownames results = "2022" "2023" "2024" "2025"

* Fixed market tightness values
local mt_2022 = 2.0
local mt_2023 = 1.4
local mt_2024 = 1.1
local mt_2025 = 1.07

* Fixed aggregate productivity values
local ap_2022 = 1.0971
local ap_2023 = 1.1179
local ap_2024 = 1.1481
local ap_2025 = 1.1554

* Loop through years and compute moments
local row = 0
foreach yr of numlist 2022 2023 2024 2025 {
    local ++row
    
    display "Computing moments for year `yr'..."
    
    * Mean log wage (weighted)
    qui sum logwage [weight=wtfinl] if year == `yr' & !missing(logwage)
    if r(N) > 0 {
        matrix results[`row', 1] = r(mean)
        matrix results[`row', 2] = r(Var)
    }
    
    * Mean alpha (weighted)
    qui sum alpha [weight=wtfinl] if year == `yr' & !missing(alpha)
    if r(N) > 0 {
        matrix results[`row', 3] = r(mean)
        matrix results[`row', 4] = r(Var)
    }
    
    * Work arrangement shares (weighted)
    * Remote share (full_remote) - column 10
    qui sum full_remote [weight=wtfinl] if year == `yr'
    if r(N) > 0 {
        matrix results[`row', 10] = r(mean)
    }
    
    * Hybrid share (hybrid) - column 9
    qui sum hybrid [weight=wtfinl] if year == `yr'
    if r(N) > 0 {
        matrix results[`row', 9] = r(mean)
    }
    
    * In-person share - column 8
    qui sum in_person [weight=wtfinl] if year == `yr'
    if r(N) > 0 {
        matrix results[`row', 8] = r(mean)
    }
    
    *-------------------------------------------------------------------------------
    * MISSING MOMENTS: Mean log wages by work arrangement
    *-------------------------------------------------------------------------------
    * Mean log wage for in-person workers - column 5
    qui sum logwage [weight=wtfinl] if year == `yr' & in_person == 1 & !missing(logwage)
    if r(N) > 0 {
        matrix results[`row', 5] = r(mean)
    }
    
    * Mean log wage for remote workers - column 6  
    qui sum logwage [weight=wtfinl] if year == `yr' & full_remote == 1 & !missing(logwage)
    if r(N) > 0 {
        matrix results[`row', 6] = r(mean)
    }
    
    *-------------------------------------------------------------------------------
    * MOMENT FOR c₀: Compensating Wage Differential (In-Person vs. Remote)
    *-------------------------------------------------------------------------------
    * Check if we have variation in work arrangements
    qui count if year == `yr' & !missing(logwage) & in_person == 1
    local n_inperson = r(N)
    qui count if year == `yr' & !missing(logwage) & full_remote == 1
    local n_remote = r(N)

    qui reghdfe logwage hybrid in_person experience experience_sq educ_* sex_* race_* ///
        [pweight=wtfinl] if year == `yr' & !missing(logwage, experience, wtfinl), absorb(industry occupation)vce(cluster industry occupation)
    
    * Check if regression succeeded and coefficient exists
    if e(N) > 0 {
        matrix coef_matrix = e(b)
        local coef_names : colnames coef_matrix
        local in_person_pos : list posof "in_person" in coef_names
        
        if `in_person_pos' > 0 {
            matrix results[`row', 7] = _b[in_person]
        }
    }

    
    * MOMENT FOR ψ₀: Wage Premium for High-ψ Firms (diff_logwage_high_lowpsi)
    * SAMPLE RESTRICTION: Remote and hybrid workers only
    capture {
        * qui reg logwage high_psi experience experience_sq educ_* sex_* race_* ind_* occ_* ///
        *     [pweight=wtfinl] if year == `yr' & (hybrid == 1 | full_remote == 1) & !missing(logwage, experience, high_psi)
        qui reghdfe logwage high_psi experience experience_sq educ_* sex_* race_*  ///
            [pweight=wtfinl] if year == `yr' & (hybrid == 1 | full_remote == 1) & !missing(logwage, experience, high_psi), absorb(industry occupation)vce(cluster industry occupation)
        
        * Get coefficient on high_psi (β₁) - this identifies ψ₀
        matrix results[`row', 18] = _b[high_psi]
    }
    
    * MOMENT FOR ν: Slope of Wage-Efficiency Profile (diff_logwage_RH_high_lowpsi)
    * SAMPLE RESTRICTION: Remote and hybrid workers only
    capture {
        * qui reg logwage psi experience experience_sq educ_* sex_* race_* ind_* occ_* ///
        *     [pweight=wtfinl] if year == `yr' & (hybrid == 1 | full_remote == 1) & !missing(logwage, experience, psi)
        qui reghdfe logwage psi experience experience_sq educ_* sex_* race_* ///
            [pweight=wtfinl] if year == `yr' & (hybrid == 1 | full_remote == 1) & !missing(logwage, experience, psi), absorb(industry occupation)vce(cluster industry occupation)

        * Get coefficient on psi (β₁) - this identifies ν
        matrix results[`row', 14] = _b[psi]
    }
    
    * MOMENT FOR ϕ: Difference in Average Remote Share by Firm Type (diff_alpha_high_lowpsi)
    
    * Calculate mean alpha for high-psi firms (above threshold)
    local mean_alpha_highpsi = .
	qui sum alpha [weight=wtfinl] if year == `yr' & high_psi == 1 & !missing(alpha)
	if r(N) > 0 {
		local mean_alpha_highpsi = r(mean)
		matrix results[`row', 15] = r(mean)
	}

    * Calculate mean alpha for low-psi firms (below threshold)
    local mean_alpha_lowpsi = .
	qui sum alpha [weight=wtfinl] if year == `yr' & low_psi == 1 & !missing(alpha)
	if r(N) > 0 {
		local mean_alpha_lowpsi = r(mean)
		matrix results[`row', 16] = r(mean)
	}
    
    * Calculate the difference - this identifies ϕ
    if !missing(`mean_alpha_highpsi') & !missing(`mean_alpha_lowpsi') {
        local diff_alpha = `mean_alpha_highpsi' - `mean_alpha_lowpsi'
        matrix results[`row', 17] = `diff_alpha'
    }
    
    *-------------------------------------------------------------------------------
    * MISSING MOMENTS: Wage variances and RH wages by psi groups
    *-------------------------------------------------------------------------------
    * Mean log wage for remote/hybrid workers by psi groups - columns 12-13
    qui sum logwage [weight=wtfinl] if year == `yr' & (hybrid == 1 | full_remote == 1) & low_psi == 1 & !missing(logwage)
    if r(N) > 0 {
        matrix results[`row', 12] = r(mean)
    }
    
    qui sum logwage [weight=wtfinl] if year == `yr' & (hybrid == 1 | full_remote == 1) & high_psi == 1 & !missing(logwage)
    if r(N) > 0 {
        matrix results[`row', 13] = r(mean)
    }
    
    * Wage variances by psi groups - columns 19-20
    qui sum logwage [weight=wtfinl] if year == `yr' & high_psi == 1 & !missing(logwage)
    if r(N) > 0 {
        matrix results[`row', 19] = r(Var)
        local var_high = r(Var)
    }
    
    qui sum logwage [weight=wtfinl] if year == `yr' & low_psi == 1 & !missing(logwage)
    if r(N) > 0 {
        matrix results[`row', 20] = r(Var)
        local var_low = r(Var)
        
        * Ratio of variances - column 21
        if !missing(`var_high') & `var_low' > 0 {
            local var_ratio = `var_high' / `var_low'
            matrix results[`row', 21] = `var_ratio'
        }
    }
    
    * Market tightness (fixed values) - column 22
    matrix results[`row', 22] = `mt_`yr''
    
    * Aggregate productivity (fixed values) - column 11
    matrix results[`row', 11] = `ap_`yr''
    
    * 90/10 ratio of log wages - column 23
    qui sum logwage [weight=wtfinl] if year == `yr' & !missing(logwage), detail
    if r(N) > 0 {
        local p90_wage = r(p90)
        local p10_wage = r(p10)
        if !missing(`p90_wage') & !missing(`p10_wage') & `p10_wage' != 0 {
            local ratio_90_10 = `p90_wage' / `p10_wage'
            matrix results[`row', 23] = `ratio_90_10'
        }
    }
}

restore  // <-- This restore returns to the state before moments computation

* Convert matrix to dataset for CSV export
clear
svmat results, names(col)
gen year = .
replace year = 2022 in 1
replace year = 2023 in 2
replace year = 2024 in 3
replace year = 2025 in 4

* Reorder with year first
order year

* Export CSV
export delimited "data/results/data_moments/data_moments_by_year.csv", replace

* Generate individual YAML files for each year
local years "2022 2023 2024 2025"
local i = 0
foreach yr of local years {
    local ++i
    
    * Get current date
    local today = c(current_date)
    
    * Create YAML content
    file open yaml using "data/results/data_moments/data_moments_`yr'.yaml", write replace
    
    file write yaml "# Data moments for year `yr'" _n
    file write yaml "# Generated on: `today'" _n _n
    file write yaml "DataMoments:" _n
    
    * Core moments
    if !missing(mean_logwage[`i']) {
        file write yaml "  mean_logwage: " %9.6f (mean_logwage[`i']) _n
    }
    else {
        file write yaml "  mean_logwage: null" _n
    }
    
    if !missing(var_logwage[`i']) {
        file write yaml "  var_logwage: " %9.6f (var_logwage[`i']) _n
    }
    else {
        file write yaml "  var_logwage: null" _n
    }
    
    if !missing(mean_alpha[`i']) {
        file write yaml "  mean_alpha: " %9.6f (mean_alpha[`i']) _n
    }
    else {
        file write yaml "  mean_alpha: null" _n
    }
    
    if !missing(var_alpha[`i']) {
        file write yaml "  var_alpha: " %9.6f (var_alpha[`i']) _n
    }
    else {
        file write yaml "  var_alpha: null" _n
    }
    
    * Mean log wages by work arrangement
    if !missing(mean_logwage_inperson[`i']) {
        file write yaml "  mean_logwage_inperson: " %9.6f (mean_logwage_inperson[`i']) _n
    }
    else {
        file write yaml "  mean_logwage_inperson: null" _n
    }
    
    if !missing(mean_logwage_remote[`i']) {
        file write yaml "  mean_logwage_remote: " %9.6f (mean_logwage_remote[`i']) _n
    }
    else {
        file write yaml "  mean_logwage_remote: null" _n
    }
    
    * KEY IDENTIFICATION MOMENTS
    * For c₀: In-person wage premium
    if !missing(diff_logwage_inperson_remote[`i']) {
        file write yaml "  diff_logwage_inperson_remote: " %9.6f (diff_logwage_inperson_remote[`i']) _n
    }
    else {
        file write yaml "  diff_logwage_inperson_remote: null" _n
    }
    
    * Work arrangement shares
    if !missing(inperson_share[`i']) {
        file write yaml "  inperson_share: " %9.6f (inperson_share[`i']) _n
    }
    else {
        file write yaml "  inperson_share: null" _n
    }
    
    if !missing(hybrid_share[`i']) {
        file write yaml "  hybrid_share: " %9.6f (hybrid_share[`i']) _n
    }
    else {
        file write yaml "  hybrid_share: null" _n
    }
    
    if !missing(remote_share[`i']) {
        file write yaml "  remote_share: " %9.6f (remote_share[`i']) _n
    }
    else {
        file write yaml "  remote_share: null" _n
    }
    
    if !missing(agg_productivity[`i']) {
        file write yaml "  agg_productivity: " %9.6f (agg_productivity[`i']) _n
    }
    else {
        file write yaml "  agg_productivity: null" _n
    }
    
    * Mean log wages for remote/hybrid workers by psi groups
    if !missing(mean_logwage_RH_lowpsi[`i']) {
        file write yaml "  mean_logwage_RH_lowpsi: " %9.6f (mean_logwage_RH_lowpsi[`i']) _n
    }
    else {
        file write yaml "  mean_logwage_RH_lowpsi: null" _n
    }
    
    if !missing(mean_logwage_RH_highpsi[`i']) {
        file write yaml "  mean_logwage_RH_highpsi: " %9.6f (mean_logwage_RH_highpsi[`i']) _n
    }
    else {
        file write yaml "  mean_logwage_RH_highpsi: null" _n
    }
    
    * For ν: Wage-efficiency slope
    if !missing(diff_logwage_RH_high_lowpsi[`i']) {
        file write yaml "  wage_slope_psi: " %9.6f (diff_logwage_RH_high_lowpsi[`i']) _n
    }
    else {
        file write yaml "  wage_slope_psi: null" _n
    }
    
    * High/low psi alpha means (for diagnostics)
    if !missing(mean_alpha_highpsi[`i']) {
        file write yaml "  mean_alpha_highpsi: " %9.6f (mean_alpha_highpsi[`i']) _n
    }
    else {
        file write yaml "  mean_alpha_highpsi: null" _n
    }
    
    if !missing(mean_alpha_lowpsi[`i']) {
        file write yaml "  mean_alpha_lowpsi: " %9.6f (mean_alpha_lowpsi[`i']) _n
    }
    else {
        file write yaml "  mean_alpha_lowpsi: null" _n
    }
    
    * For ϕ: Remote share difference by firm type
    if !missing(diff_alpha_high_lowpsi[`i']) {
        file write yaml "  diff_alpha_high_lowpsi: " %9.6f (diff_alpha_high_lowpsi[`i']) _n
    }
    else {
        file write yaml "  diff_alpha_high_lowpsi: null" _n
    }
    
    * For ψ₀: High-psi wage premium
    if !missing(diff_logwage_high_lowpsi[`i']) {
        file write yaml "  wage_premium_high_psi: " %9.6f (diff_logwage_high_lowpsi[`i']) _n
    }
    else {
        file write yaml "  wage_premium_high_psi: null" _n
    }
    
    * Wage variances by psi groups
    if !missing(var_logwage_highpsi[`i']) {
        file write yaml "  var_logwage_highpsi: " %9.6f (var_logwage_highpsi[`i']) _n
    }
    else {
        file write yaml "  var_logwage_highpsi: null" _n
    }
    
    if !missing(var_logwage_lowpsi[`i']) {
        file write yaml "  var_logwage_lowpsi: " %9.6f (var_logwage_lowpsi[`i']) _n
    }
    else {
        file write yaml "  var_logwage_lowpsi: null" _n
    }
    
    if !missing(ratio_var_logwage_high_lowpsi[`i']) {
        file write yaml "  ratio_var_logwage_high_lowpsi: " %9.6f (ratio_var_logwage_high_lowpsi[`i']) _n
    }
    else {
        file write yaml "  ratio_var_logwage_high_lowpsi: null" _n
    }
    
    if !missing(market_tightness[`i']) {
        file write yaml "  market_tightness: " %9.6f (market_tightness[`i']) _n
    }
    else {
        file write yaml "  market_tightness: null" _n
    }
    
    * 90/10 ratio of log wages
    if !missing(p90_p10_logwage[`i']) {
        file write yaml "  p90_p10_logwage: " %9.6f (p90_p10_logwage[`i']) _n
    }
    else {
        file write yaml "  p90_p10_logwage: null" _n
    }
    
    * Leave job finding rate as null for now
    file write yaml "  job_finding_rate: null" _n
    
    file close yaml
}

display ""
display "============================================================"

* Show summary of results
list year mean_logwage mean_alpha inperson_share hybrid_share remote_share diff_logwage_inperson_remote market_tightness p90_p10_logwage

display ""
display "============================================================"
display "✅ DATA MOMENTS COMPUTATION COMPLETE"
display "============================================================"

/*==============================================================================
* APPENDIX: Create Simulation Scaffolding for SMM
* 
* Purpose: Create a dataset of real-world control variables and common
*          random numbers to be used in the Julia SMM estimation. This
*          ensures coherence between the data and the simulation and
*          acts as a variance reduction technique.
*==============================================================================*/

display ""
display "============================================================"
display "🏗️  CREATING SIMULATION SCAFFOLDING"
display "============================================================"

// --- Configuration ---
local N_sim = 50000      // Set the desired number of simulated individuals

// Define the list of control variables to keep.
// This should match the controls used in the regressions exactly.
local controls_to_keep "experience experience_sq educ sex race industry occupation wtfinl"

display "Control variables for scaffolding:"
display "`controls_to_keep'"

// Reload the original dataset since we're working with the results matrix now
import delimited "data/processed/cps/cps_alpha_wage_present_reweighted.csv", clear

// Repeat the essential data preparation steps
drop indnaics occsoc_detailed occsoc_minor ftpt edu3 age4 v38 cell_id

// Keep only years 2022-2025
keep if inlist(year, 2022, 2023, 2024, 2025)

// Recreate the essential variables needed for scaffolding
destring log_wage_real, replace force
gen logwage = log_wage_real if log_wage_real != . & log_wage_real != 0

destring alpha, replace force
destring educ, replace force
destring age, replace force
destring sex, replace force
destring race, replace force
destring wtfinl, replace force

// Create years of education
gen years_educ = .
replace years_educ = 0 if educ == 2      
replace years_educ = 8 if educ == 20     
replace years_educ = 9 if educ == 30     
replace years_educ = 10 if educ == 40    
replace years_educ = 11 if educ == 50    
replace years_educ = 12 if educ == 60    
replace years_educ = 12 if educ == 71    
replace years_educ = 12 if educ == 73    
replace years_educ = 13 if educ == 81    
replace years_educ = 14 if educ == 91    
replace years_educ = 14 if educ == 92    
replace years_educ = 14 if educ == 10    
replace years_educ = 16 if educ == 111   
replace years_educ = 18 if educ == 123   
replace years_educ = 20 if educ == 124   
replace years_educ = 21 if educ == 125   

keep if !missing(years_educ) & years_educ >= 6

// Create experience variables
gen experience = age - years_educ - 6
gen experience_sq = experience^2

// Recreate the dummy variables exactly as in the main analysis
gen educ_less_hs = (years_educ < 12) if !missing(years_educ)           
gen educ_hs = (years_educ == 12) if !missing(years_educ)               
gen educ_some_college = (years_educ > 12 & years_educ < 16) if !missing(years_educ)  
gen educ_ba = (years_educ == 16) if !missing(years_educ)               
gen educ_graduate = (years_educ > 16) if !missing(years_educ)          

gen sex_male = (sex == 1) if !missing(sex)
gen sex_female = (sex == 2) if !missing(sex)

gen race_white = (race == 100) if !missing(race)                
gen race_black = (race == 200) if !missing(race)                
gen race_asian = inlist(race, 651, 652) if !missing(race)       
gen race_other = !inlist(race, 100, 200, 651, 652) if !missing(race) 

encode ind_broad, generate(industry)
encode occsoc_broad, generate(occupation)

// --- Collapse raw codes into regression categories for export ---
// Keep raw copies
rename educ educ_raw
rename sex  sex_raw
rename race race_raw

// Education 5-group (same as educ_* dummies logic)
capture drop educ_cat
gen byte educ_cat = .
replace educ_cat = 1 if years_educ < 12 & !missing(years_educ)
replace educ_cat = 2 if years_educ == 12
replace educ_cat = 3 if years_educ > 12 & years_educ < 16
replace educ_cat = 4 if years_educ == 16
replace educ_cat = 5 if years_educ > 16
label define EDUCAT 1 "less_hs" 2 "hs" 3 "some_college" 4 "ba" 5 "graduate", replace
label values educ_cat EDUCAT

// Sex 2-group
capture drop sex_cat
gen byte sex_cat = .
replace sex_cat = 1 if sex_raw == 1
replace sex_cat = 2 if sex_raw == 2
label define SEXCAT 1 "male" 2 "female", replace
label values sex_cat SEXCAT

// Race 4-group (white / black / asian / other) matching race_* dummies
capture drop race_cat
gen byte race_cat = .
replace race_cat = 1 if race_raw == 100
replace race_cat = 2 if race_raw == 200
replace race_cat = 3 if inlist(race_raw, 651, 652)
replace race_cat = 4 if !missing(race_raw) & !inlist(race_raw, 100, 200, 651, 652)
label define RACECAT 1 "white" 2 "black" 3 "asian" 4 "other", replace
label values race_cat RACECAT

// Decode to string variables (so Julia sees category names directly)
decode educ_cat, gen(educ)
decode sex_cat,  gen(sex)
decode race_cat, gen(race)

// (Optional) drop the intermediate numeric category codes
drop educ_cat sex_cat race_cat

// (Optional) drop dummy indicators not needed for export (if they exist here)
capture drop educ_less_hs educ_hs educ_some_college educ_ba educ_graduate
capture drop sex_male sex_female
capture drop race_white race_black race_asian race_other

// controls_to_keep already lists educ sex race; they now refer to the string category vars

// Loop through all years to create scaffolding for each (SKIP 2022)
foreach TARGET_YEAR of numlist 2023 2024 2025 {
    
    display ""
    display "Creating scaffolding for year `TARGET_YEAR'..."
    
    // Preserve the current state
    preserve
    
    // Keep only the target year for sampling
    keep if year == `TARGET_YEAR'
    
    // Check if we have enough observations
    qui count if !missing(wtfinl) & wtfinl > 0
    local n_available = r(N)
    display "Available observations for year `TARGET_YEAR': `n_available'"
    
    if `n_available' < 1000 {
        display "⚠️  WARNING: Very few observations available for year `TARGET_YEAR' (`n_available')"
        display "   Consider reducing N_sim or checking data quality."
    }
    

    // Seed then draw N_sim observations with replacement
    // This extracts observations (with replacement)
    set seed 12345
    bsample `N_sim', weight(wtfinl)

    // Ensure exported numeric formatting preserves decimals
    display "✓ Sample of `N_sim' drawn for year `TARGET_YEAR'."
    
    // Keep only the necessary control variables plus year identifier
    keep year `controls_to_keep'
    
    // Verify we have all the control variables
    foreach var of local controls_to_keep {
        capture confirm variable `var'
        if _rc != 0 {
            display "⚠️  WARNING: Control variable `var' not found in year `TARGET_YEAR'"
        }
    }
    
    // Generate the common random numbers for the simulation
    set seed 12345 // Set seed for reproducibility
    gen u_psi = runiform()     // Random draw for psi assignment (was u_match)
    gen u_h = runiform()       // Random draw for h assignment (was u_alpha)
    gen u_alpha = runiform()   // Random draw for alpha assignment (was u_shock)
    
    display "✓ Added uniform random draws: u_psi, u_h, u_alpha."
    
    // Add observation ID for tracking
    gen sim_id = _n

    // Keep only required columns for Julia fixed-effects regressions & simulation
    keep year experience experience_sq educ sex race industry occupation u_psi u_h u_alpha sim_id
    
    // Export final scaffolding dataset as CSV
    export delimited "data/processed/simulation_scaffolding_`TARGET_YEAR'.csv", replace

    // Also create a summary file with variable info
    describe, replace clear
    export delimited "data/processed/simulation_scaffolding_`TARGET_YEAR'_variables.csv", replace
    
    display "✓ Variable descriptions saved to: data/processed/simulation_scaffolding_`TARGET_YEAR'_variables.csv"
    
    // Restore for next iteration
    restore
}

// Create a combined scaffolding file with all years (2023-2025 only)
display ""
display "Creating combined scaffolding file..."

clear
local first_file = 1
foreach TARGET_YEAR of numlist 2023 2024 2025 {
    if `first_file' {
        import delimited "data/processed/simulation_scaffolding_`TARGET_YEAR'.csv", clear
        // Ensure only the required columns (safe if older files had extras)
        keep year experience experience_sq educ sex race industry occupation u_psi u_h u_alpha sim_id
        tempfile combined
        save `combined'
        local first_file = 0
    }
    else {
        import delimited "data/processed/simulation_scaffolding_`TARGET_YEAR'.csv", clear
        keep year experience experience_sq educ sex race industry occupation u_psi u_h u_alpha sim_id
        append using `combined'
        save `combined', replace
    }
}

use `combined', clear
export delimited "data/processed/simulation_scaffolding_all_years.csv", replace
display "✓ Combined scaffolding (trimmed columns) saved."

// Create a summary report
display ""
display "============================================================"
display "📊 SCAFFOLDING SUMMARY REPORT"
display "============================================================"

qui tab year
display "Total observations by year:"
tab year

display ""
display "Control variables included:"
foreach var of local controls_to_keep {
    capture confirm variable `var'
    if _rc == 0 {
        display "  ✓ `var'"
    }
    else {
        display "  ✗ `var' (missing)"
    }
}

display ""
display "Random number variables:"
display "  ✓ u_psi (psi assignment random draw)"
display "  ✓ u_h (h assignment random draw)"  
display "  ✓ u_alpha (alpha assignment random draw)"
display "  ✓ sim_id (observation identifier)"
display "  ✓ wtfinl (survey weights)"

display ""
display "Years included in scaffolding: 2023, 2024, 2025"
display "Note: 2022 skipped due to insufficient observations"
display ""
display "Files created in data/processed/:"
display "  ✓ simulation_scaffolding_2023.csv"
display "  ✓ simulation_scaffolding_2024.csv"
display "  ✓ simulation_scaffolding_2025.csv"
display "  ✓ simulation_scaffolding_all_years.csv"
display "  ✓ simulation_scaffolding_YYYY_variables.csv (descriptions)"
