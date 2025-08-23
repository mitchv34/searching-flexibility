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
local HIGH_PSI_PERCENTILE = 90  // Define high-psi firms as top 10% (90th percentile and above)

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

* Create psi threshold for high/low psi classification using single percentile
* FIXED: More explicit threshold calculation
capture drop psi_h_limit high_psi low_psi

* Calculate 90th percentile threshold by year more explicitly
gen psi_h_limit = .
forvalues yr = 2022/2025 {
    qui sum psi if year == `yr', detail
    replace psi_h_limit = r(p90) if year == `yr'
    display "Year `yr': 90th percentile threshold = " %8.4f r(p90)
}

* Generate high_psi and low_psi dummies based on the defined threshold
gen high_psi = (psi >= psi_h_limit) if !missing(psi) & !missing(psi_h_limit)
gen low_psi = (psi < psi_h_limit) if !missing(psi) & !missing(psi_h_limit)

* Update high_psi_flag to match our new high_psi definition for consistency
replace high_psi_flag = high_psi if !missing(psi)

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

* Create work arrangement indicators (making sure they're numeric)
destring full_inperson, replace force
destring hybrid, replace force
destring full_remote, replace force

* Create consistent variable names
gen in_person = full_inperson

* Verify work arrangement categories are mutually exclusive
gen work_cat_sum = in_person + hybrid + full_remote
assert work_cat_sum == 1

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
preserve

* Initialize results matrix (4 years x 22 moments)
matrix results = J(4, 22, .)

* Set column names
matrix colnames results = mean_logwage var_logwage mean_alpha var_alpha mean_logwage_inperson mean_logwage_remote diff_logwage_inperson_remote inperson_share hybrid_share remote_share agg_productivity mean_logwage_RH_lowpsi mean_logwage_RH_highpsi diff_logwage_RH_high_lowpsi mean_alpha_highpsi mean_alpha_lowpsi diff_alpha_high_lowpsi diff_logwage_high_lowpsi var_logwage_highpsi var_logwage_lowpsi ratio_var_logwage_high_lowpsi market_tightness

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
        [pweight=wtfinl] if year == `yr' & !missing(logwage, experience, wtfinl), absorb(industry occupation) vce(robust)
    
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
            [pweight=wtfinl] if year == `yr' & (hybrid == 1 | full_remote == 1) & !missing(logwage, experience, high_psi), absorb(industry occupation) vce(robust)
        
        * Get coefficient on high_psi (β₁) - this identifies ψ₀
        matrix results[`row', 18] = _b[high_psi]
    }
    
    * MOMENT FOR ν: Slope of Wage-Efficiency Profile (diff_logwage_RH_high_lowpsi)
    * SAMPLE RESTRICTION: Remote and hybrid workers only
    capture {
        * qui reg logwage psi experience experience_sq educ_* sex_* race_* ind_* occ_* ///
        *     [pweight=wtfinl] if year == `yr' & (hybrid == 1 | full_remote == 1) & !missing(logwage, experience, psi)
        qui reghdfe logwage psi experience experience_sq educ_* sex_* race_* ///
            [pweight=wtfinl] if year == `yr' & (hybrid == 1 | full_remote == 1) & !missing(logwage, experience, psi), absorb(industry occupation) vce(robust)

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
}

restore

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
    
    * Leave job finding rate as null for now
    file write yaml "  job_finding_rate: null" _n
    
    file close yaml
}

display ""
display "============================================================"

* Show summary of results
list year mean_logwage mean_alpha inperson_share hybrid_share remote_share diff_logwage_inperson_remote market_tightness

