/*==============================================================================
* estimate_gamma1.do
*
* Purpose: Estimate the matching function elasticity (gamma_1) using the
*          Hall & Schulhofer-Wohl (2018) methodology.
*
* Inputs:  data/processed/matching_function_data_panel_bls_unified.csv
* Outputs: An estimate for gamma_1 saved to data/results/calibrated_parameters.yaml
*==============================================================================*/

clear all
set more off

* Set working directory to project root
cd "/project/high_tech_ind/searching-flexibility"

* Create results directory if it doesn't exist
capture mkdir "data/results"

* Load the dataset you just created
import delimited "data/processed/matching_function_data_panel_bls_unified.csv", clear

* --- Prepare data for panel regression ---

* Create a numeric state ID
encode state, gen(state_id)

* Parse the date variable (note: Stata converts variable names to lowercase)
* The date comes in as YYYY-MM-DD format from pandas
gen date_parsed = date(date, "YMD")
format date_parsed %td

* Create month-year variable for time fixed effects
gen month_year = mofd(date_parsed)
format month_year %tm

* Create year variable for sample splitting
gen year = year(date_parsed)

* Declare the data as a panel (cross-section: state, time: month-year)
xtset state_id month_year

* --- Save results to YAML file ---
file open yamlfile using "data/results/calibrated_parameters.yaml", write replace text

* Write YAML header
file write yamlfile "# Calibrated Parameters" _n
file write yamlfile "# Generated on: " "`c(current_date)'" " at " "`c(current_time)'" _n
file write yamlfile "# Source: estimate_gamma1.do" _n
file write yamlfile "" _n

* --- 1. Full Sample Regression ---
display " "
display as txt "=== FULL SAMPLE ESTIMATION ==="
display as txt "Running panel fixed-effects regression (full sample)..."

xtreg ln_f ln_theta i.month_year, fe vce(robust)

* Extract coefficients for full sample
scalar coeff_ln_theta_full = _b[ln_theta]
scalar gamma_1_full = 1 - coeff_ln_theta_full
scalar se_ln_theta_full = _se[ln_theta]
scalar se_gamma_1_full = se_ln_theta_full

display as txt "Full sample - Coefficient on ln_theta: " as result %9.4f coeff_ln_theta_full
display as txt "Full sample - Estimated gamma_1: " as result %9.4f gamma_1_full

* Write full sample results to YAML
file write yamlfile "# Matching function elasticity parameter (full sample)" _n
file write yamlfile "γ₁:" _n
file write yamlfile "  value: " %9.6f (gamma_1_full) _n
file write yamlfile "  standard_error: " %9.6f (se_gamma_1_full) _n
file write yamlfile "  description: 'Matching function elasticity parameter from Hall & Schulhofer-Wohl (2018) methodology'" _n
file write yamlfile "  source: 'Estimated from BLS state-level panel data (full sample)'" _n
file write yamlfile "" _n
file write yamlfile "  # Auxiliary regression information" _n
file write yamlfile "  regression_info:" _n
file write yamlfile "    ln_theta_coefficient: " %9.6f (coeff_ln_theta_full) _n
file write yamlfile "    ln_theta_se: " %9.6f (se_ln_theta_full) _n
file write yamlfile "    r_squared_within: " %9.6f (e(r2_w)) _n
file write yamlfile "    n_observations: " %9.0f (e(N)) _n
file write yamlfile "    n_groups: " %9.0f (e(N_g)) _n
file write yamlfile "    relationship: 'gamma_1 = 1 - coefficient_on_ln_theta'" _n
file write yamlfile "" _n

* --- 2. Pre-COVID Sample (excluding 2020-2021) ---
display " "
display as txt "=== PRE-COVID for ROBUSTNESS CHECK ==="
display as txt "Running panel fixed-effects regression (pre-COVID: excluding 2020-2021)..."

xtreg ln_f ln_theta i.month_year if year < 2020, fe vce(robust)

* Extract coefficients for pre-COVID sample
scalar coeff_ln_theta_pre = _b[ln_theta]
scalar gamma_1_pre = 1 - coeff_ln_theta_pre
scalar se_ln_theta_pre = _se[ln_theta]
scalar se_gamma_1_pre = se_ln_theta_pre

display as txt "Pre-COVID - Coefficient on ln_theta: " as result %9.4f coeff_ln_theta_pre
display as txt "Pre-COVID - Estimated gamma_1: " as result %9.4f gamma_1_pre

* Write pre-COVID results to YAML
file write yamlfile "# Robustness check: Pre-COVID sample (excluding 2020-2021)" _n
file write yamlfile "pre_γ₁:" _n
file write yamlfile "  value: " %9.6f (gamma_1_pre) _n
file write yamlfile "  standard_error: " %9.6f (se_gamma_1_pre) _n
file write yamlfile "  description: 'Matching function elasticity parameter (pre-COVID robustness check)'" _n
file write yamlfile "  source: 'Estimated from BLS state-level panel data (2015-2019)'" _n
file write yamlfile "" _n
file write yamlfile "  # Auxiliary regression information" _n
file write yamlfile "  regression_info:" _n
file write yamlfile "    ln_theta_coefficient: " %9.6f (coeff_ln_theta_pre) _n
file write yamlfile "    ln_theta_se: " %9.6f (se_ln_theta_pre) _n
file write yamlfile "    r_squared_within: " %9.6f (e(r2_w)) _n
file write yamlfile "    n_observations: " %9.0f (e(N)) _n
file write yamlfile "    n_groups: " %9.0f (e(N_g)) _n
file write yamlfile "    sample_period: '2015-2019'" _n
file write yamlfile "    relationship: 'gamma_1 = 1 - coefficient_on_ln_theta'" _n
file write yamlfile "" _n

* --- 3. Post-COVID Sample (2022 onwards) ---
display " "
display as txt "=== POST-COVID for ROBUSTNESS CHECK ==="
display as txt "Running panel fixed-effects regression (post-COVID: 2022 onwards)..."

xtreg ln_f ln_theta i.month_year if year >= 2022, fe vce(robust)

* Extract coefficients for post-COVID sample
scalar coeff_ln_theta_post = _b[ln_theta]
scalar gamma_1_post = 1 - coeff_ln_theta_post
scalar se_ln_theta_post = _se[ln_theta]
scalar se_gamma_1_post = se_ln_theta_post

display as txt "Post-COVID - Coefficient on ln_theta: " as result %9.4f coeff_ln_theta_post
display as txt "Post-COVID - Estimated gamma_1: " as result %9.4f gamma_1_post

* Write post-COVID results to YAML
file write yamlfile "# Robustness check: Post-COVID sample (2022 onwards)" _n
file write yamlfile "post_γ₁:" _n
file write yamlfile "  value: " %9.6f (gamma_1_post) _n
file write yamlfile "  standard_error: " %9.6f (se_gamma_1_post) _n
file write yamlfile "  description: 'Matching function elasticity parameter (post-COVID robustness check)'" _n
file write yamlfile "  source: 'Estimated from BLS state-level panel data (2022-2025)'" _n
file write yamlfile "" _n
file write yamlfile "  # Auxiliary regression information" _n
file write yamlfile "  regression_info:" _n
file write yamlfile "    ln_theta_coefficient: " %9.6f (coeff_ln_theta_post) _n
file write yamlfile "    ln_theta_se: " %9.6f (se_ln_theta_post) _n
file write yamlfile "    r_squared_within: " %9.6f (e(r2_w)) _n
file write yamlfile "    n_observations: " %9.0f (e(N)) _n
file write yamlfile "    n_groups: " %9.0f (e(N_g)) _n
file write yamlfile "    sample_period: '2022-2025'" _n
file write yamlfile "    relationship: 'gamma_1 = 1 - coefficient_on_ln_theta'" _n

file close yamlfile

* --- Summary Results ---
display " "
display as txt "=== SUMMARY OF RESULTS ==="
display as txt "Full sample gamma_1:     " as result %9.4f gamma_1_full " (SE: " %9.4f se_gamma_1_full ")"
display as txt "Pre-COVID gamma_1:       " as result %9.4f gamma_1_pre " (SE: " %9.4f se_gamma_1_pre ")"
display as txt "Post-COVID gamma_1:      " as result %9.4f gamma_1_post " (SE: " %9.4f se_gamma_1_post ")"

display " "
display as txt "Results saved to: data/results/calibrated_parameters.yaml"
display as txt "Estimation complete!"