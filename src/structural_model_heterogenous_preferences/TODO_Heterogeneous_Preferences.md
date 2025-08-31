# To-Do List: Heterogeneous Preferences Model

## COMPLETED ITEMS (✅ Done)

### 1. Update Model Setup & Parameters ✅

-   [x] Add New Parameter `k` (Gamma shape parameter) to `Primitives` in `ModelSetup.jl` (line 28).
-   [x] Add `z_dist::Gamma` field to `Primitives` struct (line 29).
-   [x] Add `k: 15.0` to `model_parameters.yaml`.
-   [x] Update `create_primitives_from_yaml` to read `k` parameter (line 201).
-   [x] Update `validated_Primitives` call to include `k` and `z_dist=Gamma(k, 1.0)` (lines 229-230).
-   [x] Update `validate!` function to check `k > 0` (line 77).
-   [x] Update `update_primitives_results` in `ModelEstimation.jl` to accept and propagate `k` (line 38).

### 2. Modify the Core Model Solver ✅

-   [x] Add `using Expectations, Distributions` imports to `ModelSolver.jl`.
-   [x] Implement `calculate_expected_flow_surplus` function (lines 238-253) that integrates over z \~ Gamma(k,1).
-   [x] Define helper functions: `psi_top_of_h`, `psi_bottom_of_h_z`, `optimal_alpha_given_z` (lines 306-351).
-   [x] Replace deterministic `s_flow` calculation with `calculate_expected_flow_surplus(prim)` (line 70).
-   [x] Implement `calculate_average_policies` function (lines 6-31) for computing expected α and wage policies.

### 3. Results Structure Simplification ✅

-   [x] Simplified `Results` struct constructor to not pre-compute policies (lines 117-133).
-   [x] Policies (`α_policy`, `w_policy`) are computed on-demand in moment calculation.

## CRITICAL MISSING ITEMS (❌ Must Fix)

### 1. **CRITICAL: Complete var_alpha Calculation**

-   [x] **BUG fixed**: `compute_model_moments` now accumulates E\[α\] and E\[α²\] and computes `var_alpha = E[α²] - (E[α])²`.
    -   Code changes: added `sum_alpha_sq` accumulation and included `:var_alpha` in the returned `full` moments map (see `ModelEstimation.jl`).

### 2. **CRITICAL: Fix Policy Computation in Moments**

-   [x] **BUG fixed**: `compute_model_moments` now calls the expected-policy helper (integrates over z) and uses the returned expected α and w policies (`expected_policy_grids` / `calculate_average_policies`) when aggregating moments.
    -   Code changes: `ModelSolver.jl` implements the expected-policy integration; `ModelEstimation.jl` uses it to compute moments.

### 3. **Update Functions Type Safety**

-   [x] `update_primitives_results` now mirrors ASCII/unicode parameter keys, constructs `z_dist` from `k` when needed, and preserves grid recomputation for `h` when `a_h`/`b_h` change. Full AD-friendly type promotion still needs a pass.

    -   Files: `ModelEstimation.jl` (key normalization, z_dist creation).

## MISSING ESTIMATION INFRASTRUCTURE (📋 For Full Parity)

### 4. **Missing Test Suite**

-   [ ] Copy `tests/` folder from `structural_model_new` to `structural_model_heterogenous_preferences`.
-   [ ] Tests needed: `moments_vs_parameters.jl`, `objective_function_profile.jl`, `single_parameter_estimation.jl`.
-   [ ] Modify tests to include `k` parameter and `var_alpha` moment.

### 5. **Missing Parameter Profiling**

-   [ ] Copy `profile_run.jl` from `structural_model_new`.
-   [ ] Add parameter profiling for `k` parameter to verify identification of `var_alpha`.

### 6. **Enhanced Estimation Functions**

-   [ ] The `structural_model_new` has more comprehensive estimation functions (lines 698-841 in ModelEstimation.jl).
-   [ ] Missing advanced optimization setup, warm starting, and solver state management.
-   [ ] Consider copying if advanced estimation capabilities are needed.

### 7. **Moment-Based Estimation Strategy**

-   [ ] Set target moments to include E\[α\] (to identify c₀) and Var(α) (to identify k).
-   [ ] Add routines to compute empirical E\[α\] and Var(α) from dataset.
-   [ ] Update estimation scripts to use these moments for parameter identification.
-   [x] Empirical moment helper implemented: `compute_empirical_alpha_moments(path; ...)` now reads CSV and returns mean & var.
-   \[\~\] Wiring empirical moments into the estimation objective is available via `objective_function` once you supply `target_moments`. A small helper to build the estimation payload from a sweep or dataset is recommended next.

## DIAGNOSTIC & VALIDATION TASKS

### 8. **Model Verification**

-   [ ] Verify `s_flow` matrix from `calculate_expected_flow_surplus` is smooth and AD-friendly.
-   [ ] Test that varying `k` produces sensible changes in `var_alpha`.
-   [ ] Test that varying `c₀` produces sensible changes in `mean_alpha`.

### 9. **Parameter Sweep Diagnostics**

-   [ ] Generate plots of E\[α\] and Var(α) vs. c₀ and k.
-   [ ] Verify smooth, monotonic relationships.
-   [ ] Generate objective function profiles for all parameters to confirm identification.

## DELTA: Work completed in this session

-   Implemented expected-policy integration over z and wired it into moment calculation (`ModelSolver.jl`, `ModelEstimation.jl`).
-   Fixed `var_alpha` computation and included it in the returned moments (`ModelEstimation.jl`).
-   Added empirical moments helper `compute_empirical_alpha_moments` (`ModelEstimation.jl`).
-   Added coarse parameter sweep script `tests/parameter_sweep_k_c0.jl` and produced `tests/sweep_k_c0.csv`.
-   Added analysis script `tests/analysis_sweep_k_c0.jl` and `load_sweep_targets` to produce heatmaps and elasticities.

Files touched/added (representative): - `src/structural_model_heterogenous_preferences/ModelSolver.jl` - `src/structural_model_heterogenous_preferences/ModelEstimation.jl` - `src/structural_model_heterogenous_preferences/tests/parameter_sweep_k_c0.jl` - `src/structural_model_heterogenous_preferences/tests/analysis_sweep_k_c0.jl` - `src/structural_model_heterogenous_preferences/tests/sweep_k_c0.csv`

## Next recommended fixes (short list)

-   Finalize AD-friendly type handling in `update_primitives_results` (important if you plan to use ForwardDiff over parameters).
-   Wire `compute_empirical_alpha_moments` into the estimation payload builder (helper to create `p` for `objective_function`).
-   Add small unit tests for `load_sweep_targets`, `compute_empirical_alpha_moments`, and `compute_model_moments` (happy path + 1-2 edge cases).

## PRIORITY ORDER FOR IMMEDIATE FIXES

### **MUST FIX IMMEDIATELY (Model is broken without these)**:

1.  **Fix `var_alpha` calculation** - model claims to compute it but doesn't
2.  **Fix policy computation** - using zeros instead of actual expected policies makes all moments meaningless

### **SHOULD FIX SOON (For robust estimation)**:

3.  Enhance `update_primitives_results` function for type safety
4.  Add comprehensive test suite
5.  Add parameter profiling capabilities

### **NICE TO HAVE (For full feature parity)**:

6.  Copy advanced estimation infrastructure
7.  Add sophisticated diagnostic plots

## CURRENT STATUS

The heterogeneous preferences model has made significant progress on the core mathematical framework (preference shocks, integration, analytical solutions) but has **critical bugs** in the moment calculation that make it unsuitable for estimation until fixed. The model architecture is sound, but the moment computation returns meaningless values due to using zeros for policies and not actually computing the variance of alpha despite claiming to do so.