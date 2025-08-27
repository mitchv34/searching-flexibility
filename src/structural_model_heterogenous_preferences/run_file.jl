using Pkg
project_root = joinpath(@__DIR__, "..", "..")
Pkg.activate(project_root)
Pkg.instantiate()

# using Distributed
# addprocs(9)

const ROOT = @__DIR__
# @everywhere begin
#     using Random, Statistics, SharedArrays, ForwardDiff
#     using Optimization, OptimizationOptimJL, Optim
#     include(joinpath($ROOT, "ModelSetup.jl"))
#     include(joinpath($ROOT, "ModelSolver.jl"))
#     include(joinpath($ROOT, "ModelEstimation.jl"))
# end
using Random, Statistics
# Removed Term to avoid constant redefinition issues in distributed runs
using Printf
using CairoMakie
using PrettyTables

include(joinpath(ROOT, "ModelPlotting.jl"))
include(joinpath(ROOT, "ModelSetup.jl"))
include(joinpath(ROOT, "ModelSolver.jl"))
include(joinpath(ROOT, "ModelEstimation.jl"))

# Configuration
const SCRIPT_DIR = dirname(@__FILE__)
const RESULTS_DIR = joinpath(SCRIPT_DIR, "distributed_mpi_search", "output",  "results")
const OUTPUT_DIR = joinpath(SCRIPT_DIR, "distributed_mpi_search", "figures")
const PLOTS_DIR = OUTPUT_DIR

# Create output directories
mkpath(OUTPUT_DIR)
mkpath(PLOTS_DIR)


parameters = ["aₕ", "bₕ", "c₀", "μ", "χ", "A₀", "A₁", "ν", "ψ₀", "ϕ", "κ₀", "ξ"]
moments_to_use = [
    # --- Skill Distribution Parameters (aₕ, bₕ) ---
    # - mean_logwage
      "var_logwage",
      "mean_alpha",
      "var_alpha",
      "inperson_share",
      "remote_share",
      "p90_p10_logwage",
      # - agg_productivity
      "diff_alpha_high_lowpsi",
      "wage_premium_high_psi",
      # - wage_slope_psi
      "market_tightness",
      "wage_alpha",
      "wage_alpha_curvature"
]


# Load and parse the JSON file
results_file = "/project/high_tech_ind/searching-flexibility/src/structural_model_heterogenous_preferences/distributed_mpi_search/output/results/mpi_search_results_job3054090_latest.json"
results = JSON3.read( read(results_file, String) )
config = "/project/high_tech_ind/searching-flexibility/src/structural_model_heterogenous_preferences/distributed_mpi_search/mpi_search_config_test.yaml"
best_params = Dict(Symbol(k) => Float64(v) for (k, v) in results[:best_params])
simulated_data = "/project/high_tech_ind/searching-flexibility/data/processed/simulation_scaffolding_2024.feather"
begin
    # Modify some parameters
    prim, res = initializeModel(config);
    prim, res = update_primitives_results(prim, res, best_params)
    new_params = Dict(
        # # --- TECHNOLOGY: Unleash the sorting engines ---
        :ψ₀ => 0.125,  # A high baseline to make remote work generally viable
        :ϕ  => 1.8,  # CRITICAL: A very high skill-remote complementarity
        :ν  => 2.0,  # CRITICAL: A very high return to firm technology

        # # --- PREFERENCES: Allow for clear choices ---
        :c₀ => 0.03,  # A moderate amenity value
        :μ  => 0.05, # Low noise to make choices sharp
        # :χ  => 1.0,  # High curvature to encourage corner solutions

        # # --- SKILL: A more disperse and skilled workforce ---
        :aₕ => 1.0,
        :bₕ => 1.0,

        :ξ  => 0.2,

        # # --- MACRO & SEARCH: Start with a plausible baseline ---
        # # We will need to re-calibrate these after we see the effect of the new tech params.
        # :A₀ => 0.0,
        # :A₁ => 1.0,
        :κ₀ => 115.0 # Start with a moderately high vacancy cost
    )

    prim, res = update_primitives_results(prim, res, new_params)
    solve_model(prim, res, verbose=false, λ_S_init = 0.01, λ_u_init = 0.01, tol = 1e-8, max_iter = 25_000)
    # Print all parameters
    for p in Symbol.(parameters)
        println("Parameter: ", p, " = ", getfield(prim, p))
    end
    println()
    # Compute model moments
    model_moments = compute_model_moments(prim, res, simulated_data)

    # Load data moments from YAML (explicit path for 2024)
    data_moments_path = joinpath(ROOT, "..", "..", "data", "results", "data_moments", "data_moments_2024.yaml")
    data_moments_raw = load_moments_from_yaml(data_moments_path)

    # Prepare ordered list (use intersection of keys to avoid noise)
    exclude_keys = Set([:degeneracy_flag, :degeneracy_issues_count])
    desired_keys = Symbol.(moments_to_use)
    filtered_keys = [k for k in desired_keys if k ∈ keys(model_moments) && k ∉ exclude_keys]

    header = ["Moment", "Data", "Model", "Diff"]
    rows = Vector{Vector{Any}}()
    for k in filtered_keys
        model_val = model_moments[k]
        data_val = haskey(data_moments_raw, k) ? data_moments_raw[k] : missing
        data_val_num = (data_val isa Number) ? Float64(data_val) : missing
        diff_val = ismissing(data_val_num) ? missing : (model_val - data_val_num)
        push!(rows, Any[
            String(k),
            ismissing(data_val_num) ? missing : round(data_val_num, digits=3),
            round(model_val, digits=3),
            ismissing(diff_val) ? missing : round(diff_val, digits=3)
        ])
    end
    # Convert to matrix for PrettyTables (heterogeneous types allowed as Any)
    table_matrix = reduce(vcat, (reshape(r, 1, :) for r in rows))
    println("\nMoments (rounded to 3 decimals) vs Data 2024:")
    pretty_table(table_matrix; header=header, formatters = ft_printf("%.3f"))
end


# Use plotting helpers in ModelPlotting for consistency
s_flow = calculate_logit_flow_surplus_with_curvature(prim)
fig_s1, fig_s2, fig_s3, fig_s4 =plot_s_flow_diagnostics(s_flow, prim)
fig_s1 |> display

# Display core diagnostics
fig_s2 |> display
fig_s3 |> display
fig_s4 |> display


# Solution diagnostics
# # --- Generate diagnostic plots (integration with ModelPlotting) ---
# # Employment distribution heatmap
fig_emp =plot_employment_distribution(res, prim)
# # Employment distribution with marginals
fig_emp_marg =plot_employment_distribution_with_marginals(res, prim)

fig_surplus = plot_surplus_function(res, prim)

# fig_alpha =plot_alpha_policy(res, prim)
fig_alpha = plot_avg_alpha(prim, res)

# fig_wage_pol =plot_wage_policy(res, prim)
fig_wage_pol = plot_avg_wage(prim, res)

# fig_wage_amenity =plot_wage_amenity_tradeoff(res, prim)

# fig_outcome_skill =plot_outcomes_by_skill(res, prim)

# fig_work_arrangement =plot_work_arrangement_regimes(res, prim)

# fig_work_arrangement_viable =plot_work_arrangement_regimes(res, prim, gray_nonviable=true)

# fig_alpha_by_firm =plot_alpha_policy_by_firm_type(res, prim)

save(joinpath(ROOT, "temp", "fig_s1.png"), fig_s1)
save(joinpath(ROOT, "temp", "fig_s2.png"), fig_s2)
save(joinpath(ROOT, "temp", "fig_s3.png"), fig_s3)
save(joinpath(ROOT, "temp", "fig_s4.png"), fig_s4)
save(joinpath(ROOT, "temp", "fig_emp.png"), fig_emp)
save(joinpath(ROOT, "temp", "fig_emp_marg.png"), fig_emp_marg)
save(joinpath(ROOT, "temp", "fig_surplus.png"), fig_surplus)
save(joinpath(ROOT, "temp", "fig_alpha.png"), fig_alpha)
save(joinpath(ROOT, "temp", "fig_wage_pol.png"), fig_wage_pol)
save(joinpath(ROOT, "temp", "fig_outcome_skill.png"), fig_outcome_skill)
# save(joinpath(ROOT, "temp", "fig_work_arrangement.png"), fig_work_arrangement)
# save(joinpath(ROOT, "temp", "fig_work_arrangement_viable.png"), fig_work_arrangement_viable)
# save(joinpath(ROOT, "temp", "fig_alpha_by_firm.png"), fig_alpha_by_firm)
