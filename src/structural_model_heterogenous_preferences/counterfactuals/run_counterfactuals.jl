# Main script for running counterfactual experiments
# src/structural_model_heterogenous_preferences/counterfactuals/run_counterfactuals.jl

using Pkg
Pkg.activate("../../..") # Activate project at repo root
Pkg.instantiate()

using YAML, DataFrames, Printf, CSV
using Plots, PlotlyJS
plotlyjs()

# --- Project Setup ---
const ROOT = joinpath(@__DIR__, "..", "..", "..")
include(joinpath(ROOT, "src", "structural_model_heterogenous_preferences", "ModelSetup.jl"))
include(joinpath(ROOT, "src", "structural_model_heterogenous_preferences", "ModelSolver.jl"))
include(joinpath(ROOT, "src", "structural_model_heterogenous_preferences", "ModelEstimation.jl"))

# Include counterfactual analysis modules
include("complementarity_analysis.jl")
include("rto_analysis.jl")
include("plotting_utils.jl")
include("solver_extensions.jl")

"""
Loads a set of parameters from a YAML file, creates the Primitives,
and returns the solved model equilibrium (prim, res).
"""
function load_and_solve_model(param_file::String)
    println("Loading and solving for: $param_file")
    base_config_path = joinpath(ROOT, "src", "structural_model_heterogenous_preferences", "model_parameters.yaml")
    prim_base, res_base = initializeModel(base_config_path)
    estimated_params = YAML.load_file(param_file)
    params_to_update = Dict(Symbol(k) => v for (k,v) in estimated_params)
    prim, res = update_params_and_resolve(prim_base, res_base; params_to_update=params_to_update)
    println("...solved.")
    return prim, res
end

println("-"^60)
println("STEP 1: SOLVING THE THREE CORE ECONOMIES")

prim_2019, res_2019 = load_and_solve_model(joinpath(ROOT, "output", "estimation_results", "params_2019.yaml"))
prim_2024, res_2024 = load_and_solve_model(joinpath(ROOT, "output", "estimation_results", "params_2024.yaml"))

println("Creating and solving the Counterfactual (CF) Hybrid World...")
prod_params_2019 = (A0=prim_2019.A₀, A1=prim_2019.A₁, ψ₀=prim_2019.ψ₀, ν=prim_2019.ν, ϕ=prim_2019.ϕ)
pref_params_2024 = (c₀=prim_2024.c₀, χ=prim_2024.χ, μ=prim_2024.μ)
params_cf = merge(Dict(pairs(prod_params_2019)), Dict(pairs(pref_params_2024)))
prim_cf, res_cf = update_params_and_resolve(prim_2024, res_2024; params_to_update=params_cf)
println("...CF model solved.")

println("-"^60)
println("STEP 2: RUNNING DECOMPOSITION EXPERIMENTS")

moments_2019 = compute_model_moments(prim_2019, res_2019)
moments_2024 = compute_model_moments(prim_2024, res_2024)
moments_cf   = compute_model_moments(prim_cf, res_cf)

E_alpha_2019 = moments_2019[:mean_alpha]
E_alpha_2024 = moments_2024[:mean_alpha]
E_alpha_cf   = moments_cf[:mean_alpha]

delta_total_alpha = E_alpha_2024 - E_alpha_2019
delta_pref_alpha  = E_alpha_cf - E_alpha_2019
delta_tech_alpha  = E_alpha_2024 - E_alpha_cf

prod_2019 = moments_2019[:agg_productivity]
prod_2024 = moments_2024[:agg_productivity]
prod_cf   = moments_cf[:agg_productivity]

delta_total_prod = prod_2024 - prod_2019
delta_pref_prod  = prod_cf - prod_2019
delta_tech_prod  = prod_2024 - prod_cf

ineq_2019 = moments_2019[:var_logwage]
ineq_2024 = moments_2024[:var_logwage]
ineq_cf   = moments_cf[:var_logwage]

delta_total_ineq = ineq_2024 - ineq_2019
delta_pref_ineq  = ineq_cf - ineq_2019
delta_tech_ineq  = ineq_2024 - ineq_cf

println("\n--- Decomposition Results ---")
results_df = DataFrame(
    Outcome = ["E[alpha]", "Productivity", "Var(log w)"],
    TotalChange = [delta_total_alpha, delta_total_prod, delta_total_ineq],
    DueToPreferences = [delta_pref_alpha, delta_pref_prod, delta_pref_ineq],
    DueToTechnology = [delta_tech_alpha, delta_tech_prod, delta_tech_ineq]
)
println(results_df)

# Create results directory
results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)

# Save decomposition results
CSV.write(joinpath(results_dir, "decomposition_results.csv"), results_df)
println("Decomposition results saved to results/decomposition_results.csv")

# --- Optional: Run Advanced Experiments ---
run_advanced = true  # Set to false to skip computationally intensive experiments

if run_advanced
    # Run complementarity experiment (CF 4)
    complementarity_results = run_complementarity_experiment(
        prim_2024, res_2024;
        phi_grid_size=3,  # Start small for testing
        nu_grid_size=3,
        phi_range=0.1,
        nu_range=0.1
    )
    
    # Run RTO mandate experiment (CF 5)
    rto_results = run_rto_experiment(
        prim_2024, res_2024;
        alpha_max_values=[0.3, 0.5, 0.7, 0.9]
    )
    
    # Generate all plots
    generate_all_plots(results_dir)
    
    println("\n" * "="^60)
    println("ALL COUNTERFACTUAL EXPERIMENTS COMPLETED!")
    println("Results saved in: $results_dir")
    println("="^60)
else
    println("\nAdvanced experiments skipped. Set run_advanced=true to run them.")
end
