# Minimal test of counterfactual framework
# src/structural_model_heterogenous_preferences/counterfactuals/minimal_test.jl

println("Starting minimal counterfactual test...")

using Pkg
Pkg.activate("../../..")

using YAML, DataFrames, Printf

# --- Project Setup ---
const ROOT = joinpath(@__DIR__, "..", "..", "..")

# IMPORTANT: Change to the root directory so relative paths in YAML work correctly
cd(ROOT)

include(joinpath(ROOT, "src", "structural_model_heterogenous_preferences", "ModelSetup.jl"))
include(joinpath(ROOT, "src", "structural_model_heterogenous_preferences", "ModelSolver.jl"))
include(joinpath(ROOT, "src", "structural_model_heterogenous_preferences", "ModelEstimation.jl"))

println("✓ Successfully included all model files")

# Test the helper function
function load_and_solve_model(param_file::String)
    println("Loading and solving for: $param_file")
    
    # First check if we can load a basic model
    base_config_path = joinpath(ROOT, "src", "structural_model_heterogenous_preferences", "model_parameters.yaml")
    if !isfile(base_config_path)
        println("Warning: model_parameters.yaml not found, trying alternative...")
        # Look for any yaml file in the directory
        yaml_files = filter(f -> endswith(f, ".yaml") || endswith(f, ".yml"), 
                           readdir(joinpath(ROOT, "src", "structural_model_heterogenous_preferences")))
        if !isempty(yaml_files)
            base_config_path = joinpath(ROOT, "src", "structural_model_heterogenous_preferences", yaml_files[1])
            println("Using: $base_config_path")
        else
            error("No YAML configuration file found")
        end
    end
    
    prim_base, res_base = initializeModel(base_config_path)
    println("✓ Successfully initialized base model")
    
    # Load estimated parameters
    estimated_params = YAML.load_file(param_file)
    println("✓ Successfully loaded parameters from $param_file")
    println("Parameters found: $(keys(estimated_params))")
    
    # Convert to symbols
    params_to_update = Dict(Symbol(k) => v for (k,v) in estimated_params)
    
    # Update and solve
    prim, res = update_params_and_resolve(prim_base, res_base; params_to_update=params_to_update)
    println("✓ Successfully updated parameters and resolved model")
    
    return prim, res
end

# Test with just one parameter file first
try
    println("\n" * "="^50)
    println("TESTING WITH 2024 PARAMETERS")
    prim_2024, res_2024 = load_and_solve_model(joinpath(ROOT, "output", "estimation_results", "params_2024.yaml"))
    
    # Test computing moments
    println("\nTesting moment computation...")
    moments_2024 = compute_model_moments(prim_2024, res_2024)
    println("✓ Successfully computed moments")
    
    # Print key moments
    key_moments = [:mean_alpha, :agg_productivity, :var_logwage]
    println("\nKey moments for 2024:")
    for moment in key_moments
        if haskey(moments_2024, moment)
            println("  $moment: $(round(moments_2024[moment], digits=6))")
        else
            println("  $moment: NOT FOUND")
        end
    end
    
    println("\n" * "="^50)
    println("✓ MINIMAL TEST SUCCESSFUL!")
    println("The counterfactual framework should work with your model.")
    
catch e
    println("\n" * "="^50)
    println("✗ MINIMAL TEST FAILED!")
    println("Error: $e")
    println("\nThis suggests we need to adapt the framework to your specific model structure.")
end
