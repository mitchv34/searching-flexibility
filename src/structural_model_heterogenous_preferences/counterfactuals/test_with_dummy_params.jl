# Test counterfactual framework with dummy parameters
# src/structural_model_heterogenous_preferences/counterfactuals/test_with_dummy_params.jl

println("Testing counterfactual framework with dummy parameters...")

using Pkg
Pkg.activate("../../..") 

using YAML, DataFrames, Printf

# --- Project Setup ---
const ROOT = joinpath(@__DIR__, "..", "..", "..")
cd(ROOT)  # Change to root for relative paths

include(joinpath(ROOT, "src", "structural_model_heterogenous_preferences", "ModelSetup.jl"))
include(joinpath(ROOT, "src", "structural_model_heterogenous_preferences", "ModelSolver.jl"))
include(joinpath(ROOT, "src", "structural_model_heterogenous_preferences", "ModelEstimation.jl"))

println("✓ Successfully included all model files")

# Test 1: Initialize base model
println("\n" * "="^50)
println("TEST 1: INITIALIZING BASE MODEL")

base_config_path = joinpath(ROOT, "src", "structural_model_heterogenous_preferences", "model_parameters.yaml")
prim_base, res_base = initializeModel(base_config_path)
println("✓ Successfully initialized base model")

# Test 2: Test moment computation on base model
println("\n" * "="^50) 
println("TEST 2: COMPUTING MOMENTS ON BASE MODEL")

moments_base = compute_model_moments(prim_base, res_base)
println("✓ Successfully computed moments")

# Print key moments
key_moments = [:mean_alpha, :agg_productivity, :var_logwage]
println("\nKey moments for base model:")
for moment in key_moments
    if haskey(moments_base, moment)
        println("  $moment: $(round(moments_base[moment], digits=6))")
    else
        println("  $moment: NOT FOUND in moments")
        println("  Available moments: $(keys(moments_base))")
        break
    end
end

# Test 3: Test parameter updating
println("\n" * "="^50)
println("TEST 3: TESTING PARAMETER UPDATING")

# Create some dummy parameter changes (small changes to avoid convergence issues)
dummy_params_2019 = Dict(
    :A₀ => prim_base.A₀ * 0.95,  # Slightly lower technology in 2019
    :ψ₀ => prim_base.ψ₀ * 0.9,   # Lower remote work productivity in 2019
    :c₀ => prim_base.c₀ * 1.1    # Slightly higher cost of remote work in 2019
)

dummy_params_2024 = Dict(
    :A₀ => prim_base.A₀ * 1.05,  # Slightly higher technology in 2024
    :ψ₀ => prim_base.ψ₀ * 1.1,   # Higher remote work productivity in 2024
    :c₀ => prim_base.c₀ * 0.9    # Lower cost of remote work in 2024
)

try
    # Test 2019 parameters
    prim_2019, res_2019 = update_params_and_resolve(prim_base, res_base; params_to_update=dummy_params_2019)
    moments_2019 = compute_model_moments(prim_2019, res_2019)
    println("✓ Successfully updated to 2019 parameters and computed moments")
    
    # Test 2024 parameters  
    prim_2024, res_2024 = update_params_and_resolve(prim_base, res_base; params_to_update=dummy_params_2024)
    moments_2024 = compute_model_moments(prim_2024, res_2024)
    println("✓ Successfully updated to 2024 parameters and computed moments")
    
    # Test 4: Simple decomposition
    println("\n" * "="^50)
    println("TEST 4: SIMPLE DECOMPOSITION ANALYSIS")
    
    # Calculate changes
    change_alpha = moments_2024[:mean_alpha] - moments_2019[:mean_alpha]
    change_prod = moments_2024[:agg_productivity] - moments_2019[:agg_productivity]  
    change_ineq = moments_2024[:var_logwage] - moments_2019[:var_logwage]
    
    println("\nChanges from 2019 to 2024 (dummy parameters):")
    println("  Δ Mean Alpha: $(round(change_alpha, digits=6))")
    println("  Δ Productivity: $(round(change_prod, digits=6))")
    println("  Δ Wage Inequality: $(round(change_ineq, digits=6))")
    
    # Create hybrid counterfactual (2019 tech + 2024 preferences)
    # For this simple test, let's just use technology parameters from 2019
    hybrid_params = Dict(
        :A₀ => dummy_params_2019[:A₀],   # 2019 technology
        :ψ₀ => dummy_params_2019[:ψ₀],   # 2019 remote productivity  
        :c₀ => dummy_params_2024[:c₀]    # 2024 preferences (cost)
    )
    
    prim_hybrid, res_hybrid = update_params_and_resolve(prim_base, res_base; params_to_update=hybrid_params)
    moments_hybrid = compute_model_moments(prim_hybrid, res_hybrid)
    println("✓ Successfully created and solved hybrid counterfactual")
    
    # Simple decomposition
    total_change_alpha = moments_2024[:mean_alpha] - moments_2019[:mean_alpha]
    pref_effect_alpha = moments_hybrid[:mean_alpha] - moments_2019[:mean_alpha]
    tech_effect_alpha = moments_2024[:mean_alpha] - moments_hybrid[:mean_alpha]
    
    println("\nDecomposition of Alpha change:")
    println("  Total change: $(round(total_change_alpha, digits=6))")
    println("  Due to preferences: $(round(pref_effect_alpha, digits=6))")
    println("  Due to technology: $(round(tech_effect_alpha, digits=6))")
    println("  Sum check: $(round(pref_effect_alpha + tech_effect_alpha, digits=6))")
    
    println("\n" * "="^50)
    println("✓ ALL TESTS SUCCESSFUL!")
    println("The counterfactual framework is working correctly.")
    println("Once you have estimated parameters, you can run the full analysis.")
    
catch e
    println("\n" * "="^50)
    println("✗ TEST FAILED!")
    println("Error: $e")
    
    # Print more detailed error info
    if isa(e, MethodError)
        println("\nMethodError details:")
        println("  Function: $(e.f)")
        println("  Arguments: $(e.args)")
    end
end
