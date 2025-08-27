#!/usr/bin/env julia
#==========================================================================================
# Quick Test: Estimated Parameters Integration
# Author: Generated for searching-flexibility project
# Date: 2025-08-27
# Description: Test that estimated parameters work with model setup
==========================================================================================#

using Pkg
Pkg.activate(".")

println("🧪 Testing Estimated Parameters Integration")
println("="^60)

# Test that we can load the estimated parameters and create a model
try
    # Include model setup
    include("../ModelSetup.jl")
    println("✅ ModelSetup.jl loaded successfully")
    
    # Test loading estimated parameters
    est_params_path = "../../../data/results/estimated_parameters/estimated_parameters_2024.yaml"
    println("📋 Testing parameter loading from: $est_params_path")
    
    if isfile(est_params_path)
        prim = create_primitives_from_yaml(est_params_path)
        println("✅ Primitives created successfully")
        println("   📊 n_h = $(prim.n_h), n_ψ = $(prim.n_ψ)")
        println("   📈 Key parameters:")
        println("      psi_0 = $(prim.ψ₀)")
        println("      c0 = $(prim.c₀)")
        println("      mu = $(prim.μ)")
        println("      kappa0 = $(prim.κ₀)")
        println("      a_h = $(prim.aₕ), b_h = $(prim.bₕ)")
        
        # Test creating results
        res = Results(prim)
        println("✅ Results struct created successfully")
        
        println("\n🎉 Estimated parameters integration test PASSED!")
        println("   The counterfactual experiments should work with these parameters.")
        
    else
        println("❌ Estimated parameters file not found: $est_params_path")
    end
    
catch e
    println("❌ Error testing estimated parameters: $e")
    println("\nFull error:")
    showerror(stdout, e, catch_backtrace())
end
