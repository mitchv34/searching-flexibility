#!/usr/bin/env julia
#==========================================================================================
# Quick Test Script for Counterfactual Experiments
# Author: Generated for searching-flexibility project
# Date: 2025-08-27
# Description: Simple test to verify experiment infrastructure works
==========================================================================================#

using Pkg
Pkg.activate(".")

println("🧪 Testing Counterfactual Experiment Infrastructure")
println("="^60)

# Test 1: Check if configuration file can be loaded
println("📋 Test 1: Loading configuration file...")
try
    using YAML
    config_path = joinpath(@__DIR__, "counterfactual_config.yaml")
    config = YAML.load_file(config_path)
    println("   ✅ Configuration loaded successfully")
    println("   📊 Found $(length(keys(config))) top-level sections")
    
    # Check each experiment config
    for i in 1:6
        # Simplified check
        exp_keys = [k for k in keys(config) if startswith(k, "Experiment$(i)")]
        if !isempty(exp_keys)
            println("   📝 Experiment $i configuration found")
        else
            println("   ⚠️  Experiment $i configuration missing")
        end
    end
    
catch e
    println("   ❌ Error loading configuration: $e")
end

# Test 2: Check if model files can be included
println("\n📦 Test 2: Checking model file dependencies...")
model_files = [
    "../ModelSetup.jl",
    "../ModelSolver.jl", 
    "../ModelEstimation.jl"
]

for file in model_files
    full_path = joinpath(@__DIR__, file)
    if isfile(full_path)
        println("   ✅ Found: $file")
    else
        println("   ❌ Missing: $file")
    end
end

# Test 3: Check if base model parameters file exists
println("\n📊 Test 3: Checking base model parameters...")
try
    base_params_path = "src/structural_model_heterogenous_preferences/model_parameters.yaml"
    if isfile(base_params_path)
        println("   ✅ Base model parameters found: $base_params_path")
    else
        println("   ⚠️  Base model parameters not found at: $base_params_path")
        # Try alternative location
        alt_path = "../model_parameters.yaml"
        if isfile(joinpath(@__DIR__, alt_path))
            println("   ✅ Found alternative: $alt_path")
        end
    end
catch e
    println("   ❌ Error checking base parameters: $e")
end

# Test 4: Check results directory
println("\n📁 Test 4: Checking results directory...")
results_dir = joinpath(@__DIR__, "results")
if isdir(results_dir)
    println("   ✅ Results directory exists: $results_dir")
    println("   📊 Current contents: $(length(readdir(results_dir))) items")
else
    println("   ❌ Results directory missing: $results_dir")
end

# Test 5: Check experiment scripts
println("\n📜 Test 5: Checking experiment scripts...")
experiment_scripts = [
    "experiment_1_no_remote_work.jl",
    "experiment_2_remote_tech_levels.jl",
    "experiment_3_remote_mandate.jl",
    "experiment_4_search_frictions.jl",
    "experiment_5_unemployment_benefits.jl",
    "experiment_6_bargaining_power.jl"
]

all_scripts_exist = true
for script in experiment_scripts
    script_path = joinpath(@__DIR__, script)
    if isfile(script_path)
        println("   ✅ $script")
    else
        println("   ❌ $script")
        all_scripts_exist = false
    end
end

# Test 6: Check master runner
println("\n🎮 Test 6: Checking master runner...")
runner_path = joinpath(@__DIR__, "run_all_experiments.jl")
if isfile(runner_path)
    println("   ✅ Master runner exists: run_all_experiments.jl")
else
    println("   ❌ Master runner missing: run_all_experiments.jl")
end

# Summary
println("\n" * "="^60)
println("🎯 TEST SUMMARY")
println("="^60)

if all_scripts_exist
    println("✅ All experiment scripts are present")
    println("🚀 Infrastructure appears ready for counterfactual experiments")
    println()
    println("💡 To run experiments:")
    println("   • Single experiment: julia experiment_1_no_remote_work.jl")
    println("   • Multiple experiments: julia run_all_experiments.jl --experiments 1,2")
    println("   • All experiments: julia run_all_experiments.jl")
    println()
    println("⚠️  Note: Experiment 3 requires implementing solve_model_rto function")
else
    println("❌ Some scripts are missing - check the setup")
end

println("\n🎉 Test completed!")
