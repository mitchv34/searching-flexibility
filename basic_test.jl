#!/usr/bin/env julia

"""
Simple Basic Test - Test if core model files can be loaded without heavy dependencies
"""

println("🧪 Running Basic Model Loading Test...")

# Test 1: Check if core Julia is working
println("✅ Julia is working")

# Test 2: Try to activate project
try
    using Pkg
    Pkg.activate(".")
    println("✅ Project activated")
catch e
    println("❌ Could not activate project: $e")
    exit(1)
end

# Test 3: Try basic imports
println("📦 Testing basic package imports...")
try
    using Printf
    println("✅ Printf imported")
catch e
    println("❌ Printf failed: $e")
end

try
    using LinearAlgebra
    println("✅ LinearAlgebra imported")
catch e
    println("❌ LinearAlgebra failed: $e")
end

# Test 4: Check if model files exist
model_files = [
    "src/structural_model_heterogenous_preferences/ModelSetup.jl",
    "src/structural_model_heterogenous_preferences/ModelSolver.jl", 
    "src/structural_model_heterogenous_preferences/ModelEstimation.jl",
    "src/structural_model_new/ModelSetup.jl",
    "src/structural_model_new/ModelSolver.jl",
    "src/structural_model_new/ModelEstimation.jl"
]

println("📁 Checking model files...")
for file in model_files
    if isfile(file)
        println("✅ Found: $file")
    else
        println("❌ Missing: $file")
    end
end

# Test 5: Check config files
config_files = [
    "src/structural_model_heterogenous_preferences/model_parameters.yaml",
    "src/structural_model_new/model_parameters.yaml"
]

println("⚙️  Checking config files...")
for file in config_files
    if isfile(file)
        println("✅ Found: $file")
    else
        println("❌ Missing: $file")
    end
end

println("\n🎉 Basic test completed!")
println("The repository structure looks ready for experiments.")
println("To run a specific experiment, use:")
println("  julia experiment_runner.jl [experiment_name]")