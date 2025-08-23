#!/usr/bin/env julia

# Simple test script to debug the MPI search objective function
# This tests a single parameter evaluation to see what's happening

using Pkg
Pkg.activate("/project/high_tech_ind/searching-flexibility")

# Load required packages
using YAML, Printf
using Statistics, LinearAlgebra
using Dates

# Set paths
const ROOT = "/project/high_tech_ind/searching-flexibility"
const MODEL_DIR = joinpath(ROOT, "src", "structural_model_heterogenous_preferences")
const MPI_SEARCH_DIR = joinpath(MODEL_DIR, "distributed_mpi_search")

# Include model files
include(joinpath(MODEL_DIR, "ModelSetup.jl"))
include(joinpath(MODEL_DIR, "ModelSolver.jl"))  
include(joinpath(MODEL_DIR, "ModelEstimation.jl"))

println("🔍 Testing objective function with debug output")
println("=" ^ 50)

# Load target moments
config_file = joinpath(MPI_SEARCH_DIR, "mpi_search_config.yaml")
config = YAML.load_file(config_file)

# Load target moments
target_moments_config = config["MPISearchConfig"]["target_moments"]
data_file_path = joinpath(ROOT, target_moments_config["data_file"])

if !isfile(data_file_path)
    println("❌ Target moments data file not found: $data_file_path")
    exit(1)
end

target_moments_data = YAML.load_file(data_file_path)
moments_to_use = target_moments_config["moments_to_use"]
TARGET_MOMENTS = Dict()

println("Loading target moments:")
for moment_name in moments_to_use
    if haskey(target_moments_data["DataMoments"], moment_name)
        value = target_moments_data["DataMoments"][moment_name]
        if value !== nothing
            TARGET_MOMENTS[Symbol(moment_name)] = value
            println("  $moment_name: $value")
        end
    end
end

# Test function
function test_objective_function(params::Dict{String, Float64})
    """Test a single parameter evaluation with extensive debugging"""
    
    println("\n🔍 Testing parameters: $params")
    
    # Try to find a valid config file
    base_config_file = joinpath(MODEL_DIR, "model_parameters.yaml")
    
    if !isfile(base_config_file)
        println("❌ Base config file not found: $base_config_file")
        
        # Try alternative configs
        alt_configs = [
            joinpath(ROOT, "config", "search_config.yaml"),
            joinpath(MPI_SEARCH_DIR, "mpi_search_config.yaml"),
            joinpath(MODEL_DIR, "config.yaml")
        ]
        
        for alt_config in alt_configs
            if isfile(alt_config)
                base_config_file = alt_config
                println("✓ Found alternative config: $alt_config")
                break
            end
        end
        
        if !isfile(base_config_file)
            println("❌ No valid config file found!")
            return 1e10
        end
    end
    
    println("📂 Using config file: $base_config_file")
    
    # Initialize baseline model
    println("🔧 Initializing baseline model...")
    try
        prim_base, res_base = initializeModel(base_config_file)
    catch e
        println("❌ Model initialization failed: $e")
        println("This might be a file path issue in the config.")
        return 1e10
    end
    
    # Print baseline parameters
    println("📊 Baseline parameters:")
    println("  A₁: $(prim_base.A₁)")
    println("  c₀: $(prim_base.c₀)")
    println("  ν: $(prim_base.ν)")
    println("  χ: $(prim_base.χ)")
    println("  aₕ: $(prim_base.aₕ)")
    println("  bₕ: $(prim_base.bₕ)")
    
    # Convert string keys to symbols
    symbol_params = Dict(Symbol(k) => v for (k, v) in params)
    println("🔧 Updating with: $symbol_params")
    
    # Update parameters
    prim_new, res_new = update_primitives_results(prim_base, res_base, symbol_params)
    
    # Verify parameters changed
    println("✅ Updated parameters:")
    for (param_sym, new_val) in symbol_params
        try
            actual_val = getfield(prim_new, param_sym)
            changed = !(new_val ≈ actual_val)
            println("  $param_sym: $new_val → $actual_val (changed: $changed)")
        catch e
            println("  ❌ $param_sym: not found in primitives! $e")
        end
    end
    
    # Solve model
    println("⚙️  Solving model...")
    try
        solve_model(prim_new, res_new; 
                    tol=1e-6, 
                    max_iter=15000, 
                    verbose=false
                )
        println("✓ Model solved successfully")
    catch e
        println("❌ Model solving failed: $e")
        return 1e10
    end
    
    # Compute moments
    println("📈 Computing model moments...")
    try
        model_moments = compute_model_moments(prim_new, res_new)
        println("✓ Computed $(length(model_moments)) model moments")
        
        # Print a few moments
        println("Sample model moments:")
        for (i, (key, value)) in enumerate(model_moments)
            if i <= 3
                println("  $key: $value")
            end
        end
        
        # Calculate objective
        println("📏 Calculating distance to target moments...")
        objective = compute_distance(model_moments, TARGET_MOMENTS, nothing, nothing)
        println("🎯 Objective value: $objective")
        
        return objective
        
    catch e
        println("❌ Moment computation failed: $e")
        return 1e10
    end
end

# Test with two different parameter sets
println("\n" * ("=" ^ 50))
println("TEST 1: Baseline parameters from config")

params1 = Dict(
    "a_h" => 2.0,
    "b_h" => 5.0,
    "c0" => 0.05,
    "mu" => 0.05,
    "chi" => 7.0,
    "A1" => 1.0,
    "nu" => 0.25,
    "psi_0" => 0.75,
    "phi" => 0.1,
    "kappa0" => 1.4
)

obj1 = test_objective_function(params1)

println("\n" * ("=" ^ 50))
println("TEST 2: Different parameters")

params2 = Dict(
    "a_h" => 1.8,
    "b_h" => 4.5,
    "c0" => 0.06,
    "mu" => 0.04,
    "chi" => 8.0,
    "A1" => 1.1,
    "nu" => 0.3,
    "psi_0" => 0.72,
    "phi" => 0.12,
    "kappa0" => 1.3
)

obj2 = test_objective_function(params2)

println("\n" * ("=" ^ 50))
println("COMPARISON:")
println("Objective 1: $obj1")
println("Objective 2: $obj2")
println("Different results: $(obj1 != obj2)")

if obj1 == obj2
    println("❌ PROBLEM: Both evaluations return the same value!")
    println("   This suggests parameters are not being applied correctly.")
else
    println("✅ GOOD: Different parameters give different objectives")
end
