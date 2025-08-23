#!/usr/bin/env julia

# Simple test to see if parameter changes affect the objective function
# This bypasses config file issues by using a manual setup

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

println("🔍 Testing objective function problem")
println("=" ^ 50)

# Load target moments manually
target_moments_config_file = joinpath(MPI_SEARCH_DIR, "mpi_search_config.yaml")
config = YAML.load_file(target_moments_config_file)

# Load target moments
target_moments_config = config["MPISearchConfig"]["target_moments"]
data_file_path = joinpath(ROOT, target_moments_config["data_file"])

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

# Test the same approach as in mpi_search.jl
function test_mpi_objective(params::Dict{String, Float64})
    """This replicates the exact approach from mpi_search.jl"""
    
    println("\n🔍 Testing parameters: $params")
    
    # Use the EXACT same config file path logic as mpi_search.jl 
    base_config_file = joinpath(MODEL_DIR, "model_parameters.yaml")
    
    if !isfile(base_config_file)
        println("❌ Base config file not found: $base_config_file")
        
        # Try alternative config file locations (EXACT same as mpi_search.jl)
        alt_configs = [
            joinpath(ROOT, "config", "search_config.yaml"),
            joinpath(MPI_SEARCH_DIR, "mpi_search_config.yaml"),
            joinpath(MODEL_DIR, "config.yaml")
        ]
        
        base_config_file = nothing
        for alt_config in alt_configs
            if isfile(alt_config)
                base_config_file = alt_config
                println("   ✓ Found alternative config: $alt_config")
                break
            end
        end
        
        if base_config_file === nothing
            println("   ❌ No valid config file found!")
            return 1e10
        end
    else
        println("✓ Using config file: $base_config_file")
    end
    
    # Try to initialize model
    local prim_base, res_base
    try
        prim_base, res_base = initializeModel(base_config_file)
        println("✓ Model initialized successfully")
    catch e
        println("❌ Model initialization failed: $e")
        println("Let's try the MPI search config instead...")
        
        # Try the MPI search config as fallback
        try
            prim_base, res_base = initializeModel(joinpath(MPI_SEARCH_DIR, "mpi_search_config.yaml"))
            println("✓ Model initialized with MPI search config")
        catch e2
            println("❌ Both config files failed: $e2")
            return 1e10
        end
    end
    
    # Print baseline parameters
    println("📊 Baseline parameters:")
    println("  A₁: $(prim_base.A₁)")
    println("  c₀: $(prim_base.c₀)")
    println("  ν: $(prim_base.ν)")
    println("  χ: $(prim_base.χ)")
    
    # Convert string keys to symbols (EXACT same as mpi_search.jl)
    symbol_params = Dict(Symbol(k) => v for (k, v) in params)
    println("🔧 Updating with: $symbol_params")
    
    # Update primitives (EXACT same call as mpi_search.jl)
    prim_new, res_new = update_primitives_results(prim_base, res_base, symbol_params)
    
    # Verify parameters changed
    println("✅ Parameter verification:")
    for (param_sym, new_val) in symbol_params
        try
            actual_val = getfield(prim_new, param_sym)
            changed = !(new_val ≈ actual_val)
            println("  $param_sym: $new_val → $actual_val (changed: $changed)")
        catch e
            println("  ❌ $param_sym: not found in primitives! $e")
        end
    end
    
    # Solve model (EXACT same call as mpi_search.jl)
    println("⚙️  Solving model...")
    solve_model(prim_new, res_new; 
                tol=1e-6, 
                max_iter=15000, 
                verbose=false,
                λ_S_init=0.01,
                λ_u_init=0.01
            )
    
    # Compute moments (EXACT same call as mpi_search.jl)
    println("📈 Computing model moments...")
    model_moments = compute_model_moments(prim_new, res_new)
    
    # Print a few computed moments
    println("📊 Sample computed moments:")
    moment_count = 0
    for (key, value) in model_moments
        if moment_count < 3
            println("  $key: $value")
            moment_count += 1
        else
            break
        end
    end
    
    # Calculate distance (EXACT same call as mpi_search.jl)
    println("📏 Calculating distance...")
    objective = compute_distance(
        model_moments, 
        TARGET_MOMENTS,
        nothing,  # weighting_matrix
        nothing   # matrix_moment_order
    )
    
    println("🎯 Objective value: $objective")
    return objective
end

# Test with the EXACT parameter ranges from your results
println("\n" * ("=" ^ 50))
println("TEST 1: Parameter set from your results")

# These are from the first parameter vector in your results
params1 = Dict(
    "a_h" => 1.5000457763671875,
    "b_h" => 4.666656494140625,
    "c0" => 0.0676141357421875,
    "mu" => 0.0401605224609375,
    "chi" => 8.02886962890625,
    "A1" => 0.895977783203125,
    "nu" => 0.39546356201171884,
    "psi_0" => 0.7269210815429688,
    "phi" => 0.06525726318359375,
    "kappa0" => 1.391326904296875
)

obj1 = test_mpi_objective(params1)

println("\n" * ("=" ^ 50))
println("TEST 2: Second parameter set from your results")

# These are from the second parameter vector
params2 = Dict(
    "a_h" => 2.0000457763671875,
    "b_h" => 5.666656494140625,
    "c0" => 0.0476141357421875,
    "mu" => 0.060160522460937504,
    "chi" => 6.02886962890625,
    "A1" => 1.095977783203125,
    "nu" => 0.2454635620117188,
    "psi_0" => 0.7769210815429688,
    "phi" => 0.11525726318359375,
    "kappa0" => 1.5913269042968752
)

obj2 = test_mpi_objective(params2)

println("\n" * ("=" ^ 50))
println("RESULTS COMPARISON:")
println("Objective 1: $obj1")
println("Objective 2: $obj2")
println("Same result: $(obj1 == obj2)")
println("Expected from your file: 17.86206858480663")
println("Match expected: $(obj1 ≈ 17.86206858480663)")

if obj1 == obj2
    println("\n❌ CONFIRMED PROBLEM: Both evaluations return the same value!")
    println("   This confirms that parameter changes are not affecting the model.")
    println("   Possible causes:")
    println("   1. Parameters not being applied correctly in update_primitives_results")
    println("   2. Model solver not using updated parameters")
    println("   3. Moment computation not reflecting parameter changes")
    println("   4. Same cached results being returned")
else
    println("\n✅ GOOD: Different parameters give different objectives")
end
