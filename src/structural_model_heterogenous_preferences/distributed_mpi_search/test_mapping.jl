#!/usr/bin/env julia

# Test the parameter name mapping fix

using Pkg
Pkg.activate("/project/high_tech_ind/searching-flexibility")

using YAML, Printf

# Set paths
const ROOT = "/project/high_tech_ind/searching-flexibility"
const MODEL_DIR = joinpath(ROOT, "src", "structural_model_heterogenous_preferences")
const MPI_SEARCH_DIR = joinpath(MODEL_DIR, "distributed_mpi_search")

# Include model files
include(joinpath(MODEL_DIR, "ModelSetup.jl"))
include(joinpath(MODEL_DIR, "ModelSolver.jl"))  
include(joinpath(MODEL_DIR, "ModelEstimation.jl"))

println("🔍 Testing parameter name mapping fix")
println("=" ^ 50)

# Load config with the corrected parameter names
config_file = joinpath(MPI_SEARCH_DIR, "mpi_search_config.yaml")
config = YAML.load_file(config_file)

# Test function with parameter mapping
function test_parameter_mapping()
    """Test that parameters are now correctly mapped and applied"""
    
    # Initialize with MPI search config (to avoid file path issues)
    println("🔧 Initializing model...")
    prim_base, res_base = initializeModel(config_file)
    
    println("📊 Baseline parameters:")
    println("  A₁: $(prim_base.A₁)")
    println("  c₀: $(prim_base.c₀)")
    println("  ν: $(prim_base.ν)")
    println("  χ: $(prim_base.χ)")
    
    # Test parameters with OLD naming (what your search results used) - use DIFFERENT values from baseline
    old_style_params = Dict(
        "A1" => 1.5,      # baseline is 1.0, so this should change
        "nu" => 0.4,      # baseline is 0.25, so this should change  
        "chi" => 5.0,     # baseline is 7.0, so this should change
        "c0" => 0.08      # baseline is 0.05, so this should change
    )
    
    # Apply the same mapping as in the fixed code
    symbol_params = Dict(Symbol(k) => v for (k, v) in old_style_params)
    
    param_name_mapping = Dict(
        :A1 => :A₁,
        :nu => :ν, 
        :chi => :χ,
        :kappa0 => :κ₀,
        :mu => :μ,
        :psi_0 => :ψ₀,
        :a_h => :aₕ,
        :phi => :ϕ,
        :c0 => :c₀,
        :b_h => :bₕ
    )
    
    mapped_params = Dict()
    for (param_name, param_value) in symbol_params
        correct_name = get(param_name_mapping, param_name, param_name)
        mapped_params[correct_name] = param_value
    end
    
    println("🔧 Original params: $old_style_params")
    println("🔧 Mapped params: $mapped_params")
    
    # Update primitives
    prim_new, res_new = update_primitives_results(prim_base, res_base, mapped_params)
    
    # Verify changes
    println("✅ Parameter verification:")
    all_applied = true
    
    # Store baseline values for comparison
    baseline_values = Dict(
        :A₁ => prim_base.A₁,
        :c₀ => prim_base.c₀,
        :ν => prim_base.ν,
        :χ => prim_base.χ
    )
    
    for (param_sym, new_val) in mapped_params
        try
            baseline_val = get(baseline_values, param_sym, nothing)
            actual_val = getfield(prim_new, param_sym)
            
            if baseline_val !== nothing
                changed_from_baseline = !(baseline_val ≈ actual_val)
                correctly_updated = (new_val ≈ actual_val)
                println("  $param_sym: $(baseline_val) → $actual_val (expected: $new_val)")
                println("    Changed from baseline: $changed_from_baseline, Correctly set: $correctly_updated")
                
                if !correctly_updated
                    all_applied = false
                end
            else
                correctly_updated = (new_val ≈ actual_val)
                println("  $param_sym: $actual_val (expected: $new_val, correct: $correctly_updated)")
                if !correctly_updated
                    all_applied = false
                end
            end
        catch e
            println("  ❌ $param_sym: Error accessing parameter: $e")
            all_applied = false
        end
    end
    
    if all_applied
        println("✅ SUCCESS: All parameters were correctly applied!")
    else
        println("❌ FAILURE: Some parameters were not applied")
    end
    
    # Quick solve and moment computation to see if objective changes
    println("⚙️  Solving model with new parameters...")
    solve_model(prim_new, res_new; tol=1e-6, max_iter=5000, verbose=false)
    
    println("📈 Computing moments...")
    model_moments = compute_model_moments(prim_new, res_new)
    
    # Print first few moments
    println("📊 Sample moments with updated parameters:")
    for (i, (key, value)) in enumerate(model_moments)
        if i <= 3
            println("  $key: $value")
        end
    end
    
    return all_applied
end

# Test the mapping
success = test_parameter_mapping()

if success
    println("\n🎉 PROBLEM SOLVED!")
    println("The parameter name mapping should now work correctly.")
    println("Your MPI search should now produce different objective values for different parameters.")
else
    println("\n❌ Issue still exists - further investigation needed")
end
