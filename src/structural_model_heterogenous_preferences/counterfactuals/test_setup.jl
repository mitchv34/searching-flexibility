# Simple test script to verify counterfactual setup
# src/structural_model_heterogenous_preferences/counterfactuals/test_setup.jl

println("Testing counterfactual analysis setup...")

# Test 1: Check if we can activate the project
try
    using Pkg
    Pkg.activate("../../..")
    println("✓ Project activation successful")
catch e
    println("✗ Project activation failed: $e")
end

# Test 2: Check if parameter files exist
param_files = [
    "/project/high_tech_ind/searching-flexibility/output/estimation_results/params_2019.yaml",
    "/project/high_tech_ind/searching-flexibility/output/estimation_results/params_2024.yaml"
]

for file in param_files
    if isfile(file)
        println("✓ Found parameter file: $(basename(file))")
    else
        println("✗ Missing parameter file: $file")
    end
end

# Test 3: Check if model files exist
ROOT = joinpath(@__DIR__, "..", "..", "..")
model_files = [
    joinpath(ROOT, "src", "structural_model_heterogenous_preferences", "ModelSetup.jl"),
    joinpath(ROOT, "src", "structural_model_heterogenous_preferences", "ModelSolver.jl"), 
    joinpath(ROOT, "src", "structural_model_heterogenous_preferences", "ModelEstimation.jl")
]

for file in model_files
    if isfile(file)
        println("✓ Found model file: $(basename(file))")
    else
        println("✗ Missing model file: $file")
    end
end

# Test 4: Try to load basic packages
basic_packages = ["YAML", "DataFrames", "CSV"]
for pkg in basic_packages
    try
        eval(Meta.parse("using $pkg"))
        println("✓ Successfully loaded $pkg")
    catch e
        println("✗ Failed to load $pkg: $e")
    end
end

# Test 5: Check if we can include the counterfactual analysis files
cf_files = [
    "complementarity_analysis.jl",
    "rto_analysis.jl", 
    "plotting_utils.jl",
    "solver_extensions.jl"
]

for file in cf_files
    file_path = joinpath(@__DIR__, file)
    if isfile(file_path)
        println("✓ Found counterfactual file: $file")
        try
            # Just check syntax, don't execute
            read(file_path, String)
            println("  - File is readable")
        catch e
            println("  ✗ File has issues: $e")
        end
    else
        println("✗ Missing counterfactual file: $file")
    end
end

println("\nSetup test completed!")
println("If all tests pass, you can run the counterfactual analysis.")
println("To run: include(\"run_counterfactuals.jl\")")
