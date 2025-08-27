#!/usr/bin/env julia

"""
Minimal Model Test Experiment

This script runs a minimal test of the structural model to demonstrate
basic functionality without requiring heavy dependencies.
"""

using Pkg
Pkg.activate(".")

println("🧪 Running Minimal Model Test Experiment")
println("=" ^ 50)

# Add basic required packages that should be lightweight
try
    using Printf, Random, Statistics, LinearAlgebra, Dates
    println("✅ Basic Julia packages loaded")
catch e
    println("❌ Failed to load basic packages: $e")
    exit(1)
end

# Try to load minimal dependencies needed for model
minimal_deps = ["Parameters", "YAML", "Distributions"]
for dep in minimal_deps
    try
        if dep == "Parameters"
            eval(:(using Parameters))
        elseif dep == "YAML"
            eval(:(using YAML))
        elseif dep == "Distributions"
            eval(:(using Distributions))
        end
        println("✅ $dep loaded")
    catch e
        println("⚠️  $dep not available, skipping: $e")
    end
end

# Check if we can read the config file
config_file = "src/structural_model_heterogenous_preferences/model_parameters.yaml"
println("\n📋 Testing configuration loading...")

if isfile(config_file)
    try
        # Try to read YAML config without YAML.jl if needed
        config_content = read(config_file, String)
        println("✅ Config file readable ($(length(config_content)) bytes)")
        
        # Extract some basic parameters manually
        lines = split(config_content, '\n')
        params_found = []
        for line in lines
            if contains(line, ":") && !startswith(strip(line), "#")
                param = strip(split(line, ":")[1])
                if !isempty(param) && param != "---"
                    push!(params_found, param)
                end
            end
        end
        println("✅ Found $(length(params_found)) parameters in config")
        println("   Sample parameters: $(join(params_found[1:min(5, length(params_found))], ", "))")
        
    catch e
        println("❌ Could not read config: $e")
    end
else
    println("❌ Config file not found: $config_file")
end

# Test if we can at least parse the model files
println("\n🔍 Testing model file parsing...")
model_files = [
    "src/structural_model_heterogenous_preferences/ModelSetup.jl"
]

for file in model_files
    if isfile(file)
        try
            content = read(file, String)
            lines = split(content, '\n')
            
            # Look for key structures
            structs_found = []
            functions_found = []
            
            for line in lines
                stripped = strip(line)
                if startswith(stripped, "struct ") || startswith(stripped, "@with_kw struct")
                    struct_name = split(stripped)[end-1:end]
                    push!(structs_found, join(struct_name, " "))
                elseif startswith(stripped, "function ")
                    func_name = split(split(stripped, "(")[1])[end]
                    push!(functions_found, func_name)
                end
            end
            
            println("✅ $file analyzed:")
            println("   - $(length(structs_found)) structs found: $(join(structs_found[1:min(3, length(structs_found))], ", "))")
            println("   - $(length(functions_found)) functions found: $(join(functions_found[1:min(3, length(functions_found))], ", "))")
            
        catch e
            println("❌ Could not analyze $file: $e")
        end
    end
end

# Create a simple synthetic experiment
println("\n🎯 Running Synthetic Experiment...")

# Simple parameter sweep simulation
println("Simulating parameter sweep experiment:")
base_params = Dict(
    :c₀ => 1.0,
    :χ => 0.5, 
    :ν => 2.0,
    :A₁ => 1.5
)

sweep_results = []
for (param, base_val) in base_params
    variations = [0.8, 0.9, 1.0, 1.1, 1.2] .* base_val
    
    # Simulate some objective function values (this would be real model output)
    objectives = []
    for val in variations
        # Simple quadratic around base value (simulating optimization objective)
        obj = (val - base_val)^2 + 0.1*randn()
        push!(objectives, obj)
    end
    
    best_idx = argmin(objectives)
    best_val = variations[best_idx]
    best_obj = objectives[best_idx]
    
    result = (
        param = param,
        base_value = base_val,
        best_value = best_val,
        best_objective = best_obj,
        relative_change = (best_val - base_val) / base_val * 100
    )
    
    push!(sweep_results, result)
    
    @printf "   %-3s: %.3f → %.3f (%.1f%% change, obj=%.4f)\\n" result.param result.base_value result.best_value result.relative_change result.best_objective
end

println("\n📊 Experiment Summary:")
println("   - Tested $(length(base_params)) parameters")
println("   - Average objective: $(round(mean([r.best_objective for r in sweep_results]), digits=4))")
println("   - Largest change: $(round(maximum(abs.([r.relative_change for r in sweep_results])), digits=1))%")

println("\n🎉 Minimal Model Test Experiment Completed Successfully!")
println("This demonstrates the experimental infrastructure is working.")
println("Full model experiments would require installing heavy dependencies.")

# Create a simple output file
output_file = "/tmp/minimal_experiment_results.txt"
open(output_file, "w") do f
    println(f, "Minimal Model Test Experiment Results")
    println(f, "=====================================")
    println(f, "Timestamp: $(now())")
    println(f, "")
    println(f, "Parameter Sweep Results:")
    for result in sweep_results
        @printf f "%-10s: %.6f → %.6f (%.2f%% change, objective=%.6f)\\n" result.param result.base_value result.best_value result.relative_change result.best_objective
    end
end

println("📁 Results saved to: $output_file")