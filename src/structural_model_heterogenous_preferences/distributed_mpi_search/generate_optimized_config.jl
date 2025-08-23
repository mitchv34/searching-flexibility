#!/usr/bin/env julia

# Utility Script: Generate Optimized Config from MPI Search Results
# This script extracts the best parameters from MPI search results and creates
# a ready-to-use config file for run_file.jl

using Pkg
Pkg.activate("/project/high_tech_ind/searching-flexibility")

using JSON3, YAML
using Printf, Dates

# Parse command line arguments for job ID
JOB_ID = if length(ARGS) >= 1
    strip(ARGS[1])  # Job ID as string
else
    nothing  # Analyze all jobs if no arguments
end

println("⚙️  MPI Search Config Generator")
println("=" ^ 50)
if JOB_ID !== nothing
    println("📊 Analyzing job ID: $JOB_ID")
else
    println("📊 Analyzing most recent job")
end

# Configuration
const SCRIPT_DIR = dirname(@__FILE__)
const RESULTS_DIR = joinpath(SCRIPT_DIR, "output")
const OUTPUT_DIR = if JOB_ID !== nothing
    "figures/mpi_analysis/job_$JOB_ID"
else
    "figures/mpi_analysis"
end

# Create output directory
mkpath(OUTPUT_DIR)

"""Find and load the most recent MPI search results for specified job"""
function load_latest_results()
    try
        # Filter files based on job ID if specified
        if JOB_ID !== nothing
            pattern = "mpi_search_results_job$(JOB_ID)_"
            files = filter(f -> startswith(f, pattern) && endswith(f, ".json"), 
                            readdir(RESULTS_DIR))
            if isempty(files)
                error("No MPI search results found for job $JOB_ID!")
            end
        else
            files = filter(f -> startswith(f, "mpi_search_results_") && endswith(f, ".json"), 
                            readdir(RESULTS_DIR))
            if isempty(files)
                error("No MPI search results found!")
            end
        end
        
        # Sort by modification time, most recent first
        full_paths = [joinpath(RESULTS_DIR, f) for f in files]
        sorted_files = sort(full_paths, by=mtime, rev=true)
        
        # Try files in order until we find one that parses successfully
        for file_path in sorted_files
            try
                println("📁 Attempting to load: $(basename(file_path))")
                
                content = read(file_path, String)
                results = JSON3.read(content)
                
                println("✅ Successfully loaded: $(basename(file_path))")
                return results, file_path
            catch json_error
                @warn "Failed to parse $(basename(file_path)): $json_error"
                continue
            end
        end
        
        error("No valid JSON files found")
    catch e
        error("Failed to load results: $e")
    end
end

"""Generate a ready-to-use config YAML file from best parameters"""
function generate_config_from_best_params(results, save_prefix::String)
    if !haskey(results, "best_params") || !haskey(results, "parameter_names")
        @warn "Missing best parameters for config generation"
        return
    end
    
    best_params = results["best_params"]
    param_names = results["parameter_names"]
    
    println("\n" * "="^80)
    println("⚙️  GENERATING CONFIG FROM BEST PARAMETERS")
    println("="^80)
    
    # Parameter mapping from search space to YAML config
    param_mapping = Dict(
        :κ₀ => "kappa0",
        :κ₁ => "kappa1", 
        :β => "beta",
        :δ => "delta",
        :b => "b",
        :ξ => "xi",
        :A₀ => "A0",
        :A₁ => "A1",
        :ψ₀ => "psi_0",
        :ϕ => "phi",
        :ν => "nu",
        :c₀ => "c0",
        :χ => "chi",
        :γ₀ => "gamma0",
        :γ₁ => "gamma1",
        :aₕ => "a_h",
        :bₕ => "b_h",
        :μ => "mu"
    )
    
    # Load the template config to preserve other settings
    template_config_path = joinpath(dirname(@__FILE__), "..", "model_parameters.yaml")
    
    if !isfile(template_config_path)
        @warn "Template config file not found at: $template_config_path"
        @warn "Creating minimal config instead"
        config_dict = Dict(
            "ModelParameters" => Dict{String, Any}(),
            "ModelGrids" => Dict(
                "n_psi" => 100,
                "psi_min" => 0.0,
                "psi_max" => 1.0,
                "psi_data" => "data/results/psi_distribution_Pre_COVID.csv",
                "psi_column" => "psi",
                "psi_weight" => "probability_mass",
                "n_h" => 101,
                "h_min" => 0.05,
                "h_max" => 1.0
            ),
            "ModelSolverOptions" => Dict(
                "tol" => 1e-8,
                "max_iter" => 50000,
                "verbose" => false,
                "print_freq" => 50,
                "lambda_S_init" => 0.01,
                "lambda_u_init" => 0.01,
                "initial_S" => nothing
            )
        )
    else
        try
            config_dict = YAML.load_file(template_config_path)
            println("✅ Loaded template config from: $template_config_path")
        catch e
            @warn "Failed to load template config: $e"
            return
        end
    end
    
    # Update ModelParameters with best values
    model_params = config_dict["ModelParameters"]
    
    println("\n📊 PARAMETER UPDATES:")
    println("-" ^ 60)
    
    # Handle the JSON object structure for best_params
    for (param_symbol, yaml_key) in param_mapping
        param_str = string(param_symbol)
        if haskey(best_params, param_str)
            old_value = get(model_params, yaml_key, "N/A")
            new_value = best_params[param_str]
            model_params[yaml_key] = new_value
            println("  $yaml_key: $old_value → $(round(new_value, digits=6))")
        else
            @warn "Parameter $param_str not found in best_params"
        end
    end
    
    # Generate output config file
    output_config_path = joinpath(OUTPUT_DIR, "optimized_model_parameters.yaml")
    
    try
        YAML.write_file(output_config_path, config_dict)
        println("\n✅ Optimized config saved to: $output_config_path")
        
        # Also save to a standard location for easy use
        standard_config_path = joinpath(dirname(@__FILE__), "..", "optimized_model_parameters.yaml")
        YAML.write_file(standard_config_path, config_dict)
        println("✅ Optimized config also saved to: $standard_config_path")
        
        # Display the new config summary
        println("\n📋 NEW CONFIGURATION SUMMARY:")
        println("=" ^ 40)
        for (param_symbol, yaml_key) in param_mapping
            if haskey(model_params, yaml_key)
                value = model_params[yaml_key]
                println("  $yaml_key: $(round(value, digits=6))")
            end
        end
        
        # Add best objective value as comment in the file
        if haskey(results, "best_objective")
            best_obj = results["best_objective"]
            
            # Read the file and add a comment at the top
            content = read(output_config_path, String)
            comment_header = "# Optimized parameters from MPI search\n# Best objective value: $(round(best_obj, digits=8))\n# Generated: $(Dates.now())\n\n"
            
            open(output_config_path, "w") do io
                write(io, comment_header * content)
            end
            
            open(standard_config_path, "w") do io
                write(io, comment_header * content)
            end
            
            println("\n🏆 Best objective value: $(round(best_obj, digits=8))")
        end
        
        return output_config_path
        
    catch e
        @error "Failed to save config file: $e"
        return nothing
    end
end

"""Display parameter comparison table"""
function display_parameter_comparison(results)
    if !haskey(results, "best_params") || !haskey(results, "parameter_names")
        return
    end
    
    best_params = results["best_params"]
    param_names = results["parameter_names"]
    
    println("\n" * "="^80)
    println("📊 PARAMETER COMPARISON: ORIGINAL vs OPTIMIZED")
    println("="^80)
    
    # Load original config for comparison
    original_config_path = joinpath(dirname(@__FILE__), "..", "model_parameters.yaml")
    if isfile(original_config_path)
        try
            original_config = YAML.load_file(original_config_path)
            original_params = original_config["ModelParameters"]
            
            # Parameter mapping
            param_mapping = Dict(
                :κ₀ => "kappa0", :κ₁ => "kappa1", :β => "beta", :δ => "delta",
                :b => "b", :ξ => "xi", :A₀ => "A0", :A₁ => "A1", :ψ₀ => "psi_0",
                :ϕ => "phi", :ν => "nu", :c₀ => "c0", :χ => "chi",
                :γ₀ => "gamma0", :γ₁ => "gamma1", :aₕ => "a_h", :bₕ => "b_h", :μ => "mu"
            )
            
            println(rpad("Parameter", 15) * rpad("Original", 15) * rpad("Optimized", 15) * rpad("Change (%)", 15))
            println("-" ^ 60)
            
            for (param_symbol, yaml_key) in param_mapping
                param_str = string(param_symbol)
                if haskey(best_params, param_str) && haskey(original_params, yaml_key)
                    original_val = original_params[yaml_key]
                    optimized_val = best_params[param_str]
                    change_pct = ((optimized_val - original_val) / original_val) * 100
                    
                    println(rpad(yaml_key, 15) * 
                           rpad(@sprintf("%.6f", original_val), 15) * 
                           rpad(@sprintf("%.6f", optimized_val), 15) * 
                           @sprintf("%.2f%%", change_pct))
                end
            end
        catch e
            @warn "Could not load original config for comparison: $e"
        end
    end
end

"""Main function"""
function main()
    try
        println("🔍 Loading MPI search results...")
        results, results_file = load_latest_results()
        
        # Extract save prefix from filename
        save_prefix = begin
            filename = basename(results_file)
            job_match = match(r"mpi_search_results_(job\d+)_", filename)
            if job_match !== nothing
                job_id = job_match.captures[1]
                String(job_id)
            else
                "latest"
            end
        end
        
        println("📁 Source file: $(basename(results_file))")
        
        # Display basic results info
        if haskey(results, "best_objective")
            println("🏆 Best objective: $(round(results["best_objective"], digits=8))")
        end
        if haskey(results, "n_evaluations")
            println("📊 Total evaluations: $(results["n_evaluations"])")
        end
        
        # Display parameter comparison
        display_parameter_comparison(results)
        
        # Generate optimized config
        config_path = generate_config_from_best_params(results, save_prefix)
        
        if config_path !== nothing
            println("\n" * "="^80)
            println("🎯 SUCCESS! Config generation complete!")
            println("="^80)
            println("📁 Optimized config saved to:")
            println("   • Analysis directory: $config_path")
            println("   • Ready-to-use: optimized_model_parameters.yaml")
            println("\n💡 To use the optimized parameters:")
            println("   1. Backup your current model_parameters.yaml")
            println("   2. Copy optimized_model_parameters.yaml to model_parameters.yaml")
            println("   3. Run your model with julia run_file.jl")
            
            # Verify the file was created and is valid YAML
            standard_config_path = joinpath(dirname(@__FILE__), "..", "optimized_model_parameters.yaml")
            if isfile(standard_config_path)
                try
                    test_config = YAML.load_file(standard_config_path)
                    println("\n✅ Generated config file is valid YAML")
                    println("📊 ModelParameters count: $(length(test_config["ModelParameters"]))")
                catch e
                    @warn "Generated config file has YAML syntax errors: $e"
                end
            else
                @warn "Standard config file was not created at expected location"
            end
        else
            println("\n❌ Config generation failed!")
        end
        
    catch e
        println("❌ Error during config generation: $e")
        rethrow(e)
    end
end

# Run the main function
main()

println("\n🎯 Config generation script complete!")
