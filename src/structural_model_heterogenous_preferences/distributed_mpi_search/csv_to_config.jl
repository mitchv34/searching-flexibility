#!/usr/bin/env julia

# Generate Optimized Config from Top Candidates CSV
# This script reads the top_candidates.csv file and creates an optimized model_parameters.yaml

using Pkg
Pkg.activate("/project/high_tech_ind/searching-flexibility")

using CSV, DataFrames, YAML
using Printf, Dates

println("⚙️  Top Candidates to Config Generator")
println("=" ^ 50)

# Configuration
const CSV_PATH = "/project/high_tech_ind/searching-flexibility/figures/mpi_analysis/job3052881/top_candidates.csv"
const TEMPLATE_PATH = "/project/high_tech_ind/searching-flexibility/src/structural_model_heterogenous_preferences/model_parameters.yaml"
const OUTPUT_PATH = "/project/high_tech_ind/searching-flexibility/src/structural_model_heterogenous_preferences/optimized_model_parameters.yaml"

"""Generate optimized config from CSV top candidates"""
function generate_config_from_csv()
    println("📊 Reading top candidates CSV...")
    
    # Read the CSV file
    if !isfile(CSV_PATH)
        error("CSV file not found: $CSV_PATH")
    end
    
    df = CSV.read(CSV_PATH, DataFrame)
    println("✅ Loaded $(nrow(df)) candidates")
    
    # Get the best candidate (rank 1)
    best_candidate = df[df.Rank .== 1, :][1, :]
    println("🏆 Best candidate objective: $(round(best_candidate.Objective, digits=8))")
    
    # Load the template YAML config
    if !isfile(TEMPLATE_PATH)
        error("Template YAML file not found: $TEMPLATE_PATH")
    end
    
    config_dict = YAML.load_file(TEMPLATE_PATH)
    println("✅ Loaded template config")
    
    # Parameter mapping from CSV columns to YAML keys
    # Based on ModelSetup.jl create_primitives_from_yaml function
    csv_to_yaml_mapping = Dict(
        # CSV column => YAML key in ModelParameters
        "κ₀" => "kappa0",      # κ₀ -> kappa0
        "A₁" => "A1",          # A₁ -> A1  
        "ψ₀" => "psi_0",       # ψ₀ -> psi_0
        "ϕ" => "phi",          # ϕ -> phi
        "ν" => "nu",           # ν -> nu
        "c₀" => "c0",          # c₀ -> c0
        "χ" => "chi",          # χ -> chi
        "μ" => "mu",           # μ -> mu
        "aₕ" => "a_h",         # aₕ -> a_h
        "bₕ" => "b_h",         # bₕ -> b_h
    )
    
    # Update ModelParameters with optimized values
    model_params = config_dict["ModelParameters"]
    
    println("\n📊 PARAMETER UPDATES:")
    println("-" ^ 60)
    
    for (csv_col, yaml_key) in csv_to_yaml_mapping
        if csv_col in names(df)
            old_value = get(model_params, yaml_key, "N/A")
            new_value = best_candidate[Symbol(csv_col)]
            model_params[yaml_key] = new_value
            println("  $(rpad(yaml_key, 12)): $(rpad(string(old_value), 15)) → $(round(new_value, digits=6))")
        else
            @warn "Column $csv_col not found in CSV"
        end
    end
    
    # Note: Some parameters not in the search (like beta, delta, b, xi, gamma0, gamma1, A0, kappa1)
    # remain at their template values
    
    # Add metadata as comments
    best_obj = best_candidate.Objective
    comment_header = """# Optimized parameters from MPI search - Top Candidates CSV
# Best objective value: $(round(best_obj, digits=8))
# Generated from: $(basename(CSV_PATH))
# Generated: $(Dates.now())
# 
# Parameters updated from search results:
#   κ₀, A₁, ψ₀, ϕ, ν, c₀, χ, μ, aₕ, bₕ
# Parameters kept from template:
#   beta, delta, b, xi, gamma0, gamma1, A0, kappa1

"""
    
    # Save the updated config
    try
        YAML.write_file(OUTPUT_PATH, config_dict)
        
        # Read back and add header comment
        content = read(OUTPUT_PATH, String)
        open(OUTPUT_PATH, "w") do io
            write(io, comment_header * content)
        end
        
        println("\n✅ Optimized config saved to: $OUTPUT_PATH")
        
        # Display summary
        println("\n📋 OPTIMIZED CONFIGURATION SUMMARY:")
        println("=" ^ 50)
        for (csv_col, yaml_key) in csv_to_yaml_mapping
            if csv_col in names(df)
                value = best_candidate[Symbol(csv_col)]
                println("  $(rpad(yaml_key, 12)): $(round(value, digits=6))")
            end
        end
        
        println("\n🏆 Best objective value: $(round(best_obj, digits=8))")
        
        return OUTPUT_PATH
        
    catch e
        @error "Failed to save config file: $e"
        return nothing
    end
end

"""Display all top candidates for comparison"""
function display_top_candidates(n_show::Int=5)
    if !isfile(CSV_PATH)
        return
    end
    
    df = CSV.read(CSV_PATH, DataFrame)
    n_candidates = min(n_show, nrow(df))
    
    println("\n" * "="^80)
    println("🏆 TOP $n_candidates CANDIDATES COMPARISON")
    println("="^80)
    
    # Show key columns
    key_cols = [:Rank, :Objective, :κ₀, :A₁, :ψ₀, :ϕ, :ν, :c₀, :χ, :μ, :aₕ, :bₕ]
    display_df = df[1:n_candidates, key_cols]
    
    # Format for better display
    for row in eachrow(display_df)
        println("Rank $(row.Rank): Obj=$(round(row.Objective, digits=6))")
        for col in key_cols[3:end]  # Skip Rank and Objective
            println("  $(rpad(string(col), 8)): $(round(row[col], digits=6))")
        end
        println()
    end
end

"""Compare original vs optimized parameters"""
function compare_configs()
    println("\n" * "="^80)
    println("📊 ORIGINAL vs OPTIMIZED PARAMETER COMPARISON")
    println("="^80)
    
    if !isfile(TEMPLATE_PATH) || !isfile(OUTPUT_PATH)
        @warn "Cannot compare - missing config files"
        return
    end
    
    original = YAML.load_file(TEMPLATE_PATH)
    optimized = YAML.load_file(OUTPUT_PATH)
    
    orig_params = original["ModelParameters"]
    opt_params = optimized["ModelParameters"]
    
    println("$(rpad("Parameter", 15))$(rpad("Original", 15))$(rpad("Optimized", 15))$(rpad("Change (%)", 15))")
    println("-" ^ 60)
    
    # Get all parameters that exist in both configs
    common_params = intersect(keys(orig_params), keys(opt_params))
    
    for param in sort(collect(common_params))
        orig_val = orig_params[param]
        opt_val = opt_params[param]
        
        if orig_val != 0
            change_pct = ((opt_val - orig_val) / orig_val) * 100
            
            println(rpad(param, 15) * 
                   rpad(@sprintf("%.6f", orig_val), 15) * 
                   rpad(@sprintf("%.6f", opt_val), 15) * 
                   @sprintf("%.2f%%", change_pct))
        else
            println(rpad(param, 15) * 
                   rpad(@sprintf("%.6f", orig_val), 15) * 
                   rpad(@sprintf("%.6f", opt_val), 15) * 
                   "N/A")
        end
    end
end

"""Main execution"""
function main()
    try
        # Show top candidates
        display_top_candidates(3)
        
        # Generate optimized config
        config_path = generate_config_from_csv()
        
        if config_path !== nothing
            # Compare configs
            compare_configs()
            
            println("\n" * "="^80)
            println("🎯 SUCCESS! Optimized config generation complete!")
            println("="^80)
            println("📁 Input CSV: $CSV_PATH")
            println("📁 Template: $TEMPLATE_PATH") 
            println("📁 Output: $OUTPUT_PATH")
            println("\n💡 To use the optimized parameters:")
            println("   1. Backup your current model_parameters.yaml")
            println("   2. Replace with optimized_model_parameters.yaml")
            println("   3. Run: julia run_file.jl")
            
            # Verify YAML validity
            try
                test_config = YAML.load_file(OUTPUT_PATH)
                println("\n✅ Generated config file is valid YAML")
                println("📊 ModelParameters: $(length(test_config["ModelParameters"])) parameters")
                println("📊 ModelGrids: $(length(test_config["ModelGrids"])) settings")
                println("📊 ModelSolverOptions: $(length(test_config["ModelSolverOptions"])) options")
            catch e
                @warn "Generated config has YAML syntax errors: $e"
            end
        else
            println("\n❌ Config generation failed!")
        end
        
    catch e
        println("❌ Error: $e")
        rethrow(e)
    end
end

# Run the script
main()

println("\n🎯 Config generation from CSV complete!")
