#!/usr/bin/env julia
#==========================================================================================
# Utility: Update Estimated Parameters
# Author: Generated for searching-flexibility project
# Date: 2025-08-27
# Description: Utility to update estimated_parameters_2024.yaml with new estimates
==========================================================================================#

using CSV, DataFrames, YAML, Dates

"""
    update_estimated_parameters(csv_file_path, output_yaml_path; 
                               objective_value=nothing, notes="")

Update the estimated parameters YAML file with new parameter estimates from a CSV file.

# Arguments
- `csv_file_path`: Path to CSV file with columns 'parameter' and 'estimate'
- `output_yaml_path`: Path to the YAML file to update
- `objective_value`: Optional final objective function value
- `notes`: Optional additional notes about the estimation

# CSV Format Expected
The CSV should have columns:
- `parameter`: Parameter name (e.g., "aₕ", "c₀", "μ", etc.)
- `estimate`: Estimated value

# Example Usage
```julia
update_estimated_parameters(
    "output/best_parameters_20250827_140128.csv",
    "data/results/estimated_parameters/estimated_parameters_2024.yaml";
    objective_value=0.00142,
    notes="Final estimates from MPI search + LBFGS refinement"
)
```
"""
function update_estimated_parameters(csv_file_path::String, output_yaml_path::String; 
                                   objective_value=nothing, notes="")
    
    println("🔄 Updating estimated parameters...")
    println("   📁 Source: $csv_file_path")
    println("   📁 Target: $output_yaml_path")
    
    # Load current YAML structure
    if isfile(output_yaml_path)
        current_config = YAML.load_file(output_yaml_path)
        println("   ✅ Loaded existing configuration")
    else
        error("YAML template file not found: $output_yaml_path")
    end
    
    # Load new estimates from CSV
    if !isfile(csv_file_path)
        error("CSV file not found: $csv_file_path")
    end
    
    estimates_df = CSV.read(csv_file_path, DataFrame)
    println("   📊 Loaded $(nrow(estimates_df)) parameter estimates")
    
    # Parameter name mapping (handle Unicode/ASCII variations)
    param_mapping = Dict(
        "aₕ" => "a_h", "a_h" => "a_h",
        "bₕ" => "b_h", "b_h" => "b_h", 
        "c₀" => "c0", "c0" => "c0",
        "μ" => "mu", "mu" => "mu",
        "χ" => "chi", "chi" => "chi",
        "ν" => "nu", "nu" => "nu",
        "ψ₀" => "psi_0", "psi_0" => "psi_0",
        "ϕ" => "phi", "phi" => "phi",
        "κ₀" => "kappa0", "kappa0" => "kappa0",
        "κ₁" => "kappa1", "kappa1" => "kappa1",
        "γ₀" => "gamma0", "gamma0" => "gamma0", 
        "γ₁" => "gamma1", "gamma1" => "gamma1",
        "β" => "beta", "beta" => "beta",
        "δ" => "delta", "delta" => "delta",
        "ξ" => "xi", "xi" => "xi",
        "A₀" => "A0", "A0" => "A0",
        "A₁" => "A1", "A1" => "A1"
    )
    
    # Update parameters in the configuration
    updated_count = 0
    for row in eachrow(estimates_df)
        param_name = string(row.parameter)
        param_value = row.estimate
        
        # Map to standard ASCII name
        yaml_param_name = get(param_mapping, param_name, param_name)
        
        if haskey(current_config["ModelParameters"], yaml_param_name)
            old_value = current_config["ModelParameters"][yaml_param_name]
            current_config["ModelParameters"][yaml_param_name] = param_value
            println("   📝 Updated $yaml_param_name: $old_value → $param_value")
            updated_count += 1
        else
            println("   ⚠️  Parameter $yaml_param_name not found in YAML structure")
        end
    end
    
    # Update metadata
    current_config["EstimationInfo"]["estimation_date"] = string(Dates.today())
    if objective_value !== nothing
        current_config["EstimationInfo"]["final_objective_value"] = objective_value
    end
    if notes != ""
        current_config["EstimationInfo"]["notes"] = string(current_config["EstimationInfo"]["notes"], "\n\nUpdate $(Dates.now()): ", notes)
    end
    
    # Save updated configuration
    YAML.write_file(output_yaml_path, current_config)
    
    println("   ✅ Updated $updated_count parameters")
    println("   💾 Saved to: $output_yaml_path")
    println("🎉 Parameter update completed!")
    
    return current_config
end

"""
    update_from_latest_results(results_dir="output"; file_pattern="best_parameters_*.csv")

Find the most recent parameter estimates and update the YAML file.
"""
function update_from_latest_results(results_dir::String="output"; 
                                  file_pattern::String="best_parameters_*.csv",
                                  target_yaml::String="data/results/estimated_parameters/estimated_parameters_2024.yaml")
    
    println("🔍 Looking for latest parameter estimates...")
    
    # Find all matching CSV files
    csv_files = []
    if isdir(results_dir)
        for file in readdir(results_dir)
            if occursin(r"best_parameters_.*\.csv", file)
                push!(csv_files, joinpath(results_dir, file))
            end
        end
    end
    
    if isempty(csv_files)
        println("   ❌ No parameter files found in $results_dir")
        return nothing
    end
    
    # Sort by modification time (most recent first)
    csv_files_with_time = [(f, stat(f).mtime) for f in csv_files]
    sort!(csv_files_with_time, by=x->x[2], rev=true)
    
    latest_file = csv_files_with_time[1][1]
    latest_time = Dates.unix2datetime(csv_files_with_time[1][2])
    
    println("   📁 Latest file: $latest_file")
    println("   🕒 Modified: $latest_time")
    
    # Try to find corresponding objective value from summary file
    objective_value = nothing
    summary_file = replace(latest_file, "best_parameters" => "optimization_summary", ".csv" => ".json")
    if isfile(summary_file)
        try
            using JSON3
            summary = JSON3.read(summary_file)
            objective_value = get(summary, "best_objective", nothing)
            println("   📊 Found objective value: $objective_value")
        catch e
            println("   ⚠️  Could not read objective value: $e")
        end
    end
    
    # Update the YAML file
    update_estimated_parameters(latest_file, target_yaml; 
                               objective_value=objective_value, 
                               notes="Auto-updated from latest optimization results")
    
    return latest_file
end

# Command line interface
if abspath(PROGRAM_FILE) == @__FILE__
    using ArgParse
    
    function parse_commandline()
        s = ArgParseSettings()
        @add_arg_table s begin
            "--csv", "-c"
                help = "Path to CSV file with parameter estimates"
                arg_type = String
            "--yaml", "-y" 
                help = "Path to YAML file to update"
                arg_type = String
                default = "data/results/estimated_parameters/estimated_parameters_2024.yaml"
            "--objective", "-o"
                help = "Final objective function value"
                arg_type = Float64
            "--notes", "-n"
                help = "Additional notes about the estimation"
                arg_type = String
                default = ""
            "--auto", "-a"
                help = "Automatically find and use latest results"
                action = :store_true
        end
        return parse_args(s)
    end
    
    args = parse_commandline()
    
    try
        if args["auto"]
            # Auto-update from latest results
            update_from_latest_results()
        elseif args["csv"] !== nothing
            # Manual update from specified CSV
            update_estimated_parameters(args["csv"], args["yaml"]; 
                                      objective_value=args["objective"],
                                      notes=args["notes"])
        else
            println("❌ Must specify either --csv or --auto")
            println("Examples:")
            println("  julia update_estimated_parameters.jl --auto")
            println("  julia update_estimated_parameters.jl --csv output/best_parameters_20250827_140128.csv")
        end
    catch e
        println("❌ Error: $e")
        exit(1)
    end
end
