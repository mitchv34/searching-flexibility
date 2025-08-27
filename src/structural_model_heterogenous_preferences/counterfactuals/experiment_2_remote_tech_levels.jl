#!/usr/bin/env julia
#==========================================================================================
# Counterfactual Experiment 2: Remote Work Technology Levels
# Author: Generated for searching-flexibility project
# Date: 2025-08-27
# Description: Tests different levels of remote work technology productivity
==========================================================================================#

using Pkg
Pkg.activate(".")

using YAML, CSV, DataFrames, Printf, Dates
using Parameters, Statistics
global config_path = "counterfactual_config.yaml"

# Include model components
include("../ModelSetup.jl")
include("../ModelSolver.jl") 
include("../ModelEstimation.jl")

"""
    run_experiment_2(config_path="counterfactual_config.yaml")

Run the Remote Work Technology Levels counterfactual experiment.
"""
# function run_experiment_2()
    
    println("="^80)
    println("COUNTERFACTUAL EXPERIMENT 2: REMOTE WORK TECHNOLOGY LEVELS")
    println("="^80)
    
    # Load configuration
    config_file = joinpath(@__DIR__, config_path)
    config = YAML.load_file(config_file)
    
    exp_config = config["Experiment2_RemoteTechLevels"]
    common_config = config["CommonSettings"]
    base_config = config["BaseModelConfig"]
    
    println("📋 Experiment: $(exp_config["name"])")
    println("📝 Description: $(exp_config["description"])")
    println("🔀 Sweep mode: $(exp_config["sweep_mode"])")
    println()
    
    # Load base model parameters
    base_params_path = base_config["config_path"]
    base_params = YAML.load_file(base_params_path)
    
    # Create output directory
    output_dir = common_config["output_directory"]
    mkpath(output_dir)
    
    # Setup baseline model
    println("🔧 Setting up baseline model...")
    baseline_prim = create_primitives_from_yaml(base_params_path)
    baseline_res = Results(baseline_prim)
    
    # Solve baseline
    println("⚙️  Solving baseline model...")
    solve_model(baseline_prim, baseline_res; 
                tol=common_config["solver_options"]["tol"],
                max_iter=common_config["solver_options"]["max_iter"],
                verbose=false)  # Quiet for multiple runs
    
    baseline_moments = compute_model_moments(baseline_prim, baseline_res)
    
    # Generate parameter combinations
    psi_0_values = exp_config["parameter_sweeps"]["ψ₀"]
    nu_values = exp_config["parameter_sweeps"]["ν"]
    
    if exp_config["sweep_mode"] == "parallel"
        # Parallel mode: pair up parameters (must be same length)
        if length(psi_0_values) != length(nu_values)
            error("In parallel mode, psi_0 and nu arrays must have the same length")
        end
        param_combinations = [(psi_0_values[i], nu_values[i]) for i in 1:length(psi_0_values)]
    else
        # Grid mode: all combinations
        param_combinations = [(p, n) for p in psi_0_values for n in nu_values]
    end
    
    println("🔄 Running $(length(param_combinations)) parameter combinations...")
    
    # Storage for results
    all_results = DataFrame()
    detailed_results = Dict()
    
    for (i, (psi_0_val, nu_val)) in enumerate(param_combinations)
        println("   📊 Run $i/$(length(param_combinations)): psi_0=$psi_0_val, nu=$nu_val")
        
        # Setup counterfactual
        param_overrides = Dict(:psi_0 => psi_0_val, :nu => nu_val)
        cf_prim, cf_res = update_primitives_results(baseline_prim, baseline_res, param_overrides)
        
        # Solve counterfactual
        try
            solve_model(cf_prim, cf_res; 
                        tol=common_config["solver_options"]["tol"],
                        max_iter=common_config["solver_options"]["max_iter"],
                        verbose=false)
            
            cf_moments = compute_model_moments(cf_prim, cf_res)
            
            # Calculate percentage changes
            pct_changes = Dict()
            for (key, baseline_val) in baseline_moments
                if haskey(cf_moments, key) && baseline_val != 0
                    cf_val = cf_moments[key]
                    pct_changes[key] = 100 * (cf_val - baseline_val) / baseline_val
                end
            end
            
            # Store results
            run_data = Dict(
                "run_id" => i,
                "psi_0" => psi_0_val,
                "nu" => nu_val,
                "moments" => cf_moments,
                "pct_changes" => pct_changes,
                "converged" => true
            )
            
            # Add to summary DataFrame
            row = DataFrame(
                run_id = i,
                psi_0 = psi_0_val,
                nu = nu_val,
                converged = true
            )
            
            # Add moment values and percentage changes
            for (key, val) in cf_moments
                row[!, Symbol("$(key)_value")] = [val]
                if haskey(pct_changes, key)
                    row[!, Symbol("$(key)_pct")] = [pct_changes[key]]
                end
            end
            
            append!(all_results, row)
            detailed_results["run_$i"] = run_data
            
        catch e
            println("     ⚠️  Failed to converge: $e")
            
            # Store failed run
            run_data = Dict(
                "run_id" => i,
                "psi_0" => psi_0_val,
                "nu" => nu_val,
                "error" => string(e),
                "converged" => false
            )
            
            row = DataFrame(
                run_id = i,
                psi_0 = psi_0_val,
                nu = nu_val,
                converged = false
            )
            
            append!(all_results, row)
            detailed_results["run_$i"] = run_data
        end
    end
    
    # Compile final results
    results = Dict(
        "experiment" => exp_config["name"],
        "description" => exp_config["description"],
        "sweep_mode" => exp_config["sweep_mode"],
        "timestamp" => string(Dates.now()),
        "baseline_moments" => baseline_moments,
        "parameter_combinations" => param_combinations,
        "summary_stats" => detailed_results,
        "n_successful" => sum(all_results.converged),
        "n_total" => nrow(all_results)
    )
    
    # Display summary
    println("\n📊 EXPERIMENT SUMMARY:")
    println("─"^60)
    println("Total runs: $(results["n_total"])")
    println("Successful: $(results["n_successful"])")
    println("Failed: $(results["n_total"] - results["n_successful"])")
    
    if results["n_successful"] > 0
        println("\n📈 PARAMETER SENSITIVITY (successful runs only):")
        successful_runs = all_results[all_results.converged .== true, :]
        
        for moment_col in names(successful_runs)
            if endswith(string(moment_col), "_pct") && eltype(successful_runs[!, moment_col]) <: Number
                moment_name = replace(string(moment_col), "_pct" => "")
                pct_values = successful_runs[!, moment_col]
                @printf("%-25s: Min %+7.2f%%, Max %+7.2f%%, Std %6.2f%%\n", 
                        moment_name, minimum(pct_values), maximum(pct_values), std(pct_values))
            end
        end
    end
    
    # Save results
    if common_config["save_results"]
        output_prefix = exp_config["output_prefix"]
        timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        
        if "yaml" in common_config["save_formats"]
            yaml_file = joinpath(output_dir, "$(output_prefix)_$(timestamp).yaml")
            YAML.write_file(yaml_file, results)
            println("\n💾 Detailed results saved to: $yaml_file")
        end
        
        if "csv" in common_config["save_formats"]
            csv_file = joinpath(output_dir, "$(output_prefix)_$(timestamp).csv")
            CSV.write(csv_file, all_results)
            println("💾 Summary table saved to: $csv_file")
        end
    end
    
    println("\n✅ Experiment 2 completed successfully!")
    return results, all_results
end

# Command line interface
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) > 0
        config_path = ARGS[1]
    else
        config_path = "counterfactual_config.yaml"
    end
    
    try
        results, summary_df = run_experiment_2(config_path)
        println("\n🎉 All done!")
    catch e
        println("\n❌ Error running experiment: $e")
        rethrow(e)
    end
end
