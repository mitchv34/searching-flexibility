#!/usr/bin/env julia
#==========================================================================================
# Counterfactual Experiment 5: Unemployment Benefit Levels
# Author: Generated for searching-flexibility project
# Date: 2025-08-27
# Description: Tests impact of different unemployment benefit levels
==========================================================================================#

using Pkg
Pkg.activate(".")

using YAML, CSV, DataFrames, Printf, Dates
using Parameters, Statistics

# Include model components
include("../ModelSetup.jl")
include("../ModelSolver.jl") 
include("../ModelEstimation.jl")

"""
    run_experiment_5(config_path="counterfactual_config.yaml")

Run the Unemployment Benefit Levels counterfactual experiment.
"""
function run_experiment_5(config_path::String="counterfactual_config.yaml")
    
    println("="^80)
    println("COUNTERFACTUAL EXPERIMENT 5: UNEMPLOYMENT BENEFIT LEVELS")
    println("="^80)
    
    # Load configuration
    config_file = joinpath(@__DIR__, config_path)
    config = YAML.load_file(config_file)
    
    exp_config = config["Experiment5_UnemploymentBenefits"]
    common_config = config["CommonSettings"]
    base_config = config["BaseModelConfig"]
    
    println("📋 Experiment: $(exp_config["name"])")
    println("📝 Description: $(exp_config["description"])")
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
                verbose=false)
    
    baseline_moments = compute_model_moments(baseline_prim, baseline_res)
    baseline_benefit = baseline_prim.b
    
    # Get benefit levels to test
    benefit_levels = exp_config["parameter_sweeps"]["b"]
    
    println("🔄 Running $(length(benefit_levels)) benefit levels...")
    
    # Storage for results
    all_results = DataFrame()
    detailed_results = Dict()
    
    for (i, b_val) in enumerate(benefit_levels)
        println("   📊 Run $i/$(length(benefit_levels)): b = $b_val")
        
        # Setup counterfactual
        param_overrides = Dict(:b => b_val)
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
            
            # Additional unemployment-related metrics
            total_unemployment = sum(cf_res.u)
            unemployment_rate = total_unemployment / (total_unemployment + sum(cf_res.n))
            avg_unemployment_value = sum(cf_res.U .* cf_res.u) / total_unemployment
            market_tightness = cf_res.θ
            job_finding_rate = cf_res.p
            
            # Replacement rate (benefit relative to average wage)
            avg_wage = sum(cf_res.w_policy .* cf_res.n) / sum(cf_res.n)
            replacement_rate = b_val / avg_wage
            
            # Store results
            run_data = Dict(
                "run_id" => i,
                "benefit_level" => b_val,
                "replacement_rate" => replacement_rate,
                "unemployment_rate" => unemployment_rate,
                "total_unemployment" => total_unemployment,
                "avg_unemployment_value" => avg_unemployment_value,
                "market_tightness" => market_tightness,
                "job_finding_rate" => job_finding_rate,
                "avg_wage" => avg_wage,
                "moments" => cf_moments,
                "pct_changes" => pct_changes,
                "converged" => true
            )
            
            # Add to summary DataFrame
            row = DataFrame(
                run_id = i,
                benefit_level = b_val,
                replacement_rate = replacement_rate,
                unemployment_rate = unemployment_rate,
                total_unemployment = total_unemployment,
                avg_unemployment_value = avg_unemployment_value,
                market_tightness = market_tightness,
                job_finding_rate = job_finding_rate,
                avg_wage = avg_wage,
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
            detailed_results["benefit_$(b_val)"] = run_data
            
        catch e
            println("     ⚠️  Failed to converge: $e")
            
            # Store failed run
            run_data = Dict(
                "run_id" => i,
                "benefit_level" => b_val,
                "error" => string(e),
                "converged" => false
            )
            
            row = DataFrame(
                run_id = i,
                benefit_level = b_val,
                converged = false
            )
            
            append!(all_results, row)
            detailed_results["benefit_$(b_val)"] = run_data
        end
    end
    
    # Compile final results
    results = Dict(
        "experiment" => exp_config["name"],
        "description" => exp_config["description"],
        "timestamp" => string(Dates.now()),
        "baseline_moments" => baseline_moments,
        "baseline_benefit" => baseline_benefit,
        "benefit_levels" => benefit_levels,
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
        println("\n📈 UNEMPLOYMENT BENEFIT EFFECTS (successful runs only):")
        successful_runs = all_results[all_results.converged .== true, :]
        
        # Show key unemployment and welfare outcomes
        println("Labor Market Outcomes:")
        @printf("%-20s: Min %8.4f, Max %8.4f, Std %8.4f\n", 
                "Unemployment Rate", minimum(successful_runs.unemployment_rate), 
                maximum(successful_runs.unemployment_rate), std(successful_runs.unemployment_rate))
        @printf("%-20s: Min %8.4f, Max %8.4f, Std %8.4f\n", 
                "Job Finding Rate", minimum(successful_runs.job_finding_rate), 
                maximum(successful_runs.job_finding_rate), std(successful_runs.job_finding_rate))
        @printf("%-20s: Min %8.4f, Max %8.4f, Std %8.4f\n", 
                "Market Tightness", minimum(successful_runs.market_tightness), 
                maximum(successful_runs.market_tightness), std(successful_runs.market_tightness))
        
        println("\nWelfare and Wage Outcomes:")
        @printf("%-20s: Min %8.4f, Max %8.4f, Std %8.4f\n", 
                "Replacement Rate", minimum(successful_runs.replacement_rate), 
                maximum(successful_runs.replacement_rate), std(successful_runs.replacement_rate))
        @printf("%-20s: Min %8.4f, Max %8.4f, Std %8.4f\n", 
                "Average Wage", minimum(successful_runs.avg_wage), 
                maximum(successful_runs.avg_wage), std(successful_runs.avg_wage))
        @printf("%-20s: Min %8.4f, Max %8.4f, Std %8.4f\n", 
                "Unemploy. Value", minimum(successful_runs.avg_unemployment_value), 
                maximum(successful_runs.avg_unemployment_value), std(successful_runs.avg_unemployment_value))
        
        println("\nBenefit Level vs Unemployment Rate:")
        for row in eachrow(successful_runs)
            @printf("  b = %5.3f → Unemp Rate = %6.3f%% (Repl Rate = %5.3f)\n", 
                    row.benefit_level, row.unemployment_rate * 100, row.replacement_rate)
        end
        
        println("\nOther Economic Impacts:")
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
    
    println("\n✅ Experiment 5 completed successfully!")
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
        results, summary_df = run_experiment_5(config_path)
        println("\n🎉 All done!")
    catch e
        println("\n❌ Error running experiment: $e")
        rethrow(e)
    end
end
