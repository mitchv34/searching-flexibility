#!/usr/bin/env julia
#==========================================================================================
# Counterfactual Experiment 6: Worker Bargaining Power
# Author: Generated for searching-flexibility project
# Date: 2025-08-27
# Description: Examines different levels of worker bargaining power
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
    run_experiment_6(config_path="counterfactual_config.yaml")

Run the Worker Bargaining Power counterfactual experiment.
"""
function run_experiment_6(config_path::String="counterfactual_config.yaml")
    
    println("="^80)
    println("COUNTERFACTUAL EXPERIMENT 6: WORKER BARGAINING POWER")
    println("="^80)
    
    # Load configuration
    config_file = joinpath(@__DIR__, config_path)
    config = YAML.load_file(config_file)
    
    exp_config = config["Experiment6_BargainingPower"]
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
    baseline_xi = baseline_prim.ξ
    
    # Get bargaining power levels to test
    xi_levels = exp_config["parameter_sweeps"]["xi"]
    
    println("🔄 Running $(length(xi_levels)) bargaining power levels...")
    
    # Storage for results
    all_results = DataFrame()
    detailed_results = Dict()
    
    for (i, xi_val) in enumerate(xi_levels)
        println("   📊 Run $i/$(length(xi_levels)): ξ = $xi_val")
        
        # Setup counterfactual
        param_overrides = Dict(:ξ => xi_val)
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
            
            # Additional bargaining-related metrics
            avg_wage = sum(cf_res.w_policy .* cf_res.n) / sum(cf_res.n)
            avg_surplus = sum(cf_res.S .* cf_res.n) / sum(cf_res.n)
            avg_worker_value = sum(cf_res.U .* cf_res.u) / sum(cf_res.u)
            
            # Worker surplus share (wage minus unemployment value vs total surplus)
            employment_weight = cf_res.n ./ sum(cf_res.n)
            weighted_wage = sum(cf_res.w_policy .* employment_weight)
            weighted_unemp_val = sum(cf_res.U .* employment_weight)
            weighted_surplus = sum(cf_res.S .* employment_weight)
            worker_surplus_share = (weighted_wage - weighted_unemp_val) / weighted_surplus
            
            # Market outcomes
            market_tightness = cf_res.θ
            job_finding_rate = cf_res.p
            vacancy_filling_rate = cf_res.q
            total_vacancies = sum(cf_res.v)
            total_unemployment = sum(cf_res.u)
            unemployment_rate = total_unemployment / (total_unemployment + sum(cf_res.n))
            
            # Store results
            run_data = Dict(
                "run_id" => i,
                "xi_bargaining_power" => xi_val,
                "avg_wage" => avg_wage,
                "avg_surplus" => avg_surplus,
                "avg_worker_value" => avg_worker_value,
                "worker_surplus_share" => worker_surplus_share,
                "market_tightness" => market_tightness,
                "job_finding_rate" => job_finding_rate,
                "vacancy_filling_rate" => vacancy_filling_rate,
                "unemployment_rate" => unemployment_rate,
                "total_vacancies" => total_vacancies,
                "total_unemployment" => total_unemployment,
                "moments" => cf_moments,
                "pct_changes" => pct_changes,
                "converged" => true
            )
            
            # Add to summary DataFrame
            row = DataFrame(
                run_id = i,
                xi_bargaining_power = xi_val,
                avg_wage = avg_wage,
                avg_surplus = avg_surplus,
                avg_worker_value = avg_worker_value,
                worker_surplus_share = worker_surplus_share,
                market_tightness = market_tightness,
                job_finding_rate = job_finding_rate,
                vacancy_filling_rate = vacancy_filling_rate,
                unemployment_rate = unemployment_rate,
                total_vacancies = total_vacancies,
                total_unemployment = total_unemployment,
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
            detailed_results["xi_$(xi_val)"] = run_data
            
        catch e
            println("     ⚠️  Failed to converge: $e")
            
            # Store failed run
            run_data = Dict(
                "run_id" => i,
                "xi_bargaining_power" => xi_val,
                "error" => string(e),
                "converged" => false
            )
            
            row = DataFrame(
                run_id = i,
                xi_bargaining_power = xi_val,
                converged = false
            )
            
            append!(all_results, row)
            detailed_results["xi_$(xi_val)"] = run_data
        end
    end
    
    # Compile final results
    results = Dict(
        "experiment" => exp_config["name"],
        "description" => exp_config["description"],
        "timestamp" => string(Dates.now()),
        "baseline_moments" => baseline_moments,
        "baseline_xi" => baseline_xi,
        "xi_levels" => xi_levels,
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
        println("\n📈 BARGAINING POWER EFFECTS (successful runs only):")
        successful_runs = all_results[all_results.converged .== true, :]
        
        # Show key bargaining and welfare outcomes
        println("Bargaining and Wage Outcomes:")
        @printf("%-25s: Min %8.4f, Max %8.4f, Std %8.4f\n", 
                "Worker Surplus Share", minimum(successful_runs.worker_surplus_share), 
                maximum(successful_runs.worker_surplus_share), std(successful_runs.worker_surplus_share))
        @printf("%-25s: Min %8.4f, Max %8.4f, Std %8.4f\n", 
                "Average Wage", minimum(successful_runs.avg_wage), 
                maximum(successful_runs.avg_wage), std(successful_runs.avg_wage))
        @printf("%-25s: Min %8.4f, Max %8.4f, Std %8.4f\n", 
                "Average Surplus", minimum(successful_runs.avg_surplus), 
                maximum(successful_runs.avg_surplus), std(successful_runs.avg_surplus))
        @printf("%-25s: Min %8.4f, Max %8.4f, Std %8.4f\n", 
                "Worker Unemployment Value", minimum(successful_runs.avg_worker_value), 
                maximum(successful_runs.avg_worker_value), std(successful_runs.avg_worker_value))
        
        println("\nLabor Market Outcomes:")
        @printf("%-25s: Min %8.4f, Max %8.4f, Std %8.4f\n", 
                "Market Tightness", minimum(successful_runs.market_tightness), 
                maximum(successful_runs.market_tightness), std(successful_runs.market_tightness))
        @printf("%-25s: Min %8.4f, Max %8.4f, Std %8.4f\n", 
                "Job Finding Rate", minimum(successful_runs.job_finding_rate), 
                maximum(successful_runs.job_finding_rate), std(successful_runs.job_finding_rate))
        @printf("%-25s: Min %8.4f, Max %8.4f, Std %8.4f\n", 
                "Unemployment Rate", minimum(successful_runs.unemployment_rate), 
                maximum(successful_runs.unemployment_rate), std(successful_runs.unemployment_rate))
        
        println("\nBargaining Power vs Key Outcomes:")
        for row in eachrow(successful_runs)
            @printf("  ξ = %4.2f → Wage = %6.3f, Surplus Share = %5.3f, Unemp = %5.2f%%\n", 
                    row.xi_bargaining_power, row.avg_wage, row.worker_surplus_share, row.unemployment_rate * 100)
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
    
    println("\n✅ Experiment 6 completed successfully!")
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
        results, summary_df = run_experiment_6(config_path)
        println("\n🎉 All done!")
    catch e
        println("\n❌ Error running experiment: $e")
        rethrow(e)
    end
end
