#!/usr/bin/env julia
#==========================================================================================
# Counterfactual Experiment 4: Search Friction Variations
# Author: Generated for searching-flexibility project
# Date: 2025-08-27
# Description: Examines impact of different search market frictions
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
    run_experiment_4(config_path="counterfactual_config.yaml")

Run the Search Friction Variations counterfactual experiment.
"""
function run_experiment_4(config_path::String="counterfactual_config.yaml")
    
    println("="^80)
    println("COUNTERFACTUAL EXPERIMENT 4: SEARCH FRICTION VARIATIONS")
    println("="^80)
    
    # Load configuration
    config_file = joinpath(@__DIR__, config_path)
    config = YAML.load_file(config_file)
    
    exp_config = config["Experiment4_SearchFrictions"]
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
                verbose=false)
    
    baseline_moments = compute_model_moments(baseline_prim, baseline_res)
    
    # Generate parameter combinations
    kappa0_values = exp_config["parameter_sweeps"]["kappa0"]
    kappa1_values = exp_config["parameter_sweeps"]["kappa1"]
    
    if exp_config["sweep_mode"] == "grid"
        # Grid mode: all combinations
        param_combinations = [(k0, k1) for k0 in kappa0_values for k1 in kappa1_values]
    else
        # Parallel mode: pair up parameters (must be same length)
        if length(kappa0_values) != length(kappa1_values)
            error("In parallel mode, kappa0 and kappa1 arrays must have the same length")
        end
        param_combinations = [(kappa0_values[i], kappa1_values[i]) for i in 1:length(kappa0_values)]
    end
    
    println("🔄 Running $(length(param_combinations)) parameter combinations...")
    
    # Storage for results
    all_results = DataFrame()
    detailed_results = Dict()
    
    for (i, (kappa0_val, kappa1_val)) in enumerate(param_combinations)
        println("   📊 Run $i/$(length(param_combinations)): κ₀=$kappa0_val, κ₁=$kappa1_val")
        
        # Setup counterfactual
        param_overrides = Dict(:κ₀ => kappa0_val, :κ₁ => kappa1_val)
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
            
            # Additional search friction metrics
            market_tightness = cf_res.θ
            job_finding_rate = cf_res.p
            vacancy_filling_rate = cf_res.q
            total_vacancies = sum(cf_res.v)
            total_unemployment = sum(cf_res.u)
            
            # Store results
            run_data = Dict(
                "run_id" => i,
                "kappa0" => kappa0_val,
                "kappa1" => kappa1_val,
                "market_tightness" => market_tightness,
                "job_finding_rate" => job_finding_rate,
                "vacancy_filling_rate" => vacancy_filling_rate,
                "total_vacancies" => total_vacancies,
                "total_unemployment" => total_unemployment,
                "moments" => cf_moments,
                "pct_changes" => pct_changes,
                "converged" => true
            )
            
            # Add to summary DataFrame
            row = DataFrame(
                run_id = i,
                kappa0 = kappa0_val,
                kappa1 = kappa1_val,
                market_tightness = market_tightness,
                job_finding_rate = job_finding_rate,
                vacancy_filling_rate = vacancy_filling_rate,
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
            detailed_results["run_$i"] = run_data
            
        catch e
            println("     ⚠️  Failed to converge: $e")
            
            # Store failed run
            run_data = Dict(
                "run_id" => i,
                "kappa0" => kappa0_val,
                "kappa1" => kappa1_val,
                "error" => string(e),
                "converged" => false
            )
            
            row = DataFrame(
                run_id = i,
                kappa0 = kappa0_val,
                kappa1 = kappa1_val,
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
        println("\n📈 SEARCH FRICTION EFFECTS (successful runs only):")
        successful_runs = all_results[all_results.converged .== true, :]
        
        # Show key search market outcomes
        println("Search Market Outcomes:")
        @printf("%-20s: Min %8.4f, Max %8.4f, Std %8.4f\n", 
                "Market Tightness", minimum(successful_runs.market_tightness), 
                maximum(successful_runs.market_tightness), std(successful_runs.market_tightness))
        @printf("%-20s: Min %8.4f, Max %8.4f, Std %8.4f\n", 
                "Job Finding Rate", minimum(successful_runs.job_finding_rate), 
                maximum(successful_runs.job_finding_rate), std(successful_runs.job_finding_rate))
        @printf("%-20s: Min %8.4f, Max %8.4f, Std %8.4f\n", 
                "Vacancy Fill Rate", minimum(successful_runs.vacancy_filling_rate), 
                maximum(successful_runs.vacancy_filling_rate), std(successful_runs.vacancy_filling_rate))
        
        println("\nOther Economic Outcomes:")
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
    
    println("\n✅ Experiment 4 completed successfully!")
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
        results, summary_df = run_experiment_4(config_path)
        println("\n🎉 All done!")
    catch e
        println("\n❌ Error running experiment: $e")
        rethrow(e)
    end
end
