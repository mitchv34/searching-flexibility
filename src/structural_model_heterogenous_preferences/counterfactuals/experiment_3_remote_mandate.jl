#!/usr/bin/env julia
#==========================================================================================
# Counterfactual Experiment 3: Remote Work Mandate
# Author: Generated for searching-flexibility project
# Date: 2025-08-27
# Description: Policy experiment mandating minimum remote work levels
# Note: Uses solve_model_rto function (to be implemented) with alpha_mandated parameter
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
    solve_model_rto(prim, res; alpha_mandated=0.0, kwargs...)

Placeholder for the Remote-To-Office mandate solver function.
This function will be implemented later to solve the model with a minimum
remote work constraint alpha_mandated.

TODO: Implement this function to handle mandatory remote work levels.
"""
function solve_model_rto(prim::Primitives{T}, res::Results{T}; 
                        alpha_mandated::Float64=0.0, kwargs...) where {T<:Real}
    
    # TODO: This is a placeholder implementation
    # The actual function should modify the solver to enforce alpha >= alpha_mandated constraint
    
    println("   ⚠️  Using placeholder solve_model_rto (alpha_mandated = $alpha_mandated)")
    println("   📝 TODO: Implement constrained solver for remote work mandates")
    
    # For now, just call the regular solver and warn
    # In the actual implementation, this should modify the optimization problem
    # to include the constraint that alpha(h,psi) >= alpha_mandated
    
    return solve_model(prim, res; kwargs...)
end

"""
    run_experiment_3(config_path="counterfactual_config.yaml")

Run the Remote Work Mandate counterfactual experiment.
"""
function run_experiment_3(config_path::String="counterfactual_config.yaml")
    
    println("="^80)
    println("COUNTERFACTUAL EXPERIMENT 3: REMOTE WORK MANDATE")
    println("="^80)
    
    # Load configuration
    config_file = joinpath(@__DIR__, config_path)
    config = YAML.load_file(config_file)
    
    exp_config = config["Experiment3_RemoteMandate"]
    common_config = config["CommonSettings"]
    base_config = config["BaseModelConfig"]
    
    println("📋 Experiment: $(exp_config["name"])")
    println("📝 Description: $(exp_config["description"])")
    println("⚖️  Solver: $(exp_config["solver_function"])")
    println()
    
    # Load base model parameters
    base_params_path = base_config["config_path"]
    base_params = YAML.load_file(base_params_path)
    
    # Create output directory
    output_dir = common_config["output_directory"]
    mkpath(output_dir)
    
    # Setup baseline model (no mandate)
    println("🔧 Setting up baseline model (no mandate)...")
    baseline_prim = create_primitives_from_yaml(base_params_path)
    baseline_res = Results(baseline_prim)
    
    # Apply any parameter overrides to baseline
    if !isempty(exp_config["parameter_overrides"])
        param_overrides = Dict(Symbol(k) => v for (k,v) in exp_config["parameter_overrides"])
        baseline_prim, baseline_res = update_primitives_results(baseline_prim, baseline_res, param_overrides)
    end
    
    # Solve baseline (no mandate)
    println("⚙️  Solving baseline model (alpha_mandated = 0.0)...")
    solve_model_rto(baseline_prim, baseline_res; 
                   alpha_mandated=0.0,
                   tol=common_config["solver_options"]["tol"],
                   max_iter=common_config["solver_options"]["max_iter"],
                   verbose=false)
    
    baseline_moments = compute_model_moments(baseline_prim, baseline_res)
    
    # Get mandate levels to test
    mandate_levels = exp_config["mandate_levels"]["alpha_mandated"]
    
    println("🔄 Running $(length(mandate_levels)) mandate levels...")
    
    # Storage for results
    all_results = DataFrame()
    detailed_results = Dict()
    
    for (i, alpha_mandate) in enumerate(mandate_levels)
        println("   📊 Run $i/$(length(mandate_levels)): alpha_mandated = $alpha_mandate")
        
        # Create fresh copy for this mandate level
        cf_prim = deepcopy(baseline_prim)
        cf_res = Results(cf_prim)
        
        # Solve with mandate
        try
            solve_model_rto(cf_prim, cf_res; 
                           alpha_mandated=alpha_mandate,
                           tol=common_config["solver_options"]["tol"],
                           max_iter=common_config["solver_options"]["max_iter"],
                           verbose=false)
            
            cf_moments = compute_model_moments(cf_prim, cf_res)
            
            # Calculate percentage changes from baseline
            pct_changes = Dict()
            for (key, baseline_val) in baseline_moments
                if haskey(cf_moments, key) && baseline_val != 0
                    cf_val = cf_moments[key]
                    pct_changes[key] = 100 * (cf_val - baseline_val) / baseline_val
                end
            end
            
            # Calculate actual remote work compliance
            # (This would use the solved alpha_policy from cf_res)
            avg_alpha = mean(cf_res.α_policy[cf_res.n .> 0])  # Employment-weighted average
            compliance_rate = mean(cf_res.α_policy[cf_res.n .> 0] .>= alpha_mandate)
            
            # Store results
            run_data = Dict(
                "run_id" => i,
                "alpha_mandated" => alpha_mandate,
                "actual_avg_alpha" => avg_alpha,
                "compliance_rate" => compliance_rate,
                "moments" => cf_moments,
                "pct_changes" => pct_changes,
                "converged" => true
            )
            
            # Add to summary DataFrame
            row = DataFrame(
                run_id = i,
                alpha_mandated = alpha_mandate,
                actual_avg_alpha = avg_alpha,
                compliance_rate = compliance_rate,
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
            detailed_results["mandate_$(alpha_mandate)"] = run_data
            
        catch e
            println("     ⚠️  Failed to converge: $e")
            
            # Store failed run
            run_data = Dict(
                "run_id" => i,
                "alpha_mandated" => alpha_mandate,
                "error" => string(e),
                "converged" => false
            )
            
            row = DataFrame(
                run_id = i,
                alpha_mandated = alpha_mandate,
                converged = false
            )
            
            append!(all_results, row)
            detailed_results["mandate_$(alpha_mandate)"] = run_data
        end
    end
    
    # Compile final results
    results = Dict(
        "experiment" => exp_config["name"],
        "description" => exp_config["description"],
        "solver_function" => exp_config["solver_function"],
        "timestamp" => string(Dates.now()),
        "baseline_moments" => baseline_moments,
        "mandate_levels" => mandate_levels,
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
        println("\n📈 MANDATE POLICY EFFECTS (successful runs only):")
        successful_runs = all_results[all_results.converged .== true, :]
        
        # Show compliance and actual remote work levels
        println("Alpha Mandate → Actual Alpha (Compliance Rate):")
        for row in eachrow(successful_runs)
            @printf("  %4.1f%% → %6.1f%% (%5.1f%% compliant)\n", 
                    row.alpha_mandated * 100, 
                    row.actual_avg_alpha * 100,
                    row.compliance_rate * 100)
        end
        
        println("\nEconomic Impact by Mandate Level:")
        for moment_col in names(successful_runs)
            if endswith(string(moment_col), "_pct") && eltype(successful_runs[!, moment_col]) <: Number
                moment_name = replace(string(moment_col), "_pct" => "")
                pct_values = successful_runs[!, moment_col]
                @printf("%-25s: Min %+7.2f%%, Max %+7.2f%%, Final %+7.2f%%\n", 
                        moment_name, minimum(pct_values), maximum(pct_values), last(pct_values))
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
    
    println("\n✅ Experiment 3 completed successfully!")
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
        results, summary_df = run_experiment_3(config_path)
        println("\n🎉 All done!")
        println("\n📝 Note: Remember to implement solve_model_rto function for actual mandate constraints!")
    catch e
        println("\n❌ Error running experiment: $e")
        rethrow(e)
    end
end
