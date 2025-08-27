#!/usr/bin/env julia
#==========================================================================================
# Master Counterfactual Runner
# Author: Generated for searching-flexibility project
# Date: 2025-08-27
# Description: Run all counterfactual experiments or specific ones
==========================================================================================#

using Pkg
Pkg.activate(".")

using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--experiments", "-e"
            help = "Comma-separated list of experiments to run (1,2,3,4,5,6) or 'all'"
            arg_type = String
            default = "all"
        "--config", "-c"
            help = "Path to counterfactual configuration file"
            arg_type = String
            default = "counterfactual_config.yaml"
        "--verbose", "-v"
            help = "Verbose output"
            action = :store_true
        "--parallel", "-p"
            help = "Use parallel processing where available"
            action = :store_true
    end

    return parse_args(s)
end

"""
    run_all_experiments(config_path, experiments_to_run, verbose=false)

Run the specified counterfactual experiments.
"""
function run_all_experiments(config_path::String, experiments_to_run::Vector{Int}; verbose::Bool=false)
    
    println("="^80)
    println("MASTER COUNTERFACTUAL EXPERIMENT RUNNER")
    println("="^80)
    println("Config file: $config_path")
    println("Experiments: $(join(experiments_to_run, ", "))")
    println("Timestamp: $(Dates.now())")
    println("="^80)
    
    # Experiment definitions
    experiments = Dict(
        1 => ("No Remote Work Technology", "experiment_1_no_remote_work.jl"),
        2 => ("Remote Work Technology Levels", "experiment_2_remote_tech_levels.jl"),
        3 => ("Remote Work Mandate", "experiment_3_remote_mandate.jl"),
        4 => ("Search Friction Variations", "experiment_4_search_frictions.jl"),
        5 => ("Unemployment Benefit Levels", "experiment_5_unemployment_benefits.jl"),
        6 => ("Worker Bargaining Power", "experiment_6_bargaining_power.jl")
    )
    
    results_summary = Dict()
    total_experiments = length(experiments_to_run)
    
    for (i, exp_num) in enumerate(experiments_to_run)
        if !haskey(experiments, exp_num)
            println("⚠️  Warning: Experiment $exp_num not found, skipping...")
            continue
        end
        
        exp_name, exp_script = experiments[exp_num]
        println("\n" * "─"^80)
        println("🚀 RUNNING EXPERIMENT $exp_num ($i/$total_experiments): $exp_name")
        println("─"^80)
        
        # Construct script path
        script_path = joinpath(@__DIR__, exp_script)
        
        if !isfile(script_path)
            println("❌ Error: Script $script_path not found!")
            results_summary[exp_num] = Dict("status" => "error", "message" => "Script not found")
            continue
        end
        
        try
            # Include and run the experiment
            if verbose
                println("📂 Including script: $script_path")
            end
            
            # Run the experiment function based on the experiment number
            result = if exp_num == 1
                include(script_path)
                run_experiment_1(config_path)
            elseif exp_num == 2
                include(script_path)
                run_experiment_2(config_path)
            elseif exp_num == 3
                include(script_path)
                run_experiment_3(config_path)
            elseif exp_num == 4
                include(script_path)
                run_experiment_4(config_path)
            elseif exp_num == 5
                include(script_path)
                run_experiment_5(config_path)
            elseif exp_num == 6
                include(script_path)
                run_experiment_6(config_path)
            end
            
            results_summary[exp_num] = Dict(
                "status" => "success", 
                "name" => exp_name,
                "timestamp" => string(Dates.now())
            )
            
            println("✅ Experiment $exp_num completed successfully!")
            
        catch e
            println("❌ Error in experiment $exp_num: $e")
            if verbose
                println("Full error:")
                showerror(stdout, e, catch_backtrace())
                println()
            end
            
            results_summary[exp_num] = Dict(
                "status" => "error", 
                "name" => exp_name,
                "error" => string(e),
                "timestamp" => string(Dates.now())
            )
        end
    end
    
    # Print final summary
    println("\n" * "="^80)
    println("📊 FINAL SUMMARY")
    println("="^80)
    
    successful = 0
    failed = 0
    
    for (exp_num, result) in results_summary
        status = result["status"]
        name = get(result, "name", "Unknown")
        
        if status == "success"
            println("✅ Experiment $exp_num: $name - SUCCESS")
            successful += 1
        else
            error_msg = get(result, "error", "Unknown error")
            println("❌ Experiment $exp_num: $name - FAILED ($error_msg)")
            failed += 1
        end
    end
    
    println("\n📈 OVERALL RESULTS:")
    println("   Successful: $successful")
    println("   Failed: $failed")
    println("   Total: $(successful + failed)")
    
    if successful == length(experiments_to_run)
        println("\n🎉 All experiments completed successfully!")
    elseif successful > 0
        println("\n⚠️  Some experiments completed successfully, but $(failed) failed.")
    else
        println("\n💥 All experiments failed!")
    end
    
    return results_summary
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    using Dates
    
    # Parse command line arguments
    parsed_args = parse_commandline()
    
    config_path = parsed_args["config"]
    experiments_str = parsed_args["experiments"]
    verbose = parsed_args["verbose"]
    
    # Parse experiments to run
    if experiments_str == "all"
        experiments_to_run = [1, 2, 3, 4, 5, 6]
    else
        try
            experiments_to_run = parse.(Int, split(experiments_str, ","))
        catch e
            println("❌ Error parsing experiments list: $experiments_str")
            println("   Use format like '1,2,3' or 'all'")
            exit(1)
        end
    end
    
    # Validate experiments
    valid_experiments = [1, 2, 3, 4, 5, 6]
    invalid_experiments = setdiff(experiments_to_run, valid_experiments)
    if !isempty(invalid_experiments)
        println("❌ Invalid experiment numbers: $(join(invalid_experiments, ", "))")
        println("   Valid options: $(join(valid_experiments, ", "))")
        exit(1)
    end
    
    # Check config file exists
    if !isfile(config_path)
        println("❌ Configuration file not found: $config_path")
        exit(1)
    end
    
    try
        # Run experiments
        results = run_all_experiments(config_path, experiments_to_run; verbose=verbose)
        
        # Exit with appropriate code
        failed_count = sum(1 for (_, result) in results if result["status"] == "error")
        exit(failed_count > 0 ? 1 : 0)
        
    catch e
        println("\n💥 Fatal error running experiments: $e")
        if verbose
            showerror(stdout, e, catch_backtrace())
        end
        exit(1)
    end
end
