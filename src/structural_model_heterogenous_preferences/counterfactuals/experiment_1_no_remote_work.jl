#!/usr/bin/env julia
#==========================================================================================
# Counterfactual Experiment 1: No Remote Work Technology
# Author: Generated for searching-flexibility project
# Date: 2025-08-27
# Description: Eliminates remote work technology to measure its impact on equilibrium
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
    run_experiment_1(config_path="counterfactual_config.yaml")

Run the No Remote Work Technology counterfactual experiment.
"""
# function run_experiment_1()
    
    println("="^80)
    println("COUNTERFACTUAL EXPERIMENT 1: NO REMOTE WORK TECHNOLOGY")
    println("="^80)
    
    # Load configuration
    config_file = joinpath(@__DIR__, config_path)
    config = YAML.load_file(config_file)
    
    exp_config = config["Experiment1_NoRemoteWork"]
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
                verbose=common_config["solver_options"]["verbose"])
    
    baseline_moments = compute_model_moments(
                                            baseline_prim,
                                            baseline_res,
                                            common_config["simulation_data_path"],
                                            include_moments = Symbol.(common_config["moments_to_use"])
                                            )
    
    # Setup counterfactual model
    println("🔧 Setting up counterfactual model (No Remote Work)...")
    cf_params = deepcopy(base_params)
    
    # Apply parameter overrides
    for (param, value) in exp_config["parameter_overrides"]
        if haskey(cf_params["ModelParameters"], param)
            cf_params["ModelParameters"][param] = value
            println("   📝 Override: $param = $value")
        end
    end
    
    # Create counterfactual primitives
    param_changes = Dict(Symbol(k) => v for (k,v) in exp_config["parameter_overrides"])
    
    cf_prim, cf_res = update_primitives_results(
        baseline_prim,
        baseline_res, 
        param_changes  # Now it's positional
    );
    # Solve counterfactual
    println("⚙️  Solving counterfactual model...")
    solve_model(cf_prim, cf_res; 
                tol=common_config["solver_options"]["tol"],
                max_iter=common_config["solver_options"]["max_iter"],
                verbose=common_config["solver_options"]["verbose"])
    
    cf_moments = compute_model_moments(
                                        cf_prim,
                                        cf_res,
                                        common_config["simulation_data_path"],
                                        include_moments = Symbol.(common_config["moments_to_use"])
                                        )
    
    # Compare results
    println("📊 Computing comparison metrics...")
    
    results = Dict(
        "experiment" => exp_config["name"],
        "description" => exp_config["description"],
        "timestamp" => string(Dates.now()),
        "baseline_moments" => baseline_moments,
        "counterfactual_moments" => cf_moments,
        "parameter_changes" => exp_config["parameter_overrides"]
    )
    
    # Calculate percentage changes
    pct_changes = Dict()
    for (key, baseline_val) in baseline_moments
        if haskey(cf_moments, key) && baseline_val != 0
            cf_val = cf_moments[key]
            pct_changes[key] = 100 * (cf_val - baseline_val) / baseline_val
        end
    end
    results["percentage_changes"] = pct_changes
    
    # Display key results
    println("\n📊 KEY RESULTS COMPARISON:")
    println("─"^60)
    for (metric, pct_change) in pct_changes
        @printf("%-25s: %+7.2f%%\n", metric, pct_change)
    end
    
    # Save results
    #! Save results disabled until we run a "real" experiment
    # if common_config["save_results"]
    #     output_prefix = exp_config["output_prefix"]
    #     timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        
    #     if "yaml" in common_config["save_formats"]
    #         yaml_file = joinpath(output_dir, "$(output_prefix)_$(timestamp).yaml")
    #         YAML.write_file(yaml_file, results)
    #         println("\n💾 Results saved to: $yaml_file")
    #     end
        
    #     if "csv" in common_config["save_formats"]
    #         # Create a DataFrame for CSV output
    #         df = DataFrame()
    #         df.metric = collect(keys(baseline_moments))
    #         df.baseline = [baseline_moments[k] for k in df.metric]
    #         df.counterfactual = [cf_moments[k] for k in df.metric]
    #         df.pct_change = [get(pct_changes, k, missing) for k in df.metric]
            
    #         csv_file = joinpath(output_dir, "$(output_prefix)_$(timestamp).csv")
    #         CSV.write(csv_file, df)
    #         println("💾 Results saved to: $csv_file")
    #     end
    # end
    
    println("\n✅ Experiment 1 completed successfully!")
    return results
end

run_experiment_1()

# Command line interface
# if abspath(PROGRAM_FILE) == @__FILE__
#     if length(ARGS) > 0
#         config_path = ARGS[1]
#     else
#         config_path = "counterfactual_config.yaml"
#     end
    
#     try
#         results = run_experiment_1(config_path)
#         println("\n🎉 All done!")
#     catch e
#         println("\n❌ Error running experiment: $e")
#         rethrow(e)
#     end
# end
