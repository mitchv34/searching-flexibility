#==========================================================================================
Module: ComparativeStatics.jl
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2025-02-27
Description: Module for running comparative statics analysis on the random search 
        labor market model. Provides functionality to vary parameters systematically
        and collect results for analysis.
==========================================================================================#

module ComparativeStatics

using Parameters, Printf, DataFrames, CSV, YAML, OrderedCollections
using ..Types, ..ModelFunctions, ..ModelSolver, ..ModelRunner
using Statistics, Dates

export ParameterGrid, ComparativeStaticsConfig, ComparativeStaticsResults
export create_parameter_grid, run_comparative_statics, save_results, load_results
export plot_comparative_statics, extract_summary_statistics

#==========================================================================================
#? Data Structures for Comparative Statics
==========================================================================================#

"""
    ParameterGrid

Defines a grid of values for a specific parameter in comparative statics analysis.

# Fields
- `parameter_name::String`: Name of the parameter (e.g., "xi", "beta", "kappa0")
- `parameter_path::Vector{String}`: Path to parameter in config structure
- `values::Vector{Float64}`: Grid of values to test
- `description::String`: Description of the parameter
"""
@with_kw mutable struct ParameterGrid
    parameter_name::Union{String, Symbol}  # e.g., "ξ", :β, :c₀
    parameter_path::Vector{Union{String, Symbol}}  # e.g., ["ξ"], ["utility_function", :c₀]
    values::Vector{Float64}
    description::String = ""
end

"""
    ComparativeStaticsConfig

Configuration for comparative statics analysis.

# Fields
- `base_config_path::String`: Path to base YAML configuration file
- `parameter_grid::ParameterGrid`: Parameter to vary
- `output_dir::String`: Directory to save results
- `run_name::String`: Name for this comparative statics run
- `solver_options::Dict`: Custom solver options
- `collect_detailed_results::Bool`: Whether to save detailed equilibrium objects
"""
@with_kw struct ComparativeStaticsConfig
    base_config_path::String
    parameter_grid::ParameterGrid
    output_dir::String = "comparative_statics_results"
    run_name::String = "default_run"
    solver_options::Dict = Dict(:tol => 1e-7, :max_iter => 500, :verbose => false)
    collect_detailed_results::Bool = false
    save_plots::Bool = false
end

"""
    ComparativeStaticsResults

Container for comparative statics results.

# Fields
- `config::ComparativeStaticsConfig`: Configuration used
- `parameter_values::Vector{Float64}`: Parameter values tested
- `summary_stats::DataFrame`: Summary statistics for each parameter value
- `detailed_results::Vector{Tuple}`: Detailed results (if collected)
- `run_info::Dict`: Metadata about the run
"""
mutable struct ComparativeStaticsResults
    config::ComparativeStaticsConfig
    parameter_values::Vector{Float64}
    summary_stats::DataFrame
    detailed_results::Vector{Tuple{Primitives, Results}}
    run_info::Dict{String, Any}
end

#==========================================================================================
#? Parameter Grid Creation Functions
==========================================================================================#

"""
    create_parameter_grid(parameter_name::String, values::Vector{Float64}; kwargs...)

Creates a ParameterGrid for common model parameters.

# Arguments
- `parameter_name::String`: Name of parameter ("xi", "beta", "kappa0", etc.)
- `values::Vector{Float64}`: Values to test
- `description::String`: Optional description

# Returns
- `ParameterGrid`: Configured parameter grid
"""
function create_parameter_grid(
                                parameter_path::Vector{<:Union{String, Symbol}},
                                values::Vector{Float64}; 
                                description::String=""
                            )
    return ParameterGrid(
        parameter_name = parameter_path[end],
        parameter_path = parameter_path,
        values = values,
        description = description
    )
end

"""
    create_linspace_grid(parameter_name::String, min_val::Float64, max_val::Float64, n_points::Int)

Creates a linearly spaced parameter grid.
"""
function create_linspace_grid(
                            parameter_path::Vector{<:Union{String, Symbol}},
                            min_val::Float64,
                            max_val::Float64,
                            n_points::Int
                        )
    values = range(min_val, max_val, length=n_points) |> collect
    return create_parameter_grid(parameter_path, values)
end

"""
    create_logspace_grid(parameter_name::String, min_val::Float64, max_val::Float64, n_points::Int)

Creates a logarithmically spaced parameter grid.
"""
function create_logspace_grid(
                                parameter_path::Vector{<:Union{String, Symbol}},
                                min_val::Float64,
                                max_val::Float64,
                                n_points::Int)
    log_values = range(log(min_val), log(max_val), length=n_points)
    values = exp.(log_values)
    return create_parameter_grid(parameter_path, values)
end

"""
    create_custom_grid(parameter_name::String, values::Vector{Float64})

Creates a parameter grid with custom values.
"""
function create_custom_grid(parameter_name::String, values::Vector{Float64})
    return create_parameter_grid(parameter_name, values)
end

#==========================================================================================
#? Main Comparative Statics Functions
==========================================================================================#

"""
    run_comparative_statics(config::ComparativeStaticsConfig)

Runs comparative statics analysis by varying a single parameter across a grid.

# Arguments
- `config::ComparativeStaticsConfig`: Configuration for the analysis

# Returns
- `ComparativeStaticsResults`: Results object containing all outcomes
"""
function run_comparative_statics(config::ComparativeStaticsConfig)
    
    println("="^80)
    println("RUNNING COMPARATIVE STATICS ANALYSIS")
    println("="^80)
    println("Parameter: $(config.parameter_grid.parameter_name)")
    println("Description: $(config.parameter_grid.description)")
    println("Values: $(length(config.parameter_grid.values)) points")
    println("Range: [$(minimum(config.parameter_grid.values)), $(maximum(config.parameter_grid.values))]")
    println("="^80)
    
    # Initialize results containers
    parameter_values = config.parameter_grid.values
    n_points = length(parameter_values)
    detailed_results = Vector{Tuple{Primitives, Results}}()
    
    # Initialize summary statistics DataFrame
    summary_stats = DataFrame(
        parameter_value = Float64[],
        market_tightness = Float64[],
        job_finding_rate = Float64[],
        vacancy_filling_rate = Float64[],
        unemployment_rate = Float64[],
        avg_remote_share = Float64[],
        avg_wage = Float64[],
        total_surplus = Float64[],
        convergence_flag = Bool[],
        solve_time = Float64[]
    )
    
    # Setup output directory
    if !isdir(config.output_dir)
        mkpath(config.output_dir)
    end
    
    # Run analysis for each parameter value
    for (i, param_value) in enumerate(parameter_values)
        println("[$i/$n_points] Testing $(config.parameter_grid.parameter_name) = $param_value")
        
        start_time = time()
        
        try
            # Load and modify configuration
            prim, results = run_single_parameter_point(config, param_value)
            
            solve_time = time() - start_time
            
            # Extract summary statistics
            stats = extract_point_statistics(prim, results, param_value, solve_time)
            
            # Add to summary DataFrame with convergence flag
            stats_row = merge(stats, (convergence_flag = true,))
            push!(summary_stats, stats_row)
            
            # Store detailed results if requested
            if config.collect_detailed_results
                push!(detailed_results, (prim, results))
            end
            
            println("  ✓ Completed in $(round(stats.solve_time, digits=2))s")
            
        catch e
            @warn "Failed for $(config.parameter_grid.parameter_name) = $param_value: $e"
            
            # Add failed point with NaN values
            failed_stats = (
                parameter_value = param_value,
                market_tightness = NaN,
                job_finding_rate = NaN,
                vacancy_filling_rate = NaN,
                unemployment_rate = NaN,
                avg_remote_share = NaN,
                avg_wage = NaN,
                total_surplus = NaN,
                convergence_flag = false,
                solve_time = time() - start_time
            )
            push!(summary_stats, failed_stats)
        end
        
    end
    
    # Create results object
    run_info = Dict(
        "timestamp" => Dates.now(),
        "total_points" => n_points,
        "successful_points" => sum(summary_stats.convergence_flag),
        "failed_points" => sum(.!summary_stats.convergence_flag),
        "total_runtime" => sum(summary_stats.solve_time)
    )
    
    results_obj = ComparativeStaticsResults(
        config,
        parameter_values,
        summary_stats,
        detailed_results,
        run_info
    )
    
    # Save results
    save_results(results_obj)
    
    # Print summary
    print_analysis_summary(results_obj)
    
    return results_obj
end

"""
    run_single_parameter_point(config::ComparativeStaticsConfig, param_value::Float64)

Runs the model for a single parameter value.
"""
function run_single_parameter_point(config::ComparativeStaticsConfig, param_value::Float64)
    
    # Load base configuration
    prim, _ = Types.initializeModel(config.base_config_path)

    # Modify parameter value
    modify_parameter!(prim, config.parameter_grid.parameter_path, param_value)
    
    
    try
        # Run model with modified configuration
        prim, results = ModelRunner.run_model(
            prim; 
            verbose=false, 
            calibrate=true
        )
        
        # Solve with custom options
        ModelSolver.solve_model(prim, results; config.solver_options...)
        
        return prim, results

    catch e
        @warn "Failed for $(config.parameter_grid.parameter_name) = $param_value: $e"
        # If it fails, return empty results
        return prim, nothing
    end
end

#==========================================================================================
#? Results Processing and Analysis
==========================================================================================#

"""
    extract_point_statistics(prim::Primitives, results::Results, param_value::Float64, solve_time::Float64)

Extracts summary statistics from a single model solution.
"""
function extract_point_statistics(prim::Primitives, results::Results, param_value::Float64, solve_time::Float64)
    
    # Basic equilibrium variables
    market_tightness = results.θ
    job_finding_rate = results.p
    vacancy_filling_rate = results.q
    
    # Labor market composition
    total_unemployment = sum(results.u)
    unemployment_rate = total_unemployment / sum(prim.h_grid.pdf)
    
    # Work arrangements and wages (only for active matches)
    active_matches = results.S .> 0
    if sum(active_matches) > 0
        weights = results.n[active_matches] ./ sum(results.n[active_matches])
        avg_remote_share = sum(results.α_policy[active_matches] .* weights)
        avg_wage = sum(results.w_policy[active_matches] .* weights)
        total_surplus = sum(results.S[active_matches] .* weights)
    else
        avg_remote_share = NaN
        avg_wage = NaN
        total_surplus = NaN
    end
    
    return (
        parameter_value = param_value,
        market_tightness = market_tightness,
        job_finding_rate = job_finding_rate,
        vacancy_filling_rate = vacancy_filling_rate,
        unemployment_rate = unemployment_rate,
        avg_remote_share = avg_remote_share,
        avg_wage = avg_wage,
        total_surplus = total_surplus,
        solve_time = solve_time
    )
end

"""
    extract_summary_statistics(results::ComparativeStaticsResults)

Extracts additional summary statistics from comparative statics results.
"""
function extract_summary_statistics(results::ComparativeStaticsResults)
    df = results.summary_stats
    successful_df = df[df.convergence_flag, :]
    
    if nrow(successful_df) == 0
        @warn "No successful runs to analyze"
        return Dict()
    end
    
    param_name = results.config.parameter_grid.parameter_name
    
    summary = Dict(
        "parameter_name" => param_name,
        "n_successful" => nrow(successful_df),
        "n_total" => nrow(df),
        "success_rate" => nrow(successful_df) / nrow(df),
        
        # Parameter range
        "param_min" => minimum(successful_df.parameter_value),
        "param_max" => maximum(successful_df.parameter_value),
        "param_mean" => mean(successful_df.parameter_value),
        
        # Outcome ranges
        "unemployment_range" => [minimum(successful_df.unemployment_rate), maximum(successful_df.unemployment_rate)],
        "remote_share_range" => [minimum(skipmissing(successful_df.avg_remote_share)), maximum(skipmissing(successful_df.avg_remote_share))],
        "wage_range" => [minimum(skipmissing(successful_df.avg_wage)), maximum(skipmissing(successful_df.avg_wage))],
        
        # Correlations (if enough variation)
        "unemployment_elasticity" => nrow(successful_df) > 2 ? cor(successful_df.parameter_value, successful_df.unemployment_rate) : NaN,
        "remote_elasticity" => nrow(successful_df) > 2 ? cor(successful_df.parameter_value, successful_df.avg_remote_share) : NaN
    )
    
    return summary
end

#==========================================================================================
#? Results Saving and Loading
==========================================================================================#

"""
    save_results(results::ComparativeStaticsResults)

Saves comparative statics results to disk.
"""
function save_results(results::ComparativeStaticsResults)
    
    # Create timestamped directory
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
    run_dir = joinpath(results.config.output_dir, "$(results.config.run_name)_$timestamp")
    mkpath(run_dir)
    
    # Save summary statistics as CSV
    CSV.write(joinpath(run_dir, "summary_statistics.csv"), results.summary_stats)
    
    # Save configuration
    config_info = Dict(
        "parameter_name" => results.config.parameter_grid.parameter_name,
        "parameter_description" => results.config.parameter_grid.description,
        "parameter_values" => results.config.parameter_grid.values,
        "base_config_path" => results.config.base_config_path,
        "solver_options" => results.config.solver_options,
        "run_info" => results.run_info
    )
    YAML.write_file(joinpath(run_dir, "run_config.yaml"), config_info)
    
    # Save summary statistics
    summary_stats = extract_summary_statistics(results)
    YAML.write_file(joinpath(run_dir, "summary_analysis.yaml"), summary_stats)
    
    println("Results saved to: $run_dir")
    
    return run_dir
end

"""
    load_results(results_dir::String)

Loads comparative statics results from disk.
"""
function load_results(results_dir::String)
    
    # Load summary statistics
    summary_stats = CSV.read(joinpath(results_dir, "summary_statistics.csv"), DataFrame)
    
    # Load configuration
    config_info = YAML.load_file(joinpath(results_dir, "run_config.yaml"))
    
    # Reconstruct parameter grid
    param_grid = ParameterGrid(
        parameter_name = config_info["parameter_name"],
        parameter_path = String[],  # Not needed for loaded results
        values = config_info["parameter_values"],
        description = config_info["parameter_description"]
    )
    
    # Reconstruct config
    config = ComparativeStaticsConfig(
        base_config_path = config_info["base_config_path"],
        parameter_grid = param_grid,
        output_dir = dirname(results_dir),
        run_name = basename(results_dir),
        solver_options = config_info["solver_options"]
    )
    
    # Create results object
    results = ComparativeStaticsResults(
        config,
        config_info["parameter_values"],
        summary_stats,
        Tuple{Primitives, Results}[],  # Not saved/loaded
        config_info["run_info"]
    )
    
    return results
end

#==========================================================================================
#? Utility and Display Functions
==========================================================================================#

"""
    print_analysis_summary(results::ComparativeStaticsResults)

Prints a summary of the comparative statics analysis.
"""
function print_analysis_summary(results::ComparativeStaticsResults)
    println("\n" * "="^80)
    println("COMPARATIVE STATICS ANALYSIS SUMMARY")
    println("="^80)
    
    println("Parameter: $(results.config.parameter_grid.parameter_name)")
    println("Description: $(results.config.parameter_grid.description)")
    println("Total Points: $(results.run_info["total_points"])")
    println("Successful: $(results.run_info["successful_points"])")
    println("Failed: $(results.run_info["failed_points"])")
    println("Success Rate: $(round(100 * results.run_info["successful_points"] / results.run_info["total_points"], digits=1))%")
    println("Total Runtime: $(round(results.run_info["total_runtime"], digits=1)) seconds")
    
    # Show outcome ranges for successful runs
    successful_df = results.summary_stats[results.summary_stats.convergence_flag, :]
    if nrow(successful_df) > 0
        println("\nOUTCOME RANGES:")
        println("  Unemployment Rate: $(round(minimum(successful_df.unemployment_rate)*100, digits=2))% - $(round(maximum(successful_df.unemployment_rate)*100, digits=2))%")
        
        if !all(isnan.(successful_df.avg_remote_share))
            println("  Avg Remote Share: $(round(minimum(skipmissing(successful_df.avg_remote_share)), digits=3)) - $(round(maximum(skipmissing(successful_df.avg_remote_share)), digits=3))")
        end
        
        if !all(isnan.(successful_df.avg_wage))
            println("  Avg Wage: $(round(minimum(skipmissing(successful_df.avg_wage)), digits=3)) - $(round(maximum(skipmissing(successful_df.avg_wage)), digits=3))")
        end
    end
    
    println("="^80)
end

end # module ComparativeStatics
