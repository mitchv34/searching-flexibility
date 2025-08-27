#!/usr/bin/env julia

"""
run_search_production.jl

Production-ready driver script for ensemble local optimization using top GA candidates.
This script implements a robust second-stage GMM estimation with comprehensive logging,
error handling, and result persistence.

Features:
- Configurable optimization parameters via YAML
- Comprehensive logging and progress tracking  
- Robust error handling and recovery
- Results persistence in multiple formats
- Performance monitoring and profiling
- Parallel execution with thread safety
- Command-line interface support

Author: SearchingFlexibility Project
Date: 2025-08-27
"""

# =============================================================================
# 1. IMPORTS AND DEPENDENCIES
# =============================================================================

include("objective.jl")
include("utils.jl")

using YAML, CSV, DataFrames, Dates
using .LocalObjectiveUtils
using Optimization, OptimizationOptimJL, SciMLBase
using Arrow, LinearAlgebra, PrettyTables, Statistics
using Logging, LoggingExtras
using ArgParse, JSON3

# =============================================================================
# 2. CONSTANTS AND CONFIGURATION
# =============================================================================

const DEFAULT_CONFIG_PATH = joinpath(@__DIR__, "local_refine_config.yaml")
const LOG_DIR = joinpath(@__DIR__, "..", "..", "..", "logs")
const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "..", "output")

# Ensure required directories exist
mkpath(LOG_DIR)
mkpath(OUTPUT_DIR)

# =============================================================================
# 3. LOGGING SETUP
# =============================================================================

"""
    setup_logging(log_level::String="INFO", log_file::Union{String,Nothing}=nothing)

Configure comprehensive logging for the optimization process.
"""
function setup_logging(log_level::String="INFO", log_file::Union{String,Nothing}=nothing)
    # Parse log level
    level = if log_level == "DEBUG"
        Logging.Debug
    elseif log_level == "INFO"
        Logging.Info
    elseif log_level == "WARN"
        Logging.Warn
    elseif log_level == "ERROR"
        Logging.Error
    else
        @warn "Unknown log level: $log_level, using INFO"
        Logging.Info
    end
    
    # Setup loggers
    loggers = []
    
    # Console logger with formatting
    console_logger = ConsoleLogger(stderr, level)
    push!(loggers, console_logger)
    
    # File logger if specified
    if log_file !== nothing
        # Open file for logging
        file_io = open(log_file, "a")
        file_logger = SimpleLogger(file_io, level)
        push!(loggers, file_logger)
    end
    
    # Combine loggers
    global_logger(TeeLogger(loggers...))
    
    @info "Logging configured" level=log_level file=log_file
end

# =============================================================================
# 4. CONFIGURATION MANAGEMENT
# =============================================================================

"""
    load_and_validate_config(config_path::String) -> Dict

Load and validate the optimization configuration file.
"""
function load_and_validate_config(config_path::String)
    !isfile(config_path) && throw(ArgumentError("Config file not found: $config_path"))
    
    try
        cfg = YAML.load_file(config_path)
        
        # Validate required sections
        required_sections = ["GlobalSearchResults", "ModelInputs", "OptimizationSettings"]
        for section in required_sections
            haskey(cfg, section) || throw(ArgumentError("Missing required config section: $section"))
        end
        
        @info "Configuration loaded and validated" config_path=config_path
        return cfg
        
    catch e
        @error "Failed to load configuration" config_path=config_path exception=(e, catch_backtrace())
        rethrow(e)
    end
end

# =============================================================================
# 5. CANDIDATE SELECTION (Enhanced with Error Handling)
# =============================================================================

"""
    load_diverse_top_candidates(; config_path::String, n::Union{Int,Nothing}=nothing, 
                                quality_quantile::Float64=0.1) -> NamedTuple

Enhanced version with comprehensive error handling and validation.
"""
function load_diverse_top_candidates(;
                                    config_path::String,
                                    n::Union{Int,Nothing}=nothing,
                                    quality_quantile::Float64=0.1
                                )
    @info "Loading diverse top candidates" config_path=config_path n=n quality_quantile=quality_quantile
    
    try
        cfg = load_and_validate_config(config_path)
        gsr = cfg["GlobalSearchResults"]
        
        # Get job ID with validation
        job_id = get(gsr, "job_id", get(ENV, "LOCAL_GA_JOB_ID", nothing))
        if job_id === nothing
            throw(ArgumentError("No job ID available in config or environment variables"))
        end
        
        # Set default n if not provided
        n === nothing && (n = get(gsr, "n_top_starts", 5))
        
        # Validate inputs
        n > 0 || throw(ArgumentError("n must be positive, got: $n"))
        0.0 < quality_quantile <= 1.0 || throw(ArgumentError("quality_quantile must be in (0,1], got: $quality_quantile"))
        
        # Load GA results
        csv_path = replace(get(gsr, "latest_results_file", ""), "{job_id}" => string(job_id))
        !isfile(csv_path) && throw(ArgumentError("GA results CSV not found: $csv_path"))
        
        @info "Loading GA results" csv_path=csv_path
        df = CSV.read(csv_path, DataFrame)
        
        if isempty(df)
            throw(ArgumentError("No rows in GA results CSV: $csv_path"))
        end
        
        sort!(df, :objective)
        @info "GA results loaded" total_candidates=nrow(df) best_objective=df.objective[1]
        
        # Filter for quality
        n_quality = max(1, ceil(Int, quality_quantile * nrow(df)))
        n_quality = min(n_quality, nrow(df))  # Ensure we don't exceed available data
        quality_pool_df = first(df, n_quality)
        
        @info "Quality pool selected" pool_size=n_quality from_total=nrow(df)
        
        # Extract parameter information
        param_names_str = names(quality_pool_df)[2:end]  # Skip objective column
        param_names_sym = Symbol.(param_names_str)
        
        # Load bounds for normalization
        mpi_config_path = get(gsr, "config_path", nothing)
        if mpi_config_path === nothing
            throw(ArgumentError("MPI config path not found in GlobalSearchResults"))
        end
        
        mpi_config = YAML.load_file(mpi_config_path)
        bounds_dict = get(get(mpi_config, "MPISearchConfig", Dict()), "parameters", Dict())
        bounds_dict = get(bounds_dict, "bounds", Dict())
        
        if isempty(bounds_dict)
            throw(ArgumentError("Parameter bounds not found in MPI config"))
        end
        
        # Extract bounds
        lower_bounds = [get(bounds_dict, k, [NaN, NaN])[1] for k in param_names_str]
        upper_bounds = [get(bounds_dict, k, [NaN, NaN])[2] for k in param_names_str]
        
        if any(isnan, lower_bounds) || any(isnan, upper_bounds)
            throw(ArgumentError("Some parameter bounds are missing or invalid"))
        end
        
        ranges = upper_bounds .- lower_bounds
        
        # Normalize parameters for diversity selection
        quality_params_matrix = Matrix(quality_pool_df[!, param_names_str])
        normalized_params = (quality_params_matrix .- lower_bounds') ./ ranges'
        
        # Select diverse candidates using Farthest Point Sampling
        selected_indices = select_diverse_points(normalized_params, min(n, size(normalized_params, 1)))
        
        # Extract final results
        final_params_df = quality_pool_df[selected_indices, :]
        params = [Vector{Float64}(row) for row in eachrow(final_params_df[!, param_names_str])]
        
        @info "Diverse candidates selected" 
            n_selected=length(params) 
            objective_range=(minimum(final_params_df.objective), maximum(final_params_df.objective))
        
        return (params=params, param_names=param_names_sym, objectives=final_params_df.objective)
        
    catch e
        @error "Failed to load diverse top candidates" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

"""
    select_diverse_points(normalized_params::Matrix, n::Int) -> Vector{Int}

Select n diverse points using Farthest Point Sampling algorithm.
"""
function select_diverse_points(normalized_params::Matrix, n::Int)
    n_available = size(normalized_params, 1)
    n = min(n, n_available)
    
    selected_indices = Int[]
    
    # Start with the first point (best objective)
    push!(selected_indices, 1)
    
    if n == 1
        return selected_indices
    end
    
    # Available candidates (excluding the first)
    candidate_indices = Set(2:n_available)
    
    while length(selected_indices) < n
        max_min_dist = -1.0
        best_next_idx = -1
        
        selected_points = normalized_params[selected_indices, :]
        
        for idx in candidate_indices
            candidate_point = normalized_params[idx, :]
            # Calculate minimum distance to all selected points
            min_dist = minimum([norm(candidate_point - selected_points[i, :]) 
                               for i in 1:size(selected_points, 1)])
            
            if min_dist > max_min_dist
                max_min_dist = min_dist
                best_next_idx = idx
            end
        end
        
        push!(selected_indices, best_next_idx)
        delete!(candidate_indices, best_next_idx)
    end
    
    return selected_indices
end

# =============================================================================
# 6. PROBLEM SETUP
# =============================================================================

"""
    build_ensemble_problem(config::Dict, candidates::NamedTuple) -> NamedTuple

Build the optimization ensemble problem with enhanced error handling.
"""
function build_ensemble_problem(config::Dict, candidates::NamedTuple)
    @info "Building ensemble optimization problem" n_candidates=length(candidates.params)
    
    try
        # Extract configuration
        gsr = config["GlobalSearchResults"]
        start_points = candidates.params
        full_param_names = candidates.param_names
        
        # Handle parameter subset if specified
        subset = get(gsr, "parameter_subset", Any[])
        if subset isa Vector && !isempty(subset)
            param_names = Symbol.(subset)
            # Map subset indices
            name_to_index = Dict(full_param_names[i] => i for i in eachindex(full_param_names))
            missing_params = setdiff(param_names, keys(name_to_index))
            if !isempty(missing_params)
                throw(ArgumentError("Parameters not found in GA results: $missing_params"))
            end
            start_points = [[p[name_to_index[nm]] for nm in param_names] for p in start_points]
        else
            param_names = full_param_names
        end
        
        # Build problem context
        @info "Building problem context"
        context = build_context_from_config(config)
        
        # Create optimization function with error handling
        function safe_objective(u, p)
            try
                return evaluate_for_optimizer(u, p.context, p.param_names; 
                                            verbose=false, 
                                            solve_kwargs=Dict(:verbose => false))
            catch e
                @error "Objective evaluation failed" parameters=u exception=(e, catch_backtrace())
                # Return large value for failed evaluations with the same type as input
                T = eltype(u)
                return T(1e12)  # Use a large penalty value with the correct type
            end
        end
        
        # Build optimization problem
        f = OptimizationFunction(safe_objective, Optimization.AutoForwardDiff())
        dummy_u0 = start_points[1]
        p = (context=context, param_names=param_names)
        base_prob = OptimizationProblem(f, dummy_u0, p)
        
        # Problem function for ensemble
        prob_func = (prob, i, repeat) -> remake(prob; u0=start_points[i])
        ensemble_prob = EnsembleProblem(base_prob; prob_func=prob_func)
        
        @info "Ensemble problem built successfully" n_trajectories=length(start_points)
        
        return (
            ensemble_prob=ensemble_prob,
            base_problem=base_prob,
            start_points=start_points,
            param_names=param_names,
            context=context
        )
        
    catch e
        @error "Failed to build ensemble problem" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

"""
    build_context_from_config(cfg::Dict)

Enhanced context builder with validation.
"""
function build_context_from_config(cfg::Dict)
    try
        inputs = cfg["ModelInputs"]
        
        # Validate required inputs
        required_keys = ["config_path", "data_moments_yaml", "weighting_matrix_csv", "sim_data_path"]
        for key in required_keys
            if !haskey(inputs, key)
                throw(ArgumentError("Missing required ModelInputs key: $key"))
            end
        end
        
        moment_filter = get(inputs, "moment_filter", nothing)
        moment_filter_syms = (moment_filter isa Vector && !isempty(moment_filter)) ? 
                            Symbol.(moment_filter) : nothing
        
        @info "Building problem context" moment_filter=moment_filter_syms
        
        return setup_problem_context(; 
            config_path=inputs["config_path"],
            data_moments_yaml=inputs["data_moments_yaml"],
            weighting_matrix_csv=inputs["weighting_matrix_csv"],
            sim_data_path=inputs["sim_data_path"],
            moment_key_filter=moment_filter_syms
        )
        
    catch e
        @error "Failed to build context from config" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

# =============================================================================
# 7. OPTIMIZATION EXECUTION
# =============================================================================

"""
    run_optimization(ensemble_prob, config::Dict, n_trajectories::Int) -> Vector

Execute the ensemble optimization with comprehensive monitoring.
"""
function run_optimization(ensemble_prob, config::Dict, n_trajectories::Int)
    opt_settings = get(config, "OptimizationSettings", Dict())
    
    # Parse optimization parameters with defaults
    optimizer_name = get(opt_settings, "optimizer", "LBFGS")
    max_iters = get(opt_settings, "max_iterations", 5000)
    g_tol = get(opt_settings, "gradient_tolerance", 1e-7)
    f_abstol = get(opt_settings, "function_abstol", 1e-10)
    f_reltol = get(opt_settings, "function_reltol", 1e-10) 
    x_abstol = get(opt_settings, "parameter_abstol", 1e-10)
    x_reltol = get(opt_settings, "parameter_reltol", 1e-10)
    
    @info "Starting optimization"  optimizer_name  max_iters  g_tol n_trajectories
    
    # Select optimizer with proper options
    optimizer = if optimizer_name == "LBFGS"
        LBFGS()
    elseif optimizer_name == "NelderMead"
        NelderMead()
    elseif optimizer_name == "BFGS"
        BFGS()
    else
        @warn "Unknown optimizer: $optimizer_name, using LBFGS"
        LBFGS()
    end
    
    start_time = time()
    
    try
        solution = solve(
            ensemble_prob,
            optimizer,
            EnsembleThreads();
            trajectories=n_trajectories,
            maxiters=max_iters,
            abstol=f_abstol,
            reltol=f_reltol,
            g_tol=g_tol,
            allow_f_increases=true
        )
        
        elapsed_time = time() - start_time
        
        # Detailed logging of optimization results
        return_codes = [s.retcode for s in solution]
        n_successful = sum(s.retcode == ReturnCode.Success ? 1 : 0 for s in solution)

        @info "Optimization completed"  round(elapsed_time, digits=2) n_successful length(solution) return_codes
        
        # Log details for each trajectory
        for (i, s) in enumerate(solution)
            @info "Trajectory $i results"  s.retcode s.objective s.original.iterations s.original.f_calls s.original.g_calls
                # converged=s.original.converged
        end
        
        return solution
        
    catch e
        @error "Optimization failed" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

# =============================================================================
# 8. RESULTS ANALYSIS AND PERSISTENCE
# =============================================================================

"""
    analyze_and_save_results(solution, initial_candidates, problem_setup, config::Dict)

Comprehensive analysis and persistence of optimization results.
"""
function analyze_and_save_results(solution, initial_candidates, problem_setup, config::Dict)
    @info "Analyzing optimization results"
    
    try
        # Extract results
        final_objs = [s.objective for s in solution]
        final_params = [s.u for s in solution]
        param_names = problem_setup.param_names
        
        # Re-evaluate initial objectives using the same weighted objective function
        # This ensures fair comparison with the final objectives
        initial_objs = Float64[]
        @info "Re-evaluating initial objectives for fair comparison"
        for (i, start_point) in enumerate(initial_candidates.params)
            try
                obj_val = evaluate_for_optimizer(start_point, problem_setup.context, param_names; 
                                               verbose=false, 
                                               solve_kwargs=Dict(:verbose => false))
                push!(initial_objs, obj_val)
                @debug "Re-evaluated initial objective" point=i objective=obj_val
            catch e
                @warn "Failed to re-evaluate initial objective for point $i, using large penalty" exception=e
                push!(initial_objs, 1e12)
            end
        end
        
        # Find best solution
        best_idx = argmin(final_objs)
        improvements = initial_objs .- final_objs
        
        @info "Best solution found" 
            best_index=best_idx
            initial_obj=round(initial_objs[best_idx], digits=6)
            final_obj=round(final_objs[best_idx], digits=6)
            improvement=round(improvements[best_idx], digits=6)
        
        # Helper function to safely extract iterations
        function get_iterations(sol)
            try
                # Try to access iterations from the original result
                if hasfield(typeof(sol), :original) && hasfield(typeof(sol.original), :iterations)
                    return sol.original.iterations
                elseif hasfield(typeof(sol), :stats) && hasfield(typeof(sol.stats), :nf)
                    return sol.stats.nf  # Number of function evaluations
                else
                    return 0
                end
            catch
                return 0
            end
        end
        
        # Create results DataFrame
        results_df = DataFrame(
            start_point=1:length(solution),
            initial_objective=initial_objs,
            final_objective=final_objs,
            improvement=improvements,
            success=[s.retcode == :Success for s in solution],
            iterations=[get_iterations(s) for s in solution],
            final_parameters=[final_params[i] for i in 1:length(final_params)]
        )
        
        # Add parameter columns
        for (i, pname) in enumerate(param_names)
            results_df[!, pname] = [p[i] for p in final_params]
        end
        
        # Generate timestamp for files
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        
        # Save detailed results
        results_file = joinpath(OUTPUT_DIR, "optimization_results_$timestamp.csv")
        CSV.write(results_file, results_df)
        @info "Results saved" file=results_file
        
        # Save best parameters separately
        best_params_df = DataFrame(
            parameter=param_names,
            estimate=solution[best_idx].u
        )
        
        best_params_file = joinpath(OUTPUT_DIR, "best_parameters_$timestamp.csv")
        CSV.write(best_params_file, best_params_df)
        @info "Best parameters saved" file=best_params_file
        
        # Save summary statistics
        summary = Dict(
            "timestamp" => timestamp,
            "n_trajectories" => length(solution),
            "n_successful" => sum(results_df.success),
            "best_objective" => minimum(final_objs),
            "worst_objective" => maximum(final_objs),
            "mean_improvement" => mean(improvements),
            "total_improvement" => sum(improvements),
            "std_final_objectives" => std(final_objs),
            "convergence_rate" => mean(results_df.success),
            "best_parameters" => Dict(zip(string.(param_names), solution[best_idx].u))
        )
        
        summary_file = joinpath(OUTPUT_DIR, "optimization_summary_$timestamp.json")
        open(summary_file, "w") do io
            JSON3.pretty(io, summary)
        end
        @info "Summary saved" file=summary_file
        
        # Print results table
        print_results_table(results_df, param_names)
        print_summary_statistics(summary)
        
        return results_df, summary
        
    catch e
        @error "Failed to analyze and save results" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

"""
    print_results_table(results_df::DataFrame, param_names::Vector{Symbol})

Print a formatted results table.
"""
function print_results_table(results_df::DataFrame, param_names::Vector{Symbol})
    println("\n" * "="^120)
    println("DETAILED OPTIMIZATION RESULTS")
    println("="^120)
    
    # Prepare display data
    display_data = hcat(
        results_df.start_point,
        round.(results_df.initial_objective, digits=6),
        round.(results_df.final_objective, digits=6),
        round.(results_df.improvement, digits=6),
        [s ? "✓" : "✗" for s in results_df.success],
        results_df.iterations
    )
    
    headers = ["Start", "Initial Obj", "Final Obj", "Improvement", "Success", "Iters"]
    
    pretty_table(
        display_data;
        header=headers,
        header_crayon=crayon"yellow bold",
        crop=:none,
        alignment=[:c, :r, :r, :r, :c, :c]
    )
end

"""
    print_summary_statistics(summary::Dict)

Print summary statistics.
"""
function print_summary_statistics(summary::Dict)
    println("\n📊 SUMMARY STATISTICS:")
    println("   Total trajectories: $(summary["n_trajectories"])")
    println("   Successful convergences: $(summary["n_successful"]) ($(round(summary["convergence_rate"]*100, digits=1))%)")
    println("   Best objective: $(round(summary["best_objective"], digits=8))")
    println("   Worst objective: $(round(summary["worst_objective"], digits=8))")
    println("   Mean improvement: $(round(summary["mean_improvement"], digits=8))")
    println("   Total improvement: $(round(summary["total_improvement"], digits=8))")
    println("   Std of final objectives: $(round(summary["std_final_objectives"], digits=8))")
    
    if summary["std_final_objectives"] < 1e-6
        println("   🎯 Excellent: All runs converged to nearly the same minimum!")
    end
end

# =============================================================================
# 9. COMMAND LINE INTERFACE
# =============================================================================

"""
    parse_command_line() -> Dict

Parse command line arguments.
"""
function parse_command_line()
    s = ArgParseSettings(
        description="Production ensemble local optimization for GMM estimation",
        version="1.0.0",
        add_version=true
    )
    
    @add_arg_table! s begin
        "--config", "-c"
            help = "Path to configuration file"
            arg_type = String
            default = DEFAULT_CONFIG_PATH
        "--candidates", "-n"
            help = "Number of diverse candidates to use"
            arg_type = Int
            default = 10
        "--log-level", "-l"
            help = "Logging level"
            arg_type = String
            default = "INFO"
            range_tester = x -> x in ["DEBUG", "INFO", "WARN", "ERROR"]
        "--log-file"
            help = "Optional log file path"
            arg_type = String
        "--dry-run"
            help = "Run validation checks without optimization"
            action = :store_true
    end
    
    return parse_args(ARGS, s)
end

# =============================================================================
# 10. MAIN FUNCTION
# =============================================================================

"""
    main(args::Dict=Dict())

Main execution function with comprehensive error handling and logging.
"""
function main(args::Dict=Dict())
    # Parse command line if args not provided
    if isempty(args)
        args = parse_command_line()
    end
    
    # Setup logging
    log_file = get(args, "log-file", nothing)
    if log_file === nothing
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        log_file = joinpath(LOG_DIR, "optimization_$timestamp.log")
    end
    
    setup_logging(args["log-level"], log_file)
    
    @info "Starting production ensemble optimization" args=args
    
    try
        # Load and validate configuration
        config = load_and_validate_config(args["config"])
        
        if get(args, "dry-run", false)
            @info "Dry run mode - validation only"
            @info "Configuration validation successful"
            return nothing
        end
        
        # Load diverse candidates
        @info "Loading diverse candidates"
        candidates = load_diverse_top_candidates(
            config_path=args["config"],
            n=args["candidates"]
        )
        
        # Build ensemble problem
        @info "Building optimization problem"
        problem_setup = build_ensemble_problem(config, candidates)
        
        # Run optimization
        @info "Starting optimization"
        solution = run_optimization(problem_setup.ensemble_prob, config, length(problem_setup.start_points))
        
        # Analyze and save results
        @info "Analyzing and saving results"
        results_df, summary = analyze_and_save_results(solution, candidates, problem_setup, config)
        
        @info "Optimization completed successfully" 
            best_objective=summary["best_objective"]
            convergence_rate=summary["convergence_rate"]
        
        println("\n✅ Production optimization completed successfully!")
        println("📁 Results saved to: $(OUTPUT_DIR)")
        println("📋 Log file: $log_file")
        
        return results_df, summary
        
    catch e
        @error "Optimization failed" exception=(e, catch_backtrace())
        println("\n❌ Optimization failed. Check log file for details: $log_file")
        rethrow(e)
    end
end

# =============================================================================
# 11. SCRIPT EXECUTION
# =============================================================================

# Only run main if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    try
        main()
    catch e
        println("Fatal error: $e")
        exit(1)
    end
end
