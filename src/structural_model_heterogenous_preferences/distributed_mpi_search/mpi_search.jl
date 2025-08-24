# Distributed MPI Parameter Search for Structural Model
# This script uses ClusterManagers.jl and MPI to distribute parameter search
# across multiple nodes and cores for maximum scalability

# --- 1. Preamble and Setup ---
using Pkg
Pkg.activate("/project/high_tech_ind/searching-flexibility")

# Load required packages
using Distributed
using SlurmClusterManager
using YAML, Printf, Random
using Statistics, LinearAlgebra
using Dates
using JSON3
using QuasiMonteCarlo

# Get job ID from SLURM environment
JOB_ID = get(ENV, "SLURM_JOB_ID", "unknown")

println("🌐 Starting Distributed MPI Parameter Search")
println("=" ^ 50)
println("Julia version: $(VERSION)")
println("Job ID: $JOB_ID")
println("Number of available cores from SLURM: ", get(ENV, "SLURM_NTASKS", "Unknown"))
println("Start time: $(Dates.now())")

# --- 2. MPI Cluster Setup ---
# Add all workers allocated by SLURM
if haskey(ENV, "SLURM_NTASKS")
    n_workers = parse(Int, ENV["SLURM_NTASKS"]) - 1  # Subtract 1 for main process
    println("Adding $n_workers SLURM workers...")
    
    try
        # Use SlurmClusterManager to spawn workers on SLURM allocation
        addprocs(SlurmManager())
        println("✓ Successfully added $(nworkers()) workers")
        println("Worker IDs: $(workers())")
    catch e
        println("❌ Failed to add SLURM workers: $e")
        println("Falling back to single-threaded execution...")
    end
else
    println("⚠️  SLURM_NTASKS not found, running on single process")
end

# --- 3. Load model on all workers ---
@everywhere begin
    using Pkg
    Pkg.activate("/project/high_tech_ind/searching-flexibility")
    
    # Set paths
    const ROOT = "/project/high_tech_ind/searching-flexibility"
    const MODEL_DIR = joinpath(ROOT, "src", "structural_model_heterogenous_preferences")
    const MPI_SEARCH_DIR = joinpath(MODEL_DIR, "distributed_mpi_search")
    
    # Load required packages on workers
    using YAML, Printf, Random
    using Statistics, LinearAlgebra
    using Dates
    
    # Include model files
    include(joinpath(MODEL_DIR, "ModelSetup.jl"))
    include(joinpath(MODEL_DIR, "ModelSolver.jl"))  
    include(joinpath(MODEL_DIR, "ModelEstimation.jl"))
    
    # This will be set by the main process - we'll use @everywhere to distribute the config
    global TARGET_MOMENTS = Dict()
    global MOMENT_WEIGHTS = Dict()
end

# Helper function to sanitize NaN values for JSON serialization
function sanitize_nan_values(data)
    """
    Recursively replace NaN and Inf values with JSON-safe alternatives
    """
    if isa(data, Dict)
        return Dict(k => sanitize_nan_values(v) for (k, v) in data)
    elseif isa(data, Array)
        return [sanitize_nan_values(x) for x in data]
    elseif isa(data, Number)
        if isnan(data)
            return nothing  # JSON null
        elseif isinf(data)
            return data > 0 ? 1e10 : -1e10  # Large finite numbers
        else
            return data
        end
    else
        return data
    end
end

# Helper function to clean up old results and logs
function cleanup_old_results()
    """
    Clean up old results and logs from previous jobs before starting a new run
    """
    println("🧹 Cleaning up old results and logs...")
    
    # Clean results directory - remove files from other jobs
    results_dir = joinpath(MPI_SEARCH_DIR, "output", "results")
    if isdir(results_dir)
        old_files = readdir(results_dir)
        for file in old_files
            # Only remove JSON files that are NOT from the current job
            if endswith(file, ".json") && !contains(file, "job$(JOB_ID)")
                rm(joinpath(results_dir, file))
                println("  Removed old result from different job: $file")
            end
        end
    end
    
    # Clean logs directory - remove SLURM output files from other jobs
    logs_dir = joinpath(MPI_SEARCH_DIR, "output", "logs")
    if isdir(logs_dir)
        old_files = readdir(logs_dir)
        for file in old_files
            # Keep submit scripts and monitoring logs, remove SLURM output files from other jobs only
            if (endswith(file, ".out") || endswith(file, ".err")) && !contains(file, JOB_ID)
                try
                    rm(joinpath(logs_dir, file))
                    println("  Removed old log from different job: $file")
                catch
                    # Ignore if file is locked or doesn't exist
                end
            end
        end
    end
    
    println("✅ Cleanup completed")
end

# Helper function to clean up intermediate snapshots
function cleanup_intermediate_snapshots()
    """
    Remove all intermediate snapshots except the latest one after completion
    """
    println("🧹 Cleaning up intermediate snapshots...")
    
    results_dir = joinpath(MPI_SEARCH_DIR, "output", "results")
    if isdir(results_dir)
        all_files = readdir(results_dir)
        # Find intermediate files for this job (with timestamps, not final or latest)
        intermediate_files = filter(f -> 
            contains(f, "job$(JOB_ID)") && 
            endswith(f, ".json") && 
            !endswith(f, "_final.json") && 
            !endswith(f, "_latest.json") &&
            occursin(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.json$", f), all_files)
        
        # Sort by timestamp (extract timestamp from filename for proper chronological ordering)
        if length(intermediate_files) > 1
            # Extract timestamps and sort chronologically
            file_timestamps = []
            for file in intermediate_files
                # Extract timestamp from filename like: mpi_search_results_jobXXX_2025-08-23_10-15-00.json
                timestamp_match = match(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", file)
                if timestamp_match !== nothing
                    timestamp_str = timestamp_match.captures[1]
                    # Convert to DateTime for proper sorting
                    timestamp = DateTime(timestamp_str, "yyyy-mm-dd_HH-MM-SS")
                    push!(file_timestamps, (file, timestamp))
                end
            end
            
            # Sort by timestamp (most recent last)
            sort!(file_timestamps, by=x -> x[2])
            
            # Remove all but the most recent
            files_to_remove = [ft[1] for ft in file_timestamps[1:end-1]]
            
            for file in files_to_remove
                try
                    rm(joinpath(results_dir, file))
                    println("  Removed intermediate snapshot: $file")
                catch
                    # Ignore if file is locked or doesn't exist
                end
            end
            
            if !isempty(files_to_remove)
                latest_kept = file_timestamps[end][1]
                println("✅ Kept only the latest intermediate snapshot: $latest_kept")
            else
                println("✅ No intermediate snapshots to clean")
            end
        else
            println("✅ No intermediate snapshots to clean (≤1 file found)")
        end
    end
end

println("✓ Model loaded on all workers")

# --- 4. Load Configuration ---
# Accept config file from command line argument (passed by SLURM script)
if length(ARGS) > 0
    config_file = ARGS[1]
    if !isabspath(config_file)
        config_file = joinpath(ROOT, config_file)
    end
else
    # Fallback to default MPI search config if no argument provided
    config_file = joinpath(MPI_SEARCH_DIR, "mpi_search_config.yaml")
end

println("Using configuration file: $config_file")

# Load search configuration
if !isfile(config_file)
    println("Configuration file not found: $config_file")
    exit(1)
else
    config = YAML.load_file(config_file)
end

# Load target moments based on config specification
# Handle different config structures
if haskey(config, "target_moments")
    # Old structure: direct target_moments key
    target_moments_config = config["target_moments"]
elseif haskey(config, "MPISearchConfig") && haskey(config["MPISearchConfig"], "target_moments")
    # New structure: nested under MPISearchConfig
    target_moments_config = config["MPISearchConfig"]["target_moments"]
else
    println("❌ Could not find target_moments in config. Expected either 'target_moments' or 'MPISearchConfig.target_moments'")
    exit(1)
end

data_file_path = joinpath(ROOT, target_moments_config["data_file"])

if !isfile(data_file_path)
    println("Target moments data file not found: $data_file_path")
    exit(1)
else
    target_moments_data = YAML.load_file(data_file_path)
    
    # Extract the moments we need based on configuration
    moments_to_use = target_moments_config["moments_to_use"]
    global TARGET_MOMENTS = Dict()
    for moment_name in moments_to_use
        if haskey(target_moments_data["DataMoments"], moment_name)
            value = target_moments_data["DataMoments"][moment_name]
            if value !== nothing  # Skip null values
                TARGET_MOMENTS[Symbol(moment_name)] = value
                println("  $moment_name: $value")
            else
                println("  ⚠️  $moment_name: null (skipping)")
            end
        else
            println("  ❌ $moment_name: not found in data file")
        end
    end
    
    if isempty(TARGET_MOMENTS)
        println("❌ No valid target moments found!")
        exit(1)
    end
    
    # Load moment weights if specified
    global MOMENT_WEIGHTS = Dict()
    if haskey(target_moments_config, "moment_weights")
        for (moment, weight) in target_moments_config["moment_weights"]
            if haskey(TARGET_MOMENTS, Symbol(moment))
                MOMENT_WEIGHTS[Symbol(moment)] = weight
            end
        end
        println("Moment weights loaded: $MOMENT_WEIGHTS")
    else
        # Default equal weights
        for moment in keys(TARGET_MOMENTS)
            MOMENT_WEIGHTS[moment] = 1.0
        end
        println("Using default equal weights")
    end
end

# Distribute target moments and weights to all workers
@everywhere function set_target_moments(moments_dict, weights_dict)
    global TARGET_MOMENTS = moments_dict
    global MOMENT_WEIGHTS = weights_dict
end

@everywhere function set_config_file(file_path)
    global CONFIG_FILE_PATH = file_path
end

# Send the loaded moments and config file path to all workers
if nworkers() > 0
    @sync @distributed for w in workers()
        remotecall_wait(set_target_moments, w, TARGET_MOMENTS, MOMENT_WEIGHTS)
        remotecall_wait(set_config_file, w, config_file)
    end
    println("✓ Target moments and config file distributed to all $(nworkers()) workers")
else
    println("✓ Target moments loaded on main process")
    @everywhere CONFIG_FILE_PATH = $config_file
end

# --- Test worker setup ---
println("\n🔧 Testing worker setup...")
test_params = Dict("beta" => 0.96, "eta" => 0.5)
println("Running test evaluation on worker 1 with params: $test_params")

if nworkers() > 0
    try
        # Test evaluation on first worker
        test_result = remotecall_fetch(evaluate_objective_function, workers()[1], test_params)
        println("✅ Test evaluation successful: objective = $(test_result[1])")
        if test_result[1] >= 7e9
            println("⚠️  Test evaluation returned error penalty - investigating worker issues...")
        end
    catch e
        println("❌ Test evaluation failed: $e")
        println("This suggests worker setup issues that need to be resolved before proceeding.")
    end
else
    # Test on main process
    try
        test_result = evaluate_objective_function(test_params)
        println("✅ Test evaluation successful on main process: objective = $(test_result[1])")
    catch e
        println("❌ Test evaluation failed on main process: $e")
    end
end
println("Worker setup test completed.\n")

# --- 5. Define objective function on all workers ---
@everywhere function evaluate_objective_function(params::Dict{String, Float64})
    """
    Evaluate a single parameter set using the full model workflow:
    1. Initialize model with baseline config and parameter overrides
    2. Solve the model
    3. Compute model moments
    4. Calculate distance to target moments
    
    Returns: (objective_value, model_moments_dict)
    """
    try
        println("🔍 Starting evaluation with params: $params")
        println("🔧 Worker ID: $(myid()), PID: $(getpid())")
        println("📁 Current working directory: $(pwd())")
        
        # Debug: Check if all variables are defined
        println("🔍 Checking worker environment...")
        println("  CONFIG_FILE_PATH defined: $(isdefined(Main, :CONFIG_FILE_PATH))")
        println("  TARGET_MOMENTS defined: $(isdefined(Main, :TARGET_MOMENTS))")
        println("  MOMENT_WEIGHTS defined: $(isdefined(Main, :MOMENT_WEIGHTS))")
        
        if isdefined(Main, :CONFIG_FILE_PATH)
            println("  CONFIG_FILE_PATH value: $(CONFIG_FILE_PATH)")
        else
            println("❌ CONFIG_FILE_PATH not defined on worker!")
            return (7e9, Dict())
        end
        
        if isdefined(Main, :TARGET_MOMENTS)
            println("  TARGET_MOMENTS length: $(length(TARGET_MOMENTS))")
            println("  TARGET_MOMENTS keys: $(keys(TARGET_MOMENTS))")
        else
            println("❌ TARGET_MOMENTS not defined on worker!")
            return (7e9, Dict())
        end
        
        # Debug: Check if functions are defined
        println("🔍 Checking function availability...")
        function_checks = [
            ("initializeModel", isdefined(Main, :initializeModel)),
            ("update_primitives_results", isdefined(Main, :update_primitives_results)),
            ("solve_model", isdefined(Main, :solve_model)),
            ("compute_model_moments", isdefined(Main, :compute_model_moments)),
            ("compute_distance", isdefined(Main, :compute_distance))
        ]
        
        for (func_name, is_defined) in function_checks
            println("  $func_name: $is_defined")
            if !is_defined
                println("❌ Function $func_name not defined on worker!")
                return (7e9, Dict())
            end
        end
        
        # Check if config file exists and is accessible
        base_config_file = CONFIG_FILE_PATH
        println("📁 Using config file: $base_config_file")
        
        if !isfile(base_config_file)
            println("❌ Config file not found: $base_config_file")
            return (7e9, Dict())
        end
        
        println("✓ Config file exists")
        
        # Check if we have target moments BEFORE proceeding
        if length(TARGET_MOMENTS) == 0
            println("❌ No target moments available")
            return (9e9, Dict())  # Penalty: Missing target moments
        end
        
        println("🎯 Target moments available: $(length(TARGET_MOMENTS))")
        for (key, value) in TARGET_MOMENTS
            println("    $key: $value")
        end
        
        # Initialize model from base configuration
        println("🔧 Initializing model...")
        prim_base, res_base = initializeModel(base_config_file)
        println("✓ Model initialized successfully")
        
        # Convert string keys to symbols for model compatibility
        symbol_params = Dict(Symbol(k) => v for (k, v) in params)
        println("🔄 Converted params to symbols: $symbol_params")
        
        # Update primitives with new parameter values using ModelEstimation function
        println("📝 Updating primitives...")
        prim_new, res_new = update_primitives_results(
            prim_base, res_base, symbol_params
        )
        println("✓ Primitives updated successfully")
        
        # Solve the model using ModelSolver with config from file
        println("🧮 Solving model...")
        convergence_status = solve_model(prim_new, res_new; config=base_config_file)
        println("📊 Convergence status: $convergence_status")
        
        if convergence_status != :converged
            println("❌ Model did not converge")
            nan_moments = Dict(key => NaN for key in keys(TARGET_MOMENTS))
            return (8e9, nan_moments)  # Penalty: Model solver non-convergence
        end
        
        println("✓ Model converged successfully")
        
        # Compute model moments using ModelEstimation
        println("📈 Computing model moments...")
        moment_keys = Vector{Symbol}(collect(keys(TARGET_MOMENTS)))
        model_moments = compute_model_moments(prim_new, res_new; include=moment_keys)
        println("📊 Model moments computed: $model_moments")

        # Calculate distance using compute_distance from ModelEstimation
        println("📏 Computing distance...")
        objective = compute_distance(
            model_moments, 
            TARGET_MOMENTS,
            nothing,  # weighting_matrix
            nothing   # matrix_moment_order
        )
        
        println("✅ Evaluation completed successfully. Objective: $objective")
        
        # Return both objective and moments
        return (objective, model_moments)
        
    catch e
        # Enhanced error logging
        println("❌ ERROR in evaluate_objective_function:")
        println("  Worker ID: $(myid()), PID: $(getpid())")
        println("  Exception type: $(typeof(e))")
        println("  Exception message: $e")
        
        if isa(e, MethodError)
            println("  MethodError details:")
            println("    Function: $(e.f)")
            println("    Arguments: $(e.args)")
            println("    Argument types: $(typeof.(e.args))")
        elseif isa(e, UndefVarError)
            println("  Undefined variable: $(e.var)")
        elseif isa(e, BoundsError)
            println("  Bounds error: $(e)")
        elseif isa(e, LoadError)
            println("  Load error: $(e.error)")
            println("  File: $(e.file)")
            println("  Line: $(e.line)")
        elseif isa(e, SystemError)
            println("  System error: $(e.errnum) - $(e.prefix)")
        end
        
        # Print stack trace for debugging (first 15 frames)
        println("  Stack trace:")
        for (i, frame) in enumerate(stacktrace(catch_backtrace()))
            println("    $i: $frame")
            if i > 15  # Limit to prevent too much output
                break
            end
        end
        
        # Also print current working directory and environment
        println("  Current working directory: $(pwd())")
        println("  CONFIG_FILE_PATH: $(get(Main, :CONFIG_FILE_PATH, "UNDEFINED"))")
        println("  TARGET_MOMENTS length: $(length(TARGET_MOMENTS))")
        println("  Available functions: $(filter(x -> isa(getfield(Main, x), Function), names(Main)))")
        
        # Return large penalty with NaN moments for any errors
        nan_moments = Dict(key => NaN for key in keys(TARGET_MOMENTS))
        return (7e9, nan_moments)  # Penalty: Runtime error during evaluation
    end
end

@everywhere function evaluate_parameter_vector(params::Vector{Float64}, param_names)
    """
    Wrapper function to convert parameter vector to dict and evaluate.
    Accepts any param_names collection (workers may receive Vector{Any}).
    
    Returns: (objective_value, model_moments_dict)
    """
    # Ensure parameter names are strings (handles Symbols/Any from YAML load)
    names = string.(param_names)
    param_dict = Dict(zip(names, params))
    return evaluate_objective_function(param_dict)
end

# --- 6. Helper functions for progress tracking ---
function save_intermediate_results(parameter_vectors, objective_values, model_moments_list, param_names, completed, start_time)
    """
    Save intermediate results during the search for real-time monitoring
    """
    try
        # Filter out NaN values before finding best
        valid_indices = findall(x -> !isnan(x) && isfinite(x), objective_values)
        
        if isempty(valid_indices)
            # Use first evaluation as fallback, sanitizing NaN values
            best_idx = 1
            best_params_vector = parameter_vectors[1]
            best_objective = isnan(objective_values[1]) ? 1e10 : objective_values[1]
            best_moments = sanitize_nan_values(model_moments_list[1])
        else
            valid_objectives = objective_values[valid_indices]
            local_best_idx = argmin(valid_objectives)
            best_idx = valid_indices[local_best_idx]
            best_params_vector = parameter_vectors[best_idx]
            best_objective = objective_values[best_idx]
            best_moments = sanitize_nan_values(model_moments_list[best_idx])
        end
        
        best_params_dict = Dict(zip(param_names, best_params_vector))
        
        elapsed_time = time() - start_time
        total_evaluations = length(parameter_vectors)
        progress = completed / total_evaluations
        remaining = total_evaluations - completed
        avg_time_per_eval = elapsed_time / completed
        
        # Estimate time remaining
        eta_seconds = avg_time_per_eval * remaining
        
        # Calculate evaluation rate
        eval_rate = completed / elapsed_time
        
        # Get some statistics from completed evaluations
        valid_objectives = filter(x -> isfinite(x) && x < 1e9, objective_values[1:completed])
        obj_stats = if !isempty(valid_objectives)
            Dict(
                "mean" => mean(valid_objectives),
                "std" => std(valid_objectives),
                "min" => minimum(valid_objectives),
                "max" => maximum(valid_objectives),
                "median" => median(valid_objectives),
                "q25" => quantile(valid_objectives, 0.25),
                "q75" => quantile(valid_objectives, 0.75)
            )
        else
            Dict("mean" => NaN, "std" => NaN, "min" => NaN, "max" => NaN, 
                 "median" => NaN, "q25" => NaN, "q75" => NaN)
        end
        
        intermediate_results = Dict(
            "status" => "running",
            "job_id" => JOB_ID,
            "progress" => progress,
            "progress_pct" => progress * 100,
            "completed_evaluations" => completed,
            "total_evaluations" => total_evaluations,
            "remaining_evaluations" => remaining,
            "best_params" => best_params_dict,
            "best_params_vector" => best_params_vector,
            "best_objective" => sanitize_nan_values(best_objective),
            "best_moments" => best_moments,  # Already sanitized above
            "parameter_names" => param_names,
            "all_objectives" => sanitize_nan_values(objective_values),
            "all_params" => parameter_vectors,
            "all_moments" => sanitize_nan_values(model_moments_list),
            "elapsed_time_seconds" => elapsed_time,
            "elapsed_time_hours" => elapsed_time / 3600,
            "avg_time_per_eval" => avg_time_per_eval,
            "evaluation_rate" => eval_rate,
            "eta_seconds" => eta_seconds,
            "eta_minutes" => eta_seconds / 60,
            "eta_hours" => eta_seconds / 3600,
            "n_workers" => nworkers(),
            "n_evaluations" => completed,
            "timestamp" => string(Dates.now()),
            "objective_statistics" => sanitize_nan_values(obj_stats),
            "performance_metrics" => Dict(
                "evaluations_per_second" => eval_rate,
                "evaluations_per_minute" => eval_rate * 60,
                "evaluations_per_hour" => eval_rate * 3600,
                "parallel_efficiency" => nworkers() > 1 ? eval_rate / nworkers() : 1.0,
                "theoretical_max_rate" => nworkers() > 1 ? nworkers() / avg_time_per_eval : 1 / avg_time_per_eval
            ),
            "intermediate" => true
        )
        
        # Save with timestamp and job ID for monitoring to results directory
        results_dir = joinpath(MPI_SEARCH_DIR, "output", "results")
        mkpath(results_dir)  # Create results directory if it doesn't exist
        output_file = joinpath(results_dir, "mpi_search_results_job$(JOB_ID)_$(Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")).json")
        open(output_file, "w") do f
            JSON3.pretty(f, intermediate_results)
        end
        
        println("💾 Intermediate results saved: $(basename(output_file))")
        
    catch e
        println("⚠️  Failed to save intermediate results: $e")
    end
end

# --- 7. Generate parameter space ---
function generate_parameter_grid(config::Dict, n_samples::Int)
    """
    Generate parameter samples using Sobol quasi-random sequences for better space coverage
    This provides more uniform distribution across the parameter space than random sampling
    """
    # Handle different config structures for parameter bounds
    if haskey(config, "parameter_bounds")
        # Old structure: direct parameter_bounds dict
        param_bounds = config["parameter_bounds"]
        param_names = collect(keys(param_bounds))
    elseif haskey(config, "MPISearchConfig") && haskey(config["MPISearchConfig"], "parameters")
        # New structure: nested under MPISearchConfig
        param_config = config["MPISearchConfig"]["parameters"]
        param_names = param_config["names"]
        bounds_dict = param_config["bounds"]
        
        # Convert to dict format for compatibility
        param_bounds = Dict()
        for name in param_names
            param_bounds[name] = bounds_dict[name]
        end
    else
        error("Could not find parameter bounds in config. Expected either 'parameter_bounds' or 'MPISearchConfig.parameters'")
    end
    
    n_params = length(param_names)
    
    println("Generating Sobol sequence samples for $(n_params) parameters:")
    for (param, bounds) in param_bounds
        println("  $param: $bounds")
    end
    
    # Extract bounds into arrays for QuasiMonteCarlo
    lower_bounds = Float64[]
    upper_bounds = Float64[]
    
    for param_name in param_names
        bounds = param_bounds[param_name]
        push!(lower_bounds, bounds[1])
        push!(upper_bounds, bounds[2])
    end
    
    println("Using Sobol quasi-random sampling for better parameter space coverage...")
    
    # Generate Sobol sequence in unit hypercube [0,1]^n
    unit_hypercube_sample = QuasiMonteCarlo.sample(n_samples, n_params, SobolSample())
    
    # Manual scaling: scaled = lower + (upper - lower) * unit_sample
    scaled_samples = zeros(n_params, n_samples)
    for i in 1:n_params
        for j in 1:n_samples
            scaled_samples[i, j] = lower_bounds[i] + (upper_bounds[i] - lower_bounds[i]) * unit_hypercube_sample[i, j]
        end
    end
    
    # Convert to vector of vectors format expected by the rest of the code
    samples = [scaled_samples[:, i] for i in 1:n_samples]
    
    println("✓ Generated $(length(samples)) Sobol samples")
    println("  Sample bounds check:")
    for (i, param_name) in enumerate(param_names)
        param_values = [sample[i] for sample in samples]
        actual_min, actual_max = extrema(param_values)
        expected_min, expected_max = lower_bounds[i], upper_bounds[i]
        println("    $param_name: [$(round(actual_min, digits=4)), $(round(actual_max, digits=4))] (expected: [$(expected_min), $(expected_max)])")
    end
    
    return samples, param_names
end

# --- 8. Main optimization loop ---
function run_mpi_search(config::Dict)
    """
    Main distributed search function using validated configuration
    """
    println("\n🔍 Starting parameter search...")
    
    # Clean up old results and logs before starting
    cleanup_old_results()
    
    # Get number of samples from config and ensure it's an integer
    if haskey(config, "optimization")
        n_samples = Int(get(config["optimization"], "max_evaluations", 10000))
    elseif haskey(config, "MPISearchConfig")
        n_samples = Int(get(config["MPISearchConfig"], "n_samples", 10000))
    else
        n_samples = 10000
        println("⚠️  Using default n_samples = $n_samples")
    end
    
    # Generate parameter samples using validated bounds
    parameter_vectors, param_names = generate_parameter_grid(config, n_samples)
    
    println("Generated $(length(parameter_vectors)) parameter vectors")
    println("Parameters: $(param_names)")
    println("Number of workers: $(nworkers())")
    
    # Distribute evaluation across all workers with progress tracking
    start_time = time()
    
    println("🚀 Distributing evaluations across MPI workers...")
    println("📊 Progress will be tracked and intermediate results saved...")
    
    # Enhanced evaluation with progress tracking
    objective_values = Vector{Float64}(undef, length(parameter_vectors))
    model_moments_list = Vector{Dict}(undef, length(parameter_vectors))
    
    # Process in batches for progress reporting
    batch_size = max(1, div(length(parameter_vectors), 20))  # 20 progress updates
    
    for batch_start in 1:batch_size:length(parameter_vectors)
        batch_end = min(batch_start + batch_size - 1, length(parameter_vectors))
        batch_params = parameter_vectors[batch_start:batch_end]
        
        # Evaluate batch - now returns tuples of (objective, moments)
        batch_results = pmap(params -> evaluate_parameter_vector(params, param_names), batch_params)
        
        # Separate objectives and moments
        for (i, (obj, moments)) in enumerate(batch_results)
            idx = batch_start + i - 1
            objective_values[idx] = obj
            model_moments_list[idx] = moments
        end
        
        # Progress reporting
        completed = batch_end
        progress_pct = round((completed / length(parameter_vectors)) * 100, digits=1)
        elapsed = time() - start_time
        avg_time = elapsed / completed
        eta_seconds = avg_time * (length(parameter_vectors) - completed)
        eta_minutes = round(eta_seconds / 60, digits=1)
        
        # Find current best
        current_best_idx = argmin(objective_values[1:completed])
        current_best_obj = objective_values[current_best_idx]
        
        println("Progress: $(progress_pct)% ($(completed)/$(length(parameter_vectors))) | " *
                "Best: $(round(current_best_obj, digits=6)) | " *
                "ETA: $(eta_minutes) min | " *
                "Rate: $(round(1/avg_time, digits=2)) evals/sec")
        
        # Save intermediate results every 5% progress or every 100 evaluations
        if completed % max(100, div(length(parameter_vectors), 20)) == 0 && completed < length(parameter_vectors)
            save_intermediate_results(parameter_vectors[1:completed], 
                                     objective_values[1:completed], 
                                     model_moments_list[1:completed],
                                     param_names, completed, start_time)
        end
    end
    
    elapsed_time = time() - start_time
    
    println("Completed $(length(objective_values)) evaluations in $(round(elapsed_time, digits=2)) seconds")
    println("Average time per evaluation: $(round(elapsed_time/length(objective_values), digits=4)) seconds")
    
    # Find best parameters, filtering out NaN values
    valid_indices = findall(x -> !isnan(x) && isfinite(x), objective_values)
    
    if isempty(valid_indices)
        println("❌ All objective values are NaN or infinite! Check parameter bounds and model.")
        # Use first evaluation as fallback, replacing NaN with large value
        best_idx = 1
        best_params_vector = parameter_vectors[1]
        best_objective = isnan(objective_values[1]) ? 1e10 : objective_values[1]
        best_moments = model_moments_list[1]
        
        # Replace any NaN moments with zeros
        if haskey(best_moments, "model_moments") && any(isnan, values(best_moments["model_moments"]))
            best_moments["model_moments"] = Dict(k => isnan(v) ? 0.0 : v for (k,v) in best_moments["model_moments"])
        end
    else
        valid_objectives = objective_values[valid_indices]
        local_best_idx = argmin(valid_objectives)
        best_idx = valid_indices[local_best_idx]
        best_params_vector = parameter_vectors[best_idx]
        best_objective = objective_values[best_idx]
        best_moments = model_moments_list[best_idx]
        
        println("✓ Found $(length(valid_indices)) valid evaluations out of $(length(objective_values))")
    end
    
    # Convert best parameters to named dict
    best_params_dict = Dict(zip(param_names, best_params_vector))
    
    println("\nSEARCH RESULTS:")
    println("Best objective value: $best_objective")
    println("Best parameters:")
    for (name, value) in best_params_dict
        println("  $name: $(round(value, digits=6))")
    end
    
    println("Best model moments:")
    for (name, value) in best_moments
        if isfinite(value)
            println("  $name: $(round(value, digits=6))")
        else
            println("  $name: $value")
        end
    end
    
    # Save final results - sanitize all NaN values for JSON compatibility
    final_results = Dict(
        "status" => "completed",
        "job_id" => JOB_ID,
        "best_params" => best_params_dict,
        "best_params_vector" => best_params_vector,
        "best_objective" => sanitize_nan_values(best_objective),
        "best_moments" => sanitize_nan_values(best_moments),
        "parameter_names" => param_names,
        "all_objectives" => sanitize_nan_values(objective_values),
        "all_params" => parameter_vectors,  # Parameter vectors should be finite
        "all_moments" => sanitize_nan_values(model_moments_list),  # Sanitize all moments
        "elapsed_time" => elapsed_time,
        "n_workers" => nworkers(),
        "n_evaluations" => length(objective_values),
        "avg_time_per_eval" => elapsed_time/length(objective_values),
        "timestamp" => string(Dates.now()),
        "config_used" => config,
        "intermediate" => false,
        "final_results" => true
    )
    
    # Create a lightweight summary for quick access
    latest_results = Dict(
        "status" => "completed",
        "job_id" => JOB_ID,
        "best_params" => best_params_dict,
        "best_params_vector" => best_params_vector,
        "best_objective" => sanitize_nan_values(best_objective),
        "best_moments" => sanitize_nan_values(best_moments),
        "parameter_names" => param_names,
        "elapsed_time" => elapsed_time,
        "n_workers" => nworkers(),
        "n_evaluations" => length(objective_values),
        "avg_time_per_eval" => elapsed_time/length(objective_values),
        "timestamp" => string(Dates.now()),
        "search_summary" => Dict(
            "total_evaluations" => length(objective_values),
            "successful_evaluations" => length(filter(x -> isfinite(x) && x < 1e9, objective_values)),
            "best_objective" => sanitize_nan_values(best_objective),
            "objective_range" => begin
                valid_objs = filter(x -> isfinite(x) && x < 1e9, objective_values)
                if !isempty(valid_objs)
                    Dict("min" => minimum(valid_objs), "max" => maximum(valid_objs))
                else
                    Dict("min" => NaN, "max" => NaN)
                end
            end
        ),
        "intermediate" => false,
        "file_type" => "summary"
    )
    
    # Save final results to results directory
    results_dir = joinpath(MPI_SEARCH_DIR, "output", "results")
    mkpath(results_dir)  # Create results directory if it doesn't exist
    final_output_file = joinpath(results_dir, "mpi_search_results_job$(JOB_ID)_final.json")
    
    try
        # Save complete final results (large file with all data)
        open(final_output_file, "w") do f
            JSON3.pretty(f, final_results)
        end
        println("Final results (complete archive) saved to: $final_output_file")
        
        # Save lightweight summary results for quick access
        latest_output_file = joinpath(results_dir, "mpi_search_results_job$(JOB_ID)_latest.json")
        open(latest_output_file, "w") do f
            JSON3.pretty(f, latest_results)
        end
        println("Latest results (quick summary) saved to: $latest_output_file")
        
    catch e
        println("❌ Failed to save results: $e")
    end
    
    # Clean up intermediate snapshots, keeping only the latest one
    cleanup_intermediate_snapshots()
    
    return final_results
end

# --- 9. Execute search ---
try
    # Handle different config structures for parameter bounds
    if haskey(config, "parameter_bounds")
        # Old structure: direct parameter_bounds dict
        param_bounds = config["parameter_bounds"]
    elseif haskey(config, "MPISearchConfig") && haskey(config["MPISearchConfig"], "parameters")
        # New structure: nested under MPISearchConfig
        param_config = config["MPISearchConfig"]["parameters"]
        param_names = param_config["names"]
        bounds_array = param_config["bounds"]
    else
        println("❌ Could not find parameter bounds in config!")
    end
    
    # Print algorithm and optimization settings
    if haskey(config, "ga_params")
        println("\nGA Parameters:")
        for (key, value) in config["ga_params"]
            println("  $key: $value")
        end
    end
    
    if haskey(config, "optimization")
        println("\nOptimization settings:")
        for (key, value) in config["optimization"]
            println("  $key: $value")
        end
    elseif haskey(config, "MPISearchConfig")
        println("\nMPI Search settings:")
        mpi_config = config["MPISearchConfig"]
        println("  algorithm: $(get(mpi_config, "algorithm", "unknown"))")
        println("  n_samples: $(get(mpi_config, "n_samples", "unknown"))")
    end
    
    println("\nTarget moments configuration:")
    if haskey(config, "target_moments")
        println("  Moments to use: $(config["target_moments"]["moments_to_use"])")
    elseif haskey(config, "MPISearchConfig") && haskey(config["MPISearchConfig"], "target_moments")
        println("  Moments to use: $(config["MPISearchConfig"]["target_moments"]["moments_to_use"])")
    end
    
    println("  Loaded target moments:")
    for (key, value) in TARGET_MOMENTS
        weight = get(MOMENT_WEIGHTS, key, 1.0)
        println("    $key: $value (weight: $weight)")
    end
    
    # Run the search with real configuration
    results = run_mpi_search(config)
    
    println("MPI search completed successfully!")
    
finally
    # Clean up workers
    if nworkers() > 0
        println("Cleaning up $(nworkers()) workers...")
        rmprocs(workers())
        println("Workers removed")
    end
end

println("\n" * "=" ^ 50)
println("Distributed MPI Parameter Search Complete")
println("End time: $(Dates.now())")
