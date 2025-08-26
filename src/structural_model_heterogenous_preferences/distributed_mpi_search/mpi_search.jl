# Distributed MPI Parameter Search for Structural Model
# This script uses ClusterManagers.jl and MPI to distribute parameter search
# across multiple nodes and cores for maximum scalability

# --- 1. Preamble and Setup ---
using Pkg
# Project activation is assumed via --project flag. Only activate if no project already set (fallback interactive use).
const PROJECT_PATH = get(ENV, "SEARCH_PROJECT_PATH", "/project/high_tech_ind/searching-flexibility")
if get(ENV, "JULIA_PROJECT", nothing) === nothing
    @info "No JULIA_PROJECT detected, activating fallback project at $(PROJECT_PATH)"
    Pkg.activate(PROJECT_PATH; io=devnull)
end

# Load required packages
using Distributed
using SlurmClusterManager
using YAML, Printf, Random
using Statistics, LinearAlgebra
using InteractiveUtils # for @which in debugging logs
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
const SYSIMAGE_PATH = joinpath(@__DIR__, "MPI_GridSearch_sysimage.so")
if haskey(ENV, "SLURM_NTASKS")
    requested_tasks = parse(Int, ENV["SLURM_NTASKS"])  # total tasks allocated
    cpus_per_task = try parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")) catch; 1 end
    if get(ENV, "FORCE_SINGLE_PROCESS", "0") == "1"
        println("FORCE_SINGLE_PROCESS=1: Skipping worker launch; running single process.")
    elseif requested_tasks == 1 && cpus_per_task > 1
        # Single Slurm task but many CPUs -> spawn local workers to use cores
        println("Single-task allocation detected with $cpus_per_task CPUs; launching local workers.")
        desired_workers = max(cpus_per_task - 1, 1)  # leave 1 for master
        exeflags = "--startup-file=no --project=/project/high_tech_ind/searching-flexibility"
        if isfile(SYSIMAGE_PATH) && get(ENV, "DISABLE_CUSTOM_SYSIMAGE", "0") != "1"
            exeflags *= " --sysimage=$(SYSIMAGE_PATH)"
        else
            println("⚠️  Sysimage not used (missing or disabled); workers will JIT compile.")
        end
        exeflags *= " -e 'ENV[\"JULIA_PKG_PRECOMPILE_AUTO\"]=\"0\"'"
        try
            addprocs(desired_workers; exeflags=exeflags)
            println("✓ Added $(desired_workers) local workers (IDs: $(workers()))")
        catch e
            println("❌ Local worker launch failed: $e")
        end
    else
        println("Attempting to launch SlurmManager with SLURM_NTASKS=$requested_tasks ...")
        try
            exeflags = "--startup-file=no --project=/project/high_tech_ind/searching-flexibility"
            if isfile(SYSIMAGE_PATH) && get(ENV, "DISABLE_CUSTOM_SYSIMAGE", "0") != "1"
                exeflags *= " --sysimage=$(SYSIMAGE_PATH)"
            else
                println("⚠️  Sysimage not used (missing or disabled); workers will JIT compile.")
            end
            exeflags *= " -e 'ENV[\"JULIA_PKG_PRECOMPILE_AUTO\"]=\"0\"'"
            addprocs(SlurmManager(); exeflags=exeflags)
            sleep(2)
            println("✓ Workers added: $(nworkers()) (IDs: $(workers()))")
            if nworkers() == 0
                println("⚠️  SlurmManager added 0 workers unexpectedly; falling back to local addprocs")
                local_workers = max(requested_tasks - 1, 1)
                addprocs(local_workers; exeflags=exeflags)
                println("✓ Added $local_workers local workers (fallback)")
            end
        catch e
            println("❌ SlurmManager launch failed: $e")
            local_workers = max(requested_tasks - 1, 1)
            try
                addprocs(local_workers; exeflags=exeflags)
                println("✓ Fallback local workers added: $(nworkers())")
            catch e2
                println("❌ Fallback local addprocs also failed: $e2")
            end
        end
    end
else
    println("⚠️  SLURM_NTASKS not found, running on single process")
end

# --- 3. Load model on all workers ---
@everywhere begin
    # Workers inherit project from --project exeflags; fallback only if missing
    if get(ENV, "JULIA_PROJECT", nothing) === nothing
        using Pkg
        const PROJECT_PATH = get(ENV, "SEARCH_PROJECT_PATH", "/project/high_tech_ind/searching-flexibility")
        try
            Pkg.activate(PROJECT_PATH; io=devnull)
        catch
        end
    end

    # Set paths
    const ROOT = "/project/high_tech_ind/searching-flexibility"
    const MODEL_DIR = joinpath(ROOT, "src", "structural_model_heterogenous_preferences")
    const MPI_SEARCH_DIR = joinpath(MODEL_DIR, "distributed_mpi_search")
    
    # Load required packages on workers
    using YAML, Printf, Random
    using Statistics, LinearAlgebra, Dates
    using DataFrames, CSV, Arrow
    using FixedEffectModels
    try
        using Vcov
    catch
        # optional
    end
    
    # Include model files
    include(joinpath(MODEL_DIR, "ModelSetup.jl"))
    include(joinpath(MODEL_DIR, "ModelSolver.jl"))  
    include(joinpath(MODEL_DIR, "ModelEstimation.jl"))
    flush(stdout)
    
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

# --- Simulation moments ONLY ---
# Analytic moment pipeline removed; all moments computed from simulated data.
global SIM_DATA_PATH = nothing  # Absolute path to simulation scaffolding feather file

@everywhere function set_sim_data_path(data_path)
    global SIM_DATA_PATH = data_path
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
                try
                    TARGET_MOMENTS[Symbol(moment_name)] = Float64(value)
                    println("  $moment_name: $value (added)")
                catch e
                    println("  ❌ Failed to add $moment_name (value=$value) -> $e")
                end
            else
                println("  ⚠️  $moment_name: null (skipping)")
            end
        else
            println("  ❌ $moment_name: not found in data file")
        end
    end
    println("Loaded $(length(TARGET_MOMENTS)) target moments.")
    
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

# --- Simulation-only moment computation setup ---
begin
    if haskey(config, "MPISearchConfig") && haskey(config["MPISearchConfig"], "moment_computation")
        mc = config["MPISearchConfig"]["moment_computation"]
        data_rel = get(mc, "simulated_data", nothing)
        if data_rel === nothing
            println("❌ Config missing MPISearchConfig.moment_computation.simulated_data (simulation-only mode)")
            exit(1)
        end
        SIM_DATA_PATH = isabspath(data_rel) ? data_rel : joinpath(ROOT, data_rel)
        if !isfile(SIM_DATA_PATH)
            println("❌ Simulated data file not found at $SIM_DATA_PATH (simulation-only mode)")
            exit(1)
        end
    else
        println("❌ Config missing MPISearchConfig.moment_computation block (simulation-only mode)")
        exit(1)
    end
    println("🧪 Simulation-only moment computation using data: $(SIM_DATA_PATH)")
    if nworkers() > 0
        @sync @distributed for w in workers()
            remotecall_wait(set_sim_data_path, w, SIM_DATA_PATH)
        end
        println("✓ Simulation data path distributed to workers")
    else
        @everywhere global SIM_DATA_PATH = $(SIM_DATA_PATH)
    end
end

# --- Define objective function on all workers BEFORE testing ---
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
        verbose = get(ENV, "VERBOSE_EVAL", "0") == "1"
        if verbose
            println("🔍 Starting evaluation with params: $params | worker $(myid()) pid $(getpid())")
        end
        # Minimal validation (avoid flooding logs)
        if !isdefined(Main, :CONFIG_FILE_PATH) || !isdefined(Main, :TARGET_MOMENTS) || length(TARGET_MOMENTS) == 0
            verbose && println("❌ Missing config or target moments on worker $(myid())")
            return (7e9, Dict())
        end
        # Function availability quick check
        if !(isdefined(Main, :initializeModel) && isdefined(Main, :solve_model) && isdefined(Main, :compute_model_moments))
            verbose && println("❌ Core functions missing on worker $(myid())")
            return (7e9, Dict())
        end
        
        # Check if config file exists and is accessible
        base_config_file = CONFIG_FILE_PATH
        if !isfile(base_config_file)
            verbose && println("❌ Config file not found: $base_config_file")
            return (7e9, Dict())
        end
        
        # Initialize model from base configuration
        prim_base, res_base = initializeModel(base_config_file)
        
        # Convert string keys to symbols for model compatibility
        symbol_params = Dict(Symbol(k) => v for (k, v) in params)
        
        prim_new, res_new = update_primitives_results(
            prim_base, res_base, symbol_params
        )
        
        convergence_status = solve_model(prim_new, res_new; config=base_config_file)
        # Support both Symbol and Tuple return (e.g., (:converged, anchoring_error))
        local status_symbol
        local anchoring_error = nothing
        if convergence_status isa Tuple
            status_symbol = convergence_status[1]
            if length(convergence_status) > 1
                anchoring_error = convergence_status[2]
            end
        else
            status_symbol = convergence_status
        end
        
        if status_symbol != :converged
            nan_moments = Dict(key => NaN for key in keys(TARGET_MOMENTS))
            return (8e9, nan_moments)  # Penalty: Model solver non-convergence
        end
        
        moment_keys = Vector{Symbol}(collect(keys(TARGET_MOMENTS)))
        if SIM_DATA_PATH === nothing
            verbose && println("❌ SIM_DATA_PATH not set; cannot compute simulation moments")
            nan_moments = Dict(key => NaN for key in keys(TARGET_MOMENTS))
            return (7e9, nan_moments)
        end
        local model_moments
        try
            if verbose
                println("[simulation-only] Using simulation data at $(SIM_DATA_PATH)")
            end
            sim_df = simulate_model_data(prim_new, res_new, SIM_DATA_PATH)
            model_moments = compute_model_moments_from_simulation(prim_new, res_new, sim_df; include_moments=moment_keys)
        catch e
            println("❌ Simulation moment computation failed on worker $(myid()): $e")
            nan_moments = Dict(key => NaN for key in keys(TARGET_MOMENTS))
            return (7e9, nan_moments)
        end
        
        # Early degeneracy penalty: if any regression-based key is sentinel (9999.0) or NaN, return large penalty immediately.
        # Identify typical regression-based moments (extendable).
        degenerate = false
        if isdefined(Main, :SENTINEL_MOMENT)
            sentinel_val = Main.SENTINEL_MOMENT
            for k in (:diff_logwage_inperson_remote, :wage_alpha, :wage_alpha_curvature)
                if haskey(model_moments, k)
                    v = model_moments[k]
                    if !isfinite(v) || (v == sentinel_val)
                        degenerate = true
                        break
                    end
                end
            end
        else
            # Fallback: treat NaN in any moment as degeneracy
            degenerate = any(x -> !isfinite(x[2]), collect(model_moments))
        end
        if degenerate
            verbose && println("⚠️  Degeneracy detected; returning penalty without distance computation")
            nan_moments = Dict(key => (haskey(model_moments, key) ? model_moments[key] : NaN) for key in keys(TARGET_MOMENTS))
            return (7.5e9, nan_moments)
        end

        objective = compute_distance(
            model_moments,
            TARGET_MOMENTS,
            nothing,
            nothing
        )
    verbose && println("✅ Eval worker $(myid()) obj=$(objective)")
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
        config_path = isdefined(Main, :CONFIG_FILE_PATH) ? Main.CONFIG_FILE_PATH : "UNDEFINED"
        println("  CONFIG_FILE_PATH: $config_path")
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

# --- Test worker setup ---
println("\n🔧 Testing worker setup...")

# Build a unicode-parameter test vector using midpoints of configured bounds
function _build_unicode_test_params(config)
    # Support both legacy and new config structures
    if haskey(config, "MPISearchConfig") && haskey(config["MPISearchConfig"], "parameters")
        pconf = config["MPISearchConfig"]["parameters"]
        names = pconf["names"]
        bounds = pconf["bounds"]
        d = Dict{String, Float64}()
        for nm in names
            b = bounds[nm]
            if length(b) == 2 && all(x -> isa(x, Number), b)
                d[string(nm)] = (float(b[1]) + float(b[2])) / 2
            end
        end
        return d
    elseif haskey(config, "parameter_bounds")
        d = Dict{String, Float64}()
        for (nm, b) in config["parameter_bounds"]
            if length(b) == 2 && all(x -> isa(x, Number), b)
                d[string(nm)] = (float(b[1]) + float(b[2])) / 2
            end
        end
        return d
    else
        return Dict{String, Float64}("μ" => 0.1) # minimal fallback
    end
end

if get(ENV, "SKIP_STARTUP_TEST_EVAL", "0") == "1"
    println("⏭️  Startup test evaluation skipped (SKIP_STARTUP_TEST_EVAL=1)")
else
    test_params = _build_unicode_test_params(config)
    println("Running test evaluation on worker 1 with unicode params: $(join(["$(k)=$(round(v,digits=4))" for (k,v) in test_params], ", "))")
    if nworkers() > 0
        try
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
        try
            test_result = evaluate_objective_function(test_params)
            println("✅ Test evaluation successful on main process: objective = $(test_result[1])")
        catch e
            println("❌ Test evaluation failed on main process: $e")
        end
    end
end
println("Worker setup test completed.\n")

# --- Optional early exit for debugging segfaults / environment issues ---
if get(ENV, "EARLY_EXIT_AFTER_INIT", "0") == "1"
    println("EARLY_EXIT_AFTER_INIT=1 set -> exiting before generating parameter grid / GA loop.")
    flush(stdout)
    exit(0)
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
            "evaluations_per_minute" => eval_rate * 60,
            "evaluations_per_hour" => eval_rate * 3600,
            "parallel_efficiency" => nworkers() > 1 ? eval_rate / nworkers() : 1.0,
            "theoretical_max_rate" => nworkers() > 1 ? nworkers() / avg_time_per_eval : 1 / avg_time_per_eval,
            "intermediate" => true,
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
"""
Run a Genetic Algorithm search. Returns a final_results Dict matching the structure
produced by the random/Sobol search path so downstream analysis stays unchanged.

GA features (initial implementation):
  * Real-valued chromosomes (continuous parameters)
  * Elitism: top elite_fraction carried unchanged
  * Tournament selection (size=3) for robustness
  * Blend crossover (uniform convex combination) with crossover_rate
  * Gaussian mutation (per gene) scaled by parameter range * mutation_scale (default 0.1)
  * Bounds enforcement via clamping
  * Intermediate result snapshots each generation (or every 5 generations if large)
  * Accumulates all evaluated individuals in all_params/all_objectives/all_moments
Limitations / TODO:
  * No duplicate suppression
  * No adaptive mutation or crossover scheduling yet
  * No niching / diversity preservation other than mutation
  * Early stopping criteria (stagnation) can be added later
"""
function run_ga_search(config::Dict)
    # Allow library / single-evaluation mode: skip GA entirely if env flag set
    if get(ENV, "SKIP_GA", "0") == "1"
        println("⏭️  SKIP_GA=1 detected – skipping Genetic Algorithm search phase.")
        return Dict{String,Any}("skipped" => true, "reason" => "SKIP_GA env flag")
    end
    println("\n🧬 Starting Genetic Algorithm search...")

    # Extract GA options
    ga_conf = get(get(config, "MPISearchConfig", Dict()), "search_options", Dict())
    population_size = Int(get(ga_conf, "population_size", 200))
    mutation_rate   = Float64(get(ga_conf, "mutation_rate", 0.2))
    crossover_rate  = Float64(get(ga_conf, "crossover_rate", 0.8))
    n_generations   = Int(get(ga_conf, "n_generations", 200))
    elite_fraction  = Float64(get(ga_conf, "elite_fraction", 0.1))
    mutation_scale  = Float64(get(ga_conf, "mutation_scale", 0.1))  # NEW optional parameter
    tournament_size = Int(get(ga_conf, "tournament_size", 3))

    # Early stopping configuration (Task 3)
    es_conf = get(ga_conf, "early_stopping", Dict())
    early_stopping_enabled = get(es_conf, "enabled", false)
    es_patience = Int(get(es_conf, "patience", 30))                # gens with no sufficient improvement
    es_min_improve = Float64(get(es_conf, "min_improvement", 1e-4)) # absolute improvement threshold
    es_mode = get(es_conf, "mode", "absolute")                     # "absolute" | "relative"
    es_warmup = Int(get(es_conf, "warmup_generations", 10))         # ignore improvements before this
    println("Early stopping: enabled=$(early_stopping_enabled) mode=$(es_mode) patience=$(es_patience) min_improve=$(es_min_improve) warmup=$(es_warmup)")

    # Auto bound expansion configuration
    abe_conf = get(ga_conf, "auto_bound_expansion", Dict())
    auto_bound_enabled = get(abe_conf, "enabled", false)
    abe_edge_fraction = Float64(get(abe_conf, "edge_fraction", 0.02))          # within 2% of range counts as edge
    abe_min_generations = Int(get(abe_conf, "min_generations", 5))              # consecutive gens hitting edge
    abe_population_edge_share = Float64(get(abe_conf, "population_edge_share", 0.15)) # fraction of pop near edge
    abe_expansion_factor = Float64(get(abe_conf, "expansion_factor", 1.5))      # multiply range (upper) or extend lower
    abe_max_expansions = Int(get(abe_conf, "max_expansions_per_param", 2))
    abe_global_upper_caps = Dict{String, Float64}()
    if haskey(abe_conf, "global_upper_caps")
        for (k,v) in abe_conf["global_upper_caps"]
            try
                abe_global_upper_caps[string(k)] = Float64(v)
            catch; end
        end
    end
    abe_global_lower_caps = Dict{String, Float64}()
    if haskey(abe_conf, "global_lower_caps")
        for (k,v) in abe_conf["global_lower_caps"]
            try
                abe_global_lower_caps[string(k)] = Float64(v)
            catch; end
        end
    end
    println("Auto bound expansion: enabled=$(auto_bound_enabled) edge_fraction=$(abe_edge_fraction) min_gens=$(abe_min_generations) edge_share=$(abe_population_edge_share) factor=$(abe_expansion_factor)")

    # Parameter bounds
    param_cfg = config["MPISearchConfig"]["parameters"]
    param_names = param_cfg["names"]
    bounds_dict = param_cfg["bounds"]
    n_params = length(param_names)

    lower_bounds = [Float64(bounds_dict[name][1]) for name in param_names]
    upper_bounds = [Float64(bounds_dict[name][2]) for name in param_names]
    ranges = upper_bounds .- lower_bounds

    # Track expansion related metadata
    expansion_counts = zeros(Int, n_params)
    upper_edge_streak = zeros(Int, n_params)
    lower_edge_streak = zeros(Int, n_params)
    expansions_history = Vector{Dict}()

    println("Parameters: $(param_names)")
    println("Population size=$population_size, generations=$n_generations, crossover_rate=$crossover_rate, mutation_rate=$mutation_rate, elite_fraction=$elite_fraction")

    elite_count = max(1, round(Int, elite_fraction * population_size))
    println("Elite count per generation: $elite_count")

    # Helper: random individual
    rand_individual() = [lower_bounds[i] + rand() * ranges[i] for i in 1:n_params]

    # Initial population (Sobol for better spread if available)
    initial_pop = begin
        try
            sobol_mat = QuasiMonteCarlo.sample(population_size, n_params, SobolSample())
            [ [lower_bounds[i] + (upper_bounds[i]-lower_bounds[i]) * sobol_mat[i,j] for i in 1:n_params] for j in 1:population_size ]
        catch
            [rand_individual() for _ in 1:population_size]
        end
    end

    population = initial_pop

    # Storage of ALL evaluations (for final archive)
    all_params = Vector{Vector{Float64}}()
    all_objectives = Float64[]
    all_moments = Vector{Dict}()

    # Evaluate a batch of individuals (vector of vectors)
    function evaluate_population(pop)
        batch_results = pmap(ind -> evaluate_parameter_vector(ind, param_names), pop)
        objs = Float64[]; moms = Vector{Dict}()
        for (obj, m) in batch_results
            push!(objs, obj)
            push!(moms, m)
        end
        return objs, moms
    end

    # Tournament selection: returns index
    function tournament_select(objs::Vector{Float64})
        idxs = rand(1:length(objs), min(tournament_size, length(objs)))
        best_local = idxs[1]; best_val = objs[best_local]
        for idx in idxs[2:end]
            if objs[idx] < best_val
                best_val = objs[idx]; best_local = idx
            end
        end
        return best_local
    end

    # Blend crossover (returns two children)
    function crossover(p1::Vector{Float64}, p2::Vector{Float64})
        α = rand()  # uniform blend
        c1 = [clamp(α*p1[i] + (1-α)*p2[i], lower_bounds[i], upper_bounds[i]) for i in 1:n_params]
        c2 = [clamp((1-α)*p1[i] + α*p2[i], lower_bounds[i], upper_bounds[i]) for i in 1:n_params]
        return c1, c2
    end

    # Gaussian mutation in-place
    function mutate!(ind::Vector{Float64})
        for i in 1:n_params
            if rand() < mutation_rate
                ind[i] += mutation_scale * ranges[i] * randn()
                ind[i] = clamp(ind[i], lower_bounds[i], upper_bounds[i])
            end
        end
        return ind
    end

    start_time = time()
    best_objective = Inf
    best_params_vector = nothing
    best_moments = Dict()
    generations_completed = 0
    early_stopped = false
    early_stopping_reason = ""
    last_improvement_gen = 0
    prev_best_for_rel = nothing

    # GA-specific intermediate save (Task 2 - accurate progress)
    function save_ga_intermediate_results(all_params, all_objectives, all_moments, param_names, generation, n_generations, start_time, best_params_vector, best_objective, best_moments)
        try
            elapsed_time = time() - start_time
            evals = length(all_objectives)
            gen_progress = generation / n_generations
            # Basic stats on current population (last population slice)
            recent_slice = last(all_objectives, min(population_size, length(all_objectives)))
            finite_objs = filter(x -> isfinite(x) && x < 1e9, recent_slice)
            pop_stats = if !isempty(finite_objs)
                Dict(
                    "mean" => mean(finite_objs),
                    "std" => std(finite_objs),
                    "min" => minimum(finite_objs),
                    "max" => maximum(finite_objs),
                    "median" => median(finite_objs)
                )
            else
                Dict("mean"=>NaN,"std"=>NaN,"min"=>NaN,"max"=>NaN,"median"=>NaN)
            end
            result = Dict(
                "status" => "running",
                "job_id" => JOB_ID,
                "algorithm" => "genetic_algorithm",
                "ga_generation" => generation,
                "ga_total_generations" => n_generations,
                "ga_progress" => gen_progress,
                "ga_progress_pct" => gen_progress*100,
                "completed_evaluations" => evals,
                "best_params" => Dict(zip(param_names, best_params_vector)),
                "best_params_vector" => best_params_vector,
                "best_objective" => sanitize_nan_values(best_objective),
                "best_moments" => sanitize_nan_values(best_moments),
                "parameter_names" => param_names,
                "elapsed_time_seconds" => elapsed_time,
                "evaluation_rate" => evals / elapsed_time,
                "population_statistics" => pop_stats,
                "auto_bound_expansion" => Dict(
                    "enabled" => auto_bound_enabled,
                    "expansion_counts" => Dict(param_names[i] => expansion_counts[i] for i in 1:length(param_names)),
                    "expansions_history" => expansions_history
                ),
                "timestamp" => string(Dates.now()),
                "early_stopping" => Dict(
                    "enabled" => early_stopping_enabled,
                    "patience" => es_patience,
                    "min_improvement" => es_min_improve,
                    "mode" => es_mode,
                    "warmup_generations" => es_warmup,
                    "last_improvement_generation" => last_improvement_gen,
                    "generations_since_improvement" => generation - last_improvement_gen
                ),
                "intermediate" => true
            )
            results_dir = joinpath(MPI_SEARCH_DIR, "output", "results"); mkpath(results_dir)
            output_file = joinpath(results_dir, "mpi_search_results_job$(JOB_ID)_$(Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")).json")
            open(output_file, "w") do f; JSON3.pretty(f, result); end
            # Append compact summary line to history (NDJSON) for easy reconstruction
            history_file = joinpath(results_dir, "mpi_search_results_job$(JOB_ID)_history.ndjson")
            summary = Dict(
                "ts" => Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"),
                "generation" => generation,
                "total_generations" => n_generations,
                "completed_evaluations" => evals,
                "best_objective" => sanitize_nan_values(best_objective),
                "ga_progress" => gen_progress,
                "elapsed_time_seconds" => elapsed_time,
                "evaluation_rate" => evals / max(elapsed_time, 1e-9),
                "best_params" => Dict(zip(param_names, best_params_vector))
            )
            open(history_file, "a") do f
                JSON3.write(f, summary); write(f, '\n')
            end
            # Append to (or create) CSV history with header once
            csv_file = joinpath(results_dir, "mpi_search_results_job$(JOB_ID)_history.csv")
            header_cols = ["ts","generation","total_generations","completed_evaluations","best_objective","ga_progress","elapsed_time_seconds","evaluation_rate"]
            append!(header_cols, String.(param_names))
            row_vals = [summary["ts"], string(generation), string(n_generations), string(evals), string(sanitize_nan_values(best_objective)), string(gen_progress), string(elapsed_time), string(evals / max(elapsed_time,1e-9))]
            for v in best_params_vector
                push!(row_vals, string(v))
            end
            write_header = !isfile(csv_file)
            open(csv_file, write_header ? "w" : "a") do f
                if write_header
                    write(f, join(header_cols, ",") * "\n")
                end
                write(f, join(row_vals, ",") * "\n")
            end
            println("💾 GA intermediate saved (gen $(generation)): $(basename(output_file))")
        catch e
            println("⚠️  GA intermediate save failed (gen $(generation)): $e")
        end
    end

    # Lightweight live status writer (every generation) ----------------------------------
    # Overwrites a single JSON file so the monitor can update continuously without
    # waiting for the heavier intermediate snapshot cadence.
    function save_ga_live_status(generation, n_generations, start_time, best_params_vector, best_objective)
        try
            elapsed_time = time() - start_time
            evals = length(all_objectives)
            gen_progress = generation / n_generations
            results_dir = joinpath(MPI_SEARCH_DIR, "output", "results"); mkpath(results_dir)
            live_file = joinpath(results_dir, "mpi_search_results_job$(JOB_ID)_live.json")
            live = Dict(
                "status" => "running",
                "job_id" => JOB_ID,
                "algorithm" => "genetic_algorithm",
                "ga_generation" => generation,
                "ga_total_generations" => n_generations,
                "ga_progress" => gen_progress,
                "ga_progress_pct" => gen_progress * 100,
                "completed_evaluations" => evals,
                "best_objective" => sanitize_nan_values(best_objective),
                "elapsed_time_seconds" => elapsed_time,
                "evaluation_rate" => evals / max(elapsed_time,1e-9),
                "timestamp" => string(Dates.now()),
                "best_params" => Dict(zip(param_names, best_params_vector))
            )
            open(live_file, "w") do f
                JSON3.write(f, live)
            end
        catch e
            println("⚠️  Live status save failed (gen $(generation)): $e")
        end
    end

    # Helper utilities -------------------------------------------------------
    safe_median(v::AbstractVector{<:Real}) = isempty(v) ? NaN : median(v)
    safe_mean(v::AbstractVector{<:Real}) = isempty(v) ? NaN : mean(v)
    function filter_valid(objs)
        # keep only finite, non-extreme penalty values
        return [x for x in objs if isfinite(x) && x < 1e9]
    end
    # Evaluate initial population
    println("Evaluating initial population...")
    objs, moms = evaluate_population(population)
    for i in eachindex(population)
        push!(all_params, population[i])
        push!(all_objectives, objs[i])
        push!(all_moments, moms[i])
    end

    gen_best_idx = argmin(objs)
    best_objective = objs[gen_best_idx]; best_params_vector = population[gen_best_idx]; best_moments = moms[gen_best_idx]
    finite0 = filter_valid(objs)
    med0 = safe_median(finite0)
    invalid0 = length(objs) - length(finite0)
    println("Gen 0 | Best=$(round(best_objective, digits=6)) median=$(round(med0, digits=4)) invalid=$(invalid0)")

    # GA loop
    for gen in 1:n_generations
        # Sort indices by objective
        valid_obj = objs
        sorted_idx = sortperm(valid_obj)
        elites = [deepcopy(population[i]) for i in sorted_idx[1:elite_count]]

        # Build next population
        new_population = Vector{Vector{Float64}}()
        append!(new_population, elites)

        while length(new_population) < population_size
            i1 = tournament_select(valid_obj)
            i2 = tournament_select(valid_obj)
            parent1 = population[i1]; parent2 = population[i2]
            child1 = deepcopy(parent1); child2 = deepcopy(parent2)
            if rand() < crossover_rate
                child1, child2 = crossover(parent1, parent2)
            end
            mutate!(child1); mutate!(child2)
            push!(new_population, child1)
            if length(new_population) < population_size
                push!(new_population, child2)
            end
        end

        population = new_population
        objs, moms = evaluate_population(population)
        for i in eachindex(population)
            push!(all_params, population[i])
            push!(all_objectives, objs[i])
            push!(all_moments, moms[i])
        end
        gen_best_idx = argmin(objs)
        gen_best_obj = objs[gen_best_idx]
        if gen_best_obj < best_objective
            best_objective = gen_best_obj
            best_params_vector = population[gen_best_idx]
            best_moments = moms[gen_best_idx]
        end
    finite_gen = filter_valid(objs)
    med_obj = safe_median(finite_gen)
    invalid_gen = length(objs) - length(finite_gen)
    println("Gen $gen | Best=$(round(best_objective, digits=6)) gen_best=$(round(gen_best_obj, digits=6)) median=$(round(med_obj, digits=4)) invalid=$(invalid_gen)")
    # Always write lightweight live status (every generation)
    save_ga_live_status(gen, n_generations, start_time, best_params_vector, best_objective)

        # --- Auto bound expansion logic ---
        if auto_bound_enabled && gen >= abe_min_generations
            # Compute population edge shares for each param
            # population currently sized 'population_size'
            for i in 1:n_params
                range_i = ranges[i]
                lb = lower_bounds[i]; ub = upper_bounds[i]
                edge_band = abe_edge_fraction * range_i
                # fraction near edges
                near_lower = count(ind -> (ind[i] - lb) <= edge_band, population) / population_size
                near_upper = count(ind -> (ub - ind[i]) <= edge_band, population) / population_size
                best_val = best_params_vector[i]
                best_near_lower = (best_val - lb) <= edge_band
                best_near_upper = (ub - best_val) <= edge_band
                # Update streaks
                if best_near_lower && near_lower >= abe_population_edge_share
                    lower_edge_streak[i] += 1
                else
                    lower_edge_streak[i] = 0
                end
                if best_near_upper && near_upper >= abe_population_edge_share
                    upper_edge_streak[i] += 1
                else
                    upper_edge_streak[i] = 0
                end
                # Decide expansion (prefer direction of best)
                if expansion_counts[i] < abe_max_expansions
                    expanded = false
                    # Upper expansion
                    if upper_edge_streak[i] >= abe_min_generations && best_near_upper
                        old_ub = ub
                        increment = range_i * (abe_expansion_factor - 1)
                        new_ub = ub + increment
                        if haskey(abe_global_upper_caps, string(param_names[i]))
                            new_ub = min(new_ub, abe_global_upper_caps[string(param_names[i])])
                        end
                        if new_ub > ub + 1e-12
                            upper_bounds[i] = new_ub
                            expanded = true
                        end
                    end
                    # Lower expansion
                    if !expanded && lower_edge_streak[i] >= abe_min_generations && best_near_lower
                        old_lb = lb
                        decrement = range_i * (abe_expansion_factor - 1)
                        new_lb = lb - decrement
                        # Honor configurable global_lower_caps (set in YAML) for any param.
                        # For aₕ,bₕ you requested using global_lower_caps: aₕ: 1e-6, bₕ: 1e-6.
                        pname = string(param_names[i])
                        # (Do not enforce additional hard-coded floor here; rely on caps.)
                        if haskey(abe_global_lower_caps, pname)
                            new_lb = max(new_lb, abe_global_lower_caps[pname])
                        end
                        # Prevent crossing upper bound and require actual decrease
                        if new_lb < lb - 1e-12 && new_lb < upper_bounds[i]
                            lower_bounds[i] = new_lb
                            expanded = true
                        end
                    end
                    if expanded
                        expansion_counts[i] += 1
                        # Recompute range
                        ranges[i] = upper_bounds[i] - lower_bounds[i]
                        push!(expansions_history, Dict(
                            "generation" => gen,
                            "parameter" => param_names[i],
                            "new_lower" => lower_bounds[i],
                            "new_upper" => upper_bounds[i],
                            "expansion_index" => expansion_counts[i]
                        ))
                        println("📈 Expanded bounds for $(param_names[i]) at gen $gen -> [$(round(lower_bounds[i],digits=6)), $(round(upper_bounds[i],digits=6))]")
                        # Reset streaks for this param to avoid immediate re-trigger
                        upper_edge_streak[i] = 0; lower_edge_streak[i] = 0
                    end
                end
            end
        end

        generations_completed = gen
        # Early stopping check (after warmup)
        # Track last improvement generation (best_objective already updated if improved earlier in loop)
        if gen_best_obj == best_objective
            last_improvement_gen = gen
        end
        if early_stopping_enabled && gen >= es_warmup
            improvement = (prev_best_for_rel === nothing) ? Inf : (prev_best_for_rel - best_objective)
            rel_improvement = (prev_best_for_rel === nothing || prev_best_for_rel == 0) ? Inf : improvement / prev_best_for_rel
            sufficient = es_mode == "relative" ? (rel_improvement >= es_min_improve) : (improvement >= es_min_improve)
            if sufficient
                last_improvement_gen = gen
                prev_best_for_rel = best_objective
            elseif prev_best_for_rel === nothing
                prev_best_for_rel = best_objective
            end
            gens_since = gen - last_improvement_gen
            if gens_since >= es_patience
                early_stopped = true
                early_stopping_reason = "No improvement >= $(es_min_improve) (mode=$(es_mode)) for $(gens_since) generations (patience=$(es_patience))"
                println("🛑 Early stopping triggered at generation $gen: $early_stopping_reason")
                # Final intermediate save before break
                save_ga_intermediate_results(all_params, all_objectives, all_moments, param_names, gen, n_generations, start_time, best_params_vector, best_objective, best_moments)
                break
            end
        end

        # Periodic intermediate save using GA-aware function (Task 2)
        if (gen % max(1, Int(cld(n_generations,20))) == 0) || gen == n_generations
            save_ga_intermediate_results(all_params, all_objectives, all_moments, param_names, gen, n_generations, start_time, best_params_vector, best_objective, best_moments)
        end
    end

    elapsed_time = time() - start_time
    println("GA completed: generations=$(generations_completed) of $n_generations evaluations=$(length(all_objectives)) elapsed=$(round(elapsed_time, digits=2))s early_stopped=$(early_stopped)")

    # Log final (possibly expanded) bounds
    println("Final parameter bounds (post-expansion if any):")
    for i in 1:n_params
        println("  $(param_names[i]): [$(round(lower_bounds[i],digits=6)), $(round(upper_bounds[i],digits=6))] (expansions=$(expansion_counts[i]))")
    end

    # Finalization (reuse logic pattern from random search path)
    best_params_dict = Dict(zip(param_names, best_params_vector))
    final_results = Dict(
        "status" => "completed",
        "job_id" => JOB_ID,
        "best_params" => best_params_dict,
        "best_params_vector" => best_params_vector,
        "best_objective" => sanitize_nan_values(best_objective),
        "best_moments" => sanitize_nan_values(best_moments),
        "parameter_names" => param_names,
        "all_objectives" => sanitize_nan_values(all_objectives),
        "all_params" => all_params,
        "all_moments" => sanitize_nan_values(all_moments),
        "elapsed_time" => elapsed_time,
        "n_workers" => nworkers(),
        "n_evaluations" => length(all_objectives),
        "avg_time_per_eval" => elapsed_time / max(1,length(all_objectives)),
        "ga_generations_completed" => generations_completed,
        "ga_total_generations" => n_generations,
        "ga_progress_pct" => generations_completed / n_generations * 100,
        "early_stopped" => early_stopped,
        "early_stopping_reason" => early_stopping_reason,
        "final_parameter_bounds" => Dict(param_names[i] => [lower_bounds[i], upper_bounds[i]] for i in 1:length(param_names)),
        "auto_bound_expansion" => Dict(
            "enabled" => auto_bound_enabled,
            "expansion_counts" => Dict(param_names[i] => expansion_counts[i] for i in 1:length(param_names)),
            "expansions_history" => expansions_history
        ),
        "timestamp" => string(Dates.now()),
        "config_used" => config,
        "algorithm" => "genetic_algorithm",
        "final_results" => true
    )

    # Save final results
    results_dir = joinpath(MPI_SEARCH_DIR, "output", "results"); mkpath(results_dir)
    final_output_file = joinpath(results_dir, "mpi_search_results_job$(JOB_ID)_final.json")
    latest_output_file = joinpath(results_dir, "mpi_search_results_job$(JOB_ID)_latest.json")
    try
        open(final_output_file, "w") do f; JSON3.pretty(f, final_results); end
        open(latest_output_file, "w") do f; JSON3.pretty(f, final_results); end
        println("GA final results saved: $final_output_file")
        # Append GA final summary to NDJSON history
        try
            history_file = joinpath(results_dir, "mpi_search_results_job$(JOB_ID)_history.ndjson")
            summary = Dict(
                "ts" => Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"),
                "generation" => generations_completed,
                "total_generations" => n_generations,
                "completed_evaluations" => length(all_objectives),
                "best_objective" => sanitize_nan_values(best_objective),
                "ga_progress" => generations_completed / max(n_generations,1),
                "elapsed_time_seconds" => elapsed_time,
                "evaluation_rate" => length(all_objectives)/max(elapsed_time,1e-9),
                "best_params" => best_params_dict,
                "final" => true,
                "early_stopped" => early_stopped
            )
            open(history_file, "a") do f; JSON3.write(f, summary); write(f, '\n'); end
            # Also append to CSV history
            try
                csv_file = joinpath(results_dir, "mpi_search_results_job$(JOB_ID)_history.csv")
                header_cols = ["ts","generation","total_generations","completed_evaluations","best_objective","ga_progress","elapsed_time_seconds","evaluation_rate"]
                append!(header_cols, String.(param_names))
                write_header = !isfile(csv_file)
                row_vals = [summary["ts"], string(generations_completed), string(n_generations), string(length(all_objectives)), string(sanitize_nan_values(best_objective)), string(generations_completed / max(n_generations,1)), string(elapsed_time), string(length(all_objectives)/max(elapsed_time,1e-9))]
                for pname in param_names
                    push!(row_vals, string(best_params_dict[pname]))
                end
                open(csv_file, write_header ? "w" : "a") do f
                    if write_header
                        write(f, join(header_cols, ",") * "\n")
                    end
                    write(f, join(row_vals, ",") * "\n")
                end
            catch e2
                println("⚠️  Failed to append GA final CSV row: $e2")
            end
        catch e
            println("⚠️  Failed to append GA final summary to history: $e")
        end
    catch e
        println("❌ Failed to save GA results: $e")
    end

    return final_results
end

function run_mpi_search(config::Dict)
    """
    Main distributed search function using validated configuration
    """
    println("\n🔍 Starting parameter search...")

    # Determine algorithm (currently only Sobol/random sampling path implemented)
    algorithm = begin
        if haskey(config, "MPISearchConfig")
            get(config["MPISearchConfig"], "algorithm", "random_search")
        elseif haskey(config, "optimization")
            get(config["optimization"], "algorithm", "random_search")
        else
            "random_search"
        end
    end
    if algorithm == "genetic_algorithm"
        return run_ga_search(config)
    elseif algorithm == "bayesian_optimization"
        println("⚠️  algorithm=bayesian_optimization specified, but BO loop not implemented. Falling back to Sobol sampling.")
    else
        println("Algorithm: $algorithm (Sobol sampling path)")
    end
    
    # (Disabled initial cleanup to preserve previous run results for concurrent analysis)
    # cleanup_old_results()
    
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
        # Append final summary (random search path) to NDJSON history
        try
            history_file = joinpath(results_dir, "mpi_search_results_job$(JOB_ID)_history.ndjson")
            summary = Dict(
                "ts" => Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"),
                "completed_evaluations" => length(objective_values),
                "best_objective" => sanitize_nan_values(best_objective),
                "elapsed_time_seconds" => elapsed_time,
                "evaluation_rate" => length(objective_values)/max(elapsed_time,1e-9),
                "best_params" => best_params_dict,
                "final" => true
            )
            open(history_file, "a") do f; JSON3.write(f, summary); write(f, '\n'); end
            # Append to CSV (random search has no generation fields)
            try
                csv_file = joinpath(results_dir, "mpi_search_results_job$(JOB_ID)_history.csv")
                header_cols = ["ts","generation","total_generations","completed_evaluations","best_objective","ga_progress","elapsed_time_seconds","evaluation_rate"]
                append!(header_cols, String.(param_names))
                write_header = !isfile(csv_file)
                row_vals = [summary["ts"], "", "", string(length(objective_values)), string(sanitize_nan_values(best_objective)), "", string(elapsed_time), string(length(objective_values)/max(elapsed_time,1e-9))]
                for pname in param_names
                    push!(row_vals, string(best_params_dict[pname]))
                end
                open(csv_file, write_header ? "w" : "a") do f
                    if write_header
                        write(f, join(header_cols, ",") * "\n")
                    end
                    write(f, join(row_vals, ",") * "\n")
                end
            catch e2
                println("⚠️  Failed to append random-search final CSV row: $e2")
            end
        catch e
            println("⚠️  Failed to append final summary (random path) to history: $e")
        end
        
    catch e
        println("❌ Failed to save results: $e")
    end
    
    # Clean up intermediate snapshots, keeping only the latest one
    cleanup_intermediate_snapshots()
    # Final archival cleanup of stale files from previous jobs now that new final exists
    cleanup_old_results()
    
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
    
    # Allow skipping the full search run via environment variable for quick testing
    run_full_search = get(ENV, "RUN_FULL_SEARCH", "1") == "1"
    if run_full_search
        # Run the search with real configuration
        results = run_mpi_search(config)
        println("MPI search completed successfully!")
    else
        println("⏭️  Skipping full search run (RUN_FULL_SEARCH != 1). Test evaluation only.")
    end
    
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
