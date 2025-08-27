    ## run_search.jl
    # Driver script for ensemble local optimization using top GA candidates

    include("objective.jl") # Brings setup_problem_context, evaluate_for_optimizer, etc. into scope
    include("utils.jl")
    using YAML, CSV, DataFrames, Dates
    using .LocalObjectiveUtils
    using Optimization, OptimizationOptimJL, SciMLBase
    using Arrow, LinearAlgebra, PrettyTables, Statistics

    const LOCAL_REFINE_CONFIG_PATH = joinpath(@__DIR__, "local_refine_config.yaml")

    """
        load_diverse_top_candidates(; config_path, n=nothing, quality_quantile=0.1)

    Loads the top `n` candidates from a GA search, ensuring they are diverse.

    This function implements a two-stage selection process:
    1.  **Filter for Quality:** It first selects a pool of high-quality candidates by
        taking the top `quality_quantile` (e.g., 10%) of all evaluated points.
    2.  **Select for Diversity:** From this pool, it uses Farthest Point Sampling
        to select `n` points that are maximally separated in the normalized
        parameter space, guaranteeing a diverse set of starting points for
        local optimization.

    Args:
        config_path (String): Path to the MPI search configuration YAML file.
        n (Int, optional): The number of diverse candidates to return. Defaults to
                        `n_top_starts` in the config file.
        quality_quantile (Float64): The fraction of top candidates to consider as the
                                    high-quality pool (e.g., 0.1 for top 10%).

    Returns:
        A NamedTuple `(params, param_names)` where `params` is a Vector of
        parameter Vectors and `param_names` are their corresponding symbols.
    """
    function load_diverse_top_candidates(;
                                            config_path::String,
                                            n::Union{Int,Nothing}=nothing,
                                            quality_quantile::Float64=0.1
                                        )
        cfg = YAML.load_file(config_path)
        gsr = cfg["GlobalSearchResults"]
        job_id = get(gsr, "job_id", get(ENV, "LOCAL_GA_JOB_ID", nothing))
        job_id === nothing && error("No job id available in config or environment variables.")
        
        n === nothing && (n = get(gsr, "n_top_starts", 5))

        # --- 1. Load Full GA Results ---
        csv_path = replace(get(gsr, "latest_results_file", ""), "{job_id}" => string(job_id))
        !isfile(csv_path) && error("GA results CSV not found: $(csv_path).")
        
        df = CSV.read(csv_path, DataFrame)
        isempty(df) && error("No rows in GA results CSV: $(csv_path)")
        sort!(df, :objective) # Sort by best objective value

        # --- 2. Filter for Quality ---
        n_quality = ceil(Int, quality_quantile * nrow(df))
        quality_pool_df = first(df, n_quality)
        
        param_names_str = names(quality_pool_df)[2:end]
        param_names_sym = Symbol.(param_names_str)

        # --- 3. Normalize the Parameter Space ---
        #> Read the config used for the Global Optimizer to get the bounds 
        mpi_config = YAML.load_file(cfg["GlobalSearchResults"]["config_path"])
        bounds_dict = mpi_config["MPISearchConfig"]["parameters"]["bounds"]
        lower_bounds = [bounds_dict[k][1] for k in param_names_str]
        upper_bounds = [bounds_dict[k][2] for k in param_names_str]
        ranges = upper_bounds .- lower_bounds

        # Create a matrix of normalized parameters for the quality pool
        quality_params_matrix = Matrix(quality_pool_df[!, param_names_str])
        normalized_params = (quality_params_matrix .- lower_bounds') ./ ranges'

        # --- 4. Select for Diversity using Farthest Point Sampling ---
        selected_indices = Int[]
        
        # The first point is always the best one
        push!(selected_indices, 1)
        
        # The pool of available candidates to select from
        candidate_indices = Set(2:size(normalized_params, 1))

        while length(selected_indices) < min(n, size(normalized_params, 1))
            max_min_dist = -1.0
            best_next_idx = -1
            
            # Get the set of already selected points for distance calculation
            selected_points_matrix = normalized_params[selected_indices, :]

            for idx in candidate_indices
                candidate_point = normalized_params[idx, :]
                # Calculate distance from this candidate to all already selected points
                dists = vec(mapslices(p -> norm(p - candidate_point), selected_points_matrix, dims=2))
                min_dist = minimum(dists)
                
                if min_dist > max_min_dist
                    max_min_dist = min_dist
                    best_next_idx = idx
                end
            end
            
            push!(selected_indices, best_next_idx)
            delete!(candidate_indices, best_next_idx)
        end

        # --- 5. Return the Diverse, High-Quality Candidates ---
        final_params_df = quality_pool_df[selected_indices, :]
        
        # Convert the final DataFrame rows to a Vector of Vectors
        params = [Vector{Float64}(row) for row in eachrow(final_params_df[!, param_names_str])]
        
        @info "Selected $n diverse candidates from a quality pool of $n_quality."
        println("Final selected objective values: ", round.(final_params_df.objective, digits=4))

        return (params=params, param_names=param_names_sym)
    end

    """
        evaluate_initial_objectives(start_points, param_names, context) -> Vector{Float64}

    Evaluate the objective function at each of the initial candidate points.
    This is used for comparison with the final optimized solutions to analyze 
    the improvement achieved by local optimization.

    Args:
        start_points: Vector of parameter vectors (the initial starting points)
        param_names: Symbol vector of parameter names
        context: Problem context from setup_problem_context

    Returns:
        Vector{Float64}: Objective function values at each starting point
    """
    function evaluate_initial_objectives(start_points, param_names, context)
        println("Evaluating initial objective values at $(length(start_points)) starting points...")
        
        initial_objs = Vector{Float64}(undef, length(start_points))
        
        # Evaluate each starting point
        for i in 1:length(start_points)
            init_params = start_points[i]
            obj_val = evaluate_for_optimizer(init_params, context, param_names; verbose=false, solve_kwargs=Dict(:verbose => false))
            initial_objs[i] = obj_val
            println("Point $i: $(round(obj_val, digits=4))")
        end
        
        return initial_objs
    end

    """
        build_ensemble_problem(; config_path=LOCAL_REFINE_CONFIG_PATH, n_starts::Union{Int,Nothing}=nothing) -> NamedTuple

    Construct an `EnsembleProblem` for local refinement across top GA candidates.
    Loads YAML internally; if `n_starts` omitted uses `GlobalSearchResults.n_top_starts`.
    Returns NamedTuple: (ensemble_prob, base_problem, start_points, param_names, context)
    """
    function build_ensemble_problem(; config_path=LOCAL_REFINE_CONFIG_PATH, n_starts::Union{Int,Nothing}=nothing)
        cfg = load_local_refine_config(config_path)
        gsr = cfg["GlobalSearchResults"]
        # Load top candidates (now returns NamedTuple with vectors)
        topk = load_diverse_top_candidates(config_path=config_path, n=n_starts)
        raw_params = topk.params
        isempty(raw_params) && error("No candidates returned from GA results")

        # Full parameter name list from GA file (already supplied)
        full_pnames = topk.param_names

        # Optional subset
        subset = get(gsr, "parameter_subset", Any[])
        if subset isa Vector && !isempty(subset)
            param_names = Symbol.(subset)
            # Map subset order
            name_to_index = Dict(full_pnames[i] => i for i in eachindex(full_pnames))
            start_points = [ [p[name_to_index[nm]] for nm in param_names] for p in raw_params ]
        else
            param_names = full_pnames
            start_points = raw_params
        end

        # Build shared context once
        ctx = build_context_from_config(cfg)

        # Construct OptimizationProblem + EnsembleProblem
        f = OptimizationFunction(
            (u, p) -> evaluate_for_optimizer(u, p.context, p.param_names; verbose=false, solve_kwargs=Dict(:verbose => false)),
            Optimization.AutoForwardDiff()
        )
        dummy_u0 = start_points[1]
        p = (context = ctx, param_names = param_names)
        base_prob = OptimizationProblem(f, dummy_u0, p)
        prob_func = (prob, i, repeat) -> remake(prob; u0 = start_points[i])
        ensemble_prob = EnsembleProblem(base_prob; prob_func=prob_func)

        return (ensemble_prob=ensemble_prob,
                base_problem=base_prob,
                start_points=start_points,
                param_names=param_names,
                context=ctx)
    end

    """Load local refinement YAML config."""
    function load_local_refine_config(path::AbstractString)
        !isfile(path) && error("Local refine config not found: $path")
        return YAML.load_file(path)
    end

    """Build problem context from ModelInputs section of config."""
    function build_context_from_config(cfg::Dict)
        inputs = cfg["ModelInputs"]
        moment_filter = inputs["moment_filter"]
        moment_filter_syms = (moment_filter isa Vector && !isempty(moment_filter)) ? Symbol.(moment_filter) : nothing
        return setup_problem_context(; 
                                        config_path = inputs["config_path"],
                                        data_moments_yaml = inputs["data_moments_yaml"],
                                        weighting_matrix_csv = inputs["weighting_matrix_csv"],
                                        sim_data_path = inputs["sim_data_path"],
                                        moment_key_filter = moment_filter_syms
                                    )
    end

    # Main execution: Load top candidates and run ensemble optimization
    println("=" ^ 80)
    println("ENSEMBLE LOCAL OPTIMIZATION - NELDER-MEAD REFINEMENT")
    println("=" ^ 80)

    top = load_diverse_top_candidates(config_path=LOCAL_REFINE_CONFIG_PATH, n=20)
    ep = build_ensemble_problem(n_starts=length(top.params))

    # Evaluate initial objective values for comparison
    initial_objs = evaluate_initial_objectives(ep.start_points, ep.param_names, ep.context)

    println("\nStarting ensemble optimization with $(length(ep.start_points)) diverse candidates...")
    println("Parameter names: ", ep.param_names)
    println("⏱️  Starting optimization at $(Dates.format(now(), "HH:MM:SS"))...")

    # Run ensemble optimization with progress tracking
    start_time = time()
    nm_sol = solve(
        ep.ensemble_prob,
        NelderMead(),
        EnsembleThreads();
        trajectories = length(ep.start_points),
        maxiters = 10000,          # Proper maximum iterations
        abstol = 1e-8,             # Absolute tolerance for objective function
        reltol = 1e-8,             # Relative tolerance for objective function  
        f_reltol = 1e-10,             # Function tolerance
        x_abstol = 1e-10              # Parameter tolerance
    )

    elapsed_time = time() - start_time
    println("✅ All $(length(ep.start_points)) trajectories completed in $(round(elapsed_time, digits=1))s!")

    println("\nOptimization completed!")

    # Collect and organize results
    final_objs = [s.objective for s in nm_sol]
    final_params = [s.u for s in nm_sol]
    # Find the best solution
    best_idx = argmin(final_objs)
    println("\n🎯 BEST SOLUTION FOUND:")
    println("   Start point: $best_idx")
    println("   Final objective: $(round(final_objs[best_idx], digits=6))")
    println("   Improvement: $(round(initial_objs[best_idx] - final_objs[best_idx], digits=6))")

    # Create a comprehensive results table
    results_data = Matrix{Any}(undef, length(nm_sol), 6)
    for i in 1:length(nm_sol)
        improvement = initial_objs[i] - final_objs[i]
        convergence_status = nm_sol[i].retcode == :success ? "✓" : "✗"
        
        # Create parameter string (rounded to 4 digits)
        param_str = join([string(round(p, digits=3)) for p in final_params[i]], ", ")
        
        results_data[i, :] = [
            i,                                          # Start Point
            round(initial_objs[i], digits=4),           # Initial Obj
            round(final_objs[i], digits=4),             # Final Obj  
            round(improvement, digits=6),               # Improvement
            convergence_status,                         # Status
            param_str                                   # Parameters
        ]
    end

    # Print the results table
    println("\n" * "=" ^ 120)
    println("DETAILED OPTIMIZATION RESULTS")
    println("=" ^ 120)

    headers = ["Start", "Initial Obj", "Final Obj", "Improvement", "Status", "Final Parameters"]
    pretty_table(
        results_data,
        header = headers,
        header_crayon = crayon"yellow bold",
        crop = :none,
        alignment = [:c, :r, :r, :r, :c, :l],
        formatters = ft_round(4, [2, 3, 4])  # Round numeric columns to 4 digits
    )

    # Summary statistics
    println("\n📊 SUMMARY STATISTICS:")
    println("   Total runs: $(length(nm_sol))")
    println("   Successful convergences: $(sum(s.retcode == :success for s in nm_sol))")
    println("   Best objective: $(round(minimum(final_objs), digits=6))")
    println("   Worst objective: $(round(maximum(final_objs), digits=6))")
    println("   Average improvement: $(round(mean(initial_objs .- final_objs), digits=6))")
    println("   Total improvement: $(round(sum(initial_objs .- final_objs), digits=6))")

    println("\n✅ Optimization complete! Best parameters saved in nm_sol[$best_idx].u")
    println("=" ^ 80)