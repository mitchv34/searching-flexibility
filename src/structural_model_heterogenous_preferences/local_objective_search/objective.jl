# objective.jl
# Local objective function implementation for second-stage (local) GMM / SMM estimation.
# NOTE: Per user request, this file is NOT wrapped in a module. It `include`s the
# necessary definitions directly. Ensure you only include this once per session
# (or guard with @isdefined) to avoid method redefinition warnings.

# --- Load model components (idempotent guards) ---
# Use isdefined(Main, :Symbol) instead of @isdefined macro to avoid macro MethodErrors
if !isdefined(Main, :Primitives)
    include("../ModelSetup.jl")
end
if !(isdefined(Main, :solve_model) || isdefined(Main, :solve_model_no_anchor))
    include("../ModelSolver.jl")
end
if !(isdefined(Main, :update_primitives_results) && isdefined(Main, :compute_model_moments))
    include("../ModelEstimation.jl")
end

using CSV, DataFrames, YAML, LinearAlgebra, Printf, Random

# =====================================================================================
# Public API
# =====================================================================================

"""
    setup_problem_context(; config_path::String,
                             data_moments_yaml::String,
                             weighting_matrix_csv::String,
                             sim_data_path::String,
                             model_include_paths::Vector{String}=String[],
                             moment_key_filter::Union{Nothing,Vector{Symbol}}=nothing)

Load target (data) moments, weighting matrix, and provide a NamedTuple context used
by `evaluate_for_optimizer`.

Arguments
  config_path: YAML config for model primitives / grids.
  data_moments_yaml: Path to YAML containing DataMoments: {...} (one year) or
                     multi-year; expects top-level key "DataMoments".
  weighting_matrix_csv: CSV where first column lists moment names (order), and
                        remaining columns form a symmetric positive semi-definite W.
  sim_data_path: Path to simulation scaffolding (CSV / Feather) consumed by
                 `simulate_model_data` .
  model_include_paths: Optional list of files to `include` (e.g., solver, setup) if
                       not already brought into scope by caller. Safe to leave empty
                       when caller includes externally.
  moment_key_filter: Optional subset of moments to retain (Symbols). If provided,
                     both the weighting matrix and target moments will be filtered
                     (and rows/cols of W pruned) preserving order.

Returns NamedTuple with fields:
  config_file_path, target_moments::Dict{Symbol,Float64}, weighting_matrix::Matrix{Float64},
  matrix_moment_order::Vector{Symbol}, sim_data_path::String
"""
function setup_problem_context(; 
                                    config_path::String,
                                    data_moments_yaml::String,
                                    weighting_matrix_csv::String,
                                    sim_data_path::String,
                                    model_include_paths::Vector{String}=String[],
                                    moment_key_filter::Union{Nothing,Vector{Symbol}}=nothing
                                )

    for f in model_include_paths
        ispath(f) && include(f)
    end

    raw_yaml = YAML.load_file(data_moments_yaml)
    data_block = haskey(raw_yaml, "DataMoments") ? raw_yaml["DataMoments"] : raw_yaml
    target_moments = Dict{Symbol,Float64}()
    for (k,v) in data_block
        if !(v isa Missing) && v !== nothing && isnumeric(v)
            target_moments[Symbol(k)] = float(v)
        end
    end

    # Load weighting matrix (expect first column = moment names)
    w_df = CSV.read(weighting_matrix_csv, DataFrame)
    if ncol(w_df) < 2
        error("Weighting matrix CSV must have at least 2 columns (moment + values)")
    end
    moment_names = Symbol.(Vector(w_df[:,1]))
    W = Matrix(w_df[:, 2:end])
    size(W,1) == length(moment_names) || error("Weighting matrix dimension mismatch with moment list")
    size(W,1) == size(W,2) || error("Weighting matrix must be square")

    # Optional filtering
    if moment_key_filter !== nothing
        keep = [findfirst(==(m), moment_names) for m in moment_key_filter if m in moment_names]
        missing_keys = setdiff(moment_key_filter, moment_names)
        if !isempty(missing_keys)
            @warn "Requested filtered moment keys not found in weighting matrix" missing_keys
        end
        if isempty(keep)
            error("After filtering, no moments remain.")
        end
        W = W[keep, keep]
        moment_names = moment_names[keep]
    end

    return (config_file_path = config_path,
            target_moments = target_moments,
            weighting_matrix = W,
            matrix_moment_order = moment_names,
            sim_data_path = sim_data_path)
end

isnumeric(x) = x isa Number # Simple helper 

function build_param_dict(param_names::Vector{Symbol}, p_vec::AbstractVector{<:Real})
    length(param_names) == length(p_vec) || error("Parameter name vector length mismatch")
    return Dict(param_names[i] => float(p_vec[i]) for i in eachindex(param_names))
end

"""
    evaluate_for_optimizer(p_vec::AbstractVector{T}, problem_context::NamedTuple, param_names::Vector{Symbol};
                           solve_kwargs=Dict(), simulate_kwargs=Dict(), moment_subset::Union{Nothing,Vector{Symbol}}=nothing,
                           penalty_nonconv=8e9, penalty_degenerate=7.5e9, verbose::Bool=false) where T

Evaluate GMM/SMM objective g' W g for a given parameter vector. Designed to be low-allocation and
robust for use inside local optimizers (e.g., Optim.jl) or custom routines.
Generic implementation that supports both Float64 and ForwardDiff.Dual for automatic differentiation.
"""
function evaluate_for_optimizer(p_vec::AbstractVector{T},
                                problem_context::NamedTuple,
                                param_names::Vector{Symbol};
                                solve_kwargs=Dict(),
                                simulate_kwargs=Dict(),
                                moment_subset::Union{Nothing,Vector{Symbol}}=nothing,
                                penalty_nonconv=8e9,
                                penalty_degenerate=7.5e9,
                                verbose::Bool=false) where T
    
    try
        # 1. Map parameter vector
        params = build_param_dict(param_names, p_vec)

        # 2. Initialize primitives & results (fresh each evaluation; could cache for warm start)
        prim_base, res_base = initializeModel(problem_context.config_file_path)

        # 3. Update primitives with new params
        prim, res = update_primitives_results(prim_base, res_base, params)

        # 4. Solve model
        status = solve_model(prim, res; solve_kwargs...)
        status_symbol = status isa Tuple ? status[1] : status
        if status_symbol != :converged
            verbose && @printf("Non-convergence (status=%s). Penalizing.\n", string(status_symbol))
            return T(penalty_nonconv)  # Convert penalty to type T
        end

        # 5. Simulate model data & compute model moments
        model_moment_dict = Dict{Symbol,Float64}()
        wanted_keys = problem_context.matrix_moment_order
        if moment_subset !== nothing
            wanted_keys = [k for k in wanted_keys if k in moment_subset]
        end

        sim_df = simulate_model_data(prim, res, problem_context.sim_data_path)
        mdict = compute_model_moments(prim, res, sim_df; include_moments=wanted_keys)
        # `compute_model_moments` may return Dict{Symbol,Real}
        for k in wanted_keys
            if haskey(mdict, k) && mdict[k] isa Real
                model_moment_dict[k] = float(mdict[k])
            end
        end

        # 6. Build g vector in weighting matrix canonical order
        W_order = problem_context.matrix_moment_order
        g = Vector{T}(undef, length(W_order))  # Use generic type T instead of Float64
        for (i,key) in enumerate(W_order)
            target_val = get(problem_context.target_moments, key, NaN)
            model_val = get(model_moment_dict, key, NaN)
            if !isfinite(target_val) || !isfinite(model_val)
                return T(penalty_degenerate)  # Convert penalty to type T
            end
            g[i] = T(model_val) - T(target_val)  # Convert to type T
        end

        obj = g' * problem_context.weighting_matrix * g
        return obj
    catch e
        verbose && @printf("Error in evaluate_for_optimizer: %s\n", sprint(showerror, e))
        return T(penalty_degenerate)  # Convert penalty to type T
    end
end
