# ==========================================================================================

# estimation_experiment.jl
# ------------------------------------------------------------------------------------------
# Goal: Estimate a single parameter (c₀) three ways and benchmark time/efficiency:
#       1) Optim.jl (derivative-free)
#       2) MadNLP (via ADNLPModels)
#       3) Ipopt (via NLPModelsIpopt)
#
# Design:
# - We build a tiny synthetic model instance that mimics the structures in ModelSetup.jl.
# - We generate synthetic "data" (moments) from a known true c₀.
# - For any candidate c₀, we rebuild primitives/results, optionally solve equilibrium,
#   compute the same moments, and minimize squared errors to the synthetic data.
# - We time each solver and collect a small summary.
#
# Constraints from user:
# - Only reuse code from ModelSetup.jl and ModelSolver.jl; everything else is defined here.
# - Add comments in every line to show exactly what is going on.
# ==========================================================================================

# -- Standard libraries and packages we need for this script --
using Printf                           # For formatted printing in final summary lines
using Random                           # For seeding any randomness if needed (not used heavily)
using Statistics                       # For mean/aggregation helpers
using Dates                            # For simple timestamping if desired
using LinearAlgebra                    # For linear algebra operations

# -- Optimization packages requested by the user --
using Optim                            # Optim.jl for derivative-free box-constrained optimization
using Distributions                    # For Beta and pdf/cdf in grid recomputation
using PrettyTables                     # For nicely formatted timing/parameter summary (add with `] add PrettyTables` if needed)
using Optimization
using OptimizationOptimJL
using ForwardDiff   
# You may need to `] add` these if you don't have them

# -- Bring in only the requested code from the project --
include("src/structural_model_new/ModelSetup.jl")  # Defines Primitives, Results, thresholds, etc.
include("src/structural_model_new/ModelSolver.jl") # Defines solve_model and helpers we may reuse
include("src/structural_model_new/ModelEstimation.jl") # Defines compute_model_moments, perturb_parameters, etc.

# ==========================================================================================
# 0) Simple utility: precise timer helper to capture wall time in seconds
# ==========================================================================================
timeit(f) = begin                        # Define a tiny function timeit that takes a 0-arg function f
    t0 = time()                          #   Capture start time in seconds
    out = f()                            #   Execute the function and store output
    t1 = time()                          #   Capture end time in seconds
    return out, (t1 - t0)                #   Return (result, elapsed_seconds)
end                                      # End of timeit helper


# ==========================================================================================
# 1) YAML-based primitive builder (general): load once, then update parameters per evaluation
# ==========================================================================================
"""
    build_primitives_with_updates(updates; base_prim=nothing, yaml_path)

General builder to update one or more scalar parameters in `Primitives` and rebuild closures.

Arguments
- updates: NamedTuple, Dict, or iterator of Pairs mapping Symbol parameter names to values.
- base_prim: optional Primitives to copy and modify; if nothing, it will be loaded from YAML.
- yaml_path: path to YAML used if base_prim is nothing.

Notes
- Recreates production_fun, utility_fun, and matching_fun to reference `prim` fields, ensuring
  updated parameters are reflected without stale captured values.
- Assumes grid-related fields (e.g., n_h, n_ψ, h_grid, ψ_grid) are not changed during estimation.
"""
function build_primitives_with_updates(
                                        updates;
                                        base_prim::Union{Primitives,Nothing}=nothing,
                                        yaml_path::AbstractString="src/structural_model_new/model_parameters.yaml",
                                    )
    prim = base_prim === nothing ? create_primitives_from_yaml(yaml_path) : deepcopy(base_prim)

    # Force updates to Float64 to match concrete field types in Primitives
    upd_dict = Dict{Symbol,Float64}()
    for (k, v) in updates
        upd_dict[Symbol(k)] = float(v)
    end

    # The rest of the function is the same...
    for (k, v) in upd_dict
        if k in fieldnames(Primitives)
            setfield!(prim, k, v)
        else
            @warn "Unknown parameter in updates ignored" k
        end
    end

    # If Beta shape params changed, recompute h distribution on the existing grid
    if (:aₕ in keys(upd_dict)) || (:bₕ in keys(upd_dict))
        # Use current grid settings from prim
        n_h   = prim.n_h
        h_min = prim.h_min
        h_max = prim.h_max
        h_values = collect(range(h_min, h_max, length=n_h))
        # Map h_values to [0,1] for Beta
        h_scaled = (h_values .- h_min) ./ (h_max - h_min)
        beta_dist = Distributions.Beta(prim.aₕ, prim.bₕ)
        # Compute unnormalized pdf on grid
        h_pdf_raw = Distributions.pdf.(beta_dist, h_scaled)
        # Normalize pdf to sum to 1 (discrete approximation)
        h_pdf = h_pdf_raw ./ sum(h_pdf_raw)
        h_cdf = cumsum(h_pdf)
        # Update fields
        prim.h_grid = h_values
        prim.h_pdf  = h_pdf
        prim.h_cdf  = h_cdf
    end

    # Recreate closures to reference up-to-date prim fields
    prim.production_fun = (h, ψ, α) -> (prim.A₀ + prim.A₁*h) * ((1 - α) + α * (prim.ψ₀ * h^prim.ϕ * ψ^prim.ν))
    prim.utility_fun    = (w, α)    -> w - prim.c₀ * (1 - α)^(prim.χ + 1) / (prim.χ + 1)  # Fixed: χ instead of ϕ
    prim.matching_fun   = (V, U)    -> prim.γ₀ * U^prim.γ₁ * V^(1 - prim.γ₁)

    # Fresh Results (recomputes α_policy and thresholds)
    res = Results(prim)
    return prim, res
end


# ==========================================================================================
# 2) Objective builder: squared error to synthetic data as a function of parameters
# ==========================================================================================
function make_objective(
        data_moments::NamedTuple,
        W::AbstractMatrix{<:Real},
        base_prim::Primitives,
        param_map::NamedTuple
    )
    moment_names = keys(data_moments)
    data_vector = [getfield(data_moments, k) for k in moment_names]
    param_names = keys(param_map)
    counter = Ref(0)

    objective = function (x::AbstractVector{<:Real})
        counter[] += 1
        
        try
        # Build updates as Float64 to avoid injecting Duals into Float64-typed Primitives
        updates = Dict{Symbol,Float64}()
            for (i, pname) in enumerate(param_names)
                transform = getfield(param_map, pname)
                if transform == :log
            updates[pname] = float(exp(x[i]))
                elseif transform == :identity
            updates[pname] = float(x[i])
                else
                    error("Unknown transform: $transform")
                end
            end

            prim, res = build_primitives_with_updates(updates; base_prim=base_prim)
            solve_model(prim, res, verbose=false)
            model_moments_nt = compute_model_moments(prim, res; q_low_cut=0.25, q_high_cut=0.75)

            model_vector = [getfield(model_moments_nt, k) for k in moment_names]
            g = model_vector - data_vector
            loss = g' * W * g

            return loss

        catch e
            println("Warning: Solver failed at iteration $(counter[]). Returning Inf. Error: ", e)
            return Inf
        end
    end

    return objective, () -> counter[]
end

# ==========================================================================================
# 4) Estimation wrappers
# ==========================================================================================
function estimate_with_optim(obj, x0; lower=1e-6, upper=2.0, maxiters=50_000)
    # Run derivative-free box-constrained optimization with Optim.jl (L-BFGS in Fminbox).

    # Wrap the 0-arg execution so we can time just the optimization call
    result, seconds = timeit() do              # Start timing block
    # Build an initial vector (Optim expects a vector input for Fminbox/L-BFGS)
    x0_vec = x0 isa AbstractVector ? collect(float.(x0)) : [float(x0)]  # Accept scalar or vector
        # Construct the objective for Optim (it must accept AbstractVector)
        f = x -> obj(x)                        # Simple adapter, already matches required signature
        # Choose L-BFGS (quasi-Newton with gradient approximation) wrapped in Fminbox for bound constraints
        alg = Fminbox(LBFGS())                 # Algorithm selection (changed from NelderMead)
        # Set lower and upper bounds
    # Normalize bounds to vectors
    n = length(x0_vec)
    to_vec(b) = b isa AbstractVector ? collect(float.(b)) : fill(float(b), n)
    lower_bounds = to_vec(lower)
    upper_bounds = to_vec(upper)
        # Run the optimization with generous iterations (pass via Options, not keyword)
    return optimize(f, lower_bounds, upper_bounds, x0_vec, alg, 
                   Optim.Options(; iterations=maxiters, show_trace=true, extended_trace=true, show_every=10))
    end                                        # End of timing block

    # Extract the point estimate and objective value
    x_hat = Optim.minimizer(result)            # Estimated parameter vector
    fmin   = Optim.minimum(result)             # Final objective value

    # Attempt to get number of f calls (if available; otherwise set to missing)
    nevals = try
        Optim.f_calls(result)                  # Number of function evaluations recorded by Optim
    catch
        missing                                # Fallback if API changes
    end

    # Return a small NamedTuple with results
    return (x_hat=x_hat, fmin=fmin, seconds=seconds, nevals=nevals, result=result)
end                                            # End of estimate_with_optim

function estimate_with_sciml_lbfgs(obj, x0; lower, upper, maxiters=1000)
    # Use finite-difference gradients to keep everything Float64
    opt_func = OptimizationFunction((x, p) -> obj(x), Optimization.AutoFiniteDiff())
    opt_prob = OptimizationProblem(opt_func, x0, lb=lower, ub=upper)

    result, seconds = timeit() do
        try
            solve(opt_prob, LBFGS(), maxiters=maxiters)
        catch e
            println("The solve() command failed with an error: ", e)
            # Return a dummy object that matches the structure of a failed result
            return (retcode = SciMLBase.ReturnCode.Failure, u = x0, objective = Inf, stats = nothing)
        end
    end

    # --- FIX: This check will now work correctly ---
    if !SciMLBase.successful_retcode(result.retcode)
        println("Warning: Optimization failed or did not converge. Retcode: $(result.retcode)")
        return (x_hat=result.u, fmin=result.objective, seconds=seconds, nevals=missing, result=result)
    else
        x_hat = result.u
        fmin = result.objective
        nevals = hasproperty(result.stats, :nf) ? result.stats.nf : missing
        return (x_hat=x_hat, fmin=fmin, seconds=seconds, nevals=nevals, result=result)
    end
end

# ==========================================================================================
# 5) Main experiment runner
# ==========================================================================================
println("Starting estimation experiment...")
# Seed the random number generator for reproducibility (not essential here)
Random.seed!(1234)                         # Fix seed for reproducibility

# -- Build a base model once from YAML --
base_prim, _ = initializeModel("src/structural_model_new/model_parameters.yaml")  # Load default primitives/results

# -- Choose "true" parameter values for synthetic data (3 parameters total) --
#    Keep γ₀ and ψ₀ at their YAML defaults to avoid unintended shifts; set c₀ truth explicitly.
true_param_values = Dict{Symbol,Float64}(                                   # Dictionary of true parameters for data gen
    :c₀ => base_prim.c₀,                                                    # True c₀ (from YAML)
    # :ϕ  => base_prim.ϕ,                                                     # True ϕ (from YAML) 
    :ψ₀ => base_prim.ψ₀,                                                    # True ψ₀ (from YAML)
)


# -- Build truth primitives/results and compute synthetic data moments --
prim_true, res_true = build_primitives_with_updates(true_param_values; base_prim=base_prim)  # Apply truth values
solve_model(prim_true, res_true, verbose=false)                                              # Solve once at truth
data_moms = compute_model_moments(prim_true, res_true; q_low_cut=0.25, q_high_cut=0.75)      # Synthetic target moments

# -- Helper: per-parameter bounds (simple wide boxes; adjust if needed) --
param_bounds(sym::Symbol) =                                                   # Return (lower, upper) tuple by parameter
    sym === :c₀ ? (1e-6, 2.0) :
    # sym === :ϕ  ? ( -1.0, 5.0 ) :   # bounds for ϕ (allow negative values if utility curvature can be negative)
    sym === :ψ₀ ? (1e-6, 5.0) :
    (1e-6, 5.0)                                                               # Fallback for any other scalar

# -- Helper: initial guess given true value (mildly perturbed to be inside bounds) --
initial_guess(vtrue::Float64) = clamp(1.25*vtrue + 0.05, 1e-5, 4.9)           # Simple deterministic offset

# -- Define experiment configurations: 1p, 2p, 3p with Optim only --
experiments = [                                                                # Array of experiments to run
    (name = "1-param", params = [:c₀]),                                        # Estimate c₀
    (name = "2-param", params = [:c₀, :ψ₀]),                                    # Estimate c₀, ψ₀
    # (name = "3-param", params = [:c₀, :ψ₀, :ϕ]),                               # Estimate c₀, ϕ, ψ₀
]



# -- Run experiments and collect summaries --
summaries = Vector{NamedTuple}()                                               # To store results per experiment


n_ex = length(experiments)                          # Number of experiment configurations
results = Vector{Any}(undef, n_ex)                  # Preallocate a slot-per-experiment for thread-safety
for i in 1:n_ex
    printstyled("Running experiment $(i)/$(n_ex): $(experiments[i].name) ... ", color=:green, bold=true)
    ex = experiments[i]
    params = ex.params

    param_map = NamedTuple{Tuple(params)}([p === :c₀ || p === :ψ₀ ? :log : :identity for p in params])

    x_true_transformed = Float64[]
    x0_transformed = Float64[]
    
    # --- FIX: Transform the bounds to match the transformed parameters ---
    lowers_transformed = Float64[]
    uppers_transformed = Float64[]
    # --- END FIX ---

    for p in params
        true_val = true_param_values[p]
        initial_guess_val = initial_guess(true_val)
        lb, ub = param_bounds(p)

        transform = getfield(param_map, p)
        if transform == :log
            push!(x_true_transformed, log(true_val))
            push!(x0_transformed, log(initial_guess_val))
            # --- FIX: Transform the bounds ---
            push!(lowers_transformed, log(lb))
            push!(uppers_transformed, log(ub))
            # --- END FIX ---
        else
            push!(x_true_transformed, true_val)
            push!(x0_transformed, initial_guess_val)
            # --- FIX: Use untransformed bounds ---
            push!(lowers_transformed, lb)
            push!(uppers_transformed, ub)
            # --- END FIX ---
        end
    end

    W = Matrix(I(length(keys(data_moms))))
    obj, get_calls = make_objective(data_moms, W, base_prim, param_map)

    # --- FIX: Pass the TRANSFORMED bounds to the optimizer ---
    est = estimate_with_sciml_lbfgs(obj, x0_transformed; lower=lowers_transformed, upper=uppers_transformed, maxiters=50_000)
    # --- END FIX ---

    x_hat_untransformed = []
    # Check if estimation succeeded before trying to untransform
    if est.x_hat isa AbstractVector
        for (j, p) in enumerate(params)
            transform = getfield(param_map, p)
            if transform == :log
                push!(x_hat_untransformed, exp(est.x_hat[j]))
            else
                push!(x_hat_untransformed, est.x_hat[j])
            end
        end
    else
        x_hat_untransformed = est.x_hat # Keep the failed value (e.g., the initial guess)
    end


    results[i] = (
        name = ex.name,
        k = length(params),
        seconds = est.seconds,
        nevals = est.nevals,
        fmin = est.fmin,
        x_hat = x_hat_untransformed,
        x_true = [true_param_values[p] for p in params],
    )
    println("done. Time: $(round(est.seconds; digits=3))s, fmin: $(est.fmin), x_hat: $(x_hat_untransformed)")
end

# Collect thread-produced results into the summaries vector used later for printing

begin
    summaries = collect(results)

    # -- Print a short timing summary to stdout --
    println("\n=== Optim-only estimation scaling (1, 2, 3 parameters) ===")        # Header

    # Build rows as a Vector{Vector{Any}} (PrettyTables expects a matrix-like input, not a Vector{Tuple})
    rows_vec = [                                                                            # Build each row as a Vector{Any}
        Any[                                                                                 # Use Any so we can mix numbers and strings
            s.name,                                                                          # experiment name (String)
            s.k,                                                                             # number of estimated params (Int)
            s.seconds,                                                                       # elapsed seconds (Float64)
            s.nevals === missing ? "missing" : s.nevals,                                     # nevals (Int or "missing")
            s.fmin,                                                                          # final objective (Float64)
            string(round.(s.x_hat; digits=4)),                                               # estimated params (String)
            string(round.(s.x_true; digits=4)),                                              # true params (String)
        ] for s in summaries
    ]

    # Convert rows_vec into a rectangular Array{Any,2} for pretty_table
    nrows = length(rows_vec)                                                                 # number of experiments
    ncols = length(rows_vec[1])                                                              # number of columns
    data = Array{Any}(undef, nrows, ncols)                                                   # preallocate matrix
    for i in 1:nrows                                                                          # fill matrix row-by-row
        for j in 1:ncols
            data[i, j] = rows_vec[i][j]
        end
    end

    # Two-line header: names + units (units row can be empty where irrelevant)
    header = (
        ["experiment", "k", "time(s)", "nevals", "fmin", "x_hat", "x_true"],
        ["",           "",   "[s]",     "",       "",      "",        ""]
    )

    # Formatters: apply formatting to specific numeric columns (k, time, fmin)
    fmt_k   = ft_printf("%d", 2)               # integer formatter for 'k' (column 2)
    fmt_time= ft_printf("%7.3f", 3)            # floating formatter for time (column 3)
    fmt_fmin= ft_printf("% .4e", 5)            # scientific formatter for fmin (column 5)

    # Print the pretty table with a title, header styling, and rounded unicode frame
    pretty_table(
        data;
        header = header,
        formatters = (fmt_k, fmt_time, fmt_fmin),
        header_crayon = crayon"yellow bold",
        title = "Optim-only estimation scaling (1,2,3 params)",
        tf = tf_unicode_rounded,
        alignment = :l
    )

    # ---------------------------------------------------------------------
    # For each experiment: compute model moments at the estimated parameters
    # and print a PrettyTables table that compares target (data_moms) vs
    # model moments. The target moments are the same for all experiments.
    # ---------------------------------------------------------------------
    for i in 1:n_ex
        ex = experiments[i]                   # experiment tuple (name, params)
        s  = summaries[i]                     # corresponding summary produced above

        # Build updates dict from estimated x_hat -> param symbols
        pnames = ex.params
        xhat_vec = collect(s.x_hat)           # ensure a plain vector
        updates = Dict{Symbol,Float64}()
        for (j, pname) in enumerate(pnames)
            updates[pname] = float(xhat_vec[j])
        end

        # Build primitives/results at the estimated parameters (keep other params at truth)
        prim_est, res_est = build_primitives_with_updates(updates; base_prim=prim_true)
        solve_model(prim_est, res_est, verbose=false)
        model_moms = compute_model_moments(prim_est, res_est; q_low_cut=0.25, q_high_cut=0.75)

        # Extract moment names and values in the same order as data_moms
        moment_names = collect(keys(data_moms))
        n_m = length(moment_names)
        target_vals = [getfield(data_moms, k) for k in moment_names]
        model_vals  = [getfield(model_moms, k) for k in moment_names]

        # Build a matrix Array{Any,2} with columns: Moment | Target | Model
        mat = Array{Any}(undef, n_m, 3)
        for r in 1:n_m
            mat[r, 1] = string(moment_names[r])                 # moment name
            mat[r, 2] = target_vals[r]                          # numeric target
            mat[r, 3] = model_vals[r]                           # numeric model
        end

        # Header (two lines: names and units/blank)
        header_m = (["moment", "target", "model"], ["", "", ""])

        # Formatters: leave first column as-is, format numeric cols
        fmt_tgt = ft_printf("%10.6f", 2)
        fmt_mod = ft_printf("%10.6f", 3)

        # Create parameter summary string for the title
        param_str = join(["$(p)=$(round(updates[p]; digits=4))" for p in pnames], ", ")
        
        # Print the moment comparison table for this experiment
        pretty_table(
            mat;
            header = header_m,
            formatters = ( fmt_tgt, fmt_mod),
            header_crayon = crayon"yellow bold",
            title = "Moments — $(ex.name): target vs model | Params: $param_str",
            tf = tf_unicode_rounded,
            alignment = :l
        )
    end

    println("=====================================================================\n")  # Footer
end