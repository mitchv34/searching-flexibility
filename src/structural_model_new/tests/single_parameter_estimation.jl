using Pkg
# activate project containing Project.toml (adjust path if needed)
Pkg.activate(joinpath(@__DIR__, "../.."))
Pkg.instantiate()

using Distributed
addprocs(9)  # add local workers after activating environment

const ROOT = @__DIR__
@everywhere begin
    using Random, Statistics, SharedArrays, ForwardDiff
    using Optimization, OptimizationOptimJL, Optim
    include(joinpath($ROOT, "../ModelSetup.jl"))
    include(joinpath($ROOT, "../ModelSolver.jl"))
    include(joinpath($ROOT, "../ModelEstimation.jl"))
end
using Random, Statistics
using Term
using Printf
using CairoMakie
using PrettyTables
using Optim
using Optimization, OptimizationOptimJL


# Point to the configuration file
config = "src/structural_model_new/model_parameters.yaml"

# Select what parameters to estimate
params_to_estimate = [:aₕ, :bₕ, :c₀, :χ, :A₁, :ν, :ψ₀, :ϕ, :κ₀]

# --- Helper to build an OptimizationProblem with warm or cold starts ---
"""
    build_estimation_problem(; prim_base, res_base, target_moments, param_names, initial_guess;
                              weighting_matrix=nothing, matrix_moment_order=nothing,
                              warm_start::Bool=true,
                              solver_state_init=(λ_S_init=0.1, λ_u_init=0.1),
                              param_bounds=nothing)

Creates an OptimizationProblem for the structural estimation objective. If `warm_start`
is false, the internal `solver_state` is reset to `solver_state_init` on each
objective evaluation (cold start). If true, the same state is reused and updated (warm start).

Optionally pass bounds per parameter using:
- NamedTuple: (aₕ=(0.0, Inf), c₀=(0.0, 1.0))  # only constrains those listed
- Dict{Symbol, Tuple}: Dict(:aₕ => (0.0, Inf), :c₀ => (0.0, 1.0))

Unspecified parameters default to (-Inf, Inf).
"""
function build_estimation_problem(; prim_base,
                                    res_base,
                                    target_moments,
                                    param_names::Vector{Symbol},
                                    initial_guess::AbstractVector,
                                    weighting_matrix=nothing,
                                    matrix_moment_order=nothing,
                                    warm_start::Bool=true,
                                    solver_state_init=(λ_S_init=0.1, λ_u_init=0.1),
                                    param_bounds::Union{NamedTuple,Dict,Nothing}=nothing
                                )
    # Shared solver state reference (can be mutated across evaluations if warm_start)
    solver_state_ref = Ref(solver_state_init)

    # Prepare bounds arrays matching param order; default (-Inf, Inf)
    x0 = collect(initial_guess)
    T = eltype(x0) === Nothing ? Float64 : eltype(x0)
    n = length(x0)
    lb = fill(T(-Inf), n)
    ub = fill(T(Inf), n)

    if param_bounds !== nothing
        for (i, s) in enumerate(param_names)
            b = get(param_bounds, s, nothing)  # works for Dict and NamedTuple
            if b !== nothing
                if length(b) == 2
                    lo = T(b[1]); hi = T(b[2])
                    # allow swapped bounds; fix if needed
                    if lo > hi
                        lo, hi = hi, lo
                    end
                    lb[i] = lo
                    ub[i] = hi
                else
                    @warn "Bounds for $s must be 2 elements, got $b. Ignoring for this parameter."
                end
            end
        end
    end

    # Clamp initial guess into bounds (safe when bounds include ±Inf)
    for i in 1:n
        x0[i] = min(max(x0[i], lb[i]), ub[i])
    end

    p = (
        prim_base = prim_base,
        res_base = res_base,
        target_moments = target_moments,
        param_names = param_names,
        weighting_matrix = weighting_matrix,
        matrix_moment_order = matrix_moment_order,
        solver_state = solver_state_ref,
    )

    function obj_wrapped(x, pp)
        # Forbid NaN/±Inf and enforce bounds manually
        if any(!isfinite, x) || any(x .< lb) || any(x .> ub)
            return 1e12
        end
        if !warm_start
            pp.solver_state[] = solver_state_init
        end
        try
            return objective_function(x, pp)
        catch
            return 1e12
        end
    end

    of = OptimizationFunction(obj_wrapped)  # no gradient
    prob = OptimizationProblem(of, x0, p)#; lb = lb, ub = ub)  # lb/ub kept for reference
    return prob, p
end

# --- Single-parameter estimation for the first parameter using Nelder-Mead ---

# 1) Build "true" model and target moments (start from config values)
prim_true, res_true = initializeModel(config)
solve_model(prim_true, res_true, verbose=false)
target_moments = compute_model_moments(prim_true, res_true)


estimation_results = Dict{Symbol, Any}()
const results_lock = ReentrantLock()

# Choose which single parameter to estimate
# Threads.@threads for i in 1:length(params_to_estimate)
    i = 5
    param = params_to_estimate[i]
    true_value = getfield(prim_true, param)

    # 2) Starting guess (perturb the true value)
    perturb_size = 0.3
    initial_guess_scalar = true_value * (1 + perturb_size)
    @info "Estimating parameter" param true_value initial_guess_scalar

    # 3) Build optimization problem (toggle WarmStart here)
    WarmStart = true  # set to false for cold starts on every objective evaluation
    prob, p = build_estimation_problem(
        prim_base = prim_true,
        res_base = res_true,
        target_moments = target_moments,
        param_names = [param],
        initial_guess = [initial_guess_scalar],
        warm_start = WarmStart,
        solver_state_init = (λ_S_init = 0.1, λ_u_init = 0.1),
        param_bounds = (aₕ = (0.0, Inf), A₁ = (0.0, Inf)),  # NamedTuple; keys not listed are (-Inf, Inf)
    )

    # 4) Solve with Nelder-Mead (via Optimization -> Optim)
    opt_result = nothing
    sol = nothing
    elapsed = @elapsed begin
        sol = solve(
            prob,
            NelderMead();
            x_abstol = 1e-8, f_abstol = 1e-8, g_tol = 1e-6, iterations = 500
        )
        opt_result = sol.original
    end

    # Extract results
    est = try
        sol.u[1]
    catch
        NaN
    end
    abs_err = abs(est - true_value)

    # 5) Reporting
    Convergence = (sol.retcode == ReturnCode.Success)
    Value = true_value
    Estimate = isnan(est) ? "FAILED" : round(est, digits = 3)
    Error = isnan(abs_err) ? "N/A" : abs_err
    Evaluations = opt_result.f_calls
    true_reasons = [k for (k, v) in pairs(opt_result.stopped_by) if v]
    StopReason = isempty(true_reasons) ? "none" : join(string.(true_reasons), ", ")
    Time = round(elapsed, digits = 1)
    Iterations = opt_result.iterations

    if Convergence
        @info "Parameter $(param) estimation completed" Convergence WarmStart Value Estimate Error Time StopReason Iterations Evaluations
    else
        @error "Parameter $(param) estimation failed" Convergence WarmStart Value Estimate Error Time StopReason Iterations Evaluations
    end

    # Save the results (thread-safe update)
    lock(results_lock) do
        estimation_results[param] = Dict(
            "Convergence" => Convergence,
            "WarmStart" => WarmStart,
            "Value" => Value,
            "Estimate" => Estimate,
            "Error" => Error,
            "Time" => Time,
            "StopReason" => StopReason,
            "Iterations" => Iterations,
            "Evaluations" => Evaluations
    )
    end
end


rows = [
    [
        string(param),
        @sprintf("%.6f", result["Value"]),
        @sprintf("%.6f", result["Value"] * 1.3),  # initial guess = true_value * (1 + perturb_size)
        @sprintf("%.6f", result["Time"]),
        (result["Estimate"] == "FAILED" ? "FAILED" : @sprintf("%.6f", result["Estimate"])),
        (result["Error"] == "N/A" ? "N/A" : @sprintf("%.6f", result["Error"]))
    ]::Vector{String}
    for (param, result) in estimation_results
]
mat = Matrix{String}(undef, length(rows), length(rows[1]))
for i in 1:length(rows), j in 1:length(rows[1])
    mat[i, j] = rows[i][j]
end

pretty_table(mat; header = ["Parameter", "True value", "Initial guess", "Time (s)", "Estimate", "Abs error"])
