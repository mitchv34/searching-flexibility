using Pkg
# activate project containing Project.toml (adjust path if needed)
Pkg.activate("../../..")
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
using Combinatorics
using ProgressMeter
using JSON

Random.seed!(1234)

# Point to the configuration file
config = "src/structural_model_new/model_parameters.yaml"

# Select what parameters to estimate
# params_to_estimate = [:aₕ, :bₕ, :c₀, :χ, :A₁, :ν, :ψ₀, :ϕ, :κ₀]
params_to_estimate = [:aₕ, :bₕ, :c₀, :χ, :ν, :ψ₀, :ϕ, :κ₀]

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




function estimation_results_table(estimation_results; title::Union{Nothing, String}=nothing)
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

    pretty_table(mat; header = ["Parameter", "True value", "Initial guess", "Time (s)", "Estimate", "Abs error"], title=title)
end

# --- Single-parameter estimation for the first parameter using Nelder-Mead ---

# 1) Build "true" model and target moments (start from config values)
prim_true, res_true = initializeModel(config)
solve_model(prim_true, res_true, verbose=false)
target_moments = compute_model_moments(prim_true, res_true)


#?=========================================================================================
#? Pairs of parameters
#?=========================================================================================

# --- Pairs of parameters: threaded estimation (cold and warm) ---
function format_vec(v)
    if isempty(v)
        return "-"
    end
    return join([@sprintf("%.6f", x) for x in v], ", ")
end

# New helper: print all rows (no truncation)
function print_full_table(mat::Matrix{String}; header::Vector{String}=String[], title::Union{Nothing,String}=nothing)
    nrows, ncols = size(mat)
    # compute column widths
    col_widths = Vector{Int}(undef, ncols)
    for j in 1:ncols
        maxlen = 0
        if !isempty(header)
            maxlen = max(maxlen, length(header[j]))
        end
        for i in 1:nrows
            maxlen = max(maxlen, length(mat[i, j]))
        end
        col_widths[j] = maxlen
    end

    # printable helpers
    function padcell(s, w)
        return rpad(s, w)
    end

    if title !== nothing
        println(@bold @yellow "$(title)\n")
    end

    # header
    if !isempty(header)
        hdr = join([padcell(header[j], col_widths[j]) for j in 1:ncols], " │ ")
        println(hdr)
        println(join([repeat("─", col_widths[j]) for j in 1:ncols], "─┼─"))
    end

    # rows
    for i in 1:nrows
        println(join([padcell(mat[i, j], col_widths[j]) for j in 1:ncols], " │ "))
    end
    println() # trailing newline
end

# Updated pairs table printer: can print all rows if full=true
function pairs_estimation_results_table(estimation_results::Dict{String,Any}; title::Union{Nothing,String}=nothing, full::Bool=false)
    # Build list of entries with total absolute error, then sort descending (worst first)
    entries = Vector{NamedTuple{(:key,:r,:total),Tuple{String,Any,Float64}}}()
    for (k, r) in estimation_results
        abs_err = get(r, "AbsError", [])
        error_failed = get(r, "ErrorFailed", false)
        total = if error_failed
            Inf
        else
            isempty(abs_err) ? 0.0 : sum(Float64.(abs_err))
        end
        push!(entries, (key = k, r = r, total = total))
    end

    sort!(entries, by = e -> e.total, rev = true)  # worst (largest) first

    rows = Vector{Vector{String}}()
    for e in entries
        r = e.r
        push!(rows, [
            e.key,
            format_vec(get(r, "Value", [])),
            format_vec(get(r, "InitialGuess", [])),
            @sprintf("%.6f", get(r, "Time", 0.0)),
            (get(r, "EstimateFailed", false) ? "FAILED" : format_vec(get(r, "Estimate", []))),
            (get(r, "ErrorFailed", false) ? "N/A" : format_vec(get(r, "AbsError", []))),
            (isfinite(e.total) ? @sprintf("%.6f", e.total) : "Inf")
        ])
    end

    header = ["Parameters", "True values", "Initial guesses", "Time (s)", "Estimates", "Abs error", "Sum error"]

    if isempty(rows)
        pretty_table(String[], header = header, title=title)
        return
    end

    mat = Matrix{String}(undef, length(rows), length(rows[1]))
    for i in 1:length(rows), j in 1:length(rows[1])
        mat[i, j] = rows[i][j]
    end

    if full
        # use our full printer to avoid truncation
        print_full_table(mat; header=header, title=title)
    else
        pretty_table(mat; header = header, title=title, header_crayon = crayon"yellow bold")
    end
end


const results_pairs_lock = ReentrantLock()

# worker loop helper (run for a given WarmStart flag and store into target dict)
# now supports single-parameter runs and two perturb modes (:fixed or :random)
function run_multi_parameter_estimation!(param_list, target_dict::Dict{String,Any}, WarmStart::Bool; 
                                        perturb_mode::Symbol = :fixed, perturb_size::Float64 = 0.3,
                                        silent::Bool=false
                                        )

    total = length(param_list)
    prog = Progress(total; desc = WarmStart ? "Pairs (warm)" : "Pairs (cold)", showspeed=true)

    # prepare results directory & filename based on parameter arity + warm/cold + perturb mode
    results_dir = joinpath(ROOT, "estimation_results")
    mkpath(results_dir)
    param_count = isempty(param_list) ? 0 : length(first(param_list))
    # build filename without nested-quote interpolation to avoid parser issues
    filename = joinpath(results_dir,
                        string("results_", param_count, "_",
                                (WarmStart ? "warm" : "cold"), "_",
                                String(perturb_mode), ".json"))

    # load existing results (if any)
    existing_results = Dict{String,Any}()
    if isfile(filename)
        try
            txt = read(filename, String)
            existing_results = JSON.parse(txt)
        catch e
            @warn "Could not parse existing results file, proceeding fresh" filename e
            existing_results = Dict{String,Any}()
        end
    end

    Threads.@threads for idx in 1:total

        combo = param_list[idx]
        pname = join(string.(combo), ", ")

        # quick thread-safe check to skip combos already present (mark IN_PROGRESS to avoid races)
        skip = false
        lock(results_pairs_lock) do
            if haskey(existing_results, pname)
                # populate in-memory dict for consistency and skip
                target_dict[pname] = existing_results[pname]
                skip = true
            else
                # mark as in-progress to prevent other threads from running same combo
                existing_results[pname] = Dict("Status" => "IN_PROGRESS", "WarmStart" => WarmStart, "PerturbMode" => String(perturb_mode))
                target_dict[pname] = existing_results[pname]
                # persist immediately so other processes/threads see it
                try
                    open(filename, "w") do io
                        JSON.print(io, existing_results)
                    end
                catch e
                    @warn "Failed to write in-progress marker to file" filename e
                end
            end
        end

        if skip
            if !silent
                @info "Skipping already-computed combination" pname
            end
            next!(prog)
            continue
        end

        # true values and initial guesses (fixed or random perturb)
        true_vals = [getfield(prim_true, s) for s in combo]
        initial_guess = Vector{Float64}(undef, length(true_vals))
        if perturb_mode == :fixed
            initial_guess .= [v * (1 + perturb_size) for v in true_vals]
        elseif perturb_mode == :random
            for j in 1:length(true_vals)
                ps = rand()
                dir = rand(Bool) ? 1.0 : -1.0
                initial_guess[j] = true_vals[j] * (1 + dir * ps)
            end
        else
            error("Unknown perturb_mode: $perturb_mode")
        end

        if !silent
            @info "Running estimation for" pname "with perturb mode" perturb_mode
        end

        # build problem
        prob, p = build_estimation_problem(
            prim_base = prim_true,
            res_base = res_true,
            target_moments = target_moments,
            param_names = collect(combo),
            initial_guess = collect(initial_guess),
            warm_start = WarmStart,
            solver_state_init = (λ_S_init = 0.1, λ_u_init = 0.1),
            param_bounds = (aₕ = (0.0, Inf), A₁ = (0.0, Inf)),
        )

        # solve (derivative-free Nelder-Mead, bounds enforced inside objective)
        sol = nothing
        opt_result = nothing
        elapsed = @elapsed begin
            sol = solve(
                prob,
                Optim.NelderMead();
                x_abstol = 1e-8,
                f_abstol = 1e-8,
                g_tol     = 1e-6,
                iterations = 1000,
            )
            opt_result = sol.original
        end

        # extract estimates / errors
        est_vec = try
            collect(sol.u)
        catch
            []
        end
        abs_err_vec = isempty(est_vec) ? [] : abs.(est_vec .- collect(true_vals))

        Convergence = try 
            sol.retcode == ReturnCode.Success 
        catch 
            false 
        end
        EstimateFailed = isempty(est_vec)
        ErrorFailed = isempty(abs_err_vec)

        Iterations = try 
            opt_result.iterations 
        catch 
            missing
        end
        Evaluations = try 
            opt_result.f_calls 
        catch 
            missing 
        end
        StopReason = try 
            join(string.(keys(filter(x->x, opt_result.stopped_by))), ", ") 
        catch 
            "unknown" 
        end

        # build entry to store
        entry = Dict(
            "Convergence" => Convergence,
            "WarmStart" => WarmStart,
            "PerturbMode" => String(perturb_mode),
            "Value" => collect(true_vals),
            "InitialGuess" => collect(initial_guess),
            "Estimate" => EstimateFailed ? [] : collect(est_vec),
            "EstimateFailed" => EstimateFailed,
            "AbsError" => ErrorFailed ? [] : collect(abs_err_vec),
            "ErrorFailed" => ErrorFailed,
            "Time" => round(elapsed, digits=3),
            "Iterations" =>  Iterations,
            "Evaluations" =>  Evaluations,
            "StopReason" =>  StopReason
        )

        # thread-safe store + persist to file
        lock(results_pairs_lock) do
            existing_results[pname] = entry
            target_dict[pname] = entry
            try
                open(filename, "w") do io
                    JSON.print(io, existing_results)
                end
            catch e
                @warn "Failed to write results to file" filename e
            end
        end

        # advance atomic counter for the progress updater
        next!(prog) 
    end
    # wait for progress updater to finish
    finish!(prog)
end

# ---------- Removed redundant single-parameter threaded tests ----------
# The previous file contained several near-duplicate Threads.@threads blocks
# that ran single-parameter experiments. Those blocks are removed and replaced
# by a single, generic orchestration that uses `run_multi_parameter_estimation!`.


# --- Orchestrator: create, run and display estimations for 1..4 parameter combos ---
# This uses the generic `run_multi_parameter_estimation!` already defined above.
# For each arity k = 1..4 we run fixed-perturb and random-perturb experiments,
# both cold and warm starts, and then display results using the pairs printer.


for k in 1:4
    println(@bold @blue "\n\n============================== Running estimations for combinations of size $k ==============================\n\n")
    param_list = collect(combinations(params_to_estimate, k))
    if isempty(param_list)
        @info "No parameter combinations of size $k — skipping."
        continue
    end
    total = length(param_list)
    println(@bold @yellow "Total parameter combinations of size $k: $total\n\n")

    # Fixed perturbation (perturb_size = 0.3)
    results_fixed_warm_1 = Dict{String, Any}()
    run_multi_parameter_estimation!(
                                    param_list, # List of parameters to estimate 
                                    results_fixed_warm_1,  # Target to storage the parameters
                                    true;  # Warm-state estimation
                                    perturb_mode = :fixed,  # Random or fixed perturbation
                                    perturb_size = 0.3, # Size of the perturbation
                                    silent = true # Print or not info messages
                                    )

    # Display tables (use full=true for full rows if many columns)
    title_base = "Parameter combos (k=$k)"
    pairs_estimation_results_table(results_fixed_warm_1, title = "$title_base — fixed, warm", full=true)

    println(@bold @green "\n\n=============================================================================================================\n\n")
end 


#?=========================================================================================
#? Deprecated
#?=========================================================================================
# estimation_results_1_cold = Dict{Symbol, Any}()
# const results_lock = ReentrantLock()
# # Choose which single parameter to estimate
# Threads.@threads for i in 1:length(params_to_estimate)
#     param = params_to_estimate[i]
#     true_value = getfield(prim_true, param)

#     # 2) Starting guess (perturb the true value)
#     perturb_size = 0.3
#     initial_guess_scalar = true_value * (1 + perturb_size)
#     @info "Estimating parameter" param true_value initial_guess_scalar

#     # 3) Build optimization problem (toggle WarmStart here)
#     WarmStart = true  # set to false for cold starts on every objective evaluation
#     prob, p = build_estimation_problem(
#         prim_base = prim_true,
#         res_base = res_true,
#         target_moments = target_moments,
#         param_names = [param],
#         initial_guess = [initial_guess_scalar],
#         warm_start = WarmStart,
#         solver_state_init = (λ_S_init = 0.1, λ_u_init = 0.1),
#         param_bounds = (aₕ = (0.0, Inf), A₁ = (0.0, Inf)),  # NamedTuple; keys not listed are (-Inf, Inf)
#     )

#     # 4) Solve with Nelder-Mead (via Optimization -> Optim)
#     opt_result = nothing
#     sol = nothing
#     elapsed = @elapsed begin
#         sol = solve(
#             prob,
#             NelderMead();
#             x_abstol = 1e-8, f_abstol = 1e-8, g_tol = 1e-6, iterations = 500
#         )
#         opt_result = sol.original
#     end

#     # Extract results
#     est = try
#         sol.u[1]
#     catch
#         NaN
#     end
#     abs_err = abs(est - true_value)

#     # 5) Reporting
#     Convergence = (sol.retcode == ReturnCode.Success)
#     Value = true_value
#     Estimate = isnan(est) ? "FAILED" : round(est, digits = 3)
#     Error = isnan(abs_err) ? "N/A" : abs_err
#     Evaluations = opt_result.f_calls
#     true_reasons = [k for (k, v) in pairs(opt_result.stopped_by) if v]
#     StopReason = isempty(true_reasons) ? "none" : join(string.(true_reasons), ", ")
#     Time = round(elapsed, digits = 1)
#     Iterations = opt_result.iterations

#     if Convergence
#         @info "Parameter $(param) estimation completed" Convergence WarmStart Value Estimate Error Time StopReason Iterations Evaluations
#     else
#         @error "Parameter $(param) estimation failed" Convergence WarmStart Value Estimate Error Time StopReason Iterations Evaluations
#     end

#     # Save the results (thread-safe update)
#     lock(results_lock) do
#         estimation_results_1_cold[param] = Dict(
#             "Convergence" => Convergence,
#             "WarmStart" => WarmStart,
#             "Value" => Value,
#             "Estimate" => Estimate,
#             "Error" => Error,
#             "Time" => Time,
#             "StopReason" => StopReason,
#             "Iterations" => Iterations,
#             "Evaluations" => Evaluations
#     )
#     end
# end
# estimation_results_1_warm = Dict{Symbol, Any}()
# const results_lock = ReentrantLock()
# # Choose which single parameter to estimate
# Threads.@threads for i in 1:length(params_to_estimate)
#     param = params_to_estimate[i]
#     true_value = getfield(prim_true, param)

#     # 2) Starting guess (perturb the true value)
#     perturb_size = 0.3
#     initial_guess_scalar = true_value * (1 + perturb_size)
#     @info "Estimating parameter" param true_value initial_guess_scalar

#     # 3) Build optimization problem (toggle WarmStart here)
#     WarmStart = true  # set to false for cold starts on every objective evaluation
#     prob, p = build_estimation_problem(
#         prim_base = prim_true,
#         res_base = res_true,
#         target_moments = target_moments,
#         param_names = [param],
#         initial_guess = [initial_guess_scalar],
#         warm_start = WarmStart,
#         solver_state_init = (λ_S_init = 0.1, λ_u_init = 0.1),
#         param_bounds = (aₕ = (0.0, Inf), A₁ = (0.0, Inf)),  # NamedTuple; keys not listed are (-Inf, Inf)
#     )

#     # 4) Solve with Nelder-Mead (via Optimization -> Optim)
#     opt_result = nothing
#     sol = nothing
#     elapsed = @elapsed begin
#         sol = solve(
#             prob,
#             NelderMead();
#             x_abstol = 1e-8, f_abstol = 1e-8, g_tol = 1e-6, iterations = 500
#         )
#         opt_result = sol.original
#     end

#     # Extract results
#     est = try
#         sol.u[1]
#     catch
#         NaN
#     end
#     abs_err = abs(est - true_value)

#     # 5) Reporting
#     Convergence = (sol.retcode == ReturnCode.Success)
#     Value = true_value
#     Estimate = isnan(est) ? "FAILED" : round(est, digits = 3)
#     Error = isnan(abs_err) ? "N/A" : abs_err
#     Evaluations = opt_result.f_calls
#     true_reasons = [k for (k, v) in pairs(opt_result.stopped_by) if v]
#     StopReason = isempty(true_reasons) ? "none" : join(string.(true_reasons), ", ")
#     Time = round(elapsed, digits = 1)
#     Iterations = opt_result.iterations

#     if Convergence
#         @info "Parameter $(param) estimation completed" Convergence WarmStart Value Estimate Error Time StopReason Iterations Evaluations
#     else
#         @error "Parameter $(param) estimation failed" Convergence WarmStart Value Estimate Error Time StopReason Iterations Evaluations
#     end

#     # Save the results (thread-safe update)
#     lock(results_lock) do
#         estimation_results_1_warm[param] = Dict(
#             "Convergence" => Convergence,
#             "WarmStart" => WarmStart,
#             "Value" => Value,
#             "Estimate" => Estimate,
#             "Error" => Error,
#             "Time" => Time,
#             "StopReason" => StopReason,
#             "Iterations" => Iterations,
#             "Evaluations" => Evaluations
#     )
#     end
# end


# # Example usage:
# estimation_results_table(estimation_results_1_cold, title="Single-parameter estimation (cold starts)")
# estimation_results_table(estimation_results_1_warm, title="Single-parameter estimation (warm starts)")



# estimation_results_2_cold = Dict{Symbol, Any}()
# const results_lock_2_cold = ReentrantLock()
# Threads.@threads for i in 1:length(params_to_estimate)
#     param = params_to_estimate[i]
#     true_value = getfield(prim_true, param)

#     # 2) Starting guess (perturb the true value)
#     perturb_size = rand()  # uniform between 0 and 1
#     direction = rand(Bool) ? 1.0 : -1.0
#     initial_guess_scalar = true_value * (1 + direction * perturb_size)
#     @info "Estimating parameter" param true_value initial_guess_scalar

#     # 3) Build optimization problem (toggle WarmStart here)
#     WarmStart = true  # set to false for cold starts on every objective evaluation
#     prob, p = build_estimation_problem(
#         prim_base = prim_true,
#         res_base = res_true,
#         target_moments = target_moments,
#         param_names = [param],
#         initial_guess = [initial_guess_scalar],
#         warm_start = WarmStart,
#         solver_state_init = (λ_S_init = 0.1, λ_u_init = 0.1),
#         param_bounds = (aₕ = (0.0, Inf), A₁ = (0.0, Inf)),  # NamedTuple; keys not listed are (-Inf, Inf)
#     )

#     # 4) Solve with Nelder-Mead (via Optimization -> Optim)
#     opt_result = nothing
#     sol = nothing
#     elapsed = @elapsed begin
#         sol = solve(
#             prob,
#             NelderMead();
#             x_abstol = 1e-8, f_abstol = 1e-8, g_tol = 1e-6, iterations = 500
#         )
#         opt_result = sol.original
#     end

#     # Extract results
#     est = try
#         sol.u[1]
#     catch
#         NaN
#     end
#     abs_err = abs(est - true_value)

#     # 5) Reporting
#     Convergence = (sol.retcode == ReturnCode.Success)
#     Value = true_value
#     Estimate = isnan(est) ? "FAILED" : round(est, digits = 3)
#     Error = isnan(abs_err) ? "N/A" : abs_err
#     Evaluations = opt_result.f_calls
#     true_reasons = [k for (k, v) in pairs(opt_result.stopped_by) if v]
#     StopReason = isempty(true_reasons) ? "none" : join(string.(true_reasons), ", ")
#     Time = round(elapsed, digits = 1)
#     Iterations = opt_result.iterations

#     if Convergence
#         @info "Parameter $(param) estimation completed" Convergence WarmStart Value Estimate Error Time StopReason Iterations Evaluations
#     else
#         @error "Parameter $(param) estimation failed" Convergence WarmStart Value Estimate Error Time StopReason Iterations Evaluations
#     end

#     # Save the results (thread-safe update)
#     lock(results_lock_2_cold) do
#         estimation_results_2_cold[param] = Dict(
#             "Convergence" => Convergence,
#             "WarmStart" => WarmStart,
#             "Value" => Value,
#             "Estimate" => Estimate,
#             "Error" => Error,
#             "Time" => Time,
#             "StopReason" => StopReason,
#             "Iterations" => Iterations,
#             "Evaluations" => Evaluations
#     )
#     end
# end

# estimation_results_2_warm = Dict{Symbol, Any}()
# const results_lock_2_warm = ReentrantLock()
# # Choose which single parameter to estimate
# Threads.@threads for i in 1:length(params_to_estimate)
#     param = params_to_estimate[i]
#     true_value = getfield(prim_true, param)

#     # 2) Starting guess (perturb the true value)
#     perturb_size = rand()  # uniform between 0 and 1
#     direction = rand(Bool) ? 1.0 : -1.0
#     initial_guess_scalar = true_value * (1 + direction * perturb_size)
#     @info "Estimating parameter" param true_value initial_guess_scalar

#     # 3) Build optimization problem (toggle WarmStart here)
#     WarmStart = true  # set to false for cold starts on every objective evaluation
#     prob, p = build_estimation_problem(
#         prim_base = prim_true,
#         res_base = res_true,
#         target_moments = target_moments,
#         param_names = [param],
#         initial_guess = [initial_guess_scalar],
#         warm_start = WarmStart,
#         solver_state_init = (λ_S_init = 0.1, λ_u_init = 0.1),
#         param_bounds = (aₕ = (0.0, Inf), A₁ = (0.0, Inf)),  # NamedTuple; keys not listed are (-Inf, Inf)
#     )

#     # 4) Solve with Nelder-Mead (via Optimization -> Optim)
#     opt_result = nothing
#     sol = nothing
#     elapsed = @elapsed begin
#         sol = solve(
#             prob,
#             NelderMead();
#             x_abstol = 1e-8, f_abstol = 1e-8, g_tol = 1e-6, iterations = 500
#         )
#         opt_result = sol.original
#     end

#     # Extract results
#     est = try
#         sol.u[1]
#     catch
#         NaN
#     end
#     abs_err = abs(est - true_value)

#     # 5) Reporting
#     Convergence = (sol.retcode == ReturnCode.Success)
#     Value = true_value
#     Estimate = isnan(est) ? "FAILED" : round(est, digits = 3)
#     Error = isnan(abs_err) ? "N/A" : abs_err
#     Evaluations = opt_result.f_calls
#     true_reasons = [k for (k, v) in pairs(opt_result.stopped_by) if v]
#     StopReason = isempty(true_reasons) ? "none" : join(string.(true_reasons), ", ")
#     Time = round(elapsed, digits = 1)
#     Iterations = opt_result.iterations

#     if Convergence
#         @info "Parameter $(param) estimation completed" Convergence WarmStart Value Estimate Error Time StopReason Iterations Evaluations
#     else
#         @error "Parameter $(param) estimation failed" Convergence WarmStart Value Estimate Error Time StopReason Iterations Evaluations
#     end

#     # Save the results (thread-safe update)
#     lock(results_lock) do
#         estimation_results_2_warm[param] = Dict(
#             "Convergence" => Convergence,
#             "WarmStart" => WarmStart,
#             "Value" => Value,
#             "Estimate" => Estimate,
#             "Error" => Error,
#             "Time" => Time,
#             "StopReason" => StopReason,
#             "Iterations" => Iterations,
#             "Evaluations" => Evaluations
#     )
#     end
# end


# estimation_results_table(estimation_results_2_cold, title="Single-parameter estimation (cold starts) random perturbation")
# estimation_results_table(estimation_results_2_warm, title="Single-parameter estimation (warm starts) random perturbation")
