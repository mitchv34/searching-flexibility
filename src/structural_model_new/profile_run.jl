# profile_run.jl — TimerOutputs-based profiling for objective_function

using Pkg
# activate project containing Project.toml (adjust path if needed)
Pkg.activate("../../..")
Pkg.instantiate()

using TimerOutputs
using ProfileView


# Ensure relative includes work regardless of cwd
const _HERE_ = @__DIR__
include(joinpath(_HERE_, "helpers.jl"))
include(joinpath(_HERE_, "ModelSetup.jl"))
include(joinpath(_HERE_, "ModelSolver.jl"))
include(joinpath(_HERE_, "ModelEstimation.jl"))

function update_primitives_results_timed(
                                            prim::Primitives,
                                                res::Results,
                                                timer,
                                                params_to_update::Dict=Dict();
                                                kwargs...,
                                            )
    return TimerOutputs.@timeit timer "update_primitives (Total)" begin
        
        # --- Stage 1: Read current values or overrides (type-stable locals) ---
        # Scalars
        A₀   = get(params_to_update, :A₀, prim.A₀)
        A₁   = get(params_to_update, :A₁, prim.A₁)
        ψ₀   = get(params_to_update, :ψ₀, prim.ψ₀)
        ϕ    = get(params_to_update, :ϕ,  prim.ϕ)
        ν    = get(params_to_update, :ν,  prim.ν)
        c₀   = get(params_to_update, :c₀, prim.c₀)
        χ    = get(params_to_update, :χ,  prim.χ)
        γ₀   = get(params_to_update, :γ₀, prim.γ₀)
        γ₁   = get(params_to_update, :γ₁, prim.γ₁)

        κ₀   = get(params_to_update, :κ₀, prim.κ₀)
        κ₁   = get(params_to_update, :κ₁, prim.κ₁)
        βv   = get(params_to_update, :β,  prim.β)
        δv   = get(params_to_update, :δ,  prim.δ)
        b    = get(params_to_update, :b,  prim.b)
        ξ    = get(params_to_update, :ξ,  prim.ξ)

        # Grids and bounds
        n_ψ  = prim.n_ψ
        ψ_min = get(params_to_update, :ψ_min, prim.ψ_min)
        ψ_max = get(params_to_update, :ψ_max, prim.ψ_max)
        ψ_grid = get(params_to_update, :ψ_grid, prim.ψ_grid)
        ψ_pdf  = get(params_to_update, :ψ_pdf,  prim.ψ_pdf)
        ψ_cdf  = get(params_to_update, :ψ_cdf,  prim.ψ_cdf)

        aₕ   = get(params_to_update, :aₕ, prim.aₕ)
        bₕ   = get(params_to_update, :bₕ, prim.bₕ)
        n_h  = prim.n_h
        h_min = get(params_to_update, :h_min, prim.h_min)
        h_max = get(params_to_update, :h_max, prim.h_max)
        h_grid = get(params_to_update, :h_grid, prim.h_grid)
        h_pdf  = get(params_to_update, :h_pdf,  prim.h_pdf)
        h_cdf  = get(params_to_update, :h_cdf,  prim.h_cdf)

        # --- Stage 2: Promote numeric type if needed (match ModelEstimation logic) ---
        T_old = eltype(prim.h_grid)
        T_new = promote_type(typeof.(values(params_to_update))..., T_old)
        if T_new != T_old
            # Promote scalars
            A₀ = T_new(A₀); A₁ = T_new(A₁); ψ₀ = T_new(ψ₀); ϕ = T_new(ϕ); ν = T_new(ν)
            c₀ = T_new(c₀); χ = T_new(χ); γ₀ = T_new(γ₀); γ₁ = T_new(γ₁)
            κ₀ = T_new(κ₀); κ₁ = T_new(κ₁); βv = T_new(βv); δv = T_new(δv)
            b  = T_new(b);   ξ  = T_new(ξ)

            ψ_min = T_new(ψ_min); ψ_max = T_new(ψ_max)
            aₕ = T_new(aₕ); bₕ = T_new(bₕ)
            h_min = T_new(h_min); h_max = T_new(h_max)

            # Promote vectors
            ψ_grid = T_new.(ψ_grid); ψ_pdf = T_new.(ψ_pdf); ψ_cdf = T_new.(ψ_cdf)
            h_grid = T_new.(h_grid); h_pdf = T_new.(h_pdf); h_cdf = T_new.(h_cdf)
        end

        # --- Stage 3: Recompute h distribution only if relevant params changed ---
        recompute_h = any(k -> k in (:aₕ, :bₕ, :h_grid, :h_min, :h_max), keys(params_to_update))
        
        if recompute_h
            TimerOutputs.@timeit timer "b. Recompute h-dist" begin
                h_scaled = (h_grid .- h_min) ./ (h_max - h_min)
                beta_dist = Distributions.Beta(aₕ, bₕ)
                h_pdf_raw = pdf.(beta_dist, h_scaled)
                h_pdf = h_pdf_raw ./ sum(h_pdf_raw)
                h_cdf = cumsum(h_pdf)
                h_cdf ./= h_cdf[end]
            end
        end

        # --- Stage 4: Build validated Primitives and fresh Results ---
        new_prim = TimerOutputs.@timeit timer "c. validated_Primitives" validated_Primitives(
            A₀=A₀, A₁=A₁, ψ₀=ψ₀, ϕ=ϕ, ν=ν, c₀=c₀, χ=χ, γ₀=γ₀, γ₁=γ₁,
            κ₀=κ₀, κ₁=κ₁, β=βv, δ=δv, b=b, ξ=ξ,
            n_ψ=n_ψ, ψ_min=ψ_min, ψ_max=ψ_max, ψ_grid=ψ_grid, ψ_pdf=ψ_pdf, ψ_cdf=ψ_cdf,
            aₕ=aₕ, bₕ=bₕ, n_h=n_h, h_min=h_min, h_max=h_max, h_grid=h_grid, h_pdf=h_pdf, h_cdf=h_cdf,
        )
        new_res  = TimerOutputs.@timeit timer "d. Results Constructor" Results(new_prim)
        return new_prim, new_res
    end
end


# A timed wrapper around objective_function using TimerOutputs
function objective_function_timed(params, p, timer::TimerOutput)
	return @timeit timer "Objective Function (Total)" begin
		params_to_update = Dict(p.param_names .=> params)

		# Warm vs cold logic mirrors objective_function in ModelEstimation.jl
		initial_S = haskey(p, :last_res) ? p.last_res[].S : nothing

		# Read current solver state (with safe fallbacks)
		λ_S_start = try
			p.solver_state[].λ_S_init
		catch
			get(p, :λ_S_init, 0.01)
		end
		λ_u_start = try
			p.solver_state[].λ_u_init
		catch
			get(p, :λ_u_init, 0.01)
		end

		prim_new, res_new = @timeit timer "1. update_primitives" update_primitives_results_timed(
            p.prim_base, deepcopy(p.res_base), timer; params_to_update=params_to_update
		)

		λ_S_final, λ_u_final = @timeit timer "2. solve_model" solve_model(
			prim_new, res_new;
			initial_S = initial_S,
			verbose   = false,
			λ_S_init  = λ_S_start,
			λ_u_init  = λ_u_start,
			tol       = get(p, :tol, 1e-7),
			max_iter  = get(p, :max_iter, 25_000)
		)

		# Update caches
		if haskey(p, :last_res)
			p.last_res[] = res_new
		end
		if haskey(p, :solver_state)
			p.solver_state[] = (λ_S_init = λ_S_final, λ_u_init = λ_u_final)
		end

		model_moments = @timeit timer "3. compute_moments" compute_model_moments(prim_new, res_new)
		loss = @timeit timer "4. compute_distance" compute_distance(
			model_moments,
			p.target_moments,
			get(p, :weighting_matrix, nothing),
			get(p, :matrix_moment_order, nothing)
		)
		return loss
	end
end


prim_true, res_true = initializeModel(config)
using BenchmarkTools

using Statistics

# Benchmark update_primitives_results_timed for two cases
prim_loc = prim_true
res_loc = res_true

println("\n--- Benchmark: change aₕ (should recompute h-dist) ---")
params_ah = Dict(:aₕ => prim_loc.aₕ * 1.2)

TO1 = TimerOutput()
println("Single timed invocation (shows internal TimerOutputs):")
update_primitives_results_timed(prim_loc, deepcopy(res_loc), TO1; params_to_update=params_ah);
print_timer(TO1; allocations=true, sortby=:time)

println("\nBenchmarkTools measurement (many runs):")
ah_stats = @benchmark update_primitives_results_timed($prim_loc, deepcopy($res_loc), TimerOutput(); params_to_update=$params_ah)
display(ah_stats)

println("\n--- Benchmark: change c₀ (cheap) ---")
params_c0 = Dict(:c₀ => prim_loc.c₀ * 1.05)

TO2 = TimerOutput()
println("Single timed invocation (shows internal TimerOutputs):")
update_primitives_results_timed(prim_loc, deepcopy(res_loc), TO2; params_to_update=params_c0)
print_timer(TO2; allocations=true, sortby=:time)

println("\nBenchmarkTools measurement (many runs):")
c0_stats = @benchmark update_primitives_results_timed($prim_loc, deepcopy($res_loc), TimerOutput(); params_to_update=$params_c0)
display(c0_stats)

println("\nMedian times (ns):")
println("aₕ median: ", median(ah_stats.times))
println("c₀ median: ", median(c0_stats.times))



# ---------- One-shot profiling run ----------
begin
    const TO = TimerOutput()

    println("Initializing model (warm baseline)...")
    config = joinpath(_HERE_, "model_parameters.yaml")
    prim_true, res_true = initializeModel(config)
    solve_model(prim_true, res_true; verbose=false, λ_S_init=0.01, λ_u_init=0.01)
    target_moments = compute_model_moments(prim_true, res_true)

    # Pick a parameter and a slightly perturbed value to evaluate
    params_to_estimate = [:ν]
    true_val = prim_true.ν
    test_params = [true_val * 1.1]

    # Warm-start parameter container (NamedTuple as in ModelEstimation)
    p_warm = (
        prim_base = prim_true,
        res_base = res_true,
        target_moments = target_moments,
        param_names = params_to_estimate,
        last_res = Ref(res_true),
        solver_state = Ref((λ_S_init = 0.01, λ_u_init = 0.01)),
        tol = 1e-7,
        max_iter = 25_000,
    )

    println("Running once for JIT compilation...")
    objective_function(test_params, p_warm)
    println("Compilation complete.")

    println("\nRunning profiled version...")
    loss = objective_function_timed(test_params, p_warm, TO)
    println("Loss = ", loss)

    println("\n" * repeat("=", 40))
    println("--- Profiling Results ---")
    println(repeat("=", 40))
    print_timer(TO; allocations=true, sortby=:time)
end

# using ProfileView

# Include your UNMODIFIED model code

# --- Initialize the Model ---
println("Initializing model...") 
config = joinpath(_HERE_, "model_parameters.yaml")
prim_true, res_true = initializeModel(config)

# --- JIT Compilation Run ---
# Run the function once with a small workload to compile it
println("Compiling solve_model...")
solve_model(prim_true, res_true, verbose=false, max_iter=10)
println("Compilation complete.")

# --- The Profiled Run ---
# Now, simply "wrap" the call to your original, unmodified function with @profview
println("\nProfiling solve_model...")
# ProfileView.@profview solve_model(prim_true, res_true, verbose=false, max_iter=5000)
@profview solve_model(prim_true, res_true, verbose=false, max_iter=5000)

println("Profiling finished.")


# Benchmark objective evaluation (untimed and TimerOutputs-wrapped)
using BenchmarkTools, Statistics

# Prepare parameters and problem container for benchmarking
bench_param_names = [:ν]
bench_test_params = [prim_true.ν * 1.1]

bench_problem = (
    prim_base = prim_true,
    res_base = res_true,
    target_moments = compute_model_moments(prim_true, res_true),
    param_names = bench_param_names,
    last_res = Ref(res_true),
    solver_state = Ref((λ_S_init = 0.01, λ_u_init = 0.01)),
    tol = 1e-7,
    max_iter = 25_000,
)

# Warm run to ensure compilation
println("Warming objective function for JIT...")
objective_function(bench_test_params, bench_problem)

# Benchmark the plain objective function
println("\n--- Benchmark: objective_function (plain) ---")
plain_stats = @benchmark objective_function($bench_test_params, $bench_problem)
display(plain_stats)
println("Median (ns): ", median(plain_stats.times))

# Benchmark the TimerOutputs-wrapped objective
println("\n--- Benchmark: objective_function_timed (with TimerOutput) ---")
timed_stats = @benchmark objective_function_timed($bench_test_params, $bench_problem, TimerOutput())
display(timed_stats)
println("Median (ns): ", median(timed_stats.times))