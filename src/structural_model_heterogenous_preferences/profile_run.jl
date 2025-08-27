# profile_run.jl — TimerOutputs-based profiling for heterogeneous preferences model
using Pkg

# Ensure relative includes work regardless of cwd
const _HERE_ = @__DIR__
const _ROOT_ = joinpath(_HERE_, "..", "..")

# Activate the root project environment
Pkg.activate(_ROOT_)
Pkg.instantiate()

using TimerOutputs
using Printf
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
        # Reuse logic from the new-model profiling script; delegate to existing function
        return update_primitives_results(prim, res, params_to_update)
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

		@timeit timer "2. solve_model" solve_model(
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

        model_moments = @timeit timer "3. compute_moments" compute_model_moments(
            prim_new, res_new; 
            include = [:mean_logwage, :var_logwage, :diff_logwage_inperson_remote, 
                      :remote_share, :hybrid_share, :wage_premium_high_psi, 
                      :diff_alpha_high_lowpsi, :market_tightness]
        )
		loss = @timeit timer "4. compute_distance" compute_distance(
			model_moments,
			p.target_moments,
			get(p, :weighting_matrix, nothing),
			get(p, :matrix_moment_order, nothing)
		)
		return loss
	end
end

# ============================================================================
# OPTIMIZED IN-PLACE SOLVER FUNCTIONS FOR PROFILING
# ============================================================================

"""
Calculates B = vec(sum((1-ξ) * max(0,S) .* u_dist, dims=1)) in-place.
"""
function calculate_B!(B, S, u_dist, ξ)
    # This is a bit tricky to do fully in-place with broadcasting,
    # but a simple loop is allocation-free and just as fast.
    n_h, n_ψ = size(S)
    fill!(B, 0.0)
    for j in 1:n_ψ
        for i in 1:n_h
            B[j] += (1.0 - ξ) * max(0.0, S[i, j]) * u_dist[i]
        end
    end
end

"""
Calculates Γ = (v .* f_ψ) ./ V in-place.
"""
function calculate_Gamma!(Γ, v, f_ψ)
    V = dot(v, f_ψ) # dot product is non-allocating
    if V > 0
        @. Γ = (v * f_ψ) / V
    else
        fill!(Γ, 0.0)
    end
    return V
end

"""
In-place matrix-vector multiplication for positive surplus calculation.
Computes result = max(0, A) * x where we only multiply positive elements.
"""
function mul_positive_surplus!(result, A, x)
    fill!(result, 0.0)
    n_rows, n_cols = size(A)
    for i in 1:n_rows
        for j in 1:n_cols
            val = max(0.0, A[i, j])
            if val > 0.0
                result[i] += val * x[j]
            end
        end
    end
end

"""
In-place matrix-vector multiplication for boolean matrices.
Computes result = A * x where A is a boolean matrix.
"""
function mul_boolean!(result, A_bool, x)
    fill!(result, 0.0)
    n_rows, n_cols = size(A_bool)
    for i in 1:n_rows
        for j in 1:n_cols
            if A_bool[i, j]
                result[i] += x[j]
            end
        end
    end
end

"""
    solve_model_timed(prim, res, timer; kwargs...)

**PROFILING WRAPPER v2.0 (Updated August 2025)**

TimerOutputs wrapper for the official solve_model() function.
Updated on 2025-08-19 to use the new optimized solver.

This function provides detailed timing breakdown of the solver components:
- Flow surplus calculation
- Main iteration loop  
- Convergence checking
- Final outcome computation

Note: Uses the official optimized solver (v2.0) internally.
For legacy solver profiling, use solve_model_legacy() manually.
"""
function solve_model_timed(
                        prim::Primitives{T},
                        res::Results{T},
                        timer::TimerOutput; # Pass the timer in
                        initial_S::Union{Matrix{T}, Nothing}=nothing,
                        tol::Float64=1e-8,
                        max_iter::Int=5000,
                        verbose::Bool=true,
                        print_freq::Int=50,
                        λ_S_init::Float64 = 0.01,
                        λ_u_init::Float64 = 0.01
                    ) where {T<:Real}

    @timeit timer "solve_model (Total)" begin
        @unpack h_grid, ψ_grid, β, δ, ξ, κ₀, κ₁, n_h, n_ψ, γ₀, γ₁ = prim
        f_h = prim.h_pdf
        f_ψ = prim.ψ_pdf

        s_flow = @timeit timer "s_flow" calculate_logit_flow_surplus_with_curvature(prim)

        # --- MAJOR CHANGE: Pre-allocation of all temporary arrays ---
        @timeit timer "pre-allocation" begin
            S_final = isnothing(initial_S) ? copy(s_flow) : copy(initial_S)
            u_final = copy(prim.h_pdf)
            
            # Pre-allocate arrays that were previously created inside the loop
            S_old = similar(S_final)
            u_old = similar(u_final)
            u_dist = similar(u_final)
            
            B = Vector{T}(undef, n_ψ)
            v = Vector{T}(undef, n_ψ)
            Γ = Vector{T}(undef, n_ψ)
            
            ExpectedSearch = Vector{T}(undef, n_h)
            S_update = similar(S_final)
            
            ProbAccept = Vector{T}(undef, n_h)
            unemp_rate = Vector{T}(undef, n_h)
            u_update = similar(u_final)
            
            # Pre-allocate boolean matrix for positive surplus
            S_positive = similar(S_final, Bool)
        end

        denom = 1.0 - β * (1.0 - δ)
        ΔS = Inf
        λ_S = λ_S_init
        λ_u = λ_u_init

        @timeit timer "main_loop" for k in 1:max_iter
            @timeit timer "copy!" begin
                copy!(S_old, S_final)
                copy!(u_old, u_final)
            end

            # --- Part 1: Update aggregates and surplus (IN-PLACE) ---
            @timeit timer "part1_aggregates" begin
                L = sum(u_old)
                @. u_dist = u_old / L

                # B = vec(sum((1.0 - ξ) .* max.(0.0, S_old) .* u_dist, dims=1))
                calculate_B!(B, S_old, u_dist, ξ) # In-place helper
                
                B_integral = sum(@. max(0.0, B)^(1/κ₁) * f_ψ)
                θ = ( (1/L) * (γ₀/κ₀)^(1/κ₁) * B_integral )^(1 / (1 + γ₁/κ₁))
                
                p = γ₀ * θ^(1 - γ₁)
                q = γ₀ * θ^(-γ₁)

                # v = ((q .* B) ./ κ₀).^(1/κ₁)  # Original calculation
                @. v = ((q * B) / κ₀)^(1/κ₁)
                
                # Γ = V > 0 ? (v .* f_ψ) ./ V : zeros(T, n_ψ)
                V = calculate_Gamma!(Γ, v, f_ψ) # In-place helper

                # ExpectedSearch = max.(0.0, S_old) * Γ
                mul_positive_surplus!(ExpectedSearch, S_old, Γ) # In-place matrix-vector multiply
                
                # S_update = (s_flow .- (β * p * ξ .* ExpectedSearch)) ./ denom
                @. S_update = (s_flow - (β * p * ξ * ExpectedSearch)) / denom
                
                # S_final .= (1.0 - λ_S) .* S_old .+ λ_S .* S_update
                @. S_final = (1.0 - λ_S) * S_old + λ_S * S_update
            end
            
            # --- Part 2: Update unemployment (IN-PLACE) ---
            @timeit timer "part2_unemployment" begin
                # Create boolean matrix for current surplus
                @. S_positive = S_final > 0.0
                
                # ProbAccept = (S_final .> 0.0) * Γ
                mul_boolean!(ProbAccept, S_positive, Γ) # In-place matrix-vector multiply

                # unemp_rate = δ ./ (δ .+ p .* ProbAccept)
                @. unemp_rate = δ / (δ + p * ProbAccept)
                
                # u_update = unemp_rate .* f_h
                @. u_update = unemp_rate * f_h
                
                # u_final .= (1.0 - λ_u) .* u_old .+ λ_u .* u_update
                @. u_final = (1.0 - λ_u) * u_old + λ_u * u_update
            end

            # --- Part 3: Check for Convergence ---
            @timeit timer "part3_convergence" begin
                # More optimized convergence check to avoid temporary allocations
                ΔS = 0.0
                for i in eachindex(S_final)
                    ΔS = max(ΔS, abs(S_final[i] - S_old[i]))
                end

                if verbose && (k % print_freq == 0 || k == 1)
                    @printf("Iter %4d: ΔS = %.12e\n", k, ΔS)
                end
                if ΔS < tol
                    if verbose
                        println("Converged after $k iterations.")
                    end
                    break
                end
            end
        end

        @timeit timer "final_updates" begin
            res.u .= u_final
            res.S .= S_final
            compute_final_outcomes!(prim, res)
        end
        
        # The function needs to return the final lambdas for the profiling script
        return λ_S, λ_u 
    end # end @timeit
end

# Updated objective function to use the timed solver
function objective_function_timed_optimized(params, p, timer::TimerOutput)
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

		λ_S_final, λ_u_final = @timeit timer "2. solve_model" solve_model_timed(
			prim_new, res_new, timer; # Pass the timer here
			initial_S = initial_S,
			verbose   = false,
			λ_S_init  = λ_S_start,
			λ_u_init  = λ_u_start,
			tol       = get(p, :tol, 1e-7),
			max_iter  = get(p, :max_iter, 25_000)
		)

		# Update caches with new lambda values
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

begin
# Minimal profiling run
println("Initializing heterogeneous model for profiling...")
config = joinpath(_HERE_, "model_parameters.yaml")
prim_true, res_true = initializeModel(config)

# Warm solve
println("Running warm solve (small iterations) to JIT compile...")
solve_model(prim_true, res_true; verbose=false, max_iter=10)

# Build a tiny problem container
params_to_estimate = [:ν]
true_val = prim_true.ν
p_warm = (
    prim_base = prim_true,
    res_base = res_true,
    target_moments = compute_model_moments(
        prim_true, res_true; 
        include = [:mean_logwage, :var_logwage, :diff_logwage_inperson_remote, 
                  :remote_share, :hybrid_share, :wage_premium_high_psi, 
                  :diff_alpha_high_lowpsi, :market_tightness]
    ),
    param_names = params_to_estimate,
    last_res = Ref(res_true),
    solver_state = Ref((λ_S_init = 0.01, λ_u_init = 0.01)),
    tol = 1e-7,
    max_iter = 25_000,
)

println("Running objective once to compile...")
objective_function([true_val * 1.05], p_warm)

println("\n" * "="^80)
println("ORIGINAL SOLVER PROFILING")
println("="^80)
TO_original = TimerOutput()
println("Running profiled objective (original solver)...")
loss_original = objective_function_timed([true_val * 1.05], p_warm, TO_original)
println("Loss = ", loss_original)
print_timer(TO_original; allocations=true, sortby=:time)

println("\n" * "="^80)
println("OPTIMIZED SOLVER PROFILING")
println("="^80)
TO_optimized = TimerOutput()
println("Running profiled objective (optimized solver)...")
loss_optimized = objective_function_timed_optimized([true_val * 1.05], p_warm, TO_optimized)
println("Loss = ", loss_optimized)
print_timer(TO_optimized; allocations=true, sortby=:time)

println("\n" * "="^80)
println("PERFORMANCE COMPARISON")
println("="^80)
println("Original loss: ", loss_original)
println("Optimized loss: ", loss_optimized)
println("Loss difference: ", abs(loss_original - loss_optimized))
println("Should be very close to zero (numerical precision difference only)")

println("\nProfile run completed.")
end 