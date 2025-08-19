# profile_run.jl — TimerOutputs-based profiling for heterogeneous preferences model

using Pkg
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
    target_moments = compute_model_moments(prim_true, res_true),
    param_names = params_to_estimate,
    last_res = Ref(res_true),
    solver_state = Ref((λ_S_init = 0.01, λ_u_init = 0.01)),
    tol = 1e-7,
    max_iter = 25_000,
)

println("Running objective once to compile...")
objective_function([true_val * 1.05], p_warm)

TO = TimerOutput()
println("Running profiled objective...")
loss = objective_function_timed([true_val * 1.05], p_warm, TO)
println("Loss = ", loss)
print_timer(TO; allocations=true, sortby=:time)

println("Profile run completed.")
