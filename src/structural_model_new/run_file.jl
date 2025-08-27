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
    include(joinpath($ROOT, "ModelSetup.jl"))
    include(joinpath($ROOT, "ModelSolver.jl"))
    include(joinpath($ROOT, "ModelEstimation.jl"))
end
using Random, Statistics
using Term
using Printf
using CairoMakie
using PrettyTables

include(joinpath(ROOT, "ModelPlotting.jl"))

# --- REPL runner ---
# Point to the configuration file
config = "src/structural_model_new/model_parameters.yaml"
# Initialize the model
prim, res = initializeModel(config);
@show prim.c₀
new_prim, new_res = update_primitives_results(prim, res, Dict(:c₀ => prim.c₀ * 2));
@show new_prim.c₀



# --- 1. Central Setup (Done on the main processor) ---

# Load the model and solve for the "true" state
prim_true, res_true = initializeModel(config);
solve_model(prim_true, res_true, verbose=false)
target_moments = compute_model_moments(prim_true, res_true);

true_values = Dict(k => getfield(prim_true, k) for k in params_to_estimate)

# --- 2. Define the 1D Objective Function ---
# This function will be sent to all worker processes.
@everywhere function objective_1D(x, p)
    # x[1] is the single parameter value being tested by the optimizer
    # p contains all other fixed data
    
    # Create a dictionary of all parameters, starting with the true values
    # allow Dual numbers to be stored
    params_for_model = Dict{Symbol,Any}(p.true_values)
    params_for_model[p.param_to_estimate] = x[1]

    # Create a fresh, non-mutated copy of the primitives
    prim_new = deepcopy(p.prim_base)
    update_primitives!(prim_new; params_for_model...) # Update with the mixed dict
    
    # Solve the model with the new parameters
    res_new = Results(prim_new)
    solve_model(prim_new, res_new, verbose=false)

    # Compute moments and the final distance
    model_moments = compute_model_moments(prim_new, res_new)
    return compute_distance(model_moments, p.target_moments)
end



# --- 1. Central Setup (Done on the main processor) ---

# Load the model and solve for the "true" state
prim_true, res_true = initializeModel(config);
solve_model(prim_true, res_true, verbose=false)
target_moments = compute_model_moments(prim_true, res_true);

true_values = Dict(k => getfield(prim_true, k) for k in params_to_estimate)

# --- 2. Define the 1D Objective Function ---
# This function will be sent to all worker processes.
@everywhere function objective_1D(x, p)
    # x[1] is the single parameter value being tested by the optimizer
    # p contains all other fixed data
    
    # Create a dictionary of all parameters, starting with the true values
    # allow Dual numbers to be stored
    params_for_model = Dict{Symbol,Any}(p.true_values)
    params_for_model[p.param_to_estimate] = x[1]

    # Create a fresh, non-mutated copy of the primitives
    prim_new = deepcopy(p.prim_base)
    update_primitives!(prim_new; params_for_model...) # Update with the mixed dict
    
    # Solve the model with the new parameters
    res_new = Results(prim_new)
    solve_model(prim_new, res_new, verbose=false)

    # Compute moments and the final distance
    model_moments = compute_model_moments(prim_new, res_new)
    return compute_distance(model_moments, p.target_moments)
end


# --- 3. Run the Distributed Loop ---

println("Starting parallel 1D estimations for $(length(params_to_estimate)) parameters...")

# Use @distributed to run the for loop in parallel
# The `vcat` reducer will collect the results from each worker into a single array
results = @distributed (vcat) for param_name in params_to_estimate
    
    # --- This code block runs on a separate worker for each parameter ---
    
    # Get the true value and a starting guess for this specific parameter
    true_val = true_values[param_name]
    initial_guess = [true_val * 0.9] # Start 10% away

    # Set reasonable bounds (e.g., +/- 50% of the true value) to help the optimizer
    lower_bound = [true_val * 0.5]
    upper_bound = [true_val * 1.5]

    # The 'p' container now also includes the name of the parameter for this job
    p = (
        prim_base = prim_true,
        res_base = res_true,
        target_moments = target_moments,
        param_to_estimate = param_name,
        true_values = true_values # Pass all true values for the other fixed params
    )

    # Define the optimization function for this 1D problem
    opt_func = OptimizationFunction(objective_1D, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(opt_func, initial_guess, p, lb=lower_bound, ub=upper_bound)

    # Solve the 1D problem
    solution = solve(prob, BFGS())

    # Return a tuple of results for this parameter
    (param=param_name, true_val=true_val, estimated=solution.u[1], objective=solution.objective)
end


# --- 4. Display Results (Back on the main processor) ---

println("\n--- Parameter Recovery Results (1D Estimations) ---")
for res in results
    @printf "  -> %-5s | True: %-10.4f | Estimated: %-10.4f | Final Obj: %.2e\n" res.param res.true_val res.estimated res.objective
end