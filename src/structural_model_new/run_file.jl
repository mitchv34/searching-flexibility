using Pkg
# activate project containing Project.toml (adjust path if needed)
Pkg.activate(joinpath(@__DIR__, ".."))
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

# Solve the model
@time solve_model(prim, res, verbose=false, λ_S = 0.01, λ_u = 0.01, tol = 1e-12, max_iter = 25000)
@time solve_model(new_prim, new_res, verbose=false, λ_S = 0.01, λ_u = 0.01, tol = 1e-12, max_iter = 25000)

# Create moments 
baseline_moments = compute_model_moments(prim, res);
new_moments = compute_model_moments(new_prim, new_res);

begin
# Compare baseline vs new moments with PrettyTables
# Works for Dict, NamedTuple, or simple structs
moment_keys(m) = m isa AbstractDict ? collect(keys(m)) :
                 m isa NamedTuple    ? collect(propertynames(m)) :
                                        collect(fieldnames(typeof(m)))

moment_get(m, k::Symbol) = m isa AbstractDict ? get(m, k, missing) :
                           m isa NamedTuple    ? (k in propertynames(m) ? getproperty(m, k) : missing) :
                                                 (k in fieldnames(typeof(m)) ? getfield(m, k) : missing)

all_keys = unique(Symbol.(vcat(moment_keys(baseline_moments), moment_keys(new_moments))))
sort!(all_keys, by=string)

rows = Matrix{Any}(undef, length(all_keys), 4)
for (i, k) in enumerate(all_keys)
    b = moment_get(baseline_moments, k)
    n = moment_get(new_moments, k)
    d = (b isa Number && n isa Number) ? n - b : missing
    rows[i, 1] = String(k)
    rows[i, 2] = b
    rows[i, 3] = n
    rows[i, 4] = d
end

pretty_table(
    rows;
    header = ["Moment", "Baseline", "New", "New - Baseline"],
    title = "Model moments comparison"
)
end


begin
# Define parameter grids
chi_grid = collect(range(1.01, 3.0; length=100))
c0_grid = collect(range(0.01, 0.5; length=100))

# Worker function for χ sweep
@everywhere function sweep_chi(grid, prim_base, res_base)
    moments_list = Vector{Any}(undef, length(grid))
    for i in eachindex(grid)
        chi = grid[i]
        prim_i, res_i = update_primitives_results(prim_base, res_base, Dict(:χ => chi))
        solve_model(prim_i, res_i, verbose=false, λ_S = 0.01, λ_u = 0.01)
        moments_list[i] = compute_model_moments(prim_i, res_i)
    end
    return moments_list
end

# Worker function for c₀ sweep
@everywhere function sweep_c0(grid, prim_base, res_base)
    moments_list = Vector{Any}(undef, length(grid))
    for i in eachindex(grid)
        c0 = grid[i]
        prim_i, res_i = update_primitives_results(prim_base, res_base, Dict(:c₀ => c0))
        solve_model(prim_i, res_i, verbose=false, λ_S = 0.01, λ_u = 0.01)
        moments_list[i] = compute_model_moments(prim_i, res_i)
    end
    return moments_list
end

begin
    println("Starting distributed parameter sweeps...")

    # Run sweeps on different workers in parallel
    future_chi = remotecall(sweep_chi, 2, chi_grid, prim, res)  # Worker 2
    future_c0 = remotecall(sweep_c0, 3, c0_grid, prim, res)    # Worker 3

    # Fetch results
    println("Computing χ sweep on worker 2...")
    chi_moments_list = fetch(future_chi)
    println("Computing c₀ sweep on worker 3...")
    c0_moments_list = fetch(future_c0)

    # Extract numeric moment keys from both sweeps
    function get_numeric_keys(moments_list)
        all_keys = Symbol[]
        for m in moments_list
            append!(all_keys, moment_keys(m))
        end
        all_keys = unique(all_keys)
        sort!(all_keys, by=string)
        
        numeric_keys = Symbol[]
        for k in all_keys
            if any(i -> moment_get(moments_list[i], k) isa Number, eachindex(moments_list))
                push!(numeric_keys, k)
            end
        end
        return numeric_keys
    end

    chi_numeric_keys = get_numeric_keys(chi_moments_list)
    c0_numeric_keys = get_numeric_keys(c0_moments_list)

    # Plot χ sweep results
    println("Plotting χ sweep results...")
    nplots_chi = length(chi_numeric_keys)
    ncol = 3
    nrow_chi = max(1, cld(nplots_chi, ncol))
    fig_chi =  Figure(size = (1200, 900))

    for (idx, k) in enumerate(chi_numeric_keys)
        row = ((idx - 1) ÷ ncol) + 1
        col = ((idx - 1) % ncol) + 1
        ax = Axis(fig_chi[row, col], title=String(k), xlabel="χ", ylabel="moment")

        ys = Vector{Float64}(undef, length(chi_grid))
        @inbounds for i in eachindex(chi_grid)
            v = moment_get(chi_moments_list[i], k)
            ys[i] = v isa Number ? float(v) : NaN
        end

        lines!(ax, chi_grid, ys, color=:steelblue)
        vlines!(ax, [prim.χ], color=:red, linestyle=:dash, linewidth=2)
    end
    fig_chi[nrow_chi + 1, 1:ncol] = Label(fig_chi, "Model moments vs χ (distributed)", fontsize=18)
    display(fig_chi)

    # Plot c₀ sweep results
    println("Plotting c₀ sweep results...")
    nplots_c0 = length(c0_numeric_keys)
    nrow_c0 = max(1, cld(nplots_c0, ncol))
    fig_c0 =  Figure(size = (1200, 900))

    for (idx, k) in enumerate(c0_numeric_keys)
        row = ((idx - 1) ÷ ncol) + 1
        col = ((idx - 1) % ncol) + 1
        ax = Axis(fig_c0[row, col], title=String(k), xlabel="c₀", ylabel="moment")

        ys = Vector{Float64}(undef, length(c0_grid))
        @inbounds for i in eachindex(c0_grid)
            v = moment_get(c0_moments_list[i], k)
            ys[i] = v isa Number ? float(v) : NaN
        end

        lines!(ax, c0_grid, ys, color=:darkgreen)
        vlines!(ax, [prim.c₀], color=:red, linestyle=:dash, linewidth=2)
    end
    fig_c0[nrow_c0 + 1, 1:ncol] = Label(fig_c0, "Model moments vs c₀ (distributed)", fontsize=18)
    display(fig_c0)

    println("Parameter sweeps completed!")
    end
end


# Select what parameters to estimate
params_to_estimate = [:aₕ, :bₕ, :c₀, :χ, :A₁, :ν, :ψ₀, :ϕ, :κ₀]

# Select initial values for the parameters (#! For this test we select the same values they start with)
initial_values = [getfield(prim, k) for k in params_to_estimate]

# Create the estimation problem
p = (
    prim_base = prim,
    res_base = res,
    target_moments = baseline_moments,
    param_names = params_to_estimate,
    weighting_matrix = nothing,
    matrix_moment_order = nothing
);

# Settings for the grid around the true value
n_grid = 41
rel_width = 0.2

# Create grids
θ0 = collect(float.(initial_values))
grids = [range(param * (1 - rel_width), param * (1 + rel_width), n_grid) for param in θ0]

# Ship immutable inputs to workers once
@everywhere const P_OBJ = $p
@everywhere const THETA0 = $θ0

# Worker-side evaluation for one parameter's grid
@everywhere function eval_param_grid(i::Int, grid)
    fvals = Vector{Float64}(undef, length(grid))
    θ = copy(THETA0)
    for j in eachindex(grid)
        θ[i] = grid[j]
        fvals[j] =  objective_function(θ, P_OBJ)
    end
    return fvals
end


# Run each parameter's profile on a different core
n_params = length(params_to_estimate)
fvals_shared = SharedArray{Float64}(n_params, n_grid)
fill!(fvals_shared, NaN)

results = pmap(i -> eval_param_grid(i, grids[i]), 1:n_params)
for i in 1:n_params
    fvals_shared[i, :] = results[i]
end

# Plot on master only
fig = Figure(size = (1200, 900))
for i in eachindex(params_to_estimate)
    row = ((i - 1) ÷ 3) + 1
    col = ((i - 1) % 3) + 1
    ax = Axis(fig[row, col],
        title = string(params_to_estimate[i]),
        xlabel = "parameter value",
        ylabel = "objective value"
    )

    grid = grids[i]
    fvals = collect(view(fvals_shared, i, :))

    lines!(ax, grid, fvals, color = :steelblue)
    vlines!(ax, [θ0[i]], color = :red, linestyle = :dash, linewidth = 2)

    finite_idx = findall(isfinite, fvals)
    if !isempty(finite_idx)
        jmin = finite_idx[argmin(@view fvals[finite_idx])]
        scatter!(ax, [grid[jmin]], [fvals[jmin]], color = :orange, markersize = 8)
    end
end

fig[4, 1:3] = Label(fig, "Objective function profiles", fontsize = 18)
display(fig)




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


# Select what parameters to estimate
problematic_params = [:c₀, :χ, :A₁]

# Select initial values for the parameters (#! For this test we select the same values they start with)
initial_values = [getfield(prim, k) for k in problematic_params]

# Create the estimation problem
p = (
    prim_base = prim,
    res_base = res,
    target_moments = baseline_moments,
    param_names = params_to_estimate,
    weighting_matrix = nothing,
    matrix_moment_order = nothing
);

# Settings for the grid around the true value
n_grid = 41
rel_width = 0.05

# Create grids (distinct name to avoid colliding with the earlier `grids`)
θ0_problem = collect(float.(initial_values))
grids_problem = [range(param * (1 - rel_width), param * (1 + rel_width), n_grid) for param in θ0_problem]

# Ship immutable inputs to workers once
@everywhere const P_OBJ_new = $p
@everywhere const THETA0_new = $θ0_problem

# Define a distinct worker function that uses the problematic grids
@everywhere function eval_param_grid_problem(i::Int, grid)
    fvals = Vector{Float64}(undef, length(grid))
    θ = copy(THETA0_new)
    for j in eachindex(grid)
        θ[i] = grid[j]
        fvals[j] = objective_function(θ, P_OBJ_new)
    end
    return fvals
end

# Run each parameter's profile on a different core (use unique names)
n_params_problem = length(problematic_params)
fvals_shared_problem = SharedArray{Float64}(n_params_problem, n_grid)
fill!(fvals_shared_problem, NaN)

results_problem = pmap(i -> eval_param_grid_problem(i, grids_problem[i]), 1:n_params_problem)
for i in 1:n_params_problem
    fvals_shared_problem[i, :] = results_problem[i]
end

# Plot on master only
fig = Figure(size = (1200, 900))
for i in eachindex(params_to_estimate)
    row = ((i - 1) ÷ 3) + 1
    col = ((i - 1) % 3) + 1
    ax = Axis(fig[row, col],
        title = string(params_to_estimate[i]),
        xlabel = "parameter value",
        ylabel = "objective value"
    )

    grid = grids[i]
    fvals = collect(view(fvals_shared, i, :))

    lines!(ax, grid, fvals, color = :steelblue)
    vlines!(ax, [θ0[i]], color = :red, linestyle = :dash, linewidth = 2)

    finite_idx = findall(isfinite, fvals)
    if !isempty(finite_idx)
        jmin = finite_idx[argmin(@view fvals[finite_idx])]
        scatter!(ax, [grid[jmin]], [fvals[jmin]], color = :orange, markersize = 8)
    end
end

fig[4, 1:3] = Label(fig, "Objective function profiles", fontsize = 18)
display(fig)




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