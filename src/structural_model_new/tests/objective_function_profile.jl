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
# Initialize the model
prim, res = initializeModel(config)
# Compute a set of moments from the true params
baseline_moments = compute_model_moments(prim, res)

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
    matrix_moment_order = nothing,
    solver_state_init = (λ_S_init = 0.1, λ_u_init = 0.1),
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



# Select what parameters to estimate
problematic_params = [:c₀, :χ, :A₁]

# Select initial values for the parameters (#! For this test we select the same values they start with)
initial_values = [getfield(prim, k) for k in problematic_params]

# Create the estimation problem
p = (
    prim_base = prim,
    res_base = res,
    target_moments = baseline_moments,
    param_names = problematic_params,
    weighting_matrix = nothing,
    matrix_moment_order = nothing
);

# Settings for the grid around the true value
n_grid = 41
rel_width = 0.01

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
fig = Figure(size = (900, 400))
for i in eachindex(problematic_params)
    row = 1
    col = ((i - 1) % 3) + 1 
    ax = Axis(fig[row, col],
        title = string(problematic_params[i]),
        xlabel = "parameter value",
        ylabel = "objective value"
    )

    grid = grids_problem[i]
    fvals = collect(view(fvals_shared_problem, i, :))

    lines!(ax, grid, fvals, color = :steelblue)
    vlines!(ax, [θ0_problem[i]], color = :red, linestyle = :dash, linewidth = 2)

    finite_idx = findall(isfinite, fvals)
    if !isempty(finite_idx)
        jmin = finite_idx[argmin(@view fvals[finite_idx])]
        scatter!(ax, [grid[jmin]], [fvals[jmin]], color = :orange, markersize = 8)
    end
end

fig[2, 1:3] = Label(fig, "Objective function profiles", fontsize = 18)
display(fig)

