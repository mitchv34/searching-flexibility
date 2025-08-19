using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Distributed
addprocs(9)

const ROOT = joinpath(@__DIR__, "..")
@everywhere begin
        using Optimization, OptimizationOptimJL, Optim
        using Random, Statistics, SharedArrays, ForwardDiff
        include(joinpath($ROOT, "ModelSetup.jl"))
        include(joinpath($ROOT, "ModelSolver.jl"))
    include(joinpath($ROOT, "ModelEstimation.jl"))
end

using Random, Statistics
using Term, Printf
using CairoMakie

# Point to the heterogeneous model configuration file
config = joinpath(ROOT, "model_parameters.yaml")
# Initialize the model
prim, res = initializeModel(config);
# Solve the model
@time solve_model(prim, res);
# Compute baseline moments
baseline_moments = compute_model_moments(prim, res)

# Parameters to profile 
params_to_estimate = [:aₕ, :bₕ, :c₀, :k, :χ, :A₁, :ν, :ψ₀, :ϕ, :κ₀]
initial_values = [getfield(prim, k) for k in params_to_estimate]

# Build problem payload following structural_model_new pattern
p = (
    prim_base = prim,
    res_base = res,
    target_moments = baseline_moments,
    param_names = params_to_estimate,
    weighting_matrix = nothing,
    matrix_moment_order = nothing,
    # warm-start cache and rolling solver-state used by objective_function
    last_res = Ref(res),
    solver_state = Ref((λ_S_init = 0.1, λ_u_init = 0.1)),
    # solver controls forwarded by objective_function
    tol = 1e-7,
    max_iter = 25_000,
    λ_S_init = 0.01,
    λ_u_init = 0.01
);

# Settings for the grid around the true value
n_grid = 41
rel_width = 0.5

# Create grids
θ0 = collect(float.(initial_values))
grids = [range(param * (1 - rel_width), param * (1 + rel_width), n_grid) for param in θ0]

# Ship immutable inputs to workers once
@everywhere const POBJ = $p
@everywhere const THETA0 = $θ0

# Worker-side evaluation for one parameter's grid
@everywhere function eval_param_grid(i::Int, grid)
    fvals = Vector{Float64}(undef, length(grid))
    θ = copy(THETA0)
    for j in eachindex(grid)
        θ[i] = grid[j]
        fvals[j] =  objective_function(θ, POBJ)
    end
    return fvals
end

n_params = length(params_to_estimate)
fvals_shared = SharedArray{Float64}(n_params, n_grid)
fill!(fvals_shared, NaN)

results = pmap(i -> eval_param_grid(i, grids[i]), 1:n_params)

for i in 1:n_params
    fvals_shared[i, :] = results[i]
end

fig = Figure(size = (1000, 800))
for i in eachindex(params_to_estimate)
    row = ((i - 1) ÷ 3) + 1
    col = ((i - 1) % 3) + 1
    ax = Axis(fig[row, col], title = string(params_to_estimate[i]))
    grid = grids[i]
    fvals = collect(view(fvals_shared, i, :))
    lines!(ax, grid, fvals)
    vlines!(ax, [θ0[i]], color = :red, linestyle = :dash)
end

display(fig)


