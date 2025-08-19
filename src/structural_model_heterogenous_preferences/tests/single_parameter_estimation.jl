using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include(joinpath(@__DIR__, "..", "ModelSetup.jl"))
include(joinpath(@__DIR__, "..", "ModelSolver.jl"))
include(joinpath(@__DIR__, "..", "ModelEstimation.jl"))

prim, res = initializeModel(joinpath(@__DIR__, "..", "model_parameters.yaml"))

# Pick a single parameter to estimate (c0) for quick smoke
param = :c₀
initial = getfield(prim, param)

p = (
    prim_base = prim,
    res_base = res,
    target_moments = compute_model_moments(prim, res),
    param_names = [param],
    weighting_matrix = nothing,
    matrix_moment_order = nothing,
)

# Try a tiny grid around initial value
grid = range(initial * 0.8, initial * 1.2, length=21)
vals = Float64[]
for v in grid
    vals = push!(vals, objective_function([v], p))
end

println("Grid results for $(param):")
for (g, val) in zip(grid, vals)
    println(g, ", ", val)
end
