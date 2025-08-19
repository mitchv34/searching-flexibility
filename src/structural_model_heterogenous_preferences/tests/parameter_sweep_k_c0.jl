using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include(joinpath(@__DIR__, "../ModelSetup.jl"))
include(joinpath(@__DIR__, "../ModelSolver.jl"))
include(joinpath(@__DIR__, "../ModelEstimation.jl"))

using CSV, DataFrames, CairoMakie, Statistics

config = "src/structural_model_heterogenous_preferences/model_parameters.yaml"
prim_base, res_base = initializeModel(config)

# coarse grids
k_grid = [2.0, 5.0, 10.0, 20.0]
c0_grid = [0.01, 0.05, 0.1, 0.2]

rows = Vector{Any}()
for k in k_grid
    for c0 in c0_grid
        println("Running k=", k, ", c0=", c0)
        prim, res = update_primitives_results(prim_base, res_base, Dict(:k => k, :c₀ => c0))
        @time solve_model(prim, res, verbose=false, max_iter=2000, tol=1e-8, λ_S_init=0.05, λ_u_init=0.05)
        m = compute_model_moments(prim, res)
        push!(rows, (k=k, c0=c0, mean_alpha=m[:mean_alpha], var_alpha=m[:var_alpha]))
    end
end

df = DataFrame(rows)
CSV.write(joinpath(@__DIR__, "sweep_k_c0.csv"), df)

# plot heatmaps and save into the same tests directory as this script
png_out = joinpath(@__DIR__, "parameter_sweep_k_c0.png")
f = Figure(size=(900,400))
ax1 = Axis(f[1,1], title="mean_alpha")
heatmap!(ax1, reshape(df.mean_alpha, length(c0_grid), length(k_grid))')
ax2 = Axis(f[1,2], title="var_alpha")
heatmap!(ax2, reshape(df.var_alpha, length(c0_grid), length(k_grid))')
save(png_out, f)
println("Saved sweep CSV to: $(joinpath(@__DIR__, "sweep_k_c0.csv"))")
println("Saved sweep PNG to: $png_out")
