using Pkg
Pkg.activate("../../..")
Pkg.instantiate()

include(joinpath(@__DIR__, "../ModelSetup.jl"))
include(joinpath(@__DIR__, "../ModelSolver.jl"))
include(joinpath(@__DIR__, "../ModelEstimation.jl"))
using CSV, DataFrames, CairoMakie, Statistics, YAML, LinearAlgebra

csv_in = joinpath(@__DIR__, "sweep_k_c0.csv")
if !isfile(csv_in)
    error("sweep CSV not found: $csv_in — run tests/parameter_sweep_k_c0.jl first")
end

out = load_sweep_targets(csv_in)
df = out[:df]
k_grid = out[:k_grid]
c0_grid = out[:c0_grid]
mean_mat = out[:mean_alpha_matrix]
var_mat = out[:var_alpha_matrix]
elast = out[:elasticities]

# Plot mean_alpha heatmap (k on x axis, c0 on y axis)
fig = Figure(size=(800,500))
ax = Axis(fig[1,1], xlabel="k", ylabel="c0", title="mean_alpha (rows=c0, cols=k)")
# Makie's heatmap expects matrix with dims (ny, nx) matching c0 x k ordering
heat = heatmap!(ax, k_grid, c0_grid, mean_mat; colormap = :viridis)
Colorbar(fig[1,2], heat, label = "mean_alpha")
# set ticks to k values
ax.xticks = (k_grid, string.(k_grid))
ax.yticks = (c0_grid, string.(c0_grid))

png_out = joinpath(@__DIR__, "analysis_mean_alpha_heatmap.png")
pdf_out = joinpath(@__DIR__, "analysis_mean_alpha_heatmap.pdf")
save(png_out, fig)
save(pdf_out, fig)
println("Saved heatmap: $png_out and $pdf_out")

# Save elasticities
CSV.write(joinpath(@__DIR__, "sweep_elasticities.csv"), elast)
# Convert DataFrame rows to vector of Dicts for YAML output without requiring Tables.jl
rows = [ Dict(string(k) => v for (k,v) in zip(names(elast), collect(eachrow(elast))[i])) for i in 1:nrow(elast) ]
YAML.write_file(joinpath(@__DIR__, "sweep_elasticities.yaml"), Dict("elasticities" => rows))
println("Saved elasticities CSV and YAML in tests/")

# Print brief summary
println("Elasticities summary:")
show(elast)
