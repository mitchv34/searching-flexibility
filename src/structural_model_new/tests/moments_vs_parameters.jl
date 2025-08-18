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