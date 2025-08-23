using Pkg
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

# Point to the heterogeneous model configuration file
config = joinpath(ROOT,  "model_parameters.yaml")
prim, res = initializeModel(config)
@time solve_model(prim, res)
baseline_moments = compute_model_moments(prim, res)

begin

@everywhere function sweep_param(grid, prim_base, res_base, pname::Symbol)
    moments_list = Vector{Any}(undef, length(grid))
    for i in eachindex(grid)
        pval = grid[i]
        prim_i, res_i = update_primitives_results(prim_base, res_base, Dict(pname => pval))
        solve_model(prim_i, res_i, verbose=false, λ_S_init = 0.01, λ_u_init = 0.01)
        moments_list[i] = compute_model_moments(prim_i, res_i)
    end
    return moments_list
end

    # Parameters to sweep
    params_to_estimate = [:aₕ, :bₕ, :c₀, :χ, :ν, :ψ₀, :ϕ, :κ₀]

    # 41-point grid with ±50% deviation around baseline value
    param_grid = function(val; n=41)
        if !(val == 0)
            lo = 0.5 * val
            hi = 1.5 * val
            lo ≤ hi ? collect(range(lo, hi; length=n)) : collect(range(hi, lo; length=n))
        else
            collect(range(-0.5, 0.5; length=n))
        end
    end

    println("Starting distributed parameter sweeps for: ", join(string.(params_to_estimate), ", "))

    wrks = workers()
    @assert !isempty(wrks) "No workers available. Ensure addprocs(...) succeeded."

    futures = Dict{Symbol, Future}()
    grids   = Dict{Symbol, Vector{Float64}}()
    for (j, pname) in pairs(params_to_estimate)
        base_val = getproperty(prim, pname)
        grid = Float64.(param_grid(base_val; n=41))
        grids[pname] = grid
        wid = wrks[mod1(j, length(wrks))]
        println("Dispatching sweep for ", String(pname), " to worker ", wid, " over range [", grid[1], ", ", grid[end], "]")
        futures[pname] = remotecall(sweep_param, wid, grid, prim, res, pname)
    end

    function get_numeric_keys(moments_list)
        all_keys = Symbol[]
        for m in moments_list
            append!(all_keys, keys(m))
        end
        all_keys = unique(all_keys)
        sort!(all_keys, by=string)
        numeric_keys = Symbol[]
        for k in all_keys
            if any(i -> haskey(moments_list[i], k) && moments_list[i][k] isa Number, eachindex(moments_list))
                push!(numeric_keys, k)
            end
        end
        return numeric_keys
    end

    for pname in params_to_estimate
        println("Fetching results for ", String(pname), "...")
        moments_list = fetch(futures[pname])
        grid = grids[pname]
        numeric_keys = get_numeric_keys(moments_list)

        println("Plotting results for ", String(pname), "...")
        nplots = length(numeric_keys)
        ncol = 4
        nrow = max(1, cld(nplots, ncol))
        fig = Figure(size = (1200, 900))

        for (idx, k) in enumerate(numeric_keys)
            row = ((idx - 1) ÷ ncol) + 1
            col = ((idx - 1) % ncol) + 1
            ax = Axis(fig[row, col], title=String(k), xlabel=String(pname), ylabel="moment")

            ys = Vector{Float64}(undef, length(grid))
            @inbounds for i in eachindex(grid)
                v = haskey(moments_list[i], k) ? moments_list[i][k] : NaN
                ys[i] = v isa Number ? float(v) : NaN
            end

            lines!(ax, grid, ys, color=:steelblue)
            vlines!(ax, [float(getproperty(prim, pname))], color=:red, linestyle=:dash, linewidth=2)
        end

        fig[nrow + 1, 1:ncol] = Label(fig, "Model moments vs $(String(pname)) (distributed)", fontsize=18)
        display(fig)

        out_png = joinpath(ROOT, "moments_vs_$(String(pname)).png")
        save(out_png, fig)
        println("Saved: ", out_png)
    end

    println("All parameter sweeps completed!")
end