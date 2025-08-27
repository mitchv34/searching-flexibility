using Pkg
Pkg.activate("/project/high_tech_ind/searching-flexibility")
Pkg.instantiate()

using Distributed
addprocs(9)

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


const FIGURES_DIR = joinpath(ROOT, "figures", "structural_model_heterogenous_preferences", "tests", "cross_moment_check")

using Random, Statistics, Term, Printf, CairoMakie, PrettyTables

const N_GRID = 41 # Number of param_grid points
const DEV = 0.2  # ±20% deviation

# CRITICAL FIX: Distribute MODEL_CONFIG and all necessary functions to workers
@everywhere begin
    # Make sure workers have access to the config
    const WORKER_MODEL_CONFIG = $MODEL_CONFIG
    # Worker function: generic parameter sweep with fresh solves each time
    function sweep_param(param_grid, prim_base, res_base, pname::Symbol)
        moments_list = Vector{Any}(undef, length(param_grid))
        for i in eachindex(param_grid)
            pval = param_grid[i]
            try
                # Update primitives with new parameter value
                prim_i, res_i = update_primitives_results(prim_base, res_base, Dict(pname => pval))
                
                # Use config-based solver (now workers have access to WORKER_MODEL_CONFIG)
                solve_model(prim_i, res_i, config=WORKER_MODEL_CONFIG)
                
                # Compute moments
                moments_list[i] = compute_model_moments(prim_i, res_i)
                
            catch e
                println("Worker: Parameter $(pname)=$(pval) failed: $(e)")
                
                # Try to get a template from a successful computation
                if i > 1 && !isnothing(moments_list[i-1]) && isa(moments_list[i-1], Dict)
                    # Use the previous successful computation as template
                    nan_moments = Dict(k => NaN for k in keys(moments_list[i-1]))
                else
                    # More complete fallback
                    try
                        baseline_moments = compute_model_moments(prim_base, res_base)
                        nan_moments = Dict(k => NaN for k in keys(baseline_moments))
                    catch
                        # Ultimate fallback with comprehensive moment list
                        nan_moments = Dict{Symbol, Float64}(
                            :wfh_share => NaN, :avg_commute => NaN, :avg_surplus => NaN, 
                            :theta => NaN, :wage_premium => NaN, :unemployment_rate => NaN,
                            :job_finding_rate => NaN, :avg_wage => NaN, :wage_variance => NaN,
                            :alpha_mean => NaN, :alpha_variance => NaN, :productivity_premium => NaN,
                            :wage_gap => NaN, :output_gap => NaN, :welfare => NaN
                        )
                    end
                end
                moments_list[i] = nan_moments
            end
        end
        return moments_list
    end
end

begin
    # Parameters to sweep
    params_to_estimate = [:aₕ, :bₕ, :c₀, :μ, :χ, :A₁, :ν, :ψ₀, :ϕ, :κ₀]
    # Initialize and solve the model for the first time
    prim, res = initializeModel(MODEL_CONFIG)
    @time solve_model(prim, res, config=MODEL_CONFIG)

    println("Starting distributed parameter sweeps for: ", join(string.(params_to_estimate), ", "))

    # Assign workers round-robin
    wrks = workers()
    @assert !isempty(wrks) "No workers available. Ensure addprocs(...) succeeded."

    # Launch sweeps
    futures = Dict{Symbol, Future}()
    grids   = Dict{Symbol, Vector{Float64}}()
    for (j, pname) in pairs(params_to_estimate)
        base_val = getproperty(prim, pname)
        min_val = 1e-6  # Minimum value to avoid zero division
        if !(base_val == 0)
            # Ensure lower bound is strictly positive - use smaller range for stability
            lo = max(min_val, (1 - DEV) * base_val)
            hi = (1 + DEV) * base_val
            # Handle cases where val is negative
            lo_val, hi_val = min(lo, hi), max(lo, hi)
            param_grid =  collect(range(lo_val, hi_val; length=N_GRID))
        else
            # Fallback when baseline is zero: symmetric absolute window
            param_grid = collect(range(-0.2, 0.2; length=N_GRID))
        end

        grids[pname] = param_grid
        wid = wrks[mod1(j, length(wrks))]
        println("Dispatching sweep for ", String(pname), " to worker ", wid, " over range [", param_grid[1], ", ", param_grid[end], "]")
        futures[pname] = remotecall(sweep_param, wid, param_grid, prim, res, pname)
    end

    # Helper to extract numeric keys from Dict{Symbol, Float64} moments
    function get_numeric_keys(moments_list)
        all_keys = Symbol[]
        for m in moments_list
            if isa(m, Dict)
                append!(all_keys, keys(m))
            end
        end
        all_keys = unique(all_keys)
        sort!(all_keys, by=string)
        numeric_keys = Symbol[]
        for k in all_keys
            if any(i -> isa(moments_list[i], Dict) && haskey(moments_list[i], k) && moments_list[i][k] isa Number, eachindex(moments_list))
                push!(numeric_keys, k)
            end
        end
        return numeric_keys
    end

    # Collect, plot, and save for each parameter
    # Create figures directory if it doesn't exist
    mkpath(FIGURES_DIR)
    
    for pname in params_to_estimate
        println("Fetching results for ", String(pname), "...")
        moments_list = fetch(futures[pname])
        param_grid = grids[pname]
        numeric_keys = get_numeric_keys(moments_list)

        println("Found $(length(numeric_keys)) numeric moments: $(numeric_keys)")

        if length(numeric_keys) == 0
            println("⚠️  No valid moments found for $(pname), skipping plot...")
            continue
        end

        println("Plotting results for ", String(pname), "...")
        nplots = length(numeric_keys)
        ncol = 4
        nrow = max(1, cld(nplots, ncol))
        fig = Figure(size = (1200, 900))

        for (idx, k) in enumerate(numeric_keys)
            row = ((idx - 1) ÷ ncol) + 1
            col = ((idx - 1) % ncol) + 1
            ax = Axis(fig[row, col], title=String(k), xlabel=String(pname), ylabel="moment")

            ys = Vector{Float64}(undef, length(param_grid))
            @inbounds for i in eachindex(param_grid)
                if isa(moments_list[i], Dict) && haskey(moments_list[i], k)
                    v = moments_list[i][k]
                    ys[i] = v isa Number ? float(v) : NaN
                else
                    ys[i] = NaN
                end
            end

            # Only plot if we have some non-NaN values
            if any(!isnan, ys)
                lines!(ax, param_grid, ys, color=:steelblue)
                vlines!(ax, [float(getproperty(prim, pname))], color=:red, linestyle=:dash, linewidth=2)
            else
                text!(ax, 0.5, 0.5, text="No valid data", align=(:center, :center))
            end
        end

        fig[nrow + 1, 1:ncol] = Label(fig, "Model moments vs $(String(pname)) (distributed)", fontsize=18)
        display(fig)

        out_png = joinpath(FIGURES_DIR, "moments_vs_$(String(pname)).png")
        save(out_png, fig)
        println("Saved: ", out_png)
    end

    println("All parameter sweeps completed!")
end

