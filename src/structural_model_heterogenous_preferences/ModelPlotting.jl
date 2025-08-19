#==========================================================================================
Module: ModelPlotting.jl
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2025-02-27
Description: Contains all plotting functions for visualizing the random search 
        distributions, surplus functions, policy functions, and various analyses.
==========================================================================================#

# module ModelPlotting

using CairoMakie, LaTeXStrings, StatsBase, Printf
include(joinpath(@__DIR__, "ModelSetup.jl"))
include(joinpath(@__DIR__, "ModelEstimation.jl"))
using Roots
using Dates

export plot_employment_distribution, plot_employment_distribution_with_marginals,
       plot_surplus_function, plot_alpha_policy, plot_wage_policy,
       plot_wage_amenity_tradeoff, plot_outcomes_by_skill,
       plot_alpha_derivation_and_policy, plot_work_arrangement_regimes,
       plot_alpha_policy_by_firm_type, plot_z_distribution, plot_s_flow_diagnostics,
       create_figure, create_axis, plot_avg_alpha, plot_avg_wage

const COLORS = ["#0072B2", "#D55E00", "#009E73", "#F0E442"]
const BACKGROUND_COLOR = "#FAFAFA"
const FONT_CHOICE = "CMU Serif"
const FONT_SIZE_MANUSCRIPT = 14
const FONT_SIZE_PRESENTATION = 24
const FONT_SIZE = FONT_SIZE_PRESENTATION

function create_figure(;type="normal")
    if type == "wide"
        size = (1200, 800)
        scale = 1.5
    elseif type == "tall"
        size = (800, 1200)
        scale = 1.5
    elseif type == "ultrawide"
        size = (1400, 600)
        scale = 1.5
    elseif type == "ultra"
        size = (1600, 1200)
        scale = 1.5
    else
        size = (800, 600)
        scale = 1.0
    end
    return Figure(
        size = size, 
        fontsize = FONT_SIZE * scale,
        backgroundcolor = BACKGROUND_COLOR,
        fonts = (; regular=FONT_CHOICE, italic=FONT_CHOICE, bold=FONT_CHOICE)
    )
end

"""
    plot_z_distribution(prim)

Plot the distribution of heterogeneous disutility c_i = c₀ * z using the
model primitives. Returns a `Figure`.
"""
function plot_z_distribution(prim)
    # compute expectation and grid, and print diagnostics
    expected_z = prim.k
    expected_c = prim.c₀ * expected_z

    println("--- Expected Disutility ---")
    @printf("E[z] = %.2f\n", expected_z)
    @printf("E[cᵢ] = c₀ * E[z] = %.2f * %.2f = %.2f\n", prim.c₀, expected_z, expected_c)

    z_max_quantile = 0.999
    z_grid = range(0.001, quantile(prim.z_dist, z_max_quantile), length=400)
    x_axis_c = prim.c₀ .* z_grid
    y_axis_pdf = pdf.(prim.z_dist, z_grid) ./ prim.c₀

    fig = create_figure(type="wide")
    ax = create_axis(fig[1,1], "Distribution of Heterogeneous Preferences (k=$(prim.k))",
                     "Disutility Parameter (cᵢ)", "Probability Density")

    lines!(ax, x_axis_c, y_axis_pdf; color=:steelblue, linewidth=2, label="PDF of cᵢ = c₀*z")
    vlines!(ax, [expected_c]; color=:red, linewidth=2, linestyle=:dash, label="E[cᵢ] = $(round(expected_c, digits=2))")
    axislegend(ax; position = :rt)
    return fig
end


"""
    plot_s_flow_diagnostics(s_flow, prim)

Create a set of diagnostic figures for the 2D `s_flow` matrix. Returns a
tuple of Figures: (heatmap, heatmap_contours, slices_and_marginals)
"""
function plot_s_flow_diagnostics(s_flow::AbstractMatrix, prim)
    h_vals = prim.h_grid
    ψ_vals = prim.ψ_grid

    dims = size(s_flow)
    mid_row = Int(clamp(div(dims[1], 2), 1, dims[1]))
    mid_col = Int(clamp(div(dims[2], 2), 1, dims[2]))

    # Heatmap
    fig1 = create_figure(type="wide")
    ax1 = create_axis(fig1[1,1], "s_flow heatmap", "ψ", "h")
    hm = heatmap!(ax1, ψ_vals, h_vals, s_flow; colormap=:viridis)
    Colorbar(fig1[1,2], hm; label="s_flow")

    # Heatmap + contours
    fig2 = create_figure(type="wide")
    ax2 = create_axis(fig2[1,1], "s_flow heatmap + contours", "ψ", "h")
    heatmap!(ax2, ψ_vals, h_vals, s_flow; colormap=:thermal)
    contour!(ax2, ψ_vals, h_vals, s_flow', color=:black, linewidth=1)

    # Slices and marginals
    fig3 = create_figure(type="wide")
    ax3 = create_axis(fig3[1,1], "Slices: middle row & middle column", "index", "s_flow")
    lines!(ax3, ψ_vals, s_flow[mid_row, :], color=:red, label="row = $mid_row")
    lines!(ax3, h_vals, s_flow[:, mid_col], color=:blue, label="col = $mid_col")
    axislegend(ax3, position = :rb)

    # Marginals
    fig4 = create_figure(type="ultrawide")
    ax4a = create_axis(fig4[1,1], "Marginal over ψ (average across h)", "ψ", "mean s_flow")
    marg_x = vec(mean(s_flow, dims=1))
    lines!(ax4a, ψ_vals, marg_x, color=:steelblue, linewidth=2)

    ax4b = create_axis(fig4[1,2], "Marginal over h (average across ψ)", "h", "mean s_flow")
    marg_y = vec(mean(s_flow, dims=2))
    lines!(ax4b, h_vals, marg_y, color=:tomato, linewidth=2)

    return fig1, fig2, fig3, fig4
end

function create_axis(where_, title, xlabel, ylabel)
    ax = Axis(
        where_, 
        title = title, 
        xlabel = xlabel, 
        ylabel = ylabel,
        xticklabelsize = FONT_SIZE ,
        xtickalign = 1, 
        xticksize = 10, 
        yticklabelsize = FONT_SIZE ,
        ytickalign = 1, 
        yticksize = 10, 
        xgridvisible = false, 
        ygridvisible = false, 
        topspinevisible = false, 
        rightspinevisible = false
    )
    return ax
end

function plot_employment_distribution(results, prim)
    h_label = latexstring("Worker Skill (\$h\$)")
    ψ_label = latexstring("Firm Remote Efficiency (\$\\psi\$)")
    fig = create_figure()
    ax = create_axis(fig[1, 1], latexstring("Equilibrium Employment Distribution \$n(h, \\psi)\$"), h_label, ψ_label)
    hmap = heatmap!(ax, prim.h_grid, prim.ψ_grid, results.n', colormap = :viridis)
    Colorbar(fig[1, 2], hmap, label = "Mass of Workers")
    return fig
end

function plot_employment_distribution_with_marginals(results, prim)
    fig = create_figure(type="normal")
    h_vals = prim.h_grid
    ψ_vals = prim.ψ_grid
    n_dist = results.n
    n_h_marginal = vec(sum(n_dist, dims=2))
    n_ψ_marginal = vec(sum(n_dist, dims=1))
    ax_main = Axis(fig[2, 1], xlabel = "Worker Skill (h)", ylabel = "Firm Remote Efficiency (ψ)")
    ax_top = Axis(fig[1, 1], ylabel = "Density")
    ax_right = Axis(fig[2, 2], xlabel = "Density")
    contourf!(ax_main, h_vals, ψ_vals, n_dist, colormap = :viridis, levels=15)
    lines!(ax_top, h_vals, n_h_marginal, color=COLORS[1], linewidth=3)
    lines!(ax_right, n_ψ_marginal, ψ_vals, color=COLORS[2], linewidth=3)
    linkxaxes!(ax_main, ax_top)
    linkyaxes!(ax_main, ax_right)
    hidedecorations!(ax_top, grid = false, ticks=false)
    hidedecorations!(ax_right, grid = false, ticks=false)
    colgap!(fig.layout, 1, 5)
    rowgap!(fig.layout, 1, 5)
    Label(fig[0, :], "Employment Distribution with Marginals", fontsize = FONT_SIZE*1.2, font=:bold)
    return fig
end

function plot_surplus_function(results, prim)
    h_label = latexstring("Worker Skill (\$h\$)")
    ψ_label = latexstring("Firm Remote Efficiency (\$\\psi\$)")
    fig = create_figure()
    ax = create_axis(fig[1, 1], latexstring("Equilibrium Match Surplus \$S(h, \\psi)\$"), h_label, ψ_label)
    hmap = heatmap!(ax, prim.h_grid, prim.ψ_grid, results.S, colormap = :plasma)
    Colorbar(fig[1, 2], hmap, label = "Match Surplus")
    contour!(ax, prim.h_grid, prim.ψ_grid, results.S, levels = [0.0], color = :white, linestyle = :dash, linewidth = 4)
    return fig
end

function plot_alpha_policy(results, prim)
    h_label = latexstring("Worker Skill (\$h\$)")
    ψ_label = latexstring("Firm Remote Efficiency (\$\\psi\$)")
    
    fig = create_figure()
    ax = create_axis(fig[1, 1],
                    latexstring("Optimal Remote Work Share \$\\alpha^*(h, \\psi)\$"),
                    h_label,
                    ψ_label)

    # Plot the base heatmap for optimal remote work share
    hmap = heatmap!(ax, prim.h_grid, prim.ψ_grid, results.α_policy',
                    colormap = :coolwarm, colorrange = (0, 1))
    
    Colorbar(fig[1, 2], hmap, label = "Remote Share (α)", ticks = 0:0.2:1)
    
    # Create a mask: cells where the surplus S is negative (i.e. workers are not hired)
    # Note: We assume results.S is of the same dimension as α_policy (with h as rows, ψ as columns)
    gray_mask = map(x -> x < 0 ? 1.0 : NaN, results.S)
    
    # Overlay grey areas on cells where S < 0.
    # The mask is transposed to align with the heatmap orientation.
    heatmap!(ax, prim.h_grid, prim.ψ_grid, gray_mask',
              colormap = cgrad([:gray]),
              interpolate = false,
              transparency = true)

    return fig
end

function plot_wage_policy(results, prim)
    h_label = latexstring("Worker Skill (\$h\$)")
    ψ_label = latexstring("Firm Remote Efficiency (\$\\psi\$)")
    
    fig = create_figure()
    ax = create_axis(   fig[1, 1], 
                        latexstring("Equilibrium Wage Policy \$w^{*}(h, \\psi)\$"),
                        h_label,
                        ψ_label
                    )

    hmap = heatmap!(ax, prim.h_grid, prim.ψ_grid, results.w_policy',
                    colormap = :coolwarm)
    # Create a mask: cells where the surplus S is negative (i.e. workers are not hired)
    # Note: We assume results.S is of the same dimension as α_policy (with h as rows, ψ as columns)
    gray_mask = map(x -> x < 0 ? 1.0 : NaN, results.S)
    
    # Overlay grey areas on cells where S < 0.
    # The mask is transposed to align with the heatmap orientation.
    heatmap!(ax, prim.h_grid, prim.ψ_grid, gray_mask',
              colormap = cgrad([:gray]),
              interpolate = false,
              transparency = true)
    Colorbar(fig[1, 2], hmap, label = "Wage (w)")
    
    return fig
end

"""
    plot_avg_alpha(prim, res)

Plot the expected (average) remote work share (avg_alpha) over the (h, ψ) grid.
Requires: calculate_average_policies from ModelEstimation.jl.
Returns a Figure.
"""
function plot_avg_alpha(prim, res)
    avg_alpha, _, _ = calculate_average_policies(prim, res)
    h_label = latexstring("Worker Skill (\$h\$)")
    ψ_label = latexstring("Firm Remote Efficiency (\$\\psi\$)")
    fig = create_figure()
    ax = create_axis(fig[1, 1], latexstring("Expected Remote Work Share \$\\alpha^{*}(h, \\psi)\$"), h_label, ψ_label)
    hmap = heatmap!(ax, prim.h_grid, prim.ψ_grid, avg_alpha', colormap = :coolwarm, colorrange = (0, 1))
    Colorbar(fig[1, 2], hmap, label = "E[α]", ticks = 0:0.2:1)
    return fig
end

"""
    plot_avg_wage(prim, res)

Plot the expected (average) wage (avg_wage) over the (h, ψ) grid.
Requires: calculate_average_policies from ModelEstimation.jl.
Returns a Figure.
"""
function plot_avg_wage(prim, res)
    _, _, avg_wage = calculate_average_policies(prim, res)
    h_label = latexstring("Worker Skill (\$h\$)")
    ψ_label = latexstring("Firm Remote Efficiency (\$\\psi\$)")
    fig = create_figure()
    ax = create_axis(fig[1, 1], latexstring("Expected Wage \$w^{*}(h, \\psi) \$"), h_label, ψ_label)
    hmap = heatmap!(ax, prim.h_grid, prim.ψ_grid, avg_wage', colormap = :viridis)
    Colorbar(fig[1, 2], hmap, label = "E[w]")
    return fig
end


"""
    plot_wage_amenity_tradeoff(results, prim)

Plots the wage-amenity trade-off showing how wages change with remote work share
for different skill levels.
"""
function plot_wage_amenity_tradeoff(results, prim)
    
    fig = create_figure()
    ax = create_axis(
                        fig[1, 1],
                        latexstring("Wage-Amenity Trade-off"),
                        latexstring("Remote Share \$(\\alpha)\$"), 
                        latexstring("Wage \$(w)\$"))

    # Select a few skill levels to plot (e.g., low, medium, high)
    n_h = length(prim.h_grid)
    percentiles = [0.3, 0.6, 0.9]
    h_indices = floor.(Int, percentiles .* (n_h - 1)) .+ 1
    
    for (i, h_idx) in enumerate(h_indices)
        h_val = prim.h_grid[h_idx]
        α_slice = results.α_policy[h_idx, :]
        w_slice = results.w_policy[h_idx, :]
        
        # Filter out non-positive surplus matches where wages might be meaningless
        active_matches = results.S[h_idx, :] .> 0
        
        lines!(ax, α_slice[active_matches], w_slice[active_matches],
                label = "h = $(round(h_val, digits=2))",
                color = COLORS[i], linewidth = 3)
    end
    
    axislegend(ax, position = :rb) # Add a legend
    
    return fig
end

"""
    plot_outcomes_by_skill(results, prim)

Plots key labor market outcomes (unemployment rate, value of unemployment, 
average wage, average remote share) by worker skill level.
"""
function plot_outcomes_by_skill(results, prim)
    h_label = latexstring("Worker Skill (\$h\$)")
    
    # Create a 2x2 figure layout (ultrawide for better horizontal resolution)
    fig = create_figure(type="ultra")
    
    # --- Unemployment Rate ---
    ax1 = create_axis(fig[1, 1], "Unemployment Rate by Skill", h_label, "Unemployment Rate")
        unemp_rate_h = results.u ./ prim.h_pdf
        lines!(ax1, prim.h_grid, unemp_rate_h, color = COLORS[1], linewidth = 3)
    
    # --- Value of Unemployment ---
    ax2 = create_axis(fig[1, 2], "Value of Unemployment by Skill", h_label, "Value U(h)")
    lines!(ax2, prim.h_grid, results.U, color = COLORS[2], linewidth = 3)
    
    # --- Average Wage of Employed Workers ---
    ax3 = create_axis(fig[2, 1], "Average Wage by Skill", h_label, "Average Wage")
        avg_wages = zeros(prim.n_h)
    for i_h in 1:prim.n_h
        n_h_slice = results.n[i_h, :]
        total_employed_h = sum(n_h_slice)
        if total_employed_h > 0
            job_dist_h = n_h_slice ./ total_employed_h
            avg_wages[i_h] = sum(results.w_policy[i_h, :] .* job_dist_h)
        end
    end
    lines!(ax3, prim.h_grid, avg_wages, color = COLORS[3], linewidth = 3)

    # --- Average Remote Work Share of Employed Workers ---
    ax4 = create_axis(fig[2, 2], "Average Remote Work Share by Skill", h_label, "Average α")
        avg_remote = zeros(prim.n_h)
    for i_h in 1:prim.n_h
        n_h_slice = results.n[i_h, :]
        total_employed_h = sum(n_h_slice)
        if total_employed_h > 0
            job_dist_h = n_h_slice ./ total_employed_h
            avg_remote[i_h] = sum(results.α_policy[i_h, :] .* job_dist_h)
        end
    end
    lines!(ax4, prim.h_grid, avg_remote, color = COLORS[1], linewidth = 3)
    
    return fig
end


# end # module ModelPlotting
