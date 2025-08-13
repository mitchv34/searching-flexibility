#==========================================================================================
Module: ModelPlotting.jl
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2025-02-27
Description: Contains all plotting functions for visualizing the random search 
        labor market model results. Includes functions for plotting employment 
        distributions, surplus functions, policy functions, and various analyses.
==========================================================================================#

module ModelPlotting

using CairoMakie, LaTeXStrings, StatsBase
using ..Types
using ..ModelFunctions
using Roots
using Dates

export plot_employment_distribution, plot_employment_distribution_with_marginals,
       plot_surplus_function, plot_alpha_policy, plot_wage_policy,
       plot_wage_amenity_tradeoff, plot_outcomes_by_skill,
       plot_alpha_derivation_and_policy, plot_work_arrangement_regimes,
       plot_alpha_policy_by_firm_type, create_figure, create_axis,
       # New benchmark economy exports
       BenchmarkEconomy, compute_benchmark_outcomes, plot_benchmark_summary,
       # New stylized facts exports  
       compute_sorting_measures, compute_wage_inequality_measures,
       plot_remote_wage_correlations, plot_within_occupation_analysis,
       run_wage_regressions, create_simulated_dataset

#==========================================================================================
#? Plotting Configuration and Utilities
==========================================================================================#

# Color palette and styling constants
# const COLORS = [
#     "#23373B",   # primary
#     "#EB811B",   # secondary
#     "#14B03E",   # tertiary
#     "#F2C94C",   # highlight
# ]

const COLORS = [
    "#0072B2",   # > Blue
    "#D55E00",   # > Burnt Orange
    "#009E73",   # > Teal
    "#F0E442",   # > Yellow
]

const BACKGROUND_COLOR = "#FAFAFA"

const FONT_CHOICE = "CMU Serif"
const FONT_SIZE_MANUSCRIPT = 14
const FONT_SIZE_PRESENTATION = 24
const FONT_SIZE = FONT_SIZE_PRESENTATION

"""
    create_figure(;type="normal")

Creates a standardized Figure object with consistent styling.

# Arguments
- `type::String`: Figure type - "normal", "wide", "tall", "ultrawide", or "ultra"

# Returns
- `Figure`: A Makie Figure object with appropriate size and styling
"""
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
    create_axis(where_, title, xlabel, ylabel)

Creates a standardized Axis object with consistent styling.

# Arguments
- `where_`: Position in the figure layout
- `title::String`: Axis title
- `xlabel::String`: X-axis label
- `ylabel::String`: Y-axis label

# Returns
- `Axis`: A Makie Axis object with consistent styling
"""
function create_axis(where_, title, xlabel, ylabel)
    ax = Axis(
        # User parameters
        where_, 
        title = title, 
        xlabel = xlabel, 
        ylabel = ylabel,
        # Tick parameters
        # xticks
        xticklabelsize = FONT_SIZE ,
        xtickalign = 1, 
        xticksize = 10, 
        # yticks
        yticklabelsize = FONT_SIZE ,
        ytickalign = 1, 
        yticksize = 10, 
        # Grid parameters
        xgridvisible = false, 
        ygridvisible = false, 
        topspinevisible = false, 
        rightspinevisible = false
    )
    return ax
end

#==========================================================================================
#? Employment and Matching Plots
==========================================================================================#

"""
    plot_employment_distribution(results, prim)

Plots the equilibrium employment distribution n(h, ψ) as a heatmap.
This shows how workers and firms sort in equilibrium.
"""
function plot_employment_distribution(results, prim)
    # Define labels for clarity
    h_label = latexstring("Worker Skill (\$h\$)")
    ψ_label = latexstring("Firm Remote Efficiency (\$\\psi\$)")
    
    fig = create_figure()
    
    ax = create_axis(fig[1, 1], 
                    latexstring("Equilibrium Employment Distribution \$n(h, \\psi)\$"), 
                    h_label, 
                    ψ_label
    )

    # Use the transpose of n for correct orientation with Axis
    # h should be on the x-axis, ψ on the y-axis
    hmap = heatmap!(ax, prim.h_grid.values, prim.ψ_grid.values, results.n',
                    colormap = :viridis) # or :thermal, :inferno
    
    Colorbar(fig[1, 2], hmap, label = "Mass of Workers")
    
    return fig
end

"""
    plot_employment_distribution_with_marginals(results, prim)

Plots the employment distribution with marginal distributions for both worker skill
and firm types.
"""
function plot_employment_distribution_with_marginals(results, prim)
    # --- Setup ---
    fig = create_figure(type="normal")
    h_vals = prim.h_grid.values
    ψ_vals = prim.ψ_grid.values
    n_dist = results.n

    # --- Calculate Marginal Distributions ---
    # Marginal distribution of employed workers over skill h
    n_h_marginal = vec(sum(n_dist, dims=2))
    # Marginal distribution of employed workers over firm type ψ
    n_ψ_marginal = vec(sum(n_dist, dims=1))

    # --- Create a Grid Layout ---
    # The main plot is in the bottom-left.
    # Top plot is for h-marginal, right plot is for ψ-marginal.
    ax_main = Axis(fig[2, 1], xlabel = "Worker Skill (h)", ylabel = "Firm Remote Efficiency (ψ)")
    ax_top = Axis(fig[1, 1], ylabel = "Density")
    ax_right = Axis(fig[2, 2], xlabel = "Density")

    # --- Main Contour Plot ---
    # Using contourf for filled contours, which is often clearer than lines
    contourf!(ax_main, h_vals, ψ_vals, n_dist, colormap = :viridis, levels=15)

    # --- Marginal Plots ---
    # Top: Marginal density over h
    lines!(ax_top, h_vals, n_h_marginal, color=COLORS[1], linewidth=3)
    # Right: Marginal density over ψ
    lines!(ax_right, n_ψ_marginal, ψ_vals, color=COLORS[2], linewidth=3)

    # --- Linking and Tidying ---
    # Link axes so they pan/zoom together
    linkxaxes!(ax_main, ax_top)
    linkyaxes!(ax_main, ax_right)

    # Hide unnecessary decorations
    hidedecorations!(ax_top, grid = false, ticks=false)
    hidedecorations!(ax_right, grid = false, ticks=false)
    colgap!(fig.layout, 1, 5) # Reduce space between columns
    rowgap!(fig.layout, 1, 5) # Reduce space between rows

    # Add a title to the whole figure
    Label(fig[0, :], "Employment Distribution with Marginals", fontsize = FONT_SIZE*1.2, font=:bold)

    return fig
end

"""
    plot_surplus_function(results, prim)

Plots the equilibrium match surplus S(h, ψ) showing where the largest gains from trade are.
"""
function plot_surplus_function(results, prim)
    h_label = latexstring("Worker Skill (\$h\$)")
    ψ_label = latexstring("Firm Remote Efficiency (\$\\psi\$)")
    
    fig = create_figure()
    ax = create_axis(fig[1, 1],
                    latexstring("Equilibrium Match Surplus \$S(h, \\psi)\$"),
                    h_label, ψ_label)

    hmap = heatmap!(ax, prim.h_grid.values, prim.ψ_grid.values, results.S,
                    colormap = :plasma)
    
    Colorbar(fig[1, 2], hmap, label = "Match Surplus")
    
    # Add a contour plot to show the zero-surplus line
    contour!(ax, prim.h_grid.values, prim.ψ_grid.values, results.S,
            levels = [0.0], color = :white, linestyle = :dash, linewidth = 4)
    
    return fig
end

#==========================================================================================
#? Policy Function Plots
==========================================================================================#

"""
    plot_alpha_policy(results, prim)

Plots the optimal remote work share α*(h, ψ) across worker-firm pairs.
"""
function plot_alpha_policy(results, prim)
    h_label = latexstring("Worker Skill (\$h\$)")
    ψ_label = latexstring("Firm Remote Efficiency (\$\\psi\$)")
    
    fig = create_figure()
    ax = create_axis(fig[1, 1],
                    latexstring("Optimal Remote Work Share \$\\alpha^*(h, \\psi)\$"),
                    h_label,
                    ψ_label)

    # Plot the base heatmap for optimal remote work share
    hmap = heatmap!(ax, prim.h_grid.values, prim.ψ_grid.values, results.α_policy',
                    colormap = :coolwarm, colorrange = (0, 1))
    
    Colorbar(fig[1, 2], hmap, label = "Remote Share (α)", ticks = 0:0.2:1)
    
    # Create a mask: cells where the surplus S is negative (i.e. workers are not hired)
    # Note: We assume results.S is of the same dimension as α_policy (with h as rows, ψ as columns)
    gray_mask = map(x -> x < 0 ? 1.0 : NaN, results.S)
    
    # Overlay grey areas on cells where S < 0.
    # The mask is transposed to align with the heatmap orientation.
    heatmap!(ax, prim.h_grid.values, prim.ψ_grid.values, gray_mask',
              colormap = cgrad([:gray]),
              interpolate = false,
              transparency = true)

    return fig
end

"""
    plot_wage_policy(results, prim)

Plots the equilibrium wage policy w*(h, ψ) showing how wages vary across matches.
"""
function plot_wage_policy(results, prim)
    h_label = latexstring("Worker Skill (\$h\$)")
    ψ_label = latexstring("Firm Remote Efficiency (\$\\psi\$)")
    
    fig = create_figure()
    ax = create_axis(   fig[1, 1], 
                        latexstring("Equilibrium Wage Policy \$w^{*}(h, \\psi)\$"),
                        h_label,
                        ψ_label
                    )

    hmap = heatmap!(ax, prim.h_grid.values, prim.ψ_grid.values, results.w_policy',
                    colormap = :coolwarm)
    # Create a mask: cells where the surplus S is negative (i.e. workers are not hired)
    # Note: We assume results.S is of the same dimension as α_policy (with h as rows, ψ as columns)
    gray_mask = map(x -> x < 0 ? 1.0 : NaN, results.S)
    
    # Overlay grey areas on cells where S < 0.
    # The mask is transposed to align with the heatmap orientation.
    heatmap!(ax, prim.h_grid.values, prim.ψ_grid.values, gray_mask',
              colormap = cgrad([:gray]),
              interpolate = false,
              transparency = true)
    Colorbar(fig[1, 2], hmap, label = "Wage (w)")
    
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
    n_h = length(prim.h_grid.values)
    percentiles = [0.3, 0.6, 0.9]
    h_indices = floor.(Int, percentiles .* (n_h - 1)) .+ 1
    
    for (i, h_idx) in enumerate(h_indices)
        h_val = prim.h_grid.values[h_idx]
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

#==========================================================================================
#? Outcome Analysis Plots
==========================================================================================#

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
    unemp_rate_h = results.u ./ prim.h_grid.pdf
    lines!(ax1, prim.h_grid.values, unemp_rate_h, color = COLORS[1], linewidth = 3)
    
    # --- Value of Unemployment ---
    ax2 = create_axis(fig[1, 2], "Value of Unemployment by Skill", h_label, "Value U(h)")
    lines!(ax2, prim.h_grid.values, results.U, color = COLORS[2], linewidth = 3)
    
    # --- Average Wage of Employed Workers ---
    ax3 = create_axis(fig[2, 1], "Average Wage by Skill", h_label, "Average Wage")
    avg_wages = zeros(prim.h_grid.n)
    for i_h in 1:prim.h_grid.n
        n_h_slice = results.n[i_h, :]
        total_employed_h = sum(n_h_slice)
        if total_employed_h > 0
            job_dist_h = n_h_slice ./ total_employed_h
            avg_wages[i_h] = sum(results.w_policy[i_h, :] .* job_dist_h)
        end
    end
    lines!(ax3, prim.h_grid.values, avg_wages, color = COLORS[3], linewidth = 3)
    
    # --- Average Remote Work Share of Employed Workers ---
    ax4 = create_axis(fig[2, 2], "Average Remote Work Share by Skill", h_label, "Average α")
    avg_remote = zeros(prim.h_grid.n)
    for i_h in 1:prim.h_grid.n
        n_h_slice = results.n[i_h, :]
        total_employed_h = sum(n_h_slice)
        if total_employed_h > 0
            job_dist_h = n_h_slice ./ total_employed_h
            avg_remote[i_h] = sum(results.α_policy[i_h, :] .* job_dist_h)
        end
    end
    lines!(ax4, prim.h_grid.values, avg_remote, color = COLORS[1], linewidth = 3)
    
    return fig
end

#==========================================================================================
#? Specialized Analysis Plots
==========================================================================================#

"""
    plot_alpha_derivation_and_policy(prim, res; h_idx_fixed, ψ_indices_to_vary, h_idx_secondary=nothing, plot_policy_curve=true)

Creates a dual panel plot showing the marginal benefit/cost intersection for α* determination
and the resulting policy function. Optionally includes a second skill level for comparison.
"""
function plot_alpha_derivation_and_policy(
                                            prim::Primitives,
                                            res::Results; 
                                            h_idx_fixed::Int, 
                                            ψ_indices_to_vary::Vector{Int},
                                            h_idx_secondary::Union{Int, Nothing} = nothing,
                                            plot_policy_curve::Bool = true)
    
    # --- Setup the Figure ---
    fig = Figure(size = (1200, 550), 
                    fontsize = FONT_SIZE,
                    fonts = (; regular=FONT_CHOICE, italic=FONT_CHOICE, bold=FONT_CHOICE))

    # --- Define common elements ---
    α_grid = 0.0:0.01:1.0
    
    # --- Panel 1: MB-MC Intersection Diagram (Marshallian Cross) ---
    
    ax1 = create_axis(fig[1, 1],
                    "First Order Condition",
                    "Marginal Value",
                    latexstring("Remote Work Share \$(\\alpha)\$"))

    # Use the fixed h_idx provided as an argument
    h_val_fixed = prim.h_grid.values[h_idx_fixed]
    
    # Plot the upward-sloping MC curve (now with x and y swapped)
    mrs_func = α -> -ModelFunctions.evaluate_derivative(prim.utility_function, "α", 0.0, α, h_val_fixed) / 
                        ModelFunctions.evaluate_derivative(prim.utility_function, "w", 0.0, α, h_val_fixed)
    marginal_cost_values = mrs_func.(α_grid)
    lines!(ax1, marginal_cost_values, α_grid, color = COLORS[1], linewidth = 3)
    # Annotate the plot only if showing a single h 
    if isnothing(h_idx_secondary) 
        text!(ax1, 
        mean(marginal_cost_values),
        0.6, text=latexstring("Marginal Cost \$\\left(\\frac{\\partial c}{\\partial \\alpha}\\right)\$"), 
        fontsize=FONT_SIZE*0.8, 
        color = COLORS[1],
        align=(:center, :bottom))
    end

    # Secondary skill level MC curve (if provided)
    mrs_func_secondary = nothing  # Initialize outside the if block
    if h_idx_secondary !== nothing
        h_val_secondary = prim.h_grid.values[h_idx_secondary]
        mrs_func_secondary = α -> -ModelFunctions.evaluate_derivative(prim.utility_function, "α", 0.0, α, h_val_secondary) / 
                                    ModelFunctions.evaluate_derivative(prim.utility_function, "w", 0.0, α, h_val_secondary)
        marginal_cost_values_secondary = mrs_func_secondary.(α_grid)
        
        lines!(ax1, marginal_cost_values_secondary, α_grid, 
            color = COLORS[1], linewidth = 3, linestyle = :dash)
    end

    # Store the optimal (ψ, α*) pairs to plot in the second panel
    optimal_points = []
    optimal_points_secondary = []
    label_vert = [
        L"\frac{\partial Y(\psi_{\text{low}})}{\partial \alpha}",
        L"\frac{\partial Y(\psi_{\text{med}})}{\partial \alpha}",
        L"\frac{\partial Y(\psi_{\text{high}})}{\partial \alpha}"
    ]
    # Loop through the provided ψ indices to plot multiple MB curves
    if ( ψ_indices_to_vary != nothing ) && ( length(ψ_indices_to_vary) > 0)
        for (i, ψ_idx) in enumerate(ψ_indices_to_vary)
            ψ_val = prim.ψ_grid.values[ψ_idx]
            
            # Calculate the vertical MB line for primary skill level
            marginal_benefit = ModelFunctions.evaluate_derivative(prim.production_function, :α, h_val_fixed, 0.5, ψ_val)
            vlines!(ax1, [marginal_benefit],
                    color = COLORS[i+1],
                    linewidth = 2.5, label = "MB for ψ=$(round(ψ_val, digits=2))")
            
            # Annotate the plot only if showing a single h 
            if isnothing(h_idx_secondary) 
                text!(ax1, 
                    marginal_benefit - 0.03,
                    0.995, text=label_vert[i],
                    fontsize=FONT_SIZE*0.7, 
                    color = COLORS[i+1],
                    align=(:center, :top))
            end
            # Robustly find the optimal α* for primary skill level
            foc_obj = α -> marginal_benefit - mrs_func(α)
            local α_star
            try
                if foc_obj(0.0) <= 0; α_star = 0.0;
                elseif foc_obj(1.0) >= 0; α_star = 1.0;
                else; α_star = find_zero(foc_obj, (0.0, 1.0), Bisection(), atol=1e-4); end
            catch e
                @warn "Error finding α* for primary skill: $e"
                α_star = NaN
            end
            
            # Plot the intersection point and projection line for primary
            if !isnan(α_star)
                mrs_at_star = mrs_func(α_star)
                hlines!(ax1, [α_star], color = :gray, linestyle = :dot, linewidth = 1.5)
                
                # Store the result for Panel 2
                push!(optimal_points, (ψ_val, α_star))
                
                scatter!(ax1, [mrs_at_star], [α_star], 
                    color = COLORS[i+1], markersize = 16, strokewidth = 1, strokecolor = :black)
                # text!(ax1, 
                #     mrs_at_star,
                #     α_star + 0.05, text=L"\alpha^*(\psi_{\text{low}}, h)",
                #     fontsize=FONT_SIZE*0.8, 
                #     color = COLORS[i+1],
                #     align=(:left, :center))
            end

            # Handle secondary skill level if provided
            if h_idx_secondary !== nothing && mrs_func_secondary !== nothing
                # Calculate marginal benefit for secondary skill level
                marginal_benefit_secondary = ModelFunctions.evaluate_derivative(prim.production_function, :α, h_val_secondary, 0.5, ψ_val)
                
                # Vertical line for secondary skill level
                vlines!(ax1, [marginal_benefit_secondary],
                    color = COLORS[i+1], linestyle = :dash, alpha = 0.6,
                    linewidth = 2.5, label = "MB for ψ=$(round(ψ_val, digits=2))")
            

                # Add debug print
                println("Secondary MB for ψ=$(ψ_val): $(marginal_benefit_secondary)")
                
                # Find optimal α* for secondary skill level
                foc_obj_secondary = α -> marginal_benefit_secondary - mrs_func_secondary(α)
                local α_star_secondary
                try
                    if foc_obj_secondary(0.0) <= 0; α_star_secondary = 0.0;
                    elseif foc_obj_secondary(1.0) >= 0; α_star_secondary = 1.0;
                    else; α_star_secondary = find_zero(foc_obj_secondary, (0.0, 1.0), Bisection(), atol=1e-4); end
                    
                    println("Secondary α* for ψ=$(ψ_val): $(α_star_secondary)")
                catch e
                    @warn "Error finding α* for secondary skill: $e"
                    α_star_secondary = NaN
                end
                
                # Plot intersection for secondary (with lower alpha and dashed style)
                if !isnan(α_star_secondary)
                    mrs_at_star_secondary = mrs_func_secondary(α_star_secondary)
                    scatter!(ax1, [mrs_at_star_secondary], [α_star_secondary], 
                        color = (COLORS[i+1], 0.6), markersize = 12, strokewidth = 1, 
                        strokecolor = :black, marker = :diamond)
                    
                    # Store for Panel 2
                    push!(optimal_points_secondary, (ψ_val, α_star_secondary))
                end
            end
        end
    end 
    # Add legend to Panel 1
    # axislegend(ax1, position = :rt)

    # Remove x-ticks for axis 1
    hidexdecorations!(ax1, ticks=true, ticklabels=true, grid=false)

    # --- Panel 2: Optimal Policy Function α*(ψ) ---

    ax2 = create_axis(fig[1, 2],
                    latexstring("Optimal Policy \$\\alpha^*(\\psi)\$"),
                    latexstring("Firm Remote Efficiency \$(\\psi)\$"),
                    "")
    
    # Calculate the full policy curve for the fixed h across all ψ
    ψ_grid_full = prim.ψ_grid.values
    α_policy_curve = res.α_policy[h_idx_fixed, :]

    if plot_policy_curve
        # Plot the continuous policy curve for primary skill level
        lines!(ax2, ψ_grid_full, α_policy_curve, color = COLORS[1], linewidth = 3,
               label = "h=$(round(h_val_fixed, digits=2))")
        
        # Plot policy curve for secondary skill level if provided
        if h_idx_secondary !== nothing
            h_val_secondary = prim.h_grid.values[h_idx_secondary]
            α_policy_curve_secondary = res.α_policy[h_idx_secondary, :]
            lines!(ax2, ψ_grid_full, α_policy_curve_secondary, 
                   color = (COLORS[1], 0.6), linewidth = 3, linestyle = :dash,
                   label = "h=$(round(h_val_secondary, digits=2))")
        end
    end

    # Debug print for optimal points
    println("Primary optimal points: $(length(optimal_points))")
    println("Secondary optimal points: $(length(optimal_points_secondary))")

    # Overlay the scatter points from Panel 1 for primary skill level
    for (i, point) in enumerate(optimal_points)
        ψ_val, α_star = point
        
        # Draw dotted reference lines
        hlines!(ax2, [α_star], color = :gray,  linestyle = :dot, linewidth = 1.5)
        vlines!(ax2, [ψ_val], color = :gray, linestyle = :dot, linewidth = 1.5)

        scatter!(ax2, [ψ_val], [α_star],
                color = COLORS[i+1], markersize = 16, strokewidth = 1, strokecolor = :black)
    end

    # Overlay scatter points for secondary skill level
    if h_idx_secondary !== nothing
        for (i, point) in enumerate(optimal_points_secondary)
            ψ_val, α_star = point
            
            scatter!(ax2, [ψ_val], [α_star],
                    color = (COLORS[i+1], 0.6), markersize = 12, strokewidth = 1, 
                    strokecolor = :black, marker = :diamond)
            
            # Connect primary and secondary points with arrows if they're different
            if i <= length(optimal_points)
                primary_ψ, primary_α = optimal_points[i]
                
                # Only add annotation if the points are different
                if abs(primary_α - α_star) > 1e-6  # Using small epsilon for float comparison
                    annotation!(ax2, primary_ψ, primary_α, ψ_val, α_star,
                        path = Ann.Paths.Line(),
                        style = Ann.Styles.LineArrow(),
                        color = COLORS[i+1],
                        linewidth = 2.5,
                        labelspace = :data
                    )
                end
            end
        end
    end
    
    
    hideydecorations!(ax2)  # Hide y-ticks for the second panel
    # Set custom x-ticks for ax2 to show actual ψ values
    
    ψ_tick_labels = [
        L"\psi_{\text{min}}",  L"\psi_{\text{max}}"
    ]
    ψ_tick_vals = [
        prim.ψ_grid.min, prim.ψ_grid.max
    ]
    if !isempty(ψ_indices_to_vary)
        # Extract ψ values from the indices
        ψ_tick_vals = vcat(
            ψ_tick_vals,
            [prim.ψ_grid.values[idx] for idx in ψ_indices_to_vary]
        )
        ψ_tick_labels = vcat(
            ψ_tick_labels,
            [
                L"\psi_{\text{low}}",
                L"\psi_{\text{med}}",
                L"\psi_{\text{high}}"
            ][1:length(ψ_indices_to_vary)]
        )

        # Set tick colors to match the point colors
        # ax2.xticklabelcolor = [COLORS[i+1] for i in 1:length(ψ_indices_to_vary)]
    end

    # Set the ticks
    ax2.xticks = (ψ_tick_vals, ψ_tick_labels)
    
    # Set the x-axis limits to match the ψ grid
    xlims!(ax2, prim.ψ_grid.min, prim.ψ_grid.max)

    # Link y-axes for consistent scaling of α
    linkyaxes!(ax1, ax2)
    
    return fig
end

"""
    plot_work_arrangement_regimes(results, prim; gray_nonviable=true)

Plots the work arrangement regimes showing regions of full in-person, hybrid, 
and full remote work. Optionally grays out non-viable combinations where S < 0.

# Arguments
- `results::Results`: Model results containing policy functions and surplus
- `prim::Primitives`: Model primitives containing grids
- `gray_nonviable::Bool`: Whether to gray out combinations where S < 0 (default: true)
"""
function plot_work_arrangement_regimes(results, prim; gray_nonviable::Bool=false)
    # Create a figure and axis
    fig = create_figure()
    ax = create_axis(
                        fig[1, 1], 
                        "Work Arrangement Regimes",
                        latexstring("Worker Skill \$(h)\$"),
                        latexstring("Firm Remote Efficiency \$(\\psi)\$")
                    )

    # --- Data Extraction ---
                    
    # Get the y-axis limits from the grid for shading
    ψ_min, ψ_max = prim.ψ_grid.min, prim.ψ_grid.max

    # Get the x-axis limits from the grid
    h_vals = prim.h_grid.values

    # Get the ψ_bottom and ψ_top lines from results
    # Replace Inf with ψ_max and -Inf with ψ_min
    
    ψ_bottom = copy(results.ψ_bottom)
    ψ_bottom[ψ_bottom .> ψ_max] .= ψ_max
    ψ_bottom[ψ_bottom .< ψ_min] .= ψ_min

    ψ_top = results.ψ_top
    ψ_top[ψ_top .> ψ_max] .= ψ_max
    ψ_top[ψ_top .< ψ_min] .= ψ_min

    # --- Shading the Regions ---
    
    # Region 1: Full In-Person (α = 0)
    # This region is from the bottom of the plot up to the ψ_bottom line.
    band!(ax, h_vals, fill(ψ_min, length(h_vals)), ψ_bottom,
        color = (COLORS[1], 0.3), label = latexstring("Full In Person \$(\\alpha=0)\$"))

    # Region 2: Hybrid Work (0 < α < 1)
    # This region is the band between the ψ_bottom and ψ_top lines.
    band!(ax, h_vals, ψ_bottom, ψ_top,
            color = (COLORS[2], 0.3), label = latexstring("Hybrid \$(0 < \\alpha < 1)\$"))

    # Region 3: Full Remote (α = 1)
    # This region is from the ψ_top line up to the top of the plot.
    band!(ax, h_vals, ψ_top, fill(ψ_max, length(h_vals)),
            color = (COLORS[3], 0.3), label = latexstring("Full Remote \$(\\alpha=1)\$"))

    # --- Plotting the Threshold Lines ---
    
    # Plot ψ_bottom(h)
    lines!(ax, h_vals, ψ_bottom, 
            color = COLORS[1], 
            linewidth = 3)

    # Plot ψ_top(h)
    lines!(ax, h_vals, ψ_top, 
            color = COLORS[3], 
            linewidth = 3)

    # --- Gray out non-viable combinations (optional) ---
    if gray_nonviable
        # Create a mask: cells where the surplus S is negative (i.e. workers are not hired)
        gray_mask = map(x -> x < 0 ? 1.0 : NaN, results.S)
        
        # Overlay grey areas on cells where S < 0.
        # Try without transpose first to match grid dimensions
        heatmap!(ax, prim.h_grid.values, prim.ψ_grid.values, gray_mask,
                  colormap = cgrad([:gray]),
                  interpolate = false,
                  transparency = true,
                  alpha = 1.0)
    end

    # --- Final Touches ---
    # Set plot limits to match the grids
    xlims!(ax, prim.h_grid.min, prim.h_grid.max)
    ylims!(ax, prim.ψ_grid.min, prim.ψ_grid.max)
    
    # Add a legend on the lower left corner
    axislegend(ax, position = :lb) # Position legend in the bottom-left
    

    return fig
end

"""
    plot_alpha_policy_by_firm_type(results, prim)

Plots how the optimal remote work share varies with firm efficiency for different worker types.
"""
function plot_alpha_policy_by_firm_type(results, prim)
    # Create a figure and axis
    fig = create_figure()
    ax = create_axis(
        fig[1, 1], 
        "Optimal Remote Share by Firm Efficiency",
        latexstring("Firm Remote Efficiency \$(\\psi)\$"),
        latexstring("Optimal Remote Share \$(\\alpha^*)\$")
    )

    # --- Data Extraction ---
    # Get the grid for the x-axis
    ψ_vals = prim.ψ_grid.values
    
    # Get the full alpha policy matrix
    α_policy = results.α_policy

    # --- Select Representative Worker Types ---
    # Choose three skill levels: low, medium, and high
    n_h = prim.h_grid.n
    h_indices_to_plot = [
        round(Int, n_h * 0.1),  # A low-skill worker (25th percentile)
        round(Int, n_h * 0.50),  # A medium-skill worker (50th percentile)
        round(Int, n_h * 0.9)   # A high-skill worker (75th percentile)
    ]
    
    # --- Plotting Loop ---
    for (i, h_idx) in enumerate(h_indices_to_plot)
        # Get the skill value for the legend
        h_val = prim.h_grid.values[h_idx]
        
        # Extract the corresponding row from the alpha policy matrix.
        # This gives α*(h, ψ) for a fixed h, across all ψ.
        α_slice = α_policy[h_idx, :]
        
        # Create the line plot
        lines!(ax, ψ_vals, α_slice,
            label = "h = $(round(h_val, digits=2))",
            color = COLORS[i], 
            linewidth = 3)
    end
    
    # --- Final Touches ---
    # Set y-axis limits to be between 0 and 1
    ylims!(ax, -0.05, 1.05)
    
    # Add a legend to the plot
    axislegend(ax, position = :lt) # Position legend in the bottom-left

    return fig
end

#==========================================================================================
#? Benchmark Economy Structure and Analysis
==========================================================================================#

"""
    BenchmarkEconomy

Structure to hold all benchmark economy results for comparison and analysis.

# Fields
- `primitives::Primitives`: The parameter structure used for this benchmark
- `results::Results`: The equilibrium results from the model solution
- `aggregate_outcomes::NamedTuple`: Aggregate labor market statistics
- `sorting_measures::NamedTuple`: Measures of worker-firm sorting patterns
- `wage_inequality::NamedTuple`: Wage inequality and dispersion measures
- `work_arrangements::NamedTuple`: Work arrangement shares and statistics
- `timestamp::String`: When this benchmark was computed
- `description::String`: Optional description of this benchmark scenario
"""
struct BenchmarkEconomy
    primitives::Primitives
    results::Results
    aggregate_outcomes::NamedTuple
    sorting_measures::NamedTuple
    wage_inequality::NamedTuple
    work_arrangements::NamedTuple
    timestamp::String
    description::String
end

"""
    BenchmarkEconomy(results, prim; description="")

Constructor to create a BenchmarkEconomy from model results.

# Arguments
- `results::Results`: The equilibrium results from model solution
- `prim::Primitives`: The primitives used to generate these results
- `description::String`: Optional description of this benchmark

# Returns
- `BenchmarkEconomy`: Complete benchmark economy structure
"""
function BenchmarkEconomy(results::Results, prim::Primitives; description::String="Baseline Economy")
    # Compute all outcomes
    outcomes = compute_benchmark_outcomes(results, prim)
    
    return BenchmarkEconomy(
        prim,
        results,
        outcomes.aggregate,
        outcomes.sorting,
        outcomes.wage_inequality,
        outcomes.work_arrangements,
        string(now()),
        description
    )
end

"""
    compute_benchmark_outcomes(results, prim)

Computes key aggregate and distributional outcomes for the benchmark economy.
Returns a NamedTuple with organized benchmark statistics.
"""
function compute_benchmark_outcomes(results::Results, prim::Primitives)
    # --- Aggregate Outcomes ---
    
    # Total unemployment and population
    total_unemployment = sum(results.u)
    total_population = sum(prim.h_grid.pdf)
    aggregate_unemployment_rate = total_unemployment / total_population
    
    # Market tightness and vacancies
    market_tightness = results.θ
    total_vacancies = sum(results.v)
    
    # Average wage (employment-weighted)
    total_employment = sum(results.n)
    average_wage = sum(results.w_policy .* results.n) / total_employment
    
    aggregate_outcomes = (
        unemployment_rate = aggregate_unemployment_rate,
        market_tightness = market_tightness,
        total_vacancies = total_vacancies,
        total_employment = total_employment,
        average_wage = average_wage
    )
    
    # --- Distributional Outcomes: Sorting Measures ---
    sorting_measures = compute_sorting_measures(results, prim)
    
    # --- Wage Inequality Measures ---
    wage_inequality = compute_wage_inequality_measures(results, prim)
    
    # --- Work Arrangement Outcomes ---
    work_arrangements = compute_work_arrangement_outcomes(results, prim, total_employment)
    
    return (
        aggregate = aggregate_outcomes,
        sorting = sorting_measures,
        wage_inequality = wage_inequality,
        work_arrangements = work_arrangements
    )
end

"""
    compute_work_arrangement_outcomes(results, prim, total_employment)

Computes work arrangement statistics.
"""
function compute_work_arrangement_outcomes(results::Results, prim::Primitives, total_employment::Float64)
    # Create masks for different work arrangements
    in_person_mask = results.α_policy .== 0.0
    hybrid_mask = (results.α_policy .> 0.0) .& (results.α_policy .< 1.0)
    full_remote_mask = results.α_policy .== 1.0
    
    # Only count employed workers (where n > 0)
    employed_mask = results.n .> 0
    
    share_in_person = sum(results.n[in_person_mask .& employed_mask]) / total_employment
    share_hybrid = sum(results.n[hybrid_mask .& employed_mask]) / total_employment
    share_full_remote = sum(results.n[full_remote_mask .& employed_mask]) / total_employment
    
    # Average α in the economy (employment-weighted)
    average_alpha = sum(results.α_policy .* results.n) / total_employment
    
    return (
        share_in_person = share_in_person,
        share_hybrid = share_hybrid,
        share_full_remote = share_full_remote,
        average_alpha = average_alpha
    )
end

"""
    compute_sorting_measures(results, prim)

Computes measures of sorting between worker skill and firm efficiency.
"""
function compute_sorting_measures(results::Results, prim::Primitives)
    # Get employed pairs only
    employed_mask = results.n .> 0
    
    # Create vectors of h and ψ values for employed workers, weighted by employment
    h_employed = Float64[]
    ψ_employed = Float64[]
    weights = Float64[]
    
    for i_h in 1:prim.h_grid.n, i_ψ in 1:prim.ψ_grid.n
        if employed_mask[i_h, i_ψ]
            employment_mass = results.n[i_h, i_ψ]
            # Add this many observations (proportional to employment mass)
            n_obs = round(Int, employment_mass * 10000)  # Scale for numerical precision
            append!(h_employed, fill(prim.h_grid.values[i_h], n_obs))
            append!(ψ_employed, fill(prim.ψ_grid.values[i_ψ], n_obs))
            append!(weights, fill(1.0, n_obs))
        end
    end
    
    # Conditional mean E[ψ|h]
    conditional_means = zeros(prim.h_grid.n)
    for i_h in 1:prim.h_grid.n
        total_employment_h = sum(results.n[i_h, :])
        if total_employment_h > 0
            conditional_means[i_h] = sum(prim.ψ_grid.values .* results.n[i_h, :]) / total_employment_h
        end
    end
    
    # Spearman's rank correlation
    spearman_corr = StatsBase.corspearman(h_employed, ψ_employed)
    
    # Mutual information (approximated using binning)
    mutual_info = compute_mutual_information(h_employed, ψ_employed, weights)
    
    return (
        conditional_means = conditional_means,
        spearman_correlation = spearman_corr,
        mutual_information = mutual_info
    )
end

"""
    compute_mutual_information(h_vec, ψ_vec, weights)

Computes mutual information between h and ψ using histogram binning.
"""
function compute_mutual_information(h_vec::Vector{Float64}, ψ_vec::Vector{Float64}, weights::Vector{Float64})
    # Simple binning approach for mutual information
    n_bins = 10
    
    # Create joint histogram
    h_edges = range(minimum(h_vec), maximum(h_vec), length=n_bins+1)
    ψ_edges = range(minimum(ψ_vec), maximum(ψ_vec), length=n_bins+1)
    
    joint_hist = zeros(n_bins, n_bins)
    h_hist = zeros(n_bins)
    ψ_hist = zeros(n_bins)
    
    total_weight = sum(weights)
    
    for (h_val, ψ_val, w) in zip(h_vec, ψ_vec, weights)
        h_bin = min(n_bins, max(1, searchsortedfirst(h_edges[2:end], h_val)))
        ψ_bin = min(n_bins, max(1, searchsortedfirst(ψ_edges[2:end], ψ_val)))
        
        joint_hist[h_bin, ψ_bin] += w
        h_hist[h_bin] += w
        ψ_hist[ψ_bin] += w
    end
    
    # Convert to probabilities
    joint_prob = joint_hist ./ total_weight
    h_prob = h_hist ./ total_weight
    ψ_prob = ψ_hist ./ total_weight
    
    # Compute mutual information
    mi = 0.0
    for i in 1:n_bins, j in 1:n_bins
        if joint_prob[i, j] > 0 && h_prob[i] > 0 && ψ_prob[j] > 0
            mi += joint_prob[i, j] * log(joint_prob[i, j] / (h_prob[i] * ψ_prob[j]))
        end
    end
    
    return mi
end

"""
    compute_wage_inequality_measures(results, prim)

Computes various measures of wage inequality.
"""
function compute_wage_inequality_measures(results::Results, prim::Primitives)
    # Create wage vector weighted by employment
    wages = Float64[]
    employed_mask = results.n .> 0
    
    for i_h in 1:prim.h_grid.n, i_ψ in 1:prim.ψ_grid.n
        if employed_mask[i_h, i_ψ]
            employment_mass = results.n[i_h, i_ψ]
            n_obs = round(Int, employment_mass * 10000)
            append!(wages, fill(results.w_policy[i_h, i_ψ], n_obs))
        end
    end
    
    sort!(wages)
    n_wages = length(wages)
    
    # Percentiles
    p10_idx = max(1, round(Int, 0.1 * n_wages))
    p50_idx = max(1, round(Int, 0.5 * n_wages))
    p90_idx = max(1, round(Int, 0.9 * n_wages))
    
    p10_wage = wages[p10_idx]
    p50_wage = wages[p50_idx]
    p90_wage = wages[p90_idx]
    
    # Percentile ratios
    ratio_90_10 = p90_wage / p10_wage
    ratio_90_50 = p90_wage / p50_wage
    ratio_50_10 = p50_wage / p10_wage
    
    # Gini coefficient
    gini = compute_gini_coefficient(wages)
    
    # Skill premium (highest vs lowest skill workers)
    skill_premium = compute_skill_premium(results, prim, employed_mask)
    
    # Within-group inequality (wage dispersion within same skill level)
    within_group_90_10 = compute_within_group_inequality(results, prim, employed_mask)
    
    return (
        gini_coefficient = gini,
        percentile_90_10 = ratio_90_10,
        percentile_90_50 = ratio_90_50,
        percentile_50_10 = ratio_50_10,
        skill_premium = skill_premium,
        within_group_90_10 = within_group_90_10
    )
end

"""
    compute_skill_premium(results, prim, employed_mask)

Computes skill premium as ratio of top 10% to bottom 10% skill workers' wages.
"""
function compute_skill_premium(results::Results, prim::Primitives, employed_mask::BitMatrix)
    highest_skill_wages = Float64[]
    lowest_skill_wages = Float64[]
    
    # Top 10% and bottom 10% of skill distribution
    top_skill_cutoff = round(Int, 0.9 * prim.h_grid.n)
    bottom_skill_cutoff = round(Int, 0.1 * prim.h_grid.n)
    
    for i_h in 1:bottom_skill_cutoff, i_ψ in 1:prim.ψ_grid.n
        if employed_mask[i_h, i_ψ]
            employment_mass = results.n[i_h, i_ψ]
            n_obs = round(Int, employment_mass * 10000)
            append!(lowest_skill_wages, fill(results.w_policy[i_h, i_ψ], n_obs))
        end
    end
    
    for i_h in top_skill_cutoff:prim.h_grid.n, i_ψ in 1:prim.ψ_grid.n
        if employed_mask[i_h, i_ψ]
            employment_mass = results.n[i_h, i_ψ]
            n_obs = round(Int, employment_mass * 10000)
            append!(highest_skill_wages, fill(results.w_policy[i_h, i_ψ], n_obs))
        end
    end
    
    return mean(highest_skill_wages) / mean(lowest_skill_wages)
end

"""
    compute_gini_coefficient(wages)

Computes the Gini coefficient for a wage distribution.
"""
function compute_gini_coefficient(wages::Vector{Float64})
    n = length(wages)
    sorted_wages = sort(wages)
    
    # Gini formula: G = (2 * sum(i * w_i)) / (n * sum(w_i)) - (n+1)/n
    numerator = sum(i * sorted_wages[i] for i in 1:n)
    denominator = n * sum(sorted_wages)
    
    return (2 * numerator) / denominator - (n + 1) / n
end

"""
    compute_within_group_inequality(results, prim, employed_mask)

Computes within-group wage inequality (90/10 ratio within skill groups).
"""
function compute_within_group_inequality(results::Results, prim::Primitives, employed_mask::BitMatrix)
    within_group_ratios = Float64[]
    
    for i_h in 1:prim.h_grid.n
        # Get wages for this skill level
        skill_wages = Float64[]
        for i_ψ in 1:prim.ψ_grid.n
            if employed_mask[i_h, i_ψ]
                employment_mass = results.n[i_h, i_ψ]
                n_obs = round(Int, employment_mass * 1000)
                append!(skill_wages, fill(results.w_policy[i_h, i_ψ], n_obs))
            end
        end
        
        if length(skill_wages) >= 10  # Need minimum observations
            sort!(skill_wages)
            n_wages = length(skill_wages)
            p10_idx = max(1, round(Int, 0.1 * n_wages))
            p90_idx = max(1, round(Int, 0.9 * n_wages))
            
            ratio = skill_wages[p90_idx] / skill_wages[p10_idx]
            push!(within_group_ratios, ratio)
        end
    end
    
    return mean(within_group_ratios)
end

"""
    plot_benchmark_summary(benchmark::BenchmarkEconomy)

Creates a comprehensive summary plot of benchmark economy outcomes.
"""
function plot_benchmark_summary(benchmark::BenchmarkEconomy)
    fig = create_figure(type="ultra")
    
    # Create a grid layout for multiple panels
    # Panel 1: Aggregate outcomes (text summary)
    ax1 = Axis(fig[1, 1], title="Aggregate Outcomes")
    hidedecorations!(ax1)
    hidespines!(ax1)
    
    agg_text = """
    Unemployment Rate: $(round(benchmark.aggregate_outcomes.unemployment_rate * 100, digits=2))%
    Market Tightness: $(round(benchmark.aggregate_outcomes.market_tightness, digits=3))
    Average Wage: $(round(benchmark.aggregate_outcomes.average_wage, digits=2))
    Average Remote Share: $(round(benchmark.work_arrangements.average_alpha, digits=3))
    """
    
    text!(ax1, 0.1, 0.5, text=agg_text, fontsize=FONT_SIZE)
    
    # Panel 2: Work arrangement shares
    ax2 = create_axis(fig[1, 2], "Work Arrangement Shares", "Arrangement", "Share")
    
    arrangements = ["In-Person", "Hybrid", "Full Remote"]
    shares = [benchmark.work_arrangements.share_in_person, 
              benchmark.work_arrangements.share_hybrid, 
              benchmark.work_arrangements.share_full_remote]
    
    barplot!(ax2, 1:3, shares, color=COLORS[1:3])
    ax2.xticks = (1:3, arrangements)
    
    # Panel 3: Sorting measures - plot ranks instead of levels
    ax3 = create_axis(fig[2, 1], 
                        latexstring("Sorting (\$\\mathbb{E}[\\psi \\mid h] \$)"),
                        "Worker Skill Percentile",
                        "Mean Firm Efficiency Percentile"
                    )
    
    # Convert conditional means to ranks
    worker_percentiles = (1:benchmark.primitives.h_grid.n) ./ benchmark.primitives.h_grid.n * 100
    
    # Convert conditional means to firm efficiency percentiles
    firm_percentiles = zeros(benchmark.primitives.h_grid.n)
    for i_h in 1:benchmark.primitives.h_grid.n
        conditional_mean_psi = benchmark.sorting_measures.conditional_means[i_h]
        # Find what percentile this conditional mean corresponds to in the ψ distribution
        rank = sum(benchmark.primitives.ψ_grid.values .<= conditional_mean_psi) / benchmark.primitives.ψ_grid.n
        firm_percentiles[i_h] = rank * 100
    end
    
    lines!(ax3, worker_percentiles, firm_percentiles, color=COLORS[1], linewidth=3)
    # Add 45-degree line for reference (perfect positive sorting)
    lines!(ax3, [0, 100], [0, 100], color=:gray, linestyle=:dash, linewidth=2)
    
    # Panel 4: Inequality summary
    ax4 = Axis(fig[2, 2], title="Inequality Measures")
    hidedecorations!(ax4)
    hidespines!(ax4)
    
    ineq_text = """
    Gini Coefficient: $(round(benchmark.wage_inequality.gini_coefficient, digits=3))
    90/10 Ratio: $(round(benchmark.wage_inequality.percentile_90_10, digits=2))
    Skill Premium: $(round(benchmark.wage_inequality.skill_premium, digits=2))
    Spearman Correlation: $(round(benchmark.sorting_measures.spearman_correlation, digits=3))
    """
    
    text!(ax4, 0.1, 0.5, text=ineq_text, fontsize=FONT_SIZE)
    
    return fig
end

#==========================================================================================
#? Stylized Facts Analysis  
==========================================================================================#

"""
    create_simulated_dataset(benchmark::BenchmarkEconomy)

Creates a simulated dataset from a BenchmarkEconomy for regression analysis.
"""
function create_simulated_dataset(benchmark::BenchmarkEconomy)
    return create_simulated_dataset(benchmark.results, benchmark.primitives)
end

"""
    create_simulated_dataset(results, prim)

Creates a simulated dataset for regression analysis, with each observation 
representing a (h, ψ) pair weighted by employment mass.
"""
function create_simulated_dataset(results::Results, prim::Primitives)
    # Initialize vectors
    h_vec = Float64[]
    ψ_vec = Float64[]
    wage_vec = Float64[]
    alpha_vec = Float64[]
    remote_dummy_vec = Int[]
    weights_vec = Float64[]
    
    employed_mask = results.n .> 0
    
    for i_h in 1:prim.h_grid.n, i_ψ in 1:prim.ψ_grid.n
        if employed_mask[i_h, i_ψ]
            employment_mass = results.n[i_h, i_ψ]
            
            # Each observation represents this employment mass
            push!(h_vec, prim.h_grid.values[i_h])
            push!(ψ_vec, prim.ψ_grid.values[i_ψ])
            push!(wage_vec, results.w_policy[i_h, i_ψ])
            push!(alpha_vec, results.α_policy[i_h, i_ψ])
            push!(remote_dummy_vec, results.α_policy[i_h, i_ψ] > 0 ? 1 : 0)
            push!(weights_vec, employment_mass)
        end
    end
    
    return (
        h = h_vec,
        ψ = ψ_vec,
        wage = wage_vec,
        log_wage = log.(wage_vec),
        alpha = alpha_vec,
        remote_dummy = remote_dummy_vec,
        weights = weights_vec
    )
end

"""
    run_wage_regressions(dataset)

Runs wage regressions to demonstrate stylized facts.
Returns regression results as NamedTuple.
"""
function run_wage_regressions(dataset::NamedTuple)
    # Weighted correlations
    total_weight = sum(dataset.weights)
    weighted_mean_log_wage = sum(dataset.log_wage .* dataset.weights) / total_weight
    weighted_mean_ψ = sum(dataset.ψ .* dataset.weights) / total_weight
    
    # Simple correlation between log wage and ψ
    corr_wage_psi = cor(hcat(dataset.log_wage, dataset.ψ), StatsBase.AnalyticWeights(dataset.weights))[1, 2]
    
    # Remote work premium (simple comparison)
    remote_wages = dataset.log_wage[dataset.remote_dummy .== 1]
    remote_weights = dataset.weights[dataset.remote_dummy .== 1]
    inperson_wages = dataset.log_wage[dataset.remote_dummy .== 0]
    inperson_weights = dataset.weights[dataset.remote_dummy .== 0]
    
    avg_remote_wage = sum(remote_wages .* remote_weights) / sum(remote_weights)
    avg_inperson_wage = sum(inperson_wages .* inperson_weights) / sum(inperson_weights)
    
    remote_premium = avg_remote_wage - avg_inperson_wage
    
    return (
        correlation_wage_psi = corr_wage_psi,
        remote_wage_premium = remote_premium,
        avg_remote_wage = exp(avg_remote_wage),
        avg_inperson_wage = exp(avg_inperson_wage)
    )
end

"""
    plot_remote_wage_correlations(benchmark::BenchmarkEconomy; dataset=nothing)

Creates plots demonstrating Stylized Fact I: Remote work correlates with higher wages.
"""
function plot_remote_wage_correlations(benchmark::BenchmarkEconomy; dataset=nothing)
    return plot_remote_wage_correlations(benchmark.results, benchmark.primitives; dataset=dataset)
end

"""
    plot_remote_wage_correlations(results, prim; dataset=nothing)

Creates plots demonstrating Stylized Fact I: Remote work correlates with higher wages.
"""
function plot_remote_wage_correlations(results::Results, prim::Primitives; dataset=nothing)
    if dataset === nothing
        dataset = create_simulated_dataset(results, prim)
    end
    
    fig = create_figure(type="wide")
    
    # Panel 1: Wage vs. Teleworkability (Firm Efficiency ψ)
    ax1 = create_axis(fig[1, 1], 
                    "Wage vs. Teleworkability", 
                    latexstring("Firm Remote Efficiency \$(\\psi)\$"), 
                    "Average Wage")
    
    # Bin ψ values and compute conditional means
    n_bins = 10
    ψ_bins = range(minimum(dataset.ψ), maximum(dataset.ψ), length=n_bins+1)
    conditional_wages = Float64[]
    bin_centers = Float64[]
    
    for i in 1:n_bins
        mask = (dataset.ψ .>= ψ_bins[i]) .& (dataset.ψ .< ψ_bins[i+1])
        if sum(mask) > 0
            weights_in_bin = dataset.weights[mask]
            wages_in_bin = dataset.wage[mask]
            avg_wage = sum(wages_in_bin .* weights_in_bin) / sum(weights_in_bin)
            push!(conditional_wages, avg_wage)
            push!(bin_centers, (ψ_bins[i] + ψ_bins[i+1]) / 2)
        end
    end
    
    lines!(ax1, bin_centers, conditional_wages, color=COLORS[1], linewidth=3)
    scatter!(ax1, bin_centers, conditional_wages, color=COLORS[1], markersize=8)
    
    # Panel 2: Remote Wage Premium Bar Chart
    ax2 = create_axis(fig[1, 2], 
                    "Remote Wage Premium", 
                    "Work Arrangement", 
                    "Average Wage")
    
    regression_results = run_wage_regressions(dataset)
    
    avg_wages = [regression_results.avg_inperson_wage, regression_results.avg_remote_wage]
    arrangement_labels = ["In-Person", "Hybrid/Remote"]
    
    barplot!(ax2, 1:2, avg_wages, color=[COLORS[1], COLORS[2]])
    ax2.xticks = (1:2, arrangement_labels)
    
    # Add premium annotation
    premium_pct = (regression_results.avg_remote_wage / regression_results.avg_inperson_wage - 1) * 100
    text!(ax2, 1.5, max(avg_wages...) * 1.1, 
            text="Premium: $(round(premium_pct, digits=1))%", 
            align=(:center, :bottom))
    
    return fig
end

"""
    plot_within_occupation_analysis(benchmark::BenchmarkEconomy; dataset=nothing)

Creates plots demonstrating Stylized Fact II: Within occupations, remote workers earn more.
"""
function plot_within_occupation_analysis(benchmark::BenchmarkEconomy; dataset=nothing)
    return plot_within_occupation_analysis(benchmark.results, benchmark.primitives; dataset=dataset)
end

"""
    plot_within_occupation_analysis(results, prim; dataset=nothing)

Creates plots demonstrating Stylized Fact II: Within occupations, remote workers earn more.
"""
function plot_within_occupation_analysis(results::Results, prim::Primitives; dataset=nothing)
    if dataset === nothing
        dataset = create_simulated_dataset(results, prim)
    end
    
    fig = create_figure(type="ultrawide")
    
    employed_mask = results.n .> 0
    
    # Convert percentiles to actual ψ indices
    n_ψ = prim.ψ_grid.n
    ψ_indices = [max(1, min(n_ψ, round(Int, p * n_ψ))) for p in [0.25, 0.5, 0.75]]
    
    within_occupation_gaps = Float64[]
    skill_differences = Float64[]
    occupation_labels = String[]
    
    for (i, ψ_idx) in enumerate(ψ_indices)
        ψ_val = prim.ψ_grid.values[ψ_idx]
        
        # Get all matches for this occupation (ψ level)
        inperson_wages = Float64[]
        inperson_skills = Float64[]
        inperson_weights = Float64[]
        
        remote_wages = Float64[]
        remote_skills = Float64[]
        remote_weights = Float64[]
        
        for i_h in 1:prim.h_grid.n
            if employed_mask[i_h, ψ_idx]
                employment_mass = results.n[i_h, ψ_idx]
                wage = results.w_policy[i_h, ψ_idx]
                skill = prim.h_grid.values[i_h]
                alpha = results.α_policy[i_h, ψ_idx]
                
                if alpha == 0.0  # In-person
                    push!(inperson_wages, wage)
                    push!(inperson_skills, skill)
                    push!(inperson_weights, employment_mass)
                else  # Remote/Hybrid
                    push!(remote_wages, wage)
                    push!(remote_skills, skill)
                    push!(remote_weights, employment_mass)
                end
            end
        end
        
        # Calculate averages if both groups exist
        if length(inperson_wages) > 0 && length(remote_wages) > 0
            avg_inperson_wage = sum(inperson_wages .* inperson_weights) / sum(inperson_weights)
            avg_remote_wage = sum(remote_wages .* remote_weights) / sum(remote_weights)
            
            avg_inperson_skill = sum(inperson_skills .* inperson_weights) / sum(inperson_weights)
            avg_remote_skill = sum(remote_skills .* remote_weights) / sum(remote_weights)
            
            wage_gap = (avg_remote_wage / avg_inperson_wage - 1) * 100
            skill_diff = avg_remote_skill - avg_inperson_skill
            
            push!(within_occupation_gaps, wage_gap)
            push!(skill_differences, skill_diff)
            push!(occupation_labels, "ψ=$(round(ψ_val, digits=2))")
        end
    end
    
    # Panel 1: Within-occupation wage gaps
    ax1 = create_axis(fig[1, 1], 
                      "Within-Occupation Remote Wage Premium", 
                      "Occupation (ψ level)", 
                      "Wage Premium (%)")
    
    barplot!(ax1, 1:length(within_occupation_gaps), within_occupation_gaps, 
             color=COLORS[2])
    ax1.xticks = (1:length(occupation_labels), occupation_labels)
    
    # Panel 2: Skill differences explaining the gap
    ax2 = create_axis(fig[1, 2], 
                      "Skill Differences Within Occupation", 
                      "Occupation (ψ level)", 
                      "Skill Difference (Remote - In-Person)")
    
    barplot!(ax2, 1:length(skill_differences), skill_differences, 
             color=COLORS[3])
    ax2.xticks = (1:length(occupation_labels), occupation_labels)
    
    # Panel 3: Scatter plot showing relationship
    ax3 = create_axis(fig[1, 3], 
                      "Wage Premium vs. Skill Difference", 
                      "Skill Difference", 
                      "Wage Premium (%)")
    
    scatter!(ax3, skill_differences, within_occupation_gaps, 
             color=COLORS[1], markersize=12)
    
    # Add trend line if we have enough points
    if length(skill_differences) >= 3
        # Simple linear fit
        X = hcat(ones(length(skill_differences)), skill_differences)
        β = X \ within_occupation_gaps
        x_trend = range(minimum(skill_differences), maximum(skill_differences), length=100)
        y_trend = β[1] .+ β[2] .* x_trend
        lines!(ax3, x_trend, y_trend, color=COLORS[2], linestyle=:dash, linewidth=2)
    end
    
    return fig
end

end # module ModelPlotting
