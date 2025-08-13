#==========================================================================================
Module: ComparativeStaticsPlotting.jl
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2025-02-27
Description: Plotting functions for comparative statics analysis results.
        Creates visualizations showing how model outcomes vary with parameters.
==========================================================================================#

module ComparativeStaticsPlotting

using CairoMakie, LaTeXStrings, DataFrames, Statistics
using ..ComparativeStatics

export plot_comparative_statics_summary, plot_parameter_elasticities,
       plot_outcome_vs_parameter, plot_comparative_statics_grid,
       save_comparative_plots

#==========================================================================================
#? Plotting Configuration
==========================================================================================#

# Color palette and styling (consistent with ModelPlotting)
const COLORS = [
    "#23373B",   # primary
    "#EB811B",   # secondary  
    "#14B03E",   # tertiary
    "#F2C94C",   # highlight
    "#8B5CF6",   # purple
    "#EF4444",   # red
]

const FONT_CHOICE = "CMU Serif"
const FONT_SIZE = 18

function create_cs_figure(;type="normal")
    if type == "wide"
        size = (1400, 600)
    elseif type == "tall"
        size = (700, 1000)
    elseif type == "grid"
        size = (1200, 1000)
    else
        size = (800, 600)
    end
    
    return Figure(
        size = size,
        fontsize = FONT_SIZE,
        fonts = (; regular=FONT_CHOICE, italic=FONT_CHOICE, bold=FONT_CHOICE)
    )
end

function create_cs_axis(where_, title, xlabel, ylabel)
    return Axis(
        where_,
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
        xticklabelsize = FONT_SIZE-2,
        yticklabelsize = FONT_SIZE-2,
        xgridvisible = true,
        ygridvisible = true,
        xgridcolor = (:gray, 0.3),
        ygridcolor = (:gray, 0.3)
    )
end

#==========================================================================================
#? Main Plotting Functions
==========================================================================================#

"""
    plot_comparative_statics_summary(results::ComparativeStaticsResults)

Creates a comprehensive summary plot showing all key outcomes vs the parameter.
"""
function plot_comparative_statics_summary(results::ComparativeStaticsResults)
    
    # Filter to successful runs
    df = results.summary_stats[results.summary_stats.convergence_flag, :]
    if nrow(df) == 0
        error("No successful runs to plot")
    end
    
    param_name = results.config.parameter_grid.parameter_name
    param_desc = results.config.parameter_grid.description
    
    # Create figure with 2x3 grid
    fig = create_cs_figure(type="grid")
    
    # 1. Market Tightness
    ax1 = create_cs_axis(fig[1,1], "Market Tightness", param_desc, "θ")
    lines!(ax1, df.parameter_value, df.market_tightness, 
           color=COLORS[1], linewidth=3)
    scatter!(ax1, df.parameter_value, df.market_tightness,
            color=COLORS[1], markersize=8)
    
    # 2. Job Finding Rate  
    ax2 = create_cs_axis(fig[1,2], "Job Finding Rate", param_desc, "p")
    lines!(ax2, df.parameter_value, df.job_finding_rate,
           color=COLORS[2], linewidth=3)
    scatter!(ax2, df.parameter_value, df.job_finding_rate,
            color=COLORS[2], markersize=8)
    
    # 3. Unemployment Rate
    ax3 = create_cs_axis(fig[1,3], "Unemployment Rate", param_desc, "Unemployment Rate")
    lines!(ax3, df.parameter_value, df.unemployment_rate .* 100,
           color=COLORS[3], linewidth=3)
    scatter!(ax3, df.parameter_value, df.unemployment_rate .* 100,
            color=COLORS[3], markersize=8)
    
    # 4. Average Remote Share (if available)
    if !all(isnan.(df.avg_remote_share))
        ax4 = create_cs_axis(fig[2,1], "Average Remote Share", param_desc, "α")
        valid_remote = .!isnan.(df.avg_remote_share)
        lines!(ax4, df.parameter_value[valid_remote], df.avg_remote_share[valid_remote],
               color=COLORS[4], linewidth=3)
        scatter!(ax4, df.parameter_value[valid_remote], df.avg_remote_share[valid_remote],
                color=COLORS[4], markersize=8)
    end
    
    # 5. Average Wage (if available)
    if !all(isnan.(df.avg_wage))
        ax5 = create_cs_axis(fig[2,2], "Average Wage", param_desc, "w")
        valid_wage = .!isnan.(df.avg_wage)
        lines!(ax5, df.parameter_value[valid_wage], df.avg_wage[valid_wage],
               color=COLORS[5], linewidth=3)
        scatter!(ax5, df.parameter_value[valid_wage], df.avg_wage[valid_wage],
                color=COLORS[5], markersize=8)
    end
    
    # 6. Total Surplus (if available)
    if !all(isnan.(df.total_surplus))
        ax6 = create_cs_axis(fig[2,3], "Total Surplus", param_desc, "S")
        valid_surplus = .!isnan.(df.total_surplus)
        lines!(ax6, df.parameter_value[valid_surplus], df.total_surplus[valid_surplus],
               color=COLORS[6], linewidth=3)
        scatter!(ax6, df.parameter_value[valid_surplus], df.total_surplus[valid_surplus],
                color=COLORS[6], markersize=8)
    end
    
    # Add overall title
    Label(fig[0, :], "Comparative Statics: $(param_desc) ($(param_name))",
          fontsize=FONT_SIZE+4, font=:bold)
    
    return fig
end

"""
    plot_outcome_vs_parameter(results::ComparativeStaticsResults, outcome::String; kwargs...)

Plots a single outcome variable against the parameter with confidence intervals if multiple runs.
"""
function plot_outcome_vs_parameter(results::ComparativeStaticsResults, outcome::String;
                                  title::String="", show_trend::Bool=true)
    
    df = results.summary_stats[results.summary_stats.convergence_flag, :]
    if nrow(df) == 0
        error("No successful runs to plot")
    end
    
    param_name = results.config.parameter_grid.parameter_name
    param_desc = results.config.parameter_grid.description
    
    # Get outcome data
    if !hasproperty(df, Symbol(outcome))
        error("Outcome '$outcome' not found in results")
    end
    
    y_data = getproperty(df, Symbol(outcome))
    
    # Handle missing values
    valid_idx = .!isnan.(y_data)
    x_data = df.parameter_value[valid_idx]
    y_data = y_data[valid_idx]
    
    if length(y_data) == 0
        error("No valid data for outcome '$outcome'")
    end
    
    # Create figure
    fig = create_cs_figure()
    
    # Outcome-specific formatting
    ylabel_text, y_transform = get_outcome_formatting(outcome)
    y_plot = y_transform.(y_data)
    
    if isempty(title)
        title = "$(ylabel_text) vs $(param_desc)"
    end
    
    ax = create_cs_axis(fig[1,1], title, param_desc, ylabel_text)
    
    # Main plot
    lines!(ax, x_data, y_plot, color=COLORS[1], linewidth=3)
    scatter!(ax, x_data, y_plot, color=COLORS[1], markersize=10)
    
    # Add trend line if requested
    if show_trend && length(x_data) > 2
        # Fit linear trend
        X = [ones(length(x_data)) x_data]
        β = X \ y_plot
        trend_y = X * β
        
        lines!(ax, x_data, trend_y, color=COLORS[2], linestyle=:dash, linewidth=2, label="Linear Trend")
        
        # Calculate R²
        ss_res = sum((y_plot .- trend_y).^2)
        ss_tot = sum((y_plot .- mean(y_plot)).^2)
        r_squared = 1 - ss_res / ss_tot
        
        # Add R² to plot
        text!(ax, 0.05, 0.95, text="R² = $(round(r_squared, digits=3))", space=:relative, fontsize=FONT_SIZE-2)
    end
    
    return fig
end

"""
    plot_parameter_elasticities(results_list::Vector{ComparativeStaticsResults})

Plots elasticities across multiple parameters for comparison.
"""
function plot_parameter_elasticities(results_list::Vector{ComparativeStaticsResults})
    
    # Extract elasticities for each parameter
    param_names = String[]
    unemployment_elasticities = Float64[]
    remote_elasticities = Float64[]
    
    for results in results_list
        summary = ComparativeStatics.extract_summary_statistics(results)
        
        push!(param_names, results.config.parameter_grid.parameter_name)
        push!(unemployment_elasticities, get(summary, "unemployment_elasticity", NaN))
        push!(remote_elasticities, get(summary, "remote_elasticity", NaN))
    end
    
    # Create figure
    fig = create_cs_figure(type="wide")
    
    # Unemployment elasticities
    ax1 = create_cs_axis(fig[1,1], "Parameter Elasticities", "Parameter", "Unemployment Elasticity")
    
    valid_unemployment = .!isnan.(unemployment_elasticities)
    if any(valid_unemployment)
        barplot!(ax1, (1:length(param_names))[valid_unemployment], 
                unemployment_elasticities[valid_unemployment],
                color=COLORS[1])
        ax1.xticks = (1:length(param_names), param_names)
        ax1.xticklabelrotation = π/4
    end
    
    # Remote work elasticities  
    ax2 = create_cs_axis(fig[1,2], "", "Parameter", "Remote Work Elasticity")
    
    valid_remote = .!isnan.(remote_elasticities)
    if any(valid_remote)
        barplot!(ax2, (1:length(param_names))[valid_remote],
                remote_elasticities[valid_remote],
                color=COLORS[2])
        ax2.xticks = (1:length(param_names), param_names)
        ax2.xticklabelrotation = π/4
    end
    
    # Add horizontal line at zero
    hlines!(ax1, [0], color=:black, linestyle=:dash, alpha=0.5)
    hlines!(ax2, [0], color=:black, linestyle=:dash, alpha=0.5)
    
    return fig
end

"""
    plot_comparative_statics_grid(results_dict::Dict{String, ComparativeStaticsResults})

Creates a grid plot comparing multiple parameters and outcomes.
"""
function plot_comparative_statics_grid(results_dict::Dict{String, ComparativeStaticsResults})
    
    param_names = collect(keys(results_dict))
    n_params = length(param_names)
    
    # Key outcomes to plot
    outcomes = ["unemployment_rate", "avg_remote_share", "avg_wage"]
    outcome_labels = ["Unemployment Rate (%)", "Avg Remote Share", "Avg Wage"]
    
    # Create figure
    fig = Figure(size=(400*n_params, 300*length(outcomes)), fontsize=FONT_SIZE)
    
    for (i, outcome) in enumerate(outcomes)
        for (j, param_name) in enumerate(param_names)
            results = results_dict[param_name]
            df = results.summary_stats[results.summary_stats.convergence_flag, :]
            
            if nrow(df) > 0 && hasproperty(df, Symbol(outcome))
                ax = Axis(fig[i, j], 
                         title = j == 1 ? outcome_labels[i] : "",
                         xlabel = i == length(outcomes) ? param_name : "",
                         xgridvisible = true, ygridvisible = true)
                
                y_data = getproperty(df, Symbol(outcome))
                if outcome == "unemployment_rate"
                    y_data = y_data .* 100  # Convert to percentage
                end
                
                valid_idx = .!isnan.(y_data)
                if any(valid_idx)
                    lines!(ax, df.parameter_value[valid_idx], y_data[valid_idx],
                           color=COLORS[mod(j-1, length(COLORS))+1], linewidth=2)
                    scatter!(ax, df.parameter_value[valid_idx], y_data[valid_idx],
                            color=COLORS[mod(j-1, length(COLORS))+1], markersize=6)
                end
            end
        end
    end
    
    # Add parameter names as column labels
    for (j, param_name) in enumerate(param_names)
        Label(fig[0, j], param_name, fontsize=FONT_SIZE+2, font=:bold)
    end
    
    return fig
end

#==========================================================================================
#? Utility Functions
==========================================================================================#

"""
    get_outcome_formatting(outcome::String)

Returns appropriate ylabel and transformation function for different outcomes.
"""
function get_outcome_formatting(outcome::String)
    outcome_formats = Dict(
        "unemployment_rate" => ("Unemployment Rate (%)", x -> x * 100),
        "market_tightness" => ("Market Tightness (θ)", x -> x),
        "job_finding_rate" => ("Job Finding Rate (p)", x -> x),
        "vacancy_filling_rate" => ("Vacancy Filling Rate (q)", x -> x),
        "avg_remote_share" => ("Average Remote Share (α)", x -> x),
        "avg_wage" => ("Average Wage (w)", x -> x),
        "total_surplus" => ("Total Surplus (S)", x -> x)
    )
    
    if haskey(outcome_formats, outcome)
        return outcome_formats[outcome]
    else
        return (outcome, x -> x)
    end
end

"""
    save_comparative_plots(results::ComparativeStaticsResults, output_dir::String="")

Saves all comparative statics plots for a given results object.
"""
function save_comparative_plots(results::ComparativeStaticsResults, output_dir::String="")
    
    if isempty(output_dir)
        output_dir = joinpath(results.config.output_dir, "plots")
    end
    
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    param_name = results.config.parameter_grid.parameter_name
    
    try
        # 1. Summary plot
        fig_summary = plot_comparative_statics_summary(results)
        save(joinpath(output_dir, "$(param_name)_summary.png"), fig_summary)
        
        # 2. Individual outcome plots
        outcomes = ["unemployment_rate", "market_tightness", "avg_remote_share", "avg_wage"]
        
        for outcome in outcomes
            try
                fig = plot_outcome_vs_parameter(results, outcome)
                save(joinpath(output_dir, "$(param_name)_$(outcome).png"), fig)
            catch e
                @warn "Failed to create plot for $outcome: $e"
            end
        end
        
        println("Plots saved to: $output_dir")
        
    catch e
        @warn "Error saving plots: $e"
    end
end

end # module ComparativeStaticsPlotting
