# Plotting utilities for counterfactual analysis
# src/structural_model_heterogenous_preferences/counterfactuals/plotting_utils.jl

using Plots, PlotlyJS
plotlyjs()

"""
Create visualization for the decomposition results.
"""
function plot_decomposition_results(results_df; save_path=nothing)
    # Create a grouped bar plot
    outcomes = results_df.Outcome
    n_outcomes = length(outcomes)
    
    # Set up the plot
    p = plot(
        title="Decomposition of Changes (2019-2024)",
        xlabel="Outcome Variable",
        ylabel="Change",
        legend=:topright,
        size=(800, 600)
    )
    
    x_pos = 1:n_outcomes
    bar_width = 0.25
    
    # Plot bars for each component
    bar!(p, x_pos .- bar_width, results_df.TotalChange, 
         label="Total Change", alpha=0.8, color=:blue, width=bar_width)
    bar!(p, x_pos, results_df.DueToPreferences, 
         label="Due to Preferences", alpha=0.8, color=:red, width=bar_width)
    bar!(p, x_pos .+ bar_width, results_df.DueToTechnology, 
         label="Due to Technology", alpha=0.8, color=:green, width=bar_width)
    
    # Customize x-axis
    plot!(p, xticks=(x_pos, outcomes), xrotation=45)
    
    # Add horizontal line at zero
    hline!(p, [0], color=:black, linestyle=:dash, alpha=0.5, label="")
    
    if save_path !== nothing
        savefig(p, save_path)
        println("Decomposition plot saved to: $save_path")
    end
    
    return p
end

"""
Create heatmap for complementarity analysis results.
"""
function plot_complementarity_heatmap(results_df, outcome_var; save_path=nothing)
    # Reshape data for heatmap
    phi_vals = sort(unique(results_df.phi))
    nu_vals = sort(unique(results_df.nu))
    
    # Create matrix for the outcome variable
    outcome_matrix = zeros(length(nu_vals), length(phi_vals))
    
    for (i, nu_val) in enumerate(nu_vals)
        for (j, phi_val) in enumerate(phi_vals)
            row_idx = findfirst((results_df.phi .== phi_val) .& (results_df.nu .== nu_val))
            if row_idx !== nothing
                outcome_matrix[i, j] = results_df[row_idx, outcome_var]
            else
                outcome_matrix[i, j] = NaN
            end
        end
    end
    
    # Create heatmap
    p = heatmap(
        phi_vals, nu_vals, outcome_matrix,
        title="$outcome_var across ϕ-ν Grid",
        xlabel="ϕ (Complementarity Parameter)",
        ylabel="ν (Technology Parameter)",
        color=:viridis,
        size=(700, 500)
    )
    
    # Add contour lines
    contour!(p, phi_vals, nu_vals, outcome_matrix, 
             levels=10, color=:white, alpha=0.6, linewidth=1)
    
    if save_path !== nothing
        savefig(p, save_path)
        println("Complementarity heatmap saved to: $save_path")
    end
    
    return p
end

"""
Create line plot for RTO mandate results.
"""
function plot_rto_results(results_df; save_path=nothing)
    # Filter out NaN values
    valid_rows = .!isnan.(results_df.mean_alpha)
    clean_df = results_df[valid_rows, :]
    
    # Create subplots
    p1 = plot(
        clean_df.alpha_max, clean_df.mean_alpha,
        title="Mean Remote Work Share",
        xlabel="Maximum α Allowed",
        ylabel="E[α]",
        marker=:circle,
        linewidth=2,
        markersize=6,
        color=:blue,
        legend=false
    )
    
    p2 = plot(
        clean_df.alpha_max, clean_df.change_agg_productivity,
        title="Change in Aggregate Productivity",
        xlabel="Maximum α Allowed",
        ylabel="Δ Productivity",
        marker=:circle,
        linewidth=2,
        markersize=6,
        color=:red,
        legend=false
    )
    
    p3 = plot(
        clean_df.alpha_max, clean_df.change_var_logwage,
        title="Change in Wage Inequality",
        xlabel="Maximum α Allowed",
        ylabel="Δ Var(log w)",
        marker=:circle,
        linewidth=2,
        markersize=6,
        color=:green,
        legend=false
    )
    
    # Combine plots
    p = plot(p1, p2, p3, layout=(1, 3), size=(1200, 400))
    
    if save_path !== nothing
        savefig(p, save_path)
        println("RTO results plot saved to: $save_path")
    end
    
    return p
end

"""
Generate all plots for the counterfactual analysis.
"""
function generate_all_plots(results_dir)
    println("\n" * "-"^60)
    println("GENERATING PLOTS")
    
    # Create plots directory if it doesn't exist
    plots_dir = joinpath(results_dir, "plots")
    mkpath(plots_dir)
    
    try
        # Load decomposition results and plot
        decomp_file = joinpath(results_dir, "decomposition_results.csv")
        if isfile(decomp_file)
            decomp_df = CSV.read(decomp_file, DataFrame)
            plot_decomposition_results(decomp_df; 
                save_path=joinpath(plots_dir, "decomposition_results.png"))
        end
        
        # Load complementarity results and plot
        comp_file = joinpath(results_dir, "complementarity_results.csv")
        if isfile(comp_file)
            comp_df = CSV.read(comp_file, DataFrame)
            plot_complementarity_heatmap(comp_df, :mean_alpha; 
                save_path=joinpath(plots_dir, "complementarity_mean_alpha.png"))
            plot_complementarity_heatmap(comp_df, :agg_productivity; 
                save_path=joinpath(plots_dir, "complementarity_productivity.png"))
        end
        
        # Load RTO results and plot
        rto_file = joinpath(results_dir, "rto_results.csv")
        if isfile(rto_file)
            rto_df = CSV.read(rto_file, DataFrame)
            plot_rto_results(rto_df; 
                save_path=joinpath(plots_dir, "rto_mandate_results.png"))
        end
        
        println("All plots generated successfully!")
        
    catch e
        println("Error generating plots: $e")
    end
end
