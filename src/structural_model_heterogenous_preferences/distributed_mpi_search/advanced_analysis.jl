#!/usr/bin/env julia

# Advanced MPI Search Analysis with Makie
# Creates publication-quality diagnostic plots and comprehensive analysis

using Pkg
Pkg.activate("/project/high_tech_ind/searching-flexibility")

using JSON3, YAML
using CairoMakie
using Statistics, StatsBase, LinearAlgebra
using Printf, Dates
using DataFrames, CSV
using KernelDensity
using Random; Random.seed!(123)  # For reproducibility

# Parse command line arguments for job ID
JOB_ID = if length(ARGS) >= 1
    strip(ARGS[1])  # Job ID as string
else
    nothing  # Analyze all jobs if no arguments
end

println("🎨 Advanced MPI Search Analysis with Makie")
println("=" ^ 50)
if JOB_ID !== nothing
    println("📊 Analyzing job ID: $JOB_ID")
else
    println("📊 Analyzing all available jobs")
end

# Set Makie theme for publication quality
set_theme!(Theme(
    fontsize=12,
    Axis=(
        titlefont="TeX Gyre Heros Bold",
        labelfont="TeX Gyre Heros",
        ticklabelfont="TeX Gyre Heros",
        spinewidth=1.5,
        xtickwidth=1.5,
        ytickwidth=1.5,
    ),
    Legend=(
        framewidth=1.5,
        labelfont="TeX Gyre Heros",
    )
))

# Configuration
const SCRIPT_DIR = dirname(@__FILE__)
const RESULTS_DIR = joinpath(SCRIPT_DIR, "output")
const OUTPUT_DIR = if JOB_ID !== nothing
    "figures/mpi_analysis/job_$JOB_ID"
else
    "figures/mpi_analysis"
end
const PLOTS_DIR = OUTPUT_DIR

# Create output directories
mkpath(OUTPUT_DIR)
mkpath(PLOTS_DIR)

"""Find and load the most recent MPI search results for specified job"""
function load_latest_results()
    try
        # Filter files based on job ID if specified
        if JOB_ID !== nothing
            pattern = "mpi_search_results_job$(JOB_ID)_"
            files = filter(f -> startswith(f, pattern) && endswith(f, ".json"), 
                            readdir(RESULTS_DIR))
            if isempty(files)
                error("No MPI search results found for job $JOB_ID!")
            end
        else
            files = filter(f -> startswith(f, "mpi_search_results_") && endswith(f, ".json"), 
                            readdir(RESULTS_DIR))
            if isempty(files)
                error("No MPI search results found!")
            end
        end
        
        # Sort by modification time, most recent first
        full_paths = [joinpath(RESULTS_DIR, f) for f in files]
        sorted_files = sort(full_paths, by=mtime, rev=true)
        
        # Try files in order until we find one that parses successfully
        for file_path in sorted_files
            try
                println("📁 Attempting to load: $(basename(file_path))")
                
                content = read(file_path, String)
                results = JSON3.read(content)
                
                println("✅ Successfully loaded: $(basename(file_path))")
                return results, file_path
            catch json_error
                @warn "Failed to parse $(basename(file_path)): $json_error"
                continue
            end
        end
        
        error("No valid JSON files found")
    catch e
        error("Failed to load results: $e")
    end
end

"""Display top N candidates leaderboard"""
function create_top_candidates_analysis(results, save_prefix::String, top_n::Int=5)
    if !haskey(results, "all_params") || !haskey(results, "all_objectives")
        @warn "Missing data for top candidates analysis"
        return
    end
    
    all_params = results["all_params"]
    all_objectives = results["all_objectives"]
    param_names = results["parameter_names"]
    
    # Create pairs and filter valid results
    objective_param_pairs = [(obj, params) for (obj, params) in zip(all_objectives, all_params) if obj < 1e8]
    
    if length(objective_param_pairs) < top_n
        @warn "Not enough valid results for top $top_n analysis"
        return
    end
    
    # Sort by objective value (best first)
    sorted_pairs = sort(objective_param_pairs, by=x->x[1])
    top_candidates = sorted_pairs[1:min(top_n, length(sorted_pairs))]
    
    println("\n" * "="^80)
    println("🏆 TOP $top_n CANDIDATES LEADERBOARD")
    println("="^80)
    
    # Create DataFrame for top candidates
    df_data = Dict{String, Any}()
    df_data["Rank"] = Int[]
    df_data["Objective"] = Float64[]
    df_data["Improvement_Percent"] = Float64[]
    
    # Initialize parameter columns
    for param_name in param_names
        df_data[string(param_name)] = Float64[]
    end
    
    best_obj = top_candidates[1][1]
    
    for (i, (obj, params)) in enumerate(top_candidates)
        improvement = i == 1 ? 0.0 : ((best_obj - obj) / abs(best_obj)) * 100
        
        push!(df_data["Rank"], i)
        push!(df_data["Objective"], obj)
        push!(df_data["Improvement_Percent"], improvement)
        
        # Add parameter values using indices
        for (param_idx, param_name) in enumerate(param_names)
            param_val = params[param_idx]
            push!(df_data[string(param_name)], param_val)
        end
    end
    
    # Create DataFrame
    df = DataFrame(df_data)
    
    # Save to CSV
    csv_path = joinpath(PLOTS_DIR, "$(save_prefix)/top_candidates.csv")
    CSV.write(csv_path, df)
    println("💾 Top candidates saved to: $csv_path")
    
    # Display summary
    println(df)
    
    # Create visual comparison plot
    fig = Figure(size=(1400, 1000))
    
    # Objective values comparison
    ax1 = Axis(fig[1, 1],
               title="Top $top_n Candidates - Objective Values",
               xlabel="Rank",
               ylabel="Objective Value")
    
    objectives_top = [pair[1] for pair in top_candidates]
    barplot!(ax1, 1:length(objectives_top), objectives_top, 
             color=range(colorant"darkgreen", colorant"lightgreen", length=length(objectives_top)))
    
    # Parameter comparison for top candidates
    n_params = length(param_names)
    n_cols = min(3, n_params)
    n_rows = ceil(Int, n_params / n_cols)
    
    for (param_idx, param_name) in enumerate(param_names)
        row = 1 + ceil(Int, param_idx / n_cols)
        col = ((param_idx - 1) % n_cols) + 1
        
        if row <= 3  # Limit to avoid overcrowding
            ax = Axis(fig[row, col],
                     title="$param_name - Top $top_n",
                     xlabel="Rank",
                     ylabel="Parameter Value")
            
            param_values = [pair[2][param_idx] for pair in top_candidates]
            barplot!(ax, 1:length(param_values), param_values,
                    color=range(colorant"darkblue", colorant"lightblue", length=length(param_values)))
            
            # Add value labels on bars
            for (i, val) in enumerate(param_values)
                text!(ax, i, val + 0.02 * (maximum(param_values) - minimum(param_values)), 
                     text=@sprintf("%.3f", val), align=(:center, :bottom), fontsize=10)
            end
        end
    end
    
    plot_path = joinpath(PLOTS_DIR, "$(save_prefix)/top_candidates.png")
    save(plot_path, fig)
    println("📊 Top candidates analysis saved: $plot_path")
    
    return top_candidates
end

"""Create parameter trajectories for best candidates evolution"""
function create_parameter_trajectories(results, save_prefix::String)
    if !haskey(results, "all_params") || !haskey(results, "all_objectives")
        @warn "Missing data for parameter trajectories"
        return
    end
    
    all_params = results["all_params"]
    all_objectives = results["all_objectives"]
    param_names = results["parameter_names"]
    
    # Find indices where new best was found
    best_indices = Int[]
    best_objectives = Float64[]
    best_params_evolution = []
    
    current_best = Inf
    for (i, (obj, params)) in enumerate(zip(all_objectives, all_params))
        if obj < current_best && obj < 1e8
            current_best = obj
            push!(best_indices, i)
            push!(best_objectives, obj)
            push!(best_params_evolution, params)
        end
    end
    
    if length(best_indices) < 3
        @warn "Not enough improvements for trajectory analysis"
        return
    end
    
    println("📈 Creating parameter trajectory analysis for $(length(best_indices)) improvements...")
    
    # Create trajectory plots
    n_params = length(param_names)
    n_cols = 3
    n_rows = ceil(Int, n_params / n_cols)
    
    fig = Figure(size=(1400, 300 * n_rows))
    
    for (param_idx, param_name) in enumerate(param_names)
        row = ceil(Int, param_idx / n_cols)
        col = ((param_idx - 1) % n_cols) + 1
        
        ax = Axis(fig[row, col],
                 title="Best $param_name Evolution",
                 xlabel="Evaluation Number",
                 ylabel=string(param_name))
        
        # Extract parameter values for this parameter
        param_values = [params[param_idx] for params in best_params_evolution]
        
        # Plot trajectory
        lines!(ax, best_indices, param_values, color=:steelblue, linewidth=3)
        scatter!(ax, best_indices, param_values, color=:red, markersize=6)
        
        # Add trend analysis
        if length(param_values) > 4
            # Calculate moving average
            window_size = max(2, length(param_values) ÷ 3)
            if window_size < length(param_values)
                moving_avg = [mean(param_values[max(1,i-window_size+1):i]) for i in window_size:length(param_values)]
                avg_indices = best_indices[window_size:end]
                lines!(ax, avg_indices, moving_avg, color=:orange, linewidth=2, linestyle=:dash)
            end
        end
        
        # Add annotations for significant changes
        if length(param_values) > 1
            changes = abs.(diff(param_values))
            significant_changes = findall(changes .> 0.1 * std(param_values))
            if !isempty(significant_changes) && length(significant_changes) < 5
                for change_idx in significant_changes
                    scatter!(ax, [best_indices[change_idx + 1]], [param_values[change_idx + 1]], 
                            color=:yellow, markersize=12, marker=:star5, strokewidth=2, strokecolor=:black)
                end
            end
        end
    end
    
    # Add overall convergence info
    Label(fig[0, :], text="Parameter Evolution During $(length(best_indices)) Improvements | " *
                          "Best: $(round(minimum(best_objectives), digits=8)) | " *
                          "Span: Eval $(minimum(best_indices)) → $(maximum(best_indices))",
          fontsize=14)
    
    plot_path = joinpath(PLOTS_DIR, "$(save_prefix)/parameter_trajectories.png")
    save(plot_path, fig)
    println("📊 Parameter trajectories saved: $plot_path")
end

"""Create moment trajectories for best candidates evolution"""
function create_moment_trajectories(results, save_prefix::String)
    if !haskey(results, "all_moments") || !haskey(results, "all_objectives")
        @warn "Missing moments data for moment trajectories"
        return
    end
    
    all_moments = results["all_moments"]
    all_objectives = results["all_objectives"]
    
    # Find indices where new best was found
    best_indices = Int[]
    best_objectives = Float64[]
    best_moments_evolution = []
    
    current_best = Inf
    for (i, (obj, moments)) in enumerate(zip(all_objectives, all_moments))
        if obj < current_best && obj < 1e8
            current_best = obj
            push!(best_indices, i)
            push!(best_objectives, obj)
            push!(best_moments_evolution, moments)
        end
    end
    
    if length(best_indices) < 3
        @warn "Not enough improvements for moment trajectory analysis"
        return
    end
    
    # Get moment names from the first valid moments dict
    moment_names = String[]
    for moments in best_moments_evolution
        if !isempty(moments)
            moment_names = collect(keys(moments))
            break
        end
    end
    
    if isempty(moment_names)
        @warn "No valid moment names found"
        return
    end
    
    println("📈 Creating moment trajectory analysis for $(length(best_indices)) improvements...")
    
    # Create trajectory plots
    n_moments = length(moment_names)
    n_cols = 3
    n_rows = ceil(Int, n_moments / n_cols)
    
    fig = Figure(size=(1400, 300 * n_rows))
    
    for (moment_idx, moment_name) in enumerate(moment_names)
        row = ceil(Int, moment_idx / n_cols)
        col = ((moment_idx - 1) % n_cols) + 1
        
        ax = Axis(fig[row, col],
                 title="Best $moment_name Evolution",
                 xlabel="Evaluation Number",
                 ylabel=string(moment_name))
        
        # Extract moment values for this moment
        moment_values = Float64[]
        valid_indices = Int[]
        
        for (i, moments) in enumerate(best_moments_evolution)
            if haskey(moments, moment_name) && isfinite(moments[moment_name])
                push!(moment_values, moments[moment_name])
                push!(valid_indices, best_indices[i])
            end
        end
        
        if length(moment_values) < 2
            text!(ax, 0.5, 0.5, text="Insufficient data", align=(:center, :center), space=:relative)
            continue
        end
        
        # Plot trajectory
        lines!(ax, valid_indices, moment_values, color=:steelblue, linewidth=3)
        scatter!(ax, valid_indices, moment_values, color=:red, markersize=6)
        
        # Add trend analysis
        if length(moment_values) > 4
            # Calculate moving average
            window_size = max(2, length(moment_values) ÷ 3)
            if window_size < length(moment_values)
                moving_avg = [mean(moment_values[max(1,i-window_size+1):i]) for i in window_size:length(moment_values)]
                avg_indices = valid_indices[window_size:end]
                lines!(ax, avg_indices, moving_avg, color=:orange, linewidth=2, linestyle=:dash)
            end
        end
        
        # Add annotations for significant changes
        if length(moment_values) > 1
            changes = abs.(diff(moment_values))
            if std(moment_values) > 0
                significant_changes = findall(changes .> 0.1 * std(moment_values))
                if !isempty(significant_changes) && length(significant_changes) < 5
                    for change_idx in significant_changes
                        if change_idx + 1 <= length(valid_indices)
                            scatter!(ax, [valid_indices[change_idx + 1]], [moment_values[change_idx + 1]], 
                                    color=:yellow, markersize=12, marker=:star5, strokewidth=2, strokecolor=:black)
                        end
                    end
                end
            end
        end
    end
    
    # Add overall convergence info
    Label(fig[0, :], text="Moment Evolution During $(length(best_indices)) Improvements | " *
                          "Best Objective: $(round(minimum(best_objectives), digits=8)) | " *
                          "Span: Eval $(minimum(valid_indices)) → $(maximum(valid_indices))",
          fontsize=14)
    
    plot_path = joinpath(PLOTS_DIR, "$(save_prefix)/moment_trajectories.png")
    save(plot_path, fig)
    println("📊 Moment trajectories saved: $plot_path")
end

"""Create moment sensitivity analysis plots"""
function create_moment_sensitivity(results, save_prefix::String)
    if !haskey(results, "all_moments") || !haskey(results, "all_params") || !haskey(results, "all_objectives")
        @warn "Missing data for moment sensitivity analysis"
        return
    end
    
    all_moments = results["all_moments"]
    all_params = results["all_params"]
    all_objectives = results["all_objectives"]
    param_names = results["parameter_names"]
    
    # Filter for valid data (convergent solutions)
    valid_mask = [obj < 1e8 for obj in all_objectives]
    valid_moments = all_moments[valid_mask]
    valid_params = all_params[valid_mask]
    
    if length(valid_moments) < 10
        @warn "Insufficient valid data for moment sensitivity analysis"
        return
    end
    
    # Get moment names from the first valid moments dict
    moment_names = String[]
    for moments in valid_moments
        if !isempty(moments) && all(isfinite(v) for v in values(moments))
            moment_names = collect(keys(moments))
            break
        end
    end
    
    if isempty(moment_names)
        @warn "No valid moment names found for sensitivity analysis"
        return
    end
    
    println("📊 Creating moment sensitivity analysis...")
    
    # Create correlation matrix between parameters and moments
    n_params = length(param_names)
    n_moments = length(moment_names)
    
    # Prepare data matrices
    param_matrix = hcat([Float64[params[i] for params in valid_params] for i in 1:n_params]...)
    moment_matrix = zeros(length(valid_moments), n_moments)
    
    # Fill moment matrix
    for (i, moments) in enumerate(valid_moments)
        for (j, moment_name) in enumerate(moment_names)
            if haskey(moments, moment_name) && isfinite(moments[moment_name])
                moment_matrix[i, j] = moments[moment_name]
            else
                moment_matrix[i, j] = NaN
            end
        end
    end
    
    # Calculate correlations
    correlations = zeros(n_params, n_moments)
    for i in 1:n_params
        for j in 1:n_moments
            valid_both = .!isnan.(moment_matrix[:, j])
            if sum(valid_both) > 5
                correlations[i, j] = cor(param_matrix[valid_both, i], moment_matrix[valid_both, j])
            else
                correlations[i, j] = NaN
            end
        end
    end
    
    # Create heatmap
    fig = Figure(size=(max(400, 100 * n_moments), max(300, 50 * n_params)))
    ax = Axis(fig[1, 1],
             title="Parameter-Moment Sensitivity Analysis",
             xlabel="Target Moments",
             ylabel="Parameters")
    
    # Replace NaN with 0 for visualization
    correlations_viz = replace(correlations, NaN => 0.0)
    
    hm = heatmap!(ax, correlations_viz, colormap=:RdBu, colorrange=(-1, 1))
    
    # Set ticks and labels
    ax.xticks = (1:n_moments, moment_names)
    ax.yticks = (1:n_params, string.(param_names))
    ax.xticklabelrotation = π/4
    
    # Add colorbar
    Colorbar(fig[1, 2], hm, label="Correlation Coefficient")
    
    # Add correlation values as text
    for i in 1:n_params
        for j in 1:n_moments
            if !isnan(correlations[i, j])
                text!(ax, j, i, text=@sprintf("%.2f", correlations[i, j]),
                     align=(:center, :center),
                     color=abs(correlations[i, j]) > 0.5 ? :white : :black,
                     fontsize=10)
            end
        end
    end
    
    plot_path = joinpath(PLOTS_DIR, "$(save_prefix)/moment_sensitivity.png")
    save(plot_path, fig)
    println("📊 Moment sensitivity saved: $plot_path")
end

"""Analyze moment mismatch for the best candidate - TEMPORARILY DISABLED"""
#TODO:  Fix best candidates function
function analyze_best_candidate_moments(results, save_prefix::String)
    println("⚠️  Moment mismatch analysis temporarily disabled due to dependency issues")

    # Original function commented out to avoid PrettyTables/Crayons issues
    if !haskey(results, "best_params") || !haskey(results, "best_objective")
        @warn "Missing best candidate data for moment analysis"
        return
    end
    
    best_params = results["best_params"]
    best_objective = results["best_objective"]
    
    println("\n" * "="^80)
    println("🔬 MOMENT MISMATCH ANALYSIS - BEST CANDIDATE")
    println("="^80)
    println("🏆 Best Objective Value: $(round(best_objective, digits=8))")
    
    # Try to load the model and compute moments
    # This requires the model functions to be available
    try
        # Include necessary model files (adjust paths as needed)
        include("../ModelSetup.jl")  # Adjust path to your model setup
        include("../ModelSolver.jl")  # Adjust path to your model setup
        include("../ModelEstimation.jl")  # Adjust path to your model setup

        basic_configuration = "../model_parameters.yaml"

        println("📊 Re-running model with best parameters...")
        
        # Extract parameter values
        param_dict = Dict()
        for (key, value) in best_params
            param_dict[Symbol(key)] = value
        end
        
        # Initialize primitives with basic configuration
        prim, res = initializeModel(basic_configuration)
        prim_new, res_new = update_primitives_results(prim, res, param_dict)
        # Solve the model
        solve_model(prim_new, res_new, config = basic_configuration)

        # Compute model moments
        model_moments = compute_model_moments(prim, res)
        
        # Load target moments from data
        #! HARDCODED FIX!!!!!!
        #TODO: Fix this mess
        target_moments = Dict(
            :mean_logwage =>  2.650321,
            :var_logwage =>  0.110030,
            :mean_alpha =>  0.012160,
            :var_alpha =>  0.011152,
            :inperson_share =>  0.984192,
            :hybrid_share =>  0.005423,
            :remote_share =>  0.010385,
            :agg_productivity =>  1.148100,
            :mean_alpha_lowpsi =>  0.012160,
            :market_tightness =>  1.100000
        )
            
        
        # Create comparison table
        moment_names = collect(keys(target_moments))
        table_data = []
        
        println("\n📋 MOMENT COMPARISON TABLE:")
        println("-" * 60)
        
        header = ["Moment", "Data Value", "Model Value", "Absolute Diff", "% Difference"]
        
        total_sse = 0.0
        max_pct_diff = 0.0
        worst_moment = ""
        
        for moment_name in moment_names
            data_val = target_moments[moment_name]
            model_val = get(model_moments, moment_name, NaN)
            
            if !isnan(model_val) && data_val != 0
                abs_diff = abs(model_val - data_val)
                pct_diff = abs_diff / abs(data_val) * 100
                
                total_sse += abs_diff^2
                
                if pct_diff > max_pct_diff
                    max_pct_diff = pct_diff
                    worst_moment = string(moment_name)
                end
                
                push!(table_data, [
                    string(moment_name),
                    @sprintf("%.6f", data_val),
                    @sprintf("%.6f", model_val),
                    @sprintf("%.6f", abs_diff),
                    @sprintf("%.2f%%", pct_diff)
                ])
            else
                push!(table_data, [
                    string(moment_name),
                    @sprintf("%.6f", data_val),
                    "NaN",
                    "NaN",
                    "NaN"
                ])
            end
        end
        
        # Print formatted table
        # TODO either fix or replace with a CSV save 
        # pretty_table(table_data, header=header, 
        #             crop=:none,
        #             alignment=[:l, :r, :r, :r, :r],
        #             header_crayon=crayon"bold blue",
        #             subheader_crayon=crayon"green")
        
        println("\n📊 MOMENT ANALYSIS SUMMARY:")
        println("• Total SSE: $(round(total_sse, digits=8))")
        println("• RMSE: $(round(sqrt(total_sse/length(moment_names)), digits=8))")
        println("• Worst Moment: $worst_moment ($(round(max_pct_diff, digits=2))% error)")
        
        # Create visual moment comparison
        fig = Figure(size=(1200, 800))
        
        # Extract numeric values for plotting
        data_values = [target_moments[name] for name in moment_names]
        model_values = [get(model_moments, name, NaN) for name in moment_names]
        valid_mask = .!isnan.(model_values)
        
        if sum(valid_mask) > 0
            # Moment comparison scatter plot
            ax1 = Axis(fig[1, 1],
                      title="Model vs Data Moments",
                      xlabel="Data Value",
                      ylabel="Model Value")
            
            data_vals_clean = data_values[valid_mask]
            model_vals_clean = model_values[valid_mask]
            
            scatter!(ax1, data_vals_clean, model_vals_clean, 
                    color=:steelblue, markersize=10, alpha=0.7)
            
            # Add 45-degree line
            min_val = min(minimum(data_vals_clean), minimum(model_vals_clean))
            max_val = max(maximum(data_vals_clean), maximum(model_vals_clean))
            lines!(ax1, [min_val, max_val], [min_val, max_val], 
                  color=:red, linestyle=:dash, linewidth=2)
            
            # Add moment labels
            moment_names_clean = moment_names[valid_mask]
            for (i, name) in enumerate(moment_names_clean)
                text!(ax1, data_vals_clean[i], model_vals_clean[i], 
                     text=string(name), offset=(5, 5), fontsize=8)
            end
            
            # Percentage errors bar plot
            ax2 = Axis(fig[1, 2],
                      title="Percentage Errors by Moment",
                      xlabel="Moment",
                      ylabel="Absolute % Error")
            
            pct_errors = [abs(model_vals_clean[i] - data_vals_clean[i]) / abs(data_vals_clean[i]) * 100 
                         for i in 1:length(data_vals_clean)]
            
            barplot!(ax2, 1:length(pct_errors), pct_errors,
                    color=ifelse.(pct_errors .> 10, :red, :steelblue))
            
            ax2.xticks = (1:length(moment_names_clean), string.(moment_names_clean))
            ax2.xticklabelrotation = π/4
            
            # Add horizontal line at 5% and 10% error
            hlines!(ax2, [5, 10], color=[:orange, :red], linestyle=:dash, linewidth=2)
            
            # Residuals plot
            ax3 = Axis(fig[2, 1:2],
                      title="Moment Residuals (Model - Data)",
                      xlabel="Moment",
                      ylabel="Residual")
            
            residuals = model_vals_clean .- data_vals_clean
            barplot!(ax3, 1:length(residuals), residuals,
                    color=ifelse.(residuals .> 0, :red, :blue))
            
            ax3.xticks = (1:length(moment_names_clean), string.(moment_names_clean))
            ax3.xticklabelrotation = π/4
            hlines!(ax3, [0], color=:black, linewidth=2)
        end
        
        plot_path = joinpath(PLOTS_DIR, "$(save_prefix)/moment_analysis.png")
        save(plot_path, fig)
        println("📊 Moment analysis saved: $plot_path")
        
    catch e
        @warn "Could not perform moment analysis: $e"
        println("⚠️  Moment analysis requires model functions to be available")
        println("   Make sure the model setup files are properly included")
        
        # Fallback: just show what we know from the objective
        println("\n📊 OBJECTIVE BREAKDOWN (if available):")
        if haskey(results, "objective_components")
            components = results["objective_components"]
            for (component, value) in components
                println("• $component: $(round(value, digits=6))")
            end
        else
            println("• Total Objective: $(round(best_objective, digits=8))")
            println("• (Individual moment mismatches not available)")
        end
    end
end

"""Create comprehensive convergence analysis"""
function create_convergence_analysis(results, save_prefix)
    if !haskey(results, "all_objectives")
        @warn "No objective values found for convergence analysis"
        return
    end
    
    objectives = collect(results["all_objectives"])
    n_evals = length(objectives)

    # Subsample for plotting if too many evaluations
    subsample_stride = n_evals > 2000 ? max(1, n_evals ÷ 1000) : 1
    plot_indices = 1:subsample_stride:n_evals

    # Vectorized best-so-far (cumulative minimum)
    best_history = accumulate(min, objectives)

    # Subsampled best history for plotting
    best_history_plot = best_history[plot_indices]
    plot_eval_nums = collect(plot_indices)
    
    # Create figure with multiple subplots
    fig = Figure(size=(1400, 1000))
    
    # Main convergence plot
    ax1 = Axis(fig[1, 1:2], 
               title="Convergence History",
               xlabel="Evaluation Number",
               ylabel="Best Objective Value")

    lines!(ax1, plot_eval_nums, best_history_plot, color=:steelblue, linewidth=3)
    if length(plot_eval_nums) < 2000
        scatter!(ax1, plot_eval_nums, best_history_plot, color=:steelblue, markersize=4, alpha=0.7)
    end

    # Add improvement phases (subsampled)
    if n_evals > 100
        phases = [1, n_evals÷4, n_evals÷2, 3*n_evals÷4, n_evals]
        colors = [:red, :orange, :green, :purple]
        for i in 1:length(phases)-1
            start_idx, end_idx = phases[i], phases[i+1]
            idxs = plot_indices[(plot_indices .>= start_idx) .& (plot_indices .<= end_idx)]
            if !isempty(idxs)
                lines!(ax1, idxs, best_history[idxs], color=colors[i], linewidth=2, alpha=0.8)
            end
        end
    end
    
    # Objective distribution
    ax2 = Axis(fig[1, 3], 
               title="Objective Distribution",
               xlabel="Objective Value",
               ylabel="Density")
    
    # Filter out penalty values for distribution
    valid_objectives = objectives[objectives .< 1e8]
    if !isempty(valid_objectives)
        hist!(ax2, valid_objectives, bins=30, color=(:steelblue, 0.7), normalization=:pdf)
        
        # Add KDE if enough points
        if length(valid_objectives) > 20
            kde_result = kde(valid_objectives)
            lines!(ax2, kde_result.x, kde_result.density, color=:red, linewidth=3)
        end
    end
    
    # Convergence rate analysis
    ax3 = Axis(fig[2, 1], 
               title="Convergence Rate",
               xlabel="Evaluation Number",
               ylabel="Improvement Rate")
    
    if length(best_history) > 10
        # Calculate improvement rate (negative of derivative)
        window_size = max(10, n_evals ÷ 50)
        improvement_rate = Float64[]
        eval_nums = Float64[]
        
        for i in window_size:n_evals-window_size
            start_val = best_history[i-window_size+1]
            end_val = best_history[i+window_size]
            rate = (start_val - end_val) / (2 * window_size)  # Improvement per evaluation
            push!(improvement_rate, rate)
            push!(eval_nums, i)
        end
        
        lines!(ax3, eval_nums, improvement_rate, color=:darkgreen, linewidth=2)
        hlines!(ax3, [0], color=:red, linestyle=:dash, linewidth=2)
    end
    
    # Success rate over time
    ax4 = Axis(fig[2, 2], 
               title="Success Rate",
               xlabel="Evaluation Number",
               ylabel="Success Rate (%)")
    
    # Calculate success rate (non-penalty values)
    window_size = max(50, n_evals ÷ 20)
    success_rates = Float64[]
    eval_centers = Float64[]
    
    for i in window_size:window_size:n_evals
        start_idx = max(1, i - window_size + 1)
        end_idx = min(n_evals, i)
        window_objectives = objectives[start_idx:end_idx]
        success_rate = 100 * sum(window_objectives .< 1e8) / length(window_objectives)
        push!(success_rates, success_rate)
        push!(eval_centers, (start_idx + end_idx) / 2)
    end
    
    lines!(ax4, eval_centers, success_rates, color=:purple, linewidth=3)
    scatter!(ax4, eval_centers, success_rates, color=:purple, markersize=6)
    ylims!(ax4, 0, 100)
    
    # Parameter space exploration
    ax5 = Axis(fig[2, 3], 
            title="Exploration Progress",
            xlabel="Evaluation Number", 
            ylabel="Parameter Space Coverage")
    
    # # if haskey(results, "all_params") && length(results["all_params"]) > 10
    #     # Calculate cumulative parameter space coverage (subsampled)
    #     all_params = results["all_params"]
    #     n_params =  length(all_params[1])

    #     # Subsample for coverage calculation
    #     coverage_stride = n_evals > 2000 ? max(10, n_evals ÷ 200) : 1
    #     coverage_indices = 10:coverage_stride:n_evals
    #     coverage_history = Float64[]
    #     # for i in coverage_indices
    #         i = coverage_indices[100]
    #         sample_params = all_params[1:i]
    #         n_small = 5000
    #         if length(sample_params) > n_small
    #             # we take a random sample of the sample_params of size n_small
    #             subsample_indices = sample(1:length(sample_params), n_small, replace=false)
    #             sample_params = sample_params[subsample_indices]
    #             sample_params = hcat(all_params...)'
    #         end
    # #         try
    #             cov_matrix = cov(sample_params)
    #             coverage = sqrt(det(cov_matrix + 1e-10 * I))  # Add small regularization
    # #             push!(coverage_history, coverage)
    # #         catch
    # #             push!(coverage_history, 0.0)
    # #         end
    # #     end

    # #     if !isempty(coverage_history)
    # #         lines!(ax5, coverage_indices, coverage_history, color=:darkorange, linewidth=3)
    # #     end
    # # end
    
    # Add overall statistics
    if haskey(results, "best_objective")
        best_obj = results["best_objective"]
        total_time = get(results, "elapsed_time", 0)
        n_workers = get(results, "n_workers", 1)
        
        Label(fig[0, :], text="Best Objective: $(round(best_obj, digits=6)) | " *
                              "Total Time: $(round(total_time/60, digits=1)) min | " *
                              "Workers: $n_workers | " *
                              "Evaluations: $n_evals",
              fontsize=14)
    end
    
    plot_path = joinpath(PLOTS_DIR, "$(save_prefix)/convergence_analysis.png")
    save(plot_path, fig)
    println("📊 Convergence analysis saved: $plot_path")
end

"""Create parameter correlation and sensitivity analysis"""
function create_parameter_analysis(results, save_prefix::String)
    if !haskey(results, "all_params") || !haskey(results, "all_objectives")
        @warn "Missing parameter data for analysis"
        return
    end
    
    param_names = results["parameter_names"]
    all_params = results["all_params"]
    all_objectives = results["all_objectives"]

    # Randomly subsample vectors if we have more than that
    n_select = 5000
    n_total = length(all_params)
    if n_total > n_select
        subsample_indices = sample(1:n_total, n_select, replace=false)
        all_params = all_params[subsample_indices]
        all_objectives = all_objectives[subsample_indices]
        println("📊 Subsampled $(length(subsample_indices)) parameters from $n_total for analysis")
    end

    # Convert to matrix and filter valid results
    param_matrix = hcat(all_params...)'
    objectives = collect(all_objectives)
    valid_mask = objectives .< 1e8
    
    if sum(valid_mask) < 10
        @warn "Too few valid results for parameter analysis"
        return
    end
    
    param_matrix_clean = param_matrix[valid_mask, :]
    objectives_clean = objectives[valid_mask]
    n_params = length(param_names)
    
    # Create correlation heatmap
    fig1 = Figure(size=(800, 700))
    ax1 = Axis(fig1[1, 1], 
               title="Parameter Correlation Matrix",
               xlabel="Parameters",
               ylabel="Parameters")
    
    # Calculate correlation matrix
    corr_matrix = cor(param_matrix_clean)
    
    # Create heatmap
    hm = heatmap!(ax1, corr_matrix, colormap=:RdBu, colorrange=(-1, 1))
    
    # Add parameter labels
    ax1.xticks = (1:n_params, string.(param_names))
    ax1.yticks = (1:n_params, string.(param_names))
    ax1.xticklabelrotation = π/4
    
    Colorbar(fig1[1, 2], hm, label="Correlation")
    
    plot_path1 = joinpath(PLOTS_DIR, "$(save_prefix)/parameter_correlations.png")
    save(plot_path1, fig1)
    println("📊 Parameter correlations saved: $plot_path1")
    
    # Parameter sensitivity analysis
    fig2 = Figure(size=(1200, 800))
    
    # Calculate correlations with objective
    obj_correlations = [abs(cor(param_matrix_clean[:, i], objectives_clean)) for i in 1:n_params]
    sorted_indices = sortperm(obj_correlations, rev=true)
    
    # Top parameters subplot
    top_n = min(6, n_params)
    n_cols = 3
    n_rows = 2
    
    for (plot_idx, param_idx) in enumerate(sorted_indices[1:top_n])
        row = ceil(Int, plot_idx / n_cols)
        col = ((plot_idx - 1) % n_cols) + 1
        
        ax = Axis(fig2[row, col],
                 title="$(param_names[param_idx]) (r=$(round(obj_correlations[param_idx], digits=3)))",
                 xlabel="Parameter Value",
                 ylabel="Objective Value")
        
        # Scatter plot with color gradient
        param_values = param_matrix_clean[:, param_idx]
        scatter!(ax, param_values, objectives_clean, 
                color=objectives_clean, colormap=:viridis, markersize=6, alpha=0.7)
        
        # Add best point
        if haskey(results, "best_params")
            best_param = results["best_params"][string(param_names[param_idx])]
            best_obj = results["best_objective"]
            scatter!(ax, [best_param], [best_obj], 
                    color=:red, markersize=12, marker=:star5)
        end
        
        # Add trend line if correlation is significant
        if abs(obj_correlations[param_idx]) > 0.3 && length(param_values) > 20
            # Simple linear regression
            X = hcat(ones(length(param_values)), param_values)
            β = X \ objectives_clean
            x_trend = range(minimum(param_values), maximum(param_values), length=100)
            y_trend = β[1] .+ β[2] .* x_trend
            lines!(ax, x_trend, y_trend, color=:red, linewidth=2, linestyle=:dash)
        end
    end
    
    plot_path2 = joinpath(PLOTS_DIR, "$(save_prefix)/parameter_sensitivity.png")
    save(plot_path2, fig2)
    println("📊 Parameter sensitivity saved: $plot_path2")
end

"""Create performance and efficiency analysis"""
function create_performance_analysis(results, save_prefix::String)
    fig = Figure(size=(1200, 800))
    
    # Timing analysis
    if haskey(results, "elapsed_time") && haskey(results, "n_evaluations")
        total_time = results["elapsed_time"]
        n_evals = results["n_evaluations"]
        n_workers = get(results, "n_workers", 1)
        avg_time = results["avg_time_per_eval"]
        
        # Create timing summary
        ax1 = Axis(fig[1, 1],
                  title="Performance Summary",
                  ylabel="Value")
        
        categories = ["Total Time\n(min)", "Avg Time/Eval\n(sec)", "Throughput\n(eval/sec)", "Workers"]
        values = [total_time/60, avg_time, 1/avg_time, n_workers]
        
        barplot!(ax1, 1:length(categories), values, color=[:steelblue, :orange, :green, :purple])
        ax1.xticks = (1:length(categories), categories)
        ax1.xticklabelrotation = π/6
        
        # Efficiency metrics
        ax2 = Axis(fig[1, 2],
                  title="Parallel Efficiency",
                  xlabel="Metric",
                  ylabel="Value")
        
        # Calculate theoretical vs actual performance
        theoretical_speedup = n_workers
        actual_speedup = n_workers  # Assumes perfect scaling for now
        efficiency = actual_speedup / theoretical_speedup * 100
        
        efficiency_metrics = ["Parallel Efficiency (%)", "Worker Utilization (%)"]
        efficiency_values = [efficiency, 95.0]  # Placeholder for utilization
        
        barplot!(ax2, 1:length(efficiency_metrics), efficiency_values, 
                color=[:darkgreen, :darkblue])
        ax2.xticks = (1:length(efficiency_metrics), efficiency_metrics)
        ax2.xticklabelrotation = π/6
        ylims!(ax2, 0, 100)
        
        # Add horizontal line at 100%
        hlines!(ax2, [100], color=:red, linestyle=:dash, linewidth=2)
    end
    
    # Resource utilization over time (if available)
    ax3 = Axis(fig[2, 1:2],
              title="Search Progress Timeline",
              xlabel="Time (minutes)",
              ylabel="Cumulative Best Objective")
    
    if haskey(results, "all_objectives")
        objectives = collect(results["all_objectives"])
        n_evals = length(objectives)
        total_time = get(results, "elapsed_time", n_evals)
        
        # Create time axis
        time_points = range(0, total_time/60, length=n_evals)
        
        # Build best history
        best_history = Float64[]
        current_best = Inf
        for obj in objectives
            if obj < current_best
                current_best = obj
            end
            push!(best_history, current_best)
        end
        
        lines!(ax3, time_points, best_history, color=:steelblue, linewidth=3)
        
        # Add annotations for major improvements
        improvements = findall(diff(best_history) .< -0.01 * abs(best_history[1]))
        if !isempty(improvements) && length(improvements) < 20
            scatter!(ax3, time_points[improvements], best_history[improvements],
                    color=:red, markersize=8, marker=:star5)
        end
    end
    
    plot_path = joinpath(PLOTS_DIR, "$(save_prefix)/performance_analysis.png")
    save(plot_path, fig)
    println("📊 Performance analysis saved: $plot_path")
end

"""Main analysis function"""
# function run_advanced_analysis()
    println("🔍 Loading MPI search results...")
    results, results_file = load_latest_results();
    
    # Use fixed filename prefix for overwriting plots (without directory path)
    save_prefix = begin
        filename = basename(results_file)
        # Extract job ID from filename like "mpi_search_results_job3052881_2025-08-22_03-35-34.json"
        job_match = match(r"mpi_search_results_(job\d+)_", filename)
        if job_match !== nothing
            job_id = job_match.captures[1]
            String(job_id)
        else
            ""
        end
    end
    # Ensure the directory exists
    mkpath(PLOTS_DIR * "/" * save_prefix)

    println("🎨 Creating publication-quality plots...")
    
    # Generate all analysis plots
    create_convergence_analysis(results, save_prefix)
    create_parameter_analysis(results, save_prefix)
    create_performance_analysis(results, save_prefix)

    # NEW ENHANCED ANALYSIS FEATURES
    println("\n🔬 Running enhanced diagnostic analysis...")
    
    # 1. Top 5 candidates leaderboard
    top_candidates = create_top_candidates_analysis(results, save_prefix, 5)
    
    # 2. Parameter trajectories for best candidates evolution
    create_parameter_trajectories(results, save_prefix)
    
    # 3. Moment trajectories for best candidates evolution
    create_moment_trajectories(results, save_prefix)
    
    # 4. Moment sensitivity analysis
    create_moment_sensitivity(results, save_prefix)
    
    # 5. Moment mismatch analysis for best candidate (commented out due to dependencies)
    analyze_best_candidate_moments(results, save_prefix)
    
    # Print summary
    println("\n" * "="^60)
    println("📊 ADVANCED ANALYSIS COMPLETE")
    println("="^60)
    println("📁 Source: $(basename(results_file))")
    println("📈 Plots saved to: $PLOTS_DIR")
    
    if haskey(results, "best_objective")
        println("🏆 Best objective: $(round(results["best_objective"], digits=6))")
    end
    
    if haskey(results, "n_evaluations")
        println("📊 Total evaluations: $(results["n_evaluations"])")
    end
    
    # Enhanced summary with new diagnostics
    if !isnothing(top_candidates) && length(top_candidates) >= 2
        second_best = top_candidates[2][1]
        best = top_candidates[1][1]
        gap = abs(second_best - best) / abs(best) * 100
        println("🥈 Gap to 2nd best: $(round(gap, digits=4))%")
    end
    
    println("⏱️  Analysis completed at: $(Dates.now())")
    println("\n📊 GENERATED PLOTS:")
    println("   • Convergence Analysis: $(save_prefix)_convergence_analysis.png")
    println("   • Parameter Correlations: $(save_prefix)_parameter_correlations.png") 
    println("   • Parameter Sensitivity: $(save_prefix)_parameter_sensitivity.png")
    println("   • Performance Analysis: $(save_prefix)_performance_analysis.png")
    println("   • Top Candidates: $(save_prefix)_top_candidates.png")
    println("   • Parameter Trajectories: $(save_prefix)_parameter_trajectories.png")
    println("   • Moment Trajectories: $(save_prefix)_moment_trajectories.png")
    println("   • Moment Sensitivity: $(save_prefix)_moment_sensitivity.png")
    println("   • Moment Analysis: $(save_prefix)_moment_analysis.png")

println("\n🎯 Advanced MPI analysis complete!")
