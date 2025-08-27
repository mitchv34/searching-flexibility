#!/usr/bin/env julia
# Clean Ground Truth Overlay Analysis Script

using Pkg
Pkg.activate("/project/high_tech_ind/searching-flexibility")

using JSON3, YAML
using CairoMakie
using Statistics, StatsBase, LinearAlgebra
using Printf, Dates
using DataFrames, CSV
using Random; Random.seed!(123)

println("🎨 Advanced MPI Search Analysis (Ground Truth Overlays)")

JOB_ID = length(ARGS) >= 1 ? strip(ARGS[1]) : nothing

# Theme
set_theme!(Theme(fontsize=12))

const SCRIPT_DIR = dirname(@__FILE__)
const RESULTS_DIR = joinpath(SCRIPT_DIR, "output", "results")
const OUTPUT_DIR = joinpath(SCRIPT_DIR, "figures")
const PLOTS_DIR = OUTPUT_DIR
mkpath(PLOTS_DIR)

# ---------------- Utility -----------------
function adaptive_valid_mask(objectives::AbstractVector{<:Real}; penalty_threshold=1e8, min_required=25)
    finite_mask = isfinite.(objectives)
    finite_vals = objectives[finite_mask]
    if isempty(finite_vals); return falses(length(objectives)); end
    base_mask = finite_mask .& (objectives .< penalty_threshold)
    if count(base_mask) >= min_required; return base_mask; end
    sorted_vals = sort(finite_vals)
    k = min(length(sorted_vals), max(min_required, Int(clamp(round(length(sorted_vals)*0.01), 1, length(sorted_vals)))))
    kth_val = sorted_vals[k]
    p95 = quantile(sorted_vals, min(0.95, max(0.5, 1 - min_required/length(sorted_vals))))
    adaptive_threshold = min(kth_val, p95)
    return finite_mask .& (objectives .<= adaptive_threshold)
end

function report_objective_distribution(objectives)
    finite_vals = filter(isfinite, objectives)
    isempty(finite_vals) && return
    q = quantile
    println(@sprintf("🔎 Objectives: n=%d min=%.4g median=%.4g max=%.4g", length(finite_vals), minimum(finite_vals), q(finite_vals,0.5), maximum(finite_vals)))
end

# -------------- Load results ----------------
function load_latest_results(job_id::Union{Nothing,String})
    isdir(RESULTS_DIR) || error("Results dir missing: $(RESULTS_DIR)")
    files = filter(f->endswith(f,".json"), readdir(RESULTS_DIR; join=true))
    files = filter(f->!occursin("live",f) && !occursin("history",f), files)
    if job_id !== nothing
        job_files = filter(f->occursin("job$(job_id)", f), files)
        finals = filter(f->occursin("final", f), job_files)
        chosen = isempty(finals) ? job_files : finals
    else
        finals = filter(f->occursin("final", f), files)
        chosen = isempty(finals) ? files : finals
    end
    isempty(chosen) && error("No results JSON found")
    sort!(chosen)
    rf = last(chosen)
    println("📁 Using results file: $(basename(rf))")
    raw = JSON3.read(read(rf,String))
    return Dict{String,Any}(pair for pair in pairs(raw)), rf
end

# -------------- Ground truth loaders ----------------
const PROJECT_ROOT = "/project/high_tech_ind/searching-flexibility"
function load_ground_truth_params(; path=get(ENV, "GROUND_TRUTH_PARAM_FILE", joinpath(PROJECT_ROOT, "data", "results", "estimated_parameters", "estimated_parameters_2024.yaml")))
    if !isfile(path); @warn "GT param file missing: $(path)"; return Dict{String,Float64}(); end
    y = YAML.load_file(path)
    haskey(y, "ModelParameters") || return Dict{String,Float64}()
    gt = Dict{String,Float64}()
    for (k,v) in y["ModelParameters"]
        v isa Real || continue
        gt[string(k)] = float(v)
    end
    gt
end
function load_ground_truth_moments(; path=get(ENV, "GROUND_TRUTH_MOMENTS_FILE", joinpath(PROJECT_ROOT, "data", "results", "data_moments", "data_moments_groundtruth_2024.yaml")))
    if !isfile(path); @warn "GT moments file missing: $(path)"; return Dict{String,Float64}(); end
    y = YAML.load_file(path)
    haskey(y, "DataMoments") || return Dict{String,Float64}()
    dm = Dict{String,Float64}()
    for (k,v) in y["DataMoments"]
        v isa Real || continue
        dm[string(k)] = float(v)
    end
    dm
end
GROUND_TRUTH_PARAMS = load_ground_truth_params()
GROUND_TRUTH_MOMENTS = load_ground_truth_moments()
println("✅ GT loaded: params=$(length(GROUND_TRUTH_PARAMS)) moments=$(length(GROUND_TRUTH_MOMENTS))")

# -------------- Plots ----------------
function create_parameter_trajectories(results, save_prefix)
    haskey(results,"all_params") && haskey(results,"all_objectives") || return
    params = results["all_params"]; objs = results["all_objectives"]; names = results["parameter_names"]
    best_idx = Int[]; best_params = Vector{Vector{Float64}}(); best_objs = Float64[]
    cur = Inf
    valid = adaptive_valid_mask(objs)
    for (i,(p,o,v)) in enumerate(zip(params,objs,valid))
        if v && o < cur
            cur = o; push!(best_idx,i); push!(best_params,p); push!(best_objs,o)
        end
    end
    length(best_idx) < 2 && return
    ncols=3; nrows=ceil(Int,length(names)/ncols)
    fig=Figure(size=(420*ncols,280*nrows))
    for (pi,pn) in enumerate(names)
        row=ceil(Int,pi/ncols); col=((pi-1)%ncols)+1
        ax=Axis(fig[row,col], title=string(pn), xlabel="Evaluation", ylabel="Value")
        vals=[bp[pi] for bp in best_params]
        lines!(ax,best_idx,vals,color=:steelblue,linewidth=3)
        scatter!(ax,best_idx,vals,color=:steelblue,markersize=6)
        # GT line solid red
        for variant in (string(pn), replace(string(pn),"₀"=>"0"), replace(string(pn),"₁"=>"1"))
            if haskey(GROUND_TRUTH_PARAMS, variant)
                hlines!(ax,[GROUND_TRUTH_PARAMS[variant]], color=:red, linewidth=3)
                break
            end
        end
    end
    Label(fig[0,:], text="Parameter Trajectories (Best Improvements)", fontsize=16)
    out=joinpath(PLOTS_DIR,"$(save_prefix)_parameter_trajectories.png"); save(out,fig)
    println("📊 Parameter trajectories: $(out)")
end

function create_parameter_analysis(results, save_prefix)
    haskey(results,"all_params") && haskey(results,"all_objectives") || return
    params=results["all_params"]; objs=results["all_objectives"]; names=results["parameter_names"]
    mat=hcat(params...)'
    valid=adaptive_valid_mask(objs)
    matv=mat[valid,:]; objv=objs[valid]
    length(objv)<10 && return
    logobj=log.(objv .- minimum(objv) .+ 1e-9)
    corrabs=[abs(cor(matv[:,i],logobj)) for i in 1:length(names)]
    order=sortperm(corrabs, rev=true)
    ncols=min(4,length(names)); nrows=ceil(Int,length(names)/ncols)
    fig=Figure(size=(420*ncols,300*nrows))
    for (plot_i,idxp) in enumerate(order)
        row=ceil(Int,plot_i/ncols); col=((plot_i-1)%ncols)+1
        ax=Axis(fig[row,col], title="$(names[idxp]) (|r|=$(round(corrabs[idxp],digits=3)))", xlabel=string(names[idxp]), ylabel="log obj")
        vals=matv[:,idxp]
        scatter!(ax, vals, logobj, color=:gray30, markersize=5)
        for variant in (string(names[idxp]), replace(string(names[idxp]),"₀"=>"0"), replace(string(names[idxp]),"₁"=>"1"))
            if haskey(GROUND_TRUTH_PARAMS, variant)
                vlines!(ax,[GROUND_TRUTH_PARAMS[variant]], color=:red, linewidth=3)
                break
            end
        end
    end
    Label(fig[0,:], text="Parameter Sensitivity (log objective)", fontsize=16)
    out=joinpath(PLOTS_DIR,"$(save_prefix)_parameter_sensitivity.png"); save(out,fig)
    println("📊 Parameter sensitivity: $(out)")
end

function create_moment_trajectories(results, save_prefix)
    haskey(results,"all_moments") && haskey(results,"all_objectives") || return
    moms=results["all_moments"]; objs=results["all_objectives"]
    best_idx=Int[]; best_moms=Vector{Dict{String,Float64}}(); cur=Inf
    for (i,(mm,o)) in enumerate(zip(moms,objs))
        if isfinite(o) && o<cur
            cur=o
            dict_mm=Dict{String,Float64}()
            for (k,v) in mm
                v isa Real || continue
                dict_mm[string(k)] = float(v)
            end
            push!(best_idx,i); push!(best_moms,dict_mm)
        end
    end
    length(best_idx)<2 && return
    mset=Set{String}(); for mm in best_moms; for k in keys(mm); push!(mset,k); end; end
    mnames=collect(mset)
    ncols=3; nrows=ceil(Int,length(mnames)/ncols)
    fig=Figure(size=(420*ncols,300*nrows))
    for (mi,mn) in enumerate(mnames)
        row=ceil(Int,mi/ncols); col=((mi-1)%ncols)+1
        ax=Axis(fig[row,col], title=mn, xlabel="Evaluation", ylabel="Value")
        vals=Float64[]; idxs=Int[]
        for (j,mm) in enumerate(best_moms)
            if haskey(mm,mn)
                push!(vals,mm[mn]); push!(idxs,best_idx[j])
            end
        end
        length(vals)<2 && (text!(ax,0.5,0.5,text="Insufficient",space=:relative); continue)
        lines!(ax, idxs, vals, color=:steelblue, linewidth=3)
        scatter!(ax, idxs, vals, color=:steelblue, markersize=6)
        if haskey(GROUND_TRUTH_MOMENTS, mn)
            hlines!(ax,[GROUND_TRUTH_MOMENTS[mn]], color=:red, linewidth=3)
        end
    end
    Label(fig[0,:], text="Moment Trajectories", fontsize=16)
    out=joinpath(PLOTS_DIR,"$(save_prefix)_moment_trajectories.png"); save(out,fig)
    println("📊 Moment trajectories: $(out)")
end

function create_top_candidates_analysis(results, save_prefix; top_n=5)
    haskey(results,"all_params") && haskey(results,"all_objectives") || return
    params=results["all_params"]; objs=results["all_objectives"]; names=results["parameter_names"]
    valid=adaptive_valid_mask(objs)
    tuples=[(objs[i],params[i]) for i in eachindex(objs) if valid[i]]
    isempty(tuples) && return
    sort!(tuples, by=x->x[1])
    top=tuples[1:min(top_n,length(tuples))]
    df=DataFrame(Rank=Int[], Objective=Float64[])
    for pn in names; df[!,string(pn)]=Float64[]; end
    for (i,(o,p)) in enumerate(top)
        push!(df.Rank,i); push!(df.Objective,o)
        for (j,pn) in enumerate(names)
            push!(df[!,string(pn)], p[j])
        end
    end
    csv=joinpath(PLOTS_DIR,"$(save_prefix)_top_candidates.csv")
    CSV.write(csv,df)
    println("🏆 Top candidates CSV: $(csv)")
end

function run_all()
    results, rf = load_latest_results(JOB_ID)
    haskey(results,"all_objectives") && report_objective_distribution(results["all_objectives"])
    ts=Dates.format(now(),"yyyymmdd_HHMMSS")
    prefix = JOB_ID === nothing ? "alljobs_$(ts)" : "job$(JOB_ID)_$(ts)"
    create_parameter_trajectories(results,prefix)
    create_parameter_analysis(results,prefix)
    create_moment_trajectories(results,prefix)
    create_top_candidates_analysis(results,prefix)
    println("✅ Analysis complete")
end

run_all()


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
        
        plot_path = joinpath(PLOTS_DIR, "$(save_prefix)_moment_analysis.png")
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
    # Determine burn-in skip: ignore early sharp drop for clarity
    # Skip first N improvements or first BURN_FRAC of evaluations (whichever larger)
    improvements = findall(diff(best_history) .< 0)
    burn_frac = 0.02
    burn_improvements = 10
    burn_idx = 1
    if !isempty(improvements)
        if length(improvements) >= burn_improvements
            burn_idx = improvements[burn_improvements]  # index before Nth improvement
        else
            burn_idx = last(improvements)
        end
    end
    burn_idx = max(burn_idx, Int(clamp(round(n_evals * burn_frac),1,n_evals)))

    # Subsampled best history for plotting
    best_history_plot = best_history[plot_indices]
    plot_eval_nums = collect(plot_indices)
    # Overlay truncated (post burn-in) for detail
    truncated_mask = plot_eval_nums .>= burn_idx
    truncated_evals = plot_eval_nums[truncated_mask]
    truncated_best = best_history_plot[truncated_mask]
    
    # Create figure with multiple subplots
    fig = Figure(size=(1400, 1000))
    
    # Main convergence plot
    ax1 = Axis(fig[1, 1:2], 
               title="Convergence History",
               xlabel="Evaluation Number",
               ylabel="Best Objective Value")

    lines!(ax1, plot_eval_nums, best_history_plot, color=(:steelblue,0.35), linewidth=2)
    lines!(ax1, truncated_evals, truncated_best, color=:steelblue, linewidth=3)
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
    valid_objectives = objectives[adaptive_valid_mask(objectives)]
    # Outlier filtering (upper tail) for clearer density: remove top 1% & bottom 0.5%
    if length(valid_objectives) > 50
        q_low = quantile(valid_objectives, 0.005)
        q_high = quantile(valid_objectives, 0.99)
        filtered = valid_objectives[(valid_objectives .>= q_low) .& (valid_objectives .<= q_high)]
    else
        filtered = valid_objectives
    end
    if !isempty(filtered)
        hist!(ax2, filtered, bins=30, color=(:steelblue, 0.7), normalization=:pdf)
        if length(filtered) > 20
            kde_result = kde(filtered)
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
    
    plot_path = joinpath(PLOTS_DIR, "$(save_prefix)_convergence_analysis.png")
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
    valid_mask = adaptive_valid_mask(objectives)
    
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
    
    plot_path1 = joinpath(PLOTS_DIR, "$(save_prefix)_parameter_correlations.png")
    save(plot_path1, fig1)
    println("📊 Parameter correlations saved: $plot_path1")
    
    # Parameter sensitivity analysis (ALL parameters, log objective)
    # Safely transform objective values to log scale (shift if any <= 0)
    obj_vals = copy(objectives_clean)
    if any(obj_vals .<= 0)
        shift = abs(minimum(obj_vals)) + 1e-6
        obj_vals .= obj_vals .+ shift
    end
    log_obj_vals = log.(obj_vals)

    # Correlations with log objective
    obj_correlations = [abs(cor(param_matrix_clean[:, i], log_obj_vals)) for i in 1:n_params]
    sorted_indices = sortperm(obj_correlations, rev=true)  # order by strength

    # Dynamic grid sizing
    n_cols = min(4, n_params)
    n_rows = ceil(Int, n_params / n_cols)
    fig2 = Figure(size=(350 * n_cols, 300 * n_rows))

    for (plot_idx, param_idx) in enumerate(sorted_indices)  # include ALL params
        row = ceil(Int, plot_idx / n_cols)
        col = ((plot_idx - 1) % n_cols) + 1

        ax = Axis(fig2[row, col],
                 title="$(param_names[param_idx]) (|r|=$(round(obj_correlations[param_idx], digits=3)))",
                 xlabel="Parameter Value",
                 ylabel="log(Objective Value)")

        param_values = param_matrix_clean[:, param_idx]
        scatter!(ax, param_values, log_obj_vals,
                 color=log_obj_vals, colormap=:viridis, markersize=5, alpha=0.75)

        # Add best point (if available and valid)
        if haskey(results, "best_params") && haskey(results, "best_objective")
            best_param = results["best_params"][string(param_names[param_idx])]
            best_obj = results["best_objective"]
            if best_obj > 0
                scatter!(ax, [best_param], [log(best_obj)], color=:red, markersize=10, marker=:star5)
            end
        end

        # Trend line if notable correlation
        if abs(obj_correlations[param_idx]) > 0.3 && length(param_values) > 20
            X = hcat(ones(length(param_values)), param_values)
            β = X \ log_obj_vals
            x_trend = range(minimum(param_values), maximum(param_values), length=100)
            y_trend = β[1] .+ β[2] .* x_trend
            lines!(ax, x_trend, y_trend, color=:red, linewidth=2, linestyle=:dash)
        end
    end

    plot_path2 = joinpath(PLOTS_DIR, "$(save_prefix)_parameter_sensitivity.png")
    save(plot_path2, fig2)
    println("📊 Parameter sensitivity (all params, log objective) saved: $plot_path2")
end

"""Create performance and efficiency analysis"""
function create_performance_analysis(results, save_prefix::String)
    fig = Figure(size=(1200, 800))
    
    # Timing analysis
    if haskey(results, "elapsed_time") && (haskey(results, "n_evaluations") || haskey(results, "completed_evaluations"))
        total_time = results["elapsed_time"]
        n_evals = haskey(results, "n_evaluations") ? results["n_evaluations"] : results["completed_evaluations"]
        n_workers = get(results, "n_workers", 1)
        avg_time = haskey(results, "avg_time_per_eval") ? results["avg_time_per_eval"] : (total_time / max(1,n_evals))
        
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
        
        # Burn-in removal for clarity (same logic as convergence)
        improvements = findall(diff(best_history) .< 0)
        burn_improvements = 10
        burn_idx = 1
        if !isempty(improvements)
            if length(improvements) >= burn_improvements
                burn_idx = improvements[burn_improvements]
            else
                burn_idx = last(improvements)
            end
        end
        burn_idx = max(burn_idx, Int(clamp(round(n_evals*0.02),1,n_evals)))
        lines!(ax3, time_points, best_history, color=(:steelblue,0.35), linewidth=2)
        lines!(ax3, time_points[burn_idx:end], best_history[burn_idx:end], color=:steelblue, linewidth=3)
        ylims!(ax3, minimum(best_history[burn_idx:end])*0.999, maximum(best_history[burn_idx:end])*1.001)
        
        # Add annotations for major improvements
        improvements = findall(diff(best_history) .< -0.01 * abs(best_history[1]))
        if !isempty(improvements) && length(improvements) < 20
            scatter!(ax3, time_points[improvements], best_history[improvements],
                    color=:red, markersize=8, marker=:star5)
        end
    end
    
    plot_path = joinpath(PLOTS_DIR, "$(save_prefix)_performance_analysis.png")
    save(plot_path, fig)
    println("📊 Performance analysis saved: $plot_path")
end

function model_fit_analysis(results)
    # --- Resolve configuration source ---------------------------------------------------
    config_obj = nothing
    used_fallback = false
    if haskey(results, :config_used)
        try
            config_obj = results[:config_used]
        catch
            config_obj = nothing
        end
    end
    if config_obj === nothing
        # Hardcoded fallback per user request
        fallback_path = joinpath(SCRIPT_DIR, "mpi_search_config_test.yaml")
        if isfile(fallback_path)
            try
                config_obj = YAML.load_file(fallback_path)
                used_fallback = true
            catch e
                @warn "Failed to load fallback config at $(fallback_path): $e"
            end
        else
            @warn "Fallback config file not found: $(fallback_path)"
        end
    end

    # Guard against missing keys
    if config_obj === nothing || !haskey(config_obj, "MPISearchConfig") || !haskey(config_obj["MPISearchConfig"], "target_moments")
        @warn "Cannot perform model_fit_analysis: missing target moments configuration"
        return
    end

    target_cfg = config_obj["MPISearchConfig"]["target_moments"]
    if !(haskey(target_cfg, "data_file") && haskey(target_cfg, "moments_to_use"))
        @warn "Target moments configuration incomplete (data_file / moments_to_use)"
        return
    end

    data_file_moments = String(target_cfg["data_file"])
    moments_list = Vector{String}(target_cfg["moments_to_use"])
    if !isfile(data_file_moments)
        @warn "Data moments file not found: $(data_file_moments)"
        return
    end
    data_moments = load_moments_from_yaml(data_file_moments, include = moments_list)

    # --- Extract best params & moments safely ------------------------------------------
    if !haskey(results, :best_params) || !haskey(results, :best_moments)
        @warn "Results missing best_params or best_moments"
        return
    end
    best_params = Dict(results[:best_params])
    model_moments = Dict(results[:best_moments])

    println("\n" * "="^80)
    println("📊 MOMENT COMPARISON SUMMARY")
    println("="^80)
    if used_fallback
        println("(Using hardcoded fallback config: mpi_search_config_test.yaml)")
    end
    
    # Prepare table data
    moment_names = String[]
    data_values = Float64[]
    model_values = Float64[]
    absolute_diffs = Float64[]
    percent_diffs = Float64[]
    
    # Calculate statistics for each moment
    for (moment_name, data_val) in data_moments
        model_val = get(model_moments, moment_name, NaN)
        
        push!(moment_names, string(moment_name))
        push!(data_values, data_val)
        push!(model_values, model_val)
        
        if !isnan(model_val) && data_val != 0
            abs_diff = abs(model_val - data_val)
            pct_diff = abs_diff / abs(data_val) * 100
            push!(absolute_diffs, abs_diff)
            push!(percent_diffs, pct_diff)
        else
            push!(absolute_diffs, NaN)
            push!(percent_diffs, NaN)
        end
    end
    
    # Create the table data matrix
    table_data = hcat(
        moment_names,
        [@sprintf("%.6f", val) for val in data_values],
        [isnan(val) ? "NaN" : @sprintf("%.6f", val) for val in model_values],
        [isnan(val) ? "NaN" : @sprintf("%.6f", val) for val in absolute_diffs],
        [isnan(val) ? "NaN" : @sprintf("%.2f%%", val) for val in percent_diffs]
    )
    
    # Create headers
    headers = ["Moment", "Data Value", "Model Value", "Abs. Difference", "% Error"]
    
    # Print the table
    pretty_table(table_data, 
                header=headers,
                alignment=[:l, :r, :r, :r, :r],
                crop=:none,
                formatters=ft_printf("%.6f", [2, 3, 4]),
                header_crayon=crayon"bold blue",
                subheader_crayon=crayon"bold green")
    
    # Calculate summary statistics
    valid_pct_diffs = percent_diffs[.!isnan.(percent_diffs)]
    valid_abs_diffs = absolute_diffs[.!isnan.(absolute_diffs)]
    
    if !isempty(valid_pct_diffs)
        println("\n📈 FIT SUMMARY STATISTICS:")
        println("─"^50)
        println("• Number of moments: $(length(moment_names))")
        println("• Valid comparisons: $(length(valid_pct_diffs))")
        println("• Mean absolute error: $(round(mean(valid_abs_diffs), digits=6))")
        println("• Root mean squared error: $(round(sqrt(mean(valid_abs_diffs.^2)), digits=6))")
        println("• Mean percentage error: $(round(mean(valid_pct_diffs), digits=2))%")
        println("• Max percentage error: $(round(maximum(valid_pct_diffs), digits=2))%")
        println("• Moments with <5% error: $(sum(valid_pct_diffs .< 5)) / $(length(valid_pct_diffs))")
        println("• Moments with <10% error: $(sum(valid_pct_diffs .< 10)) / $(length(valid_pct_diffs))")
        
        # Identify best and worst fits
        best_idx = argmin(valid_pct_diffs)
        worst_idx = argmax(valid_pct_diffs)
        valid_names = moment_names[.!isnan.(percent_diffs)]
        
        println("\n🏆 BEST FIT: $(valid_names[best_idx]) ($(round(valid_pct_diffs[best_idx], digits=2))% error)")
        println("⚠️  WORST FIT: $(valid_names[worst_idx]) ($(round(valid_pct_diffs[worst_idx], digits=2))% error)")
    end
    
    # Save summary to CSV as well
    summary_df = DataFrame(
        Moment = moment_names,
        Data_Value = data_values,
        Model_Value = model_values,
        Absolute_Difference = absolute_diffs,
        Percent_Error = percent_diffs
    )
    
    csv_path = joinpath(PLOTS_DIR, "moment_comparison_summary.csv")
    CSV.write(csv_path, summary_df)
    println("\n💾 Moment comparison saved to: $csv_path")
end

"""Main analysis function"""
# function run_advanced_analysis()
    println("🔍 Loading MPI search results...")
    results, results_file = load_latest_results();
    if haskey(results, "all_objectives")
        try
            report_objective_distribution(results["all_objectives"])
        catch e
            @warn "Objective distribution diagnostic failed" exception=e
        end
    end
    
    # Simple filename prefix from job ID
    job_id = JOB_ID !== nothing ? JOB_ID : "latest"
    save_prefix = string("mpi_search_", job_id)

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
    
    model_fit_analysis(results)
    
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
    println("   • Objective Sensitivity: $(save_prefix)_objective_sensitivity.png")
    println("   • Sensitivity Heatmap: $(save_prefix)_sensitivity_heatmap.png")

    println("\n🎯 Advanced MPI analysis complete!")



