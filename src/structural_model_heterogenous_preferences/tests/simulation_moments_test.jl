# simulation_moments_test.jl
# 
# Test the new simulation-based moment computation functionality
# Tests Feather file format and collects timing statistics

# Setup parallel computing
using Distributed, SharedArrays
n_workers = 12 # 9 workers for parallel, 0 for serial
if n_workers > 0 && nprocs() == 1
    addprocs(n_workers)
end

# Setup paths and configuration
const ROOT = dirname(dirname(dirname(@__DIR__)))
const MODEL_CONFIG = "/project/high_tech_ind/searching-flexibility/src/structural_model_heterogenous_preferences/distributed_mpi_search/mpi_search_config.yaml"
const FIGURES_DIR = joinpath(ROOT, "figures", "structural_model_heterogenous_preferences", "tests", "simulation_moments_test")

# Load packages on all processors
@everywhere begin
    using Pkg
    Pkg.activate($ROOT)
    
    using Random, Statistics, Distributions, LinearAlgebra, ForwardDiff
    using Printf, Term, YAML
    using CSV, DataFrames
    using JSON3, YAML
    using QuadGK
    using FixedEffectModels
    using CairoMakie, ProgressMeter
    using Arrow  # For feather file support
    using SharedArrays  # For parallel shared arrays
    
    # Include model files
    include(joinpath($ROOT, "src", "structural_model_heterogenous_preferences", "ModelSetup.jl"))
    include(joinpath($ROOT, "src", "structural_model_heterogenous_preferences", "ModelSolver.jl"))
    include(joinpath($ROOT, "src", "structural_model_heterogenous_preferences", "ModelEstimation.jl"))
end

# Main processor setup
using Random, Statistics, Term, Printf, CairoMakie, ProgressMeter, CSV, DataFrames
using Arrow  # For feather file support
using BenchmarkTools, SharedArrays

println("="^80)
println("SIMULATION-BASED MOMENTS TESTING")
println("="^80)
println("Number of processors: $(nprocs())")
println("Number of workers: $(nworkers())")

# Initialize the model
println("Initializing model...")
prim, res = initializeModel(MODEL_CONFIG);

# Solve the model
println("Solving model...")
@time solve_model(prim, res, config=MODEL_CONFIG);

# Global timing collector
timing_results = Float64[]

# Global timing collector
timing_results = Float64[]

# Helper function to time evaluations and collect results
function timed_compute_moments(prim, res, data_path; include_moments=nothing)
    elapsed_time = @elapsed result = compute_model_moments(prim, res, data_path; include_moments=include_moments)
    push!(timing_results, elapsed_time)
    return result, elapsed_time
end

# Define moments to test
test_moments = Symbol.([
        # "mean_logwage", 
        "var_logwage",
        "mean_alpha",
        "var_alpha",
        "inperson_share",
        "remote_share",
        "p90_p10_logwage",
        # "agg_productivity",
        "diff_alpha_high_lowpsi",
        "wage_premium_high_psi",  
        "wage_slope_psi",         
        "market_tightness"
    ])

println("Testing moments: $(join(test_moments, ", "))")

# Test simulation data paths
test_feather_path = "/project/high_tech_ind/searching-flexibility/data/processed/simulation_scaffolding_2024.feather"

println("\n" * "="^80)
println("TESTING FEATHER-BASED MOMENT COMPUTATION")
println("="^80)

if isfile(test_feather_path)
    println("Testing Feather file: $test_feather_path")
    
    # Test the new dispatch method with Feather and timing
    println("Computing moments from Feather simulation data...")
    feather_moments, feather_time = timed_compute_moments(prim, res, test_feather_path; include_moments=test_moments)
    @printf("✓ Feather computation completed in %.4f seconds\n", feather_time)
    
    println("\nFeather-based moments:")
    println("-"^50)
    for moment in test_moments
        if haskey(feather_moments, moment)
            @printf("%-30s : %12.8f\n", String(moment), feather_moments[moment])
        else
            @printf("%-30s : %12s\n", String(moment), "NOT FOUND")
        end
    end
else
    @warn "Feather test file not found: $test_feather_path"
    exit(1)
end

println("\n" * "="^80)
println("REPEATED EVALUATIONS FOR TIMING ANALYSIS")
println("="^80)

# Perform multiple evaluations to get better timing statistics
n_repeats = 10
println("Performing $n_repeats repeated evaluations for timing analysis...")

if isfile(test_feather_path)
    println("\nTiming Feather evaluations...")
    for i in 1:n_repeats
        _, elapsed = timed_compute_moments(prim, res, test_feather_path; include_moments=test_moments)
        @printf("  Evaluation %2d: %.4f seconds\n", i, elapsed)
    end
else
    @warn "Cannot perform repeated evaluations - Feather file not found"
end

println("\n" * "="^80)
println("PERFORMANCE BENCHMARKING")
println("="^80)

if isfile(test_feather_path)
    using BenchmarkTools
    
    println("Benchmarking Feather-based computation...")
    feather_benchmark = @benchmark compute_model_moments($prim, $res, $test_feather_path; include_moments=$test_moments)
    display(feather_benchmark)
else
    @warn "Cannot perform benchmarking - Feather file not found"
end

println("\n" * "="^80)
println("ERROR HANDLING TESTS")
println("="^80)

# Test with invalid file path
println("Testing error handling with invalid file path...")
try
    compute_model_moments(prim, res, "/invalid/path/test.csv"; include_moments=test_moments)
catch e
    println("✓ Correctly caught error: $(typeof(e))")
    println("  Message: $(e.msg)")
end

# Test with unsupported file format
println("\nTesting error handling with unsupported file format...")
try
    compute_model_moments(prim, res, "/tmp/test.txt"; include_moments=test_moments)
catch e
    println("✓ Correctly caught error: $(typeof(e))")
    println("  Message: $(e.msg)")
end

# Test with missing moments
println("\nTesting behavior with missing moments...")
try
    missing_moments = [:non_existent_moment, :another_missing_moment]
    if isfile(test_feather_path)
        result, _ = timed_compute_moments(prim, res, test_feather_path; include_moments=missing_moments)
        println("✓ Function completed with missing moments")
        println("  Result keys: $(keys(result))")
    end
catch e
    println("⚠ Unexpected error with missing moments: $e")
end

println("\n" * "="^80)
println("VISUALIZATION: MOMENT VALUES")
println("="^80)

# Create figures directory if it doesn't exist
mkpath(FIGURES_DIR)

if @isdefined(feather_moments)
    # Create moment values plot
    fig = Figure(size=(1200, 600))
    
    # Extract moment values for plotting
    moment_names = String[]
    feather_vals = Float64[]
    
    for moment in test_moments
        if haskey(feather_moments, moment)
            push!(moment_names, String(moment))
            push!(feather_vals, feather_moments[moment])
        end
    end
    
    if !isempty(moment_names)
        # Main moment values plot
        ax1 = Axis(fig[1, 1], 
                  title="Moment Values from Feather Data",
                  xlabel="Moment Index", 
                  ylabel="Value")
        
        x_pos = 1:length(moment_names)
        barplot!(ax1, x_pos, feather_vals, color=:steelblue, alpha=0.7)
        
        # Add moment names as x-axis labels (rotated)
        ax1.xticks = (x_pos, moment_names)
        ax1.xticklabelrotation = π/4
        
        display(fig)
        
        # Save the figure
        out_png = joinpath(FIGURES_DIR, "feather_moments_values.png")
        save(out_png, fig)
        println("Moment values plot saved to: $out_png")
    end
end

println("\n" * "="^80)
println("OBJECTIVE FUNCTION PROFILING")
println("="^80)

# Parameters to profile around the true values
@everywhere params_to_estimate = [:aₕ, :bₕ, :c₀, :μ, :χ, :A₁, :ν, :ψ₀, :ϕ, :κ₀]
initial_values = [getfield(prim, k) for k in params_to_estimate]

println("Parameters to profile:")
for (i, param) in enumerate(params_to_estimate)
    @printf("  %-10s : %12.6f\n", String(param), initial_values[i])
end

# Compute baseline "true" moments for the objective function
println("\nComputing baseline 'true' moments...")
println("  Using simulation-based moment computation")
println("  Simulation data path: $test_feather_path")
baseline_moments = compute_model_moments(prim, res, test_feather_path)
println("✓ Baseline moments computed ($(length(baseline_moments)) moments)")

# Build problem payload for objective function (following ModelEstimation.jl pattern)
println("\nSetting up objective function...")
println("  Mode: Simulation-based moment computation")
println("  Initialization: COLD-START (no warm-start cache)")
println("  Target moments: $(length(baseline_moments))")

begin
# Settings for the parameter grids
n_grid = 41
rel_width = 0.5  # 50% deviation around true parameter

println("\nGrid settings:")
@printf("  Grid points per parameter: %d\n", n_grid)
@printf("  Relative width: %.1f%%\n", rel_width * 100)

# Create symmetric grids around true values
θ0 = collect(float.(initial_values))
grids = [range(param * (1 - rel_width), param * (1 + rel_width), n_grid) for param in θ0]

@everywhere function objective_function_test(param_name, param_value, baseline_moments, prim, res, test_feather_path)
    # try
        # Update primitives with new parameter value
        prim_new, res_new = update_primitives_results(prim, res, Dict(param_name => param_value))
        # Re-solve the model (verbose=false for parallel execution)
        status = solve_model(prim_new, res_new, verbose=false)
        
        # # Check if model converged
        # if status != :converged
        #     return 1e10  # Return penalty for non-convergence
        # end
        
        # Compute the moments
        model_moments = compute_model_moments(prim_new, res_new, test_feather_path)
        # Compute the square distance between moments
        moment_distance = sum((model_moments[i] - baseline_moments[i])^2 for i in keys(model_moments))
        return moment_distance
    # catch e
    #     @warn "Error in objective_function_test for $param_name = $param_value: $e"
    #     return 1e10  # Return penalty for any errors
    # end
end

println("  Parameter ranges:")
for (i, param) in enumerate(params_to_estimate)
    @printf("    %-10s : [%8.4f, %8.4f]\n", String(param), first(grids[i]), last(grids[i]))
end

# Ship immutable inputs to workers once
@everywhere const SIMDATAPATH = $test_feather_path
@everywhere const MOMENTS = $baseline_moments
@everywhere const PRIM = $prim
@everywhere const RES = $res

# Worker-side evaluation for one parameter's grid
@everywhere function eval_param_grid(i::Int, grid)
    fvals = Vector{Float64}(undef, length(grid))
    for j in eachindex(grid)
        param_name = params_to_estimate[i]
        param_value = grid[j]
        # try
            fvals[j] = objective_function_test(param_name, param_value, MOMENTS, PRIM, RES, SIMDATAPATH)
        # catch e
            # If evaluation fails, assign a large penalty value
        #     @warn "Objective function failed for parameter $i, grid point $j: $e"
        #     fvals[j] = 1e10
        # end
    end
    return fvals
end


n_params = length(params_to_estimate)
fvals_shared = SharedArray{Float64}(n_params, n_grid)
fill!(fvals_shared, NaN)

# Setup parallel progress tracking
println("\nEvaluating objective function across parameter grids...")
progress = Progress(n_params, desc="Parameter grids: ")
channel = RemoteChannel(()->Channel{Bool}(1))

# Async progress updater
@async begin
    while take!(channel)
        next!(progress)
    end
end

# Parallel evaluation with progress updates
results = pmap(i -> begin
    out = eval_param_grid(i, grids[i])
    put!(channel, true)  # update progress for this finished job
    out
end, 1:n_params)

put!(channel, false)  # end updating
# Collect results
for i in 1:n_params
    fvals_shared[i, :] = results[i]
end

println("✓ Objective function evaluation completed")
end

begin
# Create objective function profile plots
println("\nCreating objective function profile plots...")

fig = Figure(size = (1200, 900))
for i in eachindex(params_to_estimate)
    row = ((i - 1) ÷ 3) + 1
    col = ((i - 1) % 3) + 1
    ax = Axis(fig[row, col], 
              title = string(params_to_estimate[i]),
              xlabel = "Parameter Value",
              ylabel = "Objective Function")
    
    grid = grids[i]
    fvals = collect(view(fvals_shared, i, :))
    
    # Plot the objective function
    lines!(ax, grid, fvals, color = :blue, linewidth = 2)
    
    # Add vertical line at true parameter value
    vlines!(ax, [θ0[i]], color = :red, linestyle = :dash, linewidth = 2)
    
    # Add some styling
    ax.xgridvisible = true
    ax.ygridvisible = true
end

# Add a main title
supertitle = Label(fig[0, :], "Objective Function Profiles Around True Parameters", fontsize = 16)

                   
# Save the objective function profile figure
obj_png = joinpath(FIGURES_DIR, "objective_function_profiles.png")
save(obj_png, fig)
println("✓ Objective function profiles saved to: $obj_png")
display(fig)
end 
# Print some summary statistics
println("\nObjective Function Profile Summary:")
for i in eachindex(params_to_estimate)
    fvals = collect(view(fvals_shared, i, :))
    min_val = minimum(fvals)
    max_val = maximum(fvals)
    true_idx = argmin(abs.(collect(grids[i]) .- θ0[i]))
    true_val = fvals[true_idx]
    
    @printf("  %-10s : min=%.2e, max=%.2e, at_true=%.2e\n", 
            String(params_to_estimate[i]), min_val, max_val, true_val)
end

println("\n" * "="^80)
println("SIMULATION MOMENTS TEST COMPLETED")
println("="^80)
println("✓ Feather-based moment computation tested")
println("✓ Performance benchmarking completed")
println("✓ Error handling validated")
println("✓ Timing analysis completed")
println("✓ Objective function profiling completed")

# TIMING SUMMARY STATISTICS
if !isempty(timing_results)
    println("\n" * "="^80)
    println("TIMING SUMMARY STATISTICS")
    println("="^80)
    println("Total evaluations performed: $(length(timing_results))")
    @printf("Mean evaluation time:        %.6f seconds\n", mean(timing_results))
    @printf("Median evaluation time:      %.6f seconds\n", median(timing_results))
    @printf("Minimum evaluation time:     %.6f seconds\n", minimum(timing_results))
    @printf("Maximum evaluation time:     %.6f seconds\n", maximum(timing_results))
    @printf("Standard deviation:          %.6f seconds\n", std(timing_results))
    
    # Additional statistics
    @printf("95th percentile:             %.6f seconds\n", quantile(timing_results, 0.95))
    @printf("5th percentile:              %.6f seconds\n", quantile(timing_results, 0.05))
else
    println("\n⚠ No timing data collected")
end

if @isdefined(feather_moments)
    println("\nMoment Computation Summary:")
    println("  Moments successfully computed: $(length(feather_moments))")
    println("  Requested moments: $(length(test_moments))")
    missing_moments = setdiff(test_moments, keys(feather_moments))
    if !isempty(missing_moments)
        println("  Missing moments: $(join(missing_moments, ", "))")
    else
        println("  ✓ All requested moments computed successfully")
    end
end

println("="^80)
