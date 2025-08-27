#!/usr/bin/env julia

"""
Batch Local Optimization Search Script

This script is called by the SLURM job array to run local optimization
on a specific batch of starting points.

Usage: julia run_batch_search.jl <batch_id> <start_index> <end_index> <batch_size>
"""

using YAML
using CSV
using DataFrames
using LinearAlgebra
using Statistics
using JSON
using Dates
using Random

# Import our local modules
include("../../../../ModelSolver.jl")
include("../../../../objective.jl")

function parse_command_line_args()
    if length(ARGS) != 4
        error("Usage: julia run_batch_search.jl <batch_id> <start_index> <end_index> <batch_size>")
    end
    
    batch_id = parse(Int, ARGS[1])
    start_index = parse(Int, ARGS[2])
    end_index = parse(Int, ARGS[3])
    batch_size = parse(Int, ARGS[4])
    
    return batch_id, start_index, end_index, batch_size
end

function load_config()
    config_path = "local_refine_config.yaml"
    if !isfile(config_path)
        error("Configuration file not found: $config_path")
    end
    return YAML.load_file(config_path)
end

function euclidean_distance(x, y)
    return sqrt(sum((x .- y).^2))
end

function farthest_point_sampling(candidates, n_diverse)
    n_total = size(candidates, 1)
    if n_diverse >= n_total
        return collect(1:n_total)
    end
    
    selected_indices = [1]  # Start with first candidate
    candidates_matrix = Matrix(candidates)
    
    for _ in 2:n_diverse
        max_min_distance = -Inf
        best_candidate = 1
        
        for i in 1:n_total
            if i in selected_indices
                continue
            end
            
            # Find minimum distance to already selected points
            min_distance = minimum([euclidean_distance(candidates_matrix[i, :], 
                                                     candidates_matrix[j, :]) 
                                  for j in selected_indices])
            
            if min_distance > max_min_distance
                max_min_distance = min_distance
                best_candidate = i
            end
        end
        
        push!(selected_indices, best_candidate)
    end
    
    return selected_indices
end

function load_batch_candidates(config, batch_id, start_index, end_index, batch_size)
    """Load the specific candidates for this batch"""
    
    println("📋 Loading candidates for batch $batch_id...")
    
    # Load all candidates first
    candidates_file = get(config, "candidates_file", "candidates.csv")
    if !isfile(candidates_file)
        error("Candidates file not found: $candidates_file")
    end
    
    all_candidates = CSV.read(candidates_file, DataFrame)
    total_available = nrow(all_candidates)
    
    println("   📊 Total candidates available: $total_available")
    println("   🎯 Batch range: $start_index to $end_index")
    
    if end_index > total_available
        println("   ⚠️  Warning: Requested end index ($end_index) exceeds available candidates ($total_available)")
        end_index = min(end_index, total_available)
    end
    
    # Apply quality filter
    n_quality = get(config, "n_quality_filter", 1000)
    if total_available > n_quality
        println("   🎯 Applying quality filter: keeping top $n_quality candidates")
        # Sort by objective value (assuming lower is better)
        sort!(all_candidates, :objective_value)
        quality_filtered = all_candidates[1:n_quality, :]
    else
        println("   ✅ All candidates pass quality filter")
        quality_filtered = all_candidates
    end
    
    # Apply diversity sampling to the quality-filtered candidates
    param_columns = config["param_columns"]
    diversity_data = quality_filtered[:, param_columns]
    
    n_diverse = get(config, "n_diverse", 500)
    if nrow(quality_filtered) > n_diverse
        println("   🎲 Applying diversity sampling: selecting $n_diverse diverse candidates")
        diverse_indices = farthest_point_sampling(diversity_data, n_diverse)
        diverse_candidates = quality_filtered[diverse_indices, :]
    else
        println("   ✅ All quality candidates are diverse enough")
        diverse_candidates = quality_filtered
    end
    
    # Now select the batch range from the diverse candidates
    actual_end = min(end_index, nrow(diverse_candidates))
    actual_start = min(start_index, actual_end)
    
    if actual_start > nrow(diverse_candidates)
        error("Batch start index ($start_index) exceeds available diverse candidates ($(nrow(diverse_candidates)))")
    end
    
    batch_candidates = diverse_candidates[actual_start:actual_end, :]
    
    println("   ✅ Selected $(nrow(batch_candidates)) candidates for optimization")
    
    return batch_candidates
end

function run_batch_optimization(config, candidates, batch_id)
    """Run optimization on the batch of candidates"""
    
    println("🚀 Starting batch optimization...")
    
    # Extract optimization parameters
    objective_config = config["objective"]
    max_iters = get(config, "max_iters", 10000)
    
    # Setup optimization results storage
    results = []
    
    n_candidates = nrow(candidates)
    println("   🎯 Optimizing $n_candidates candidates")
    
    for (i, candidate_row) in enumerate(eachrow(candidates))
        start_time = time()
        
        print("   🔍 Candidate $i/$n_candidates: ")
        
        try
            # Extract starting point
            param_columns = config["param_columns"]
            x0 = [candidate_row[col] for col in param_columns]
            
            # Create the objective function
            obj_func = x -> objective_function(x, objective_config)
            
            # Setup and solve optimization problem
            using OptimizationOptimJL
            using SciMLBase
            
            prob = OptimizationProblem(obj_func, x0)
            solve_kwargs = Dict{Symbol, Any}()
            if haskey(config, "verbose") && !config["verbose"]
                solve_kwargs[:verbose] = false
            end
            
            sol = solve(prob, NelderMead(); maxiters=max_iters, solve_kwargs...)
            
            # Store results
            result = Dict(
                "batch_id" => batch_id,
                "candidate_index" => i,
                "original_objective" => candidate_row.objective_value,
                "optimized_objective" => sol.objective,
                "optimization_time" => time() - start_time,
                "iterations" => get(sol, :iterations, missing),
                "converged" => sol.retcode == SciMLBase.ReturnCode.Success,
                "starting_point" => x0,
                "optimized_point" => sol.u,
                "improvement" => candidate_row.objective_value - sol.objective
            )
            
            push!(results, result)
            
            improvement = result["improvement"]
            if improvement > 0
                println("✅ Improved by $(round(improvement, digits=4)) in $(round(result["optimization_time"], digits=2))s")
            else
                println("⚪ No improvement ($(round(improvement, digits=4))) in $(round(result["optimization_time"], digits=2))s")
            end
            
        catch e
            println("❌ Error: $e")
            error_result = Dict(
                "batch_id" => batch_id,
                "candidate_index" => i,
                "error" => string(e),
                "optimization_time" => time() - start_time,
                "converged" => false
            )
            push!(results, error_result)
        end
    end
    
    return results
end

function save_batch_results(results, batch_id, config)
    """Save the optimization results for this batch"""
    
    println("💾 Saving batch results...")
    
    # Create output filename
    output_file = "optimization_results_batch_$batch_id.json"
    
    # Create summary statistics
    successful_results = filter(r -> get(r, "converged", false), results)
    n_successful = length(successful_results)
    n_total = length(results)
    
    improvements = [r["improvement"] for r in successful_results if haskey(r, "improvement")]
    
    summary = Dict(
        "batch_id" => batch_id,
        "timestamp" => string(now()),
        "total_candidates" => n_total,
        "successful_optimizations" => n_successful,
        "success_rate" => n_successful / n_total,
        "mean_improvement" => isempty(improvements) ? 0.0 : mean(improvements),
        "max_improvement" => isempty(improvements) ? 0.0 : maximum(improvements),
        "total_runtime" => sum([r["optimization_time"] for r in results if haskey(r, "optimization_time")])
    )
    
    # Combine summary and detailed results
    output_data = Dict(
        "summary" => summary,
        "detailed_results" => results
    )
    
    # Save to file
    open(output_file, "w") do f
        JSON.print(f, output_data, 4)
    end
    
    println("   ✅ Results saved to: $output_file")
    println("   📊 Success rate: $(round(summary["success_rate"] * 100, digits=1))%")
    
    if n_successful > 0
        println("   📈 Mean improvement: $(round(summary["mean_improvement"], digits=4))")
        println("   🎯 Best improvement: $(round(summary["max_improvement"], digits=4))")
    end
    
    return output_file
end

function main()
    println("===============================================================================")
    println("🔬 BATCH LOCAL OPTIMIZATION")
    println("===============================================================================")
    println("📅 Start time: $(now())")
    
    # Parse command line arguments
    batch_id, start_index, end_index, batch_size = parse_command_line_args()
    
    println("📋 Batch Configuration:")
    println("   • Batch ID: $batch_id")
    println("   • Index range: $start_index to $end_index")
    println("   • Batch size: $batch_size")
    
    try
        # Load configuration
        config = load_config()
        println("✅ Configuration loaded")
        
        # Load candidates for this batch
        candidates = load_batch_candidates(config, batch_id, start_index, end_index, batch_size)
        
        # Run optimization
        results = run_batch_optimization(config, candidates, batch_id)
        
        # Save results
        output_file = save_batch_results(results, batch_id, config)
        
        println("")
        println("===============================================================================")
        println("✅ BATCH $batch_id COMPLETED SUCCESSFULLY")
        println("📊 Results saved to: $output_file")
        println("⏰ Total time: $(now())")
        println("===============================================================================")
        
    catch e
        println("")
        println("===============================================================================")
        println("❌ BATCH $batch_id FAILED")
        println("💥 Error: $e")
        println("⏰ Failed at: $(now())")
        println("===============================================================================")
        
        # Print stack trace for debugging
        showerror(stdout, e, catch_backtrace())
        
        exit(1)
    end
end

# Run the main function
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
