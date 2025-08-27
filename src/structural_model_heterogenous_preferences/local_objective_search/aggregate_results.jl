#!/usr/bin/env julia

"""
Aggregate Batch Optimization Results

This script collects and aggregates results from all batch optimization jobs,
providing comprehensive analysis and reporting.

Usage: julia aggregate_results.jl [output_directory]
"""

using JSON
using CSV
using DataFrames
using Statistics
using YAML
using Dates
using PrettyTables

function parse_args()
    if length(ARGS) > 1
        error("Usage: julia aggregate_results.jl [output_directory]")
    end
    
    if length(ARGS) == 1
        return ARGS[1]
    else
        # Try to get from config
        config = load_config()
        return get(config["SlurmConfig"], "output_dir", "output/mpi_results")
    end
end

function load_config()
    config_path = "local_refine_config.yaml"
    if !isfile(config_path)
        error("Configuration file not found: $config_path")
    end
    return YAML.load_file(config_path)
end

function find_batch_results(output_dir)
    """Find all batch result files in the output directory"""
    
    if !isdir(output_dir)
        error("Output directory not found: $output_dir")
    end
    
    result_files = []
    summary_files = []
    error_files = []
    
    for file in readdir(output_dir)
        if startswith(file, "optimization_results_batch_") && endswith(file, ".json")
            push!(result_files, joinpath(output_dir, file))
        elseif startswith(file, "batch_") && endswith(file, "_summary.json")
            push!(summary_files, joinpath(output_dir, file))
        elseif startswith(file, "batch_") && endswith(file, "_error.json")
            push!(error_files, joinpath(output_dir, file))
        end
    end
    
    return result_files, summary_files, error_files
end

function load_batch_results(result_files)
    """Load and combine all batch results"""
    
    all_results = []
    batch_summaries = []
    
    for file in result_files
        try
            data = JSON.parsefile(file)
            
            # Extract batch summary
            if haskey(data, "summary")
                push!(batch_summaries, data["summary"])
            end
            
            # Extract detailed results
            if haskey(data, "detailed_results")
                append!(all_results, data["detailed_results"])
            end
            
        catch e
            println("⚠️  Warning: Could not load $file: $e")
        end
    end
    
    return all_results, batch_summaries
end

function load_error_info(error_files)
    """Load information about failed batches"""
    
    error_info = []
    
    for file in error_files
        try
            data = JSON.parsefile(file)
            push!(error_info, data)
        catch e
            println("⚠️  Warning: Could not load error file $file: $e")
        end
    end
    
    return error_info
end

function analyze_results(all_results, batch_summaries, error_info, config)
    """Perform comprehensive analysis of optimization results"""
    
    println("📊 ANALYZING OPTIMIZATION RESULTS")
    println("=" ^ 50)
    
    # Basic statistics
    total_candidates = length(all_results)
    successful_results = filter(r -> get(r, "converged", false), all_results)
    n_successful = length(successful_results)
    n_failed = total_candidates - n_successful
    n_error_batches = length(error_info)
    
    println("📈 Overall Statistics:")
    println("   • Total candidates processed: $total_candidates")
    println("   • Successful optimizations: $n_successful")
    println("   • Failed optimizations: $n_failed")
    println("   • Success rate: $(round(n_successful/total_candidates * 100, digits=1))%")
    println("   • Failed batches: $n_error_batches")
    
    if n_successful == 0
        println("❌ No successful optimizations found!")
        return Dict()
    end
    
    # Performance analysis
    improvements = [r["improvement"] for r in successful_results if haskey(r, "improvement")]
    runtimes = [r["optimization_time"] for r in all_results if haskey(r, "optimization_time")]
    
    # Filter for actual improvements
    positive_improvements = filter(x -> x > 0, improvements)
    
    println("")
    println("🎯 Improvement Analysis:")
    if !isempty(improvements)
        println("   • Mean improvement: $(round(mean(improvements), digits=6))")
        println("   • Median improvement: $(round(median(improvements), digits=6))")
        println("   • Max improvement: $(round(maximum(improvements), digits=6))")
        println("   • Min improvement: $(round(minimum(improvements), digits=6))")
        println("   • Candidates with positive improvement: $(length(positive_improvements)) ($(round(length(positive_improvements)/length(improvements)*100, digits=1))%)")
    end
    
    println("")
    println("⏱️  Runtime Analysis:")
    if !isempty(runtimes)
        println("   • Mean runtime: $(round(mean(runtimes), digits=2))s")
        println("   • Median runtime: $(round(median(runtimes), digits=2))s")
        println("   • Total runtime: $(round(sum(runtimes), digits=2))s")
        println("   • Max runtime: $(round(maximum(runtimes), digits=2))s")
    end
    
    # Batch-level analysis
    println("")
    println("📦 Batch Analysis:")
    if !isempty(batch_summaries)
        batch_success_rates = [s["success_rate"] for s in batch_summaries]
        batch_improvements = [s["mean_improvement"] for s in batch_summaries if haskey(s, "mean_improvement")]
        
        println("   • Number of completed batches: $(length(batch_summaries))")
        println("   • Mean batch success rate: $(round(mean(batch_success_rates) * 100, digits=1))%")
        if !isempty(batch_improvements)
            println("   • Mean batch improvement: $(round(mean(batch_improvements), digits=6))")
        end
    end
    
    # Find best results
    if !isempty(positive_improvements)
        println("")
        println("🏆 Top Improvements:")
        
        # Sort by improvement
        sorted_results = sort(successful_results, by=r -> get(r, "improvement", 0), rev=true)
        top_results = sorted_results[1:min(10, length(sorted_results))]
        
        for (i, result) in enumerate(top_results)
            improvement = get(result, "improvement", 0)
            if improvement > 0
                batch_id = get(result, "batch_id", "unknown")
                candidate_idx = get(result, "candidate_index", "unknown")
                println("   $i. Batch $batch_id, Candidate $candidate_idx: improvement = $(round(improvement, digits=6))")
            end
        end
    end
    
    # Analysis summary
    analysis = Dict(
        "total_candidates" => total_candidates,
        "successful_optimizations" => n_successful,
        "success_rate" => n_successful / total_candidates,
        "mean_improvement" => isempty(improvements) ? 0.0 : mean(improvements),
        "median_improvement" => isempty(improvements) ? 0.0 : median(improvements),
        "max_improvement" => isempty(improvements) ? 0.0 : maximum(improvements),
        "positive_improvements" => length(positive_improvements),
        "positive_improvement_rate" => isempty(improvements) ? 0.0 : length(positive_improvements) / length(improvements),
        "mean_runtime" => isempty(runtimes) ? 0.0 : mean(runtimes),
        "total_runtime" => isempty(runtimes) ? 0.0 : sum(runtimes),
        "batch_summaries" => batch_summaries,
        "error_batches" => length(error_info),
        "analysis_timestamp" => string(now())
    )
    
    return analysis
end

function create_results_dataframe(all_results)
    """Convert results to a structured DataFrame"""
    
    if isempty(all_results)
        return DataFrame()
    end
    
    # Prepare data for DataFrame
    df_data = []
    
    for result in all_results
        row = Dict(
            "batch_id" => get(result, "batch_id", missing),
            "candidate_index" => get(result, "candidate_index", missing),
            "converged" => get(result, "converged", false),
            "original_objective" => get(result, "original_objective", missing),
            "optimized_objective" => get(result, "optimized_objective", missing),
            "improvement" => get(result, "improvement", missing),
            "optimization_time" => get(result, "optimization_time", missing),
            "iterations" => get(result, "iterations", missing)
        )
        push!(df_data, row)
    end
    
    return DataFrame(df_data)
end

function save_aggregated_results(analysis, all_results, output_dir)
    """Save the aggregated analysis and results"""
    
    println("")
    println("💾 Saving aggregated results...")
    
    # Save comprehensive analysis
    analysis_file = joinpath(output_dir, "aggregated_analysis.json")
    open(analysis_file, "w") do f
        JSON.print(f, analysis, 4)
    end
    
    # Save detailed results as CSV
    if !isempty(all_results)
        df = create_results_dataframe(all_results)
        csv_file = joinpath(output_dir, "all_optimization_results.csv")
        CSV.write(csv_file, df)
        println("   📊 Detailed results saved to: $csv_file")
    end
    
    # Save best results
    successful_results = filter(r -> get(r, "converged", false), all_results)
    if !isempty(successful_results)
        improvements = [get(r, "improvement", 0) for r in successful_results]
        positive_mask = improvements .> 0
        
        if any(positive_mask)
            best_results = successful_results[positive_mask]
            sort!(best_results, by=r -> get(r, "improvement", 0), rev=true)
            
            best_file = joinpath(output_dir, "best_optimization_results.json")
            open(best_file, "w") do f
                JSON.print(f, best_results[1:min(50, length(best_results))], 4)
            end
            println("   🏆 Best results saved to: $best_file")
        end
    end
    
    # Create summary report
    report_file = joinpath(output_dir, "optimization_summary_report.txt")
    open(report_file, "w") do f
        write(f, "ENSEMBLE LOCAL OPTIMIZATION SUMMARY REPORT\n")
        write(f, "=" ^ 50 * "\n")
        write(f, "Generated: $(now())\n\n")
        
        write(f, "OVERALL PERFORMANCE:\n")
        write(f, "- Total candidates: $(analysis["total_candidates"])\n")
        write(f, "- Successful optimizations: $(analysis["successful_optimizations"])\n")
        write(f, "- Success rate: $(round(analysis["success_rate"] * 100, digits=1))%\n")
        write(f, "- Positive improvements: $(analysis["positive_improvements"])\n")
        write(f, "- Positive improvement rate: $(round(analysis["positive_improvement_rate"] * 100, digits=1))%\n\n")
        
        write(f, "IMPROVEMENT STATISTICS:\n")
        write(f, "- Mean improvement: $(round(analysis["mean_improvement"], digits=6))\n")
        write(f, "- Median improvement: $(round(analysis["median_improvement"], digits=6))\n")
        write(f, "- Maximum improvement: $(round(analysis["max_improvement"], digits=6))\n\n")
        
        write(f, "RUNTIME STATISTICS:\n")
        write(f, "- Mean runtime per candidate: $(round(analysis["mean_runtime"], digits=2))s\n")
        write(f, "- Total computation time: $(round(analysis["total_runtime"], digits=2))s\n")
        write(f, "- Total computation time: $(round(analysis["total_runtime"]/3600, digits=2)) hours\n\n")
        
        if analysis["error_batches"] > 0
            write(f, "ERRORS:\n")
            write(f, "- Failed batches: $(analysis["error_batches"])\n\n")
        end
    end
    
    println("   📋 Analysis saved to: $analysis_file")
    println("   📄 Summary report saved to: $report_file")
    
    return analysis_file, report_file
end

function main()
    println("===============================================================================")
    println("📊 ENSEMBLE OPTIMIZATION RESULTS AGGREGATION")
    println("===============================================================================")
    println("📅 Start time: $(now())")
    
    try
        # Parse arguments
        output_dir = parse_args()
        println("📁 Output directory: $output_dir")
        
        # Find result files
        println("\n🔍 Searching for result files...")
        result_files, summary_files, error_files = find_batch_results(output_dir)
        
        println("   📊 Found $(length(result_files)) result files")
        println("   📋 Found $(length(summary_files)) summary files")
        println("   ❌ Found $(length(error_files)) error files")
        
        if isempty(result_files)
            error("No optimization result files found in $output_dir")
        end
        
        # Load results
        println("\n📥 Loading optimization results...")
        all_results, batch_summaries = load_batch_results(result_files)
        error_info = load_error_info(error_files)
        
        println("   ✅ Loaded $(length(all_results)) individual results")
        println("   ✅ Loaded $(length(batch_summaries)) batch summaries")
        
        # Load configuration
        config = load_config()
        
        # Analyze results
        println("")
        analysis = analyze_results(all_results, batch_summaries, error_info, config)
        
        # Save aggregated results
        if !isempty(analysis)
            analysis_file, report_file = save_aggregated_results(analysis, all_results, output_dir)
        end
        
        println("")
        println("===============================================================================")
        println("✅ AGGREGATION COMPLETED SUCCESSFULLY")
        println("📊 Processed $(length(all_results)) optimization results")
        println("⏰ Completed at: $(now())")
        println("===============================================================================")
        
    catch e
        println("")
        println("===============================================================================")
        println("❌ AGGREGATION FAILED")
        println("💥 Error: $e")
        println("⏰ Failed at: $(now())")
        println("===============================================================================")
        
        showerror(stdout, e, catch_backtrace())
        exit(1)
    end
end

# Run the main function
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
