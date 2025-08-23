#!/usr/bin/env julia

# Enhanced MPI Search Progress Monitor
# Real-time monitoring with progress tracking, ETA, and performance metrics

using JSON3, YAML
using Dates, Printf
using Statistics

# Parse command line arguments for job ID
JOB_ID = if length(ARGS) >= 1
    strip(ARGS[1])  # Job ID as string
else
    nothing  # Monitor all jobs if no arguments
end

println("📊 Enhanced MPI Search Progress Monitor")
println("=" ^ 60)
if JOB_ID !== nothing
    println("🎯 Monitoring job ID: $JOB_ID")
else
    println("🎯 Monitoring all available jobs")
end

# Configuration
const SCRIPT_DIR = dirname(@__FILE__)
const RESULTS_DIR = joinpath(SCRIPT_DIR, "output")
const LOG_DIR = joinpath(SCRIPT_DIR, "..", "..", "..", "logs")

# Create output directories if they don't exist
mkpath(RESULTS_DIR)
mkpath(LOG_DIR)

# Color codes for terminal output
const COLORS = Dict(
    :red => "\033[0;31m",
    :green => "\033[0;32m",
    :blue => "\033[0;34m",
    :yellow => "\033[1;33m",
    :purple => "\033[0;35m",
    :cyan => "\033[0;36m",
    :bold => "\033[1m",
    :reset => "\033[0m"
)

function colorize(text::String, color::Symbol)
    return "$(COLORS[color])$text$(COLORS[:reset])"
end

function find_latest_results_file()
    """
    Find the most recent results file for the specified job ID (or any job)
    """
    try
        if JOB_ID !== nothing
            pattern = "mpi_search_results_job$(JOB_ID)_*.json"
        else
            pattern = "mpi_search_results_job*_*.json"
        end
        
        # Find all matching files
        files = filter(f -> occursin(r"mpi_search_results_job.*\.json$", f), readdir(RESULTS_DIR))
        
        if JOB_ID !== nothing
            files = filter(f -> occursin("job$(JOB_ID)_", f), files)
        end
        
        if isempty(files)
            return nothing
        end
        
        # Sort by modification time (most recent first)
        full_paths = [joinpath(RESULTS_DIR, f) for f in files]
        sorted_files = sort(full_paths, by=f -> stat(f).mtime, rev=true)
        
        return sorted_files[1]
        
    catch e
        println("❌ Error finding results file: $e")
        return nothing
    end
end

function parse_search_progress(results_file::String)
    """
    Extract progress information from the results file
    """
    try
        data = JSON3.read(read(results_file, String))
        
        # Extract basic progress info
        status = get(data, "status", "unknown")
        
        # Handle both intermediate and final results files
        if haskey(data, "completed_evaluations") && haskey(data, "total_evaluations")
            # Intermediate results file
            progress = get(data, "progress", 0.0)
            completed = get(data, "completed_evaluations", 0)
            total = get(data, "total_evaluations", 0)
            elapsed_time = get(data, "elapsed_time_seconds", get(data, "elapsed_time", 0.0))
            avg_time_per_eval = get(data, "avg_time_per_eval", 0.0)
        else
            # Final results file - extract from arrays
            all_objectives = get(data, "all_objectives", [])
            completed = length(all_objectives)
            total = completed
            progress = completed > 0 ? 1.0 : 0.0
            elapsed_time = get(data, "elapsed_time", get(data, "total_time", 0.0))
            avg_time_per_eval = completed > 0 ? elapsed_time / completed : 0.0
        end
        
        best_objective = get(data, "best_objective", Inf)
        n_workers = get(data, "n_workers", 1)
        timestamp = get(data, "timestamp", "unknown")
        
        # Calculate derived metrics
        progress_pct = progress * 100
        remaining = max(0, total - completed)
        
        # Estimate time remaining
        if avg_time_per_eval > 0 && remaining > 0
            eta_seconds = avg_time_per_eval * remaining
            eta_minutes = eta_seconds / 60
            eta_hours = eta_minutes / 60
        else
            eta_seconds = 0
            eta_minutes = 0
            eta_hours = 0
        end
        
        # Calculate evaluation rate
        eval_rate = if elapsed_time > 0 && completed > 0
            completed / elapsed_time
        else
            0.0
        end
        
        return Dict(
            "status" => status,
            "progress_pct" => progress_pct,
            "completed" => completed,
            "total" => total,
            "remaining" => remaining,
            "elapsed_time" => elapsed_time,
            "elapsed_hours" => elapsed_time / 3600,
            "avg_time_per_eval" => avg_time_per_eval,
            "eta_seconds" => eta_seconds,
            "eta_minutes" => eta_minutes,
            "eta_hours" => eta_hours,
            "eval_rate" => eval_rate,
            "best_objective" => best_objective,
            "n_workers" => n_workers,
            "timestamp" => timestamp,
            "file_path" => results_file
        )
        
    catch e
        println("❌ Error parsing results file: $e")
        return nothing
    end
end

function format_time(seconds::Float64)
    """
    Format time in a human-readable way
    """
    if seconds < 60
        return @sprintf("%.1fs", seconds)
    elseif seconds < 3600
        return @sprintf("%.1fm", seconds / 60)
    else
        hours = floor(seconds / 3600)
        minutes = (seconds % 3600) / 60
        return @sprintf("%.0fh %.1fm", hours, minutes)
    end
end

function format_number(n::Int)
    """
    Format number with thousands separators
    """
    str = string(n)
    if length(str) <= 3
        return str
    end
    
    # Add commas every 3 digits from the right
    result = ""
    for (i, char) in enumerate(reverse(str))
        if i > 1 && (i - 1) % 3 == 0
            result = "," * result
        end
        result = char * result
    end
    return result
end

function format_rate(rate::Float64)
    """
    Format evaluation rate in a human-readable way
    """
    if rate < 1
        return @sprintf("%.2f evals/sec", rate)
    elseif rate < 60
        return @sprintf("%.1f evals/sec", rate)
    else
        return @sprintf("%.1f evals/min", rate * 60)
    end
end

function display_progress_bar(progress_pct::Float64, width::Int=50)
    """
    Display a text-based progress bar
    """
    filled = Int(round(progress_pct * width / 100))
    empty = width - filled
    
    bar = colorize("█" ^ filled, :green) * colorize("░" ^ empty, :cyan)
    percentage = colorize(@sprintf("%6.2f%%", progress_pct), :bold)
    
    return "[$bar] $percentage"
end

function display_progress_info(progress_info::Dict)
    """
    Display comprehensive progress information
    """
    println("\n" * "=" ^ 60)
    println(colorize("📊 MPI SEARCH PROGRESS REPORT", :bold))
    println("=" ^ 60)
    
    # Status and timestamp
    status_color = progress_info["status"] == "running" ? :green : 
                   progress_info["status"] == "completed" ? :blue : :yellow
    println("🎯 Status: ", colorize(uppercase(progress_info["status"]), status_color))
    println("🕐 Last Update: ", colorize(progress_info["timestamp"], :cyan))
    println("📁 Results File: ", colorize(basename(progress_info["file_path"]), :purple))
    
    println("\n" * colorize("📈 EVALUATION PROGRESS", :bold))
    println("-" ^ 40)
    
    # Progress bar
    progress_bar = display_progress_bar(Float64(progress_info["progress_pct"]))
    println("Progress: $progress_bar")
    
    # Sample counts
    completed = colorize(format_number(progress_info["completed"]), :green)
    total = colorize(format_number(progress_info["total"]), :blue)
    remaining = colorize(format_number(progress_info["remaining"]), :yellow)
    
    println("Samples:  $completed / $total completed ($remaining remaining)")
    
    # Performance metrics
    println("\n" * colorize("⚡ PERFORMANCE METRICS", :bold))
    println("-" ^ 40)
    
    elapsed_str = colorize(format_time(progress_info["elapsed_time"]), :green)
    println("Elapsed Time: $elapsed_str")
    
    rate_str = colorize(format_rate(progress_info["eval_rate"]), :blue)
    println("Evaluation Rate: $rate_str")
    
    workers_str = colorize(string(progress_info["n_workers"]), :purple)
    println("Active Workers: $workers_str")
    
    avg_time_str = colorize(@sprintf("%.3fs", progress_info["avg_time_per_eval"]), :cyan)
    println("Avg Time/Eval: $avg_time_str")
    
    # Time estimates
    if progress_info["remaining"] > 0
        println("\n" * colorize("⏰ TIME ESTIMATES", :bold))
        println("-" ^ 40)
        
        eta_str = colorize(format_time(progress_info["eta_seconds"]), :yellow)
        println("Estimated Remaining: $eta_str")
        
        if progress_info["eta_hours"] > 0
            completion_time = now() + Dates.Second(round(Int, progress_info["eta_seconds"]))
            completion_str = colorize(Dates.format(completion_time, "yyyy-mm-dd HH:MM"), :green)
            println("Estimated Completion: $completion_str")
        end
    end
    
    # Optimization progress
    println("\n" * colorize("🎯 OPTIMIZATION STATUS", :bold))
    println("-" ^ 40)
    
    if isfinite(progress_info["best_objective"])
        best_obj_str = colorize(@sprintf("%.6f", progress_info["best_objective"]), :green)
        println("Best Objective: $best_obj_str")
    else
        println("Best Objective: ", colorize("Not yet available", :yellow))
    end
    
    # Efficiency metrics
    if progress_info["n_workers"] > 1 && progress_info["eval_rate"] > 0
        theoretical_max_rate = progress_info["n_workers"] * (1 / progress_info["avg_time_per_eval"])
        efficiency = (progress_info["eval_rate"] / theoretical_max_rate) * 100
        efficiency_str = colorize(@sprintf("%.1f%%", efficiency), :blue)
        println("Parallel Efficiency: $efficiency_str")
    end
    
    println("=" ^ 60)
end

function save_progress_log(progress_info::Dict)
    """
    Save progress information to a log file for historical tracking
    """
    try
        log_file = joinpath(LOG_DIR, "mpi_progress_$(JOB_ID !== nothing ? "job$(JOB_ID)" : "all").log")
        
        # Create log entry
        log_entry = @sprintf("%s | Progress: %6.2f%% | Completed: %6d/%d | Rate: %6.2f evals/sec | ETA: %s | Best: %.6f\n",
            progress_info["timestamp"],
            progress_info["progress_pct"],
            progress_info["completed"],
            progress_info["total"],
            progress_info["eval_rate"],
            format_time(Float64(progress_info["eta_seconds"])),
            progress_info["best_objective"]
        )
        
        # Append to log file
        open(log_file, "a") do f
            write(f, log_entry)
        end
        
        println("📝 Progress logged to: ", colorize(basename(log_file), :purple))
        
    catch e
        println("⚠️  Failed to save progress log: $e")
    end
end

function main()
    """
    Main monitoring function
    """
    println(colorize("🔍 Searching for latest results...", :blue))
    
    results_file = find_latest_results_file()
    
    if results_file === nothing
        println("❌ No results files found for job $(JOB_ID !== nothing ? JOB_ID : "any")")
        println("💡 Make sure the MPI search is running and has started saving intermediate results")
        return
    end
    
    println("✅ Found results file: ", colorize(basename(results_file), :green))
    
    # Parse progress information
    progress_info = parse_search_progress(results_file)
    
    if progress_info === nothing
        println("❌ Failed to parse progress information")
        return
    end
    
    # Display progress information
    display_progress_info(progress_info)
    
    # Save to log
    save_progress_log(progress_info)
    
    # Summary message
    if progress_info["status"] == "running"
        if progress_info["eta_hours"] < 1
            time_msg = colorize(format_time(Float64(progress_info["eta_seconds"])), :green)
            println("\n🚀 Search is actively running! Estimated completion in $time_msg")
        else
            time_msg = colorize(format_time(Float64(progress_info["eta_seconds"])), :yellow)
            println("\n⏳ Search is running but will take some time. ETA: $time_msg")
        end
    elseif progress_info["status"] == "completed"
        println("\n🎉 Search has completed successfully!")
    else
        println("\n⚠️  Search status is uncertain: $(progress_info["status"])")
    end
end

# Run the monitor
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
