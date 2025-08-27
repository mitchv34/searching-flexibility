#!/usr/bin/env julia
# Auto Monitor for MPI Parameter Search
# Periodically reads the "latest" JSON (if final) or scans for newest intermediate
# plus the live file and prints concise status lines. Designed to be run from login node.

using JSON3, Dates, Printf
const _HAS_TERM = let ok = false
    try
        @eval using Term
        ok = true
    catch
        ok = false
    end
    ok
end

SEARCH_DIR = @__DIR__
RESULTS_DIR = joinpath(SEARCH_DIR, "output", "results")
LOGS_DIR = joinpath(SEARCH_DIR, "output", "logs")

const REFRESH_SECONDS = try parse(Float64, get(ENV, "MONITOR_REFRESH", "30")) catch; 30.0 end
const JOB_ID = get(ENV, "JOB_ID", nothing)

function find_latest_file()
    if !isdir(RESULTS_DIR); return nothing end
    files = filter(f->startswith(f, "mpi_search_results_job") && endswith(f, ".json"), readdir(RESULTS_DIR))
    isempty(files) && return nothing
    # choose most recent by mtime
    mtimes = Dict(f=>stat(joinpath(RESULTS_DIR,f)).mtime for f in files)
    sort!(files, by=f->mtimes[f], rev=true)
    return joinpath(RESULTS_DIR, first(files))
end

function load_json(path)
    try
        return JSON3.read(read(path,String))
    catch
        return nothing
    end
end

"""Build multi-line formatted status lines (first line summary, subsequent lines key/value)."""
function summarize(d)
    if d === nothing
        return ["no-data"]
    end
    # Core fields
    status = get(d, "status", "?")
    gen    = get(d, "ga_generation", get(d, "ga_generations_completed", missing))
    tot    = get(d, "ga_total_generations", get(d, "ga_total_generations", missing))
    prog   = Float64(get(d, "ga_progress_pct", get(d, "ga_progress", 0.0)))
    best   = Float64(get(d, "best_objective", NaN))
    evals  = get(d, "completed_evaluations", get(d, "n_evaluations", missing))
    rate   = get(d, "evaluation_rate", get(d, "avg_time_per_eval", missing))
    early  = get(d, "early_stopped", get(get(d, "early_stopping", Dict()), "early_stopped", false))
    ts     = get(d, "timestamp", "?")
    # Optional GA params
    pop    = get(d, "ga_population_size", missing)
    mut    = get(d, "ga_mutation_rate", missing)
    cross  = get(d, "ga_crossover_rate", missing)
    expands = get(d, "ga_bound_expansions", get(d, "bound_expansions", missing))
    last_expand = get(d, "last_bound_expansion_param", get(d, "last_expanded_param", missing))
    diversity = get(d, "ga_population_diversity", missing)
    median_obj = get(d, "ga_median_objective", missing)
    # First line summary
    head = @sprintf "%s gen=%s/%s (%.2f%%)" status string(gen) string(tot) prog
    # Build kv lines
    kv = Dict(
        "evals" => evals,
        "best" => isnan(best) ? "NaN" : @sprintf("%.6f", best),
        "median" => median_obj,
        "rate" => rate,
        "early" => early,
        "population" => pop,
        "mutation" => mut,
        "crossover" => cross,
        "diversity" => diversity,
        "bound_expansions" => expands,
        "last_expanded" => last_expand,
        "ts" => ts,
    )
    # Filter missing
    lines = String[]
    push!(lines, head)
    for (k,v) in kv
        (v === missing || v === nothing) && continue
        push!(lines, string("\t", k, "=", v))
    end
    if _HAS_TERM
        # Apply styling: first line bold cyan status, subsequent keys dim cyan
        try
            using Term: style, bold, color
            if !isempty(lines)
                lines[1] = style(lines[1], bold, color"bright_blue")
                for i in 2:length(lines)
                    # split key=value
                    if occursin("=", lines[i])
                        k,v = split(lines[i], "=", limit=2)
                        lines[i] = string(style(k, color"cyan"), "=", style(v, color"yellow"))
                    end
                end
            end
        catch
            # ignore styling errors
        end
    end
    return lines
end

function monitor_loop()
    println("📡 Auto monitor started (refresh=$(REFRESH_SECONDS)s) -- CTRL+C to stop")
    last_line_print = time()
    while true
        live_file = glob_first("mpi_search_results_job*_live.json")
        latest_path = live_file === nothing ? find_latest_file() : live_file
        data = latest_path === nothing ? nothing : load_json(latest_path)
        for ln in summarize(data)
            println(ln)
        end
        flush(stdout)
        sleep(REFRESH_SECONDS)
    end
end

function glob_first(pattern)
    if !isdir(RESULTS_DIR); return nothing end
    candidates = filter(f->occursin(r"_live.json$", f), readdir(RESULTS_DIR))
    isempty(candidates) && return nothing
    # pick newest
    mtimes = Dict(f=>stat(joinpath(RESULTS_DIR,f)).mtime for f in candidates)
    sort!(candidates, by=f->mtimes[f], rev=true)
    return joinpath(RESULTS_DIR, first(candidates))
end

if abspath(PROGRAM_FILE) == @__FILE__
    monitor_loop()
end
