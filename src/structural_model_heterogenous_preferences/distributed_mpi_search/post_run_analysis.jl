#!/usr/bin/env julia
using Pkg
Pkg.activate("/project/high_tech_ind/searching-flexibility")

using JSON3, Statistics, Dates, Printf

const ROOT = "/project/high_tech_ind/searching-flexibility"
const MODEL_DIR = joinpath(ROOT, "src", "structural_model_heterogenous_preferences")
const MPI_SEARCH_DIR = joinpath(MODEL_DIR, "distributed_mpi_search")
const RESULTS_DIR = joinpath(MPI_SEARCH_DIR, "output", "results")

job_id = get(ENV, "SLURM_JOB_ID", length(ARGS) > 0 ? ARGS[1] : "unknown")

function find_final_file(job_id)
    if !isdir(RESULTS_DIR)
        println("No results directory.")
        return nothing
    end
    final = joinpath(RESULTS_DIR, "mpi_search_results_job$(job_id)_final.json")
    if isfile(final)
        return final
    end
    # Fallback: pick most recent *_final.json
    finals = filter(f -> occursin(r"_final.json$", f), readdir(RESULTS_DIR))
    if isempty(finals)
        return nothing
    end
    sort!(finals)
    return joinpath(RESULTS_DIR, finals[end])
end

final_path = find_final_file(job_id)
if final_path === nothing
    println("No final results file found; exiting.")
    exit(0)
end

println("Analyzing final results file: $final_path")
data = JSON3.read(read(final_path, String))

objectives = Vector{Float64}(data["all_objectives"])
valid = filter(x -> isfinite(x) && x < 1e9, objectives)
params_matrix = data["all_params"]
names = data["parameter_names"]

function topk(k)
    idxs = collect(eachindex(objectives))
    filter!(i -> isfinite(objectives[i]) && objectives[i] < 1e9, idxs)
    sort!(idxs, by = i -> objectives[i])
    first(idxs, min(k, length(idxs)))
end

kidx = topk(10)
top_list = [Dict(
    "rank" => r,
    "objective" => objectives[i],
    "params" => Dict(string(names[j]) => params_matrix[i][j] for j in eachindex(names))
) for (r,i) in enumerate(kidx)]

quantiles = Dict(
    "min" => isempty(valid) ? NaN : minimum(valid),
    "q10" => isempty(valid) ? NaN : quantile(valid, 0.10),
    "q25" => isempty(valid) ? NaN : quantile(valid, 0.25),
    "median" => isempty(valid) ? NaN : quantile(valid, 0.50),
    "q75" => isempty(valid) ? NaN : quantile(valid, 0.75),
    "q90" => isempty(valid) ? NaN : quantile(valid, 0.90),
    "max" => isempty(valid) ? NaN : maximum(valid)
)

elapsed = get(data, "elapsed_time", NaN)
avg_time = get(data, "avg_time_per_eval", NaN)
nevals = get(data, "n_evaluations", length(objectives))
nworkers = get(data, "n_workers", 1)

est_full_runtime = isnan(avg_time) ? NaN : avg_time * nevals / max(nworkers,1)

summary = Dict(
    "job_id" => job_id,
    "timestamp" => string(Dates.now()),
    "n_evaluations" => nevals,
    "n_workers" => nworkers,
    "elapsed_time_sec" => elapsed,
    "avg_time_per_eval" => avg_time,
    "objective_quantiles" => quantiles,
    "top10" => top_list,
    "best_objective" => get(data, "best_objective", NaN),
    "estimated_total_runtime_adjusted" => est_full_runtime,
    "source_file" => final_path
)

out_json = joinpath(RESULTS_DIR, "post_analysis_job$(job_id).json")
open(out_json, "w") do f
    JSON3.pretty(f, summary)
end

# Text summary
out_txt = joinpath(RESULTS_DIR, "post_analysis_job$(job_id).txt")
open(out_txt, "w") do f
    println(f, "Post-Run Analysis (Job $job_id)")
    println(f, repeat("=", 60))
    println(f, "Final file: $final_path")
    println(f, @sprintf("Elapsed: %.2f s | Avg eval: %.4f s | Workers: %d", elapsed, avg_time, nworkers))
    println(f, "Objectives quantiles:")
    for (k,v) in quantiles
        println(f, @sprintf("  %-5s : %g", k, v))
    end
    println(f, "\nTop 5 parameter sets:")
    for entry in first(top_list, min(5, length(top_list)))
        println(f, @sprintf("  #%d obj=%.6f", entry["rank"], entry["objective"]))
    end
end

println("Post-run analysis written to:\n  $out_json\n  $out_txt")