# Local Objective Search (Second Stage GMM/SMM Refinement)

This folder contains the tooling to take the output of the distributed / global search (GA or Sobol) and perform focused local refinement (e.g. LBFGS, multi-start polishing) on the best candidate parameter vectors.

## Key Files

| File | Purpose |
|-----------------------------|-------------------------------------------|
| `objective.jl` | Implements `setup_problem_context` and `evaluate_for_optimizer` (computes g' W g). Loads core model code (setup, solver, estimation) with guards (no module wrapper). |
| `utils.jl` | Utility module (`LocalObjectiveUtils`) with JSON GA results loader, nth / top-k candidate selection, moment key extraction, (stub) parameter transforms, caching hooks. |
| `run_search.jl` | Entry script: demo single evaluation; `evaluate_nth_candidate(n)` to re-evaluate nth GA candidate; `load_top_n_candidates(cfg)` and `build_ensemble_problem(cfg)` for multi-start local refinement via SciML `EnsembleProblem`. Supports env trigger `RUN_NTH`. |
| `local_refine_config.yaml` | Configuration for local refinement: path template to GA results with `{job_id}` placeholder, penalties, model/moment paths, optimizer and output options. |

## Workflow Overview

1.  Run (or have already run) the distributed GA search in `distributed_mpi_search/` which produces JSON result snapshots (e.g. `mpi_search_results_job3055218_latest.json`).
2.  Point the local refinement config at the desired job via `{job_id}` placeholder + either a `job_id` value inside the YAML or the environment variable `LOCAL_GA_JOB_ID`.
3.  Use `RUN_NTH=<rank>` to evaluate a ranked candidate (after objective sorting & de-duplication) or build a loop to process the top K (see snippet below).
4.  Use `build_ensemble_problem` to construct an `EnsembleProblem` and launch a parallel multi-start local optimizer (`LBFGS`) across top candidates.
5.  (Planned) Persist refined solutions and produce consolidated summaries under `local_objective_search/output`.

## Configuration (`local_refine_config.yaml`)

Important sections:

``` yaml
GlobalSearchResults:
    latest_results_file: ".../mpi_search_results_job{job_id}_latest.json"  # {job_id} placeholder
    job_id: 3055218                   # Fallback if env LOCAL_GA_JOB_ID not set
    n_top_starts: 10                  # Intended number of multi-starts (future use)
    min_relative_obj_gap: 1e-5        # Filters near-duplicates by objective
    deduplicate_exact: true
    parameter_subset: []              # Restrict parameters (empty -> all)
```

Model & moments:

``` yaml
ModelInputs:
    config_path: path/to/model/config_estimation.yaml
    data_moments_yaml: data/results/data_moments/data_moments_2024.yaml
    weighting_matrix_csv: data/results/data_moments/smm_weighting_matrix_2024.csv
    sim_data_path: data/processed/simulation_scaffolding_2024.feather
    moment_filter: []   # Optional subset of moment keys (Symbols as strings)
```

Penalties should mirror those in `objective.jl` unless intentionally altered.

## Environment Variables

| Variable | Effect |
|----------------------------------------|--------------------------------|
| `LOCAL_GA_JOB_ID` | Overrides `job_id` in config for `{job_id}` substitution in GA results path. |
| `RUN_NTH` | When set, `run_search.jl` evaluates the nth distinct best GA candidate. |

## Running Examples

Evaluate the best (rank 1) candidate using the job id inside the config:

``` bash
julia --project src/structural_model_heterogenous_preferences/local_objective_search/run_search.jl RUN_NTH=1
```

Override job id at runtime and evaluate the 3rd candidate:

``` bash
LOCAL_GA_JOB_ID=3056000 RUN_NTH=3 julia --project \
    src/structural_model_heterogenous_preferences/local_objective_search/run_search.jl
```

Just run demo (single hard‑coded parameter vector) – no env vars:

``` bash
julia --project src/structural_model_heterogenous_preferences/local_objective_search/run_search.jl
```

## Programmatic Use (Top K loop)

``` julia
include("src/structural_model_heterogenous_preferences/local_objective_search/run_search.jl")
cfg_path = "src/structural_model_heterogenous_preferences/local_objective_search/local_refine_config.yaml"
for k in 1:5
        println("-- Evaluating rank $k --")
        evaluate_nth_candidate(k; config_path=cfg_path)
end
```

## Distinct Candidate Logic

Candidates are sorted ascending by objective.

A candidate is considered *distinct* if:

1.  Its parameter vector differs (when `deduplicate_exact=true`).
2.  Its objective differs from the last accepted distinct candidate by at least `min_relative_obj_gap` (relative gap: \|Δf\| / max(\|f_prev\|, 1e-12)).

## Moment Keys

Use `extract_moment_keys(load_ga_results(path))` to inspect the moment names present in the GA JSON (`best_moments` block). To restrict moments for local evaluation supply `moment_filter` in config (string list of symbols).

## Multi-Start Ensemble Local Refinement

The functions `load_top_n_candidates` and `build_ensemble_problem` (added in `run_search.jl`) let you spin up a SciML `EnsembleProblem` for parallel local polishing of GA solutions.

Minimal example:

``` julia
using YAML, Optimization, OptimizationOptimJL, SciMLBase
include("src/structural_model_heterogenous_preferences/local_objective_search/run_search.jl")

cfg = YAML.load_file("src/structural_model_heterogenous_preferences/local_objective_search/local_refine_config.yaml")
nt = build_ensemble_problem(cfg; n_starts=cfg["GlobalSearchResults"]["n_top_starts"])

# Choose optimizer (LBFGS via OptimizationOptimJL)
alg = LBFGS()

# Solve ensemble in threaded parallel (or replace with EnsembleDistributed())
res = solve(nt.ensemble_prob, alg, EnsembleThreads(); trajectories=length(nt.start_points), iterations=1000)

final_objs = [r.objective for r in res]
best_idx = argmin(final_objs)
println("Best objective: ", final_objs[best_idx])
println("Best params: ", res[best_idx].u)
```

Notes:

-   Automatic differentiation is enabled via `Optimization.AutoForwardDiff()` if `evaluate_for_optimizer` is differentiable w.r.t. parameters.
-   If AD fails (non-differentiable regions / conditionals), switch to `OptimizationFunction((u,p)->..., Optimization.NoAD())` or supply a custom gradient.
-   Subset parameter optimization: declare `parameter_subset` in `local_refine_config.yaml`; the builder maps GA full vectors accordingly.

## Planned Enhancements (Not Yet Implemented)

-   Persist each ensemble trajectory result (parameter vector, objective, convergence status) to JSON/CSV.
-   Warm-start caching of `(prim, res)` objects to accelerate nearby evaluations.
-   Parameter transformations (log / logistic) to impose bounds smoothly.
-   Automatic top-K summary and consolidated CSV / JSON output.

## Troubleshooting

| Issue | Cause | Fix |
|---------------------------|---------------------------|-------------------|
| Error: GA results JSON not found | Wrong job id or path template not substituted | Ensure `LOCAL_GA_JOB_ID` or `job_id` in config matches an existing results file. |
| Error: No job id provided | Neither env nor config specified a job id | Set `LOCAL_GA_JOB_ID` or add `job_id` under `GlobalSearchResults`. |
| Penalty (8e9 / 7.5e9) returned | Non-convergence or missing simulation/moment code | Implement / import `simulate_model_data` & `compute_model_moments`, or inspect solver convergence. |

## Minimal Dependency Assumptions

The script relies on JSON3, YAML, DataFrames, CSV already being in the project (as used by the global search). No new packages are introduced here.

------------------------------------------------------------------------

Feel free to request the multi-start optimizer scaffold next; the structure above is designed to plug it in cleanly.