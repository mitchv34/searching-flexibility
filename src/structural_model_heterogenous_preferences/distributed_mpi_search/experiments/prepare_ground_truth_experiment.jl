#!/usr/bin/env julia
"""
prepare_ground_truth_experiment.jl
Generates synthetic (ground truth) target moments from the estimated 2024 parameters.

Output: data/results/data_moments/data_moments_groundtruth_2024.yaml (+ json twin)
Then you can launch a recovery MPI search pointing the config to this file.

Environment overrides (optional):
  GROUND_TRUTH_PARAM_FILE  path to parameter YAML (default estimated_parameters_2024.yaml)
  OUTPUT_MOMENTS_FILE      path to write moments YAML
  BASE_MPI_CONFIG          base mpi_search_config.yaml for grids & moment list
"""

using Pkg
Pkg.activate("/project/high_tech_ind/searching-flexibility")

using YAML, JSON3, Dates, Printf

const PROJECT_ROOT = "/project/high_tech_ind/searching-flexibility"
const DEFAULT_PARAM_FILE = joinpath(PROJECT_ROOT, "data", "results", "estimated_parameters", "estimated_parameters_2024.yaml")
const DEFAULT_OUTPUT_MOMENTS_FILE = joinpath(PROJECT_ROOT, "data", "results", "data_moments", "data_moments_groundtruth_2024.yaml")
const DEFAULT_MPI_CONFIG = joinpath(PROJECT_ROOT, "src", "structural_model_heterogenous_preferences", "distributed_mpi_search", "mpi_search_config.yaml")

PARAM_FILE = get(ENV, "GROUND_TRUTH_PARAM_FILE", DEFAULT_PARAM_FILE)
OUTPUT_MOMENTS_FILE = get(ENV, "OUTPUT_MOMENTS_FILE", DEFAULT_OUTPUT_MOMENTS_FILE)
BASE_MPI_CONFIG = get(ENV, "BASE_MPI_CONFIG", DEFAULT_MPI_CONFIG)

println("🧪 Ground Truth Moment Generation")
println("  Parameters: $(PARAM_FILE)")
println("  Base MPI Config: $(BASE_MPI_CONFIG)")
println("  Output Moments: $(OUTPUT_MOMENTS_FILE)")

@assert isfile(PARAM_FILE) "Parameter file not found: $(PARAM_FILE)"
@assert isfile(BASE_MPI_CONFIG) "MPI config file not found: $(BASE_MPI_CONFIG)"

param_yaml = YAML.load_file(PARAM_FILE)
@assert haskey(param_yaml, "ModelParameters") "ModelParameters missing in parameter file"
raw_params = param_yaml["ModelParameters"]

ground_truth_params = Dict{Symbol,Float64}()
for (k,v) in raw_params
    if v isa Real
        ground_truth_params[Symbol(String(k))] = float(v)
    end
end
println("  Loaded $(length(ground_truth_params)) numeric ground truth parameters")

cfg = YAML.load_file(BASE_MPI_CONFIG)
@assert haskey(cfg, "MPISearchConfig") "MPISearchConfig block missing"
msc = cfg["MPISearchConfig"]
@assert haskey(msc, "target_moments") "target_moments missing in MPISearchConfig"
tm_cfg = msc["target_moments"]
moments_to_use = Vector{String}(tm_cfg["moments_to_use"])
println("  Moments to compute: ", join(moments_to_use, ", "))

@assert haskey(msc, "moment_computation") "moment_computation missing"
mcfg = msc["moment_computation"]
@assert haskey(mcfg, "simulated_data") "simulated_data missing from moment_computation"
sim_data_rel = String(mcfg["simulated_data"])
sim_data_path = isabspath(sim_data_rel) ? sim_data_rel : joinpath(PROJECT_ROOT, sim_data_rel)
@assert isfile(sim_data_path) "Simulation scaffolding file not found: $(sim_data_path)"

# --- Include model code (assumes these files exist) ---
model_dir = joinpath(PROJECT_ROOT, "src", "structural_model_heterogenous_preferences")
include_if_exists(path) = (isfile(path) ? include(path) : @warn("Missing include file: $(path)"))
include_if_exists(joinpath(model_dir, "ModelSetup.jl"))
include_if_exists(joinpath(model_dir, "ModelSolver.jl"))
include_if_exists(joinpath(model_dir, "ModelEstimation.jl"))
include_if_exists(joinpath(model_dir, "Simulation.jl"))

if !@isdefined(initializeModel)
    error("initializeModel not defined after includes; ensure model source files are available")
end

println("⚙️  Initializing model...")
prim, res = initializeModel(BASE_MPI_CONFIG)

if @isdefined(update_primitives_results)
    prim, res = update_primitives_results(prim, res, ground_truth_params)
else
    # Fallback: assign directly if fields exist
    for (k,v) in ground_truth_params
        try
            setfield!(prim, k, v)
        catch
            # ignore missing fields
        end
    end
end

println("🧮 Solving model under ground truth parameters ...")
if @isdefined(solve_model)
    try
        solve_model(prim, res; config=BASE_MPI_CONFIG)
    catch e
        @warn "solve_model failed: $(e)" exception=(e, catch_backtrace())
    end
else
    @warn "solve_model not defined; skipping solve"
end

println("🧪 Simulating data ...")
sim_df = nothing
if @isdefined(simulate_model_data)
    try
        sim_df = simulate_model_data(prim, res, sim_data_path)
    catch e
        @warn "simulate_model_data failed: $(e)" exception=(e, catch_backtrace())
    end
else
    @warn "simulate_model_data not defined; cannot compute simulation-based moments"
end

moments_dict = Dict{Symbol,Float64}()
if sim_df !== nothing && @isdefined(compute_model_moments_from_simulation)
    try
        sel_syms = Symbol.(moments_to_use)
        md = compute_model_moments_from_simulation(prim, res, sim_df; include_moments=sel_syms)
        for m in moments_to_use
            if haskey(md, Symbol(m)) && md[Symbol(m)] isa Real
                moments_dict[Symbol(m)] = float(md[Symbol(m)])
            end
        end
    catch e
        @warn "Moment computation failed: $(e)" exception=(e, catch_backtrace())
    end
else
    @warn "Simulation dataframe or moment function unavailable; writing nulls"
end

out_yaml = Dict(
    "Meta" => Dict(
        "generated_from" => basename(PARAM_FILE),
        "generation_date" => string(Dates.now()),
        "simulation_scaffolding" => sim_data_rel,
        "note" => "Synthetic ground truth target moments for recovery experiment"
    ),
    "DataMoments" => Dict{String,Any}()
)
for m in moments_to_use
    val = get(moments_dict, Symbol(m), nothing)
    out_yaml["DataMoments"][m] = val
end

mkpath(dirname(OUTPUT_MOMENTS_FILE))
open(OUTPUT_MOMENTS_FILE, "w") do io
    YAML.write(io, out_yaml)
end
println("✅ Wrote ground truth YAML: $(OUTPUT_MOMENTS_FILE)")

json_path = replace(OUTPUT_MOMENTS_FILE, ".yaml" => ".json")
open(json_path, "w") do io
    JSON3.pretty(io, out_yaml)
end
println("📝 JSON mirror: $(json_path)")

println("Done. Now launch MPI search with config pointing to this moments file.")
