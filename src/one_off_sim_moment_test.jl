using Pkg
Pkg.activate("/project/high_tech_ind/searching-flexibility")

using YAML, DataFrames, Arrow, Random, InteractiveUtils

# Include model files directly (fresh definitions)
include("/project/high_tech_ind/searching-flexibility/src/structural_model_heterogenous_preferences/ModelSetup.jl")
include("/project/high_tech_ind/searching-flexibility/src/structural_model_heterogenous_preferences/ModelSolver.jl")
include("/project/high_tech_ind/searching-flexibility/src/structural_model_heterogenous_preferences/ModelEstimation.jl")

println("Loaded model files (fresh session)")

# Load a baseline config file referenced by mpi_search_config if available
config_path = "/project/high_tech_ind/searching-flexibility/src/structural_model_heterogenous_preferences/distributed_mpi_search/mpi_search_config.yaml"
config = YAML.load_file(config_path)

# Extract parameter bounds midpoints for a quick test param set (using new structure assumed)
param_cfg = config["MPISearchConfig"]["parameters"]
param_names = param_cfg["names"]
bounds = param_cfg["bounds"]
params_mid = Dict{Symbol, Float64}()
for nm in param_names
    b = bounds[nm]
    params_mid[Symbol(nm)] = (b[1] + b[2]) / 2
end

# Initialize base model (needs a base primitives config file; reusing config used in estimation code)
# Assuming same config file is acceptable for initializeModel
prim_base, res_base = initializeModel(config_path)

# Update to mid parameters and solve
prim_test, res_test = update_primitives_results(prim_base, res_base, params_mid)
status = solve_model(prim_test, res_test; verbose=false, tol=1e-8, max_iter=50_000)
println("Solve raw status: ", status)
status_symbol = status isa Tuple ? status[1] : status
println("Parsed status symbol: ", status_symbol)
if status_symbol != :converged
    error("Test solve did not converge (status=$(status)); aborting moments test")
end

# Path to simulation scaffolding (pick 2024 file referenced in logs)
sim_path = "/project/high_tech_ind/searching-flexibility/data/processed/simulation_scaffolding_2024.feather"
@assert isfile(sim_path) "Missing simulation scaffolding file: $sim_path"

sim_df = simulate_model_data(prim_test, res_test, sim_path)
println("Simulated rows: ", nrow(sim_df))

###############################
# Load target moments (replicate mpi_search.jl logic)
###############################
tm_conf = config["MPISearchConfig"]["target_moments"]
data_moments_file = joinpath("/project/high_tech_ind/searching-flexibility", tm_conf["data_file"])
@assert isfile(data_moments_file) "Target moments data file missing: $(data_moments_file)"
raw_data_moments = YAML.load_file(data_moments_file)
moments_to_use = tm_conf["moments_to_use"]

TARGET_MOMENTS = Dict{Symbol,Float64}()
for m in moments_to_use
    if haskey(raw_data_moments["DataMoments"], m)
        val = raw_data_moments["DataMoments"][m]
        if val !== nothing
            TARGET_MOMENTS[Symbol(m)] = Float64(val)
        else
            @info "Skipping null data moment $(m)"
        end
    else
        @warn "Requested data moment $(m) not found in file"
    end
end
@assert !isempty(TARGET_MOMENTS) "No target moments loaded"

# Build vector of moment keys we actually request from simulation
moment_keys = collect(keys(TARGET_MOMENTS))

###############################
# Compute model moments (simulation path) for required keys
###############################
moments = compute_model_moments_from_simulation(prim_test, res_test, sim_df; include_moments=moment_keys)

println("Model (simulation) moments used for objective:")
for k in moment_keys
    mv = get(moments, k, NaN)
    println("  ", k, " = ", mv)
end

###############################
# Compute objective (distance) like evaluate_objective_function
###############################
objective_value = compute_distance(moments, TARGET_MOMENTS, nothing, nothing)
println("\nObjective value (distance to data moments): ", objective_value)

println("\nMoment comparison (model vs data | diff):")
for k in moment_keys
    model_v = get(moments, k, NaN)
    data_v = TARGET_MOMENTS[k]
    diff_v = model_v - data_v
    println("  ", rpad(string(k), 28), " model=", round(model_v, digits=6), " data=", round(data_v, digits=6), " diff=", round(diff_v, digits=6))
end

# Verify dispatch via @which
println("Dispatch location for compute_model_moments_from_simulation:")
@show @which compute_model_moments_from_simulation(prim_test, res_test, sim_df)

println("ONE-OFF TEST COMPLETED OK")
