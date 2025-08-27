# Precompilation script for MPI distributed search (extended)
using Pkg
Pkg.activate("/project/high_tech_ind/searching-flexibility")

using Distributed, ClusterManagers, SlurmClusterManager, MPI
using Parameters, Distributions, ForwardDiff, QuadGK, Interpolations
using DataFrames, CSV, Arrow
using YAML, JSON3, OrderedCollections
using FixedEffectModels
using QuasiMonteCarlo
using Statistics, LinearAlgebra, Random, Dates, Printf
# Term removed to avoid constant redefinition on workers

println("✓ Core & extended packages loaded")

const MODEL_DIR = "/project/high_tech_ind/searching-flexibility/src/structural_model_heterogenous_preferences"
const MPI_SEARCH_DIR = "/project/high_tech_ind/searching-flexibility/src/structural_model_heterogenous_preferences/distributed_mpi_search"

function _tryinclude(fname)
    try
        include(joinpath(MODEL_DIR, fname))
        println("✓ $fname precompiled")
    catch e
        println("⚠️  Could not precompile $fname: $e")
    end
end

_tryinclude("ModelSetup.jl")
_tryinclude("ModelSolver.jl")
_tryinclude("ModelEstimation.jl")

# Config & basic YAML / JSON
try
    config_file = joinpath(MPI_SEARCH_DIR, "mpi_search_config.yaml")
    if isfile(config_file)
        cfg = YAML.load_file(config_file)
        println("✓ YAML config load")
    end
    json_str = JSON3.write(Dict("a"=>1, "b"=>[1,2,3]))
    JSON3.read(json_str)
    println("✓ JSON ops")
catch e
    println("⚠️  Config/JSON precompile issue: $e")
end

# Light solver + simulation warmup (guarded by env)
do_sim = get(ENV, "PRECOMPILE_SIM", "1") != "0"
solver_steps = try parse(Int, get(ENV, "PRECOMPILE_SOLVER_STEPS", "50")) catch; 50 end

try
    if @isdefined initializeModel
        prim, res = initializeModel(joinpath(MPI_SEARCH_DIR, "mpi_search_config.yaml"))
        println("✓ initializeModel")
        # Quick solve (reduced iterations)
        solve_model(prim, res; max_iter=solver_steps, tol=1e-4, verbose=false)
        println("✓ solve_model warmup")
        # Analytic moments compile
        compute_model_moments(prim, res; include=[:mean_alpha, :var_alpha])
        println("✓ analytic moments warmup")
        if do_sim && @isdefined simulate_model_data
            # Attempt simulation; small sample by truncating after load
            sim_path = joinpath("/project/high_tech_ind/searching-flexibility", "data", "processed", "simulation_scaffolding_2024.feather")
            if isfile(sim_path)
                sim_df_full = simulate_model_data(prim, res, sim_path)
                sim_df = sim_df_full[1:min(1000, nrow(sim_df_full)), :]
                compute_model_moments(prim, res, sim_df; include=[:mean_logwage, :mean_alpha])
                println("✓ simulation moments warmup (subset)")
            else
                # Synthetic minimal DF for compile if file absent
                sim_df = DataFrame(u_h=rand(100), u_psi=rand(100), u_alpha=rand(100),
                                   h_values=rand(100), ψ_values=rand(100), alpha=rand(100),
                                   base_wage=rand(100), compensating_diff=rand(100),
                                   logwage=rand(100), experience=rand(100), experience_sq=rand(100),
                                   industry=repeat(["A","B"], 50), occupation=repeat(["X","Y"],50),
                                   educ=repeat(["HS","BA"],50), sex=repeat(["M","F"],50), race=repeat(["W","O"],50))
                compute_model_moments_from_simulation(prim, res, sim_df; include_moments=[:mean_logwage])
                println("✓ synthetic simulation warmup")
            end
        end
        compute_distance(Dict(:mean_alpha=>0.1), Dict(:mean_alpha=>0.05))
        println("✓ distance function warmup")
    else
        println("⚠️  initializeModel not defined – skipped solver warmup")
    end
catch e
    println("⚠️  Warmup phase issue (continuing): $e")
end

println("🎯 Extended precompilation script completed")
