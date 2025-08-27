#!/usr/bin/env julia
# Minimal System Image Builder for MPI Search
# Focuses only on packages actually needed by the search / model evaluation loop,
# avoiding heavy plotting / GUI / image stacks present in the full project.

using Pkg

const ROOT = "/project/high_tech_ind/searching-flexibility"
const THIS_DIR = @__DIR__
const ENV_DIR = joinpath(THIS_DIR, "sysimage_env")

required_packages = [
    # Parallel / cluster
    "Distributed", "ClusterManagers", "SlurmClusterManager", "MPI",
    # Core model & math
    "Parameters", "Distributions", "ForwardDiff", "QuadGK", "Interpolations",
    # Data / IO
    "DataFrames", "CSV", "Arrow", "YAML", "JSON3", "OrderedCollections",
    # Estimation / regression
    "FixedEffectModels",
    # Sampling / misc
    "QuasiMonteCarlo",
    # Stdlib-like deps explicitly (help inference)
    "Statistics", "LinearAlgebra", "Random", "Dates", "Printf", "Term"
]

println("🏗️  Minimal sysimage build starting")
println("Environment dir: $ENV_DIR")
mkpath(ENV_DIR)
Pkg.activate(ENV_DIR)

# Add packages (respect JULIA_PKG_PRECOMPILE_AUTO setting; recommend setting to 0 when invoking)
for pkg in required_packages
    try
        Pkg.add(pkg)
    catch e
        @warn "Failed to add $pkg" e
    end
end

using PackageCompiler

# Precompile script to touch core model code paths without heavy loops
precompile_script = joinpath(THIS_DIR, "precompile_mpi_minimal.jl")
open(precompile_script, "w") do io
    print(io, """
        using Pkg
        Pkg.activate(raw\"$ENV_DIR\")
        using Distributed, ClusterManagers, SlurmClusterManager, MPI
        using Parameters, Distributions, ForwardDiff, QuadGK, Interpolations
        using DataFrames, CSV, Arrow, YAML, JSON3, OrderedCollections
        using FixedEffectModels, QuasiMonteCarlo
        using Statistics, LinearAlgebra, Random, Dates, Printf, Term

        const ROOT = raw\"$ROOT\"
        const MODEL_DIR = joinpath(ROOT, "src", "structural_model_heterogenous_preferences")
        include(joinpath(MODEL_DIR, "ModelSetup.jl"))
        include(joinpath(MODEL_DIR, "ModelSolver.jl"))
        include(joinpath(MODEL_DIR, "ModelEstimation.jl"))
        try
            cfg_path = joinpath(MODEL_DIR, "distributed_mpi_search", "mpi_search_config.yaml")
            if isfile(cfg_path) && @isdefined initializeModel
                prim, res = initializeModel(cfg_path)
                # light warmup
                solve_model(prim, res; max_iter=10, tol=1e-4, verbose=false)
                compute_model_moments(prim, res; include=[:mean_alpha])
            end
        catch e
            @warn "Warmup issue" e
        end
        println("✓ Minimal precompile script complete")
        """)
end

sysimage_path = joinpath(THIS_DIR, "MPI_GridSearch_sysimage.so")
println("🧱 Target sysimage: $sysimage_path")

try
    create_sysimage(required_packages; sysimage_path=sysimage_path, precompile_execution_file=precompile_script, cpu_target="generic", filter_stdlibs=false)
    println("✅ Minimal sysimage built")
    run(`julia --startup-file=no --sysimage=$sysimage_path -e "println(\"Sysimage smoke OK\")"`)
catch e
    println("❌ Failed building minimal sysimage: $e")
end

rm(precompile_script, force=true)
println("🏁 Done")
