#!/usr/bin/env julia
using Pkg
Pkg.activate("/project/high_tech_ind/searching-flexibility")
println("Rebuilding MPI Grid Search sysimage (ordered driver)...")

# Minimal-but-complete package set required at runtime for the distributed
# parameter search (simulation-only objective). This list is derived from
# explicit `using` statements across: mpi_search.jl, ModelSetup/Solver/Estimation,
# and the monitoring / analysis helpers that run during the job (JSON/YAML IO).
#
# Classification:
#  - Cluster / parallel: Distributed, ClusterManagers, SlurmClusterManager, MPI
#  - Core model & math: Parameters, Distributions, ForwardDiff, QuadGK, Interpolations, QuasiMonteCarlo
#  - Data / IO: DataFrames, CSV, Arrow, YAML, JSON3, OrderedCollections (for OrderedDict)
#  - Estimation: FixedEffectModels, Vcov (optional but present; keeps regression VCOV paths compiled)
#  - Stdlib utilities (still list to force inclusion ordering): Statistics, LinearAlgebra, Random, Dates, Printf
#  - Preferences: pulled in indirectly but listing prevents first-use JIT of extension code
#
# Excluded from earlier broader list: Term (UI only / tests), Logging (stdlib already), CairoMakie/Plots/etc (post-run analysis only),
# KernelDensity/PrettyTables/Roots/TimerOutputs (not needed for core search loop), Optimization/Optim (analytic path removed).
required_pkgs = [
    # Cluster / parallel
    "Distributed", "ClusterManagers", "SlurmClusterManager", "MPI",
    # Data & IO
    "YAML", "JSON3", "OrderedCollections", "DataFrames", "CSV", "Arrow",
    # Model & math
    "Parameters", "Distributions", "ForwardDiff", "QuadGK", "Interpolations", "QuasiMonteCarlo",
    # Estimation / regression
    "FixedEffectModels", "Vcov",
    # Misc / stdlib related (force ordering)
    "Statistics", "LinearAlgebra", "Random", "Dates", "Printf",
    # Indirect dependencies to avoid extension JIT
    "Preferences"
]
project_deps = Set(keys(Pkg.project().dependencies))
to_add = [p for p in required_pkgs if !(p in project_deps)]
if !isempty(to_add)
    @info "Adding missing packages" to_add
    try
        Pkg.add(to_add)
    catch e
        @warn "Failed adding some packages" error=e
    end
else
    @info "All required packages already present (skipping add)"
end
Pkg.instantiate()

using PackageCompiler

driver_path = joinpath(@__DIR__, "sysimage_precompile_driver.jl")
if !isfile(driver_path)
    error("Missing driver script: $(driver_path)")
end

output_so = joinpath(@__DIR__, "MPI_GridSearch_sysimage.so")
@info "Building sysimage" output_so

# Determine portable CPU target (env override allowed). Use extremely portable default.
cpu_target = get(ENV, "SYSIMAGE_CPU_TARGET", "generic")
@info "Using cpu_target" cpu_target

# Remove any existing sysimage first (avoid stale incompatible image lingering)
if isfile(output_so)
    try
        rm(output_so; force=true)
        @info "Removed existing sysimage before rebuild" output_so
    catch e
        @warn "Could not remove existing sysimage (continuing)" error=e
    end
end

try
    # Use packages list for dependency capture; execution file ensures correct include order.
    create_sysimage(required_pkgs;
        sysimage_path = output_so,
        precompile_execution_file = driver_path,
        cpu_target = cpu_target,
    )
    println("✅ Sysimage rebuilt at: " * output_so)
    # Quick smoke test to confirm it can at least load on this node
    try
    run(`julia --startup-file=no --sysimage=$(output_so) -e "println(\"✓ Sysimage smoke test OK\")"`)
    catch e
        @warn "Smoke test failed; sysimage may still be host-specific" error=e
    end
    println("Hint: export SYSIMAGE_CPU_TARGET='generic' (or 'generic;skylake;skylake-avx2') before rebuilding to adjust portability.")
catch e
    println("❌ Failed to create system image: $e")
    println("Proceed without sysimage or adjust SYSIMAGE_CPU_TARGET and retry.")
end
