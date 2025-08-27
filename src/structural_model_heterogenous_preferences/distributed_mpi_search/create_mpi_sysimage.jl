# Create System Image for MPI Distributed Search
# This script creates an optimized Julia system image to reduce startup time
# and improve performance for the distributed MPI parameter search

using Pkg
Pkg.activate("/project/high_tech_ind/searching-flexibility")

# Add PackageCompiler if not already present
try
    using PackageCompiler
catch
    println("Installing PackageCompiler...")
    Pkg.add("PackageCompiler")
    using PackageCompiler
end

# Add required packages for MPI functionality
required_packages = [
    # Parallel / cluster
    "Distributed",
    "ClusterManagers",
    "SlurmClusterManager",
    "MPI",
    # Model & math
    "Parameters",
    "Distributions",
    "ForwardDiff",
    "QuadGK",
    "Interpolations",
    # Data & IO
    "DataFrames",
    "CSV",
    "Arrow",
    "YAML",
    "JSON3",
    "OrderedCollections",
    # Estimation / regression
    "FixedEffectModels",
    # Misc utilities
    "QuasiMonteCarlo",
    "Statistics",
    "LinearAlgebra",
    "Random",
    "Dates",
    "Printf",
    "Term"
]

println("🏗️  CREATING MPI SYSTEM IMAGE")
println("=" ^ 40)

# Check if all required packages are available
println("📦 Checking required packages...")
for pkg in required_packages
    try
        eval(:(using $(Symbol(pkg))))
        println("  ✓ $pkg")
    catch e
        println("  ❌ $pkg - Installing...")
        Pkg.add(pkg)
        eval(:(using $(Symbol(pkg))))
        println("  ✓ $pkg (installed)")
    end
end

# Set paths
const ROOT = "/project/high_tech_ind/searching-flexibility"
const MPI_SEARCH_DIR = joinpath(ROOT, "src", "structural_model_heterogenous_preferences", "distributed_mpi_search")
const MODEL_DIR = joinpath(ROOT, "src", "structural_model_heterogenous_preferences")

# Create precompilation script
precompile_script = joinpath(MPI_SEARCH_DIR, "precompile_mpi.jl")

precompile_code = """
# Precompilation script for MPI distributed search (extended)
using Pkg
Pkg.activate(\"$ROOT\")

using Distributed, ClusterManagers, SlurmClusterManager, MPI
using Parameters, Distributions, ForwardDiff, QuadGK, Interpolations
using DataFrames, CSV, Arrow
using YAML, JSON3, OrderedCollections
using FixedEffectModels
using QuasiMonteCarlo
using Statistics, LinearAlgebra, Random, Dates, Printf
# Term removed to reduce risk of constant redefinition across processes

println(\"✓ Core & extended packages loaded\")

const MODEL_DIR = \"$MODEL_DIR\"
const MPI_SEARCH_DIR = \"$MPI_SEARCH_DIR\"

function _tryinclude(fname)
    try
        include(joinpath(MODEL_DIR, fname))
        println(\"✓ \$fname precompiled\")
    catch e
        println(\"⚠️  Could not precompile \$fname: \$e\")
    end
end

_tryinclude(\"ModelSetup.jl\")
_tryinclude(\"ModelSolver.jl\")
_tryinclude(\"ModelEstimation.jl\")

# Config & basic YAML / JSON
try
    config_file = joinpath(MPI_SEARCH_DIR, \"mpi_search_config.yaml\")
    if isfile(config_file)
        cfg = YAML.load_file(config_file)
        println(\"✓ YAML config load\")
    end
    json_str = JSON3.write(Dict(\"a\"=>1, \"b\"=>[1,2,3]))
    JSON3.read(json_str)
    println(\"✓ JSON ops\")
catch e
    println(\"⚠️  Config/JSON precompile issue: \$e\")
end

# Light solver + simulation warmup (guarded by env)
do_sim = get(ENV, \"PRECOMPILE_SIM\", \"1\") != \"0\"
solver_steps = try parse(Int, get(ENV, \"PRECOMPILE_SOLVER_STEPS\", \"50\")) catch; 50 end

try
    if @isdefined initializeModel
        prim, res = initializeModel(joinpath(MPI_SEARCH_DIR, \"mpi_search_config.yaml\"))
        println(\"✓ initializeModel\")
        # Quick solve (reduced iterations)
        solve_model(prim, res; max_iter=solver_steps, tol=1e-4, verbose=false)
        println(\"✓ solve_model warmup\")
        # Analytic moments compile
        compute_model_moments(prim, res; include=[:mean_alpha, :var_alpha])
        println(\"✓ analytic moments warmup\")
        if do_sim && @isdefined simulate_model_data
            # Attempt simulation; small sample by truncating after load
            sim_path = joinpath(\"$ROOT\", \"data\", \"processed\", \"simulation_scaffolding_2024.feather\")
            if isfile(sim_path)
                sim_df_full = simulate_model_data(prim, res, sim_path)
                sim_df = sim_df_full[1:min(1000, nrow(sim_df_full)), :]
                compute_model_moments(prim, res, sim_df; include=[:mean_logwage, :mean_alpha])
                println(\"✓ simulation moments warmup (subset)\")
            else
                # Synthetic minimal DF for compile if file absent
                sim_df = DataFrame(u_h=rand(100), u_psi=rand(100), u_alpha=rand(100),
                                   h_values=rand(100), ψ_values=rand(100), alpha=rand(100),
                                   base_wage=rand(100), compensating_diff=rand(100),
                                   logwage=rand(100), experience=rand(100), experience_sq=rand(100),
                                   industry=repeat([\"A\",\"B\"], 50), occupation=repeat([\"X\",\"Y\"],50),
                                   educ=repeat([\"HS\",\"BA\"],50), sex=repeat([\"M\",\"F\"],50), race=repeat([\"W\",\"O\"],50))
                compute_model_moments_from_simulation(prim, res, sim_df; include_moments=[:mean_logwage])
                println(\"✓ synthetic simulation warmup\")
            end
        end
        compute_distance(Dict(:mean_alpha=>0.1), Dict(:mean_alpha=>0.05))
        println(\"✓ distance function warmup\")
    else
        println(\"⚠️  initializeModel not defined – skipped solver warmup\")
    end
catch e
    println(\"⚠️  Warmup phase issue (continuing): \$e\")
end

println(\"🎯 Extended precompilation script completed\")
"""

# Write precompilation script
open(precompile_script, "w") do f
    write(f, precompile_code)
end

println("📝 Created precompilation script: $precompile_script")

# Create system image (match naming used elsewhere)
sysimage_path = joinpath(MPI_SEARCH_DIR, "MPI_GridSearch_sysimage.so")

println("🚀 Creating system image...")
println("  Output: $sysimage_path")
println("  This may take several minutes...")

if get(ENV, "SKIP_SYSIMAGE", "0") == "1"
    println("⏭️  SKIP_SYSIMAGE=1 set; skipping system image build (precompile script executed).")
else
    try
        create_sysimage(
            required_packages;
            sysimage_path = sysimage_path,
            precompile_execution_file = precompile_script,
            cpu_target = "generic",  # cross-node compatibility
            filter_stdlibs = false
        )
        println("✅ System image created successfully!")
        if isfile(sysimage_path)
            size_mb = round(stat(sysimage_path).size / (1024^2), digits=1)
            println("📏 System image size: $(size_mb) MB")
            println("🧪 Testing system image...")
            run(`julia --startup-file=no --sysimage=$sysimage_path -e "println(\"✓ System image test successful\")"`)
            println("🎉 System image is ready. Use: julia --sysimage=$sysimage_path your_script.jl")
        else
            println("❌ System image file missing after build")
        end
    catch e
        println("❌ Failed to create system image: $e")
        println("💡 Proceed without it; startup will be slower.")
    end
end

# Clean up precompilation script
rm(precompile_script, force=true)
println("🧹 Cleaned up temporary files")
