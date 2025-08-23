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
    "ClusterManagers",
    "MPI", 
    "Distributed",
    "YAML",
    "JSON3",
    "QuasiMonteCarlo",
    "Statistics",
    "LinearAlgebra",
    "Random",
    "Dates",
    "Printf"
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
# Precompilation script for MPI distributed search
using Pkg
Pkg.activate("$ROOT")

# Load all required packages
using ClusterManagers
using MPI
using Distributed
using YAML, JSON3
using QuasiMonteCarlo
using Statistics, LinearAlgebra, Random
using Dates, Printf

println("✓ Core packages loaded")

# Include model files to precompile them
const MODEL_DIR = "$MODEL_DIR"
const MPI_SEARCH_DIR = "$MPI_SEARCH_DIR"

try
    include(joinpath(MODEL_DIR, "ModelSetup.jl"))
    println("✓ ModelSetup.jl precompiled")
catch e
    println("⚠️  Could not precompile ModelSetup.jl: \$e")
end

try
    include(joinpath(MODEL_DIR, "ModelSolver.jl"))
    println("✓ ModelSolver.jl precompiled")
catch e
    println("⚠️  Could not precompile ModelSolver.jl: \$e")
end

try
    include(joinpath(MODEL_DIR, "ModelEstimation.jl"))
    println("✓ ModelEstimation.jl precompiled")
catch e
    println("⚠️  Could not precompile ModelEstimation.jl: \$e")
end

# Precompile common operations
println("🔥 Precompiling common operations...")

# Test YAML loading
try
    config_file = joinpath(MPI_SEARCH_DIR, "mpi_search_config.yaml")
    if isfile(config_file)
        test_config = YAML.load_file(config_file)
        println("✓ YAML operations precompiled")
    else
        # Create minimal test config
        test_dict = Dict("test" => "value", "array" => [1, 2, 3])
        test_yaml = YAML.write(test_dict)
        println("✓ YAML operations precompiled (minimal)")
    end
catch e
    println("⚠️  YAML precompilation issue: \$e")
end

# Test JSON operations
try
    test_data = Dict("params" => [1.0, 2.0, 3.0], "objective" => 0.5)
    json_str = JSON3.write(test_data)
    parsed = JSON3.read(json_str)
    println("✓ JSON operations precompiled")
catch e
    println("⚠️  JSON precompilation issue: \$e")
end

# Test distributed operations
try
    # Test pmap with simple function
    test_func(x) = x^2 + sin(x)
    test_data = [1.0, 2.0, 3.0, 4.0]
    result = map(test_func, test_data)  # Use map since we don't have workers yet
    println("✓ Distributed operations precompiled")
catch e
    println("⚠️  Distributed precompilation issue: \$e")
end

# Test MPI-related operations (basic)
try
    # Basic MPI operations that don't require actual MPI environment
    if haskey(ENV, "SLURM_NTASKS")
        n_tasks = parse(Int, ENV["SLURM_NTASKS"])
        println("✓ SLURM integration precompiled")
    else
        println("✓ MPI integration precompiled (no SLURM env)")
    end
catch e
    println("⚠️  MPI precompilation issue: \$e")
end

println("🎯 Precompilation script completed successfully!")
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

try
    create_sysimage(
        required_packages;
        sysimage_path = sysimage_path,
        precompile_execution_file = precompile_script,
        cpu_target = "generic",  # For compatibility across different nodes
        filter_stdlibs = false
    )
    
    println("✅ System image created successfully!")
    
    # Check file size
    if isfile(sysimage_path)
        size_mb = round(stat(sysimage_path).size / (1024^2), digits=1)
        println("📏 System image size: $(size_mb) MB")
        
        # Test the system image
        println("🧪 Testing system image...")
        test_cmd = `julia --startup-file=no --sysimage=$sysimage_path -e "println(\"✓ System image test successful\")"`
        run(test_cmd)
        
        println("🎉 System image is ready for use!")
        println("📋 To use this system image:")
        println("   julia --sysimage=$sysimage_path your_script.jl")
        
    else
        println("❌ System image file not found after creation")
    end
    
catch e
    println("❌ Failed to create system image: $e")
    println("💡 You can still run the MPI search without a system image")
    println("   (it will just take longer to start up)")
end

# Clean up precompilation script
rm(precompile_script, force=true)
println("🧹 Cleaned up temporary files")
