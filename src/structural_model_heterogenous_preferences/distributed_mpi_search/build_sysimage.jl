#!/usr/bin/env julia
# Build a custom sysimage for the distributed GA search to reduce JIT latency
# Output: MPI_GridSearch_sysimage.so in this directory

using PackageCompiler
using Pkg

const ROOT = dirname(dirname(dirname(@__FILE__)))  # /project/.../src
const PROJECT = dirname(ROOT)                       # project root
Pkg.activate(PROJECT)
Pkg.instantiate()

# Precompile execution script: include main model components
precompile_statements = joinpath(@__DIR__, "precompile_statements.jl")
open(precompile_statements, "w") do io
    println(io, "using Distributed")
    println(io, "using YAML, Random, Statistics, LinearAlgebra, DataFrames, CSV, Arrow, FixedEffectModels, Vcov, QuasiMonteCarlo")
    # Include model files so their methods get compiled
    println(io, "include(\"../ModelSetup.jl\")")
    println(io, "include(\"../ModelSolver.jl\")")
    println(io, "include(\"../ModelEstimation.jl\")")
end

image_path = joinpath(@__DIR__, "MPI_GridSearch_sysimage.so")

create_sysimage([
    :YAML,:Random,:Statistics,:LinearAlgebra,:DataFrames,:CSV,:Arrow,:FixedEffectModels,:Vcov,:QuasiMonteCarlo,
    :SlurmClusterManager,:PackageCompiler,:StatsBase,:Interpolations,:QuadGK,:Distributions
]; precompile_execution_file=precompile_statements, sysimage_path=image_path, cpu_target="generic")

println("[Sysimage] Built sysimage at $(image_path)")
