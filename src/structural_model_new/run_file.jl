using BenchmarkTools
using Random, Statistics

include("ModelSetup.jl")
include("ModelSolver.jl")
include("ModelEstimation.jl")

# --- REPL runner ---
# Point to the configuration file
config = "src/structural_model_new/model_parameters.yaml"
# Initialize the model
prim, res = initializeModel(config)
# Solve the model
solve_model(prim, res, verbose=false)
# Compute model moments (original function)
moments_1 = compute_model_moments(prim, res)


