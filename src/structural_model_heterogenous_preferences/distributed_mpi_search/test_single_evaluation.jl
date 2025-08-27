#!/usr/bin/env julia
# Quick single objective evaluation test (no GA loop)
using Pkg
Pkg.activate("/project/high_tech_ind/searching-flexibility")

const ROOT = "/project/high_tech_ind/searching-flexibility"
const SEARCH_DIR = joinpath(ROOT, "src", "structural_model_heterogenous_preferences", "distributed_mpi_search")

using YAML

config_path = joinpath(SEARCH_DIR, "mpi_search_config.yaml")
config = YAML.load_file(config_path)

include(joinpath(dirname(SEARCH_DIR), "ModelSetup.jl"))
include(joinpath(dirname(SEARCH_DIR), "ModelSolver.jl"))
include(joinpath(dirname(SEARCH_DIR), "ModelEstimation.jl"))
# Set flag to skip GA inside mpi_search when we include it
ENV["SKIP_GA"] = "1"
include(joinpath(SEARCH_DIR, "mpi_search.jl"))  # defines evaluate_objective_function after setup (GA skipped)

# Build midpoint parameter dict (reuse internal helper logic if available)
function midpoint_params(cfg)
    d = Dict{String,Float64}()
    if haskey(cfg, "MPISearchConfig") && haskey(cfg["MPISearchConfig"], "parameters")
        pconf = cfg["MPISearchConfig"]["parameters"]
        names = pconf["names"]
        bounds = pconf["bounds"]
        for nm in names
            b = bounds[nm]
            if length(b)==2 && all(x->isa(x, Number), b)
                d[string(nm)] = (float(b[1]) + float(b[2]))/2
            end
        end
    elseif haskey(cfg, "parameter_bounds")
        for (nm,b) in cfg["parameter_bounds"]
            if length(b)==2 && all(x->isa(x, Number), b)
                d[string(nm)] = (float(b[1]) + float(b[2]))/2
            end
        end
    end
    return d
end

params = midpoint_params(config)
println("Testing single evaluation with params: ", params)
obj, moments = evaluate_objective_function(params)
println("Objective value: ", obj)
println("Moments returned (subset): ", first(collect(moments), min(10, length(moments))))

missing_count = count(isnan, values(moments))
println("NaN moments: ", missing_count, " / ", length(moments))
if obj >= 7e9 || missing_count > 0
    println("❌ Test indicates issues (penalty objective or NaN moments)")
    exit(1)
else
    println("✅ Single evaluation test passed")
end
