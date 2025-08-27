using Distributed, ClusterManagers, SlurmClusterManager, MPI
using YAML, JSON3, OrderedCollections, DataFrames, CSV, Arrow
using Parameters, Distributions, ForwardDiff, QuadGK, Interpolations, QuasiMonteCarlo
using FixedEffectModels, Vcov
using Statistics, LinearAlgebra, Random, Dates, Printf
using Preferences

# Include model sources in dependency order
include(joinpath(dirname(@__DIR__), "ModelSetup.jl"))
include(joinpath(dirname(@__DIR__), "ModelSolver.jl"))
include(joinpath(dirname(@__DIR__), "ModelEstimation.jl"))
# Light-touch include of mpi_search for its function defs (avoid running full search)
include(joinpath(@__DIR__, "mpi_search.jl"))

println("Precompile driver loaded with expanded runtime package set.")

# --- Warmup Section ---------------------------------------------------------
# Goal: exercise hot paths so they are precompiled into the sysimage.
# Controlled by environment variables so CI / quick rebuilds can skip heavy work.
#   WARMUP_SOLVE=1 (default)  -> run truncated solve
#   WARMUP_SIM=1 (default)    -> run simulation-based moments on small slice
#   WARMUP_OBJECTIVE=1        -> evaluate objective once (requires config)
#   WARMUP_REG=1              -> run a tiny FixedEffectModels regression
#
const _do_solve      = get(ENV, "WARMUP_SOLVE", "1") == "1"
const _do_sim        = get(ENV, "WARMUP_SIM", "1") == "1"
const _do_obj        = get(ENV, "WARMUP_OBJECTIVE", "1") == "1"
const _do_reg        = get(ENV, "WARMUP_REG", "1") == "1"
const _solver_steps  = try parse(Int, get(ENV, "WARMUP_SOLVER_STEPS", "40")) catch; 40 end

try
	# Use existing config if present
	config_path = joinpath(@__DIR__, "mpi_search_config.yaml")
	cfg = isfile(config_path) ? YAML.load_file(config_path) : Dict()
	if _do_solve || _do_sim || _do_obj
		if @isdefined initializeModel
			prim, res = initializeModel(config_path)
			println("✓ initializeModel")
			if _do_solve && @isdefined solve_model
				try
					solve_model(prim, res; max_iter=_solver_steps, tol=1e-4, verbose=false)
					println("✓ truncated solve_model warmup ($_solver_steps iterations)")
				catch e
					println("⚠️  solve_model warmup issue: $e")
				end
			end
			if _do_sim && @isdefined simulate_model_data
				# Try feather file first; otherwise synthesize minimal DataFrame
				sim_path = joinpath(dirname(@__DIR__), "..", "data", "processed", "simulation_scaffolding_2024.feather")
				local sim_df
				try
					if isfile(sim_path)
						sim_df_full = simulate_model_data(prim, res, sim_path)
						sim_df = sim_df_full[1:min(500, nrow(sim_df_full)), :]
						println("✓ simulate_model_data subset (" * string(nrow(sim_df)) * " rows)")
					else
						sim_df = DataFrame(u_h=rand(200), u_psi=rand(200), u_alpha=rand(200),
										   h_values=rand(200), ψ_values=rand(200), alpha=rand(200),
										   base_wage=rand(200), compensating_diff=rand(200),
										   logwage=rand(200), experience=rand(200), experience_sq=rand(200),
										   industry=repeat(["A","B"], 100), occupation=repeat(["X","Y"],100),
										   educ=repeat(["HS","BA"],100), sex=repeat(["M","F"],100), race=repeat(["W","O"],100))
						println("✓ synthetic simulation DataFrame constructed")
					end
					if @isdefined compute_model_moments_from_simulation
						try
							compute_model_moments_from_simulation(prim, res, sim_df; include_moments=[:mean_logwage])
							println("✓ compute_model_moments_from_simulation warmup")
						catch e
							println("⚠️  simulation moments warmup issue: $e")
						end
					end
				catch e
					println("⚠️  simulate_model_data warmup issue: $e")
				end
			end
			if _do_obj && @isdefined evaluate_objective_function
				# Build a mid-point parameter dict from bounds, if present
				local pdict = Dict{String,Float64}()
				if haskey(cfg, "parameter_bounds")
					for (k,v) in cfg["parameter_bounds"]
						if isa(v, Vector) && length(v)==2 && all(x->isa(x,Number), v)
							pdict[string(k)] = (float(v[1])+float(v[2]))/2
						end
					end
				elseif haskey(cfg, "MPISearchConfig") && haskey(cfg["MPISearchConfig"], "parameters")
					pconf = cfg["MPISearchConfig"]["parameters"]
					if haskey(pconf, "names") && haskey(pconf, "bounds")
						for nm in pconf["names"]
							b = pconf["bounds"][nm]
							if isa(b, Vector) && length(b)==2 && all(x->isa(x,Number), b)
								pdict[string(nm)] = (float(b[1])+float(b[2]))/2
							end
						end
					end
				end
				if !isempty(pdict)
					try
						obj_val, mm = evaluate_objective_function(pdict)
						println("✓ objective warmup: objective=$(round(obj_val,digits=4)) (|moments|=$(length(mm)))")
					catch e
						println("⚠️  objective warmup issue: $e")
					end
				end
			end
		else
			println("⚠️  initializeModel not defined; skipping model warmups")
		end
	end
	if _do_reg
		try
			df_reg = DataFrame(y = randn(300), x = randn(300), fe = repeat(1:10, inner=30))
			reg(df_reg, @formula(y ~ x + fe), Vcov.cluster(:fe))
			println("✓ FixedEffectModels regression warmup")
		catch e
			println("⚠️  regression warmup issue: $e")
		end
	end
catch e
	println("⚠️  General warmup block error (continuing): $e")
end

println("Warmup section complete.")
