using BenchmarkTools
using Random, Statistics
using Term
using Printf
using CairoMakie

include("ModelSetup.jl")
include("ModelSolver.jl")
include("ModelEstimation.jl")

# --- REPL runner ---
# Point to the configuration file
config = "src/structural_model_new/model_parameters.yaml"
# Initialize the model
prim, res = initializeModel(config)
# Solve the model
solve_model(prim, res, verbose=true)

# Solution diagnostics
include("ModelPlotting.jl")
# --- Generate diagnostic plots (integration with ModelPlotting) ---
# Employment distribution heatmap
fig_emp = ModelPlotting.plot_employment_distribution(res, prim)
# Employment distribution with marginals
fig_emp_marg = ModelPlotting.plot_employment_distribution_with_marginals(res, prim)

fig_surplus = ModelPlotting.plot_surplus_function(res, prim)

fig_alpha = ModelPlotting.plot_alpha_policy(res, prim)

fig_wage_pol = ModelPlotting.plot_wage_policy(res, prim)

fig_wage_amenity = ModelPlotting.plot_wage_amenity_tradeoff(res, prim)

fig_outcome_skill = ModelPlotting.plot_outcomes_by_skill(res, prim)

fig_work_arrangement = ModelPlotting.plot_work_arrangement_regimes(res, prim)

fig_work_arrangement_viable = ModelPlotting.plot_work_arrangement_regimes(res, prim, gray_nonviable=true)

fig_alpha_by_firm = ModelPlotting.plot_alpha_policy_by_firm_type(res, prim)


# Compute and save baseline moments
moments_baseline = compute_model_moments(prim, res);
save_moments_to_yaml(moments_baseline, "baseline_moments.yaml")

# Test 1: Basic moment computation
println("Test 1 - Moment computation: ", isa(moments_baseline, NamedTuple) ? "PASS" : "FAIL")

# Test 2: YAML save/load roundtrip
moments_loaded = load_moments_from_yaml("baseline_moments.yaml")
println("Test 2 - YAML roundtrip: ", moments_loaded.mean_logwage ≈ moments_baseline.mean_logwage ? "PASS" : "FAIL")

# Test 3: Parameter perturbation
perturbed = perturb_parameters(prim; scale=0.1)
println("Test 3 - Parameter perturbation: ", length(perturbed) == 7 ? "PASS" : "FAIL")

# Test 4: Estimation objective evaluation
obj_val = estimation_objective(prim, res; data_moments=moments_baseline, A₁=prim.A₁*1.1)
println("Test 4 - Objective evaluation: ", obj_val >= 0.0 ? "PASS" : "FAIL")

# Test 5: Full parameter recovery test with detailed comparison
println("\n" * "="^60)
println("PARAMETER RECOVERY TEST WITH DETAILED COMPARISON")
println("="^60)

# Store original parameter values
original_params = Dict(
    :A₁ => prim.A₁, :ψ₀ => prim.ψ₀, :ν => prim.ν,
    :a_h => prim.a_h, :b_h => prim.b_h,
    :χ => prim.χ, :c₀ => prim.c₀
)

# Store original moments
original_moments = moments_baseline

# Test: Objective is ~0 at original parameters
obj_at_original = estimation_objective(prim, res; data_moments=original_moments)
println("Test 4a - Objective at original params: ", isapprox(obj_at_original, 0.0; atol=1e-10) ? "PASS" : @sprintf("FAIL (%.6e)", obj_at_original))

# Explore objective curvature around baseline for 4 key parameters and plot (2x2)
printstyled("\n" * repeat("-", 80) * "\n"; color=:cyan, bold=true)
printstyled("Objective profiles around baseline parameters\n"; color=:cyan, bold=true)

param_syms = [:A₁, :ψ₀, :ν, :χ]
rel_grid = collect(range(0.8, 1.2, length=25))  # ±20% around baseline
obj_profiles = Dict{Symbol, Tuple{Vector{Float64}, Vector{Float64}}}()

for s in param_syms
    base = getfield(prim, s)
    xvals = max.(1e-8, base .* rel_grid)
    n = length(xvals)
    yvals = Vector{Float64}(undef, n)

    printstyled(@sprintf("Evaluating objective on grid for %-3s (base=%.6g) ... ", String(s), base);
                color=:light_blue, bold=false)

    Threads.@threads for i in 1:n
        xv = xvals[i]
        # use thread-local copies to avoid race conditions
        prim_local = deepcopy(prim)
        res_local = deepcopy(res)
        setfield!(prim_local, s, xv)
        # evaluate objective on the thread-local model
        obj = estimation_objective(prim_local, res_local; data_moments=original_moments)
        yvals[i] = obj
    end

    obj_profiles[s] = (xvals, yvals)
    min_obj, argmin_idx = findmin(yvals)
    printstyled(@sprintf("done. min obj=%.6e at %.6g\n", min_obj, xvals[argmin_idx]);
                color=:green, bold=false)
end

# Plot 2x2 objective curves with baseline marker
f = Figure(size=(1000, 800))
for (idx, s) in enumerate(param_syms)
    row = Int(ceil(idx / 2))
    col = (idx % 2 == 0) ? 2 : 1
    ax = Axis(f[row, col]; title=@sprintf("Objective vs %s", String(s)), xlabel=@sprintf("%s", String(s)), ylabel="Objective")
    xvals, yvals = obj_profiles[s]
    lines!(ax, xvals, yvals, color=:dodgerblue)
    vlines!(ax, [getfield(prim, s)]; color=:red, linestyle=:dash, linewidth=2)
end
display(f) 

# Perturb parameters and track them
perturbed_params = perturb_parameters(prim; scale=0.15)
println("\nPerturbed parameters:")
for (k, v) in perturbed_params
    println("  $k: $(round(getfield(prim, k), digits=4)) → $(round(v, digits=4))")
end

# Apply perturbations
for (k, v) in perturbed_params
    setfield!(prim, k, v)
end

# Run estimation to recover parameters
target_moments = original_moments
result = simple_estimation(prim, res, target_moments; max_iter=10000, step_size=0.01)
recovery_success = result[:objective] < 1e-3

# Get estimated parameters
estimated_params = result[:params]

# Solve with estimated parameters to get estimated moments
for (k, v) in estimated_params
    setfield!(prim, k, v)
end
_, new_res = update_params_and_resolve!(prim, res; verbose=false)
estimated_moments = compute_model_moments(prim, new_res)

println("Test 5 - Parameter recovery: ", recovery_success ? "PASS" : "FAIL")

# Print comparison tables
println("\n" * "="^80)
println("PARAMETER COMPARISON TABLE")
println("="^80)
@printf "%-12s %12s %12s %12s %12s\n" "Parameter" "Original" "Perturbed" "Estimated" "Error %"
println("-"^80)
for param in [:A₁, :ψ₀, :ν, :a_h, :b_h, :χ, :c₀]
    orig = original_params[param]
    pert = perturbed_params[param]
    est = estimated_params[param]
    error_pct = abs(est - orig) / orig * 100
    @printf "%-12s %12.4f %12.4f %12.4f %12.2f%%\n" param orig pert est error_pct
end

println("\n" * "="^80)
println("MOMENTS COMPARISON TABLE")
println("="^80)
@printf "%-35s %12s %12s %12s\n" "Moment" "Original" "Estimated" "Error %"
println("-"^80)

moment_fields = fieldnames(typeof(original_moments))
for field in moment_fields
    orig = getfield(original_moments, field)
    est = getfield(estimated_moments, field)
    error_pct = abs(est - orig) / abs(orig) * 100
    @printf "%-35s %12.4f %12.4f %12.2f%%\n" field orig est error_pct
end

println("\nFinal objective value: $(round(result[:objective], digits=6))")
println("Recovery $(recovery_success ? "SUCCESSFUL" : "FAILED")")
println("="^80)


