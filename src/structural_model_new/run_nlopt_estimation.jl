# -------------------------
# Runner: NLopt estimation script
# -------------------------
using Printf
using NLopt

# --- 0) Include core model code ---
include("ModelSetup.jl")
include("ModelSolver.jl")
include("ModelEstimation.jl")
include("NLoptEstimation.jl")

using .NLoptEstimation

# -------------------------
# 1) Load configuration and solve baseline model
# -------------------------
config = "src/structural_model_new/model_parameters.yaml"
prim, res = initializeModel(config)
solve_model(prim, res, verbose=false)

# -------------------------
# 2) Compute baseline moments (targets)
# -------------------------
moments_baseline = compute_model_moments(prim, res)
println(@sprintf("Baseline objective vs self: %.3e",
                 sum((getfield(moments_baseline, k) - getfield(moments_baseline, k))^2
                     for k in fieldnames(typeof(moments_baseline)))))

original_params = Dict(
    :A₁ => prim.A₁, :ψ₀ => prim.ψ₀, :ν => prim.ν,
    :a_h => prim.a_h, :b_h => prim.b_h,
    :χ  => prim.χ,  :c₀ => prim.c₀,
    :ϕ  => hasfield(typeof(prim), :ϕ) ? prim.ϕ : 0.0,
    :κ₀ => hasfield(typeof(prim), :κ₀) ? prim.κ₀ : 0.0,
)
original_moments = moments_baseline

# -------------------------
# 3) Perturb parameters
# -------------------------
perturbed_params = perturb_parameters(prim; scale=0.15)
for (k, v) in perturbed_params
    setfield!(prim, k, v)
end
if haskey(perturbed_params, :ϕ);  setfield!(prim, :ϕ, perturbed_params[:ϕ]); end
if haskey(perturbed_params, :κ₀); setfield!(prim, :κ₀, perturbed_params[:κ₀]); end

_, res = update_params_and_resolve!(prim, res; verbose=false)
println("Model perturbed. Starting NLopt estimation...")

# -------------------------
# 4) Configure and run NLopt estimation
# -------------------------
nlopt_opts = (
    algorithm = :LN_BOBYQA,   # robust derivative-free with bounds
    maxeval = 500,
    xtol_rel = 1e-6,
    ftol_rel = 1e-8,
    # You can also pass lower_bounds/upper_bounds here as Dict or Vector if desired
)

result = estimate_with_nlopt(prim, res, moments_baseline; nlopt_opts=nlopt_opts, verbose=true)

println("\n=== NLopt Estimation Results ===")
println("Status: ", result[:status])
println(@sprintf("Objective: %.6e", result[:objective]))
println("Parameters:")
for k in [:A₁, :ψ₀, :ν, :a_h, :b_h, :χ, :c₀, :ϕ, :κ₀]
    @printf("  %-3s = %12.6f\n", String(k), result[:params][k])
end

# -------------------------
# 5) Compare params and moments
# -------------------------
estimated_params = result[:params]
prim_est = deepcopy(prim)
for (k, v) in estimated_params
    setfield!(prim_est, k, v)
end
_, res_est = update_params_and_resolve!(prim_est, res; verbose=false)
estimated_moments = compute_model_moments(prim_est, res_est)

recovery_success = result[:objective] < 1e-3

println("\n" * "="^80)
println("PARAMETER COMPARISON TABLE")
println("="^80)
@printf "%-12s %12s %12s %12s %12s\n" "Parameter" "Original" "Perturbed" "Estimated" "Error %"
println("-"^80)
for param in [:A₁, :ψ₀, :ν, :a_h, :b_h, :χ, :c₀, :ϕ, :κ₀]
    orig = original_params[param]
    pert = perturbed_params[param]
    est  = estimated_params[param]
    error_pct = abs(est - orig) / max(abs(orig), eps()) * 100
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
    est  = getfield(estimated_moments, field)
    error_pct = abs(est - orig) / (abs(orig) > 0 ? abs(orig) : eps()) * 100
    @printf "%-35s %12.4f %12.4f %12.2f%%\n" field orig est error_pct
end

println("\nFinal objective value: $(round(result[:objective], digits=6))")
println("Recovery ", recovery_success ? "SUCCESSFUL" : "FAILED")
println("="^80)