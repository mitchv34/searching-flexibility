using Printf

# -------------------------
# Runner: estimation script
# -------------------------
# This script performs a parameter recovery test using the model code in this
# folder. It follows these high-level steps:
#  1. Load model primitives and initialize a Results object
#  2. Compute baseline model-implied moments (used as targets)
#  3. Perturb parameters to move away from baseline
#  4. Run Ipopt to re-estimate parameters to match the baseline moments
#  5. Compare original, perturbed, and estimated parameters and moments

# --- 0) Include core model code ---
# We include the core files once from this top-level runner so that docstrings and
# method definitions are registered in the session a single time. Other helper
# modules (like `IpoptEstimation.jl`) assume these are already loaded.
include("ModelSetup.jl")     # defines `Primitives`, `Results`, and grid builders
include("ModelSolver.jl")    # solver routines (solve_inner_loop, update_* helpers)
include("ModelEstimation.jl")# estimation helpers (update_primitives!, objective wrappers)
include("IpoptEstimation.jl")# high-level Ipopt wrapper (estimate_with_ipopt)

# Make the IpoptEstimation module available in this scope
using .IpoptEstimation

# -------------------------
# 1) Load configuration and solve baseline model
# -------------------------
# Path to YAML config with parameters and grid settings. Keep this in the repo
# so the experiment is reproducible and portable.
config = "src/structural_model_new/model_parameters.yaml"

# Create primitives and results using the canonical initializer. This sets up
# parameter values, PDF/CDF grids, and any default policies embedded in Results.
prim, res = initializeModel(config)

# Solve the model once at baseline so Results holds a consistent equilibrium.
# `verbose=false` keeps console output minimal during automated runs.
solve_model(prim, res, verbose=false)

# -------------------------
# 2) Compute baseline moments (targets)
# -------------------------
# Compute the model-implied moments that we'll try to recover later. These are
# saved and used as the target moments for the estimation routine.
moments_baseline = compute_model_moments(prim, res)

# This print is a quick sanity check: objective computed against itself should be 0.
println(@sprintf("Baseline objective vs self: %.3e",
                 sum((getfield(moments_baseline, k) - getfield(moments_baseline, k))^2
                     for k in fieldnames(typeof(moments_baseline)))))

# Keep copies of the original parameters and moments so we can compare later.
# Use a Dict for parameters for easy indexing by symbol.
original_params = Dict(
    :A₁ => prim.A₁, :ψ₀ => prim.ψ₀, :ν => prim.ν,
    :a_h => prim.a_h, :b_h => prim.b_h,
    :χ  => prim.χ,  :c₀ => prim.c₀,
    :ϕ  => hasfield(typeof(prim), :ϕ) ? prim.ϕ : 0.0,
    :κ₀ => hasfield(typeof(prim), :κ₀) ? prim.κ₀ : 0.0
)
original_moments = moments_baseline

# -------------------------
# 3) Perturb parameters to move away from baseline
# -------------------------
# We perturb parameters to create a non-trivial starting point for estimation.
# `perturb_parameters` should return a Dict mapping symbols to new values. The
# scale controls the magnitude of the perturbation (15% here).
perturbed_params = perturb_parameters(prim; scale=0.15)

# Apply the perturbations to `prim` in-place so subsequent solves start from
# the perturbed parameter vector.
for (k, v) in perturbed_params
    setfield!(prim, k, v)
end

# Ensure perturbed params for newly added parameters are set on prim if present
if haskey(perturbed_params, :ϕ)
    setfield!(prim, :ϕ, perturbed_params[:ϕ])
end
if haskey(perturbed_params, :κ₀)
    setfield!(prim, :κ₀, perturbed_params[:κ₀])
end

# Resolve the model after perturbation to have a consistent Results object.
# This is important because the estimator expects `res` to be a warm-startable state.
_, res = update_params_and_resolve!(prim, res; verbose=false)
println("Model perturbed. Starting Ipopt estimation...")

# -------------------------
# 4) Configure and run Ipopt estimation
# -------------------------
# Ipopt options are passed as strings (Ipopt expects strings for option names).
# We choose moderately strict tolerances and L-BFGS Hessian approximation because
# the objective is a black-box (calls the model solver) and computing exact
# second derivatives is not practical here.
ipopt_opts = Dict(
    "max_iter" => 300,
    "tol" => 1e-6,
    "acceptable_tol" => 1e-4,
    # `print_level` controls Ipopt verbosity; set to 0 to silence solver output.
    "print_level" => 5,
    "sb" => "yes",  # suppress banner
    # We tell Ipopt to approximate jacobian via finite differences
    "jacobian_approximation" => "finite-difference-values",
    # Limited-memory Hessian (L-BFGS) is more robust for black-box objectives
    "hessian_approximation" => "limited-memory",
)

# Call the estimator. It expects `prim`, `res`, and a NamedTuple of target moments.
# It returns a Dict with `:params`, `:objective`, and `:status` for reporting.
result = estimate_with_ipopt(prim, res, moments_baseline; ipopt_opts=ipopt_opts, verbose=true)

# Print a compact summary of results so the user sees status and objective.
println("\n=== Ipopt Estimation Results ===")
println("Status: ", result[:status])
println(@sprintf("Objective: %.6e", result[:objective]))
println("Parameters:")
for k in [:A₁, :ψ₀, :ν, :a_h, :b_h, :χ, :c₀, :ϕ, :κ₀]
    # Print each estimated parameter with a consistent numeric format
    @printf("  %-3s = %12.6f\n", String(k), result[:params][k])
end

# -------------------------
# 5) Prepare comparisons (original vs perturbed vs estimated)
# -------------------------
# Extract estimated parameters from the result; this is a Dict keyed by symbol.
estimated_params = result[:params]

# To compute model-implied moments for the estimated parameter vector we apply
# the estimated params to a copy of the primitives (so we don't override the
# current `prim` used during optimization) and resolve to obtain `res_est`.
prim_est = deepcopy(prim)
for (k, v) in estimated_params
    setfield!(prim_est, k, v)
end
_, res_est = update_params_and_resolve!(prim_est, res; verbose=false)
estimated_moments = compute_model_moments(prim_est, res_est)

# Define a simple recovery success criterion (objective below threshold).
recovery_success = result[:objective] < 1e-3

# -------------------------
# 6) Print parameter and moments comparison tables
# -------------------------
# Nicely formatted comparison tables help inspect which parameters/moments were
# recovered and the magnitude of the estimation error.
println("\n" * "="^80)
println("PARAMETER COMPARISON TABLE")
println("="^80)
@printf "%-12s %12s %12s %12s %12s\n" "Parameter" "Original" "Perturbed" "Estimated" "Error %"
println("-"^80)
for param in [:A₁, :ψ₀, :ν, :a_h, :b_h, :χ, :c₀, :ϕ, :κ₀]
    # Extract values for comparison and compute percent error relative to original
    orig = original_params[param]
    pert = perturbed_params[param]
    est  = estimated_params[param]
    error_pct = abs(est - orig) / abs(orig) * 100
    @printf "%-12s %12.4f %12.4f %12.4f %12.2f%%\n" param orig pert est error_pct
end

println("\n" * "="^80)
println("MOMENTS COMPARISON TABLE")
println("="^80)
@printf "%-35s %12s %12s %12s\n" "Moment" "Original" "Estimated" "Error %"
println("-"^80)

# Iterate over fields of the NamedTuple of moments and print errors for each.
moment_fields = fieldnames(typeof(original_moments))
for field in moment_fields
    orig = getfield(original_moments, field)
    est  = getfield(estimated_moments, field)
    # Use absolute relative error; protect division by zero by taking abs(orig)
    error_pct = abs(est - orig) / (abs(orig) > 0 ? abs(orig) : eps()) * 100
    @printf "%-35s %12.4f %12.4f %12.2f%%\n" field orig est error_pct
end

println("\nFinal objective value: $(round(result[:objective], digits=6))")
println("Recovery ", recovery_success ? "SUCCESSFUL" : "FAILED")
println("="^80)
