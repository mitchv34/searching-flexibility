# ModelEstimation.jl depends on types and functions defined in ModelSetup.jl and
# ModelSolver.jl (e.g. `Primitives`, `Results`, `update_params_and_resolve`).
# To avoid re-defining docstrings and symbols when files are included multiple
# times in the same session, do NOT include the core files here. The top-level
# runner should include `ModelSetup.jl` and `ModelSolver.jl` once before
# including `ModelEstimation.jl`.

using Random, Statistics, Distributions, LinearAlgebra, ForwardDiff
using Interpolations  # Needed for linear_interpolation in simulate_model_data
using Printf, YAML

# --- Moment / Regression Constants (added) ---
# Minimum sample sizes for reliable regression-based moments
const MIN_RH_ABS = 50              # Remote + Hybrid combined observations
const MIN_INPERSON_ABS = 25        # In-person observations (lowered from 50 to reduce false degeneracy flags)
const MIN_PSI_GROUP = 30           # Each ψ quantile group size
const MIN_CLUSTER_LEVELS = 5       # Minimum distinct levels for industry / occupation

# Sentinel value used when a regression-based moment cannot be computed due to degeneracy
const SENTINEL_MOMENT = 9999.0
using CSV, DataFrames
using JSON3, YAML
using QuadGK # For numerical integration in Gumbel model
using FixedEffectModels # For regression-based moment computation

# Global cache for simulation scaffolding datasets (path => DataFrame) to avoid repeated disk I/O
const _SIM_SCAFFOLD_CACHE = Dict{String, DataFrame}()

"""
    _weighted_quantile(values, weights, q)

Calculates the q-th quantile for a set of values with corresponding weights.
This is necessary for calculating quantiles from gridded model output.
"""
function _weighted_quantile(values::AbstractVector{T}, weights::AbstractVector{T}, q::Real) where {T<:Real}
    if !(0 <= q <= 1); throw(ArgumentError("Quantile q must be in [0,1]")); end
    n = length(values)
    if n == 0; return NaN; end
    if length(weights) != n; throw(ArgumentError("values and weights must have same length")); end
    # Normalize weights
    w = copy(weights)
    s = sum(w)
    if s <= 0; throw(ArgumentError("Weights must sum to positive value")); end
    w ./= s
    # Sort by values
    idx = sortperm(values)
    vs = values[idx]
    ws = w[idx]
    cdf = cumsum(ws)
    # Find first index where cdf >= q
    for (i, c) in enumerate(cdf)
        if c >= q
            return vs[i]
        end
    end
    return last(vs)
end

function update_primitives_results(
                                    prim::Primitives,
                                    res::Results,
                                    params_to_update::Dict=Dict();
                                    kwargs...,
                                )
    # --- Stage 1: Read current values or overrides (type-stable locals) ---
    # Scalars
    A₀   = get(params_to_update, :A₀, prim.A₀)
    A₁   = get(params_to_update, :A₁, prim.A₁)
    ψ₀   = get(params_to_update, :ψ₀, prim.ψ₀)
    ϕ    = get(params_to_update, :ϕ,  prim.ϕ)
    ν    = get(params_to_update, :ν,  prim.ν)
    c₀   = get(params_to_update, :c₀, prim.c₀)
    χ    = get(params_to_update, :χ,  prim.χ)
    γ₀   = get(params_to_update, :γ₀, prim.γ₀)
    γ₁   = get(params_to_update, :γ₁, prim.γ₁)

    κ₀   = get(params_to_update, :κ₀, prim.κ₀)
    κ₁   = get(params_to_update, :κ₁, prim.κ₁)
    βv   = get(params_to_update, :β,  prim.β)
    δv   = get(params_to_update, :δ,  prim.δ)
    b    = get(params_to_update, :b,  prim.b)
    ξ    = get(params_to_update, :ξ,  prim.ξ)
    # Gumbel location parameter
    μ    = get(params_to_update, :μ, prim.μ)

    # Grids and bounds
    n_ψ   = prim.n_ψ
    ψ_min = get(params_to_update, :ψ_min, prim.ψ_min)
    ψ_max = get(params_to_update, :ψ_max, prim.ψ_max)
    ψ_grid = get(params_to_update, :ψ_grid, prim.ψ_grid)
    ψ_pdf  = get(params_to_update, :ψ_pdf,  prim.ψ_pdf)
    ψ_cdf  = get(params_to_update, :ψ_cdf,  prim.ψ_cdf)

    aₕ   = get(params_to_update, :aₕ, prim.aₕ)
    bₕ   = get(params_to_update, :bₕ, prim.bₕ)
    n_h  = prim.n_h
    h_min = get(params_to_update, :h_min, prim.h_min)
    h_max = get(params_to_update, :h_max, prim.h_max)
    h_grid = get(params_to_update, :h_grid, prim.h_grid)
    h_pdf  = get(params_to_update, :h_pdf,  prim.h_pdf)
    h_cdf  = get(params_to_update, :h_cdf,  prim.h_cdf)

    # (Beta shape parameter lower bounds enforced externally via search config.)

    # --- Stage 2: Promote numeric type if needed ---
    T_old = eltype(prim.h_grid)
    T_new = promote_type(typeof.(values(params_to_update))..., T_old)
    if T_new != T_old
        # Promote scalars
        A₀ = T_new(A₀); A₁ = T_new(A₁); ψ₀ = T_new(ψ₀); ϕ = T_new(ϕ); ν = T_new(ν)
        c₀ = T_new(c₀); χ = T_new(χ); γ₀ = T_new(γ₀); γ₁ = T_new(γ₁)
        κ₀ = T_new(κ₀); κ₁ = T_new(κ₁); βv = T_new(βv); δv = T_new(δv)
        b  = T_new(b);   ξ  = T_new(ξ);   μ  = T_new(μ)

        ψ_min = T_new(ψ_min); ψ_max = T_new(ψ_max)
        aₕ = T_new(aₕ); bₕ = T_new(bₕ)
        h_min = T_new(h_min); h_max = T_new(h_max)

        # Promote vectors
        ψ_grid = T_new.(ψ_grid); ψ_pdf = T_new.(ψ_pdf); ψ_cdf = T_new.(ψ_cdf)
        h_grid = T_new.(h_grid); h_pdf = T_new.(h_pdf); h_cdf = T_new.(h_cdf)
    end

    # --- Stage 3: Recompute h distribution only if relevant params changed ---
    recompute_h = any(k -> k in (:aₕ, :bₕ, :h_grid, :h_min, :h_max), keys(params_to_update))
    if recompute_h
        h_scaled = (h_grid .- h_min) ./ (h_max - h_min)
        beta_dist = Distributions.Beta(aₕ, bₕ)
        h_pdf_raw = pdf.(beta_dist, h_scaled)
        h_pdf = h_pdf_raw ./ sum(h_pdf_raw)
        h_cdf = cumsum(h_pdf)
        h_cdf ./= h_cdf[end]
    end

    # --- Stage 4: Build validated Primitives and fresh Results ---
    new_prim = validated_Primitives(
        A₀=A₀, A₁=A₁, ψ₀=ψ₀, ϕ=ϕ, ν=ν, c₀=c₀, χ=χ, γ₀=γ₀, γ₁=γ₁, μ=μ,
        κ₀=κ₀, κ₁=κ₁, β=βv, δ=δv, b=b, ξ=ξ,
        n_ψ=n_ψ, ψ_min=ψ_min, ψ_max=ψ_max, ψ_grid=ψ_grid, ψ_pdf=ψ_pdf, ψ_cdf=ψ_cdf,
        aₕ=aₕ, bₕ=bₕ, n_h=n_h, h_min=h_min, h_max=h_max, h_grid=h_grid, h_pdf=h_pdf, h_cdf=h_cdf,
    )

    new_res = Results(new_prim)
    return new_prim, new_res
end

function update_params_and_resolve(prim::Primitives, res::Results; params_to_update=Dict() , kwargs...)
    KW = Dict(kwargs)
    new_prim, new_res = update_primitives_results(prim, res, params_to_update; kwargs...)
    tol        = get(KW, :tol, 1e-7)
    max_iter   = get(KW, :max_iter, 25000)
    verbose    = get(KW, :verbose, false)
    λ_S        = get(KW, :λ_S, 0.01)
    λ_u        = get(KW, :λ_u, 0.01)
    solve_model(new_prim, new_res; tol=tol, max_iter=max_iter, verbose=verbose, λ_S=λ_S, λ_u=λ_u)
    return new_prim, new_res
end

function _zero_moments(θ::T; keys::Union{Nothing, Vector{Symbol}}=nothing) where {T<:Real}
    default_keys = [
        :mean_logwage, :var_logwage, :mean_alpha, :var_alpha,
        :inperson_share, :hybrid_share, :remote_share,
        :agg_productivity, :diff_alpha_high_lowpsi,
        :market_tightness, :job_finding_rate
    ]
    use_keys = isnothing(keys) ? default_keys : keys
    out = Dict{Symbol, T}()
    for k in use_keys
        out[k] = k == :market_tightness ? θ : zero(T)
    end
    return out
end

function _find_quantile_index(cdf::Vector{T}, quantile::T, n::Int)::Int where {T<:Real}
    idx = findfirst(>=(ForwardDiff.value(quantile)), ForwardDiff.value.(cdf))
    return idx === nothing ? n : idx
end

function _calculate_policy_arrays(prim::Primitives)
    @unpack n_h, n_ψ, h_grid, ψ_grid, A₀, A₁, ψ₀, ν, ϕ, c₀, μ, χ = prim
    
    h_col = reshape(h_grid, n_h, 1)
    ψ_row = reshape(ψ_grid, 1, n_ψ)
    
    A_h = A₀ .+ A₁ .* h_col
    g = @. ψ₀ * exp(ν * ψ_row + ϕ * h_col)
    
    # This is the deterministic value function V(α)
    V_alpha(α, h_idx, ψ_idx) = (A_h[h_idx] * ((1-α) + α * g[h_idx, ψ_idx])) - (c₀ * (1-α)^(1 + χ) / (1 + χ))

    # --- REVISED: Pre-calculate the LOG of the integral for numerical stability ---
    log_integral_val = Matrix{eltype(h_grid)}(undef, n_h, n_ψ)
    
    Threads.@threads for j in 1:n_ψ
        for i in 1:n_h
            # --- STEP 1: Find the maximum value of V(α) for rescaling ---
            V_func(a) = V_alpha(a, i, j)
            # Sample on a fine grid to find a robust maximum
            alpha_grid_fine = 0.0:0.01:1.0
            V_max = maximum(V_func(a) for a in alpha_grid_fine)

            # --- STEP 2: Define the STABLE integrand ---
            # The exponent is now guaranteed to be <= 0, preventing overflow.
            stable_integrand(α) = exp((V_func(α) - V_max) / μ)

            # --- STEP 3: Use quadgk on the stable integrand ---
            integral_of_exp, _ = quadgk(stable_integrand, 0.0, 1.0)

            # Under extreme curvature parameters integral_of_exp can underflow to 0.0 → log(0) = -Inf
            # That propagates to p_alpha_func => Inf / Inf → NaN PDF values → NaN CDF knots (triggering Interpolations warning).
            # Guard: if integral_of_exp is nonpositive or NaN, approximate with a tiny epsilon preserving maximum location.
            if !(isfinite(integral_of_exp)) || integral_of_exp <= 0.0
                integral_of_exp = eps(Float64)
            end

            # --- STEP 4: Combine results using the numerically stable formula ---
            log_integral_val[i, j] = (V_max / μ) + log(integral_of_exp)
        end
    end

    # --- REVISED: The PDF function now works in log-space for stability ---
    function p_alpha_func(α, h_idx, ψ_idx)
        log_numerator = V_alpha(α, h_idx, ψ_idx) / μ
        log_denominator = log_integral_val[h_idx, ψ_idx]
        
        # p(α) = exp(log_numerator - log_denominator)
        return exp(log_numerator - log_denominator)
    end

    return p_alpha_func # Return the PDF function
end

function calculate_average_policies(
                                    prim::Primitives{T}, 
                                    res::Results{T};
                                    αtol_inperson::Float64 = 0.1,
                                    αtol_remote::Float64 = 0.9
                                ) where {T<:Real}
    
    @unpack n_h, n_ψ, β, δ, ξ, c₀ = prim
    p_alpha_func = _calculate_policy_arrays(prim)
    # Placeholder throw removed – function now fully implemented below
    avg_alpha = Matrix{T}(undef, n_h, n_ψ)
    avg_alpha_sq = Matrix{T}(undef, n_h, n_ψ)
    avg_wage = Matrix{T}(undef, n_h, n_ψ)
    prob_inperson = Matrix{T}(undef, n_h, n_ψ)
    prob_remote = Matrix{T}(undef, n_h, n_ψ)
    
    # --- NEW: Matrices for the partial expectations of log wages ---
    exp_logw_inperson = Matrix{T}(undef, n_h, n_ψ)
    exp_logw_remote = Matrix{T}(undef, n_h, n_ψ)
    
    # --- NEW: Add matrices for the RH subsample ---
    prob_RH = Matrix{T}(undef, n_h, n_ψ)
    exp_logw_RH = Matrix{T}(undef, n_h, n_ψ)
    
    base_wage_component = (1 - prim.β) .* res.U .+ prim.ξ .* res.S .* (1 - prim.β * (1 - prim.δ))

    Threads.@threads for j in 1:n_ψ
        for i in 1:n_h
            # --- E[α] and E[α²] (same as before) ---
            avg_alpha[i, j], _ = quadgk(α -> α * p_alpha_func(α, i, j), 0.0, 1.0, rtol=1e-6)
            avg_alpha_sq[i, j], _ = quadgk(α -> α^2 * p_alpha_func(α, i, j), 0.0, 1.0, rtol=1e-6)
            
            # --- Probabilities (same as before) ---
            prob_inperson[i, j], _ = quadgk(α -> p_alpha_func(α, i, j), 0.0, αtol_inperson, rtol=1e-6)
            prob_remote[i, j], _ = quadgk(α -> p_alpha_func(α, i, j), αtol_remote, 1.0, rtol=1e-6)

            # --- NEW: Calculate probabilities and partial expectations for the RH subsample ---
            
            # P(RH | h,ψ) = P(α > α_tol) = ∫ from α_tol to 1 of p(α|h,ψ) dα
            prob_RH[i, j], _ = quadgk(α -> p_alpha_func(α, i, j), αtol_inperson, 1.0, rtol=1e-6)
            
            # Define the wage and log-wage functions (same as before)
            wage_func(α) = base_wage_component[i, j] + c₀ * (1 - α)
            log_wage_func(α) = log(max(1e-12, wage_func(α)))

            # E[log(w) * I(In-Person) | h,ψ] = ∫ from 0 to α_tol of log(w(α))p(α|h,ψ) dα
            exp_logw_inperson[i, j], _ = quadgk(α -> log_wage_func(α) * p_alpha_func(α, i, j), 0.0, αtol_inperson, rtol=1e-6)
            
            # E[log(w) * I(Remote) | h,ψ] = ∫ from 1-α_tol to 1 of log(w(α))p(α|h,ψ) dα
            exp_logw_remote[i, j], _ = quadgk(α -> log_wage_func(α) * p_alpha_func(α, i, j), αtol_remote, 1.0, rtol=1e-6)

            # E[log(w) * I(RH) | h,ψ] = ∫ from α_tol to 1 of log(w(α))p(α|h,ψ) dα
            exp_logw_RH[i, j], _ = quadgk(α -> log_wage_func(α) * p_alpha_func(α, i, j), αtol_inperson, 1.0, rtol=1e-6)
        end
    end
    
    @. avg_wage = base_wage_component + c₀ * (1.0 - avg_alpha)
    
    # Return all computed matrices
    return avg_alpha, avg_alpha_sq, avg_wage, 
           prob_inperson, prob_remote, 
           exp_logw_inperson, exp_logw_remote,
           prob_RH, exp_logw_RH # Add the new outputs
end

"""
    simulate_model_data(prim, res, path_to_data)

Simulate model data using the equilibrium distribution from solved model.
Loads simulation scaffolding data and generates (h, ψ, α) matches with wages.

# Arguments
- `prim::Primitives`: Solved model primitives
- `res::Results`: Solved model results
- `path_to_data::String`: Path to simulation scaffolding data (.csv or .feather)

# Returns
- `DataFrame`: Simulated dataset with columns for h_values, ψ_values, alpha, wage, logwage, etc.
"""
function simulate_model_data(prim::Primitives, res::Results, path_to_data::String)
    # --- Load (or retrieve) base simulation scaffolding ---
    base_df = let p = path_to_data
        if haskey(_SIM_SCAFFOLD_CACHE, p)
            _SIM_SCAFFOLD_CACHE[p]
        else
            # Detect format robustly
            if endswith(p, ".feather")
                _SIM_SCAFFOLD_CACHE[p] = DataFrame(Arrow.Table(p))
            elseif endswith(p, ".csv")
                _SIM_SCAFFOLD_CACHE[p] = CSV.read(p, DataFrame)
            else
                error("Unsupported file format for simulation scaffolding: $(basename(p)). Use .feather or .csv")
            end
            _SIM_SCAFFOLD_CACHE[p]
        end
    end

    # Work on a fresh copy because we add columns (mutations must not affect cache)
    sim_scaffold = copy(base_df)

    n_sim = nrow(sim_scaffold)

    # --- Step 1: Create the Samplers from the Equilibrium Distribution `n` ---

    # Marginal distribution of employed worker types (h)
    n_h_dist = vec(sum(res.n, dims=2))
    n_h_cdf = cumsum(n_h_dist) ./ sum(n_h_dist)

    # Conditional distributions of firm types (ψ) given worker type (h)
    # n_psi_given_h_cdf is a matrix where each row is a conditional CDF for that h_idx
    n_psi_given_h_dist = res.n ./ sum(res.n, dims=2)
    n_psi_given_h_cdf = cumsum(n_psi_given_h_dist, dims=2)

    # --- Step 2: Vectorized Draw of Match Types (h, ψ) ---

    # Get the pre-drawn uniform numbers
    u_h_draws = sim_scaffold.u_h
    u_psi_draws = sim_scaffold.u_psi

    # Vectorized draw for worker type index (h_idx)
    # `searchsortedfirst` finds the first index in the CDF >= the random draw
    h_idx_draws = searchsortedfirst.(Ref(n_h_cdf), u_h_draws)
    h_values = prim.h_grid[h_idx_draws]
    # Add the values and index of h to the dataframe
    sim_scaffold.h_values = h_values
    sim_scaffold.h_idx = h_idx_draws

    # Vectorized draw for firm type index (ψ_idx)
    # This is the clever part: we use the drawn h_idx to select the correct row
    # of the conditional CDF matrix for each simulated worker.
    ψ_idx_draws = zeros(Int, n_sim)
    for i in 1:n_sim
        h_idx = h_idx_draws[i]
        # Get the conditional CDF for this specific worker type
        conditional_cdf = @view n_psi_given_h_cdf[h_idx, :]
        ψ_idx_draws[i] = searchsortedfirst(conditional_cdf, u_psi_draws[i])
    end

    ψ_values = prim.ψ_grid[ψ_idx_draws]
    sim_scaffold.ψ_values = ψ_values
    sim_scaffold.ψ_idx = ψ_idx_draws

    # --- Step 3: Draw Alpha and Calculate Wages for each (h, ψ) Match ---

    # --- Step 1: Pre-calculate necessary components ---
    p_alpha_func = _calculate_policy_arrays(prim) # The PDF engine
    base_wage_component = (1 - prim.β) .* res.U .+ prim.ξ .* res.S .* (1 - prim.β * (1 - prim.δ))

    # Pre-allocate the output column
    alpha_draws = zeros(n_sim)

    # --- Step 2: The Optimized Simulation Loop (Group by match type) ---

    # Group the DataFrame by the (h,ψ) index pairs. This is the key optimization.
    grouped_sims = groupby(sim_scaffold, [:h_idx, :ψ_idx])

    alpha_fine_grid = 0.0:0.05:1.0 # Grid for numerical CDF

    for group in grouped_sims
        # For each group, all workers have the same (h,ψ) type
        h_idx = group.h_idx[1]
        ψ_idx = group.ψ_idx[1]
        
        # Get the pre-drawn random numbers for just the workers in this group
        u_alpha_group = group.u_alpha
        
        # --- Create the Inverse CDF Sampler ONCE for this group ---
        pdf_values = [p_alpha_func(α, h_idx, ψ_idx) for α in alpha_fine_grid]
        # Sanitize any non-finite values (Inf/NaN) produced by extreme parameters → set to 0 so they don't poison CDF
        @inbounds for k in eachindex(pdf_values)
            if !isfinite(pdf_values[k]) || pdf_values[k] < 0
                pdf_values[k] = 0.0
            end
        end
        
        # Handle cases where the PDF is zero everywhere (no remote work possible)
        if sum(pdf_values) < 1e-12
            # These workers will all have alpha = 0. The pre-allocated zero is correct.
            continue 
        end
        
        cdf_values = cumsum(pdf_values)
        total_mass = cdf_values[end]
        if !(isfinite(total_mass)) || total_mass <= 0
            # Degenerate fallback: assign all probability to alpha=0 (already preallocated zeros)
            continue
        end
        cdf_values ./= total_mass
        
        # Use the robust deduplication and extrapolation logic
        keep_indices = [true; cdf_values[2:end] .!= cdf_values[1:end-1]]
        knots = cdf_values[keep_indices]
        vals = alpha_fine_grid[keep_indices]

        # Final guard: if any remaining NaNs (shouldn't after sanitation) skip interpolation
        if any(!isfinite, knots) || any(!isfinite, vals)
            continue
        end
        
        # Create linear interpolator for inverse CDF sampling
        alpha_sampler = linear_interpolation(knots, vals, extrapolation_bc=Flat())

        # --- Apply the sampler to all workers in this group ---
        group_indices = parentindices(group)[1]
        alpha_draws[group_indices] = alpha_sampler.(u_alpha_group)
    end

    # Add the alpha draws to the final DataFrame
    sim_scaffold.alpha = alpha_draws

    base_wage = zeros(n_sim)
    for i ∈ 1:n_sim
        h_idx = sim_scaffold.h_idx[i]
        ψ_idx = sim_scaffold.ψ_idx[i]
        base_wage[i] = base_wage_component[h_idx, ψ_idx]
    end

    # Calculate the wage for each simulated worker
    sim_scaffold.base_wage = base_wage

    # Use the full, non-linear cost function for the compensating differential
    @. sim_scaffold.compensating_diff = prim.c₀ * (1 - sim_scaffold.alpha)^(1 + prim.χ) / (1 + prim.χ)

    sim_scaffold.wage = sim_scaffold.base_wage .+ sim_scaffold.compensating_diff
    # sim_scaffold.logwage = sim_scaffold.base_wage .+ sim_scaffold.compensating_diff #> This is a test DO NOT CHANGE
    sim_scaffold.logwage = log.(max.(1e-12, sim_scaffold.wage))

    return sim_scaffold
end


function compute_model_moments(
                                prim::Primitives{T},
                                res::Results{T};
                                q_low_cut::T=T(0.9),
                                q_high_cut::T=T(0.9),
                                αtol::T = T(0.2),
                                include::Union{Nothing, Vector{Symbol}, Symbol}=nothing
                            ) where {T<:Real}

    
    default_keys = [
        # Core distribution & dispersion
        :mean_logwage, :var_logwage, :p90_p10_logwage,
        :mean_alpha, :var_alpha,
        # Labor supply shares
        :inperson_share, :hybrid_share, :remote_share,
        # Production / technology
        :agg_productivity,
        :diff_alpha_high_lowpsi,
        :market_tightness, :job_finding_rate,
        # Wage differential & technology slope moments (needed for GA target set)
        :diff_logwage_inperson_remote, :wage_premium_high_psi, :wage_slope_psi,
        # Optional regression curvature diagnostics
        :wage_alpha, :wage_alpha_curvature
    ]

    # FIXED: Properly determine which moments to return
    if include === :all
        include_keys = default_keys
    elseif isa(include, Vector{Symbol})
        include_keys = include  # Use exactly what was requested
    elseif include === nothing
        include_keys = default_keys
    else
        throw(ArgumentError("`include` must be nothing, :all, or Vector{Symbol}"))
    end

    n::Matrix{T} = res.n

    # --- Call the NEW, fully upgraded policy function ---
    avg_alpha_policy, avg_alpha_sq_policy, avg_wage_policy, 
    prob_inperson_policy, prob_remote_policy, 
    exp_logw_inperson_policy, exp_logw_remote_policy,
    prob_RH_policy, exp_logw_RH_policy = 
        calculate_average_policies(prim, res; αtol_inperson=Float64(αtol), αtol_remote=1.0-Float64(αtol))

    @unpack h_grid, ψ_grid, ψ_cdf, ψ₀, ν, ϕ, A₀, A₁ = prim
    
    valid::BitMatrix = (n .> 0.0) .& (avg_wage_policy .> 0.0)

    total_emp::T = sum(n .* valid)
    if !(ForwardDiff.value(total_emp) > 0)
        return _zero_moments(res.θ; keys=include_keys)
    end
    
    idx_low_end::Int = _find_quantile_index(ψ_cdf, q_low_cut, length(ψ_grid))
    idx_high_start::Int = max(_find_quantile_index(ψ_cdf, .9, length(ψ_grid)), idx_low_end + 1)

    #> Masks
    is_low_psi = zeros(Bool, size(n)); is_low_psi[:, 1:idx_low_end] .= true;
    is_high_psi = zeros(Bool, size(n)); is_high_psi[:, idx_high_start:end] .= true;

    #> Accumulators 
    h_col = reshape(h_grid, prim.n_h, 1) # Create the column vector once
    production = @. (A₀ + A₁ * h_col) * ( (1 - avg_alpha_policy) + avg_alpha_policy * (ψ₀ * exp(ν * ψ_grid' + ϕ * h_col)) )

    #> Final moments
    # Define the correct denominators first for clarity
    total_employment = sum(n)
    total_valid_employment = sum(n .* valid)

    # --- Unconditional Means (denominator is total_valid_employment) ---
    mean_alpha = total_valid_employment > 0 ? sum(avg_alpha_policy .* n .* valid) / total_valid_employment : 0.0
    mean_logwage = total_valid_employment > 0 ? sum(log.(max.(1e-12, avg_wage_policy)) .* n .* valid) / total_valid_employment : 0.0
    
    # --- Unconditional Variances (use correct means and denominator) ---
    var_alpha = total_valid_employment > 0 ? sum(((avg_alpha_policy .- mean_alpha).^2) .* n .* valid) / total_valid_employment : 0.0
    var_logwage = total_valid_employment > 0 ? sum(((log.(max.(1e-12, avg_wage_policy)) .- mean_logwage).^2) .* n .* valid) / total_valid_employment : 0.0
    
    # --- P90/P10 Log Wage Ratio ---
    # We need to flatten the matrices of log wages and weights into vectors
    flat_logw_policy = vec(log.(max.(1e-12, avg_wage_policy))[valid])
    flat_weights = vec(n[valid])
    
    if !isempty(flat_weights) && sum(flat_weights) > 0
        p10_logwage = _weighted_quantile(flat_logw_policy, flat_weights, 0.10)
        p90_logwage = _weighted_quantile(flat_logw_policy, flat_weights, 0.90)
        p90_p10_logwage = p90_logwage - p10_logwage
    else
        p90_p10_logwage = T(NaN)
    end
    
    # --- Shares (denominator is total_employment) ---
    # This is now the expectation of the probability
    inperson_share = total_employment > 0 ? sum(prob_inperson_policy .* n) / total_employment : 0.0
    remote_share = total_employment > 0 ? sum(prob_remote_policy .* n) / total_employment : 0.0
    hybrid_share = 1.0 - inperson_share - remote_share

    # --- Other Aggregate Moments ---
    agg_productivity = total_employment > 0 ? sum(production .* n .* valid) / total_employment : 0.0
    market_tightness = res.θ
    job_finding_rate = res.p
    
    # --- Conditional Means ---
    sum_n_low_valid = sum(n .* is_low_psi .* valid)
    sum_n_high_valid = sum(n .* is_high_psi .* valid)
    
    mean_alpha_lowpsi = sum_n_low_valid > 0 ? sum(avg_alpha_policy .* n .* is_low_psi .* valid) / sum_n_low_valid : 0.0
    mean_alpha_highpsi = sum_n_high_valid > 0 ? sum(avg_alpha_policy .* n .* is_high_psi .* valid) / sum_n_high_valid : 0.0
    diff_alpha_high_lowpsi = mean_alpha_highpsi - mean_alpha_lowpsi

    # --- WAGE MOMENTS ---

    # --- NEW: Correctly Calculated Conditional Mean Wages ---
    
    # Numerator: E[log(w) * I(In-Person)]
    sum_logw_inperson_total = sum(exp_logw_inperson_policy .* n)
    # Denominator: P(In-Person) = E[I(In-Person)]
    sum_prob_inperson_total = sum(prob_inperson_policy .* n)
    
    mean_logwage_inperson = sum_prob_inperson_total > 0 ? sum_logw_inperson_total / sum_prob_inperson_total : 0.0

    # Do the same for remote
    sum_logw_remote_total = sum(exp_logw_remote_policy .* n)
    sum_prob_remote_total = sum(prob_remote_policy .* n)
    
    mean_logwage_remote = sum_prob_remote_total > 0 ? sum_logw_remote_total / sum_prob_remote_total : 0.0
    
    # The final moment for c₀
    diff_logwage_inperson_remote = mean_logwage_inperson - mean_logwage_remote

    # --- NEW: Correctly Calculated Conditional Mean Wages for RH Subsample ---
    
    # Denominator for low-psi RH workers: P(RH, Low ψ)
    sum_prob_RH_low_total = sum(prob_RH_policy .* n .* is_low_psi .* valid)
    # Numerator for low-psi RH workers: E[log(w) * I(RH, Low ψ)]
    sum_logw_RH_low_total = sum(exp_logw_RH_policy .* n .* is_low_psi .* valid)
    
    mean_logwage_RH_lowpsi = sum_prob_RH_low_total > 0 ? sum_logw_RH_low_total / sum_prob_RH_low_total : 0.0

    # Do the same for high-psi
    sum_prob_RH_high_total = sum(prob_RH_policy .* n .* is_high_psi .* valid)
    sum_logw_RH_high_total = sum(exp_logw_RH_policy .* n .* is_high_psi .* valid)
    
    mean_logwage_RH_highpsi = sum_prob_RH_high_total > 0 ? sum_logw_RH_high_total / sum_prob_RH_high_total : 0.0
    
    # The final moment for ψ₀ (Wage Premium)
    wage_premium_high_psi = mean_logwage_RH_highpsi - mean_logwage_RH_lowpsi
    
    # The final moment for ν (Wage Slope): in Stata this is the coefficient from
    # a regression of logwage on continuous ψ among Remote+Hybrid (RH).
    # We approximate that here analytically (no observed controls) using the
    # weighted covariance slope over the RH subsample:
    #   slope = Cov_w(logwage, ψ | RH) / Var_w(ψ | RH)
    # This replaces the earlier proxy assignment wage_slope_psi = wage_premium_high_psi.
    begin
        RH_mask = (prob_RH_policy .> 0) .& valid
        weights_RH = n .* RH_mask
        total_w_RH = sum(weights_RH)
        if total_w_RH > 0
            # Weighted means
            logw_matrix = log.(max.(1e-12, avg_wage_policy))
            ψ_mat = reshape(ψ_grid, 1, length(ψ_grid))
            ψ_expanded = repeat(ψ_mat, size(n,1), 1)
            μ_logw_RH = sum(logw_matrix .* weights_RH) / total_w_RH
            μ_ψ_RH    = sum(ψ_expanded .* weights_RH) / total_w_RH
            cov_num   = sum((logw_matrix .- μ_logw_RH) .* (ψ_expanded .- μ_ψ_RH) .* weights_RH)
            var_denom = sum(((ψ_expanded .- μ_ψ_RH).^2) .* weights_RH)
            wage_slope_psi = var_denom > 0 ? cov_num / var_denom : zero(T)
        else
            wage_slope_psi = zero(T)
        end
    end

    # Create full dictionary of ALL computed moments
    full = Dict{Symbol, T}(
        :mean_logwage => mean_logwage,
        :var_logwage => var_logwage,
        :p90_p10_logwage => p90_p10_logwage,
        :mean_alpha => mean_alpha,
        :var_alpha => var_alpha,
        :inperson_share => inperson_share,
        :hybrid_share => hybrid_share,
        :remote_share => remote_share,
        :agg_productivity => agg_productivity,
        :diff_alpha_high_lowpsi => diff_alpha_high_lowpsi,
        :market_tightness => market_tightness,
        :job_finding_rate => job_finding_rate,
        :diff_logwage_inperson_remote => diff_logwage_inperson_remote, # For c₀
        :wage_premium_high_psi => wage_premium_high_psi,             # For ψ₀
        :wage_slope_psi => wage_slope_psi                          # For ν
    )

    # FIXED: Only return the moments that were actually requested
    result = Dict{Symbol, T}()
    for k in include_keys
        if haskey(full, k)
            result[k] = full[k]
        else
            @warn "Requested moment $k not available, skipping"
            # Don't add it to result - this prevents phantom moments
        end
    end

    return result
end

"""
    compute_model_moments(prim, res, data_path; include_moments=nothing, kwargs...)

Method overload that takes a file path to simulation data (CSV or Feather format).
Loads the simulation data and calls the simulation-based moment computation.

# Arguments
- `prim::Primitives`: The solved model primitives
- `res::Results`: The solved model results
- `data_path::String`: Path to simulation scaffolding data (CSV or Feather)
- `include_moments::Union{Nothing, Vector{Symbol}}=nothing`: Which moments to compute
- `kwargs...`: Additional arguments passed to the simulation-based computation

# Returns
- `Dict{Symbol, Real}`: Dictionary of computed moments
"""
function compute_model_moments(
                                prim::Primitives,
                                res::Results,
                                data_path::String;
                                include_moments::Union{Nothing, Vector{Symbol}}=nothing,
                                q_low_cut::Float64=0.25,
                                q_high_cut::Float64=0.75,
                                #? We assume that less thatn 20% of time working remote is full in person (less than one day of 5) and similar fro full remote.
                                αtol::Float64=0.2
                            )
    
    # Generate simulated data using the simulation function
    simulated_data = simulate_model_data(prim, res, data_path)
    
    # Call the simulation-based moment computation
    return compute_model_moments_from_simulation(
        prim, res, simulated_data;
        include_moments=include_moments,
        q_low_cut=q_low_cut,
        q_high_cut=q_high_cut,
        αtol=αtol
    )
end

"""
    compute_model_moments_from_simulation(prim, res, simulated_data; kwargs...)

Compute model moments from pre-simulated data. This is the optimized version
that works with DataFrame input containing simulation results.
"""
function compute_model_moments_from_simulation(
    prim::Primitives,
    res::Results,
    simulated_data::DataFrame;
    q_low_cut::Float64=0.25,   # LOW ψ = ψ <= 25th percentile (aligned with Stata do-file)
    q_high_cut::Float64=0.75,  # HIGH ψ = ψ >= 75th percentile (aligned with Stata do-file)
    #? We assume that less thatn 20% of time working remote is full in person (less than one day of 5) and similar fro full remote.
    αtol::Float64=0.2,
    include_moments::Union{Nothing, Vector{Symbol}}=nothing
)
    
    df = simulated_data
    # NOTE: α in the simulation is the structural remote share decision variable.
    # In the empirical (Stata) construction we now reconstruct in_person / hybrid / remote from the
    # same α thresholds (≤0.2, ≥0.8). For interpretability, mean_alpha / var_alpha compare the
    # structural distribution to the proxy distribution derived identically in data, closing the
    # previous definition gap.
    
    # Determine which moments to compute
    default_keys = [
        :mean_logwage, :var_logwage, :p90_p10_logwage,
        :mean_alpha, :var_alpha,
        :inperson_share, :hybrid_share, :remote_share,
        :agg_productivity,
        :diff_alpha_high_lowpsi,
        :market_tightness, :job_finding_rate,
        :diff_logwage_inperson_remote,
        :wage_premium_high_psi,
        :wage_slope_psi,
        # Added to ensure availability for estimation & alignment with data_moments YAML
        :wage_alpha, :wage_alpha_curvature
    ]
    
    include_keys = include_moments === nothing ? default_keys : include_moments
    
    # --- 1. Pre-computation and Column Creation (Vectorized) ---
    
    # Unconditional moments that are easy to calculate upfront
    mean_logwage_uncond = mean(df.logwage)
    var_logwage_uncond = var(df.logwage)
    mean_alpha_uncond = mean(df.alpha)
    var_alpha_uncond = var(df.alpha)
    p90_p10_logwage = quantile(df.logwage, 0.9) - quantile(df.logwage, 0.1)

    # Create all necessary grouping and classification columns at once
    df.is_inperson = df.alpha .<= αtol
    df.is_remote   = df.alpha .>= (1.0 - αtol)
    df.is_hybrid   = .!df.is_inperson .& .!df.is_remote
    # New curvature-related columns
    df.alpha_sq = df.alpha .^ 2
    
    psi_q_high = quantile(df.ψ_values, q_high_cut)
    psi_q_low  = quantile(df.ψ_values, q_low_cut)
    df.is_high_psi = df.ψ_values .>= psi_q_high
    df.is_low_psi  = df.ψ_values .<= psi_q_low
    
    # --- 2. High-Performance Grouped Calculations ---
    
    # Calculate shares
    shares = combine(df, 
        :is_inperson => mean => :inperson_share,
        :is_hybrid   => mean => :hybrid_share,
        :is_remote   => mean => :remote_share
    )

    # Calculate conditional alpha means
    alpha_by_psi = combine(groupby(df, :is_high_psi), 
        :alpha => mean => :mean_alpha
    )
    # Ensure we have both groups, even if one is empty
    if nrow(alpha_by_psi) < 2
        mean_alpha_highpsi = mean_alpha_lowpsi = mean_alpha_uncond
    else
        mean_alpha_highpsi = alpha_by_psi[alpha_by_psi.is_high_psi .== true, :mean_alpha][1]
        mean_alpha_lowpsi  = alpha_by_psi[alpha_by_psi.is_high_psi .== false, :mean_alpha][1]
    end
    diff_alpha_high_lowpsi = mean_alpha_highpsi - mean_alpha_lowpsi


    n_total = nrow(df)
    n_remote = sum(df.is_remote)
    n_hybrid = sum(df.is_hybrid)
    n_inperson = sum(df.is_inperson)
    n_rh = n_remote + n_hybrid
    n_highψ = sum(df.is_high_psi)
    n_lowψ  = sum(.!df.is_high_psi)

    degeneracy_issues = String[]
    if n_rh < MIN_RH_ABS;          push!(degeneracy_issues, "tiny_RH_subsample") end
    if n_inperson < MIN_INPERSON_ABS; push!(degeneracy_issues, "tiny_inperson") end
    if n_highψ < MIN_PSI_GROUP || n_lowψ < MIN_PSI_GROUP; push!(degeneracy_issues, "unbalanced_psi_groups") end

    # Cluster level checks (only if columns exist)
        # Cluster level checks (only if columns exist) — NOT a degeneracy trigger by itself.
        function _levels_ok(col)
            col ∈ names(df) ? length(unique(df[!, col])) >= MIN_CLUSTER_LEVELS : false
        end
        cluster_levels_ok = _levels_ok(:industry) && _levels_ok(:occupation)

        # Degeneracy only if sample size / composition issues (exclude pure cluster insufficiency)
        degeneracy = any(issue -> issue != "insufficient_cluster_levels", degeneracy_issues)


    # Base controls (Mincer equation components)
    base_controls = term(:experience) + term(:experience_sq)
    demographic_fe = fe(:educ) + fe(:sex) + fe(:race)
    job_fe = fe(:industry) + fe(:occupation)    
    # Complete control set
    full_controls = base_controls + demographic_fe + job_fe

    # --- 3. Preference Parameter (c₀) ---
    # Moment for c₀: Compensating Wage Differential
    # We use FixedEffectModels.jl for performance, mirroring Stata's reghdfe.
    # The controls (X) are assumed to be in the DataFrame from the scaffolding.
    # We absorb industry and occupation as high-dimensional fixed effects.
    # Ensure regression-based moments always have a defined value (avoid UndefVarError if degeneracy)
    diff_logwage_inperson_remote = SENTINEL_MOMENT
    wage_alpha = SENTINEL_MOMENT
    wage_alpha_curvature = SENTINEL_MOMENT
    if degeneracy
        # Throttle warnings: only print up to DEGEN_WARN_LIMIT per process.
        # Use module-level refs (initialize once per process). Avoid assigning into Main explicitly
        # to prevent scope errors inside worker function contexts.
        if !isdefined(@__MODULE__, :_DEGEN_WARN_COUNT_REF)
            global _DEGEN_WARN_COUNT_REF = Ref(0)
        end
        if !isdefined(@__MODULE__, :_DEGEN_WARN_LIMIT_REF)
            lim = try parse(Int, get(ENV, "DEGEN_WARN_LIMIT", "10")) catch; 10 end
            global _DEGEN_WARN_LIMIT_REF = Ref(lim)
        end
        local count_ref = _DEGEN_WARN_COUNT_REF
        local limit_ref = _DEGEN_WARN_LIMIT_REF
        if count_ref[] < limit_ref[]
            @warn "DEGENERACY TRIGGERED: skipping regression-based moments. Sentinel moments set to $SENTINEL_MOMENT." 
            count_ref[] += 1
            if count_ref[] == limit_ref[]
                @warn "Further degeneracy warnings suppressed (limit=$(limit_ref[]))."
            end
        end
        # println("DEGENERACY TRIGGERED: skipping regression-based moments.")
        # issues_str = isempty(degeneracy_issues) ? "none listed" : join(degeneracy_issues, ", ")
        # println("  Issues: $issues_str")
        # println("  Sample sizes:")
        # println("    total                = $n_total")
        # println("    in-person (≤ αtol)   = $n_inperson (min required = $MIN_INPERSON_ABS)")
        # println("    remote (≥ 1-αtol)    = $n_remote")
        # println("    hybrid               = $n_hybrid")
        # println("    remote+hybrid (RH)   = $n_rh (min required = $MIN_RH_ABS)")
        # println("  ψ group sizes:")
        # println("    high ψ               = $n_highψ")
        # println("    low ψ                = $n_lowψ")
        # println("    min each group       = $MIN_PSI_GROUP")
        # if :industry ∈ names(df)
        #     println("    #industry levels      = $(length(unique(df.industry))) (min $MIN_CLUSTER_LEVELS)")
        # end
        # if :occupation ∈ names(df)
        #     println("    #occupation levels    = $(length(unique(df.occupation))) (min $MIN_CLUSTER_LEVELS)")
        # end
        # println("  Sentinel moments set to $SENTINEL_MOMENT.")
    # Values remain at sentinel defaults assigned above
    else
        try
            formula_c₀ = term(:logwage) ~ term(:is_hybrid) + term(:is_inperson) + full_controls
            vcov_spec = cluster_levels_ok ? Vcov.cluster(:industry, :occupation) : Vcov.robust()
            reg_c₀ = reg(df, formula_c₀, vcov_spec);
            diff_logwage_inperson_remote = coef(reg_c₀)[2]
        catch
            diff_logwage_inperson_remote = SENTINEL_MOMENT
            push!(degeneracy_issues, "regression_c₀_failed")
            degeneracy = any(issue -> issue != "insufficient_cluster_levels", degeneracy_issues)
        end
        # Regression for wage_alpha & wage_alpha_curvature (logwage ~ alpha + alpha_sq + controls)
        try
            formula_χ = term(:logwage) ~ term(:alpha) + term(:alpha_sq) + full_controls
            vcov_specχ = cluster_levels_ok ? Vcov.cluster(:industry, :occupation) : Vcov.robust()
            reg_χ = reg(df, formula_χ, vcov_specχ)
            # Coeff ordering: alpha, alpha_sq (since no intercept due to FE absorption of large sets)
            βχ = coef(reg_χ)
            wage_alpha = βχ[1]
            wage_alpha_curvature = length(βχ) >= 2 ? βχ[2] : SENTINEL_MOMENT
        catch
            wage_alpha = SENTINEL_MOMENT
            wage_alpha_curvature = SENTINEL_MOMENT
            push!(degeneracy_issues, "regression_chi_failed")
        end
    end

    # --- 4. Technology Parameters (ψ₀, ν, ϕ) ---

    # First, create the necessary subsamples and grouping variables
    ψ_q_high = quantile(df.ψ_values, q_high_cut)
    ψ_q_low  = quantile(df.ψ_values, q_low_cut)
    df.high_ψ = df.ψ_values .>= ψ_q_high

    # Create the Remote/Hybrid (RH) subsample for the wage regressions
    rh_sample = @view df[df.is_hybrid .| df.is_remote, :]

    if degeneracy
        wage_premium_high_psi = SENTINEL_MOMENT
        wage_slope_psi = SENTINEL_MOMENT
    else
        try
            formula_ψ₀ = term(:logwage) ~ term(:high_ψ)  + full_controls
            vcov_spec = cluster_levels_ok ? Vcov.cluster(:industry, :occupation) : Vcov.robust()
            reg_ψ₀ = reg(rh_sample, formula_ψ₀, vcov_spec);
            wage_premium_high_psi = coef(reg_ψ₀)[1]
        catch
            wage_premium_high_psi = SENTINEL_MOMENT
            push!(degeneracy_issues, "regression_ψ₀_failed")
            degeneracy = any(issue -> issue != "insufficient_cluster_levels", degeneracy_issues)
        end
        try
            mincer_formula_ν = term(:logwage) ~ term(:ψ_values) + full_controls
            vcov_spec2 = cluster_levels_ok ? Vcov.cluster(:industry, :occupation) : Vcov.robust()
            reg_ν = reg(df, mincer_formula_ν, vcov_spec2);
            wage_slope_psi = coef(reg_ν)[1]
        catch
            wage_slope_psi = SENTINEL_MOMENT
            push!(degeneracy_issues, "regression_ν_failed")
            degeneracy = any(issue -> issue != "insufficient_cluster_levels", degeneracy_issues)
        end
    end

    # Moment for ϕ: Difference in Average Remote Share
    mean_alpha_high_ψ = mean(df.alpha[df.high_ψ])
    mean_alpha_low_ψ  = mean(df.alpha[.! df.high_ψ])
    diff_alpha_high_lowpsi = degeneracy ? SENTINEL_MOMENT : (mean_alpha_high_ψ - mean_alpha_low_ψ)

    # --- 4. Aggregate & Search Moments ---
    agg_productivity = hasproperty(df, :production) ? mean(df.production) : 0.0
    market_tightness = res.θ
    job_finding_rate = res.p

    # --- 5. Assemble the Final Dictionary ---
    full = Dict{Symbol, Float64}(
        :mean_logwage => mean_logwage_uncond,
        :var_logwage => var_logwage_uncond,
        :p90_p10_logwage => p90_p10_logwage,
        :mean_alpha => mean_alpha_uncond,
        :var_alpha => var_alpha_uncond,
        :inperson_share => shares.inperson_share[1],
        :hybrid_share => shares.hybrid_share[1],
        :remote_share => shares.remote_share[1],
        :diff_logwage_inperson_remote => diff_logwage_inperson_remote,
        :wage_premium_high_psi => wage_premium_high_psi,
        :wage_slope_psi => wage_slope_psi,
        :diff_alpha_high_lowpsi => diff_alpha_high_lowpsi,
        :agg_productivity => agg_productivity,
        :market_tightness => market_tightness,
        :job_finding_rate => job_finding_rate,
    :wage_alpha => wage_alpha,
    :wage_alpha_curvature => wage_alpha_curvature,
        :degeneracy_flag => degeneracy ? 1.0 : 0.0
    )
    if degeneracy
        full[:degeneracy_issues_count] = length(degeneracy_issues)
    end
    
    # Filter to only requested moments
    result = Dict{Symbol, Float64}()
    for k in include_keys
        if haskey(full, k)
            result[k] = full[k]
        else
            @warn "Requested moment $k not available, skipping"
        end
    end
    
    return result
end

# -- Multiple dispatch wrapper so callers can just use `compute_model_moments` --
"""
    compute_model_moments(prim, res, simulated_data::DataFrame; include=nothing, kwargs...)

Dispatch wrapper that forwards to `compute_model_moments_from_simulation` so the caller
can use a unified interface whether moments are analytic or simulation-based.
Pass `include` (Vector{Symbol}) to select a subset of moments.
"""
function compute_model_moments(
    prim::Primitives,
    res::Results,
    simulated_data::DataFrame; include=nothing, kwargs...
)
    include_moments = include === nothing ? nothing : Vector{Symbol}(include)
    return compute_model_moments_from_simulation(prim, res, simulated_data; include_moments=include_moments, kwargs...)
end


function save_moments_to_yaml(moments::Union{NamedTuple, AbstractDict}, filename::String)
    md = Dict{String, Any}()
    for k in keys(moments)
        v = moments isa AbstractDict ? moments[k] : getproperty(moments, k)
        md[string(k)] = v
    end
    YAML.write_file(filename, md)
    println("Saved moments to: $filename")
end

function load_moments_from_yaml(filename::String; include::Union{Vector{Symbol}, Vector{String}, Nothing}=nothing)
    moments_dict_keys = ["DataMoments", "ModelMoments"]
    moments_dict_file = YAML.load_file(filename)
    if moments_dict_keys[1] ∈ keys(moments_dict_file)
        moments_dict = moments_dict_file[moments_dict_keys[1]]
    elseif moments_dict_keys[2] ∈ keys(moments_dict_file)
        moments_dict = moments_dict_file[moments_dict_keys[2]]
    else
        error("YAML file must contain either 'DataMoments' or 'ModelMoments' as a top-level key.")
    end
    if !isnothing(include)
        include_syms = Symbol.(include)
        moments_dict = Dict(k => v for (k,v) in pairs(moments_dict) if Symbol(k) in include_syms)
    end
    return Dict(Symbol(k) => v for (k,v) in pairs(moments_dict))
end

function compute_distance(
                            model_moments::Union{NamedTuple, AbstractDict},
                            data_moments::Union{NamedTuple, AbstractDict},
                            weighting_matrix::Union{Matrix, Nothing}=nothing,
                            matrix_moment_order::Union{Vector{Symbol}, Nothing}=nothing
                        )
    current_moment_order = Symbol.(collect(keys(model_moments)))
    get_val(m, k) = m isa AbstractDict ? (haskey(m, k) ? m[k] : error("missing moment $k")) :
                            (k in propertynames(m) ? getproperty(m, k) : error("missing moment $k"))

    g = [ get_val(model_moments, k) - get_val(data_moments, k) for k in current_moment_order ]

    if isnothing(weighting_matrix)
        return dot(g, g)
    end

    if !isnothing(matrix_moment_order) && current_moment_order != matrix_moment_order
        perm_indices = indexin(current_moment_order, matrix_moment_order)
        W = weighting_matrix[perm_indices, perm_indices]
    else
        W = weighting_matrix
    end

    return g' * W * g
end

function objective_function(params, p)
    params_to_update = Dict(p.param_names .=> params)
    
    # Optional warm-start S if available
    initial_S = haskey(p, :last_res) ? p.last_res[].S : nothing
    
    # Solve with updated params (deepcopy res_base to avoid mutation across calls)
    prim_new, res_new = update_primitives_results(
                                                    p.prim_base, 
                                                    deepcopy(p.res_base),
                                                    params_to_update
                                                    );
    
    # --- REVISED: Use the non-adaptive solver and check status ---
    status = solve_model(
        prim_new, res_new;
        initial_S = initial_S,
        tol      = get(p, :tol, 1e-8),
        max_iter = get(p, :max_iter, 100_000),
        verbose  = false,
        λ_S_init = get(p, :λ_S_init, 0.05),
        λ_u_init = get(p, :λ_u_init, 0.05)
    )

    # --- NEW: Penalize non-convergence ---
    if status != :converged
        return 1e10 # Return a massive penalty
    end

    # Update warm-start cache
    if haskey(p, :last_res)
        p.last_res[] = res_new
    end
    
    # --- REVISED: Support both simulation-based and analytical moment computation ---
    if haskey(p, :simulation_data_path)
        # Use simulation-based moment computation
        model_moments = compute_model_moments(
            prim_new, res_new, p.simulation_data_path;
            include_moments=collect(keys(p.target_moments))
        )
    else
        # Use analytical moment computation (original method)
        model_moments = compute_model_moments(
            prim_new, res_new; 
            include=collect(keys(p.target_moments))
        )
    end

    return compute_distance(
        model_moments,
        p.target_moments,
        get(p, :weighting_matrix, nothing),
        get(p, :matrix_moment_order, nothing)
    )
end

function perturb_parameters(prim::Primitives;
                            param_list::Vector{Symbol} = Symbol[],
                            scale::Float64=0.1)
    params = Dict{Symbol, Float64}()
    params[:A₁] = prim.A₁ * (1.0 + scale * randn())
    params[:ψ₀] = prim.ψ₀ * (1.0 + scale * randn())
    params[:ν] = max(0.1, prim.ν * (1.0 + scale * randn()))
    params[:aₕ] = max(0.5, prim.aₕ * (1.0 + scale * randn()))
    params[:bₕ] = max(0.5, prim.bₕ * (1.0 + scale * randn()))
    params[:χ] = max(0.1, prim.χ * (1.0 + scale * randn()))
    params[:c₀] = max(0.01, prim.c₀ * (1.0 + scale * randn()))
    if hasfield(typeof(prim), :ϕ)
        params[:ϕ] = max(1e-6, prim.ϕ * (1.0 + scale * randn()))
    end
    if hasfield(typeof(prim), :κ₀)
        params[:κ₀] = max(1e-6, prim.κ₀ * (1.0 + scale * randn()))
    end
    return params
end

"""
    setup_estimation_problem(prim_base, res_base, target_moments, params_to_estimate;
                             initial_param_guess,
                             use_warm_start::Bool=true,
                             ad_backend=AutoForwardDiff(),
                             lower_bound=nothing,
                             upper_bound=nothing,
                             tol::Real=1e-7,
                             max_iter::Integer=25_000,
                             λ_S_init::Real=0.01,
                             λ_u_init::Real=0.01,
                             weighting_matrix=nothing,
                             matrix_moment_order=nothing,
                             simulation_data_path=nothing,
                             burnin_kwargs=NamedTuple())

Build an OptimizationProblem for estimation. Choose warm vs cold start via `use_warm_start`.

- If `use_warm_start == true`, perform a burn-in solve at `initial_param_guess` to seed `last_res`.
- Otherwise, use `res_base` as the initial state (cold start).
- If `simulation_data_path` is provided, uses simulation-based moment computation.
"""
function setup_estimation_problem(
    prim_base, res_base, target_moments, params_to_estimate;
    initial_param_guess,
    use_warm_start::Bool=true,
    ad_backend=AutoForwardDiff(),
    lower_bound=nothing,
    upper_bound=nothing,
    tol::Real=1e-7,
    max_iter::Integer=25_000,
    λ_S_init::Real=0.01,
    λ_u_init::Real=0.01,
    weighting_matrix=nothing,
    matrix_moment_order=nothing,
    simulation_data_path=nothing,
    burnin_kwargs=NamedTuple()
)
    # --- Stage 1: optional burn-in to create a warm starting Results ---
    res_for_cache = res_base
    if use_warm_start
        try
            @info "setup_estimation_problem: warm start burn-in..."
            params_dict = Dict(params_to_estimate .=> initial_param_guess)
            prim_burn, res_burn = update_primitives_results(prim_base, deepcopy(res_base), params_dict)

            # Merge user burn-in kwargs with defaults
            burn_defaults = (verbose=false, tol=tol, max_iter=max_iter, λ_S_init=λ_S_init, λ_u_init=λ_u_init)
            burn_opts = merge(burn_defaults, burnin_kwargs)
            solve_model(prim_burn, res_burn;
                        verbose=burn_opts.verbose, tol=burn_opts.tol, max_iter=burn_opts.max_iter,
                        λ_S_init=burn_opts.λ_S_init, λ_u_init=burn_opts.λ_u_init)
            res_for_cache = res_burn
        catch e
            @warn "Burn-in failed, falling back to cold start: $e"
            res_for_cache = res_base
        end
    else
        @info "setup_estimation_problem: cold start (no burn-in)"
    end

    # --- Stage 2: build problem context ---
    last_res_ref = Ref(res_for_cache)

    p = (
        prim_base = prim_base,
        res_base = res_base,
        target_moments = target_moments,
        param_names = params_to_estimate,
        last_res = last_res_ref,         # enables warm starts in objective_function
        weighting_matrix = weighting_matrix,
        matrix_moment_order = matrix_moment_order,
        # --- REVISED: Pass robust solver controls ---
        tol = 1e-8,
        max_iter = 100_000,
        λ_S_init = 0.05,
        λ_u_init = 0.05,
        # --- NEW: Optional simulation data path ---
        simulation_data_path = simulation_data_path
    )

    # Create OptimizationProblem (SciML) with chosen AD
    opt_func = OptimizationFunction(objective_function, ad_backend)
    prob = OptimizationProblem(
        opt_func,
        collect(initial_param_guess),
        p;
        lb = lower_bound,
        ub = upper_bound
    )

    @info "setup_estimation_problem: created (warm=$(use_warm_start)) with $(length(params_to_estimate)) parameter(s)"
    return prob
end
