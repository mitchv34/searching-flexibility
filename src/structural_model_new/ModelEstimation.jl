# ModelEstimation.jl depends on types and functions defined in ModelSetup.jl and
# ModelSolver.jl (e.g. `Primitives`, `Results`, `update_params_and_resolve!`).
# To avoid re-defining docstrings and symbols when files are included multiple
# times in the same session, do NOT include the core files here. The top-level
# runner should include `ModelSetup.jl` and `ModelSolver.jl` once before
# including `ModelEstimation.jl`.

using Random, Statistics, Distributions
using Printf
using Term
"""
    update_primitives!(prim::Primitives; kwargs...) -> NamedTuple

Update primitive parameters in-place with minimal recomputation and return flags:
(:recompute_h_dist, :recreate_functions, :recompute_alpha).

- Recompute h distribution iff `a_h` or `b_h` changed.
- Recreate closures iff production/utility/matching parameters changed.
- α-policy should be recomputed iff α-affecting parameters changed.
"""
function update_primitives!(prim::Primitives; kwargs...)
    KW = Dict(kwargs)
    prim_fields = Set(fieldnames(Primitives))

    # Determine which primitive fields are being updated
    old_vals = Dict{Symbol,Any}()
    param_keys = Symbol[]
    for (k, v) in KW
        if k in prim_fields
            push!(param_keys, k)
            old_vals[k] = getfield(prim, k)
        end
    end

    # Flags for minimal recomputations
    h_dist_keys = Set([:a_h, :b_h])
    func_keys   = Set([:A₀, :A₁, :ψ₀, :ϕ, :ν, :c₀, :χ, :γ₀, :γ₁])
    alpha_keys  = union(func_keys, Set([:ψ_min, :ψ_max]))

    recompute_h_dist   = any(k -> (k in h_dist_keys) && (haskey(KW, k) && KW[k] != old_vals[k]), param_keys)
    recreate_functions = any(k -> k in func_keys, param_keys)
    recompute_alpha    = recreate_functions || any(k -> k in alpha_keys, param_keys)

    # Apply primitive updates
    for k in param_keys
        setfield!(prim, k, KW[k])
    end

    # Update h distribution if needed (keep same grid points)
    if recompute_h_dist
        h_values = prim.h_grid
        h_min, h_max = prim.h_min, prim.h_max
        h_scaled = (h_values .- h_min) ./ (h_max - h_min)
        beta_dist = Distributions.Beta(prim.a_h, prim.b_h)
        h_pdf_raw = pdf.(beta_dist, h_scaled)
        h_pdf = h_pdf_raw ./ sum(h_pdf_raw)
        h_cdf = cumsum(h_pdf)
        prim.h_pdf = h_pdf
        prim.h_cdf = h_cdf
    end

    # Rebuild functional forms if needed
    if recreate_functions
        A₀, A₁, ψ₀, ϕ, ν = prim.A₀, prim.A₁, prim.ψ₀, prim.ϕ, prim.ν
        c₀, χ = prim.c₀, prim.χ
        γ₀, γ₁ = prim.γ₀, prim.γ₁
        prim.production_fun = (h, ψ, α) -> (A₀ + A₁*h) * ((1 - α) + α * (ψ₀ * h^ϕ * ψ^ν))
        prim.utility_fun    = (w, α)    -> w - c₀ * (1 - α)^(χ + 1) / (χ + 1)
        prim.matching_fun   = (V, U)    -> γ₀ * U^γ₁ * V^(1 - γ₁)
    end

    return (recompute_h_dist=recompute_h_dist,
            recreate_functions=recreate_functions,
            recompute_alpha=recompute_alpha)
end

"""
    fresh_results(prim::Primitives, old_res::Results=nothing; alpha_changed::Union{Bool,Nothing}=nothing) -> Results

Create a new Results object. If `alpha_changed` is false and `old_res` is provided and
dimensions match, reuse `α_policy`, `ψ_bottom`, and `ψ_top` from `old_res` to avoid
unnecessary recomputation in practice (the constructor computes them anyway, so we overwrite).
"""
function fresh_results(prim::Primitives, old_res::Results=nothing; alpha_changed::Union{Bool,Nothing}=nothing)
    need_alpha = alpha_changed === nothing ? (old_res === nothing ? true : false) : alpha_changed
    new_res = Results(prim)
    if !need_alpha && old_res !== nothing
        same_dims = size(new_res.α_policy) == size(old_res.α_policy)
        if same_dims
            new_res.α_policy .= old_res.α_policy
            new_res.ψ_bottom = copy(old_res.ψ_bottom)
            new_res.ψ_top = copy(old_res.ψ_top)
        end
    end
    return new_res
end

"""
    update_params_and_resolve!(prim::Primitives, res::Results; kwargs...)

Convenience wrapper: update primitives, build a fresh Results (reusing α when allowed),
then solve. Returns `(prim, new_res)`.
"""
function update_params_and_resolve!(prim::Primitives, res::Results; kwargs...)
    KW = Dict(kwargs)
    flags = update_primitives!(prim; kwargs...)
    new_res = fresh_results(prim, res; alpha_changed=flags.recompute_alpha)
    tol        = 1e-7
    max_iter   = 10000
    verbose    = false
    λ_S        = 0.01
    λ_u        = 0.01
    solve_model(prim, new_res; tol=tol, max_iter=max_iter, verbose=verbose, λ_S=λ_S, λ_u=λ_u)
    return prim, new_res
end

"""
    compute_model_moments(prim::Primitives, res::Results) -> NamedTuple

Compute model-implied moments from the steady-state cross-section of employed workers.

Returns:
- mean_logwage: E[log w*] under the distribution n(h, ψ)
- var_logwage: Var[log w*] under the distribution n(h, ψ)
- mean_logwage_inperson: E[log w* | α* ≈ 0]
- mean_logwage_remote:   E[log w* | α* ≈ 1]
- diff_logwage_inperson_remote: difference of the above two means
- hybrid_share: total mass of employed workers with interior α* (0 < α* < 1)
- agg_productivity: E[Y(h, ψ, α*)] weighted by n(h, ψ)
- dlogw_dpsi_mean_RH: E[(1/w) ∂w*/∂ψ | α*>0]
- mean_logwage_RH_lowpsi: E[log w* | ψ in bottom quantile, α*>0]
- mean_logwage_RH_highpsi: E[log w* | ψ in top quantile, α*>0]
- diff_logwage_RH_high_lowpsi: Difference between the two above (High − Low)
- mean_alpha_highpsi: E[α* | ψ in top quantile]
- mean_alpha_lowpsi: E[α* | ψ in bottom quantile]
- diff_alpha_high_lowpsi: Difference in α* means (High − Low)
- var_logwage_highpsi: Var[log w* | ψ in top quantile]
- var_logwage_lowpsi: Var[log w* | ψ in bottom quantile]
- ratio_var_logwage_high_lowpsi: var_high / var_low (0.0 if var_low==0)
- market_tightness: θ (market tightness)
"""


"""
    compute_model_moments(prim::Primitives, res::Results; q_low_cut=0.25, q_high_cut=0.75) -> NamedTuple{(:mean_logwage, :var_logwage, :mean_logwage_inperson, :mean_logwage_remote, :diff_logwage_inperson_remote, :hybrid_share, :agg_productivity, :dlogw_dpsi_mean_RH, :mean_logwage_RH_lowpsi, :mean_logwage_RH_highpsi, :diff_logwage_RH_high_lowpsi, :mean_alpha_highpsi, :mean_alpha_lowpsi, :diff_alpha_high_lowpsi, :var_logwage_highpsi, :var_logwage_lowpsi, :ratio_var_logwage_high_lowpsi, :market_tightness), NTuple{18, Float64}}

Compute model-implied moments from the steady-state cross-section of employed workers.

## Economic Interpretation:
- **Log wage moments**: Capture wage dispersion and sorting patterns across worker-firm matches
- **Work arrangement moments**: Measure remote/hybrid adoption and its wage premium/penalty
- **Firm productivity grouping**: Tests model's predictions about high-ψ vs low-ψ firm behavior
- **Derivative moment**: Captures wage-productivity elasticity among flexible arrangements

## Returns:
Comprehensive set of theoretical moments for structural estimation, including unconditional
wage statistics, conditional moments by work arrangement and firm type, and market aggregates.
"""
function compute_model_moments(
    prim::Primitives, 
    res::Results;
    q_low_cut::Float64=0.25, 
    q_high_cut::Float64=0.75
)::NamedTuple{(:mean_logwage, :var_logwage, :mean_logwage_inperson, :mean_logwage_remote, 
              :diff_logwage_inperson_remote, :hybrid_share, :agg_productivity, :dlogw_dpsi_mean_RH, 
              :mean_logwage_RH_lowpsi, :mean_logwage_RH_highpsi, :diff_logwage_RH_high_lowpsi, 
              :mean_alpha_highpsi, :mean_alpha_lowpsi, :diff_alpha_high_lowpsi, 
              :var_logwage_highpsi, :var_logwage_lowpsi, :ratio_var_logwage_high_lowpsi, 
              :market_tightness), NTuple{18, Float64}}

    # Extract key objects
    n::Matrix{Float64} = res.n           # Employment distribution n(h,ψ) 
    w::Matrix{Float64} = res.w_policy    # Equilibrium wages w*(h,ψ)
    α::Matrix{Float64} = res.α_policy    # Optimal work arrangements α*(h,ψ)
    
    @unpack h_grid, ψ_grid, ψ_cdf, production_fun, A₁, ψ₀, ϕ, ν, χ, ξ, β, δ = prim

    total_emp::Float64 = sum(n)
    
    # Early return for degenerate case
    if !(total_emp > 0)
        return _zero_moments(res.θ)
    end

    # Pre-compute masks and tolerance
    αtol::Float64 = 1e-8
    valid::BitMatrix = (n .> 0.0) .& (w .> 0.0)
    
    # Determine ψ quantile boundaries for firm type grouping
    n_ψ::Int = length(ψ_grid)
    idx_low_end::Int = _find_quantile_index(ψ_cdf, q_low_cut, n_ψ)
    idx_high_start::Int = max(_find_quantile_index(ψ_cdf, q_high_cut, n_ψ), idx_low_end + 1)

    # Initialize all accumulators for single-pass computation
    # Unconditional wage moments
    sum_wts::Float64 = 0.0
    sum_logw::Float64 = 0.0
    sum_logw2::Float64 = 0.0
    
    # Work arrangement conditional moments  
    sum_n_inperson::Float64 = 0.0; sum_logw_inperson::Float64 = 0.0
    sum_n_remote::Float64 = 0.0; sum_logw_remote::Float64 = 0.0
    sum_n_interior::Float64 = 0.0  # Hybrid workers (0 < α* < 1)
    
    # Productivity moment
    sum_production::Float64 = 0.0
    
    # Wage-productivity elasticity among remote/hybrid workers
    # Economic interpretation: ∂log(w)/∂ψ measures how responsive wages are to firm 
    # productivity improvements, conditional on offering flexible work arrangements
    sum_dlogw_dpsi_RH::Float64 = 0.0
    sum_n_RH::Float64 = 0.0
    
    # ψ-group conditional moments (for first-stage estimation)
    # Low-ψ firms (bottom quantile)
    sum_n_low::Float64 = 0.0; sum_alpha_low::Float64 = 0.0
    sum_logw_low::Float64 = 0.0; sum_logw2_low::Float64 = 0.0
    sum_logw_RH_low::Float64 = 0.0; sum_n_RH_low::Float64 = 0.0
    
    # High-ψ firms (top quantile) 
    sum_n_high::Float64 = 0.0; sum_alpha_high::Float64 = 0.0
    sum_logw_high::Float64 = 0.0; sum_logw2_high::Float64 = 0.0
    sum_logw_RH_high::Float64 = 0.0; sum_n_RH_high::Float64 = 0.0

    # Single pass through all (h,ψ) combinations
    for (i_h, h) in enumerate(h_grid)
        for (i_ψ, ψ) in enumerate(ψ_grid)
            n_cell::Float64 = n[i_h, i_ψ]
            
            # Skip zero-employment cells
            n_cell > 0.0 || continue
            
            w_cell::Float64 = w[i_h, i_ψ]
            α_cell::Float64 = α[i_h, i_ψ]
            
            # Determine cell characteristics
            is_valid::Bool = valid[i_h, i_ψ]  # Positive wage
            is_inperson::Bool = α_cell <= αtol
            is_remote::Bool = α_cell >= (1.0 - αtol) 
            is_interior::Bool = (α_cell > αtol) && (α_cell < (1.0 - αtol))
            is_RH::Bool = α_cell > αtol  # Remote or hybrid
            is_low_psi::Bool = i_ψ <= idx_low_end
            is_high_psi::Bool = i_ψ >= idx_high_start

            # Aggregate productivity (all employed workers)
            sum_production += production_fun(h, ψ, α_cell) * n_cell
            
            # Skip wage-based moments if wage is non-positive
            if !is_valid
                # Still accumulate α moments and employment counts
                if is_low_psi
                    sum_n_low += n_cell
                    sum_alpha_low += α_cell * n_cell
                end
                if is_high_psi
                    sum_n_high += n_cell
                    sum_alpha_high += α_cell * n_cell
                end
                if is_interior
                    sum_n_interior += n_cell
                end
                continue
            end
            
            logw_cell::Float64 = log(w_cell)
            
            # Unconditional wage moments (all valid employed workers)
            sum_wts += n_cell
            sum_logw += logw_cell * n_cell
            sum_logw2 += (logw_cell^2) * n_cell
            
            # Work arrangement conditional moments
            if is_inperson
                sum_n_inperson += n_cell
                sum_logw_inperson += logw_cell * n_cell
            elseif is_remote
                sum_n_remote += n_cell  
                sum_logw_remote += logw_cell * n_cell
            end
            
            if is_interior
                sum_n_interior += n_cell
            end
            
            # Remote/hybrid wage-productivity elasticity
            # Economic model: ∂w*/∂ψ = A₁h(∂g/∂ψ)[ξα*/(1-β(1-δ)) - (1-α*)/χ]
            # This captures how wage responds to firm productivity for flexible arrangements
            if is_RH
                # Marginal productivity of firm type: ∂g/∂ψ where g(h,ψ) = ψ₀h^ϕψ^ν
                dg_dpsi::Float64 = ψ₀ * (h^ϕ) * (ν * ψ^(ν - 1))
                
                # Wage derivative incorporating Nash bargaining and arrangement costs
                # First term: remote productivity advantage (ξ = remote productivity factor)
                # Second term: arrangement cost savings (χ = cost elasticity parameter)
                dw_dpsi::Float64 = A₁ * h * dg_dpsi * (
                    (ξ * α_cell) / (1.0 - β * (1.0 - δ)) - (1.0 - α_cell) / χ
                )
                
                sum_dlogw_dpsi_RH += (dw_dpsi / w_cell) * n_cell
                sum_n_RH += n_cell
            end
            
            # ψ-group moments (for identification of firm heterogeneity effects)
            if is_low_psi
                sum_n_low += n_cell
                sum_alpha_low += α_cell * n_cell
                sum_logw_low += logw_cell * n_cell
                sum_logw2_low += (logw_cell^2) * n_cell
                
                if is_RH
                    sum_logw_RH_low += logw_cell * n_cell
                    sum_n_RH_low += n_cell
                end
            end
            
            if is_high_psi
                sum_n_high += n_cell
                sum_alpha_high += α_cell * n_cell
                sum_logw_high += logw_cell * n_cell
                sum_logw2_high += (logw_cell^2) * n_cell
                
                if is_RH
                    sum_logw_RH_high += logw_cell * n_cell
                    sum_n_RH_high += n_cell
                end
            end
        end
    end

    # Compute final moments with numerical safeguards
    mean_logwage::Float64, var_logwage::Float64 = _compute_mean_var(sum_logw, sum_logw2, sum_wts)
    
    mean_logwage_inperson::Float64 = sum_n_inperson > 0 ? sum_logw_inperson / sum_n_inperson : 0.0
    mean_logwage_remote::Float64 = sum_n_remote > 0 ? sum_logw_remote / sum_n_remote : 0.0
    diff_logwage_inperson_remote::Float64 = mean_logwage_inperson - mean_logwage_remote
    
    hybrid_share::Float64 = sum_n_interior / total_emp
    agg_productivity::Float64 = sum_production / total_emp
    dlogw_dpsi_mean_RH::Float64 = sum_n_RH > 0 ? sum_dlogw_dpsi_RH / sum_n_RH : 0.0
    
    # First-stage moments for firm heterogeneity identification
    mean_logwage_RH_lowpsi::Float64 = sum_n_RH_low > 0 ? sum_logw_RH_low / sum_n_RH_low : 0.0
    mean_logwage_RH_highpsi::Float64 = sum_n_RH_high > 0 ? sum_logw_RH_high / sum_n_RH_high : 0.0
    diff_logwage_RH_high_lowpsi::Float64 = mean_logwage_RH_highpsi - mean_logwage_RH_lowpsi
    
    mean_alpha_lowpsi::Float64 = sum_n_low > 0 ? sum_alpha_low / sum_n_low : 0.0
    mean_alpha_highpsi::Float64 = sum_n_high > 0 ? sum_alpha_high / sum_n_high : 0.0
    diff_alpha_high_lowpsi::Float64 = mean_alpha_highpsi - mean_alpha_lowpsi
    
    _, var_logwage_lowpsi::Float64 = _compute_mean_var(sum_logw_low, sum_logw2_low, sum_n_low)
    _, var_logwage_highpsi::Float64 = _compute_mean_var(sum_logw_high, sum_logw2_high, sum_n_high)
    ratio_var_logwage_high_lowpsi::Float64 = var_logwage_lowpsi > 0 ? var_logwage_highpsi / var_logwage_lowpsi : 0.0

    return (
        mean_logwage = mean_logwage,
        var_logwage = var_logwage,
        mean_logwage_inperson = mean_logwage_inperson,
        mean_logwage_remote = mean_logwage_remote,
        diff_logwage_inperson_remote = diff_logwage_inperson_remote,
        hybrid_share = hybrid_share,
        agg_productivity = agg_productivity,
        dlogw_dpsi_mean_RH = dlogw_dpsi_mean_RH,
        mean_logwage_RH_lowpsi = mean_logwage_RH_lowpsi,
        mean_logwage_RH_highpsi = mean_logwage_RH_highpsi,
        diff_logwage_RH_high_lowpsi = diff_logwage_RH_high_lowpsi,
        mean_alpha_highpsi = mean_alpha_highpsi,
        mean_alpha_lowpsi = mean_alpha_lowpsi,
        diff_alpha_high_lowpsi = diff_alpha_high_lowpsi,
        var_logwage_highpsi = var_logwage_highpsi,
        var_logwage_lowpsi = var_logwage_lowpsi,
        ratio_var_logwage_high_lowpsi = ratio_var_logwage_high_lowpsi,
        market_tightness = res.θ,
    )
end

# Helper functions for cleaner code
function _zero_moments(θ::Float64)::NamedTuple
    return (
        mean_logwage = 0.0, var_logwage = 0.0, mean_logwage_inperson = 0.0,
        mean_logwage_remote = 0.0, diff_logwage_inperson_remote = 0.0, hybrid_share = 0.0,
        agg_productivity = 0.0, dlogw_dpsi_mean_RH = 0.0, mean_logwage_RH_lowpsi = 0.0,
        mean_logwage_RH_highpsi = 0.0, diff_logwage_RH_high_lowpsi = 0.0, 
        mean_alpha_highpsi = 0.0, mean_alpha_lowpsi = 0.0, diff_alpha_high_lowpsi = 0.0,
        var_logwage_highpsi = 0.0, var_logwage_lowpsi = 0.0, 
        ratio_var_logwage_high_lowpsi = 0.0, market_tightness = θ,
    )
end

function _find_quantile_index(cdf::Vector{Float64}, quantile::Float64, n::Int)::Int
    idx = findfirst(>=(quantile), cdf)
    return idx === nothing ? n : idx
end

function _compute_mean_var(sum_x::Float64, sum_x2::Float64, sum_wts::Float64)::Tuple{Float64, Float64}
    if sum_wts > 0
        μ = sum_x / sum_wts
        σ2 = max(0.0, (sum_x2 / sum_wts) - μ^2)
        return μ, σ2
    else
        return 0.0, 0.0
    end
end

"""
    estimation_objective(prim, res; data_moments::NamedTuple, kwargs...)

Update parameters (via kwargs), resolve, compute model moments, and return a simple
least-squares objective vs provided `data_moments`.
"""
function estimation_objective(prim::Primitives, res::Results; data_moments::NamedTuple, kwargs...)
    # If no overrides, use the current solution `res` directly to avoid drift.
    if length(kwargs) == 0
        model_moments = compute_model_moments(prim, res)
        keys = fieldnames(typeof(data_moments))
        return sum((getfield(model_moments, k) - getfield(data_moments, k))^2 for k in keys)
    end

    # With overrides: temporarily update primitives, resolve, compute, then restore.
    KW = Dict(kwargs)
    prim_fields = Set(fieldnames(Primitives))
    saved = Dict{Symbol,Any}()
    for (k, _) in KW
        if k in prim_fields
            saved[k] = getfield(prim, k)
        end
    end

    try
        _, res_new = update_params_and_resolve!(prim, res; kwargs...)
        model_moments = compute_model_moments(prim, res_new)
    finally
        # Restore primitive fields and rebuild closures if needed
        if !isempty(saved)
            update_primitives!(prim; saved...)
        end
    end

    keys = fieldnames(typeof(data_moments))
    return sum((getfield(model_moments, k) - getfield(data_moments, k))^2 for k in keys)
end

using YAML

"""
    save_moments_to_yaml(moments::NamedTuple, filename::String)

Save model moments to a YAML file for later use as target moments in estimation.
"""
function save_moments_to_yaml(moments::NamedTuple, filename::String)
    moments_dict = Dict(string(k) => v for (k, v) in pairs(moments))
    YAML.write_file(filename, moments_dict)
    println("Saved moments to: $filename")
end

"""
    load_moments_from_yaml(filename::String) -> NamedTuple

Load target moments from a YAML file and convert back to NamedTuple.
"""
function load_moments_from_yaml(filename::String)
    moments_dict = YAML.load_file(filename)
    moment_keys = Symbol.(collect(Base.keys(moments_dict)))
    moment_values = collect(Base.values(moments_dict))
    return NamedTuple{Tuple(moment_keys)}(moment_values)
end

"""
    perturb_parameters(prim::Primitives; scale::Float64=0.1) -> Dict{Symbol, Float64}

Create perturbed parameter values for testing parameter recovery.
Returns a dictionary of parameter perturbations.
"""
function perturb_parameters(prim::Primitives; scale::Float64=0.1)
    # Key parameters to perturb for testing
    params = Dict{Symbol, Float64}()
    
    # Production parameters
    params[:A₁] = prim.A₁ * (1.0 + scale * randn())
    params[:ψ₀] = prim.ψ₀ * (1.0 + scale * randn())
    params[:ν] = max(0.1, prim.ν * (1.0 + scale * randn()))
    
    # Human capital distribution
    params[:a_h] = max(0.5, prim.a_h * (1.0 + scale * randn()))
    params[:b_h] = max(0.5, prim.b_h * (1.0 + scale * randn()))
    
    # Cost parameters
    params[:χ] = max(0.1, prim.χ * (1.0 + scale * randn()))
    params[:c₀] = max(0.01, prim.c₀ * (1.0 + scale * randn()))

    # Additional parameters (if present on Primitives)
    if hasfield(typeof(prim), :ϕ)
        params[:ϕ] = max(1e-6, prim.ϕ * (1.0 + scale * randn()))
    end
    if hasfield(typeof(prim), :κ₀)
        params[:κ₀] = max(1e-6, prim.κ₀ * (1.0 + scale * randn()))
    end
    
    return params
end

"""
    simple_estimation(prim::Primitives, res::Results, target_moments::NamedTuple;
                      max_iter::Int=50, step_size::Float64=0.01, tol::Float64=1e-4,
                      seed::Union{Int,Nothing}=nothing, step_decay::Float64=0.95,
                      decay_every::Int=50, min_step_size::Float64=1e-4,
                      verbose::Bool=true) -> Dict

Simple gradient-free estimation using random search with step-size annealing.
- step_size: std dev of relative multiplicative perturbations per proposal.
- step_decay: multiplicative decay factor applied every `decay_every` iters.
- min_step_size: lower bound for the annealed step size.
- seed: set RNG seed for reproducibility when provided.
"""
function simple_estimation(prim::Primitives, res::Results, target_moments::NamedTuple;
                           max_iter::Int=50, step_size::Float64=0.01, tol::Float64=1e-4,
                           seed::Union{Int,Nothing}=nothing, step_decay::Float64=0.95,
                           decay_every::Int=50, min_step_size::Float64=1e-4,
                           verbose::Bool=true)

    seed !== nothing && Random.seed!(seed)

    # Initialize with current parameters
    best_params = Dict{Symbol, Float64}(
        :A₁ => prim.A₁, :ψ₀ => prim.ψ₀, :ν => prim.ν,
        :a_h => prim.a_h, :b_h => prim.b_h,
        :χ => prim.χ, :c₀ => prim.c₀
    )

    best_obj = estimation_objective(prim, res; data_moments=target_moments, best_params...)

    if verbose
        hdr = @sprintf("Random search (annealed) | init_step=%.5f | decay=%.3f/%d | min_step=%.5f",
                       step_size, step_decay, decay_every, min_step_size)
        printstyled("\n" * "="^80 * "\n"; color=:cyan, bold=true)
        printstyled(hdr * "\n"; color=:cyan, bold=true)
        seed !== nothing && printstyled(@sprintf("Seed = %d\n", seed); color=:light_black, bold=false)
        printstyled(@sprintf("Initial objective: %.6e\n", best_obj); color=:magenta, bold=true)
        printstyled("-"^80 * "\n"; color=:cyan)
    end

    cur_step = step_size

    for iter in 1:max_iter
        # Anneal step size
        if decay_every > 0 && iter % decay_every == 0
            new_step = max(min_step_size, cur_step * step_decay)
            if verbose && new_step != cur_step
                printstyled(@sprintf("Anneal @ iter %d: step_size %.6f → %.6f\n",
                                     iter, cur_step, new_step); color=:yellow, bold=true)
            end
            cur_step = new_step
        end

        # Random perturbation around current best
        candidate_params = copy(best_params)
        for (k, v) in candidate_params
            candidate_params[k] = max(0.01, v * (1.0 + cur_step * randn()))
            # Keep ν above a softer lower bound (just in case)
            if k == :ν
                candidate_params[k] = max(0.05, candidate_params[k])
            end
        end

        # Evaluate objective
        try
            obj = estimation_objective(prim, res; data_moments=target_moments, candidate_params...)

            if obj < best_obj
                best_params = candidate_params
                best_obj = obj
                if verbose
                    printstyled(@sprintf("Iter %6d | New best obj = %.6e | step=%.6f\n",
                                         iter, best_obj, cur_step); color=:green, bold=true)
                end
            else
                if verbose && (iter % 50 == 0)
                    printstyled(@sprintf("Iter %6d | obj = %.6e | step=%.6f\n",
                                         iter, obj, cur_step); color=:blue, bold=false)
                end
            end

            if best_obj < tol
                if verbose
                    printstyled(@sprintf("Converged at iter %d with obj = %.6e\n",
                                         iter, best_obj); color=:green, bold=true)
                    printstyled("="^80 * "\n"; color=:cyan, bold=true)
                end
                break
            end
        catch e
            if verbose
                printstyled(@sprintf("Iter %6d | proposal failed (%s)\n", iter, typeof(e)),
                            color=:red, bold=true)
            end
            continue
        end
    end

    return Dict(:params => best_params, :objective => best_obj)
end

"""
    test_parameter_recovery(prim::Primitives, res::Results; verbose::Bool=true) -> Bool

Full test: solve model, save moments, perturb parameters, then try to recover them.
Returns true if recovery was successful (objective < tol), false otherwise.
"""
function test_parameter_recovery(prim::Primitives, res::Results; verbose::Bool=true, tol::Float64=1e-5)
    # 1. Save current moments as "true" moments
    true_moments = compute_model_moments(prim, res)
    temp_file = "temp_target_moments.yaml"
    save_moments_to_yaml(true_moments, temp_file)
    
    # 2. Perturb parameters
    perturbed_params = perturb_parameters(prim; scale=0.15)
    if verbose
        println("Perturbed parameters:")
        for (k, v) in perturbed_params
            println("  $k: $(getfield(prim, k)) → $v")
        end
    end
    
    # 3. Apply perturbations
    for (k, v) in perturbed_params
        setfield!(prim, k, v)
    end
    
    # 4. Try to recover
    target_moments = load_moments_from_yaml(temp_file)
    result = simple_estimation(prim, res, target_moments; max_iter=1000, step_size=0.001)
    
    # 5. Clean up
    rm(temp_file, force=true)
    
    success = result[:objective] < tol
    if verbose
        println("Recovery $(success ? "PASSED" : "FAILED"): Final objective = $(round(result[:objective], digits=6))")
    end
    
    return success
end

