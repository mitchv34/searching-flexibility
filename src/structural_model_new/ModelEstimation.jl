include("ModelSetup.jl")
include("ModelSolver.jl")

using Random, Statistics, Distributions
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
    func_keys   = Set([:A₀, :A₁, :ψ₀, :ϕ, :ν, :c₁, :χ, :γ₀, :γ₁])
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
        c₁, χ = prim.c₁, prim.χ
        γ₀, γ₁ = prim.γ₀, prim.γ₁
        prim.production_fun = (h, ψ, α) -> (A₀ + A₁*h) * ((1 - α) + α * (ψ₀ * h^ϕ * ψ^ν))
        prim.utility_fun    = (w, α)    -> w - c₁ * (1 - α)^(χ + 1) / (χ + 1)
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
    tol        = get(KW, :tol, 1e-7)
    max_iter   = get(KW, :max_iter, 5000)
    verbose    = get(KW, :verbose, true)
    print_freq = get(KW, :print_freq, 50)
    λ_S        = get(KW, :λ_S, 0.1)
    λ_u        = get(KW, :λ_u, 0.1)
    solve_model(prim, new_res; tol=tol, max_iter=max_iter, verbose=verbose, print_freq=print_freq, λ_S=λ_S, λ_u=λ_u)
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
function compute_model_moments(
                                prim::Primitives, res::Results;
                                # Quantile cutoffs (defaults: bottom/top quartiles)
                                q_low_cut=0.25, q_high_cut=0.75
                                )
    n = res.n
    w = res.w_policy
    α = res.α_policy
    @unpack h_grid, ψ_grid, ψ_cdf, production_fun, A₁, ψ₀, ϕ, ν, χ, ξ, β, δ = prim

    total_emp = sum(n)
    if !(total_emp > 0)
        return (
            mean_logwage = 0.0,
            var_logwage = 0.0,
            mean_logwage_inperson = 0.0,
            mean_logwage_remote = 0.0,
            diff_logwage_inperson_remote = 0.0,
            hybrid_share = 0.0,
            agg_productivity = 0.0,
            dlogw_dpsi_mean_RH = 0.0,
            mean_logwage_RH_lowpsi = 0.0,
            mean_logwage_RH_highpsi = 0.0,
            diff_logwage_RH_high_lowpsi = 0.0,
            mean_alpha_highpsi = 0.0,
            mean_alpha_lowpsi = 0.0,
            diff_alpha_high_lowpsi = 0.0,
            var_logwage_highpsi = 0.0,
            var_logwage_lowpsi = 0.0,
            ratio_var_logwage_high_lowpsi = 0.0,
            market_tightness = res.θ,
        )
    end

    # Mask to avoid log of non-positive wages and zero-weight cells
    valid = (n .> 0.0) .& (w .> 0.0)

    # Unconditional mean/variance of log wages across employed
    if any(valid)
        logw_all = log.(w[valid])
        wts_all  = n[valid]
        wts_all_norm = wts_all ./ sum(wts_all)
        μ = sum(logw_all .* wts_all_norm)
        σ2 = sum(((logw_all .- μ).^2) .* wts_all_norm)
    else
        μ = 0.0; σ2 = 0.0
    end

    # Conditional by α* groups
    αtol = 1e-8
    mask_inperson = valid .& (α .<= αtol)
    mask_remote   = valid .& (α .>= 1.0 - αtol)
    mask_interior = (α .> αtol) .& (α .< 1.0 - αtol)

    function cond_mean(mask)
        if any(mask)
            logw = log.(w[mask])
            wts  = n[mask]
            s = sum(wts)
            if s > 0
                wtsn = wts ./ s
                return sum(logw .* wtsn)
            end
        end
        return 0.0
    end

    μ_inperson = cond_mean(mask_inperson)
    μ_remote   = cond_mean(mask_remote)
    diff_ir    = μ_inperson - μ_remote

    # Hybrid share as fraction of total employment
    hybrid_share = sum(n[mask_interior]) / total_emp

    # Aggregate productivity E[Y]
    EY = 0.0
    if total_emp > 0
        acc = 0.0
        for (i_h, h) in enumerate(h_grid)
            for (i_ψ, ψ) in enumerate(ψ_grid)
                acc += production_fun(h, ψ, α[i_h, i_ψ]) * n[i_h, i_ψ]
            end
        end
        EY = acc / total_emp
    end

    # Derivative-based moment over S_RH (α*>0)
    mask_RH = valid .& (α .> αtol)
    dlogw_dpsi_mean_RH = 0.0
    denom_RH = sum(n[mask_RH])
    if denom_RH > 0
        # Compute ∂g/∂ψ and ∂w/∂ψ using provided formula
        acc = 0.0
        for (i_h, h) in enumerate(h_grid)
            for (i_ψ, ψ) in enumerate(ψ_grid)
                if mask_RH[i_h, i_ψ]
                    g_psi = ψ₀ * (h^ϕ) * (ν * ψ^(ν - 1))
                    α_star = α[i_h, i_ψ]
                    dw_dpsi = A₁ * h * g_psi * ( (ξ * α_star) / (1.0 - β * (1.0 - δ)) - (1.0 - α_star) / χ )
                    acc += (dw_dpsi / w[i_h, i_ψ]) * n[i_h, i_ψ]
                end
            end
        end
        dlogw_dpsi_mean_RH = acc / denom_RH
    end

    # First-stage (theoretical) high-vs-low ψ moment among remote/hybrid (α*>0)
    # Find index boundaries on the ψ grid using the ψ CDF
    n_ψ = length(ψ_grid)
    # Low group: indices with CDF ≤ q_low_cut
    idx_low_end = begin
        idx = findfirst(>=(q_low_cut), ψ_cdf)
        idx === nothing ? n_ψ : idx
    end
    # High group: indices with CDF ≥ q_high_cut
    idx_high_start = begin
        idx = findfirst(>=(q_high_cut), ψ_cdf)
        idx === nothing ? n_ψ : idx
    end
    # Ensure disjoint groups (optional): shift high start if overlapping
    if idx_high_start <= idx_low_end
        idx_high_start = min(idx_low_end + 1, n_ψ)
    end

    # Accumulate conditional expectations over n(h,ψ) restricted to α*>0 and ψ in group
    num_low = 0.0; den_low = 0.0
    num_high = 0.0; den_high = 0.0
    if total_emp > 0
        for (i_h, _) in enumerate(h_grid)
            # Low ψ slice
            for i_ψ in 1:idx_low_end
                if valid[i_h, i_ψ] && α[i_h, i_ψ] > αtol
                    num_low  += log(w[i_h, i_ψ]) * n[i_h, i_ψ]
                    den_low  += n[i_h, i_ψ]
                end
            end
            # High ψ slice
            for i_ψ in idx_high_start:n_ψ
                if valid[i_h, i_ψ] && α[i_h, i_ψ] > αtol
                    num_high += log(w[i_h, i_ψ]) * n[i_h, i_ψ]
                    den_high += n[i_h, i_ψ]
                end
            end
        end
    end
    mean_logwage_RH_lowpsi  = den_low  > 0 ? (num_low  / den_low)  : 0.0
    mean_logwage_RH_highpsi = den_high > 0 ? (num_high / den_high) : 0.0
    diff_RH_high_lowpsi = mean_logwage_RH_highpsi - mean_logwage_RH_lowpsi

    # Grouped α* means by ψ (no α*>0 restriction), weighted by n(h,ψ)
    num_a_low = 0.0; den_a_low = 0.0
    num_a_high = 0.0; den_a_high = 0.0
    for (i_h, _) in enumerate(h_grid)
        for i_ψ in 1:idx_low_end
            if n[i_h, i_ψ] > 0.0
                num_a_low += α[i_h, i_ψ] * n[i_h, i_ψ]
                den_a_low += n[i_h, i_ψ]
            end
        end
        for i_ψ in idx_high_start:n_ψ
            if n[i_h, i_ψ] > 0.0
                num_a_high += α[i_h, i_ψ] * n[i_h, i_ψ]
                den_a_high += n[i_h, i_ψ]
            end
        end
    end
    mean_alpha_lowpsi  = den_a_low  > 0 ? (num_a_low  / den_a_low)  : 0.0
    mean_alpha_highpsi = den_a_high > 0 ? (num_a_high / den_a_high) : 0.0
    diff_alpha_high_lowpsi = mean_alpha_highpsi - mean_alpha_lowpsi

    # Conditional variances of log wages by ψ group (no α restriction), weighted by n
    sum_n_low = 0.0; sum_logw_low = 0.0; sum_logw2_low = 0.0
    sum_n_high = 0.0; sum_logw_high = 0.0; sum_logw2_high = 0.0
    for (i_h, _) in enumerate(h_grid)
        for i_ψ in 1:idx_low_end
            if valid[i_h, i_ψ]
                lw = log(w[i_h, i_ψ])
                wt = n[i_h, i_ψ]
                sum_n_low += wt
                sum_logw_low += lw * wt
                sum_logw2_low += (lw^2) * wt
            end
        end
        for i_ψ in idx_high_start:n_ψ
            if valid[i_h, i_ψ]
                lw = log(w[i_h, i_ψ])
                wt = n[i_h, i_ψ]
                sum_n_high += wt
                sum_logw_high += lw * wt
                sum_logw2_high += (lw^2) * wt
            end
        end
    end
    var_logwage_lowpsi = 0.0
    var_logwage_highpsi = 0.0
    if sum_n_low > 0
        μL = sum_logw_low / sum_n_low
        var_logwage_lowpsi = max(0.0, (sum_logw2_low / sum_n_low) - μL^2)
    end
    if sum_n_high > 0
        μH = sum_logw_high / sum_n_high
        var_logwage_highpsi = max(0.0, (sum_logw2_high / sum_n_high) - μH^2)
    end
    ratio_var_logwage_high_lowpsi = (var_logwage_lowpsi > 0) ? (var_logwage_highpsi / var_logwage_lowpsi) : 0.0

    return (
        mean_logwage = μ,
        var_logwage = σ2,
        mean_logwage_inperson = μ_inperson,
        mean_logwage_remote = μ_remote,
        diff_logwage_inperson_remote = diff_ir,
        hybrid_share = hybrid_share,
        agg_productivity = EY,
        dlogw_dpsi_mean_RH = dlogw_dpsi_mean_RH,
        mean_logwage_RH_lowpsi = mean_logwage_RH_lowpsi,
        mean_logwage_RH_highpsi = mean_logwage_RH_highpsi,
        diff_logwage_RH_high_lowpsi = diff_RH_high_lowpsi,
        mean_alpha_highpsi = mean_alpha_highpsi,
        mean_alpha_lowpsi = mean_alpha_lowpsi,
        diff_alpha_high_lowpsi = diff_alpha_high_lowpsi,
        var_logwage_highpsi = var_logwage_highpsi,
        var_logwage_lowpsi = var_logwage_lowpsi,
        ratio_var_logwage_high_lowpsi = ratio_var_logwage_high_lowpsi,
    market_tightness = res.θ,
    )
end
"""
    estimation_objective(prim, res; data_moments::NamedTuple, kwargs...)

Update parameters (via kwargs), resolve, compute model moments, and return a simple
least-squares objective vs provided `data_moments`.
"""
function estimation_objective(prim::Primitives, res::Results; data_moments::NamedTuple, kwargs...)
    _, res_new = update_params_and_resolve!(prim, res; kwargs...)
    model_moments = compute_model_moments(prim, res_new)
    keys = fieldnames(typeof(data_moments))
    return sum((getfield(model_moments, k) - getfield(data_moments, k))^2 for k in keys)
end

