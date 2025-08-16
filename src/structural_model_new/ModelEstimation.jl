# ModelEstimation.jl depends on types and functions defined in ModelSetup.jl and
# ModelSolver.jl (e.g. `Primitives`, `Results`, `update_params_and_resolve`).
# To avoid re-defining docstrings and symbols when files are included multiple
# times in the same session, do NOT include the core files here. The top-level
# runner should include `ModelSetup.jl` and `ModelSolver.jl` once before
# including `ModelEstimation.jl`.

using Random, Statistics, Distributions, LinearAlgebra , ForwardDiff
using Printf, Term, YAML

"""
update_primitives_results(prim::Primitives, res::Results; params_to_update::Dict=Dict(), kwargs...)

Update a Primitives object by applying a set of parameter overrides and
rebuilding any dependent primitive quantities, then construct a fresh
Results object from the updated primitives.

This function does not mutate the provided `prim` or `res`; it returns a
new validated `Primitives` instance and a new `Results` instance
constructed from it.

Arguments
- prim::Primitives
    The current primitives object whose fields provide default values.
- res::Results
    The current results object (kept for API symmetry). It is not read
    deeply by this routine other than to match the original signature.
- params_to_update::Dict (optional, default = Dict())
    A dictionary mapping Symbol keys to new values that should override
    the corresponding fields in `prim`. Keys are expected to match the
    field names of `Primitives`. If a key is not present in this dict,
    the value from `prim` is retained.
- kwargs...
    Additional keyword arguments are accepted but not used by the
    current implementation.

Behavior
- Builds an updated parameter map `pairs` by taking all fields of `prim`
    and replacing any fields present in `params_to_update`.
- Detects whether the h-distribution needs to be recomputed. By
    default, this happens when any of the h-distribution parameters
    (e.g. :a_h, :b_h) included in `params_to_update` change relative to
    the corresponding value in `prim`.
- If the h-distribution is recomputed:
    - The function rescales `h_grid` to [0,1] using `h_min` and `h_max`.
    - It evaluates a Beta distribution with parameters (a_h, b_h) on the
        scaled grid to produce an unnormalized PDF, normalizes it to sum to
        1, and computes the cumulative distribution (CDF) by cumulative sum
        and renormalization.
    - The updated `:h_pdf` and `:h_cdf` are placed into the parameter map.
- Constructs a new, validated `Primitives` object (via `validated_Primitives`)
    from the updated parameter map. Note: arrays such as ψ_grid/ψ_pdf/ψ_cdf
    are converted to Vector{Float64} in the new object.
- Constructs a new `Results` instance from the validated primitives:
    new_res = Results(new_prim).
- Returns `(new_prim, new_res)`.

Return
- Tuple{Primitives, Results}
    A validated, updated primitives instance and the Results object derived
    from it.

Notes and caveats
- The function currently always constructs a new `Results` object and may
    precompute policies (e.g., α_policy) even when not required; there is a
- Keys in `params_to_update` must match the field names of `Primitives`.
    Be careful about ASCII vs. Unicode symbol differences (for example,
    `:a_h` vs `:aₕ`) — mismatched symbol names will not update the intended
    field.
- Comparison of "old" vs "new" values uses !=; if parameters are arrays
    or other complex objects, ensure that the intended notion of change is
    captured by this comparison.
- Validation performed by `validated_Primitives` can raise errors if the
    updated parameters are invalid.

Example
    # Replace selected primitives and get updated objects
    new_prim, new_res = update_primitives_results(prim, res, Dict(:a_h => 2.0, :b_h => 5.0))
"""
function update_primitives_results(
                            prim::Primitives,
                            res::Results,
                            params_to_update::Dict=Dict()
                            ; kwargs...)
    prim_fields = Set(fieldnames(Primitives))
    pairs = Dict([ k => (haskey(params_to_update, k) ? params_to_update[k] : getfield(prim, k)) for k in prim_fields ])
    
    h_dist_keys = Set([:a_h, :b_h])
    alpha_keys  = Set([:A₀, :A₁, :ψ₀, :ϕ, :ν, :c₀, :χ, :γ₀, :γ₁])

    # Flags for alpha recomputation
    old_values = Dict(k => getfield(prim, k) for k in keys(params_to_update))
    recompute_h_dist = any(k -> (k in h_dist_keys) && (pairs[k] != old_values[k]), keys(params_to_update))
    recompute_alpha  = any(k -> (k in alpha_keys)  && (pairs[k] != old_values[k]), keys(params_to_update))
    
    # Update h distribution if needed (keep same grid points)
    if true #recompute_h_dist
        #TODO figure out why it allways evaluates to false
        h_values = pairs[:h_grid]
        h_min, h_max = pairs[:h_min], pairs[:h_max]
        h_scaled = (h_values .- h_min) ./ (h_max - h_min)
        beta_dist = Distributions.Beta(pairs[:aₕ], pairs[:bₕ])
        h_pdf_raw = pdf.(beta_dist, h_scaled)
        h_pdf = h_pdf_raw ./ sum(h_pdf_raw)
        pairs[:h_pdf] = h_pdf
        h_cdf = cumsum(h_pdf)
        h_cdf /= h_cdf[end]
        pairs[:h_cdf] = h_cdf
    end

    # Create new primitives 
    new_prim = validated_Primitives(
        A₀=pairs[:A₀], A₁=pairs[:A₁], ψ₀=pairs[:ψ₀], ϕ=pairs[:ϕ], ν=pairs[:ν], c₀=pairs[:c₀], χ=pairs[:χ], γ₀=pairs[:γ₀], γ₁=pairs[:γ₁],
        κ₀=pairs[:κ₀], κ₁=pairs[:κ₁], β=pairs[:β], δ=pairs[:δ], b=pairs[:b], ξ=pairs[:ξ],
        n_ψ=pairs[:n_ψ], ψ_min=pairs[:ψ_min], ψ_max=pairs[:ψ_max], ψ_grid=Vector{Float64}(pairs[:ψ_grid]),
        ψ_pdf=Vector{Float64}(pairs[:ψ_pdf]), ψ_cdf=Vector{Float64}(pairs[:ψ_cdf]),
        aₕ=pairs[:aₕ], bₕ=pairs[:bₕ], n_h=pairs[:n_h], h_min=pairs[:h_min], h_max=pairs[:h_max],
        h_grid=pairs[:h_grid], h_pdf=pairs[:h_pdf], h_cdf=pairs[:h_cdf]
    )

    # Create new results
    #TODO: Skip pre-computation of α_policy if not needed
    new_res = Results(new_prim)
    return new_prim, new_res
end

"""
    update_params_and_resolve(prim::Primitives, res::Results; kwargs...)

Convenience wrapper: update primitives, build a fresh Results (reusing α when allowed),
then solve. Returns `(prim, new_res)`.
"""
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

"""
_zero_moments(θ::T) where {T<:Real}

Return a NamedTuple of zero-initialized moment statistics, with market_tightness set to θ.
"""
function _zero_moments(θ::T; keys::Union{Nothing, Vector{Symbol}}=nothing) where {T<:Real}
    # default full list (keeps previous names available)
    default_keys = [
        :mean_logwage, :var_logwage, :mean_logwage_inperson, :mean_logwage_remote,
        :diff_logwage_inperson_remote, :hybrid_share, :agg_productivity, :dlogw_dpsi_mean_RH,
        :mean_logwage_RH_lowpsi, :mean_logwage_RH_highpsi, :diff_logwage_RH_high_lowpsi,
        :mean_alpha_highpsi, :mean_alpha_lowpsi, :diff_alpha_high_lowpsi,
        :var_logwage_highpsi, :var_logwage_lowpsi, :ratio_var_logwage_high_lowpsi,
        :market_tightness
    ]
    use_keys = isnothing(keys) ? default_keys : keys
    out = Dict{Symbol, T}()
    for k in use_keys
        out[k] = k == :market_tightness ? θ : zero(T)
    end
    return out
end

"""
    compute_mean_var(sum_x, sum_x2, sum_wts)

Compute the weighted mean μ and non-negative variance σ² from the sums:
μ = sum_x / sum_wts, σ² = max(0.0, (sum_x2 / sum_wts) - μ^2).

Arguments
- sum_x: sum of weighted observations
- sum_x2: sum of squared weighted observations
- sum_wts: sum of weights

Returns
- (μ, σ²) as a tuple; returns (0.0, 0.0) if sum_wts <= 0.
"""
function _compute_mean_var(sum_x::T, sum_x2::T, sum_wts::T)::Tuple{T, T} where {T<:Real}
    if ForwardDiff.value(sum_wts) > 0
        μ = sum_x / sum_wts
        σ2 = max(0.0, (sum_x2 / sum_wts) - μ^2)
        return μ, σ2
    else
        return 0.0, 0.0
    end
end

"""Return the index of the first entry in `cdf` whose ForwardDiff.value >= ForwardDiff.value(quantile); if none is found, return `n`."""
function _find_quantile_index(cdf::Vector{T}, quantile::T, n::Int)::Int where {T<:Real}
    idx = findfirst(>=(ForwardDiff.value(quantile)), ForwardDiff.value.(cdf))
    return idx === nothing ? n : idx
end

"""
    compute_model_moments(prim::Primitives{T}, res::Results{T}; q_low_cut::T=0.25, q_high_cut::T=0.75) where {T<:Real}

Compute a set of economy-level and conditional moments from an equilibrium employment
distribution and associated policy functions. This function performs a single pass
over the discrete grid of worker skill `h_grid` and remote efficiency `ψ_grid`
contained in `prim`, weighting cell-level quantities by employment `n(h,ψ)` from
`res`. The returned moments are useful for model calibration and first-stage
identification of firm heterogeneity.

Arguments
- `prim::Primitives{T}`: model primitives and grids. Expected fields (used by the
    implementation): `h_grid`, `ψ_grid`, `ψ_cdf`, `production_fun`, `A₁`, `ψ₀`,
    `ϕ`, `ν`, `χ`, `ξ`, `β`, `δ`. `production_fun(h,ψ,α)` should return total output
    produced by a worker of skill `h` at firm-type `ψ` when the firm uses arrangement
    `α`.
- `res::Results{T}`: equilibrium objects. Expected fields: `n` (employment matrix
    sized `(length(h_grid), length(ψ_grid))`), `w_policy` (equilibrium wages matrix),
    `α_policy` (optimal arrangement matrix), and `θ` (market tightness / matching
    object returned unmodified in the output).
- `q_low_cut`, `q_high_cut` (optional): quantile cutoffs in (0,1) used to define
    "low-ψ" and "high-ψ" firm groups for first-stage moments (defaults 0.25 and 0.75).
    Indices are found using `_find_quantile_index(ψ_cdf, q, n_ψ)`.

Behavior and numerical details
- The function is AD-friendly: it often unwraps Dual numbers via `ForwardDiff.value`
    when classifying arrangement types using a small tolerance `αtol = 1e-8`.
- Cells with zero employment are skipped entirely.
- Cells with non-positive wages are excluded from wage-based moments but still
    contribute to arrangement (`α`) moments and employment counts for ψ-group
    aggregates.
- Arrangement classification:
    - "in-person" if α ≈ 0 (<= αtol)
    - "remote" if α ≈ 1 (>= 1-αtol)
    - "interior" (hybrid) if α in (αtol, 1-αtol)
    - "RH" denotes remote or hybrid (α > αtol).
- The routine computes an approximate wage response to firm productivity for
    remote/hybrid cells using the analytic derivative implied by the model:
    dw/dψ = A₁ * h * (∂g/∂ψ) * [ ξ α / (1 - β(1-δ)) - (1-α)/χ ],
    and accumulates ∂log w / ∂ψ = (dw/dψ)/w for RH cells.
- Final means/variances are computed by `_compute_mean_var(sum_logw, sum_logw2, sum_wts)`.
- Degenerate case: if total employment is zero the function returns
    `_zero_moments(res.θ)`.

Returned named tuple (employment-weighted where appropriate)
- `mean_logwage` : overall mean of log wages (over employed workers with positive wage)
- `var_logwage`  : variance of log wages
- `mean_logwage_inperson` : mean log wage for in-person-only jobs
- `mean_logwage_remote`   : mean log wage for remote-only jobs
- `diff_logwage_inperson_remote` : difference (inperson − remote)
- `hybrid_share` : share of employment in interior/hybrid arrangements (α strictly between 0 and 1)
- `agg_productivity` : aggregate productivity per worker (employment-weighted average of `production_fun`)
- `dlogw_dpsi_mean_RH` : mean ∂log w / ∂ψ among remote or hybrid workers
- `mean_logwage_RH_lowpsi`, `mean_logwage_RH_highpsi` :
    mean log wages for RH workers in the low- and high-ψ groups respectively
- `diff_logwage_RH_high_lowpsi` : difference between high- and low-ψ RH mean log wages
- `mean_alpha_highpsi`, `mean_alpha_lowpsi` : employment-weighted mean α in high- and low-ψ groups
- `diff_alpha_high_lowpsi` : difference (high − low) of mean α
- `var_logwage_highpsi`, `var_logwage_lowpsi` : within-group wage variances (high / low ψ)
- `ratio_var_logwage_high_lowpsi` : ratio var_high / var_low (safe-guarded against division by zero)
- `market_tightness` : returned as `res.θ`

Complexity and implementation notes
- Time complexity is O(|h_grid| * |ψ_grid|) due to the double loop over the grid.
- Uses employment `n(h,ψ)` as weights for all aggregated statistics.
- Quantile boundaries are chosen so that the low and high groups do not overlap; the
    high-group start index is forced to be at least one above the low-group end.
- Numerical safeguards avoid division by zero (checks sum of weights before dividing).
- Designed to work with ForwardDiff dual numbers by unwrapping only where needed
    (classification and final scalar guards), while preserving differentiability of
    accumulated quantities where appropriate.

Usage
Place this docstring immediately above the `compute_model_moments` function definition.
"""
function compute_model_moments(
                                prim::Primitives{T},
                                res::Results{T};
                                q_low_cut::T=0.5,
                                q_high_cut::T=0.75,
                                include::Union{Nothing, Vector{Symbol}, Symbol}=nothing
                            ) where {T<:Real}
    # If user passes include = :all or nothing -> keep default (the previously returned set)
    default_keys = [
        :mean_logwage, :var_logwage, :mean_alpha, :diff_logwage_inperson_remote,
        :hybrid_share, :agg_productivity, :dlogw_dpsi_mean_RH,
        :mean_logwage_RH_lowpsi, :mean_logwage_RH_highpsi, :diff_logwage_RH_high_lowpsi,
        :mean_alpha_highpsi, :mean_alpha_lowpsi, :diff_alpha_high_lowpsi,
        :var_logwage_highpsi, :var_logwage_lowpsi, :ratio_var_logwage_high_lowpsi,
        :diff_logwage_inperson_remote, :market_tightness
    ]

    # Normalize include argument to a Vector{Symbol} or nothing
    if include === :all
        include_keys = default_keys
    elseif isa(include, Vector{Symbol})
        include_keys = include
    elseif include === nothing
        include_keys =[
                        :mean_logwage,                                                #> Average of log(wages)
                        :var_logwage,                                                  #> Variance of log(wages)
                        :mean_alpha,                                                    #> Average of alpha
                        #// :mean_logwage_inperson,                              #> Average of log(wages) for in-person workers
                        #// :mean_logwage_remote,                                  #> Average of log(wages) for remote workers
                        :diff_logwage_inperson_remote,                #> Difference in average log(wages) between in-person and remote workers
                        #//:hybrid_share,                                                #> Share of hybrid workers
                        :agg_productivity,                                        #> Aggregate productivity
                        :dlogw_dpsi_mean_RH,                                    #> Elasticity of log(wages) with respect to market tightness for remote workers
                        #//mean_logwage_RH_lowpsi,                            #> Average of log(wages) for remote workers in low-psi region
                        #//mean_logwage_RH_highpsi,                          #> Average of log(wages) for remote workers in high-psi region
                        :diff_logwage_RH_high_lowpsi,                  #> Difference in average log(wages) between high-psi and low-psi regions
                        #//mean_alpha_highpsi,                                    #> Average of alpha for remote workers in high-psi region
                        #//mean_alpha_lowpsi,                                      #> Average of alpha for remote workers in low-psi region
                        :diff_alpha_high_lowpsi,                            #> Difference in average alpha between high-psi and low-psi regions
                        #//var_logwage_highpsi,                                  #> Variance of log(wages) for remote workers in high-psi region
                        #//var_logwage_lowpsi,                                    #> Variance of log(wages) for remote workers in low-psi region
                        #//ratio_var_logwage_high_lowpsi,              #> Ratio of variances of log(wages) between high-psi and low-psi regions
                        :market_tightness                                                   #> Market tightness (V/U ratio)
        ]
    else
        throw(ArgumentError("`include` must be nothing, :all, or Vector{Symbol}"))
    end

    # Extract key objects
    n::Matrix{T} = res.n
    w::Matrix{T} = res.w_policy
    α::Matrix{T} = res.α_policy

    @unpack h_grid, ψ_grid, ψ_cdf = prim
    # production_fun etc. left unchanged (use `prim` fields inside)
    production_fun = (h, ψ, α) -> (prim.A₀ + prim.A₁*h) * ((1 - α) + α * (prim.ψ₀ * h^prim.ϕ * ψ^prim.ν))

    total_emp::T = sum(n)
    if !(ForwardDiff.value(total_emp) > 0)
        return _zero_moments(res.θ; keys=include_keys)
    end

    # Pre-compute masks and tolerance
    αtol::Float64 = 1e-8
    valid::BitMatrix = (n .> 0.0) .& (w .> 0.0)

    # Determine ψ quantile boundaries for firm type grouping
    n_ψ::Int = length(ψ_grid)
    idx_low_end::Int = _find_quantile_index(ψ_cdf, q_low_cut, n_ψ)
    idx_high_start::Int = max(_find_quantile_index(ψ_cdf, q_high_cut, n_ψ), idx_low_end + 1)

    # Initialize accumulators (only numeric scalars; we'll select which to return later)
    sum_alpha::T = 0.0
    total_employment::T = 0.0

    sum_wts::T = 0.0
    sum_logw::T = 0.0
    sum_logw2::T = 0.0

    sum_n_inperson::T = 0.0; sum_logw_inperson::T = 0.0
    sum_n_remote::T = 0.0; sum_logw_remote::T = 0.0
    sum_n_interior::T = 0.0

    sum_production::T = 0.0

    sum_dlogw_dpsi_RH::T = 0.0
    sum_n_RH::T = 0.0

    sum_n_low::T = 0.0; sum_alpha_low::T = 0.0
    sum_logw_low::T = 0.0; sum_logw2_low::T = 0.0
    sum_logw_RH_low::T = 0.0; sum_n_RH_low::T = 0.0

    sum_n_high::T = 0.0; sum_alpha_high::T = 0.0
    sum_logw_high::T = 0.0; sum_logw2_high::T = 0.0
    sum_logw_RH_high::T = 0.0; sum_n_RH_high::T = 0.0

    # Single pass
    for (i_h, h) in enumerate(h_grid)
        for (i_ψ, ψ) in enumerate(ψ_grid)
            n_cell::T = n[i_h, i_ψ]
            n_cell > 0.0 || continue

            alpha_cell = α[i_h, i_ψ]
            sum_alpha += alpha_cell * n_cell
            total_employment += n_cell

            w_cell::T = w[i_h, i_ψ]
            α_cell::T = α[i_h, i_ψ]

            is_valid::Bool = valid[i_h, i_ψ]
            is_inperson::Bool = ForwardDiff.value(α_cell) <= αtol
            is_remote::Bool = ForwardDiff.value(α_cell) >= (1.0 - αtol)
            is_interior::Bool = (ForwardDiff.value(α_cell) > αtol) && (ForwardDiff.value(α_cell) < (1.0 - αtol))
            is_RH::Bool = ForwardDiff.value(α_cell) > αtol
            is_low_psi::Bool = i_ψ <= idx_low_end
            is_high_psi::Bool = i_ψ >= idx_high_start

            sum_production += production_fun(h, ψ, α_cell) * n_cell

            if !is_valid
                continue
            end

            logw_cell::T = log(w_cell)

            sum_wts += n_cell
            sum_logw += logw_cell * n_cell
            sum_logw2 += (logw_cell^2) * n_cell

            if is_inperson
                sum_logw_inperson += logw_cell * n_cell
                sum_n_inperson += n_cell
            elseif is_remote
                sum_logw_remote += logw_cell * n_cell
                sum_n_remote += n_cell
            end

            if is_interior
                sum_n_interior += n_cell
            end
            # Remote/hybrid wage-productivity elasticity 
            if is_RH
                # Marginal productivity of firm type: ∂g/∂ψ where g(h,ψ) = ψ₀h^ϕψ^ν
                dg_dpsi::T = prim.ψ₀ * (h^prim.ϕ) * (prim.ν * ψ^(prim.ν - 1))

                # Wage derivative incorporating Nash bargaining and arrangement costs
                dw_dpsi::T = prim.A₁ * h * dg_dpsi * (
                    (prim.ξ * α_cell) / (1.0 - prim.β * (1.0 - prim.δ)) - (1.0 - α_cell) / prim.χ
                )
                
                sum_dlogw_dpsi_RH += (dw_dpsi / w_cell) * n_cell
                sum_n_RH += n_cell
            end

            # ψ-group moments 
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

    mean_alpha = total_employment > 0 ? sum_alpha / total_employment : 0.0
    mean_logwage, var_logwage = _compute_mean_var(sum_logw, sum_logw2, sum_wts)
    mean_logwage_inperson = ForwardDiff.value(sum_n_inperson) > 0 ? sum_logw_inperson / sum_n_inperson : 0.0
    mean_logwage_remote = ForwardDiff.value(sum_n_remote) > 0 ? sum_logw_remote / sum_n_remote : 0.0
    diff_logwage_inperson_remote = mean_logwage_inperson - mean_logwage_remote

    hybrid_share = ForwardDiff.value(sum_n_interior) / ForwardDiff.value(total_emp)
    agg_productivity = ForwardDiff.value(sum_production) / ForwardDiff.value(total_emp)
    dlogw_dpsi_mean_RH = ForwardDiff.value(sum_n_RH) > 0 ? sum_dlogw_dpsi_RH / sum_n_RH : 0.0

    mean_logwage_RH_lowpsi = ForwardDiff.value(sum_n_RH_low) > 0 ? sum_logw_RH_low / sum_n_RH_low : 0.0
    mean_logwage_RH_highpsi = ForwardDiff.value(sum_n_RH_high) > 0 ? sum_logw_RH_high / sum_n_RH_high : 0.0
    diff_logwage_RH_high_lowpsi = mean_logwage_RH_highpsi - mean_logwage_RH_lowpsi

    mean_alpha_lowpsi = ForwardDiff.value(sum_n_low) > 0 ? sum_alpha_low / sum_n_low : 0.0
    mean_alpha_highpsi = ForwardDiff.value(sum_n_high) > 0 ? sum_alpha_high / sum_n_high : 0.0
    diff_alpha_high_lowpsi = mean_alpha_highpsi - mean_alpha_lowpsi

    _, var_logwage_lowpsi = _compute_mean_var(sum_logw_low, sum_logw2_low, sum_n_low)
    _, var_logwage_highpsi = _compute_mean_var(sum_logw_high, sum_logw2_high, sum_n_high)
    ratio_var_logwage_high_lowpsi = ForwardDiff.value(var_logwage_lowpsi) > 0 ? var_logwage_highpsi / var_logwage_lowpsi : 0.0

    # Build results dictionary and only keep requested keys
    full = Dict{Symbol, T}(
        :mean_logwage => mean_logwage,
        :var_logwage => var_logwage,
        :mean_alpha => mean_alpha,
        :mean_logwage_inperson => mean_logwage_inperson,
        :mean_logwage_remote => mean_logwage_remote,
        :diff_logwage_inperson_remote => diff_logwage_inperson_remote,
        :hybrid_share => hybrid_share,
        :agg_productivity => agg_productivity,
        :dlogw_dpsi_mean_RH => dlogw_dpsi_mean_RH,
        :mean_logwage_RH_lowpsi => mean_logwage_RH_lowpsi,
        :mean_logwage_RH_highpsi => mean_logwage_RH_highpsi,
        :diff_logwage_RH_high_lowpsi => diff_logwage_RH_high_lowpsi,
        :mean_alpha_highpsi => mean_alpha_highpsi,
        :mean_alpha_lowpsi => mean_alpha_lowpsi,
        :diff_alpha_high_lowpsi => diff_alpha_high_lowpsi,
        :var_logwage_highpsi => var_logwage_highpsi,
        :var_logwage_lowpsi => var_logwage_lowpsi,
        :ratio_var_logwage_high_lowpsi => ratio_var_logwage_high_lowpsi,
        :market_tightness => res.θ
    )

    result = Dict{Symbol, T}()
    for k in include_keys
        if haskey(full, k)
            result[k] = full[k]
        else
            # if user requested a key that wasn't computed, fill with zero
            result[k] = zero(T)
        end
    end

    return result
end

"""
    save_moments_to_yaml(moments::NamedTuple, filename::String)

Save model moments to a YAML file for later use as target moments in estimation.
"""
function save_moments_to_yaml(moments::Union{NamedTuple, AbstractDict}, filename::String)
    # Convert either NamedTuple or Dict to a Dict{String, Any}
    md = Dict{String, Any}()
    for k in keys(moments)
        v = moments isa AbstractDict ? moments[k] : getproperty(moments, k)
        md[string(k)] = v
    end
    YAML.write_file(filename, md)
    println("Saved moments to: $filename")
end

function load_moments_from_yaml(filename::String)
    moments_dict = YAML.load_file(filename)
    # Return as Dict{Symbol, Any} so downstream code can index by Symbol keys
    return Dict(Symbol(k) => v for (k,v) in pairs(moments_dict))
end

"""
    perturb_parameters(prim::Primitives; scale::Float64=0.1) -> Dict{Symbol, Float64}

Create perturbed parameter values for testing parameter recovery.
Returns a dictionary of parameter perturbations.
"""
function perturb_parameters(prim::Primitives;
                            param_list::Vector{Symbol} = Symbol[],
                            scale::Float64=0.1)
    
    # Key parameters to perturb for testing
    params = Dict{Symbol, Float64}()
    
    # Production parameters
    params[:A₁] = prim.A₁ * (1.0 + scale * randn())
    params[:ψ₀] = prim.ψ₀ * (1.0 + scale * randn())
    params[:ν] = max(0.1, prim.ν * (1.0 + scale * randn()))
    
    # Human capital distribution
    params[:aₕ] = max(0.5, prim.aₕ * (1.0 + scale * randn()))
    params[:bₕ] = max(0.5, prim.bₕ * (1.0 + scale * randn()))

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
    compute_distance(model_moments::NamedTuple, data_moments::NamedTuple;
                        weighting_matrix::Union{Matrix, Nothing}=nothing,
                        matrix_moment_order::Union{Vector{Symbol}, Nothing}=nothing)

Compute a scalar distance between `model_moments` and `data_moments`.

Description
- Builds an error vector `g` containing the elementwise differences `model_moments[k] - data_moments[k]`
    for each key `k` in `model_moments` (the key iteration order of the `NamedTuple` determines the order of `g`).
- If `weighting_matrix` is `nothing`, the function returns the unweighted sum of squared errors: dot(g, g).
- If a `weighting_matrix` is provided, the function returns the quadratic form g' * W * g.
- If `matrix_moment_order` is supplied and differs from the current moment order, the weighting matrix
    is permuted to align with the ordering used to construct `g`.

Arguments
- `model_moments::NamedTuple` : Named tuple of model-implied moments (keys are moment names).
- `data_moments::NamedTuple`  : Named tuple of data moments with the same keys as `model_moments`.
- `weighting_matrix::Union{Matrix, Nothing}` (keyword, default `nothing`) :
    - If `nothing`, compute and return the unweighted sum of squared errors.
    - Otherwise a square matrix (typically symmetric) used to weight the moment errors.
- `matrix_moment_order::Union{Vector{Symbol}, Nothing}` (keyword, default `nothing`) :
    - When provided, this vector specifies the moment names in the order that the rows/columns of
    `weighting_matrix` correspond to. If it differs from the `NamedTuple` key order, the function
    will permute `weighting_matrix` so its order matches the `NamedTuple` keys before forming g' * W * g.

Returns
- A scalar (numeric) distance value:
    - `dot(g, g)` when no weighting matrix is provided.
    - `g' * W * g` when a weighting matrix is provided (after any required permutation).

Notes and requirements
- `data_moments` must contain the same keys as `model_moments`. The ordering of keys in `model_moments`
    determines the layout of the error vector `g`.
- If `matrix_moment_order` is provided, it must contain exactly the same set of symbols as the
`NamedTuple` keys (possibly in a different order) and have length equal to `size(weighting_matrix, 1)`.
- The weighting matrix should be square and conformable with the length of `g`. Typical usage assumes
    symmetric positive-definite weighting matrices, but the function will compute g' * W * g for any
    conformable numeric matrix.
- Mismatched sizes or missing keys will result in a runtime error.

Examples
- Unweighted distance:
    model = (m1=1.0, m2=2.0)
    data  = (m1=0.8, m2=2.1)
    compute_distance(model, data)         # returns 0.2^2 + (-0.1)^2

- Weighted distance with permutation:
    W = ...                              # 2×2 weighting matrix ordered as [:m2, :m1]
    compute_distance(model, data, W, [:m2, :m1])
"""
function compute_distance(
                            model_moments::Union{NamedTuple, AbstractDict},
                            data_moments::Union{NamedTuple, AbstractDict},
                            weighting_matrix::Union{Matrix, Nothing}=nothing,
                            matrix_moment_order::Union{Vector{Symbol}, Nothing}=nothing
                        )
    # Collect moment keys as symbols in the order provided by model_moments
    current_moment_order = Symbol.(collect(keys(model_moments)))

    # Helper to extract value for either NamedTuple or Dict
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
    # Unpack our fixed data and parameter names from `p`
    prim_base = p.prim_base
    res_base = p.res_base
    target_moments = p.target_moments
    param_names = p.param_names # e.g., [:A₁, :ν, :c₀]
    weighting_matrix = p.weighting_matrix
    matrix_moment_order = p.matrix_moment_order
    fixed_point_options = get(p, :fixed_point_options, Dict())

    # --- Step 1: Update primitives ---
    # Create a dictionary mapping param names to the new values from the optimizer
    params_to_update = Dict(param_names .=> params)

    # --- Step 2 update and re-solve the model ---
    prim_new, res_new = update_params_and_resolve(
                                                    prim_base, res_base; 
                                                    params_to_update=params_to_update,
                                                    fixed_point_options=fixed_point_options
                                                )

    # --- Step 3: Compute model moments ---
    model_moments = compute_model_moments(prim_new, res_new)

    # --- Step 4: Compute the distance ---
    loss = compute_distance( model_moments, target_moments, weighting_matrix, matrix_moment_order )

    return loss
end


