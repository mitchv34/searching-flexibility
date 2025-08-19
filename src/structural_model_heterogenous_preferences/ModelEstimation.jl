# ModelEstimation.jl depends on types and functions defined in ModelSetup.jl and
# ModelSolver.jl (e.g. `Primitives`, `Results`, `update_params_and_resolve`).
# To avoid re-defining docstrings and symbols when files are included multiple
# times in the same session, do NOT include the core files here. The top-level
# runner should include `ModelSetup.jl` and `ModelSolver.jl` once before
# including `ModelEstimation.jl`.

using Random, Statistics, Distributions, LinearAlgebra , ForwardDiff
using Printf, Term, YAML
using CSV, DataFrames

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
    # shape parameter for z distribution
    k    = get(params_to_update, :k, prim.k)

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

    # --- Stage 2: Promote numeric type if needed ---
    T_old = eltype(prim.h_grid)
    T_new = promote_type(typeof.(values(params_to_update))..., T_old)
    if T_new != T_old
        # Promote scalars
        A₀ = T_new(A₀); A₁ = T_new(A₁); ψ₀ = T_new(ψ₀); ϕ = T_new(ϕ); ν = T_new(ν)
        c₀ = T_new(c₀); χ = T_new(χ); γ₀ = T_new(γ₀); γ₁ = T_new(γ₁)
        κ₀ = T_new(κ₀); κ₁ = T_new(κ₁); βv = T_new(βv); δv = T_new(δv)
        b  = T_new(b);   ξ  = T_new(ξ);   k  = T_new(k)

        ψ_min = T_new(ψ_min); ψ_max = T_new(ψ_max)
        aₕ = T_new(aₕ); bₕ = T_new(bₕ)
        h_min = T_new(h_min); h_max = T_new(h_max)

        # Promote vectors
        ψ_grid = T_new.(ψ_grid); ψ_pdf = T_new.(ψ_pdf); ψ_cdf = T_new.(ψ_cdf)
        h_grid = T_new.(h_grid); h_pdf = T_new.(h_pdf); h_cdf = T_new.(h_cdf)
    end

    # --- Stage 3: Recompute h and z distribution only if relevant params changed ---
    recompute_h = any(k -> k in (:aₕ, :bₕ, :h_grid, :h_min, :h_max), keys(params_to_update))
    if recompute_h
        h_scaled = (h_grid .- h_min) ./ (h_max - h_min)
        beta_dist = Distributions.Beta(aₕ, bₕ)
        h_pdf_raw = pdf.(beta_dist, h_scaled)
        h_pdf = h_pdf_raw ./ sum(h_pdf_raw)
        h_cdf = cumsum(h_pdf)
        h_cdf ./= h_cdf[end]
    end
    recompute_z = :k in keys(params_to_update)
    if recompute_z
        z_dist = Distributions.Gamma(k, 1)
    else
        z_dist = deepcopy(prim.z_dist)
    end

    # --- Stage 4: Build validated Primitives and fresh Results ---
    new_prim = validated_Primitives(
        A₀=A₀, A₁=A₁, ψ₀=ψ₀, ϕ=ϕ, ν=ν, c₀=c₀, χ=χ, γ₀=γ₀, γ₁=γ₁,
        k=k, z_dist=z_dist,
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
        :mean_logwage_inperson, :mean_logwage_remote,
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

function _compute_mean_var(sum_x::T, sum_x2::T, sum_wts::T)::Tuple{T, T} where {T<:Real}
    if ForwardDiff.value(sum_wts) > 0
        μ = sum_x / sum_wts
        σ2 = max(0.0, (sum_x2 / sum_wts) - μ^2)
        return μ, σ2
    else
        return 0.0, 0.0
    end
end

function _find_quantile_index(cdf::Vector{T}, quantile::T, n::Int)::Int where {T<:Real}
    idx = findfirst(>=(ForwardDiff.value(quantile)), ForwardDiff.value.(cdf))
    return idx === nothing ? n : idx
end

function calculate_average_policies(prim::Primitives{T}, res::Results{T}) where {T<:Real}
    @unpack n_h, n_ψ, h_grid, ψ_grid, k, β, δ, ξ, c₀, χ = prim
    @unpack U, S = res
    
    avg_alpha = Matrix{T}(undef, n_h, n_ψ)
    avg_alpha_sq = Matrix{T}(undef, n_h, n_ψ) # For variance calculation
    avg_wage = Matrix{T}(undef, n_h, n_ψ)

    z_dist = Gamma(k, 1.0)
    E = expectation(z_dist)

    Threads.@threads for i_h in 1:n_h
        h = h_grid[i_h]
        for i_ψ in 1:n_ψ
            ψ = ψ_grid[i_ψ]

            # --- E[α] and E[α^2] ---
            alpha_func(z) = optimal_alpha_given_z(prim, h, ψ, z)
            avg_alpha[i_h, i_ψ] = E(alpha_func)
            avg_alpha_sq[i_h, i_ψ] = E(z -> alpha_func(z)^2)

            # --- E[w] ---
            wage_func(z) = begin
                alpha_star = alpha_func(z)
                base_wage = (1 - β*(1-δ)) * (U[i_h] + ξ * S[i_h, i_ψ]) - (β*δ*U[i_h])
                compensating_diff = c₀ * z * (1 - alpha_star)^(1 + χ) / (1 + χ)
                return base_wage + compensating_diff
            end
            avg_wage[i_h, i_ψ] = E(wage_func)
        end
    end
    
    return avg_alpha, avg_alpha_sq, avg_wage
end

function compute_model_moments(
                                prim::Primitives{T},
                                res::Results{T};
                                q_low_cut::T=0.5,
                                q_high_cut::T=0.75,
                                include::Union{Nothing, Vector{Symbol}, Symbol}=nothing
                            ) where {T<:Real}

    default_keys = [
        :mean_logwage, :var_logwage, :mean_alpha, :var_alpha, :diff_logwage_inperson_remote,
        :hybrid_share, :agg_productivity, :dlogw_dpsi_mean_RH,
        :mean_logwage_RH_lowpsi, :mean_logwage_RH_highpsi, :diff_logwage_RH_high_lowpsi,
        :mean_alpha_highpsi, :mean_alpha_lowpsi, :diff_alpha_high_lowpsi,
        :var_logwage_highpsi, :var_logwage_lowpsi, :ratio_var_logwage_high_lowpsi,
        :diff_logwage_inperson_remote, :market_tightness
    ]

    if include === :all
        include_keys = default_keys
    elseif isa(include, Vector{Symbol})
        include_keys = include
    elseif include === nothing
        include_keys = default_keys
    else
        throw(ArgumentError("`include` must be nothing, :all, or Vector{Symbol}"))
    end

    n::Matrix{T} = res.n
    
    # MAJOR FIX: Calculate actual expected policies instead of using zeros
    avg_alpha_policy, avg_alpha_sq_policy, avg_wage_policy = calculate_average_policies(prim, res)

    @unpack h_grid, ψ_grid, ψ_cdf = prim
    production_fun = (h, ψ, α) -> (prim.A₀ + prim.A₁*h) * ((1 - α) + α * (prim.ψ₀ * h^prim.ϕ * ψ^prim.ν))

    total_emp::T = sum(n)
    if !(ForwardDiff.value(total_emp) > 0)
        return _zero_moments(res.θ; keys=include_keys)
    end

    αtol::Float64 = 1e-8
    valid::BitMatrix = (n .> 0.0) .& (avg_wage_policy .> 0.0)

    n_ψ::Int = length(ψ_grid)
    idx_low_end::Int = _find_quantile_index(ψ_cdf, q_low_cut, n_ψ)
    idx_high_start::Int = max(_find_quantile_index(ψ_cdf, q_high_cut, n_ψ), idx_low_end + 1)

    sum_alpha::T = 0.0
    sum_alpha_sq::T = 0.0  # NEW: For variance calculation
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

    for (i_h, h) in enumerate(h_grid)
        for (i_ψ, ψ) in enumerate(ψ_grid)
            n_cell::T = n[i_h, i_ψ]
            n_cell > 0.0 || continue

            # Use the computed expected policies instead of zeros
            alpha_cell = avg_alpha_policy[i_h, i_ψ]
            alpha_sq_cell = avg_alpha_sq_policy[i_h, i_ψ]
            w_cell = avg_wage_policy[i_h, i_ψ]
            
            # Accumulate alpha moments for mean and variance
            sum_alpha += alpha_cell * n_cell
            sum_alpha_sq += alpha_sq_cell * n_cell  # NEW: For variance calculation
            total_employment += n_cell

            α_cell::T = alpha_cell  # For compatibility with existing code

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
            if is_RH
                dg_dpsi::T = prim.ψ₀ * (h^prim.ϕ) * (prim.ν * ψ^(prim.ν - 1))
                dw_dpsi::T = prim.A₁ * h * dg_dpsi * (
                    (prim.ξ * α_cell) / (1.0 - prim.β * (1.0 - prim.δ)) - (1.0 - α_cell) / prim.χ
                )
                sum_dlogw_dpsi_RH += (dw_dpsi / w_cell) * n_cell
                sum_n_RH += n_cell
            end

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
    # MAJOR FIX: Calculate var_alpha properly using E[α²] - (E[α])²
    E_alpha_sq = total_employment > 0 ? sum_alpha_sq / total_employment : 0.0
    var_alpha = E_alpha_sq - mean_alpha^2
    
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

    full = Dict{Symbol, T}(
        :mean_logwage => mean_logwage,
        :var_logwage => var_logwage,
        :mean_alpha => mean_alpha,
        :var_alpha => var_alpha,  # MAJOR FIX: Actually include var_alpha in the output
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
            result[k] = zero(T)
        end
    end

    return result
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

function load_moments_from_yaml(filename::String)
    moments_dict = YAML.load_file(filename)
    return Dict(Symbol(k) => v for (k,v) in pairs(moments_dict))
end

"""
Compute empirical mean and variance of alpha from a CSV data file.
If `weight_col` is provided, use it as observation weights.

Arguments
- `path::String`: path to CSV file
- `alpha_col::Symbol`: column name for alpha
- `weight_col::Union{Nothing, Symbol}`: optional weights column

Returns Dict(:mean_alpha => ..., :var_alpha => ...)
"""
function compute_empirical_alpha_moments(path::String; alpha_col::Symbol=:alpha, weight_col::Union{Nothing,Symbol}=nothing)
    # CSV and DataFrames are imported at the top of this file. Do not use
    # `using` inside a function (Julia requires `using` at top-level).
    df = CSV.read(path, DataFrame)
    if !(alpha_col in names(df))
        error("alpha column not found in data: $alpha_col")
    end
    α = df[!, alpha_col]
    if !isnothing(weight_col)
        if !(weight_col in names(df))
            error("weight column not found: $weight_col")
        end
        w = df[!, weight_col]
        sw = sum(w)
        μ = sum(w .* α) / sw
        μ2 = sum(w .* (α .^ 2)) / sw
    else
        μ = mean(α)
        μ2 = mean(α .^ 2)
    end
    return Dict(:mean_alpha => μ, :var_alpha => max(0.0, μ2 - μ^2))
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
    # --- WARM VS. COLD LOGIC ---
    # Check if the parameter struct 'p' has the cache.
    # If it does, use it for a warm start. If not, do a cold start.
    
    # Optional warm-start S if available
    initial_S = haskey(p, :last_res) ? p.last_res[].S : nothing
    
    # Read the current solver state
    current_solver_params = p.solver_state[] 
    λ_S_start = current_solver_params.λ_S_init
    λ_u_start = current_solver_params.λ_u_init


    # Solve with updated params (deepcopy res_base to avoid mutation across calls)
    prim_new, res_new = update_primitives_results(
                                                    p.prim_base, 
                                                    deepcopy(p.res_base),
                                                    params_to_update
                                                    );
    λ_S_final, λ_u_final = solve_model(
        prim_new, res_new,
        initial_S = initial_S,
        tol      = get(p, :tol, 1e-7),
        max_iter = get(p, :max_iter, 25_000),
        verbose  = false,
        λ_S_init      = get(p, :λ_S_init, 0.01),
        λ_u_init      = get(p, :λ_u_init, 0.01)
    )

    # Update warm-start cache
    if haskey(p, :last_res)
        p.last_res[] = res_new
        # Update the solver state cache with the new final lambdas
        p.solver_state[] = (λ_S_init = λ_S_final, λ_u_init = λ_u_final)
    end

    
    model_moments = compute_model_moments(prim_new, res_new)

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
                             burnin_kwargs=NamedTuple())

Build an OptimizationProblem for estimation. Choose warm vs cold start via `use_warm_start`.

- If `use_warm_start == true`, perform a burn-in solve at `initial_param_guess` to seed `last_res`.
- Otherwise, use `res_base` as the initial state (cold start).
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
    solver_state = Ref((λ_S_init=λ_S_init, λ_u_init=λ_u_init))

    p = (
        prim_base = prim_base,
        res_base = res_base,
        target_moments = target_moments,
        param_names = params_to_estimate,
        last_res = last_res_ref,         # enables warm starts in objective_function
        solver_state = solver_state,     # rolling λ cache (updated in objective_function)
        weighting_matrix = weighting_matrix,
        matrix_moment_order = matrix_moment_order,
        # pass-through solver controls used by objective_function via get(p, ...)
        tol = tol,
        max_iter = max_iter,
        λ_S_init = λ_S_init,
        λ_u_init = λ_u_init
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
