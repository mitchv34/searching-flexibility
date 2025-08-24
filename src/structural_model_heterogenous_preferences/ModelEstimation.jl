# ModelEstimation.jl depends on types and functions defined in ModelSetup.jl and
# ModelSolver.jl (e.g. `Primitives`, `Results`, `update_params_and_resolve`).
# To avoid re-defining docstrings and symbols when files are included multiple
# times in the same session, do NOT include the core files here. The top-level
# runner should include `ModelSetup.jl` and `ModelSolver.jl` once before
# including `ModelEstimation.jl`.

using Random, Statistics, Distributions, LinearAlgebra, ForwardDiff
using Printf, Term, YAML
using CSV, DataFrames
using QuadGK # For numerical integration in Gumbel model
# Note: Optimization packages should be imported by the calling script

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
    @unpack n_h, n_ψ, h_grid, ψ_grid, A₀, A₁, ψ₀, ν, ϕ, c₀, μ = prim
    
    h_col = reshape(h_grid, n_h, 1)
    ψ_row = reshape(ψ_grid, 1, n_ψ)
    
    A_h = A₀ .+ A₁ .* h_col
    g = @. ψ₀ * exp(ν * ψ_row + ϕ * h_col)
    
    # This is the deterministic value function V(α)
    V_alpha(α, h_idx, ψ_idx) = A_h[h_idx] * ((1-α) + α * g[h_idx, ψ_idx]) - c₀ * (1-α)

    # Pre-calculate the denominator (the integral) for every (h,ψ) pair
    integral_val = Matrix{eltype(h_grid)}(undef, n_h, n_ψ)
    Threads.@threads for j in 1:n_ψ
        for i in 1:n_h
            integrand(α) = exp(V_alpha(α, i, j) / μ)
            integral_val[i, j], _ = quadgk(integrand, 0.0, 1.0)
        end
    end

    # Now define the closure for the PDF, p(α) = exp(V/μ) / ∫exp(V/μ)
    function p_alpha_func(α, h_idx, ψ_idx)
        numerator = exp(V_alpha(α, h_idx, ψ_idx) / μ)
        return numerator / integral_val[h_idx, ψ_idx]
    end

    return p_alpha_func # Return the PDF function
end

function calculate_average_policies(prim::Primitives{T}, res::Results{T}) where {T<:Real}
    @unpack n_h, n_ψ, β, δ, ξ, c₀ = prim
    
    # Get the correctly defined PDF function
    p_alpha_func = _calculate_policy_arrays(prim)
    
    avg_alpha = Matrix{T}(undef, n_h, n_ψ)
    avg_alpha_sq = Matrix{T}(undef, n_h, n_ψ)
    avg_wage = Matrix{T}(undef, n_h, n_ψ)
    
    # Define the Base Wage Component
    flow_value_of_unemployment = (1 - β) .* res.U
    base_wage_component = flow_value_of_unemployment .+ ξ .* res.S .* (1 - β * (1 - δ))

    # Use Threads.@threads for performance
    Threads.@threads for j in 1:n_ψ
        for i in 1:n_h
            # Integrate α * p(α|h,ψ) to get E[α|h,ψ]
            integrand_alpha(α) = α * p_alpha_func(α, i, j)
            avg_alpha[i, j], _ = quadgk(integrand_alpha, 0.0, 1.0, rtol=1e-6)
            
            # Integrate α² * p(α|h,ψ) to get E[α²|h,ψ]
            integrand_alpha_sq(α) = (α^2) * p_alpha_func(α, i, j)
            avg_alpha_sq[i, j], _ = quadgk(integrand_alpha_sq, 0.0, 1.0, rtol=1e-6)
        end
    end
    
    # The expected wage is Base Wage + E[c(1-α)] = Base Wage + c₀ * (1 - E[α])
    @. avg_wage = base_wage_component + c₀ * (1.0 - avg_alpha)
    
    return avg_alpha, avg_alpha_sq, avg_wage
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
        :mean_logwage, :var_logwage,
        :mean_alpha, :var_alpha,
        :inperson_share, :hybrid_share, :remote_share,
        :agg_productivity,
        :diff_alpha_high_lowpsi,
        :market_tightness, :job_finding_rate
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

    avg_alpha_policy, avg_alpha_sq_policy, avg_wage_policy = calculate_average_policies(prim, res)

    @unpack h_grid, ψ_grid, ψ_cdf, ψ₀, ν, ϕ, A₀, A₁ = prim
    
    αtol_val::Float64 = Float64(αtol)
    valid::BitMatrix = (n .> 0.0) .& (avg_wage_policy .> 0.0)

    total_emp::T = sum(n .* valid)
    if !(ForwardDiff.value(total_emp) > 0)
        return _zero_moments(res.θ; keys=include_keys)
    end
    
    idx_low_end::Int = _find_quantile_index(ψ_cdf, q_low_cut, length(ψ_grid))
    idx_high_start::Int = max(_find_quantile_index(ψ_cdf, .9, length(ψ_grid)), idx_low_end + 1)

    #> Masks
    is_inperson = @. ForwardDiff.value(avg_alpha_policy) <= αtol_val
    is_remote = @. ForwardDiff.value(avg_alpha_policy) >= (1.0 - αtol_val)
    is_hybrid = @. (ForwardDiff.value(avg_alpha_policy) > αtol_val) && (ForwardDiff.value(avg_alpha_policy) < (1.0 - αtol_val))
    is_RH = .!is_inperson # Remote or Hybrid
    is_low_psi = zeros(Bool, size(is_hybrid)); is_low_psi[:, 1:idx_low_end] .= true;
    is_high_psi = zeros(Bool, size(is_hybrid)); is_high_psi[:, idx_high_start:end] .= true;

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
    
    # --- Shares (denominator is total_employment) ---
    inperson_share = total_employment > 0 ? sum(n .* is_inperson) / total_employment : 0.0
    remote_share = total_employment > 0 ? sum(n .* is_remote) / total_employment : 0.0
    hybrid_share = total_employment > 0 ? sum(n .* is_hybrid) / total_employment : 0.0
    
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

    # Moment for c₀: Compensating Wage Differential
    sum_n_inperson_valid = sum(n .* is_inperson .* valid)
    sum_n_remote_valid = sum(n .* is_remote .* valid)
    mean_logwage_inperson = sum_n_inperson_valid > 0 ? sum(log.(max.(1e-12, avg_wage_policy)) .* n .* is_inperson .* valid) / sum_n_inperson_valid : 0.0
    mean_logwage_remote = sum_n_remote_valid > 0 ? sum(log.(max.(1e-12, avg_wage_policy)) .* n .* is_remote .* valid) / sum_n_remote_valid : 0.0
    diff_logwage_inperson_remote = mean_logwage_inperson - mean_logwage_remote

    # Moment for ψ₀: Wage Premium for High-ψ Firms (on RH subsample)
    sum_n_RH_low_valid = sum(n .* is_low_psi .* is_RH .* valid)
    sum_n_RH_high_valid = sum(n .* is_high_psi .* is_RH .* valid)
    mean_logwage_RH_lowpsi = sum_n_RH_low_valid > 0 ? sum(log.(max.(1e-12, avg_wage_policy)) .* n .* is_low_psi .* is_RH .* valid) / sum_n_RH_low_valid : 0.0
    mean_logwage_RH_highpsi = sum_n_RH_high_valid > 0 ? sum(log.(max.(1e-12, avg_wage_policy)) .* n .* is_high_psi .* is_RH .* valid) / sum_n_RH_high_valid : 0.0
    wage_premium_high_psi = mean_logwage_RH_highpsi - mean_logwage_RH_lowpsi

    # Moment for ν: Wage Slope (Proxy)
    # We use the wage premium as a proxy for the slope in the theoretical moments.
    # The full SMM will correct this by running the actual regression.
    wage_slope_psi = wage_premium_high_psi

    # Create full dictionary of ALL computed moments
    full = Dict{Symbol, T}(
        :mean_logwage => mean_logwage,
        :var_logwage => var_logwage,
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
    
    model_moments = compute_model_moments(prim_new, res_new; include=collect(keys(p.target_moments)))

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
        λ_u_init = 0.05
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
