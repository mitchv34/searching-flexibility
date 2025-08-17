#==========================================================================================
Module: ModelSolver.jl
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2025-02-27
Description: Contains the core functions to solve for the steady-state equilibrium
        of the random search labor market model. The main entry point is the
        `solve_model` function, which orchestrates the entire solution process.
==========================================================================================#
#? Main Solver Function
#==========================================================================================#
using Parameters, Printf, Term
using ForwardDiff, Roots, Distributions
using NonlinearSolve, SciMLBase
using NLsolve

# Type-stable primitives update (moved from estimation timing prototype)
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
        b  = T_new(b);   ξ  = T_new(ξ)

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
        A₀=A₀, A₁=A₁, ψ₀=ψ₀, ϕ=ϕ, ν=ν, c₀=c₀, χ=χ, γ₀=γ₀, γ₁=γ₁,
        κ₀=κ₀, κ₁=κ₁, β=βv, δ=δv, b=b, ξ=ξ,
        n_ψ=n_ψ, ψ_min=ψ_min, ψ_max=ψ_max, ψ_grid=ψ_grid, ψ_pdf=ψ_pdf, ψ_cdf=ψ_cdf,
        aₕ=aₕ, bₕ=bₕ, n_h=n_h, h_min=h_min, h_max=h_max, h_grid=h_grid, h_pdf=h_pdf, h_cdf=h_cdf,
    )

    new_res = Results(new_prim)
    return new_prim, new_res
end
function solve_model(
                        prim::Primitives{T},
                        res::Results{T};
                        initial_S::Union{Matrix{T}, Nothing}=nothing,
                        tol::Float64=1e-7,
                        max_iter::Int=5000,
                        verbose::Bool=true,
                        print_freq::Int=100,
                        # Initial damping values
                        λ_S_init::Float64 = 0.1,
                        λ_u_init::Float64 = 0.1,
                        # Damping control parameters
                        λ_min::Float64 = 0.01,           # Higher minimum to avoid getting stuck
                        λ_max::Float64 = 0.5,            # Conservative maximum
                        λ_increase_factor::Float64 = 1.2, # More aggressive increases
                        λ_decrease_factor::Float64 = 0.7  # Less aggressive decreases
                    ) where {T<:Real}

    @unpack h_grid, ψ_grid, β, δ, ξ, κ₀, κ₁, n_h, n_ψ, γ₀, γ₁ = prim

    f_h = copy(prim.h_pdf)
    f_ψ = copy(prim.ψ_pdf)
    s_flow = calculate_flow_surplus(prim, res)

    S = if !isnothing(initial_S)
        copy(initial_S)
    else
        copy(s_flow)
    end

    u = copy(f_h)
    denom = 1.0 - β * (1.0 - δ)
    
    # Initialize adaptive damping state (decoupled for S and u)
    ΔS = Inf
    ΔS_prev = Inf
    Δu = Inf
    Δu_prev = Inf
    λ_S = λ_S_init
    λ_u = λ_u_init
    θ = NaN

    for k in 1:max_iter
        # Store state at the beginning of the iteration
        S_old = copy(S)
        u_old = copy(u)

        # --- Part 1: Update S using vectorized operations ---
        L = sum(u_old)
        u_dist = u_old ./ L

        # Compute expected value of posting a vacancy
        S_positive = max.(0.0, ForwardDiff.value.(S_old))
        weighted_surplus = (1.0 - ξ) .* S_positive .* u_dist
        B = vec(sum(weighted_surplus, dims=1))

        # Compute market tightness and meeting rates
        Integral = sum((B ./ κ₀).^(1/κ₁) .* f_ψ)
        θ = ((1/L) * Integral)^(κ₁ / (κ₁ + γ₁))
        p = θ^(1 - γ₁)
        q = θ^(-γ₁)

        # Compute the endogenous vacancy distribution
        v = ((q .* B) ./ κ₀).^(1/κ₁)
        V = v' * f_ψ
        Γ = V > 0 ? (v .* f_ψ) ./ V : zeros(T, n_ψ)

        # Compute the expected value of searching for a job
        ExpectedSearch = S_positive * Γ  # Matrix-vector multiplication
        S_update = (s_flow .- (β .* p .* ξ .* ExpectedSearch)) ./ denom

        # Apply the update to S
        S .= (1.0 - λ_S) .* S_old .+ λ_S .* S_update

        # --- Part 2: Update the unemployment distribution ---
        # Recalculate with new surplus function
        S_positive_new = max.(0.0, ForwardDiff.value.(S))
        weighted_surplus_new = (1.0 - ξ) .* S_positive_new .* u_dist
        B_new = vec(sum(weighted_surplus_new, dims=1))

        Integral_new = sum((B_new ./ κ₀).^(1/κ₁) .* f_ψ)
        θ_new = ((1/L) * Integral_new)^(κ₁ / (κ₁ + γ₁))
        p_new = θ_new^(1 - γ₁)
        q_new = θ_new^(-γ₁)

        v_new = ((q_new .* B_new) ./ κ₀).^(1/κ₁)
        V_new = sum(v_new .* f_ψ)
        Γ_new = V_new > 0 ? (v_new .* f_ψ) ./ V_new : zeros(T, n_ψ)

        # Probability of acceptance calculation
        acceptance_matrix = (ForwardDiff.value.(S) .> 0.0)
        ProbAccept = acceptance_matrix * Γ_new

        # Employment distribution calculation
        inflow = p_new .* u_old .* ProbAccept
        n_new = inflow ./ δ

        # Unemployment update
        u_update = max.(0.0, f_h .- n_new)
        u .= (1.0 - λ_u) .* u_old .+ λ_u .* u_update

        # --- Part 3: Calculate Changes ---
        ΔS_prev = ΔS
        ΔS = maximum(abs.(S .- S_old))

        Δu_prev = Δu
        Δu = maximum(abs.(u .- u_old))

        # --- Part 4: Decoupled Adaptive Damping ---
        if k > 1
            # Adapt λ_S based ONLY on ΔS
            if ΔS > ΔS_prev
                λ_S = max(λ_min, λ_S * λ_decrease_factor)
            else
                λ_S = min(λ_max, λ_S * λ_increase_factor)
            end

            # Adapt λ_u based ONLY on Δu
            if Δu > Δu_prev
                λ_u = max(λ_min, λ_u * λ_decrease_factor)
            else
                λ_u = min(λ_max, λ_u * λ_increase_factor)
            end
        end

        # --- Part 5: Print and Check for Convergence ---
        if verbose && (k % print_freq == 0 || k == 1)
            @printf("Iter %4d: ΔS = %.6e, Δu = %.6e, L = %.6f, θ = %.6f, λ_S=%.4f, λ_u=%.4f\n",
                    k, ForwardDiff.value(ΔS), ForwardDiff.value(Δu), L, θ, λ_S, λ_u)
        end

        # Convergence check is ONLY on ΔS
        if ForwardDiff.value(ΔS) < tol
            if verbose
                println(@bold @green "Converged after $k iterations (ΔS=$ΔS).")
            end
            break
        end
    end

    # Post-processing 
    L_final = sum(u)
    u_dist_final = L_final > 0 ? u ./ L_final : zeros(T, n_h)
    
    S_positive_final = max.(0.0, S)
    weighted_surplus_final = (1.0 - ξ) .* S_positive_final .* u_dist_final
    B_final = vec(sum(weighted_surplus_final, dims=1))
    
    Integral_final = sum((B_final ./ κ₀).^(1/κ₁) .* f_ψ)
    θ_final = ((1/L_final) * Integral_final)^(κ₁ / (κ₁ + γ₁))
    p_final = θ_final^(1 - γ₁)
    q_final = θ_final^(-γ₁)
    v_final = ((q_final .* B_final) ./ κ₀).^(1/κ₁)

    # Update results
    res.S .= S
    res.θ = θ_final
    res.p = p_final
    res.q = q_final
    res.v .= v_final
    res.u .= u

    calculate_final_policies!(res, prim)
    
    # Return final damping factors
    return λ_S, λ_u
end
#?=========================================================================================
#? Helper Functions
#?=========================================================================================

function calculate_flow_surplus(prim::Primitives{T}, res::Results{T}) where {T<:Real}
    @unpack h_grid, ψ_grid, b = prim
    
    # Create broadcasting-compatible arrays
    h_broadcast = h_grid  # (n_h,)
    ψ_broadcast = ψ_grid' # (1, n_ψ) - transposed for broadcasting
    
    # Vectorized production calculation
    Y = (prim.A₀ .+ prim.A₁ .* h_broadcast) .* 
        ((1 .- res.α_policy) .+ res.α_policy .* 
            (prim.ψ₀ .* (h_broadcast).^prim.ϕ .* ψ_broadcast.^prim.ν))
    
    # Vectorized utility calculation  
    non_wage_utility = -(prim.c₀ .* (1 .- res.α_policy).^(prim.χ + 1) ./ (prim.χ + 1))
    
    # Vectorized surplus calculation
    s_flow = Y .+ non_wage_utility .- (b .* h_broadcast)
    
    return s_flow
end

function calculate_final_policies!(res::Results{T}, prim::Primitives{T}) where {T<:Real}
    @unpack β, δ, ξ, b, n_h, n_ψ, h_grid = prim
    @unpack p, S, v, u, α_policy = res
    
    f_ψ = prim.ψ_pdf
    
    # Calculate vacancy distribution
    V = v' * f_ψ
    Gamma = V > 0.0 ? (v .* f_ψ) ./ V : zeros(T, n_ψ)
    
    # 1. Unemployment values - use matrix multiplication for efficiency
    exp_val = max.(0.0, ForwardDiff.value(S)) * Gamma
    res.U = (b .* h_grid .+ β .* p .* ξ .* exp_val) ./ (1.0 - β)
    
    # 2. Wage policy - let Julia handle broadcasting
    flow_utility = (1.0 - β * (1.0 - δ)) .* (res.U .+ ξ .* res.S) .- (β * δ .* res.U)
    cost_term = prim.c₀ .* (1 .- α_policy).^(prim.χ + 1) ./ (prim.χ + 1)
    res.w_policy = flow_utility .+ cost_term
    
    # 3. Employment distribution
    employment_flow = p .* (u * Gamma')
    acceptance_mask = (ForwardDiff.value.(S) .> 0.0)
    res.n = (acceptance_mask .* employment_flow) ./ δ
    
    return nothing
end


#!==========================================================================================
#! EXPERIMENTS
#!==========================================================================================
