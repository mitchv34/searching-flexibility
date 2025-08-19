#==========================================================================================
# Module: ModelSolver.jl
# Description: Contains the core functions to solve for the steady-state equilibrium.
==========================================================================================#

using Parameters, Printf, Term, Distributions, ForwardDiff
using Expectations

#?=========================================================================================
#? Main Solver Function
#?=========================================================================================

function solve_model(
                        prim::Primitives{T},
                        res::Results{T};
                        initial_S::Union{Matrix{T}, Nothing}=nothing,
                        tol::Float64=1e-8,
                        max_iter::Int=5000,
                        verbose::Bool=true,
                        print_freq::Int=50,
                        λ_S_init::Float64 = 0.01,
                        λ_u_init::Float64 = 0.01
                    ) where {T<:Real}

    @unpack h_grid, ψ_grid, β, δ, ξ, κ₀, κ₁, n_h, n_ψ, γ₀, γ₁ = prim
    f_h = copy(prim.h_pdf)
    f_ψ = copy(prim.ψ_pdf)

    # --- CHANGE: Calculate the smooth, expected flow surplus at the start ---
    s_flow = calculate_expected_flow_surplus(prim)

    # Use warm start for S if provided, otherwise use s_flow
    res.S .= isnothing(initial_S) ? copy(s_flow) : copy(initial_S)
    
    denom = 1.0 - β * (1.0 - δ)
    ΔS = Inf


    λ_S = λ_S_init # Initial step size for S updates (maybe updated in the loop #TODO)
    λ_u = λ_u_init # Initial step size for u updates (maybe updated in the loop #TODO)

    for k in 1:max_iter
        S_old = copy(res.S)
        u_old = copy(res.u)

        # --- Part 1: Update aggregates and surplus ---
        L = sum(u_old)
        u_dist = u_old ./ L

        B = vec(sum((1.0 - ξ) .* max.(0.0, S_old) .* u_dist, dims=1))
        
        B_integral = sum(B.^(1/κ₁) .* f_ψ)
        θ = ( (1/L) * (γ₀/κ₀)^(1/κ₁) * B_integral )^(1 / (1 + γ₁/κ₁))
        
        p = γ₀ * θ^(1 - γ₁)
        q = γ₀ * θ^(-γ₁)

        v = ((q .* B) ./ κ₀).^(1/κ₁)
        V = sum(v .* f_ψ)
        Γ = V > 0 ? (v .* f_ψ) ./ V : zeros(T, n_ψ)

        ExpectedSearch = max.(0.0, S_old) * Γ
        S_update = (s_flow .- (β * p * ξ .* ExpectedSearch)) ./ denom
        res.S .= (1.0 - λ_S) .* S_old .+ λ_S .* S_update

        # --- Part 2: Update unemployment ---
        ProbAccept = (res.S .> 0.0) * Γ
        unemp_rate = δ ./ (δ .+ p .* ProbAccept)
        u_update = unemp_rate .* f_h
        res.u .= (1.0 - λ_u) .* u_old .+ λ_u .* u_update

        # --- Part 3: Check for Convergence ---
        ΔS = maximum(abs.(res.S .- S_old))
        if verbose && (k % print_freq == 0 || k == 1); @printf("Iter %4d: ΔS = %.6e\n", k, ΔS); end
        if ΔS < tol; if verbose; println("Converged after $k iterations."); end; break; end
    end

    # --- CHANGE: Compute all final outcomes AFTER convergence ---
    compute_final_outcomes!(prim, res)
end

#?=========================================================================================
#? Helper Functions
#?=========================================================================================

# --- CHANGE: New function to calculate expected flow surplus ---
function calculate_expected_flow_surplus(prim::Primitives{T}) where {T<:Real}
    @unpack n_h, n_ψ, h_grid, ψ_grid, b, k = prim
    s_flow = Matrix{T}(undef, n_h, n_ψ)
    E = expectation(prim.z_dist)

    Threads.@threads for i_h in 1:n_h
        h = h_grid[i_h]
        for i_ψ in 1:n_ψ
            ψ = ψ_grid[i_ψ]
            
            joint_value_func(z) = begin
                α = optimal_alpha_given_z(prim, h, ψ, z)
                Y = (prim.A₀ + prim.A₁*h) * ((1 - α) + α * (prim.ψ₀ * exp(prim.ν * ψ + prim.ϕ * h)))
                C = prim.c₀ * z * (1 - α)^(prim.χ + 1) / (prim.χ + 1)
                return Y - C
            end
            
            s_flow[i_h, i_ψ] = E(joint_value_func) - (b * h)
        end
    end
    return s_flow
end

# --- CHANGE: New function to compute all final outcomes post-convergence ---
function compute_final_outcomes!(prim::Primitives{T}, res::Results{T}) where {T<:Real}
    @unpack β, δ, ξ, b, n_h, n_ψ, h_grid = prim
    
    # Recalculate final aggregates based on converged S and u
    L = sum(res.u)
    u_dist = res.u ./ L
    B = vec(sum((1.0 - ξ) .* max.(0.0, res.S) .* u_dist, dims=1))
    B_integral = sum(B.^(1/prim.κ₁) .* prim.ψ_pdf)
    res.θ = ( (1/L) * (prim.γ₀/prim.κ₀)^(1/prim.κ₁) * B_integral )^(1 / (1 + prim.γ₁/prim.κ₁))
    res.p = prim.γ₀ * res.θ^(1 - prim.γ₁)
    res.q = prim.γ₀ * res.θ^(-prim.γ₁)
    res.v = ((res.q .* B) ./ prim.κ₀).^(1/prim.κ₁)
    V = sum(res.v .* prim.ψ_pdf)
    Γ = V > 0 ? (res.v .* prim.ψ_pdf) ./ V : zeros(T, n_ψ)
    
    # Calculate final U(h) and n(h,ψ)
    exp_val_S = max.(0.0, res.S) * Γ
    res.U .= (b .* h_grid .+ β * res.p * ξ .* exp_val_S) ./ (1 - β)
    res.n .= (res.u * Γ') .* (res.S .> 0.0) ./ δ
end

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
