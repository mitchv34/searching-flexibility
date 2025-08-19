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
    λ_S_final, λ_u_final = λ_S, λ_u
    
    return λ_S_final, λ_u_final
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