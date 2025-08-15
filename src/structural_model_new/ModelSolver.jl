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
using ForwardDiff 
"""
    solve_model(prim::Primitives, res::Results; 
                tol::Float64=1e-7, max_iter::Int=5000, 
                verbose::Bool=true, print_freq::Int=50, 
                λ_S::Float64=0.01, λ_u::Float64=0.01)

Solves the random search model by computing the equilibrium surplus matrix `S` and 
unemployment distribution `u` using a fixed-point iteration algorithm.

# Arguments
- `prim::Primitives`: Model parameters and grids for human capital and firm productivity
- `res::Results`: Results object to be updated with equilibrium values
- `tol::Float64=1e-7`: Convergence tolerance for the surplus matrix
- `max_iter::Int=5000`: Maximum number of iterations for the fixed-point algorithm
- `verbose::Bool=true`: Whether to print convergence information
- `print_freq::Int=50`: Frequency of convergence progress messages
- `λ_S::Float64=0.1`: Damping factor for surplus updates (smaller values = more damping)
- `λ_u::Float64=0.1`: Damping factor for unemployment updates (smaller values = more damping)

# Algorithm
The solution proceeds in three phases:
1. **Setup**: Initializes distributions and calculates flow surplus
2. **Fixed-point iteration**:
    - Computes market aggregates (tightness θ, job-finding rate p, vacancy-filling rate q)
    - Calculates the endogenous vacancy distribution
    - Updates the surplus matrix and unemployment distribution
    - Checks for convergence
3. **Post-processing**: Finalizes all equilibrium objects and stores them in the results object

# Note
The function modifies the `res` object in-place, updating its fields with the
equilibrium values for surplus, market tightness, job-finding rate, vacancy-filling rate,
vacancies, unemployment, and other derived quantities.
"""
# function solve_model(   
#                         prim::Primitives,   # Model parameters and grids
#                         res::Results;       # Initial results object to be updated
#                         tol::Float64=1e-7,  # convergence tolerance for outer loop on θ
#                         max_iter::Int=5000,  # maximum number of iterations for outer loop
#                         verbose::Bool=true, # print convergence progress
#                         λ_S::Float64 = 0.01, # damping factor for surplus S
#                         λ_u::Float64 = 0.01  # damping factor for unemployment u
#                     )


#     #* Phase 0: Setup & pre-calculations
#     @unpack h_grid, ψ_grid, β, δ, ξ, κ₀, κ₁, n_h, n_ψ, γ₀, γ₁ = prim
#     f_h = prim.h_pdf
#     f_ψ = prim.ψ_pdf
#     Γ = f_ψ ./ sum(f_ψ)                                     # firm type distribution Γ(ψ)
#     # static α* and flow surplus s(h,ψ)
#     s_flow = calculate_flow_surplus(prim, res)
#     # Initialize S and u
#     S = copy(s_flow)                                       # S(h,ψ) = s(h,ψ)
#     u = copy(f_h)                                          # u₀(h)=N(h)
    
#     denom = 1.0 - β * (1.0 - δ) # denominator in the Bellman equation for surplus S(h,ψ)
    
#     ΔS = Inf # Initialize change in surplus for convergence check
#     Δu_prev = Inf
#     ΔS_prev = Inf



#     #* Phase 1: main fixed‐point on S and u
#     for k in 1:max_iter
#         S_old = copy(S)
#         u_old = copy(u)
        
#         #? --- Step 1: Calculate market aggregates based on S_old and u_old ---
#         L = sum(u)
#         u_dist = u ./ L
        
#         #> (a) Firm's expected benefit B(ψ) per filled vacancy
#         # B(ψ) = ∫(1-ξ)S(h,ψ)⁺ * (u(h)/L) dF_h(h)
#         B = [ sum((1.0-ξ) * max.(0.0, S_old[:, j]) .* u_dist) for j in 1:n_ψ ]
        
#         #> (b) Market tightness θ
#         # θ = [ (1/L) * ∫(B(ψ)/κ₀)^(1/κ₁) dF_ψ(ψ) ]^(κ₁/(κ₁+γ))
#         Integral = sum((B ./ κ₀).^(1/κ₁) .* f_ψ) 
#         θ = ((1/L) * Integral)^(κ₁ / (κ₁ + γ₁))
        
#         #> (c) Aggregate job finding (p) and vacancy filling (q) probabilities
#         p = θ^(1 - γ₁)
#         q = θ^(-γ₁)
        
#         #? --- Step 2: Calculate the ENDOGENOUS vacancy distribution Γ ---
        
#         #> (d) Vacancies per firm v(ψ) and aggregate vacancies V
#         # v(ψ) = (q*B(ψ)/κ₀)^(1/κ₁)
#         v = ( (q .* B) ./ κ₀ ).^(1/κ₁)
#         # V = ∫v(ψ)dF_ψ(ψ)
#         V = sum(v .* f_ψ) 

#         #> (e) Distribution of vacancies met by workers, Γ(ψ)
#         # Γ(ψ) = v(ψ)f_ψ(ψ) / V
#         Γ_new = V > 0 ? (v .* f_ψ) ./ V : zeros(n_ψ)

#         #? --- Step 3: Update S and u using the NEW endogenous Γ ---
        
#         #> (f) Update Surplus Matrix S
#         ExpectedSearch = sum(max.(0.0, S_old) .* Γ_new', dims=2)
#         S_update = @. (s_flow - (β * p * ξ * ExpectedSearch)) / denom 
#         S .= (1.0 - λ_S) .* S_old .+ λ_S .* S_update
#         #> (g) Update Unemployment Vector u
#         # --- JACOBI-STYLE UPDATE ---
#         # Calculate ProbAccept using S_old to be consistent with p and Γ_new
#         ProbAccept = sum((S_old .> 0.0) .* Γ_new', dims=2) # <-- USE S_old HERE
#         # ---------------------------
#         unemp_rate = @. δ / (δ + p * ProbAccept)
#         u_update = unemp_rate .* f_h 
#         u .= (1.0 - λ_u) .* u_old .+ λ_u .* u_update

#         #> (h) Check for convergence
#         # ΔS = maximum(abs.(S .- S_old))
#         ΔS = maximum(abs.(ForwardDiff.value.(S) - ForwardDiff.value.(S_old)))
#         # Δu = maximum(abs.(u .- u_old))
#         Δu = maximum(abs.(ForwardDiff.value.(u) - ForwardDiff.value.(u_old)))

#         # Initialize baseline residuals on first iteration
#         if k == 1
#             ΔS0 = ΔS
#             Δu0 = Δu
#         end

#         ΔS_prev = ΔS
#         Δu_prev = Δu

#         if verbose && k % 1 == 0
#             println(@bold @cyan "Iteration $k: ΔS=$ΔS")
#         end

#         if ΔS < tol

#             if verbose
#                 println(@bold @green "Converged after $k iters (ΔS=$ΔS).")
#             end
#             break
#         end
#     end # for k in 1:max_iter


#     if ΔS >= tol
#         @warn "Model failed to converge after $max_iter iterations. Last change: ΔS=$ΔS"
#     end

#     #* Phase 2: Post‐processing – assemble Results
    
#     # --- ONE FINAL UPDATE FOR FULL CONSISTENCY ---
#     # Recalculate all aggregate variables using the final converged S and u
#     # to ensure perfect synchronization before calculating final policies.
    
#     # (a) Final L and B
#     L_final = sum(u)
#     u_dist_final = L_final > 0 ? u ./ L_final : zeros(n_h)
#     B_final = [ sum((1.0-ξ) * max.(0.0, S[:, j]) .* u_dist_final) for j in 1:n_ψ ]
    
#     # (b) Final θ
#     Integral_final = sum((B_final ./ κ₀).^(1/κ₁) .* f_ψ)
#     θ_final = ((1/L_final) * Integral_final)^(κ₁ / (κ₁ + γ₁))
    
#     # (c) Final p and q
#     p_final = θ_final^(1 - γ₁)
#     q_final = θ_final^(-γ₁)

#     # (d) Final v
#     v_final = ( (q_final .* B_final) ./ κ₀ ).^(1/κ₁)
#     # -----------------------------------------

#     # Store the fully consistent, final values in the results object
#     res.S .= S
#     res.θ  = θ_final
#     res.p  = p_final
#     res.q  = q_final
#     res.v .= v_final
#     res.u .= u
    
#     # Calculate final wage, unemployment value, and employment distribution
#     calculate_final_policies!(res, prim)
            
# end

function solve_model(
                        prim::Primitives,
                        res::Results;
                        tol::Float64=1e-7,
                        max_iter::Int=5000,
                        verbose::Bool=true,
                        λ_S::Float64 = 0.01,
                        λ_u::Float64 = 0.01
                    )

    @unpack h_grid, ψ_grid, β, δ, ξ, κ₀, κ₁, n_h, n_ψ, γ₀, γ₁ = prim
    f_h = prim.h_pdf
    f_ψ = prim.ψ_pdf
    Γ = f_ψ ./ sum(f_ψ)
    s_flow = calculate_flow_surplus(prim, res)

    # --- AD FIX: Initialize arrays with the parameter type ---
    # Get the numeric type from a parameter that will be differentiated.
    # If AD is running, T will be a Dual type. If not, it will be Float64.
    T = eltype(prim.c₀)

    # Initialize S and u as arrays of type T.
    S = T.(s_flow)
    u = T.(f_h)
    # --- END FIX ---

    denom = 1.0 - β * (1.0 - δ)
    ΔS = Inf

    for k in 1:max_iter
        S_old = copy(S)
        u_old = copy(u)

        L = sum(u)
        u_dist = u ./ L

        # Use ForwardDiff.value for all comparisons and control flow
        B = [ sum((1.0-ξ) * max.(0.0, ForwardDiff.value.(S_old[:, j])) .* u_dist) for j in 1:n_ψ ]
        Integral = sum((B ./ κ₀).^(1/κ₁) .* f_ψ)
        θ = ((1/L) * Integral)^(κ₁ / (κ₁ + γ₁))
        p = θ^(1 - γ₁)
        q = θ^(-γ₁)
        v = ( (q .* B) ./ κ₀ ).^(1/κ₁)
        V = sum(v .* f_ψ)
        Γ_new = ForwardDiff.value(V) > 0 ? (v .* f_ψ) ./ V : zeros(n_ψ)

        ExpectedSearch = sum(max.(0.0, ForwardDiff.value.(S_old)) .* Γ_new', dims=2)
        S_update = @. (s_flow - (β * p * ξ * ExpectedSearch)) / denom
        S .= (1.0 - λ_S) .* S_old .+ λ_S .* S_update

        ProbAccept = sum((ForwardDiff.value.(S_old) .> 0.0) .* Γ_new', dims=2)
        unemp_rate = @. δ / (δ + p * ProbAccept)
        u_update = unemp_rate .* f_h
        u .= (1.0 - λ_u) .* u_old .+ λ_u .* u_update

        ΔS = maximum(abs.(ForwardDiff.value.(S) .- ForwardDiff.value.(S_old)))

        if ΔS < tol
            if verbose; println(@bold @green "Converged after $k iters (ΔS=$ΔS)."); end
            break
        end
    end

    if ΔS >= tol
        @warn "Model failed to converge after $max_iter iterations. Last change: ΔS=$ΔS"
    end

    # Post-processing
    L_final = sum(u)
    u_dist_final = L_final > 0 ? u ./ L_final : zeros(n_h)
    B_final = [ sum((1.0-ξ) * max.(0.0, ForwardDiff.value.(S[:, j])) .* u_dist_final) for j in 1:n_ψ ]
    Integral_final = sum((B_final ./ κ₀).^(1/κ₁) .* f_ψ)
    θ_final = ((1/L_final) * Integral_final)^(κ₁ / (κ₁ + γ₁))
    p_final = θ_final^(1 - γ₁)
    q_final = θ_final^(-γ₁)
    v_final = ( (q_final .* B_final) ./ κ₀ ).^(1/κ₁)

    res.S .= S
    res.θ  = θ_final
    res.p  = p_final
    res.q  = q_final
    res.v .= v_final
    res.u .= u

    calculate_final_policies!(res, prim)
end
#==========================================================================================
#? Core Inner Loop and Update Functions
==========================================================================================#

"""
    solve_inner_loop(res::Results, prim::Primitives, s_flow::Matrix{Float64}; tol::Float64=1e-8, max_iter::Int=2000, verbose::Bool=true)

Solves for the partial equilibrium objects (`S`, `v`, `u`) for a *fixed* set of
aggregate probabilities `p` and `q`.

This function iterates until the surplus matrix `S` converges. In each step, it
sequentially updates the surplus `S`, the vacancy distribution `v`, and the
unemployment distribution `u`, as these objects are mutually dependent. This is a
Gauss-Seidel-like iterative method.

# Arguments
- `res::Results`: The results object to be updated in-place.
- `prim::Primitives`: The model primitives.
- `s_flow::Matrix{Float64}`: The pre-calculated flow surplus matrix `s(h, ψ)`.
- `tol::Float64`: Convergence tolerance for the surplus matrix `S`.
- `max_iter::Int`: Maximum number of iterations for this inner loop.
- `verbose::Bool`: If true, prints convergence progress for the inner loop.

# Returns
- `nothing`: The `res` object is modified in-place.
"""
function solve_inner_loop(res::Results, prim::Primitives, s_flow::Matrix{Float64}; tol::Float64=1e-8, max_iter::Int=2000, verbose::Bool=true)
    
    # Loop until the surplus matrix S converges or max_iter is reached.
    for i in 1:max_iter
        # 1. Store the old surplus matrix to check for convergence later.
        #    `copy()` is essential to create a separate object, not just a reference.
        S_old = copy(res.S)

        # 2. Perform the sequential updates using the most recent information available.
        #    This order allows changes to propagate through the system within one iteration.
        update_surplus!(res, prim, s_flow)
        update_vacancies!(res, prim)
        update_unemployment!(res, prim)

        # 3. Check for convergence by calculating the maximum absolute difference
        #    between the new and old surplus matrices.
        diff = maximum(abs.(res.S - S_old))

        # Optional: Print progress at specified intervals to avoid flooding the console.
        if verbose && (i % 100 == 0 || i == 1)
            @printf("    Inner Iter: %4d | Surplus Diff: %.4e\n", i, diff)
        end

        if diff < tol
            if verbose
                @printf("    Inner loop converged after %d iterations.\n", i)
            end
            # If converged, exit the function successfully.
            return
        end
    end

    # If the loop completes without converging, issue a warning.
    @warn "Inner loop failed to converge after $max_iter iterations."
end

"""
    update_surplus!(res::Results, prim::Primitives, s_flow::Matrix{Float64})

    Updates the total match surplus matrix `S(h, ψ)` in-place based on the Bellman equation.

    This function implements the core recursive formula for the surplus:
    `S(h, ψ) = [s(h, ψ) - β * p * ξ * E[S(h,ψ')⁺]] / [1 - β(1-δ)]`

    For each worker type `h`, it first calculates the expected value of continuing to search,
    `E[S(h,ψ')⁺]`, which depends on the current distribution of vacancies `v(ψ)`. It then
    uses this value to update the surplus for all possible matches `(h, ψ)`.

    # Arguments
    - `res::Results`: The results object. `res.S` and `res.v` are read, and `res.S` is updated.
    - `prim::Primitives`: The model primitives, including `β`, `δ`, and `ξ`.
    - `s_flow::Matrix{Float64}`: The pre-calculated flow surplus matrix.

    # Returns
    - `nothing`: The `res.S` matrix is modified in-place.
"""
function update_surplus!(res::Results, prim::Primitives, s_flow::Matrix{Float64})
    # Unpack parameters and grid sizes for readability
    @unpack β, δ, ξ, n_h, n_ψ = prim
    @unpack p = res # p is the aggregate job finding rate, fixed for this inner loop
    
    # The denominator of the surplus equation is constant for all (h, ψ)
    denominator = 1.0 - β * (1.0 - δ)

    # Calculate the total number of vacancies V and the probability distribution
    # of meeting a firm of type ψ, denoted Γ(ψ).
    V = sum(res.v)
    Gamma = (V > 0.0) ? res.v ./ V : zeros(n_ψ)

    # We need a new matrix to store the results to avoid using partially
    # updated values from the current iteration within the same loop.
    S_new = similar(res.S)

    # Loop over each worker type h
    for i_h in 1:n_h
        # Get the surplus row for this worker from the *previous* iteration
        S_h_old = @view res.S[i_h, :]

        # Calculate the positive part of the surplus, S⁺(h, ψ)
        S_h_positive = max.(0.0, S_h_old)

        # Calculate the worker's expected surplus from finding a new job.
        # This is the integral/sum part of the formula: E[S(h,ψ')⁺] = ∫ S(h,ψ')⁺ dΓ(ψ')
        expected_search_value = sum(S_h_positive .* Gamma)

        # Calculate the numerator of the update rule. This is a vector operation
        # that computes the numerator for all ψ simultaneously for a given h.
        numerator = s_flow[i_h, :] .- (β * p * ξ * expected_search_value)

        # Update the entire row for worker h in the new surplus matrix
        S_new[i_h, :] = numerator / denominator
    end

    # After computing all new values, update the results object in-place.
    res.S .= S_new
end

"""
    update_vacancies!(res::Results, prim::Primitives)

Updates the vacancy posting vector `v(ψ)` in-place based on the free-entry condition.

For each firm type `ψ`, this function calculates the expected profit from posting one
vacancy. This depends on the probability of filling the vacancy (`q`) and the expected
surplus the firm will get, which is an average over the distribution of unemployed
workers `u(h)`.

The free-entry condition states that firms post vacancies until the marginal cost equals
the expected marginal benefit: `c'(v) = q * E_h[(1-ξ)S(h,ψ)⁺]`.
With a linear cost `c(v) = κ*v`, the condition becomes `κ = q * E_h[...]`. If the
benefit exceeds the cost, firms want to post infinite vacancies. To ensure stability,
we assume a simple convex cost `c(v) = κ₀*v + 0.5*κ₁*v²`, which gives a marginal cost
`c'(v) = κ₀ + κ₁*v`. This leads to a well-defined number of vacancies:
`v(ψ) = (q * E_h[...] - κ₀) / κ₁`.

# Arguments
- `res::Results`: The results object. `res.S` and `res.u` are read, and `res.v` is updated.
- `prim::Primitives`: The model primitives, including `κ₀`, `κ₁`, and `ξ`.

# Returns
- `nothing`: The `res.v` vector is modified in-place.
"""
function update_vacancies!(res::Results, prim::Primitives)
    # Unpack parameters and results for readability
    # Assuming κ₀ and κ₁ are in prim for a convex cost structure.
    # If only linear cost κ is used, this logic needs a small adjustment.
    @unpack κ₀, κ₁, ξ, n_h, n_ψ = prim
    @unpack q, S, u = res
    
    # Calculate the total mass of unemployed workers L and the probability
    # distribution of meeting a worker of type h, u_dist(h).
    L = sum(u)
    u_dist = (L > 0.0) ? u ./ L : zeros(n_h)

    # Loop over each firm type ψ
    for i_ψ in 1:n_ψ
        # Get the surplus column for this firm type
        S_ψ = @view S[:, i_ψ]

        # Calculate the positive part of the surplus, S⁺(h, ψ)
        S_ψ_positive = max.(0.0, S_ψ)

        # Calculate the firm's expected share of surplus from a new match.
        # This is the integral/sum: E_h[(1-ξ)S(h,ψ)⁺] = ∫ (1-ξ)S(h,ψ)⁺ u_dist(h) dh
        expected_firm_surplus = sum((1.0 - ξ) .* S_ψ_positive .* u_dist)

        # Calculate the expected marginal benefit of posting a vacancy
        marginal_benefit = q * expected_firm_surplus

        # Determine the number of vacancies to post using the inverted marginal cost.
        # From c'(v) = marginal_benefit => κ₀ + κ₁*v = marginal_benefit
        # => v = (marginal_benefit - κ₀) / κ₁
        # The number of vacancies cannot be negative.
        
        # Check if the firm can cover the fixed part of the marginal cost
        if marginal_benefit > κ₀
            num_vacancies = (marginal_benefit - κ₀) / κ₁
        else
            num_vacancies = 0.0
        end

        # Update the vacancy vector for this firm type
        res.v[i_ψ] = num_vacancies
    end
end

"""
    update_unemployment!(res::Results, prim::Primitives)

Updates the unemployment vector `u(h)` in-place based on steady-state flow conditions.

This function calculates the steady-state mass of unemployed workers for each skill
type `h`. It uses the standard "flows in = flows out" condition for unemployment:
`δ * E(h) = p * u(h) * ProbAccept(h)`.
Combined with the population identity `N(h) = E(h) + u(h)`, this yields the
unemployment rate for type `h`: `unemp_rate(h) = δ / (δ + p * ProbAccept(h))`.

The `ProbAccept(h)` term is the likelihood that a randomly met firm will result
in a positive-surplus match, and it depends on the current vacancy distribution `v(ψ)`.

# Arguments
- `res::Results`: The results object. `res.S` and `res.v` are read, and `res.u` is updated.
- `prim::Primitives`: The model primitives, including `δ` and `h_grid.pdf`.

# Returns
- `nothing`: The `res.u` vector is modified in-place.
"""
function update_unemployment!(res::Results, prim::Primitives)
    # Unpack parameters and results for readability
    @unpack δ, n_h, n_ψ, h_pdf = prim
    @unpack p, S, v = res
    

    # Calculate the total number of vacancies V and the probability distribution
    # of meeting a firm of type ψ, Γ(ψ).
    V = sum(v)
    Gamma = (V > 0.0) ? v ./ V : zeros(n_ψ)

    # Loop over each worker type h
    for i_h in 1:n_h
        # Get the surplus row for this worker type
        S_h = @view S[i_h, :]

        # Create an indicator for whether a match has positive surplus
        is_match_acceptable = S_h .> 0.0

        # Calculate the probability that a meeting results in an acceptable job offer.
        # This is the integral/sum: ProbAccept(h) = ∫ 1_{S(h,ψ)>0} dΓ(ψ)
        prob_accept = sum(is_match_acceptable .* Gamma)

        # Calculate the steady-state unemployment rate for type h.
        # The denominator is the total outflow rate from unemployment.
        outflow_rate = p * prob_accept
        unemp_rate = δ / (δ + outflow_rate)

        # Calculate the mass of unemployed workers.
        # N(h) is the total population of type h, which we take from the
        # exogenous skill distribution PDF.
        total_population_h = h_pdf[i_h]
        res.u[i_h] = unemp_rate * total_population_h
    end
end

#==========================================================================================#
#? Helper and Post-Processing Functions
#==========================================================================================#
"""
    calculate_flow_surplus(prim::Primitives, res::Results) -> Matrix{Float64}

Pre-calculates the flow surplus `s(h, ψ)` for all `(h, ψ)` pairs.

The flow surplus is the per-period joint value of a match net of the worker's
outside option (unemployment benefit). The joint value is the sum of the firm's
profit (`Y - w`) and the worker's utility (`u(w, α)`). With a quasi-linear utility
function `u(w, α) = w + non_wage_utility(α)`, the wage `w` cancels out, and the
joint value becomes `Y(α) + non_wage_utility(α)`.

The function therefore calculates:
`s(h, ψ) = [Y(α*) + non_wage_utility(α*)] - b(h)`

This value only depends on primitives and the optimal `α` policy, so it can be
computed once at the beginning of the solution process.

# Arguments
- `prim::Primitives`: The model primitives.
- `res::Results`: The results object containing the `α_policy`.

# Returns
- `Matrix{Float64}`: A matrix of size `(n_h, n_ψ)` containing the flow surplus values.
"""
function calculate_flow_surplus(prim::Primitives, res::Results)::Matrix{Float64}
    # Unpack grid sizes for clarity
    @unpack n_h, n_ψ, h_grid, ψ_grid, utility_fun, production_fun = prim

    # 1. Initialize an empty matrix `s_flow` of size (n_h, n_ψ).
    #    Note: The convention used here is that the first dimension is for worker
    #    type `h` and the second is for firm type `ψ`.
    s_flow = Matrix{Float64}(undef, n_h, n_ψ)

    # 2. Loop over all `(i_h, i_ψ)` pairs on the grid.
    for (i_h, h) in enumerate(h_grid)
        for (i_ψ, ψ) in enumerate(ψ_grid)

            # b. Get the optimal remote work share α* for this (h, ψ) pair.
            #    This was pre-calculated during the initialization of the Results struct.
            #    We assume α_policy is indexed as (h, ψ).
            α_star = res.α_policy[i_h, i_ψ]

            # c. Calculate total output Y for the optimal α*.
            Y = production_fun(h, ψ, α_star)

            # d. Calculate the non-wage component of the worker's utility.
            #    For a quasi-linear utility u(w,α) = w + f(α), this is f(α).
            #    We can calculate this by evaluating the utility function at a wage of zero,
            #    assuming the utility function is of the form u(w,...) = w + ...
            #    This is more general than hardcoding the functional form.
            non_wage_utility = utility_fun(0.0, α_star)

            # e. The joint flow value of the match is the sum of output and non-wage utility.
            joint_flow_value = Y + non_wage_utility

            # f. The flow surplus is the joint value minus the worker's outside option,
            #    the unemployment benefit `b * h`. (Assuming b(h) = prim.b * h) #! Hardcoded for now
            s = joint_flow_value - prim.b * h

            # g. Store the calculated flow surplus in the matrix.
            s_flow[i_h, i_ψ] = s
        end
    end

    # 3. Return the fully populated `s_flow` matrix.
    return s_flow
end

"""
    calculate_final_policies!(res::Results, prim::Primitives)

Calculates the final equilibrium objects after the model has been solved.

This function is called once after the main fixed-point iteration converges. It uses
the final equilibrium values of `S`, `u`, `v`, and `p` to compute:
1. The worker's value of unemployment, `U(h)`.
2. The equilibrium wage policy, `w(h, ψ)`.
3. The steady-state distribution of employed workers, `n(h, ψ)`.

# Arguments
- `res::Results`: The converged results object to be populated.
- `prim::Primitives`: The model primitives.

# Returns
- `nothing`: The `res` object is modified in-place with the final policies (`U`, `w_policy`, `n`).
"""
function calculate_final_policies!(res::Results, prim::Primitives)
    # Unpack parameters and converged results
    @unpack β, δ, ξ, b, n_h, n_ψ, h_grid, utility_fun = prim
    @unpack p, S, v, u, α_policy = res

    f_ψ = prim.ψ_pdf

    # 1. Calculate the value of unemployment U(h) for all h.
    # ---------------------------------------------------------
    # First, calculate the expected search value for each worker type, which we need for U(h).
    V = sum(v .* f_ψ)
    Gamma = (V > 0.0) ?  (v .* f_ψ) ./ V : zeros(n_ψ)

    for i_h in 1:n_h
        S_h_positive = max.(0.0, @view S[i_h, :])
        expected_search_value = sum(S_h_positive .* Gamma)
        
        # From the Bellman equation for an unemployed worker:
        # U(h) = b(h) + β * [ (1-p)U(h) + p * (U(h) + ξ*E[S(h,ψ')⁺]) ]
        # Solving for U(h) gives:
        # U(h) = (b(h) + β * p * ξ * E[S(h,ψ')⁺]) / (1 - β)
        numerator_U = b + β * p * ξ * expected_search_value # Assuming b(h) is constant prim.b
        denominator_U = 1.0 - β
        res.U[i_h] = numerator_U / denominator_U
    end

    # 2. Calculate the wage policy w(h, ψ) for all (h, ψ).
    # ---------------------------------------------------------
    # From the Nash Bargaining condition W = U + ξS and the Bellman equation for W:
    # W = u(w,α) + β[(1-δ)W + δU]
    # u(w,α) = (1-β(1-δ)) * (U + ξS) - βδU
    # We need to invert the utility function to find w.
    for (i_h, h_val) in enumerate(h_grid)
        for i_ψ in 1:n_ψ
            α_star = α_policy[i_h, i_ψ]
            
            # Calculate the required flow utility for the worker
            required_flow_utility = (1.0 - β * (1.0 - δ)) * (res.U[i_h] + ξ * res.S[i_h, i_ψ]) - (β * δ * res.U[i_h])
            
            # Invert the utility function to find the wage that delivers this utility
            # This uses the `find_implied_wage` function defined in `model_functions.jl`.
            # This function solves `u(w, α) = required_flow_utility` for `w`. 
            # For quasi linear utility this is u(w, α) = w - c(α) =  required_flow_utility => w = required_flow_utility  + c(α) 
            # Notice that -u(0, α) = c(α) => c(α) = required_flow_utility - u(0, α)
            res.w_policy[i_h, i_ψ] = required_flow_utility - utility_fun(0.0, α_star)
        end
    end

    # 3. Calculate the distribution of employed workers n(h, ψ).
    # ---------------------------------------------------------
    # From the steady-state flow condition: δ * n(h, ψ) = p * u(h) * Γ(ψ) * 1_{S>0}
    for i_h in 1:n_h
        for i_ψ in 1:n_ψ
            if S[i_h, i_ψ] > 0.0
                # Flow from unemployment into this specific job match
                inflow = p * u[i_h] * Gamma[i_ψ]
                # The stock of employed workers is the inflow rate divided by the outflow rate (δ)
                res.n[i_h, i_ψ] = inflow / δ
            else
                # No matches are formed if the surplus is not positive
                res.n[i_h, i_ψ] = 0.0
            end
        end
    end
end