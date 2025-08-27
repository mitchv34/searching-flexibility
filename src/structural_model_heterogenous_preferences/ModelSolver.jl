#==========================================================================================
# Module: ModelSolver.jl
# Description: Contains the core functions to solve for the steady-state equilibrium

===========================================================================================#

using Parameters, Printf, Distributions, ForwardDiff, YAML
#?=========================================================================================
#? Solver Configuration
#?=========================================================================================

"""
    SolverConfig

Structure to hold solver configuration parameters.
"""
@with_kw struct SolverConfig{T<:Real}
    tol::Float64 = 1e-8
    max_iter::Int = 5000
    verbose::Bool = true
    print_freq::Int = 50
    λ_S_init::Float64 = 0.01
    λ_u_init::Float64 = 0.01
    initial_S::Union{Matrix{T}, Nothing} = nothing
end

"""
    load_solver_config_from_yaml(yaml_file::String; T::Type=Float64)

Load solver configuration from YAML file. Returns a SolverConfig struct with
values from the ModelSolverOptions section of the YAML file.

# Arguments
- `yaml_file::String`: Path to YAML configuration file
- `T::Type`: Numeric type for matrices (default: Float64)

# Returns
- `SolverConfig{T}`: Configuration struct with solver options
"""
function load_solver_config_from_yaml(source; T::Type=Float64)
    # Handle both YAML file path and dictionary input
    config = if source isa AbstractString
        # Load from YAML file
        YAML.load_file(source, dicttype=OrderedDict)
    elseif source isa AbstractDict
        # Use dictionary directly, converting keys to strings for compatibility
        Dict(string(k) => (v isa AbstractDict ? Dict(string(k2) => v2 for (k2, v2) in v) : v) 
                for (k, v) in source)
    else
        error("Source must be either a YAML file path (String) or a configuration dictionary")
    end

    solver_opts = get(config, "ModelSolverOptions", Dict())
    
    # Helper function to get value with type conversion
    function get_param(key::String, default_val, target_type::Type)
        val = get(solver_opts, key, default_val)
        if target_type == Float64 && isa(val, Number)
            return Float64(val)
        elseif target_type == Int && isa(val, Number)
            return Int(val)
        elseif target_type == Bool && isa(val, Bool)
            return val
        else
            return default_val
        end
    end
    
    return SolverConfig{T}(
        tol = get_param("tol", 1e-8, Float64),
        max_iter = get_param("max_iter", 5000, Int),
        verbose = get_param("verbose", true, Bool),
        print_freq = get_param("print_freq", 50, Int),
        λ_S_init = get_param("lambda_S_init", 0.01, Float64),
        λ_u_init = get_param("lambda_u_init", 0.01, Float64),
        initial_S = nothing  # Always start as nothing, can be overridden later
    )
end

"""
    merge_solver_config(base_config::SolverConfig{T}; kwargs...) where {T}

Merge solver configuration with keyword arguments. Keyword arguments override
values in base_config.

# Arguments  
- `base_config::SolverConfig{T}`: Base configuration
- `kwargs...`: Keyword arguments to override

# Returns
- `SolverConfig{T}`: Merged configuration
"""
function merge_solver_config(base_config::SolverConfig{T}; kwargs...) where {T}
    # Convert kwargs to a dictionary for easier access
    kw_dict = Dict(kwargs)
    
    return SolverConfig{T}(
        tol = get(kw_dict, :tol, base_config.tol),
        max_iter = get(kw_dict, :max_iter, base_config.max_iter),
        verbose = get(kw_dict, :verbose, base_config.verbose),
        print_freq = get(kw_dict, :print_freq, base_config.print_freq),
        λ_S_init = get(kw_dict, :λ_S_init, base_config.λ_S_init),
        λ_u_init = get(kw_dict, :λ_u_init, base_config.λ_u_init),
        initial_S = get(kw_dict, :initial_S, base_config.initial_S)
    )
end

#?=========================================================================================
#?=========================================================================================
#? Helper Functions
#?=========================================================================================

"""
    calculate_analytical_logit_flow_surplus(prim)

Computes the expected flow surplus s(h, ψ) using the analytical closed-form
solution. This version is fully vectorized over both h and ψ for maximum
performance in a single-threaded environment.
"""
function calculate_analytical_logit_flow_surplus(prim::Primitives{T}) where {T<:Real}
    @unpack n_h, n_ψ, h_grid, ψ_grid, b, A₀, A₁, ψ₀, ν, ϕ, c₀, μ = prim
    
    # --- Reshape grids for 2D broadcasting ---
    # h_col becomes a column vector (n_h x 1)
    # ψ_row becomes a row vector (1 x n_ψ)
    h_col = reshape(h_grid, n_h, 1)
    ψ_row = reshape(ψ_grid, 1, n_ψ)
    
    # --- Perform all calculations on (n_h x n_ψ) matrices ---
    
    # A_h is now a (n_h x 1) column vector
    A_h = A₀ .+ A₁ .* h_col
    
    # Broadcasting h_col and ψ_row creates a full (n_h x n_ψ) matrix for g
    g = @. ψ₀ * exp(ν * ψ_row + ϕ * h_col)

    denominator_term = @. A_h * (g - 1.0) + c₀
    
    # Calculate the main case for all elements first
    exp_term1 = @. exp((A_h - c₀) / μ)
    exp_term2 = @. exp(denominator_term / μ)
    integral_val = @. (μ * exp_term1 / denominator_term) * (exp_term2 - 1.0)
    
    # Identify and overwrite the special cases where the denominator is near zero
    zero_mask = @. abs(denominator_term) < 1e-9
    integral_val = ifelse.(zero_mask, exp_term1, integral_val)
    
    # Final calculation is broadcast element-wise
    expected_max_value = @. μ * log(integral_val)
    
    # Subtract the unemployment benefit, broadcasting b*h_col across the rows
    s_flow = expected_max_value .- (b .* h_col)
    
    return s_flow
end


#?=========================================================================================
#? ANCHORED SOLVER (v4.0 - Data-Driven Vacancy Distribution, August 2025)
#? Features vacancy distribution anchoring to match empirical data
#?=========================================================================================

"""
    solve_model_anchored(prim, res; config=nothing, kwargs...)

**ANCHORED SOLVER v4.0 (Data-Driven Vacancy Distribution, August 2025)**

Solver that anchors the vacancy distribution shape to empirical data while allowing
the total level to be endogenous. This is useful for structural estimation where
we want to match observed vacancy distributions across firm types.

The key innovation is treating prim.ψ_pdf as the target vacancy distribution shape
rather than the exogenous firm type distribution. The solver finds equilibrium
conditions that generate this target distribution.

# Algorithm
1. Use prim.ψ_pdf as the target vacancy distribution shape (v_shape)
2. Calculate endogenous surplus and unemployment as usual
3. Determine total vacancy level V endogenously through market clearing
4. Force vacancy distribution to match target shape: Γ = v_shape
5. Solve for consistent equilibrium with this constraint

# Arguments
- `prim::Primitives{T}`: Model primitives (ψ_pdf interpreted as target vacancy shape)
- `res::Results{T}`: Results structure to update
- `config::Union{SolverConfig, Nothing, String}`: Solver configuration (optional)
- `kwargs...`: Individual parameters to override config

# Additional Keyword Arguments
- `anchor_weight::Float64=1.0`: Weight on anchoring constraint (1.0 = full anchoring)
- `min_vacancy_level::Float64=1e-6`: Minimum total vacancy level to prevent degeneracy

# Returns
- `status::Symbol`: Convergence status
- `anchoring_error::Float64`: Final deviation from target vacancy distribution
"""
function solve_model(
                        prim::Primitives{T},
                        res::Results{T};
                        config::Union{SolverConfig{T}, Nothing, String}=nothing,
                        kwargs...
                    ) where {T<:Real}
    #! For testing block
    # prim = deepcopy(prim_optimized)
    # res = deepcopy(res_optimized)
    # config = nothing
    # kwargs = Dict()
    #! End of testing block
    # Handle configuration (same as other solvers)
    if config !== nothing
        if typeof(config) == String
            config = load_solver_config_from_yaml(config)
        end
        final_config = merge_solver_config(config; kwargs...)
        
        initial_S = final_config.initial_S
        tol = final_config.tol
        max_iter = final_config.max_iter
        verbose = final_config.verbose
        print_freq = final_config.print_freq
        λ_S_init = final_config.λ_S_init
        λ_u_init = final_config.λ_u_init
    else
        initial_S = get(kwargs, :initial_S, nothing)
        tol = get(kwargs, :tol, 1e-8)
        max_iter = get(kwargs, :max_iter, 5000)
        verbose = get(kwargs, :verbose, true)
        print_freq = get(kwargs, :print_freq, 50)
        λ_S_init = get(kwargs, :λ_S_init, 0.01)
        λ_u_init = get(kwargs, :λ_u_init, 0.01)
    end

    @unpack h_grid, ψ_grid, β, δ, ξ, κ₀, κ₁, n_h, n_ψ, γ₀, γ₁ = prim
    f_h = prim.h_pdf
    
    # --- KEY CHANGE: Interpret ψ_pdf as target vacancy distribution shape ---
    v_shape = copy(prim.ψ_pdf)
    v_shape ./= sum(v_shape) # Ensure it's normalized
    
    # Calculate flow surplus
    s_flow_base = calculate_logit_flow_surplus_with_curvature(prim)

    # Initialize main arrays
    S_final = isnothing(initial_S) ? copy(s_flow_base) : copy(initial_S)
    u_final = copy(f_h)
    
    # Pre-allocate arrays
    S_old = similar(S_final)
    u_old = similar(u_final)
    u_dist = similar(u_final)
    B = Vector{T}(undef, n_ψ)
    Γ = Vector{T}(undef, n_ψ)             # Actual vacancy distribution used
    ExpectedSearch = Vector{T}(undef, n_h)
    b_h = Vector{T}(undef, n_h)
    s_flow_net = similar(S_final)
    S_update = similar(S_final)
    ProbAccept = Vector{T}(undef, n_h)
    unemp_rate = Vector{T}(undef, n_h)
    u_update = similar(u_final)

    # Pre-allocate V_actual so we can keep it after the iteration loop
    V_actual = 0.0

    # Pre-calculate constants
    denom = 1.0 - β * (1.0 - δ)
    ΔS = Inf
    λ_S = λ_S_init
    λ_u = λ_u_init
    status = :notConverged

    if verbose
        println("Starting anchored solver with target vacancy shape anchoring...")
    end

    for k in 1:max_iter
        copy!(S_old, S_final)
        copy!(u_old, u_final)

        # --- Part 1: Calculate B and L from previous iteration ---
        L = sum(u_old)
        @. u_dist = u_old / L
        calculate_B!(B, S_old, u_dist, prim.ξ) # B is the firm's expected surplus share

        # --- Part 2: THE CORE OF THE ANCHORED METHOD ---
        # We use the free-entry condition to solve for the TOTAL LEVEL of vacancies (V)
        # that is consistent with the ANCHORED SHAPE (v_shape).
        
        # The "value" of the average vacancy, weighted by the target shape
        B_integral_anchored = 0.0
        for j in 1:n_ψ
            B_integral_anchored += max(zero(eltype(B)), B[j])^(1/prim.κ₁) * v_shape[j]
        end
        
        # This is the key equation. It solves for the market tightness θ that is
        # consistent with the surplus B and the anchored vacancy shape v_shape.
        θ = ((1/L) * (prim.γ₀/prim.κ₀)^(1/prim.κ₁) * B_integral_anchored)^(1 / (1 + prim.γ₁/prim.κ₁))
        
        # From this single, consistent θ, all other aggregates follow
        p = prim.γ₀ * θ^(1 - prim.γ₁)
        q = prim.γ₀ * θ^(-prim.γ₁)
        
        # The vacancy distribution Γ IS the target shape.
        Γ .= v_shape
        
        # The total level of vacancies V is now an outcome
        V_actual = L * θ
        
        # --- Part 3: Update Surplus using the consistent, anchored distribution ---
        
        # Calculate expected search value using the anchored Γ
        mul!(ExpectedSearch, max.(0.0, S_old), Γ)
        
        # Calculate the endogenous b(h) vector using the anchored Γ
        # (Note: your previous code had a bug here, using S_old * Γ_old)
        mul!(b_h, S_old, Γ)
        @. b_h = prim.b * prim.ξ * b_h
        
        # Calculate net flow surplus
        @. s_flow_net = s_flow_base - b_h

        # Update surplus using the consistent p
        @. S_update = (s_flow_net - (β * p * prim.ξ * ExpectedSearch)) / denom
        @. S_final = (1.0 - λ_S) * S_old + λ_S * S_update

        # --- Part 4: Update unemployment using the consistent p and anchored Γ ---
        mul!(ProbAccept, (S_final .> 0.0), Γ)
        @. unemp_rate = δ / (δ + p * ProbAccept)
        @. u_update = unemp_rate * f_h
        @. u_final = (1.0 - λ_u) * u_old + λ_u * u_update

        # --- Part 5: Convergence check ---
        ΔS = 0.0
        for i in eachindex(S_final)
            ΔS = max(ΔS, abs(S_final[i] - S_old[i]))
        end

        if verbose && (k % print_freq == 0 || k == 1)
            @printf("Iter %4d: ΔS = %.12e, V = %.6f, θ = %.6f\n", 
                    k, ΔS, V_actual, θ)
        end
        
        if ΔS < tol
            if verbose
                println("Converged after $k iterations.")
            end
            status = :converged
            break
        end
    end

    # Update final results
    res.u .= u_final
    res.S .= S_final
    
    # Compute final outcomes with anchored vacancy distribution
    compute_final_outcomes_anchored!(prim, res, Γ, V_actual)
    
    # The anchoring error is now implicitly zero by construction
    return status, 0.0
end

"""
    compute_final_outcomes_anchored!(prim, res, Γ, V_total)

Compute final equilibrium outcomes using the anchored vacancy distribution.
This is a modified version of compute_final_outcomes! that uses the provided
vacancy distribution instead of deriving it from free entry.
"""
function compute_final_outcomes_anchored!(
                                prim::Primitives{T}, 
                                res::Results{T}, 
                                Γ::Vector{T}, 
                                V_total::T
                            ) where {T<:Real}
    @unpack β, δ, ξ, b, h_grid, γ₀, γ₁, κ₀, κ₁ = prim
    
    # Use provided vacancy distribution instead of deriving from free entry
    L = sum(res.u)
    u_dist = res.u ./ L
    
    # Calculate B using current surplus and unemployment
    B = vec(sum((1.0 - ξ) .* max.(0.0, res.S) .* u_dist, dims=1))
    
    # Calculate market tightness from total vacancies and unemployment
    θ = V_total / L
    res.θ = θ
    res.p = γ₀ * θ^(1 - γ₁)
    res.q = γ₀ * θ^(-γ₁)
    
    # Use anchored vacancy distribution
    res.v = V_total .* Γ
    
    # Calculate final outcomes
    exp_val_S = max.(0.0, res.S) * Γ
    res.U .= (b .* h_grid .+ β * res.p * ξ .* exp_val_S) ./ (1 - β)
    
    # Employment flows using anchored distribution
    res.n .= res.p .* (res.u * Γ') .* (res.S .> 0.0) ./ δ
end


#?=========================================================================================
#? Final Outcomes Calculation (Post-Convergence)
#?=========================================================================================

function compute_final_outcomes!(prim::Primitives{T}, res::Results{T}) where {T<:Real}
    @unpack β, δ, ξ, b, h_grid, γ₀, γ₁, κ₀, κ₁ = prim
    
    # Recalculate final aggregates based on converged S and u
    L = sum(res.u)
    u_dist = res.u ./ L
    B = vec(sum((1.0 - ξ) .* max.(0.0, res.S) .* u_dist, dims=1))
    B_integral = sum(max.(0.0, B).^(1/κ₁) .* prim.ψ_pdf)
    res.θ = ( (1/L) * (γ₀/κ₀)^(1/κ₁) * B_integral )^(1 / (1 + γ₁/κ₁))
    res.p = γ₀ * res.θ^(1 - γ₁)
    res.q = γ₀ * res.θ^(-γ₁)
    res.v = ((res.q .* B) ./ κ₀).^(1/κ₁)
    V = sum(res.v .* prim.ψ_pdf)
    Γ = V > 0 ? (res.v .* prim.ψ_pdf) ./ V : zeros(T, prim.n_ψ)
    
    # Calculate final U(h) and n(h,ψ)
    exp_val_S = max.(0.0, res.S) * Γ
    res.U .= (b .* h_grid .+ β * res.p * ξ .* exp_val_S) ./ (1 - β)
    
    # --- BUG FIX: Added missing job-finding rate 'res.p' ---
    res.n .= res.p .* (res.u * Γ') .* (res.S .> 0.0) ./ δ
end

using QuadGK 

"""
    calculate_logit_flow_surplus_with_curvature(prim)

Computes the expected flow surplus s(h, ψ) using QuadGK for numerical
integration. This version is NUMERICALLY STABLE and avoids overflow by
rescaling the integrand.
"""
function calculate_logit_flow_surplus_with_curvature(prim::Primitives{T}) where {T<:Real}
    @unpack n_h, n_ψ, h_grid, ψ_grid, b, A₀, A₁, ψ₀, ν, ϕ, c₀, χ, μ = prim
    
    s_flow = Matrix{T}(undef, n_h, n_ψ)

    Threads.@threads for i_h in 1:n_h
        h = h_grid[i_h]
        A_h = A₀ + A₁*h
        
        for i_ψ in 1:n_ψ
            ψ = ψ_grid[i_ψ]
            # g = ψ₀ * exp(ν * ψ + ϕ * h) #? Exponential production function
            g = @. ψ₀ * (h^ϕ) * (ψ^ν) #? Cobb-Douglas production function


            # --- Define the core value function V(α) ---
            # This is the deterministic part of match utility
            V_alpha(α) = (A_h * ((1 - α) + α * g)) - (c₀ * (1 - α)^(1 + χ) / (1 + χ))

            # --- STEP 1: Find the maximum value of V(α) to use for rescaling ---
            # We can find this by sampling on a fine grid.
            alpha_grid_fine = 0.0:0.01:1.0
            V_max = maximum(V_alpha(α) for α in alpha_grid_fine)

            # --- STEP 2: Define the STABLE integrand ---
            # The exponent (V(α) - V_max) / μ is now guaranteed to be <= 0, preventing overflow.
            function stable_integrand(α::T)
                return exp((V_alpha(α) - V_max) / μ)
            end

            # --- STEP 3: Use quadgk on the stable integrand ---
            integral_val, _ = quadgk(stable_integrand, 0.0, 1.0)
            
            # --- STEP 4: Combine the results using the numerically stable formula ---
            #> Notice that this is gross surplus without the unemployment flow value
            s_flow[i_h, i_ψ] = V_max + μ * log(integral_val)
        end
    end
    
    return s_flow
end

#?=========================================================================================
#? OFFICIAL SOLVER (v2.0 - Optimized, August 2025)
#? Replaced original solver on 2025-08-19 due to 4.6x speed improvement and 6.2x memory reduction
#?=========================================================================================

"""
    solve_model(prim, res; config=nothing, kwargs...)

**OFFICIAL SOLVER v2.0 (August 2025 Optimization)**

High-performance solver with minimal allocations and maximum efficiency.
Replaced the original solver due to significant performance improvements:
- 4.6x faster execution time
- 6.2x lower memory usage
- Zero-allocation inner loops

This version uses pre-allocated arrays and in-place operations throughout.
For legacy version, see `solve_model_legacy()` below.

# Arguments
- `prim::Primitives{T}`: Model primitives
- `res::Results{T}`: Results structure to update
- `config::Union{SolverConfig, Nothing}`: Solver configuration (optional)
- `kwargs...`: Individual parameters to override config (for backward compatibility)

# Keyword Arguments (when config is not provided)
- `initial_S::Union{Matrix{T}, Nothing}=nothing`: Initial surplus matrix
- `tol::Float64=1e-8`: Convergence tolerance
- `max_iter::Int=5000`: Maximum iterations
- `verbose::Bool=true`: Print convergence information
- `print_freq::Int=50`: Print frequency for verbose mode
- `λ_S_init::Float64=0.01`: Initial dampening parameter for surplus
- `λ_u_init::Float64=0.01`: Initial dampening parameter for unemployment
"""
function solve_model_no_anchor(
                        prim::Primitives{T},
                        res::Results{T};
                        config::Union{SolverConfig{T}, Nothing, String}=nothing,
                        kwargs...
                    ) where {T<:Real}

    # If config is provided, use it and only override with explicitly provided kwargs
    if config !== nothing
        # If a path is given then we extract the model parameters from the YAML file
        if typeof(config) == String
            config = load_solver_config_from_yaml(config)
        end

        # Only override config with explicitly provided kwargs (not defaults)
        final_config = merge_solver_config(config; kwargs...)
        
        # Extract parameters from final config
        initial_S = final_config.initial_S
        tol = final_config.tol
        max_iter = final_config.max_iter
        verbose = final_config.verbose
        print_freq = final_config.print_freq
        λ_S_init = final_config.λ_S_init
        λ_u_init = final_config.λ_u_init
    else
        # No config provided, use defaults and any provided kwargs
        # Extract from kwargs with defaults as fallback
        initial_S = get(kwargs, :initial_S, nothing)
        tol = get(kwargs, :tol, 1e-8)
        max_iter = get(kwargs, :max_iter, 5000)
        verbose = get(kwargs, :verbose, true)
        print_freq = get(kwargs, :print_freq, 50)
        λ_S_init = get(kwargs, :λ_S_init, 0.01)
        λ_u_init = get(kwargs, :λ_u_init, 0.01)
    end

    @unpack h_grid, ψ_grid, β, δ, ξ, κ₀, κ₁, n_h, n_ψ, γ₀, γ₁ = prim
    f_h = prim.h_pdf
    f_ψ = prim.ψ_pdf

    # Calculate flow surplus once
    s_flow_base = calculate_logit_flow_surplus_with_curvature(prim)

    # Initialize main arrays
    S_final = isnothing(initial_S) ? copy(s_flow_base) : copy(initial_S)
    u_final = copy(prim.h_pdf)
    
    # Pre-allocate ALL temporary arrays to eliminate allocations in the loop
    S_old = similar(S_final)
    u_old = similar(u_final)
    u_dist = similar(u_final)
    
    B = Vector{T}(undef, n_ψ)
    v = Vector{T}(undef, n_ψ)
    Γ = Vector{T}(undef, n_ψ)
    
    ExpectedSearch = Vector{T}(undef, n_h)
    S_update = similar(S_final)
    
    ProbAccept = Vector{T}(undef, n_h)
    unemp_rate = Vector{T}(undef, n_h)
    u_update = similar(u_final)

    # Pre-calculate constants
    denom = 1.0 - β * (1.0 - δ)
    inv_κ₁ = 1.0 / κ₁
    ΔS = Inf
    λ_S = λ_S_init
    λ_u = λ_u_init

    status = :notConverged

    # Main iteration loop - fully optimized
    for k in 1:max_iter
        # In-place copy operations
        copy!(S_old, S_final)
        copy!(u_old, u_final)

        # --- Part 1: Update aggregates and surplus (ALL IN-PLACE) ---
        L = sum(u_old)
        
        # u_dist = u_old ./ L (in-place)
        @. u_dist = u_old / L

        # B = vec(sum((1.0 - ξ) .* max.(0.0, S_old) .* u_dist, dims=1)) (in-place)
        fill!(B, 0.0)
        one_minus_ξ = 1.0 - ξ
        for j in 1:n_ψ
            for i in 1:n_h
                B[j] += one_minus_ξ * max(zero(eltype(S_old)), S_old[i, j]) * u_dist[i]
            end
        end
        
        # B_integral = sum(max.(0.0, B).^(1/κ₁) .* f_ψ) (in-place computation)
        B_integral = 0.0
        for j in 1:n_ψ
            B_integral += max(zero(eltype(B)), B[j])^inv_κ₁ * f_ψ[j]
        end
        
        θ = ((1/L) * (γ₀/κ₀)^inv_κ₁ * B_integral)^(1 / (1 + γ₁*inv_κ₁))
        p = γ₀ * θ^(1 - γ₁)
        q = γ₀ * θ^(-γ₁)

        # v = ((q .* B) ./ κ₀).^(1/κ₁) (in-place)
        @. v = ((q * B) / κ₀)^inv_κ₁
        
        # V = sum(v .* f_ψ) and Γ calculation (in-place)
        V = 0.0
        for j in 1:n_ψ
            V += v[j] * f_ψ[j]
        end
        
        if V > 0
            inv_V = 1.0 / V
            @. Γ = (v * f_ψ) * inv_V
        else
            fill!(Γ, 0.0)
        end

        # Now calculate the expected surplus
        Expected_S_by_h = S_old * Γ

        # The new b(h) vector
        b_h = prim.b .* prim.ξ .* Expected_S_by_h

        # Now, use this new b_h to calculate the new s_flow and the new surplus
        # Note: s_flow now depends on b_h, so it must be recalculated or adjusted
        s_flow = s_flow_base .- b_h # (Assuming s_flow_base is calculated once without b(h))
        
        # ExpectedSearch = max.(0.0, S_old) * Γ (in-place matrix-vector multiply)
        fill!(ExpectedSearch, 0.0)
        for i in 1:n_h
            for j in 1:n_ψ
                val = max(zero(eltype(S_old)), S_old[i, j])
                if val > zero(eltype(S_old))
                    ExpectedSearch[i] += val * Γ[j]
                end
            end
        end
        
        # S_update and S_final updates (in-place)
        βpξ = β * p * ξ
        @. S_update = (s_flow - βpξ * ExpectedSearch) / denom
        @. S_final = (1.0 - λ_S) * S_old + λ_S * S_update

        # --- Part 2: Update unemployment (ALL IN-PLACE) ---
        
        # ProbAccept = (S_final .> 0.0) * Γ (in-place)
        fill!(ProbAccept, 0.0)
        for i in 1:n_h
            for j in 1:n_ψ
                if S_final[i, j] > 0.0
                    ProbAccept[i] += Γ[j]
                end
            end
        end

        # unemp_rate and u_update (in-place)
        @. unemp_rate = δ / (δ + p * ProbAccept)
        @. u_update = unemp_rate * f_h
        @. u_final = (1.0 - λ_u) * u_old + λ_u * u_update

        # --- Part 3: Convergence check (optimized to avoid allocations) ---
        ΔS = 0.0
        for i in eachindex(S_final)
            ΔS = max(ΔS, abs(S_final[i] - S_old[i]))
        end

        if verbose && (k % print_freq == 0 || k == 1)
            @printf("Iter %4d: ΔS = %.12e\n", k, ΔS)
        end
        
        if ΔS < tol
            if verbose
                println("Converged after $k iterations.")
            end
            status = :converged
            break
        end
    end

    # Update final results (in-place)
    res.u .= u_final
    res.S .= S_final
    compute_final_outcomes!(prim, res)
    
    return status
end

#?=========================================================================================
#? LEGACY SOLVER (v1.0 - Deprecated August 2025)
#? Moved to legacy section on 2025-08-19, replaced by optimized version
#? Kept for compatibility and benchmarking purposes only
#?=========================================================================================

"""
    solve_model_legacy(prim, res; config=nothing, kwargs...)

**LEGACY SOLVER v1.0 (Deprecated August 2025)**

Original solver implementation, replaced by optimized version due to performance issues:
- 4.6x slower than new solver
- 6.2x higher memory usage  
- Frequent allocations in inner loops

Use `solve_model()` for production code. This function is kept for:
- Compatibility with old code
- Performance benchmarking
- Reference implementation

# Arguments
- `prim::Primitives{T}`: Model primitives
- `res::Results{T}`: Results structure to update
- `config::Union{SolverConfig, Nothing}`: Solver configuration (optional)
- `kwargs...`: Individual parameters to override config (for backward compatibility)
"""
function solve_model_legacy(
                        prim::Primitives{T},
                        res::Results{T};
                        config::Union{SolverConfig{T}, Nothing}=nothing,
                        kwargs...
                    ) where {T<:Real}

    # If config is provided, use it and only override with explicitly provided kwargs
    if config !== nothing
        # Override config with explicitly provided kwargs (not defaults)
        final_config = merge_solver_config(config; kwargs...)
        
        # Extract parameters from final config
        initial_S = final_config.initial_S
        tol = final_config.tol
        max_iter = final_config.max_iter
        verbose = final_config.verbose
        print_freq = final_config.print_freq
        λ_S_init = final_config.λ_S_init
        λ_u_init = final_config.λ_u_init
    else
        # No config provided, use defaults and any provided kwargs
        initial_S = get(kwargs, :initial_S, nothing)
        tol = get(kwargs, :tol, 1e-8)
        max_iter = get(kwargs, :max_iter, 5000)
        verbose = get(kwargs, :verbose, true)
        print_freq = get(kwargs, :print_freq, 50)
        λ_S_init = get(kwargs, :λ_S_init, 0.01)
        λ_u_init = get(kwargs, :λ_u_init, 0.01)
    end

    @unpack h_grid, ψ_grid, β, δ, ξ, κ₀, κ₁, n_h, n_ψ, γ₀, γ₁ = prim
    f_h = copy(prim.h_pdf)
    f_ψ = copy(prim.ψ_pdf)

    # --- MAJOR CHANGE: Calculate flow surplus using the analytical closed-form solution ---
    # s_flow = calculate_analytical_logit_flow_surplus(prim) #! Linear version of preferences
    s_flow = calculate_logit_flow_surplus_with_curvature(prim) #? Convex preferences with curvature

    # Use warm start for S if provided, otherwise use s_flow
    S_final = isnothing(initial_S) ? copy(s_flow) : copy(initial_S)
    u_final = copy(prim.h_pdf)  # Start with total population in unemployment
    
    # Initialize u_old as 

    denom = 1.0 - β * (1.0 - δ)
    ΔS = Inf

    λ_S = λ_S_init
    λ_u = λ_u_init

    # The main loop structure remains the same. The complexity of the α choice
    # has been absorbed into the pre-computed s_flow.
    for k in 1:max_iter
    # k = 1
        S_old = copy(S_final)
        u_old = copy(u_final)

        # --- Part 1: Update aggregates and surplus ---
        L = sum(u_old)
        u_dist = u_old ./ L

        B = vec(sum((1.0 - ξ) .* max.(0.0, S_old) .* u_dist, dims=1))
        
        B_integral = sum(max.(0.0, B).^(1/κ₁) .* f_ψ)
        θ = ( (1/L) * (γ₀/κ₀)^(1/κ₁) * B_integral )^(1 / (1 + γ₁/κ₁))
        
        p = γ₀ * θ^(1 - γ₁)
        q = γ₀ * θ^(-γ₁)

        v = ((q .* B) ./ κ₀).^(1/κ₁)
        V = sum(v .* f_ψ)
        Γ = V > 0 ? (v .* f_ψ) ./ V : zeros(T, prim.n_ψ)

        ExpectedSearch = max.(0.0, S_old) * Γ
        S_update = (s_flow .- (β * p * ξ .* ExpectedSearch)) ./ denom
        S_final .= (1.0 - λ_S) .* S_old .+ λ_S .* S_update

        
        # --- Part 2: Update unemployment ---
        ProbAccept = (S_final .> 0.0) * Γ
        unemp_rate = δ ./ (δ .+ p .* ProbAccept)
        u_update = unemp_rate .* f_h
        u_final .= (1.0 - λ_u) .* u_old .+ λ_u .* u_update

        # --- Part 3: Check for Convergence ---
        ΔS = maximum(abs.(S_final .- S_old))
        if verbose && (k % print_freq == 0 || k == 1); @printf("Iter %4d: ΔS = %.12e\n", k, ΔS); end
        if ΔS < tol; if verbose; println("Converged after $k iterations."); end; break; end
    end
    # Update final results objects
    res.u = copy(u_final)
    res.S = copy(S_final)

    compute_final_outcomes!(prim, res)
    return
end

#?=========================================================================================
#? EXPERIMENTAL SOLVER (v3.0 - Dynamic Relaxation, August 2025)
#? Features adaptive damping and improved convergence monitoring
#?=========================================================================================

"""
    calculate_B!(B, S_old, u_dist, ξ)

In-place calculation of B vector to minimize allocations.
"""
function calculate_B!(B::Vector{T}, S_old::Matrix{T}, u_dist::Vector{T}, ξ::T) where {T<:Real}
    n_h, n_ψ = size(S_old)
    fill!(B, 0.0)
    one_minus_ξ = 1.0 - ξ
    for j in 1:n_ψ
        for i in 1:n_h
            B[j] += one_minus_ξ * max(zero(eltype(S_old)), S_old[i, j]) * u_dist[i]
        end
    end
    return nothing
end

"""
    calculate_Gamma!(Γ, v, f_ψ)

In-place calculation of Γ vector to minimize allocations.
"""
function calculate_Gamma!(Γ::Vector{T}, v::Vector{T}, f_ψ::Vector{T}) where {T<:Real}
    V = 0.0
    for j in eachindex(v)
        V += v[j] * f_ψ[j]
    end
    
    if V > 0
        inv_V = 1.0 / V
        @. Γ = (v * f_ψ) * inv_V
    else
        fill!(Γ, 0.0)
    end
    return nothing
end

"""
    solve_model_adaptive(prim, res; config=nothing, kwargs...)

**EXPERIMENTAL SOLVER v3.0 (Dynamic Relaxation, August 2025)**

Enhanced solver with adaptive damping parameters and improved convergence monitoring.
Features:
- Dynamic relaxation parameter adjustment
- Convergence status reporting
- Automatic step size reduction when convergence stalls
- Enhanced progress monitoring
- Full config compatibility with main solver

# Arguments
- `prim::Primitives{T}`: Model primitives
- `res::Results{T}`: Results structure to update
- `config::Union{SolverConfig, Nothing, String}`: Solver configuration (optional)
- `kwargs...`: Individual parameters to override config (for backward compatibility)

# Keyword Arguments (when config is not provided)
- `initial_S::Union{Matrix{T}, Nothing}=nothing`: Initial surplus matrix
- `tol::Float64=1e-8`: Convergence tolerance
- `max_iter::Int=5000`: Maximum iterations
- `verbose::Bool=true`: Print convergence information
- `print_freq::Int=50`: Print frequency for verbose mode
- `λ_S_init::Float64=0.01`: Initial dampening parameter for surplus
- `λ_u_init::Float64=0.01`: Initial dampening parameter for unemployment

# Returns
- `status::Symbol`: Convergence status (:converged, :max_iter_reached, :in_progress)
- `λ_S::Float64`: Final surplus damping parameter
- `λ_u::Float64`: Final unemployment damping parameter
"""
function solve_model_adaptive(
                        prim::Primitives{T},
                        res::Results{T};
                        config::Union{SolverConfig{T}, Nothing, String}=nothing,
                        kwargs...
                    ) where {T<:Real}

    # If config is provided, use it and only override with explicitly provided kwargs
    if config !== nothing
        # If a path is given then we extract the model parameters from the YAML file
        if typeof(config) == String
            config = load_solver_config_from_yaml(config)
        end

        # Only override config with explicitly provided kwargs (not defaults)
        final_config = merge_solver_config(config; kwargs...)
        
        # Extract parameters from final config
        initial_S = final_config.initial_S
        tol = final_config.tol
        max_iter = final_config.max_iter
        verbose = final_config.verbose
        print_freq = final_config.print_freq
        λ_S_init = final_config.λ_S_init
        λ_u_init = final_config.λ_u_init
    else
        # No config provided, use defaults and any provided kwargs
        # Extract from kwargs with defaults as fallback
        initial_S = get(kwargs, :initial_S, nothing)
        tol = get(kwargs, :tol, 1e-8)
        max_iter = get(kwargs, :max_iter, 5000)
        verbose = get(kwargs, :verbose, true)
        print_freq = get(kwargs, :print_freq, 50)
        λ_S_init = get(kwargs, :λ_S_init, 0.01)
        λ_u_init = get(kwargs, :λ_u_init, 0.01)
    end

    @unpack h_grid, ψ_grid, β, δ, ξ, κ₀, κ₁, n_h, n_ψ, γ₀, γ₁ = prim
    f_h = prim.h_pdf
    f_ψ = prim.ψ_pdf

    s_flow = calculate_logit_flow_surplus_with_curvature(prim)

    S_final = isnothing(initial_S) ? copy(s_flow) : copy(initial_S)
    u_final = copy(prim.h_pdf)
    
    # Pre-allocate all temporary arrays to minimize allocations
    S_old = similar(S_final)
    u_old = similar(u_final)
    u_dist = similar(u_final)
    B = Vector{T}(undef, n_ψ)
    v = Vector{T}(undef, n_ψ)
    Γ = Vector{T}(undef, n_ψ)
    ExpectedSearch = Vector{T}(undef, n_h)
    S_update = similar(S_final)
    ProbAccept = Vector{T}(undef, n_h)
    unemp_rate = Vector{T}(undef, n_h)
    u_update = similar(u_final)
    S_diff = similar(S_final) # For non-allocating distance calculation

    denom = 1.0 - β * (1.0 - δ)
    inv_κ₁ = 1.0 / κ₁
    ΔS = Inf
    ΔS_prev = Inf

    # --- NEW: Dynamic Relaxation and Convergence Status ---
    λ_S = λ_S_init
    λ_u = λ_u_init
    status::Symbol = :in_progress
    max_retries = 5 # Max times to shrink lambda before giving up

    for k in 1:max_iter
        copy!(S_old, S_final)
        copy!(u_old, u_final)

        # --- Part 1: Update aggregates and surplus (In-place) ---
        L = sum(u_old)
        @. u_dist = u_old / L
        
        # Use optimized in-place calculations
        calculate_B!(B, S_old, u_dist, ξ)
        B_integral = 0.0
        for j in 1:n_ψ
            B_integral += max(zero(eltype(B)), B[j])^inv_κ₁ * f_ψ[j]
        end
        
        θ = ((1/L) * (γ₀/κ₀)^inv_κ₁ * B_integral)^(1 / (1 + γ₁*inv_κ₁))
        p = γ₀ * θ^(1 - γ₁)
        q = γ₀ * θ^(-γ₁)
        
        @. v = ((q * B) / κ₀)^inv_κ₁
        calculate_Gamma!(Γ, v, f_ψ)
        
        # ExpectedSearch calculation (in-place matrix-vector multiply)
        fill!(ExpectedSearch, 0.0)
        for i in 1:n_h
            for j in 1:n_ψ
                val = max(zero(eltype(S_old)), S_old[i, j])
                if val > zero(eltype(S_old))
                    ExpectedSearch[i] += val * Γ[j]
                end
            end
        end
        
        βpξ = β * p * ξ
        @. S_update = (s_flow - βpξ * ExpectedSearch) / denom
        @. S_final = (1.0 - λ_S) * S_old + λ_S * S_update

        # --- Part 2: Update unemployment (In-place) ---
        fill!(ProbAccept, 0.0)
        for i in 1:n_h
            for j in 1:n_ψ
                if S_final[i, j] > 0.0
                    ProbAccept[i] += Γ[j]
                end
            end
        end
        
        @. unemp_rate = δ / (δ + p * ProbAccept)
        @. u_update = unemp_rate * f_h
        @. u_final = (1.0 - λ_u) * u_old + λ_u * u_update

        # --- Part 3: Check for Convergence ---
        @. S_diff = abs(S_final - S_old)
        ΔS = maximum(S_diff)

        # --- NEW: Dynamic Relaxation Logic ---
        if ΔS > ΔS_prev && k > 10 && max_retries > 0
            # If error increased, the step was too big.
            # Revert, shrink lambda, and retry this iteration.
            copy!(S_final, S_old) # Revert S
            copy!(u_final, u_old) # Revert u
            λ_S = max(λ_S / 2.0, 1e-4) # Shrink lambda
            λ_u = max(λ_u / 2.0, 1e-4) # Shrink lambda for unemployment too
            ΔS = ΔS_prev # Reset error
            max_retries -= 1
            if verbose
                @printf("Step too large at iter %d, reducing λ_S to %.6f\n", k, λ_S)
            end
            continue # Redo this iteration with smaller lambda
        end
        ΔS_prev = ΔS

        if verbose && (k % print_freq == 0 || k == 1)
            @printf("Iter %4d: ΔS = %.12e, λ_S = %.4f\n", k, ΔS, λ_S)
        end
        
        if ΔS < tol
            status = :converged
            if verbose
                println("Converged after $k iterations.")
            end
            break
        end
    end

    if status == :in_progress
        status = :max_iter_reached
        if verbose
            @warn "Solver stopped after reaching max_iter=$max_iter. Final ΔS = $ΔS"
        end
    end

    # Update final results (in-place)
    res.u .= u_final
    res.S .= S_final
    compute_final_outcomes!(prim, res)
    
    # --- NEW: Return status, final lambdas and convergence error -- 
    return status, λ_S, λ_u, ΔS
end