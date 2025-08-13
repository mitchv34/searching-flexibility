#==========================================================================================
Title: model_functions.jl
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2025-02-26
Description: Abstract functions for the search model:
    - ProductionFunction
    - UtilityFunction
    - MatchingFunction
    - WageFunction
==========================================================================================#
module ModelFunctions
    # Load required packages
    using Parameters, Roots
    # Export functions and types 
    export AbstractModelFunction, ProductionFunction, UtilityFunction
    export MatchingFunction, WageFunction
    export evaluate, get_parameters, find_implied_wage, evaluate_derivative
    export invert_vacancy_fill, eval_prob_job_find, eval_prob_vacancy_fill, recover_tightness
    # Export factory functions for creating model functions
    export create_matching_function, create_production_function, create_utility_function
    #?=========================================================================================
    #? AbstractModelFunction: Base abstract type for all model functions
    #?=========================================================================================
    #==========================================================================================
    #* AbstractModelFunction:
        -  An abstract type that serves as a common interface for all model functions.
    ==========================================================================================#
    abstract type AbstractModelFunction end
    #==========================================================================================
    #* evaluate
        Evaluate the model function `f` with the given arguments `args...`.  
        #* Arguments
        - `f::AbstractModelFunction`: The model function to be evaluated.
        - `args...`: The arguments to be passed to the model functio.
        #* Throws
        - `ErrorException`: If the `evaluate` function is not implemented for the specific type
        of `f`.
    ==========================================================================================#    
    function evaluate(f::AbstractModelFunction, args...)
        error("evaluate not implemented for $(typeof(f))")
    end
    #==========================================================================================
    #* get_parameters
        Retrieve the parameters of the model function `f`.
        #* Arguments
        - `f::AbstractModelFunction`: The model function whose parameters are to be retrieved.
        #* Throws
        - `ErrorException`: If the `get_parameters` function is not implemented for the specific 
        type of `f`.
    ==========================================================================================#
    function get_parameters(f::AbstractModelFunction)
        error("get_parameters not implemented for $(typeof(f))")
    end
    #==========================================================================================
    #* validate_parameters
        Validate the parameters of the model function `f`.
        #* Arguments
        - `f::AbstractModelFunction`: The model function whose parameters are to be validated.
        #* Throws
        - `ErrorException`: If the `validate_parameters` function is not implemented for the 
        specific type of `f`.
    ==========================================================================================#
    function validate_parameters(f::AbstractModelFunction)
        error("validate_parameters not implemented for $(typeof(f))")
    end
    #?=========================================================================================
    #? ProductionFunction: Represents production technology
    #? - Production function that captures the relationship between worker skill, remote work,
    #?   and firm remote efficiency.
    #? - Production specific methods:
    #?   - None
    #? - Components:
    #?   - ProductivityComponent: Worker skill productivity A(h)
    #?   - RemoteEfficiencyComponent: Remote work efficiency for a firm-worker pair g(ψ, h)
    #?   - CompositeProduction: Y(h, α, ψ) = A(h) * (1 - α) + B(h) * g(α, ψ, h)
    #?=========================================================================================
    abstract type ProductionFunction <: AbstractModelFunction end
    #*=========================================================================================
    #* ProductionFunction Components:
    #*  - ProductivityComponent (Captures worker skill productivity)
    #*  - RemoteEfficiencyComponent (Captures remote work efficiency for a firm-worker pair)
    #*=========================================================================================
    abstract type ProductivityComponent <: AbstractModelFunction end
    abstract type RemoteEfficiencyComponent <: AbstractModelFunction end
    #==========================================================================================
    # ProductivityComponent.
    #  Available implementations:
    #  - LinearProductivity: A(h) = A₀ + A₁ * h
    ==========================================================================================#
    # > LinearProductivity
    @with_kw mutable struct LinearProductivity <: ProductivityComponent
        A₀::Float64              # Base productivity level
        A₁::Float64              # Skill intensity of productivity
        function LinearProductivity(A₀::Float64, A₁::Float64)
            prod_comp = new(A₀, A₁)
            validate_parameters(prod_comp)
            return prod_comp
        end
    end # end definition
    function validate_parameters(f::LinearProductivity)
        if f.A₀ < 0.0
            error("A₀ must be non-negative")
        end
        if f.A₁ < 0.0
            error("A₁ must be non-negative")
        end
    end # end validate_parameters
    function evaluate(f::LinearProductivity, h::Float64)
        return f.A₀ + f.A₁ * h
    end # end evaluate
    function evaluate_derivative(f::LinearProductivity, argument::Union{String, Symbol}, h::Float64)
        if argument == "h" || argument == :h
            return f.A₁     
        else
            error("Invalid argument: $argument. Valid options are \"h\".")
        end
    end
    #> List of available productivity components
    productivity_components = Dict(
        "LinearProductivity" => LinearProductivity
    )
    #==========================================================================================
    # RemoteEfficiencyComponent
    #  Available implementations:
    #  - LinearFirmLogWorker g(ψ, h, α) = α * ( ν * ψ - ψ₀ + ϕ * log(h) )
    #  - Interaction g(α, h, ψ) = (γ * h * ψ * α) / (1 + γ * h * ψ * α)
    #  - Exponential g(α, h, ψ) = k(h, ψ) * (1 - exp(-λ * α))
    #               where k(h, ψ) = 1 - exp(-γ * h * ψ)
    ==========================================================================================#
    # > LinearFirmLogWorker
    @with_kw mutable struct LinearFirmLogWorker <: RemoteEfficiencyComponent
        ν::Float64              # Remote efficiency scaling factor of firm remote efficiency
        ϕ::Float64              # Remote efficiency scaling factor of worker skill
        ψ₀::Float64             # Threshold for remote efficiency to be positive for production
        # Constructor
        function LinearFirmLogWorker(ν::Float64, ϕ::Float64, ψ₀::Float64)
            remote_eff_comp = new(ν, ϕ, ψ₀)
            # Validate parameters
            validate_parameters(remote_eff_comp)
            # Return the instance if parameters are valid
            return remote_eff_comp
        end
    end # end definition
    function validate_parameters(f::LinearFirmLogWorker)
        if f.ν < 0.0
            error("ν must be non-negative")
        end
        if f.ϕ < 0.0
            error("ϕ must be non-negative")
        end
        # if f.ψ₀ < 0.0
        #     error("ψ₀ must be non-negative")
        # end
    end
    function evaluate(f::LinearFirmLogWorker, ψ::Float64, h::Float64, α::Float64)
        # return ψ + f.ψ₀ + f.ϕ * log(h)
        return α * (f.ν * ψ - f.ψ₀ + f.ϕ * log(h) )
    end # end evaluate
    function evaluate_derivative(f::LinearFirmLogWorker, argument::Union{String, Symbol}, ψ::Float64, h::Float64, α::Float64)
        if argument == "α" || argument == :α
            return ψ * f.ν - f.ψ₀ + f.ϕ * log(h)
        elseif argument == "ψ" || argument == :ψ
            return α * f.ν
        elseif argument == "h" || argument == :h
            return α * f.ϕ / h
        else
            error("Invalid argument: $argument. Valid options are \"α\", \"ψ\", or \"h\".")
        end
    end
    # > Interaction
    @with_kw mutable struct Interaction <: RemoteEfficiencyComponent
        ϕ::Float64 = 1.0 # Interaction scaling parameter (phi > 0)
        B₁::Float64 = 1.0 # Skill decoupling parameter (B1 > 0)
        # Constructor with validation
        function Interaction(ϕ::Float64, B₁::Float64)
            if ϕ <= 0.0
                error("Parameter ϕ must be positive for Interaction")
            end
            new(ϕ, B₁)
        end
    end # end definition
    """
        evaluate(f::InteractionG, ψ::Float64, h::Float64, α::Float64)

    Calculates g(α, h, ψ) = (k * α) / (1 + k * α), where k = γ * h * ψ.
    """
    function evaluate(f::Interaction, ψ::Float64, h::Float64, α::Float64)
        # Input validation
        if !(0.0 <= α <= 1.0); error("α must be between 0 and 1."); end
        if h <= 0.0; error("h must be positive."); end
        if ψ < 0.0; error("ψ must be non-negative."); end # Assuming non-negative psi

        k_val = f.ϕ * h * ψ
        # Handle k_val = 0 case (e.g., if h or ψ is 0)
        if k_val == 0.0
            return 0.0
        end
        denominator = 1.0 + k_val * α
        # Denominator should always be >= 1 given constraints
        return f.B₁ * (k_val * α) / denominator
    end # end evaluate
    """
        evaluate_derivative(f::Interaction, argument::Union{String, Symbol}, ψ::Float64, h::Float64, α::Float64)

    Calculates the partial derivative of g(α, h, ψ) = (k * α) / (1 + k * α)
    with respect to the specified argument (α, h, or ψ), where k = ϕ * h * ψ.
    """
    function evaluate_derivative(f::Interaction, argument::Union{String, Symbol}, ψ::Float64, h::Float64, α::Float64)
        # Input validation
        if !(0.0 <= α <= 1.0); error("α must be between 0 and 1."); end
        if h <= 0.0; error("h must be positive."); end
        if ψ < 0.0; error("ψ must be non-negative."); end

        k_val = f.ϕ * h * ψ
        denominator = 1.0 + k_val * α
        # Denominator should always be >= 1, so denominator_sq > 0
        denominator_sq = denominator^2

        arg_sym = Symbol(argument) # Ensure symbol for comparison

        if arg_sym == :α
            # dg/dα = k / (1 + k*α)²
            return f.B₁ * k_val / denominator_sq
        elseif arg_sym == :h
            # dg/dh = (dg/dk) * (dk/dh) = [α / (1+kα)²] * (γψ)
            dk_dh = f.ϕ * ψ
            dg_dk = α / denominator_sq # Note: dg/dk = α / (1+kα)²
            return f.B₁ * dg_dk * dk_dh
            # Alternative calculation: return (f.ϕ * ψ * α) / denominator_sq
        elseif arg_sym == :ψ
            # dg/dψ = (dg/dk) * (dk/dψ) = [α / (1+kα)²] * (γh)
            dk_dψ = f.ϕ * h
            dg_dk = α / denominator_sq
            return f.B₁ * dg_dk * dk_dψ
            # Alternative calculation: return (f.ϕ * h * α) / denominator_sq
        else
            error("Invalid argument: $argument. Valid options are \"α\", \"ψ\", or \"h\".")
        end
    end # end evaluate_derivative
    # > Exponential
    @with_kw mutable struct Exponential <: RemoteEfficiencyComponent
        γ::Float64 # Scaling for k(h, ψ) = 1 - exp(-γhψ) (gamma > 0)
        λ::Float64 # Decay parameter for alpha dependence (lambda > 0)
    
        # Constructor with validation
        function Exponential(γ::Float64, λ::Float64)
            if γ <= 0.0; error("Parameter γ must be positive for Exponential"); end
            if λ <= 0.0; error("Parameter λ must be positive for Exponential"); end
            new(γ, λ)
        end
    end # end definition
    """
        _k_exp(f::Exponential, ψ::Float64, h::Float64)
    
    Internal helper to calculate k(h, ψ) = 1 - exp(-γ * h * ψ).
    """
    function _k_exp(f::Exponential, ψ::Float64, h::Float64)
        # Assuming h > 0 and ψ >= 0 checked by caller
        return 1.0 - exp(-f.γ * h * ψ)
    end
    """
        evaluate(f::Exponential, ψ::Float64, h::Float64, α::Float64)
    
    Calculates g(α, h, ψ) = (1 - exp(-γhψ)) * (1 - exp(-λα)).
    """
    function evaluate(f::Exponential, ψ::Float64, h::Float64, α::Float64)
        # Input validation
        if !(0.0 <= α <= 1.0); error("α must be between 0 and 1."); end
        if h <= 0.0; error("h must be positive."); end
        if ψ < 0.0; error("ψ must be non-negative."); end
        k_val = _k_exp(f, ψ, h)
        alpha_term = 1.0 - exp(-f.λ * α)
        return k_val * alpha_term
    end # end evaluate
    """
        evaluate_derivative(f::Exponential, argument::Union{String, Symbol}, ψ::Float64, h::Float64, α::Float64)
    
    Calculates the partial derivative of g(α, h, ψ) = k * (1 - exp(-λα))
    with respect to the specified argument (α, h, or ψ), where k = 1 - exp(-γhψ).
    """
    function evaluate_derivative(f::Exponential, argument::Union{String, Symbol}, ψ::Float64, h::Float64, α::Float64)
        # Input validation
        if !(0.0 <= α <= 1.0); error("α must be between 0 and 1."); end
        if h <= 0.0; error("h must be positive."); end
        if ψ < 0.0; error("ψ must be non-negative."); end
    
        # Pre-calculate common terms
        k_val = _k_exp(f, ψ, h)                 # k = 1 - exp(-γhψ)
        exp_neg_gamma_h_psi = exp(-f.γ * h * ψ) # exp(-γhψ)
        alpha_term_val = 1.0 - exp(-f.λ * α)    # (1 - exp(-λα))
        exp_neg_lambda_alpha = exp(-f.λ * α)    # exp(-λα)
    
        arg_sym = Symbol(argument) # Ensure symbol for comparison
    
        if arg_sym == :α
            # dg/dα = k * λ * exp(-λα)
            return k_val * f.λ * exp_neg_lambda_alpha
        elseif arg_sym == :h
            # dg/dh = (dk/dh) * (1 - exp(-λα))
            # dk/dh = γψ * exp(-γhψ)
            dk_dh = f.γ * ψ * exp_neg_gamma_h_psi
            return dk_dh * alpha_term_val
        elseif arg_sym == :ψ
            # dg/dψ = (dk/dψ) * (1 - exp(-λα))
            # dk/dψ = γh * exp(-γhψ)
            dk_dψ = f.γ * h * exp_neg_gamma_h_psi
            return dk_dψ * alpha_term_val
        else
            error("Invalid argument: $argument. Valid options are \"α\", \"ψ\", or \"h\".")
        end
    end # end evaluate_derivative
    # > BoundedInteraction
    """
    BoundedInteraction <: RemoteEfficiencyComponent

    Represents a remote efficiency component where the relative productivity factor
    increases with skill worker (h) and firm remote productivity (psi) but saturates, 
    ensuring the overall contribution g remains bounded between 0 and alpha.

    The functional form is: g(ψ, h, α) = α * k / (1 + k), where k = ϕ * ψ * h.

    # Fields
    - `ϕ::Float64`: Interaction scaling parameter (must be positive).
    """
    @with_kw mutable struct BoundedInteraction <: RemoteEfficiencyComponent
        ϕ::Float64 # Interaction scaling parameter (phi1 > 0)
        B₁::Float64 # Skill decoupling parameter (B1 > 0)
        # Constructor with validation
        function BoundedInteraction(ϕ::Float64, B₁::Float64)
            if ϕ <= 0.0
                error("Parameter ϕ must be positive for BoundedInteraction")
            end
            new(ϕ, B₁)
        end
    end # end definition
    function evaluate(f::BoundedInteraction, ψ::Float64, h::Float64, α::Float64)
        # Input validation
        if !(0.0 <= α <= 1.0); error("α must be between 0 and 1."); end
        if h <= 0.0; error("h must be positive."); end
        if ψ < 0.0; error("ψ must be non-negative."); end # Assuming non-negative psi
        # Calculate the interaction term k
        k_val = f.ϕ * h * ψ
        # Handle k_val = 0 case (e.g., if h or ψ is 0) to avoid 0/1 calculation if needed
        if k_val == 0.0
            return 0.0
        end
        # Calculate g
        # Denominator is 1 + k_val, which is >= 1 given constraints
        g_val = f.B₁ * α * k_val / (1.0 + k_val)
        # Although mathematically bounded by alpha, explicit clip can add robustness
        # and ensures it doesn't exceed 1 if alpha=1
        return max(0.0, min(g_val, 1.0))
    end # end evaluate
    """
        evaluate_derivative(f::BoundedInteraction, argument::Union{String, Symbol}, ψ::Float64, h::Float64, α::Float64)
    
        Calculates the partial derivative of g(ψ, h, α) = α * k / (1 + k)
        with respect to the specified argument (α, h, or ψ), where k = ϕ * h * ψ.
    """
    function evaluate_derivative(f::BoundedInteraction, argument::Union{String, Symbol}, ψ::Float64, h::Float64, α::Float64)
        # Input validation
        if !(0.0 <= α <= 1.0); error("α must be between 0 and 1."); end
        if h <= 0.0; error("h must be positive."); end
        if ψ < 0.0; error("ψ must be non-negative."); end
        k_val = f.ϕ * h * ψ
        denominator = 1.0 + k_val
        # Denominator is >= 1, so denominator_sq > 0
        denominator_sq = denominator^2

        arg_sym = Symbol(argument) # Ensure symbol for comparison
    
        if arg_sym == :α
            # dg/dα = k / (1 + k)
            # Handle k=0 case
            if k_val == 0.0
                return 0.0
            else
                return f.B₁ * k_val / denominator
            end
    
        elseif arg_sym == :h
            # dg/dh = α * (∂/∂h) [k / (1+k)]
            #       = α * (∂k/∂h) * (∂/∂k) [k / (1+k)]
            #       = α * (ϕ*ψ) * [1 / (1+k)²]
            dk_dh = f.ϕ * ψ
            dg_dk_term = α / denominator_sq # This is α * d/dk(k/(1+k))
            return f.B₁ * dg_dk_term * dk_dh
            # Simplified: return (α * f.ϕ * ψ) / denominator_sq
    
        elseif arg_sym == :ψ
            # dg/dψ = α * (∂/∂ψ) [k / (1+k)]
            #       = α * (∂k/∂ψ) * (∂/∂k) [k / (1+k)]
            #       = α * (ϕ*h) * [1 / (1+k)²]
            dk_dψ = f.ϕ * h
            dg_dk_term = α / denominator_sq
            return f.B₁ * dg_dk_term * dk_dψ
            # Simplified: return (α * f.ϕ * h) / denominator_sq
        else
            error("Invalid argument: $argument. Valid options are \"α\", \"ψ\", or \"h\".")
        end
    end # end evaluate_derivative
    #> List of available remote efficiency components
    remote_efficiency_components = Dict(
        "LinearFirmLogWorker" => LinearFirmLogWorker,
        "Interaction" => Interaction,
        "Exponential" => Exponential,
        "BoundedInteraction" => BoundedInteraction
    )
    #==========================================================================================
    # Create a composite production function that combines productivity and remote efficiency
    ==========================================================================================#
    @with_kw mutable struct CompositeProduction <: ProductionFunction
        productivity::ProductivityComponent
        remote_efficiency::RemoteEfficiencyComponent
        constant::Float64 = 0.0 # Default constant to 0 if not always provided
    end # end definition
    
    function evaluate(f::CompositeProduction, h::Float64, α::Float64, ψ::Float64)
        A_h = evaluate(f.productivity, h)
        g_val = evaluate(f.remote_efficiency, ψ, h, α)
        # Ensure g_val doesn't lead to negative base if needed, though model should handle
        # base = max(0.0, (1.0 - α) + g_val) # Optional: Ensure base is non-negative if A_h can be negative
        base = (1.0 - α) + g_val
        return A_h * base + f.constant
    end
    
    """
        evaluate_derivative(f::CompositeProduction, argument::Union{String, Symbol}, h::Float64, α::Float64, ψ::Float64)
    
    Calculates the partial derivative of the production function
    Y = A(h) * [ (1-α) + g(ψ, h, α) ] + constant
    with respect to the specified argument (h, α, or ψ).
    """
    function evaluate_derivative(f::CompositeProduction, argument::Union{String, Symbol}, h::Float64, α::Float64, ψ::Float64)
        # Pre-calculate component evaluations and derivatives needed
        A_h = evaluate(f.productivity, h)
        g_val = evaluate(f.remote_efficiency, ψ, h, α) # Need g itself for dY/dh
        dA_dh = evaluate_derivative(f.productivity, :h, h)
        dg_dα = evaluate_derivative(f.remote_efficiency, :α, ψ, h, α)
        dg_dh = evaluate_derivative(f.remote_efficiency, :h, ψ, h, α)
        dg_dψ = evaluate_derivative(f.remote_efficiency, :ψ, ψ, h, α)
    
        arg_sym = Symbol(argument) # Ensure symbol for comparison
    
        if arg_sym == :h
            # dY/dh = (dA/dh) * [ (1-α) + g(ψ, h, α) ] + A(h) * (dg/dh)
            return dA_dh * ( (1.0 - α) + g_val ) + A_h * dg_dh
        elseif arg_sym == :α
            # dY/dα = A(h) * [ -1 + dg/dα ]
            return A_h * (dg_dα - 1.0)
        elseif arg_sym == :ψ
            # dY/dψ = A(h) * (dg/dψ)
            return A_h * dg_dψ
        else
            error("Invalid argument: $argument. Valid options are \"h\", \"α\", or \"ψ\".")
        end
    end # end evaluate_derivative
    function get_parameters(f::CompositeProduction)
        return (
            productivity_params = f.productivity,
            remote_efficiency_params = f.remote_efficiency,
            constant = f.constant
        )
    end # end get_parameters
    #?=========================================================================================
    #? UtilityFunction: Represents worker preferences over wage and remote work: x = U(w, α, h)
    #? - Utility specific methods:
    #?   - evaluate_derivative: Evaluate the derivative of the utility function with respect to
    #?     wage w, h or α. IF THE USER SUPPLIES THE DERIVATIVE FUNCTION, ELSE: error.
    #?   - find_implied_wage: Find the wage that implies a given utility level x and remote work
    #?     level α. (IF the user defines closed form solution, ELSE: numeric solution)
    #TODO: Lets add a property to the utility function to check if we have a formula for the wage and the derivative
    #TODO: If not formula for derivative define a numerical derivative using FiniteDifference.jl
    #?=========================================================================================
    abstract type UtilityFunction <: AbstractModelFunction end
    #TODO: Consider separating the utility function into two parts: wage and remote work preferences
    #==========================================================================================
    # evaluate_derivative
    ==========================================================================================#
    function evaluate_derivative(f::UtilityFunction, args...)
        error("evaluate_derivative not implemented for $(typeof(f))")
    end
    #==========================================================================================
    # find_implied_wage 
    ==========================================================================================#
    function find_implied_wage(f::UtilityFunction, x::Float64, α::Float64, h::Float64)
        obj_fun = (w::Float64) -> evaluate(f, w, α, h) - x
        # Since parameters guarantee that utility is increasing in wage, we can use a simple
        # bisection method to find the implied wage
        return find_zero(obj_fun, (0.001, 100.0), Bisection()) # TODO: Figure out how to pin down the bounds
    end
    #==========================================================================================
    # Available implementations:
    # - PolySeparable: U(w, α, h) = (a₀ + a₁ w^η_w) - (c₀ + c₁ * h^η_h) * (1 - α)^(χ + 1)/(χ + 1)
    ==========================================================================================#
    # > QuasiLinearSkill
    @with_kw mutable struct PolySeparable <: UtilityFunction
        a₀::Float64              # Base intensity of wage preference
        a₁::Float64              # Slope of wage preference
        η_w::Float64             # Curvature of wage preference
        c₀::Float64              # Base intensity of remote work preference
        c₁::Float64              # Skill intensity of remote work preference
        η_h::Float64             # Use this for curvature on h
        χ::Float64               # Curvature of remote work preference
        # Constructor
        function PolySeparable(a₀::Float64, a₁::Float64, η_w::Float64,
                c₀::Float64, c₁::Float64, η_h::Float64, χ::Float64)
            utility = new(a₀, a₁, η_w, c₀, c₁, η_h, χ)
            # Validate parameters
            validate_parameters(utility)
            # Return the instance if parameters are valid
            return utility
        end
    end # end definition
    function validate_parameters(f::PolySeparable)
        params = [f.a₀, f.a₁, f.η_w, f.c₀, f.c₁, f.η_h, f.χ]
        param_names = ["a₀", "a₁", "η_w", "c₀", "c₁", "η_h", "χ"]
        for (param, name) in zip(params, param_names)
            if param < 0.0
                error("$name must be non-negative")
            end
        end
        if f.η_h < 1
            error("η_h must be greater than or equal to 1")
        end
    end # end validate_parameters
    function evaluate(f::PolySeparable, w::Float64, α::Float64, h::Float64)
        # Replace f.η with f.η_h
        # return  (a₀ + a₁ * w^η_w) - (f.c₀ + f.c₁ * h^f.η_h) * (1 - α)^f.χ
        return  (f.a₀ + f.a₁ * w^f.η_h) - (f.c₀ + f.c₁ * h^f.η_h) * (1 - α)^(f.χ + 1)/(f.χ + 1)
    end # end evaluate
    function evaluate_derivative(f::PolySeparable, argument::Union{String, Symbol}, 
                                w::Float64, α::Float64, h::Float64)
        # Evaluate the derivative of the utility function with respect to the given argument
        if argument == "w"
            return f.a₁ * f.η_h * w^(f.η_h - 1)
        elseif argument == "α"
            return  (f.c₀ + f.c₁ * h^f.η_h) * (1 - α)^(f.χ)
        elseif argument == "h"
            return -f.η_h * f.c₁ * h^(f.η_h - 1) * (1 - α)^(f.χ + 1)/(f.χ + 1)
        else
            error("Invalid argument: $argument")
        end
    end # end evaluate_derivative
    function find_implied_wage(f::PolySeparable, x::Float64, α::Float64, h::Float64)
        return ((x + (f.c₀ + f.c₁ * h^f.η_h) * (1 - α)^(f.χ + 1)/(f.χ + 1) - f.a₀ / f.a₁))^(1 / f.η_w)
    end # end find_implied_wage
    #> List of available utility functions
    utility_functions = Dict(
        "PolySeparable" => PolySeparable
    )
    #?=========================================================================================
    #? MatchingFunction: Represents labor market matching technology
    #? - Matching function that maps vacancies and unemployed workers to matches.
    #? - Fields:
    #?   - function parameters
    #?   - maxVacancyFillRate (maximum rate at which vacancies are filled)
    #? - Matching specific methods:
    #?   - JobFindRate: Evaluate the probability of finding a job given a sub-market tightness.
    #?   - VacancyFillRate: Evaluate the probability of filling a vacancy given a sub-market
    #?                      tightness.
    #?   - InvertVacancyFill: Solve the sub-market free entry condition for the tightness.
    #?=========================================================================================
    abstract type MatchingFunction <: AbstractModelFunction end
    # invert vacancy fill
    function invert_vacancy_fill(f::MatchingFunction, κ::Float64, Ej::Float64)
        # First, check submarket activity.
        if f.maxVacancyFillRate * Ej < κ
            return 0.0
        else
            # Define the objective function:
            # We want to solve: eval_prob_vacancy_fill(f, θ) * Ej - κ = 0.
            obj_fun = (θ::Float64) -> eval_prob_vacancy_fill(f, θ) * Ej - κ
            θ_low = 1e-8  # Lower bound
            # Pin down the bounds for bisection.
            θ_high = 1.0  # Initial guess for upper bound
            # Increase θ_high until the objective becomes negative.
            while obj_fun(θ_high) > 0
                θ_high *= 2.0
                if θ_high > 1e6
                    error("Could not bracket the solution for invert_vacancy_fill.")
                end
            end
            # Use the bisection method to find the zero.
            return find_zero(obj_fun, (θ_low, θ_high), Bisection())
        end
    end
    #==========================================================================================
    # Available implementations:
    # - CobbDouglasMatching: M(V, U) = U^γ * V^(1 - γ)
    # - CESMatching: M(V, U) = (V^γ + U^γ)^(1/γ)
    # - ExponentialMatching: M(V, U) = 1 - exp(-γ * θ)
    # - LogisticMatching: M(V, U) = (V * U )^γ / (V^γ + U^γ)
    ==========================================================================================#
    # > CobbDouglasMatching
    @with_kw mutable struct CobbDouglasMatching <: MatchingFunction
        γ::Float64               # Matching elasticity
        maxVacancyFillRate::Float64  # Maximum rate at which vacancies are filled
        # Constructor
        function CobbDouglasMatching(γ::Float64)
            matching = new(γ, Inf)
            # Validate parameters
            validate_parameters(matching)
            # Return the instance if parameters are valid
            return matching
        end
    end # end definition
    function eval_prob_job_find(f::CobbDouglasMatching, θ::Float64)
        return θ^(1 - f.γ)
    end
    function eval_prob_vacancy_fill(f::CobbDouglasMatching, θ::Float64)
        return θ^(-f.γ)
    end
    function invert_vacancy_fill(f::CobbDouglasMatching, κ::Float64, Ej::Float64)
        # Removed call to undefined 'max_q(f)'; user must supply proper bounds.
        # If needed, add proper logic here.
        if f.maxVacancyFillRate * Ej < κ
            return 0.0
        else
            return (κ / Ej)^(-1 / f.γ)
        end
    end
    function validate_parameters(f::CobbDouglasMatching)
        if f.γ <= 0.0
            error("γ must be positive")
        end
    end # end validate_parameters
    # > CESMatching
    @with_kw mutable struct CESMatching <: MatchingFunction
        γ::Float64   # CES parameter (γ > 0)
        maxVacancyFillRate::Float64  # Maximum rate at which vacancies are filled
        # Constructor
        function CESMatching(γ::Float64)
            # Compute the maximum vacancy fill rate
            matching = new(γ, Inf)
            validate_parameters(matching)
            return matching
        end
    end
    function eval_prob_job_find(f::CESMatching, θ::Float64)
        return θ * (θ^(f.γ) + 1)^(-1/f.γ)
    end
    function eval_prob_vacancy_fill(f::CESMatching, θ::Float64)
        #! TODO: Need to be super sure about this
        return (θ > 0.0) ?  ((θ^(f.γ) + 1)^(-1/f.γ)) : 0.0
    end
    function validate_parameters(f::CESMatching)
        if f.γ <= 0.0
            error("γ must be positive")
        end
    end
    function invert_vacancy_fill(f::CESMatching, κ::Float64, Ej::Float64)
        if Ej > κ
            return  ( ( Ej / κ )^f.γ - 1  )^(1/f.γ)
        else
            return 0.0
        end
    end
    function invert_vacancy_fill(f::CESMatching, q::Float64)
        if value >= 1.0 - 1e-10 # Allow for tiny float errors near 1
            return 0.0
        elseif value <= 1e-10 # Treat value near zero as zero tightness or handle appropriately
            # Depending on interpretation, could return 0.0 or Inf
            # Returning 0.0 is often safer numerically if value=0 means inactive market
            return 0.0
            # Alternatively: return Inf
        else
            return  (q^f.γ - 1  )^(1/f.γ)
        end
    end
    function recover_tightness(f::CESMatching, which::Symbol, value::Float64)
        """
        recover market tightness from either job finding rates or vacancy filling rates
        """ 
        if which == :q
            if value >= 1.0 - 1e-10 # Allow for tiny float errors near 1
                return 0.0
            elseif value <= 1e-10 # Treat value near zero as zero tightness or handle appropriately
                # Depending on interpretation, could return 0.0 or Inf
                # Returning 0.0 is often safer numerically if value=0 means inactive market
                return 0.0
                # Alternatively: return Inf
            end
            # --- Calculation ---
            # Calculate base: value^(-γ) - 1
            # Need to be careful with negative value if γ is not an integer, but value should be positive here.
            try
                base = value^(-f.γ) - 1
                # Base must be non-negative for the outer exponentiation if 1/γ is fractional
                if base < 0
                    # This shouldn't happen for valid value in (0, 1) and typical γ > 0
                        @warn "Negative base encountered in inverse matching for value=$value, γ=$(f.γ). Base = $base. Returning 0.0"
                        return 0.0
                    end
                    # Calculate θ = base^(1/γ)
                    θ = base^(1 / f.γ)
                    return θ
            catch e
                # Catch potential errors like DomainError if base is negative and exponent is fractional
                @warn "Error calculating inverse matching function for value=$value, γ=$(f.γ): $e. Returning 0.0"
                return 0.0 # Return 0 as a safe fallback
            end
        elseif which == :p
            #! TODO: Need to implement this 
            return NaN
        else
            error("Invalid argument for which: $which. Use :p or :q.")
        end
    end
    #> ExponentialMatching
    @with_kw mutable struct ExponentialMatching <: MatchingFunction
        γ::Float64   # parameter controlling the arrival rate (γ > 0)
        maxVacancyFillRate::Float64  # Maximum rate at which vacancies are filled
        # Constructor
        function ExponentialMatching(γ::Float64)
            # Compute the maximum vacancy fill rate
            # q(θ) = [1 - exp(-γ θ)]/θ. As θ→0, use L'Hôpital: limit = γ.
            matching = new(γ, γ)
            validate_parameters(matching)
            return matching
        end
    end
    function eval_prob_job_find(f::ExponentialMatching, θ::Float64)
        # p(θ) = 1 - exp(-γ * θ)
        return 1 - exp(-f.γ * θ)
    end
    function eval_prob_vacancy_fill(f::ExponentialMatching, θ::Float64)
        # q(θ) = p(θ)/θ = [1 - exp(-γ * θ)] / θ
        return (1 - exp(-f.γ * θ)) / θ
    end
    function validate_parameters(f::ExponentialMatching)
        if f.γ <= 0.0
            error("γ must be positive")
        end
    end
    #> LogisticMatching
    @with_kw mutable struct LogisticMatching <: MatchingFunction
        γ::Float64   # parameter controlling the curvature (γ > 0)
        maxVacancyFillRate::Float64  # Maximum rate at which vacancies are filled
        # Constructor: use input γ instead of f.γ for setting maxVacancyFillRate
        function LogisticMatching(γ::Float64)
            local mvr = if γ < 1
                Inf
            elseif isapprox(γ, 1.0)
                1.0
            else
                0.0
            end
            matching = new(γ, mvr)
            validate_parameters(matching)
            return matching
        end
    end
    function eval_prob_job_find(f::LogisticMatching, θ::Float64)
        # p(θ) = θ^(γ) / (1 + θ^(γ))
        return θ^(f.γ) / (1 + θ^(f.γ))
    end
    function eval_prob_vacancy_fill(f::LogisticMatching, θ::Float64)
        # q(θ) = p(θ)/θ = [θ^(γ) / (1 + θ^(γ))] / θ
        return (θ^(f.γ) / (1 + θ^(f.γ))) / θ
    end
    function validate_parameters(f::LogisticMatching)
        if f.γ <= 0.0
            error("γ must be positive")
        end
    end
    #> List of available matching functions
    matching_functions = Dict(
        "CobbDouglasMatching" => CobbDouglasMatching,
        "CESMatching" => CESMatching,
        "ExponentialMatching" => ExponentialMatching,
        "LogisticMatching" => LogisticMatching
    )
    #?=========================================================================================
    #? Initialization Functions
    #? - Create matching function
    #? - Create production function
    #? - Create utility function
    #?=========================================================================================
    #*=========================================================================================
    #* create_matching_function(type::String, params::Array{Float64})::MatchingFunction
    #* Description:
    #*  - Create a matching function of the specified type with the given parameters.   
    #* Parameters:
    #*  - type::String - The type of matching function to create (e.g., "CESMatching")
    #*  - params::Array{Float64} - Array of parameters for the matching function. ORDER MATTERS!
    #* Returns:
    #*  - MatchingFunction - An initialized matching function of the specified type
    #*=========================================================================================
    function create_matching_function(type::String, params::Array{Float64})::MatchingFunction
        if haskey(matching_functions, type)
            return matching_functions[type](params...)
        else
            # TODO: figure out how to handle user provided matching functions
            error("Unknown matching function type: $type")
        end
    end
    #*=========================================================================================
    #* create_utility_function(type::String, params::Array{Float64})::UtilityFunction
    #* Description:
    #*  - Create a utility function of the specified type with the given parameters.
    #* Parameters:
    #*  - type::String - The type of utility function to create (e.g., "QuasiLinear")
    #*  - params::Array{Float64} - Array of parameters for the matching function. ORDER MATTERS!
    #* Returns:
    #*  - UtilityFunction - An initialized utility function of the specified type
    #*=========================================================================================
    function create_utility_function(type::String, params::Array{Float64})::UtilityFunction
        if haskey(utility_functions, type)
            return utility_functions[type](params...)
        else
            # TODO: figure out how to handle user provided utility functions
            error("Unknown utility function type: $type")
        end
    end
    #*=========================================================================================
    #* create_production_function(type_productivity_component::String,
    #*                           type_remote_efficiency::String,
    #*                           params_productivity_component::Array{Float64},
    #*                           params_remote_efficiency::Array{Float64})::ProductionFunction
    #* Description:
    #*  - Create a composite production function with the specified productivity and remote
    #*    efficiency components and their parameters.
    #* Parameters:
    #*  - type_productivity_component::String - The type of productivity component to create
    #*  - type_remote_efficiency::String - The type of remote efficiency component to create
    #*  - params_productivity_component::Array{Float64} - Array of parameters for the productivity
    #*    component. ORDER MATTERS!
    #*  - params_remote_efficiency::Array{Float64} - Array of parameters for the remote efficiency
    #*    component. ORDER MATTERS!
    #* Returns:
    #*  - ProductionFunction - An initialized production function with the specified components
    #*=========================================================================================
    function create_production_function(type_productivity_component::String,
                                        type_remote_efficiency::String,
                                        params_productivity_component::Array{Float64},
                                        params_remote_efficiency::Array{Float64},
                                        constant::Float64
                                        )::ProductionFunction
        #> Create the productivity component
        if haskey(productivity_components, type_productivity_component)
            productivity_component = productivity_components[type_productivity_component](params_productivity_component...)
        else
            # TODO: figure out how to handle user provided productivity components
            error("Unknown productivity component type: $type_productivity_component")
        end
        #> Create the remote efficiency component
        if haskey(remote_efficiency_components, type_remote_efficiency)
            remote_efficiency_component = remote_efficiency_components[type_remote_efficiency](params_remote_efficiency...)
        else
            #TODO: figure out how to handle user provided remote efficiency components
            error("Unknown remote efficiency component type: $type_remote_efficiency")
        end
        #> Create the composite production function
        return CompositeProduction(productivity_component, remote_efficiency_component, constant)
    end
end # end module