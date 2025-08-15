#==========================================================================================
# Title:
# Author: Mitchell Valdes-Bobes @mitchv34
# Date: 2025-08-13
# Description: Simplified model setup with hardcoded functional forms.
==========================================================================================#
using Parameters, Roots, Random, YAML, Distributions, Term, OrderedCollections
include("helpers.jl") # Lean fit_kde_psi (pure Julia)
#?=========================================================================================
#? Model Data Structures
#?=========================================================================================
@with_kw mutable struct Primitives   
    #> Functional form parameters (stored explicitly)
    A₀::Float64
    A₁::Float64
    ψ₀::Float64
    ϕ::Float64
    ν::Float64
    c₀::Float64
    χ::Float64
    γ₀::Float64
    γ₁::Float64
    #> Model functions (simple closures)
    production_fun::Function  # Production function
    utility_fun::Function     # Utility function
    matching_fun::Function    # Matching function
    #> Market parameters
    κ₀::Float64     # Vacancy posting cost parameter
    κ₁::Float64     # Vacancy posting cost parameter
    β::Float64      # Time discount factor
    δ::Float64      # Baseline job destruction rate
    b::Float64      # Unemployment benefits
    ξ::Float64      # Worker bargaining power
    #> Grids
    #* ψ_grid Remote productivity grid
    n_ψ::Int64
    ψ_min::Float64
    ψ_max::Float64
    ψ_grid::Vector{Float64}
    ψ_pdf::Vector{Float64}
    ψ_cdf::Vector{Float64}
    #* h_grid Skill grid
    aₕ::Float64 # Parameter of the Beta distribution of skill
    bₕ::Float64 # Parameter of the Beta distribution of skill
    n_h::Int64
    h_min::Float64
    h_max::Float64
    h_grid::Vector{Float64}
    h_pdf::Vector{Float64}
    h_cdf::Vector{Float64}
    #> Constructor with validation
    function Primitives(args...)
        prim = new(args...)
        #?Validate parameter ranges
        ## > Discount factor
        if prim.β < 0 || prim.β > 1
            throw(ArgumentError("β must be in [0,1]"))
        end
        ## > Destruction rate
        if prim.δ < 0 || prim.δ > 1
            throw(ArgumentError("δ must be in [0,1]"))
        end
        ## > Posting cost parameters
        if prim.κ₀ < 0 || prim.κ₁ < 0
            throw(ArgumentError("κ₀ and κ₁ must be non negative"))
        end
        return prim
    end
end
"""
    Results

A mutable struct that stores the core value functions, aggregate market outcomes, endogenous distributions, policy functions, and thresholds for the structural model.

# Fields

- **S::Matrix{Float64}**: Total match surplus `S(h, ψ)`, an array of size `(n_h, n_ψ)`.
- **U::Vector{Float64}**: Unemployed worker value `U(h)`, an array of size `(n_h)`.
- **θ::Float64**: Aggregate market tightness (scalar).
- **p::Float64**: Job finding probability (scalar).
- **q::Float64**: Vacancy filling probability (scalar).
- **v::Vector{Float64}**: Vacancies posted by each firm type `v(ψ)`, an array of size `(n_ψ)`.
- **u::Vector{Float64}**: Mass of unemployed workers of each type `u(h)`, an array of size `(n_h)`.
- **n::Matrix{Float64}**: Mass of employed workers in each match `n(h, ψ)`, an array of size `(n_h, n_ψ)`.
- **α_policy::Matrix{Float64}**: Optimal remote work policy `α(h, ψ)`, an array of size `(n_h, n_ψ)`.
- **w_policy::Matrix{Float64}**: Optimal wage policy `w(h, ψ)`, an array of size `(n_h, n_ψ)`.
- **ψ_bottom::Vector{Float64}**: Threshold for hybrid work `ψ_bottom(h)`, an array of size `(n_h)`.
- **ψ_top::Vector{Float64}**: Threshold for full-time remote work `ψ_top(h)`, an array of size `(n_h)`.

# Constructor

    Results(prim::Primitives)

Initializes a `Results` object using the model primitives. Sets up arrays with appropriate dimensions and default values. The remote work policy (`α_policy`) and thresholds (`ψ_bottom`, `ψ_top`) are computed from the primitives, while other fields are initialized to zeros or reasonable starting guesses.

# Notes

- The wage policy (`w_policy`) cannot be pre-calculated as it depends on endogenous variables and is initialized to zeros.
- The remote work policy (`α_policy`) is transposed to match the expected dimensions.
- Throws an `ArgumentError` if `prim.h_pdf` is not a `Vector{Float64}`.
"""
mutable struct Results
    #== Core Value Functions ==#
    S::Matrix{Float64}                  # Total match surplus S(h, ψ) -> Array of size (n_h, n_ψ)
    U::Vector{Float64}                  # Unemployed worker value U(h) -> Array of size (n_h)

    #== Aggregate Market Outcomes ==#
    θ::Float64                          # Aggregate market tightness (scalar)
    p::Float64                          # Job finding probability (scalar)
    q::Float64                          # Vacancy filling probability (scalar)

    #== Firm and Worker Distributions (Endogenous) ==#
    v::Vector{Float64}                  # Vacancies posted by each firm type v(ψ) -> Array of size (n_ψ)
    u::Vector{Float64}                  # Mass of unemployed workers of each type u(h) -> Array of size (n_h)
    n::Matrix{Float64}                  # Mass of employed workers in each match n(h, ψ) -> Array of size (n_h, n_ψ)

    #== Policy Functions ==#
    α_policy::Matrix{Float64}           # Optimal remote work α(h, ψ) -> Array of size (n_h, n_ψ)
    w_policy::Matrix{Float64}           # Optimal wage w(h, ψ) -> Array of size (n_h, n_ψ)

    #== Thresholds (as before) ==#
    ψ_bottom::Vector{Float64}         # Threshold for hybrid work ψ_bottom(h)
    ψ_top::Vector{Float64}            # Threshold for full-time remote work ψ_top(h)

    # Constructor
    function Results(prim::Primitives)
        n_h = prim.n_h
        n_ψ = prim.n_ψ

        # Initialize with appropriate dimensions and default values
        S_init = zeros(n_h, n_ψ)
        U_init = zeros(n_h)
        θ_init = 1.0 # A reasonable starting guess
        p_init, q_init = 0.0, 0.0 # Will be calculated from θ
        v_init = ones(n_ψ) ./ n_ψ # e.g., uniform distribution
        
        # Start with unemployment matching population distribution
        # Ensure h_pdf is a Vector{Float64} as expected
            if !(typeof(prim.h_pdf) <: Vector{Float64})
                throw(ArgumentError("prim.h_pdf must be a Vector{Float64} for Results initialization"))
            end
            u_init = copy(prim.h_pdf)

            n_init = zeros(n_h, n_ψ)
            w_policy_init = zeros(n_h, n_ψ)
    
            # This part only depends on primitives.
            # find_thresholds_and_optimal_remote_policy returns α_policy as (n_ψ, n_h)
            ψ_bottom_calc, ψ_top_calc, α_policy_calc_psi_h = find_thresholds_and_optimal_remote_policy(prim)
            # Transpose α_policy to be (n_h, n_ψ)
            α_policy_init = permutedims(α_policy_calc_psi_h, (2,1))

            # IMPORTANT: Wage policy CANNOT be pre-calculated anymore.
            # It depends on the endogenous U(h) and S(h, ψ), which are solved for.
            # So we just initialize it to zeros.

            new(S_init, U_init, θ_init, p_init, q_init, v_init, u_init, n_init,
                α_policy_init, w_policy_init, ψ_bottom_calc, ψ_top_calc)
    end
end
"""
    create_primitives_from_yaml(yaml_file::String) :: Primitives

Load model primitives and grids from a YAML configuration file.

# Arguments
- `yaml_file::String`: Path to the YAML file containing model configuration.

# Returns
- `Primitives`: An object containing model parameters, functional forms, and discretized grids for the model.

# Description
This function reads a YAML configuration file to extract model parameters and grid specifications. It performs the following steps:
- Loads model parameters and grid definitions from the YAML file.
- Extracts scalar parameters for the model (e.g., production, utility, matching function parameters).
- Constructs the ψ (psi) grid using kernel density estimation (KDE) on provided data, with optional weighting.
- Constructs the h (skill) grid by discretizing a Beta distribution over a specified range.
- Defines closures for the production, utility, and matching functions using the loaded parameters.
- Normalizes probability density functions (PDFs) and computes cumulative distribution functions (CDFs) for the grids.
- Returns a `Primitives` object containing all parameters, functions, and grids required for the model.

# Notes
- The function assumes that the YAML file contains flat (non-nested) parameter dictionaries under "ModelParameters" and "ModelGrids".
- The ψ grid is constructed using a KDE fit to empirical data, while the h grid is based on a Beta distribution.
- The function currently contains some commented-out or legacy code for an additional utility grid, which is not used in the returned object.
"""
function create_primitives_from_yaml(yaml_file::String)::Primitives
    #> Load configuration from YAML file
    config = YAML.load_file(yaml_file, dicttype = OrderedDict)
    model_parameters = config["ModelParameters"]
    model_grids = config["ModelGrids"]

    # Extract all parameters directly from model_parameters (flat, not nested)
    κ₀    = model_parameters["kappa0"]
    κ₁    = model_parameters["kappa1"]
    β     = model_parameters["beta"]
    δ     = model_parameters["delta"]
    b     = model_parameters["b"]
    ξ     = model_parameters["xi"]
    A₀    = model_parameters["A0"]
    A₁    = model_parameters["A1"]
    ψ₀    = model_parameters["psi_0"]
    ϕ     = model_parameters["phi"]
    ν     = model_parameters["nu"]
    c₀    = model_parameters["c0"]
    χ     = model_parameters["chi"]
    γ₀    = model_parameters["gamma0"]
    γ₁    = model_parameters["gamma1"]
    aₕ   = model_parameters["a_h"]
    bₕ   = model_parameters["b_h"]
    n_ψ   = model_parameters["n_psi"]
    ψ_min  = model_grids["psi_min"]
    ψ_max  = model_grids["psi_max"]
    ψ_data = model_grids["psi_data"]
    ψ_column = model_grids["psi_column"]
    ψ_weight = model_grids["psi_weight"]
    # Create ψ_grid
    ψ_grid, ψ_pdf, ψ_cdf = fit_kde_psi(
        ψ_data, ψ_column; weights_col=ψ_weight, num_grid_points=n_ψ, boundary=(ψ_min, ψ_max))

    #* h_grid Skill grid
    n_h = model_grids["n_h"]
    h_min = model_grids["h_min"]
    h_max = model_grids["h_max"]
    # Now we fit a Beta distribution:
    # Fit a Beta distribution to h on [h_min, h_max] with shape parameters a_h, b_h
    h_values = collect(range(h_min, h_max, length=n_h))
    # Map h_values to [0,1] for Beta
    h_scaled = (h_values .- h_min) ./ (h_max - h_min)
    beta_dist = Beta(aₕ, bₕ)
    # Compute unnormalized pdf on grid
    h_pdf_raw = pdf.(beta_dist, h_scaled)
    # Normalize pdf to sum to 1 (discrete approximation)
    h_pdf = h_pdf_raw ./ sum(h_pdf_raw)
    h_cdf = cumsum(h_pdf)
    

    # Build simple  functions that can be evaluated
    production_fun = (h, ψ, α) -> (A₀ + A₁*h) * ((1 - α) + α * (ψ₀ * h^ϕ * ψ^ν))
    utility_fun    = (w, α)    -> w - c₀ * (1 - α)^(χ + 1) / (χ + 1)
    matching_fun   = (V, U)    -> γ₀ * U^γ₁ * V^(1 - γ₁)
    

    #> Create Primitives object
    return Primitives(
        A₀, A₁, ψ₀, ϕ, ν, c₀, χ, γ₀, γ₁,
        production_fun, utility_fun, matching_fun,
        κ₀, κ₁, β, δ, b, ξ,
        n_ψ, ψ_min, ψ_max, ψ_grid, ψ_pdf, ψ_cdf, 
        aₕ, bₕ, n_h, h_min, h_max, h_values, h_pdf, h_cdf
    )
end

function find_thresholds_and_optimal_remote_policy(prim::Primitives)
    # 1. Unpack parameters
    @unpack n_ψ, ψ_grid, n_h, h_grid, A₀, A₁, ψ₀, ϕ, ν, c₀, χ = prim

    # 2. Define helper functions
    A(h) = A₀ + A₁*h
    g(h, ψ) = ψ₀ * h^ϕ * ψ^ν

    # 3. Calculate thresholds analytically
    ψ_top = @. (1 / (ψ₀ * h_grid^ϕ))^(1/ν)
    
    ψ_bottom = zeros(n_h)
    for (i, h) in enumerate(h_grid)
        Ah = A₀ + A₁*h
        if Ah <= c₀
            ψ_bottom[i] = Inf
        else
            ψ_bottom[i] = ψ_top[i] * (1 - c₀ / Ah)^(1/ν)
        end
    end

    # 4. Initialize policy matrix
    α_policy = zeros(n_ψ, n_h)

    # 5. Compute the optimal α policy with CORRECTED logic
    for (i, h) in enumerate(h_grid)
        ψ_b = ψ_bottom[i]
        ψ_t = ψ_top[i]

        for (j, ψ) in enumerate(ψ_grid)
            
            # --- LOGIC CORRECTION IS HERE ---
            if isinf(ψ_b)
                # CASE 1: No hybrid region exists for this worker.
                # The world is binary: either in-person or full-remote.
                # The only threshold that matters is ψ_top.
                if ψ >= ψ_t
                    α_policy[j, i] = 1.0 # Full Remote
                else
                    α_policy[j, i] = 0.0 # Full In-Person
                end
            else
                # CASE 2: A hybrid region exists. Use the standard three-regime logic.
                if ψ <= ψ_b
                    α_policy[j, i] = 0.0 # Full In-Person
                elseif ψ >= ψ_t
                    α_policy[j, i] = 1.0 # Full Remote
                else
                    # Hybrid Work. Calculate the interior solution.
                    term = ( (A₀ + A₁*h) / c₀) * (1 - g(h, ψ))
                    # The term must be non-negative in the hybrid region by construction
                    α_policy[j, i] = 1.0 - max(0.0, term)^(1/χ)
                end
            end
        end
    end

    return ψ_bottom, ψ_top, α_policy
end
"""
    initializeModel(yaml_file::String) -> Tuple{Primitives, Results}

Initializes the model using the provided YAML file.

# Arguments
- `yaml_file::String`: Path to the YAML file containing model configuration.

# Returns
- `Tuple{Primitives, Results}`: A tuple containing the initialized `Primitives` object and the corresponding `Results` object.

# Description
This function creates the model primitives from the specified YAML file and initializes the results structure. It returns both the primitives and results for further use in the model workflow.
"""
function initializeModel(yaml_file::String)::Tuple{Primitives, Results}
    #> Create primitives from YAML file
    prim = create_primitives_from_yaml(yaml_file)
    #> Initialize results (just use contructor)
    res = Results(prim)
    #> Return primitives and results
    return prim, res
end