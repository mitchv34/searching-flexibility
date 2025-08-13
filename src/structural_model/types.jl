#==========================================================================================
Title: types.jl
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2025-02-26
Description: Data structures for the labor market model and simulation. Also includes 
            initialization functions for the model and simulation objects.
==========================================================================================#
module Types
    # Load required packages
    using Parameters, Random, YAML, Distributions, Term, OrderedCollections
    # Load functions to calibrate empirical distributions
    include("./calibrate_empirical_distributions.jl")
    # Load My Modules
    using ..ModelFunctions
    # Import specific types from ModelFunctions
    import ..ModelFunctions: MatchingFunction, ProductionFunction, UtilityFunction, evaluate, evaluate_derivative
    # Export the data structures and initialization functions
    export Primitives, Results, Worker, Economy, initializeModel, initializeEconomy
    export create_primitives_from_yaml, initializeResults, modify_parameter!
    #?=========================================================================================
    #? Model Data Structures
    #?=========================================================================================
    #==========================================================================================
    #* Primitives: Contains all parameters, grids, and functions for the labor market model. 
        #> Fields
            - `matching_function::MatchingFunction`: Function for job matching.
            - `production_function::ProductionFunction`: Function for production.
            - `utility_function::UtilityFunction`: Function for utility.
            - `κ::Float64`: Vacancy posting cost.
            - `β::Float64`: Time discount factor.
            - `δ_bar::Float64`: Baseline job destruction rate.
            - `n_ψ::Int64`: Number of remote productivity grid points.
            - `n_h::Int64`: Number of skill grid points.
            - `h_min::Float64`: Minimum worker skill.
            - `h_max::Float64`: Maximum worker skill.
            - `n_x::Int64`: Number of utility grid points.
            - `x_min::Float64`: Minimum utility.
            - `x_max::Float64`: Maximum utility.
        #> Constructor with validation
            - `Primitives(args...)`: Constructor that validates grid sizes and parameter ranges.
        ##* Validations
        - Grid sizes:
            - `ψ_grid`, `ψ_pdf`, and `ψ_cdf` must have length `n_ψ`.
            - `h_grid` and `h_pdf` must have length `n_h`.
        - Parameter ranges:
            - `β` must be in the range `(0,1)`.
            - `δ_bar` must be in the range `[0,1]`.
            - `κ` must be positive.
        - Grids:
            - `ψ_pdf` must be a probability distribution with non-negative values and sum to 1.
            - `h_pdf` must be a probability distribution with non-negative values and sum to 1.
    ==========================================================================================#
    @with_kw mutable struct Grid
        n::Int64                                  # Number of grid points
        min::Float64                              # Minimum value
        max::Float64                              # Maximum value
        values::Vector{Float64}  # Grid values
        pdf::Union{Function, Vector{Float64}}     # PDF of the grid values
        cdf::Union{Function, Vector{Float64}}     # CDF of the grid values
        function Grid(
                    values::Vector{Float64}; 
                    pdf::Union{Function, Vector{Float64}}=zeros(length(values)), 
                    cdf::Union{Function, Vector{Float64}}=zeros(length(values))
            )
            n = length(values)
            min = minimum(values)
            max = maximum(values)
            if typeof(pdf) == Vector{Float64}
                if (length(values) != n) || (length(pdf) != n) || (length(cdf) != n)
                    throw(ArgumentError("values must have length n"))
                end
                if any(pdf .< 0) || any(cdf .< 0)
                    throw(ArgumentError("pdf and cdf must be non-negative"))
                end
                if !isapprox(sum(pdf), 1.0, atol=1e-10)
                    throw(ArgumentError("pdf does not sum to 1."))
                end
                if !isapprox( maximum( abs.( cumsum(pdf) -  cdf)), 0.0, atol=1e-10)
                    throw(ArgumentError("pdf and cdf are not consistent"))
                end
                # Check if sorted and if not sort (also sort pdf and cdf)
                if !issorted(values)
                    sorted_indices = sortperm(values)
                    values = values[sorted_indices]
                    pdf = pdf[sorted_indices]
                    cdf = cdf[sorted_indices]
                end
            end
            # Create the grid object
            new(n, min, max, values, pdf, cdf)
        end
    end
    @with_kw mutable struct Primitives   
        #> Model functions
        matching_function::MatchingFunction
        production_function::ProductionFunction
        utility_function::UtilityFunction
        #> Market parameters
        κ₀::Float64     # Vacancy posting cost parameter
        κ₁::Float64     # Vacancy posting cost parameter
        β::Float64      # Time discount factor
        δ_bar::Float64  # Baseline job destruction rate
        b::Float64      # Unemployment benefits
        ξ::Float64      # Worker bargaining power
        #> Grids
        ψ_grid::Grid    # Remote productivity grid
        h_grid::Grid    # Skill grid
        x_grid::Grid    # Utility grid for search intensity (no longer used for x_policy directly)
        #> Constructor with validation
        function Primitives(args...)
            prim = new(args...)
            #?Validate parameter ranges
            ## > Discount factor
            if prim.β < 0 || prim.β > 1
                throw(ArgumentError("β must be in [0,1]"))
            end
            ## > Destruction rate
            if prim.δ_bar < 0 || prim.δ_bar > 1
                throw(ArgumentError("δ_bar must be in [0,1]"))
            end
            ## > Posting cost parameters
            if prim.κ₀ < 0 || prim.κ₁ < 0
                throw(ArgumentError("κ₀ and κ₁ must be non negative"))
            end
            return prim
        end
    end
    #==========================================================================================
    #* Results: Holds equilibrium objects from the labor market model solution.
    # Fields
        - S::Matrix{Float64}: Total match surplus S(h, ψ)
        - U::Vector{Float64}: Value of a h worker when unemployed: U(h)
        - θ::Float64: Aggregate market tightness
        - p::Float64: Aggregate job finding probability
        - q::Float64: Aggregate vacancy filling probability
        - v::Vector{Float64}: Vacancies posted by each firm type v(ψ)
        - u::Vector{Float64}: Mass of unemployed workers of each type u(h)
        - n::Matrix{Float64}: Mass of employed workers in each match n(h, ψ)
        - α_policy::Matrix{Float64}: Optimal remote work a firm offers an h worker: α(h, ψ)
        - w_policy::Matrix{Float64}: Optimal wage a firm offers an h worker: w(h, ψ)
        - ψ_bottom::Vector{Float64}: Threshold for hybrid work (depending on worker skill h)
        - ψ_top::Vector{Float64}: Threshold for full-time remote work (depending on worker skill h)
    ==========================================================================================#
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
            n_h = prim.h_grid.n
            n_ψ = prim.ψ_grid.n

            # Initialize with appropriate dimensions and default values
            S_init = zeros(n_h, n_ψ)
            U_init = zeros(n_h)
            θ_init = 1.0 # A reasonable starting guess
            p_init, q_init = 0.0, 0.0 # Will be calculated from θ
            v_init = ones(n_ψ) ./ n_ψ # e.g., uniform distribution
            
            # Start with unemployment matching population distribution
            # Ensure h_grid.pdf is a Vector{Float64} as expected
            if !(typeof(prim.h_grid.pdf) <: Vector{Float64})
                throw(ArgumentError("prim.h_grid.pdf must be a Vector{Float64} for Results initialization"))
            end
            u_init = copy(prim.h_grid.pdf)

            n_init = zeros(n_h, n_ψ)
            w_policy_init = zeros(n_h, n_ψ)
    
            # This part is good and can be kept. It only depends on primitives.
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
    #?=========================================================================================
    #? Simulation Data Structures 
    #?=========================================================================================c
    #==========================================================================================
    #* Worker: Represents an individual worker in the simulation.
    # Fields
        - id::String: Worker unique identifier
        - skill::Int: Worker skill value (index corresponding to model skill grid)
        - age::Int: Worker age (number of periods in the simulation)
        - employed::Vector{Bool}: Employment status history (true if employed)
        - wage::Vector{Float64}: Wage history
        - x_choice::Int: Worker search policy if unemployed (index in the x_policy grid)
        - firm_type::Vector{Float64}: Employer indicator/history (e.g., index from firm remote productivity grid)
        - remote_work::Vector{Float64}: Remote work fraction α history (if employed)
    ==========================================================================================#
    mutable struct Worker
        #? Worker attributes
        id::String                # Worker unique identifier
        skill::Int                # Worker skill value (index corresponding to model skill grid)
        age::Int                  # Worker age (number of periods in the simulation)
        employed::Vector{Bool}    # Employment status history (true if employed)
        wage::Vector{Float64}     # Wage history
        x_choice::Int             # Worker search policy if unemployed (index in the x_policy grid)
        firm_type::Vector{Float64} # Employer indicator/history (e.g., index from firm remote productivity grid)
        remote_work::Vector{Float64} # Remote work fraction α history (if employed)
        #? Constructor
        function Worker(
                        id::String, skill::Int, x_choice::Int; 
                        employed::Vector{Bool}=[false], wage::Vector{Float64}=[0.0],
                        firm_type::Vector{Float64}=[NaN], remote_work::Vector{Float64}=[NaN]
                    )
            #==========================================================================================
            Create a new `Worker` object.
            #* Arguments
            ##? Required
            - `id::String`: The unique identifier for the worker.
            - `skill::Int`: The skill level of the worker.
            - `x_choice::Int`: Search policy if unemployed. #! Immutable for know (might change in the future)
            ##? Optional
            - `employed::Vector{Bool}`: Employment status history. Default is `[false]` (unemployed).
            - `init_wage::Vector{Float64}`: Wage history. Default is `[0.0]` (no wage if unemployed).
            - `firm_type::Vector{Union{Float64, Missing}}`: Employer indicator/history. Default is `[NaN]` (no employer if unemployed).
            - `remote_work::Vector{Union{Float64, Missing}}`: Remote work fraction α history. Default is `[NaN]` (no remote work if unemployed).
            #* Returns
            - A new `Worker` object.
            ==========================================================================================#
            new(id, skill, 0, employed, wage, x_choice, firm_type, remote_work)
        end
    end
    #==========================================================================================
    #* Economy: Represents the economy in which workers and firms interact.
    # Fields
        - N::Int: Number of workers
        - T::Int: Number of periods to simulate
        - T_burn::Int: Number of periods to burn before collecting data
        - seed::Int: Random seed for simulation reproducibility
        - workers::Vector{Worker}: Vector of workers
        - ψ_grid::Vector{Float64}: Firm remote work efficiency grid (1D) 
        - ψ_pdf::Vector{Float64}: PDF for sampling firm types 
        - x_grid::Vector{Float64}: Utility grid from the 
        - p::Function: Match probability function p(θ)
        - wage_fun::Function: Function to compute wage: w(x, α)
        - δ_bar::Float64: Exogenous job destruction rate #! For now this is fixed at δ_bar (it might change in the future if dynamics are added thenm it will be a model outcome)
        - θ::Array{Float64,2}: Sub-market tightness, indexed by (skill, searched_utility) #> [Model outcome]
        - α_policy::Array{Float64,2}: Optimal remote work fraction as a function of (firm type, worker skill) #>[Model outcome]
        - job_finding::Vector{Float64}: Aggregate job finding rate over time #> [Simulation outcome]
        - unemployment_rate::Vector{Float64}: Aggregate unemployment rate over time #> [Simulation outcome]
        - remote_work::Vector{Float64}: Aggregate remote work fraction over time #> [Simulation outcome]
    ==========================================================================================#
    mutable struct Economy
        #* Simulation parameters 
        N::Int                               # Number of workers
        T::Int                               # Number of periods to simulate
        T_burn::Int                          # Number of periods to burn before collecting data
        seed::Int                            # Random seed
        #* Simulation objects
        workers::Vector{Worker}              # Vector of workers
        #* Model objects
        #? Model primitives
        ψ_grid::Vector{Float64}              # Firm type grid (1D)
        ψ_pdf::Vector{Float64}               # PDF for sampling firm types
        x_grid::Vector{Float64}              # Utility grid from the model
        p::Function                          # Match probability function p(θ)
        δ_bar::Float64                       # Exogenous job destruction rate #! For now this is fixed at δ_bar (it might change in the future if dynamics are added then it will be a model solution)
        #? Model solution
        θ::Array{Float64,2}                  # Sub-market tightness, indexed by (skill, x)
        α_policy::Array{Float64,2}           # Optimal remote work fraction as a function of (firm type, worker skill)
        w_policy::Array{Float64,3}           # Optimal wage as a function of (firm type, worker skill, utility)
        #* Simulation outcomes
        job_finding::Vector{Float64}         # Aggregate job finding rate over time
        unemployment_rate::Vector{Float64}   # Aggregate unemployment rate over time
        remote_work::Vector{Float64}         # Aggregate remote work fraction over time
    end
    #?=========================================================================================
    #? Helper Functions 
    #?=========================================================================================
    #==========================================================================================
    ##* create_primitives_from_yaml(yaml_file::String)::Primitives
        Create a Primitives object by reading parameters from a YAML file.

        This function reads a YAML configuration file and initializes model functions and 
        parameters according to the configuration. It handles the initialization of matching 
        functions, production functions, and utility functions, as well as grid 
        and market parameters.

        The function reads parameters in the same order they appear in the YAML file and passes 
        them to the respective constructors in that same order. This ensures that the parameter 
        order in the YAML file matches the expected order of arguments in the constructors.

    #* Arguments
    - yaml_file::String: Path to the YAML file that contains the parameter values and model 
        specification
    #* Returns
        - Primitives: Model primitives
    ==========================================================================================#
    function create_primitives_from_yaml(yaml_file::String)::Primitives
        #> Load configuration from YAML file
        config = YAML.load_file(yaml_file, dicttype = OrderedDict)
        model_config = config["ModelConfig"]
        #> Extract configuration components
        model_parameters = model_config["ModelParameters"]
        model_grids = model_config["ModelGrids"]
        model_functions = model_config["ModelFunctions"]

        #> Extract model parameters
        #TODO: Validate model_parameters and add defaults if missing
        κ₀    = model_parameters["kappa0"]          # Vacancy posting cost parameter
        κ₁    = model_parameters["kappa1"]          # Vacancy posting cost parameter
        β     = model_parameters["beta"]            # Time discount factor
        δ_bar = model_parameters["delta_bar"]       # Baseline job destruction rate
        b = model_parameters["b"]                   # Unemployment benefit
        ξ = model_parameters["xi"]                  # Worker bargaining power
        #> Extract model grids 
        #TODO: Validate model_grids and add defaults if missing
        #* Remote productivity grid parameters
        n_ψ       =  model_grids["RemoteProductivityGrid"]["n_psi"]                # Number of remote productivity grid points
        ψ_data    =  model_grids["RemoteProductivityGrid"]["data_file"]            # File with the data to construct the grid
        ψ_column  =  model_grids["RemoteProductivityGrid"]["data_column"]          # Column with the data to construct the grid
        ψ_weight  =  model_grids["RemoteProductivityGrid"]["weight_column"]        # Column with the weights for the data
        #* Skill grid parameters
        n_h       =  model_grids["SkillGrid"]["n_h"]                  # Number of skill grid points
        h_data    =  model_grids["SkillGrid"]["data_file"]  
        h_column  =  model_grids["SkillGrid"]["data_column"] 
        h_weight  =  model_grids["SkillGrid"]["weight_column"] 
        #* Utility grid parameters
        n_x       =  model_grids["UtilityGrid"]["n_x"]                  # Number of utility grid points
        x_size    =  model_grids["UtilityGrid"]["x_size"]                # Size of the utility grid
        #? Create grids
        #* Remote productivity grid
        # Compute KDE for ψ
        ψ_grid, ψ_pdf, ψ_cdf = fit_kde_psi(
                                            ψ_data,
                                            ψ_column,
                                            weights_col = ψ_weight, 
                                            num_grid_points=n_ψ,
                                            engine="python"
                                            ) 
        # Construct grid object
        ψ_grid = Grid(ψ_grid, pdf=ψ_pdf, cdf=ψ_cdf)
        #* Skill grid
        #TODO: Revise that the non-paramteric estimation is workign properly
        h_grid, h_pdf, h_cdf = fit_kde_psi(
                                            h_data,
                                            h_column,
                                            weights_col = h_weight, 
                                            num_grid_points=n_h,
                                            engine="python"
                                            ) 
        # h_grid, h_pdf, h_cdf = fit_distribution_to_data(
        #                             h_data,                 # Path to skill data 
        #                             h_column,               # Which column to use
        #                             "functions",            # Will I be returning a vector of values of functions that can be evaluated
        #                             "parametric";           # Parametric or non-Parametric estimation
        #                             distribution=Normal,    # If Parametric which distribution?
        #                             num_grid_points=n_h     # Number of grid points to fit the distribution
        #                         )
        # Construct grid object
        #if the minimum skill value h_min ≤ 0 set it to a small positive value to avoid issues with the log function
        if h_grid[1] <= 0
            h_grid[1] = 1e-6
        end
        h_grid = Grid(h_grid, pdf=h_pdf, cdf=h_cdf)
        #> Extract model functions configurations
        #TODO: Validate model_functions and add defaults if missing
        #? Matching function
        matching_function_config = model_functions["MatchingFunction"]
        matching_function_type = matching_function_config["type"]
        matching_function_params = matching_function_config["params"] |> values |> collect
        # Initialize matching function
        matching_function = create_matching_function(matching_function_type, Float64.(matching_function_params) )
        #? Utility function
        utility_function_config = model_functions["UtilityFunction"]
        utility_function_type = utility_function_config["type"]
        utility_function_params = utility_function_config["params"] |> values |> collect
        # Initialize utility function
        utility_function = create_utility_function(utility_function_type, Float64.(utility_function_params) )
        #? Production function
        production_function_config = model_functions["ProductionFunction"]
        constant = production_function_config["constant"]
        #* Productivity component
        productivity_component_config = production_function_config["ProductivityComponent"]
        productivity_component_type = productivity_component_config["type"]
        productivity_component_params = productivity_component_config["params"] |> values |> collect
        #* Remote efficiency component
        remote_work_component_config = production_function_config["RemoteEfficiencyComponent"]
        remote_work_component_type = remote_work_component_config["type"]
        remote_work_component_params = remote_work_component_config["params"] |> values |> collect
        # Initialize production function
        production_function = create_production_function(
                                                        productivity_component_type,
                                                        remote_work_component_type,
                                                        Float64.(productivity_component_params),
                                                        Float64.(remote_work_component_params),
                                                        constant
                                                    )
        
        
        
        #* Utility grid
        # x_min     =  utility_function.c₀ * utility_function.a₀ #! Hardcoded for specific utility generalize later #TODO
        # x_max     =  x_min + x_size 
        x_values    =  collect(range(start=0.0, stop=x_size, length=n_x)) # Utility grid for search intensity
        # Create a uniform distribution for the utility grid (since it not important)
        x_pdf = fill(1/n_x, n_x)
        x_cdf = cumsum(x_pdf)
        # Construct grid object
        x_grid = Grid(x_values, pdf=x_pdf, cdf=x_cdf)

        #> Create Primitives object
        return Primitives(
                            #> Model functions
                            matching_function,
                            production_function,
                            utility_function,
                            #> Market parameters
                            κ₀,
                            κ₁,
                            β,
                            δ_bar,
                            b,
                            ξ, # Add ξ here
                            #> Grids
                            ψ_grid,
                            h_grid,
                            x_grid # Add x_grid to Primitives
                        )
    end
    #==========================================================================================
    # Initialize model from YAML file returns, primitives and results objects
    #TODO: Add simulation component to initializeModel
    ==========================================================================================#
    function initializeModel(yaml_file::String)::Tuple{Primitives, Results}
        #> Create primitives from YAML file
        prim = create_primitives_from_yaml(yaml_file)
        #> Initialize results (just use contructor)
        res = Results(prim)
        #> Return primitives and results
        return prim, res
    end
    #==========================================================================================
    #* initializeEconomy(prim::Primitives, res::Results; N::Int=1000, T::Int=100, seed::Int=42)
        Initialize an Economy object for simulation based on model solution.
    #* Arguments
        #? Required
        - prim::Primitives: The model primitives
        - res::Results: The model solution results
        #? Optional
        - N::Int: Number of workers to simulate (default: 1000)
        - T::Int: Number of periods to simulate (default: 100)
        - T_burn::Int: Number of periods to burn before collecting data (default: 0)
        - seed::Int: Random seed (default: 42)
    #* Returns
        - Economy: Initialized economy for simulation
    ==========================================================================================#
    function initializeEconomy(prim::Primitives, res::Results;  N::Int=1000, T::Int=100, T_burn::Int=0, seed::Int=42)::Economy
        #* Unpack model primitives and results
        @unpack ψ_grid, ψ_pdf, x_grid, δ_bar, h_grid = prim # Removed p, n_h from prim unpack as p is now in res, n_h from h_grid
        # h_pdf is accessed via h_grid.pdf
        @unpack θ, α_policy, w_policy = res # Removed x_policy from res unpack
        # p is now a scalar in res, q is also a scalar in res.
        # The matching function p(θ) is still in prim.
        
        #* Initialize workers with random skill levels
        Random.seed!(seed)
        # Ensure h_grid.pdf is a Vector{Float64} for Weights
        if !(typeof(h_grid.pdf) <: Vector{Float64})
            throw(ArgumentError("prim.h_grid.pdf must be a Vector{Float64} for worker skill sampling"))
        end
        worker_skills = rand(1:h_grid.n, Weights(h_grid.pdf), N) # Sample worker skills from the distribution
        worker_ids = ["W$(lpad(i, 4, '0'))" for i in 1:N] # Generate worker IDs (e.g., W0001, W0002, ...)
        
        # x_policy is removed from Results, so worker_policy needs to be rethought or removed if not used.
        # For now, let's assume x_choice for Worker is still relevant and needs a default or different logic.
        # If x_choice is tied to a grid that's no longer central, this needs to be addressed.
        # Placeholder: assign a default x_choice (e.g., 1) or remove if Worker struct changes.
        default_x_choice = 1 # Placeholder
        worker_x_choices = fill(default_x_choice, N)

        #? Create array of workers
        workers = [Worker(id, skill, x_choice) for (id, skill, x_choice) in zip(worker_ids, worker_skills, worker_x_choices)]
        #* Initialize time series
        job_finding = zeros(Float64, T)
        unemployment_rate = zeros(Float64, T)
        remote_work = zeros(Float64, T)
        #* Return economy object
        return Economy(
                        N, T, T_burn, seed, workers, 
                        prim.ψ_grid.values, prim.ψ_pdf isa Function ? prim.ψ_pdf.(prim.ψ_grid.values) : prim.ψ_pdf, # ensure ψ_pdf is vector
                        prim.x_grid.values, prim.matching_function, prim.δ_bar, # Pass matching_function from prim
                        res.θ, res.α_policy, res.w_policy, # Pass relevant fields from res
                        job_finding, unemployment_rate, remote_work
                    )
    end
    #?=========================================================================================
    #? Helper Functions
    #?=========================================================================================
    #*=========================================================================================
    #* find_thresholds_and_optimal_remote_policy(prim::Primitives, res::Results)::Tuple{Vector{Float64}, Vector{Float64},Matrix{Float64}}
    #*    Find the thresholds for remote work based on the model functions and parameters.
    #*    Also, compute the optimal remote work policy for each worker type.
    #* Arguments
    #*    - prim::Primitives: Model primitives
    #*=========================================================================================
    #! TODO:CLEAN
    # function find_thresholds_and_optimal_remote_policy(prim::Primitives)::Tuple{Vector{Float64}, Vector{Float64},Matrix{Float64}}
    #     # Unpack model functions
    #     @unpack production_function, utility_function, ψ_grid, h_grid = prim
    #     # Define functions for evaluation
        
    #     # ∂Y/∂α
    #     dY_dα = (h, ψ) -> evaluate_derivative(production_function, :α, h, 0.5, ψ) # Note: Evaluating in α = 0.5 works because the derivative does not depends on α
    #     #! TODO: Make this more robust in the future
    #     # And ∂Π/∂α = ∂Y/∂α - ∂w/∂α = ∂Y/∂α - ∂w/∂α 
    #     # Interior solution means that ∂Π/∂α = 0 ⟹ Φ(h, ψ) = ∂w/∂α
    #     # By the implicit function theorem, if ∂u/∂w ≠ 0 then ∂w/∂α = - ∂u/∂α / ∂u/∂w
    #     #! For this function I have the derivative of the wage function with respect to α and w in general i need to do it numerically
    #     du_dw = (α, h) -> ModelFunctions.evaluate_derivative(utility_function, "w", 0.0, α, h) # derivative of u with respect to w (here the value of w is not important)
    #     du_dα = (α, h) -> ModelFunctions.evaluate_derivative(utility_function, "α", 0.0, α, h) # derivative of u with respect to α (here the value of w is not important)# 
    #     # Define the derivative of the wage function wrt as a function of h and ψ
    #     dw_dα = (α, h) -> - du_dα(α, h) / du_dw(α, h)
    #     # Conditions for interior solution are:
    #     # [-] ∂Y/∂α(h,ψ) > min{ ∂w/∂α(0, h), ∂w/∂α(1, h) }
    #     # [-] ∂Y/∂α(h,ψ) < max{ ∂w/∂α(0, h), ∂w/∂α(1, h) }
    #     # Pre-allocate arrays for thresholds ψ_bottom(h) and ψ_top(h)
    #     # ψ_top = psi_bar.(h_grid.values, Ref(prim))
    #     # ψ_bottom = psi_underline.(h_grid.values, Ref(prim))
    #     ψ_bottom = zeros(h_grid.n)
    #     ψ_top = zeros(h_grid.n)
    #     # Find the admisible range for ψ
    #     # Find thresholds for each skill level
    #     for (i, h_val) in enumerate(h_grid.values)
    #         # Evaluate ∂Y/∂α(h,ψ) - min{ ∂w/∂α(0, h), ∂w/∂α(1, h) } in the minimum and maximum values of ψ
    #         dw_min = min(dw_dα(0.0, h_val), dw_dα(1.0, h_val))
    #         # if (dY_dα(h_val, ψ_grid.min) - dw_min) * (dY_dα(h_val, ψ_grid.max) - dw_min) < 0
    #             # If the product is negative then there is a solution in the interval
    #             # Define the objective function to solve for the threshold
    #             ψ_bottom_obj = (ψ) -> dY_dα(h_val, ψ) - dw_min
    #             # Find the zero of the objective function Bisecting the [ψ_min, ψ_max] interval,
    #             # Bisection will work since conditions guarantee that wage is monotonic in α
    #             ψ_bottom[i] = find_zero(ψ -> ψ_bottom_obj(ψ), (-Inf, Inf), Bisection())
    #         # else
    #             # If the product is positive then there is no solution in the interval 
    #             # The interpretation is that there is no value of ψ ∈ [ψ_min, ψ_max] that satisfies the condition
    #             # This means that there all firms offer full in person work to workers with skill level h
    #             # This is equivalent to setting:
    #             # ψ_bottom[i] = Inf # Set to a value ψ > ψ_max
    #         # end
    #         # Evaluate ∂Y/∂α(h,ψ) - max{ ∂w/∂α(0, h), ∂w/∂α(1, h) } in the minimum and maximum values of ψ
    #         dw_max = max(dw_dα(0.0, h_val), dw_dα(1.0, h_val))
    #         # if (dY_dα(h_val, ψ_grid.min) - dw_max) * (dY_dα(h_val, ψ_grid.max) - dw_max) < 0
    #             # If the product is negative then there is a solution in the interval
    #             # Define the objective function to solve for the threshold
    #             ψ_top_obj = (ψ) -> dY_dα(h_val, ψ) - dw_max
    #             # Find the zero of the objective function Bisecting the [ψ_min, ψ_max] interval,
    #             # Bisection will work since conditions guarantee that wage is monotonic in α
    #             ψ_top[i] = find_zero(ψ -> ψ_top_obj(ψ), (-Inf, Inf), Bisection())
    #         # else
    #             # If the product is positive then there is no solution in the interval 
    #             # The interpretation is that there is no value of ψ ∈ [ψ_min, ψ_max] that satisfies the condition
    #             # This means that all firms offer full remote work to workers with skill level h
    #             # This is equivalent to setting:
    #             # ψ_top[i] = Inf 
    #         # end
    #     end
    #     # Pre-allocate array for optimal remote work policy α_policy(ψ, h)
    #     α_policy = zeros(ψ_grid.n, h_grid.n)
    #     # Find the optimal remote work policy for each firm type and skill level
    #     for (i_ψ, ψ_val) in enumerate(ψ_grid.values)
    #         for (i_h, h_val) in enumerate(h_grid.values)
    #             # Define the objective function for the optimal remote work policy
    #             α_obj = (α) -> dY_dα(h_val, ψ_val) - dw_dα(α, h_val)
    #             # If ψ ≤ ψ_bottom(h) then α = 0
    #             if ψ_val ≤ ψ_bottom[i_h]
    #                 α_policy[i_ψ, i_h] = 0.0
    #             # If ψ ≥ ψ_top(h) then α = 1
    #             elseif ψ_val ≥ ψ_top[i_h]
    #                 α_policy[i_ψ, i_h] = 1.0
    #             else
    #                 # Otherwise, find the zero of the objective function
    #                 try
    #                     α_policy[i_ψ, i_h] = find_zero(α -> α_obj(α), (0.0, 1.0), Bisection())
    #                 catch e
    #                     if isa(e, ArgumentError)
    #                         # Assign NaN if no solution found
    #                         α_policy[i_ψ, i_h] = NaN
    #                     else
    #                         rethrow(e)
    #                     end
    #                 end
    #             end
                
    #         end
    #     end
    #     return ψ_bottom, ψ_top, α_policy
    # end
    #! Modified version of the function to find thresholds and optimal remote work policy
    function find_thresholds_and_optimal_remote_policy(prim::Primitives)
        # Unpack for clarity
        @unpack production_function, utility_function, ψ_grid, h_grid = prim

        # --- 1. Define Helper Functions for MB and MC ---
        
        # Marginal Benefit of remote work (from production)
        # This is robust as long as ∂Y/∂α is independent of α.
        MB = (h, ψ) -> evaluate_derivative(production_function, :α, h, 0.5, ψ)

        # Marginal Cost of remote work (from worker's MRS)
        MC = (α, h) -> -evaluate_derivative(utility_function, "α", 0.0, α, h) / evaluate_derivative(utility_function, "w", 0.0, α, h)

        # --- 2. Calculate Thresholds Numerically ---
        ψ_bottom = zeros(h_grid.n)
        ψ_top = zeros(h_grid.n)

        for (i, h_val) in enumerate(h_grid.values)
            # --- Calculate ψ_bottom(h) ---
            # This is where MB = MC(at α=0).
            mc_at_zero = MC(0.0, h_val)
            ψ_bottom_obj = (ψ) -> MB(h_val, ψ) - mc_at_zero
            
            # Check if a solution exists within the grid bounds
            if ψ_bottom_obj(ψ_grid.min) * ψ_bottom_obj(ψ_grid.max) < 0
                ψ_bottom[i] = find_zero(ψ_bottom_obj, (ψ_grid.min, ψ_grid.max), Bisection())
            elseif ψ_bottom_obj(ψ_grid.min) > 0 # MB is always > MC(0)
                ψ_bottom[i] = -Inf # All firms offer α > 0
            else # MB is always < MC(0)
                ψ_bottom[i] = Inf # All firms offer α = 0
            end

            # --- Calculate ψ_top(h) ---
            # This is where MB = MC(at α=1).
            mc_at_one = MC(1.0, h_val)
            ψ_top_obj = (ψ) -> MB(h_val, ψ) - mc_at_one

            # Check if a solution exists within the grid bounds
            if ψ_top_obj(ψ_grid.min) * ψ_top_obj(ψ_grid.max) < 0
                ψ_top[i] = find_zero(ψ_top_obj, (ψ_grid.min, ψ_grid.max), Bisection())
            elseif ψ_top_obj(ψ_grid.min) > 0 # MB is always > MC(1)
                ψ_top[i] = -Inf # All firms offer α = 1
            else # MB is always < MC(1)
                ψ_top[i] = Inf # No firms offer α = 1
            end
        end

        # --- 3. Calculate Optimal Policy Matrix ---
        α_policy = zeros(h_grid.n, ψ_grid.n) # Swapped order to (h, ψ) for clarity

        for (i_h, h_val) in enumerate(h_grid.values)
            for (i_ψ, ψ_val) in enumerate(ψ_grid.values)
                
                if ψ_val <= ψ_bottom[i_h]
                    α_policy[i_h, i_ψ] = 0.0
                elseif ψ_val >= ψ_top[i_h]
                    α_policy[i_h, i_ψ] = 1.0
                else
                    # Solve for interior α: MB(ψ) - MC(α) = 0
                    foc_obj = (α) -> MB(h_val, ψ_val) - MC(α, h_val)
                    # The bracket (0,1) is safe because we are in the interior region
                    α_policy[i_h, i_ψ] = find_zero(foc_obj, (0.0, 1.0), Bisection())
                end
            end
        end
        
        # --- 4. Return Results ---
        return ψ_bottom, ψ_top, α_policy'
    end
    #==========================================================================================
    #* modify_parameter!(prim::Primitives, parameter_path::String, new_value::Any)
        Modify a parameter in the Primitives struct using a dot-notation path.

        This function allows modification of both top-level parameters (e.g., "ξ", "β") 
        and nested parameters (e.g., [:utility_function, :c₀], ["production_function", "constant"]).

    #* Arguments
        - prim::Primitives: The primitives struct to modify
        - parameter_path::Vector{Union{Symbol, String}}: Array of symbols or strings representing the parameter path
        - new_value::Any: The new value to assign to the parameter
    #* Returns
        - nothing: The function modifies the struct in place and does not return a value.
        
    #* Examples
        ```julia
        # Modify top-level parameter
        modify_parameter!(prim, [:ξ], 0.5) => Modify the bargaining power parameter from whatever it was to 0.5
        modify_parameter!(prim, [:β], 0.95) => Modify the discount factor parameter from whatever it was to 0.95

        # Modify nested parameters
        modify_parameter!(prim, [:utility_function, :c₀], 1.2) => Modify the c₀ parameter in the utility function
        modify_parameter!(prim, ["production_function", "remote_efficiency", :ν], 0.7) => Modify the ν parameter in the remote efficiency component of the production function
        modify_parameter!(prim, ["matching_function", :γ], 0.8) => Modify the γ parameter in the matching function
        ```
    ==========================================================================================#
    function modify_parameter!(
                                prim::Primitives,
                                parameter_path::Vector{<:Union{Symbol, String}},
                                new_value::Any
                            )
        # Ensure parameter_path is a vector of strings or symbols
        if !(all(x -> x isa Symbol || x isa String, parameter_path))
            throw(ArgumentError("parameter_path must be a vector of strings or symbols"))
        end
        # Convert all elements to symbols for consistency
        parameter_path = map(x -> x isa Symbol ? x : Symbol(x), parameter_path)
        # Check if the parameter path is valid
        if isempty(parameter_path)
            throw(ArgumentError("parameter_path cannot be empty"))
        end

        if length(parameter_path) == 1
            # Top-level parameter
            param_name = parameter_path[1]
            if hasfield(Primitives, param_name)
                setfield!(prim, param_name, new_value)
            else
                throw(ArgumentError("Parameter '$(path_parts[1])' not found in Primitives struct"))
            end
        elseif length(parameter_path) > 1
            # Nested parameter (e.g., [:utility_function, :c₀])
            current_field = prim
            total_depth = length(parameter_path)
            for i in 1:total_depth
                if i < total_depth
                    # Intermediate element, ensure it exists
                    if !hasfield(typeof(current_field), parameter_path[i])
                        throw(ArgumentError("Parameter '$(parameter_path[i])' not found in $(typeof(current_field))"))
                    end
                    current_field = getfield(current_field, parameter_path[i])
                else
                    if hasfield(typeof(current_field), parameter_path[i])
                        # Last element, set the value
                        setfield!(current_field, parameter_path[i], new_value)
                    else
                        throw(ArgumentError("Parameter '$(parameter_path[i])' not found in $(typeof(current_field))"))
                    end
                end
            end
        end
        
        return nothing
    end    
end # module Types