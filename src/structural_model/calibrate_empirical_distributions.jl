#==========================================================================================
Title: Generate Worker and Firm Type Distributions
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2025-02-19
Description:

#> fit_kde_psi(ψ, num_grid_points; n_points=100, bandwidth=1, boundary=false)::(Array{Float64,1}, Array{Float64,1}, Array{Float64,1})
    Fits a kernel density estimation (KDE) for the given sample `ψ` and constructs a 
    corresponding distribution DataFrame.
    using Base: return_types
using CSV: pass
# Arguments
    - `ψ::AbstractVector{<:Real}`: The input data sample for which the density is to be estimated.
    - `num_grid_points`: A parameter intended to specify the number of grid points (currently not used in the implementation).
    - `n_points::Integer=100`: The number of points to use in constructing the KDE grid.
    - `bandwidth`: The bandwidth parameter for the KDE (default is `1`).
    - `boundary`: Either `false` (default) or a tuple `(min_x, max_x)` that explicitly defines the boundaries over which the 
    KDE should be computed. If not provided, the minimum and maximum of `ψ` are used.
    # Returns
    A `DataFrame` with the following columns:
    - `:ψ_grid`: The grid points at which the density is estimated.
    - `:ψ_pdf`: The normalized probability density function (PDF) values such that the sum of the PDF equals 1.
    - `:ψ_cdf`: The cumulative distribution function (CDF), computed as the cumulative sum of the PDF values.
    # Details
    This function carries out a KDE on `ψ` using the specified parameters and computes a normalized PDF 
        over a defined grid. The cumulative density (CDF) is then determined by cumulatively summing the PDF values. 
        Note that while `num_grid_points` is included in the function signature, only `n_points` is used to define the 
            grid resolution in the current implementation.
#> fit_distribution_h()        
==========================================================================================#
using PyCall, Roots, DataFrames, CSV, Distributions

function fit_kde_psi(data_path::String, data_col::String; weights_col::String="",
                    num_grid_points::Int=100, bandwidth::Int=1, boundary::Bool=false, engine::String= "julia") 

    # data_path = "/project/high_tech_ind/WFH/WFH/data/results/psi_distribution.csv"
    # data_col = "psi"
    # weights_col = "probability_mass"
    # boundary = false
    # engine = "python"
    # num_grid_points = 100
    # Read data
    data = CSV.read(data_path, DataFrame)
    ψ = data[!, data_col]
    if weights_col == ""
        weights = ones(length(ψ))
    else
        weights = data[!, weights_col]
    end
    # Fit KDE with specified grid points
    if boundary
        min_x, max_x = boundary
    else
        min_x = minimum(ψ)
        max_x = maximum(ψ)
    end

    if engine == "julia"
        kde_result = kde(
                    ψ,
                    boundary = (min_x, max_x),
                    bandwidth=bandwidth,
                    npoints=num_grid_points
                )
        # Extract grid and density values
        ψ_grid = kde_result.x
        estimated_density = kde_result.density
        # Normalize probabilities to sum to 1
        probabilities = estimated_density ./ sum(estimated_density)
    elseif engine == "python"
        # --- Step 1: Ensure required Python packages are available
        pyimport_conda("scipy.stats", "scipy")
        pyimport_conda("numpy", "numpy")        
        # Import Python libraries
        scipy_stats = pyimport("scipy.stats")
        np = pyimport("numpy")
        # --- Step 2: Convert Julia Vector to NumPy ---
        psi_values_np = np.array(ψ)
        weights_np = np.array(weights)
        # --- Step 3: Run the Python KDE ---
        kde_result = scipy_stats.gaussian_kde(psi_values_np, weights=weights_np)
        kde_grid = np.linspace(np.min(psi_values_np), np.max(psi_values_np), num_grid_points)
        estimated_density = kde_result.evaluate(kde_grid)
        prob = estimated_density / np.sum(estimated_density)  # Ensure sum to 1
        # --- Step 4: Convert Python Output to Julia ---
        ψ_grid = Vector{Float64}(kde_grid)
        probabilities = Vector{Float64}(prob)
    end
    ψ_pdf = probabilities
    # Add cdf
    ψ_cdf = cumsum(ψ_pdf)
    return ψ_grid, ψ_pdf, ψ_cdf
end


function fit_distribution_to_data(data_path::String, data_col::String, return_types::String, method::String;
                                    weights_col::String="", num_grid_points::Int=100, distribution=Normal)
    
    # Read data
    data = CSV.read(data_path, DataFrame)
    values = data[!, data_col]
    if weights_col == ""
        weights = ones(length(values))
    else
        weights = data[!, weights_col]
    end
    if method == "kde"
        #TODO: Join with previous function to have a single function that handles both cases
    elseif method == "parametric"
        # Create grid
        _grid = range(minimum(values), stop=maximum(values), length=num_grid_points) |> collect
        parametric_dist = fit_mle(distribution, values, weights)
    else
        error("Invalid method specified, should be 'kde' or 'parametric'.")
    end

    if return_types == "vector"
        if method == "kde"
            # if method is kde, return the grid, pdf and cdf vectors
            #TODO: Join with previous function to have a single function that handles both cases
        else
            # Evaluate pdf and cdf in the grid
            _pdf = pdf(parametric_dist, _grid)
            _cdf = cdf(parametric_dist, _grid)
            return _grid, _pdf, _cdf
        end
    elseif return_types == "functions"
        if method == "kde"
            # If method is kde, interpolate the pdf and cdf functions using bsplines
            #TODO: Join with previous function to have a single function that handles both cases
        else
            # Create functions for pdf and cdf
            _pdf = x -> pdf(parametric_dist, x)
            _cdf = x -> cdf(parametric_dist, x)
            return _grid, _pdf, _cdf
        end
    else
        error("Invalid return type specified, should be 'vector' or 'functions'.")
    end
end