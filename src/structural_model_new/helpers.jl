# ==============================================================================
# Title: Lightweight helpers for empirical grid construction
# Author: Mitchell Valdes-Bobes @mitchv34 (refactor by Copilot)
# Date: 2025-08-13
# Description: Lean, fast, dependency-minimal KDE-based grid builder for ψ.
# Notes:
#  - Pure Julia implementation of a weighted Gaussian KDE (no PyCall / external engines).
#  - Bandwidth uses Silverman's rule with weighted variance and effective sample size.
#  - Returns discrete pdf normalized to sum to 1 and its cdf on the constructed grid.
# ==============================================================================

using CSV, DataFrames

"""
    fit_kde_psi(data_path::AbstractString, data_col::AbstractString;
                weights_col::AbstractString = "",
                num_grid_points::Int = 100,
                bandwidth::Union{Nothing, Real} = nothing,
                boundary::Union{Nothing, Tuple{<:Real, <:Real}} = nothing)

Construct a KDE-based discretized distribution for ψ from a CSV file.

Arguments
- data_path: Path to CSV file with ψ data.
- data_col: Column name in the CSV containing ψ values.
- weights_col: Optional column name with observation weights (defaults to equal weights).
- num_grid_points: Number of points in the returned grid (default 100).
- bandwidth: Optional kernel bandwidth; if not provided, uses Silverman's rule (weighted).
- boundary: Optional (min, max) tuple to bound the grid; defaults to data range.

Returns
- (ψ_grid::Vector{Float64}, ψ_pdf::Vector{Float64}, ψ_cdf::Vector{Float64})

Implementation details
- Uses a weighted Gaussian kernel: K(u) = (2π)^(-1/2) exp(-u^2/2).
- Bandwidth h via Silverman: h = 1.06 * σ * n_eff^(-1/5),
  where σ is the weighted std and n_eff = (∑w)^2 / ∑w^2.
- Discrete pdf is normalized to sum to 1, and cdf is its cumulative sum.
"""
function fit_kde_psi(data_path::AbstractString, data_col::AbstractString;
                        weights_col::AbstractString = "",
                        num_grid_points::Int = 100,
                        bandwidth::Union{Nothing, Real} = nothing,
                        boundary::Union{Nothing, Tuple{<:Real, <:Real}} = nothing)

    # Load data
    df = CSV.read(data_path, DataFrame)
    ψ = Float64.(df[!, data_col])
    weights = weights_col == "" ? ones(Float64, length(ψ)) : Float64.(df[!, weights_col])
    @assert length(ψ) == length(weights) "ψ and weights must have the same length"

    # Boundaries for the grid
    if boundary === nothing
        min_x = minimum(ψ)
        max_x = maximum(ψ)
        if min_x == max_x
            # Expand degenerate range slightly to avoid zero-length grid
            ϵ = max(abs(min_x), 1.0) * 1e-6
            min_x -= ϵ; max_x += ϵ
        end
    else
        min_x, max_x = boundary
    end

    # Construct grid
    ψ_grid = collect(range(min_x, max_x; length = num_grid_points))

    # Weighted statistics for bandwidth
    W = sum(weights)
    μ = sum(weights .* ψ) / W
    σ2 = sum(weights .* (ψ .- μ).^2) / W
    σ = sqrt(max(σ2, eps()))
    n_eff = (W^2) / sum(weights .^ 2)
    h = bandwidth === nothing ? 1.06 * σ * n_eff^(-1/5) : float(bandwidth)
    h = max(h, 1e-12)

    inv_norm = 1.0 / (h * sqrt(2π) * W)

    # Evaluate weighted Gaussian KDE on the grid
    ψ_pdf = similar(ψ_grid, Float64)
    @inbounds for j in eachindex(ψ_grid)
        x = ψ_grid[j]
        s = 0.0
        for i in eachindex(ψ)
            z = (x - ψ[i]) / h
            s += weights[i] * exp(-0.5 * z * z)
        end
        ψ_pdf[j] = inv_norm * s
    end

    # Discrete normalization to ensure sum(ψ_pdf) == 1 on the grid
    total = sum(ψ_pdf)
    if !(total > 0)
        # Fallback: place mass uniformly if density is numerically degenerate
        fill!(ψ_pdf, 1.0 / length(ψ_pdf))
    else
        @. ψ_pdf = ψ_pdf / total
    end

    ψ_cdf = cumsum(ψ_pdf)
    return ψ_grid, ψ_pdf, ψ_cdf
end
