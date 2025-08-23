# SMM Estimation Framework for Heterogeneous Preferences Model

This directory contains the improved SMM estimation framework with flexible moment selection and efficient computation.

## Key Improvements

### 1. Enhanced `compute_model_moments` Function

- **Fixed production function**: Now uses `exp(ν*ψ + ϕ*h)` for consistency
- **Added new share calculations**: `inperson_share`, `hybrid_share`, `remote_share`
- **Added unconditional wage premium**: `diff_logwage_high_lowpsi` for parameter identification
- **Flexible moment selection**: Use `include` parameter to compute only needed moments

### 2. Moment Selection Utilities (`moment_selection_utils.jl`)

- **Core identification moments**: Minimal set for parameter identification
- **Auxiliary moments**: Additional moments for robustness checks
- **Flexible selection**: Include/exclude moments as needed
- **Validation**: Ensures target moments are available

### 3. Complete Estimation Pipeline (`run_estimation.jl`)

- **Flexible setup**: Choose core moments, all moments, or custom subsets
- **Parameter bounds**: Reasonable constraints for optimization
- **Multiple optimizers**: L-BFGS-B with bounds or Nelder-Mead
- **Results validation**: Compare model vs target moments

## Parameter-Moment Identification Strategy

| Parameter(s) | Identifying Moment(s) | Description |
|:-------------|:---------------------|:------------|
| `aₕ`, `bₕ` | `mean_logwage`, `var_logwage` | Skill distribution |
| `c₀` | `diff_logwage_inperson_remote` | Compensating differential |
| `χ` | `remote_share` | Remote preference scale |
| `μ` | `hybrid_share` | Gumbel scale parameter |
| `ψ₀`, `ν` | `diff_logwage_high_lowpsi` | Production parameters (RH subsample) |
| `ϕ` | `diff_alpha_high_lowpsi` | Skill-remote interaction |
| `κ₀` | `market_tightness` | Search parameter |

## Quick Start

### 1. Basic Estimation (Recommended)
```julia
include("run_estimation.jl")

# Run estimation with core moments only
result, problem = main_estimation_example()
```

### 2. Robustness Check
```julia
# Run estimation with all available moments
result, problem = main_estimation_example(moment_selection=:all)
```

### 3. Custom Moment Selection
```julia
include("moment_selection_utils.jl")

# Select specific moments
selected_moments = select_moments_for_estimation(
    core_only = true,
    additional_moments = [:mean_alpha, :var_alpha],
    exclude_moments = [:market_tightness]
)

# Setup problem with custom moments
problem = setup_estimation_problem(
    "model_parameters.yaml", 
    target_moments;
    moment_selection = :core_only,
    additional_moments = [:mean_alpha, :var_alpha],
    exclude_moments = [:market_tightness]
)
```

### 4. Efficient Moment Computation
```julia
# Compute only the moments you need
core_moments = select_moments_for_estimation(core_only=true)
model_moments = compute_model_moments(prim, res; include=core_moments)
```

## Files Overview

- **`ModelEstimation.jl`**: Enhanced moments computation with flexible selection
- **`moment_selection_utils.jl`**: Utilities for moment selection and validation
- **`run_estimation.jl`**: Complete SMM estimation pipeline
- **`estimation_examples.jl`**: Comprehensive examples and demonstrations
- **`profile_run.jl`**: Performance profiling script

## Usage Examples

See `estimation_examples.jl` for comprehensive examples including:

1. **Core moments only**: Minimal identification strategy
2. **Core + additional**: Adding robustness moments
3. **Excluding moments**: Handle missing empirical moments
4. **Efficient computation**: Speed optimization techniques
5. **Moment validation**: Ensure data compatibility

## Next Steps

1. **Replace example target moments** with your empirical estimates
2. **Run estimation** using the provided scripts
3. **Validate results** by checking moment fit and parameter reasonableness
4. **Robustness checks** using auxiliary moments

## Notes

- The framework automatically handles parameter bounds and optimization constraints
- Model moments are computed efficiently using numerical integration
- Caching is implemented for warm starts during optimization
- All functions support ForwardDiff for automatic differentiation
