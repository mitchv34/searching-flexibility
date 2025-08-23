# Counterfactual Analysis Documentation

This directory contains the implementation of counterfactual experiments for the searching flexibility model. The analysis decomposes changes between 2019 and 2024 into preference and technology components, and explores policy implications.

## Files Overview

### Main Scripts
- `run_counterfactuals.jl` - Main entry point that orchestrates all experiments
- `complementarity_analysis.jl` - Implements the complementarity grid experiment (CF 4)
- `rto_analysis.jl` - Implements the return-to-office mandate experiment (CF 5)
- `plotting_utils.jl` - Visualization functions for all experiments
- `solver_extensions.jl` - Modified solver functions for constrained optimization

### Experiments Implemented

#### 1. Decomposition Experiments (CF 1-3)
These experiments decompose the total change between 2019 and 2024 into components due to:
- **Preferences**: Changes in worker preferences for remote work (c₀, χ, μ)
- **Technology**: Changes in technology parameters (A₀, A₁, ψ₀, ν, ϕ)

**Outcomes analyzed:**
- Mean remote work share E[α]
- Aggregate productivity
- Wage inequality Var(log w)

#### 2. Complementarity Analysis (CF 4)
Explores how complementarity (ϕ) and technology (ν) parameters interact by:
- Creating a grid around their 2024 estimated values
- Re-calibrating κ₀ to maintain constant unemployment rate
- Computing outcomes across the parameter space

**Key insight**: Tests whether technology and complementarity are substitutes or complements in determining remote work adoption.

#### 3. Return-to-Office Mandate (CF 5)
Simulates policy interventions that cap maximum remote work at different levels:
- Constrains α ≤ α_max for various α_max values
- Measures impact on productivity and inequality
- Provides welfare analysis of RTO policies

## Usage

### Basic Usage
```julia
# Navigate to the counterfactuals directory
cd("src/structural_model_heterogenous_preferences/counterfactuals")

# Run all experiments
include("run_counterfactuals.jl")
```

### Customization
You can modify the experiments by editing the main script:

```julia
# Skip computationally intensive experiments
run_advanced = false

# Customize grid sizes for complementarity analysis
complementarity_results = run_complementarity_experiment(
    prim_2024, res_2024;
    phi_grid_size=5,     # Increase for finer grid
    nu_grid_size=5,
    phi_range=0.2,       # Expand search range
    nu_range=0.2
)

# Customize RTO constraint levels
rto_results = run_rto_experiment(
    prim_2024, res_2024;
    alpha_max_values=[0.2, 0.4, 0.6, 0.8, 0.9]
)
```

## Prerequisites

### Required Files
1. **Estimated Parameters**: 
   - `output/estimation_results/params_2019.yaml`
   - `output/estimation_results/params_2024.yaml`

2. **Model Files**:
   - `src/structural_model_heterogenous_preferences/ModelSetup.jl`
   - `src/structural_model_heterogenous_preferences/ModelSolver.jl`
   - `src/structural_model_heterogenous_preferences/ModelEstimation.jl`

### Required Functions
The following functions must be available in your model files:

```julia
# Model initialization and solving
initializeModel(config_path)
update_params_and_resolve(prim, res; params_to_update)
solve_model(prim; kwargs...)

# Moment calculation
compute_model_moments(prim, res)
```

### Expected Moment Structure
The `compute_model_moments` function should return a dictionary/NamedTuple with:
- `:mean_alpha` - Mean remote work share
- `:agg_productivity` - Aggregate productivity measure
- `:var_logwage` - Variance of log wages

## Output Structure

### Results Directory
```
results/
├── decomposition_results.csv
├── complementarity_results.csv
├── rto_results.csv
└── plots/
    ├── decomposition_results.png
    ├── complementarity_mean_alpha.png
    ├── complementarity_productivity.png
    └── rto_mandate_results.png
```

### Key Result Files

#### `decomposition_results.csv`
| Column | Description |
|--------|-------------|
| Outcome | Variable name (E[alpha], Productivity, Var(log w)) |
| TotalChange | Total change from 2019 to 2024 |
| DueToPreferences | Change attributable to preference shifts |
| DueToTechnology | Change attributable to technology shifts |

#### `complementarity_results.csv`
| Column | Description |
|--------|-------------|
| phi, nu | Parameter values on the grid |
| mean_alpha | Mean remote work share |
| agg_productivity | Aggregate productivity |
| var_logwage | Wage inequality |
| recalibrated_kappa | Re-calibrated κ₀ value |

#### `rto_results.csv`
| Column | Description |
|--------|-------------|
| alpha_max | Maximum allowed remote work share |
| mean_alpha | Realized mean remote work share |
| change_* | Changes relative to unconstrained baseline |

## Technical Notes

### Solver Modifications for RTO Experiment
The RTO experiment requires modifying the integration bounds in the surplus calculation. The key changes are:

1. **Constrained Integration**: Integration bounds change from [0,1] to [0,α_max]
2. **Distribution Truncation**: The α distribution is truncated and renormalized
3. **Equilibrium Recalculation**: All equilibrium objects are recalculated under the constraint

### Computational Considerations
- **Complementarity Experiment**: Scales as O(n_phi × n_nu) with nested optimization
- **Memory Usage**: Large grids may require significant memory for storing results
- **Convergence**: Some parameter combinations may fail to converge

### Troubleshooting
1. **Convergence Issues**: Reduce grid size or parameter ranges
2. **Memory Issues**: Process results in batches or reduce grid resolution
3. **Missing Functions**: Ensure all required model functions are implemented

## Extension Points

### Adding New Experiments
1. Create a new analysis file (e.g., `new_experiment.jl`)
2. Implement experiment-specific functions
3. Include the file in `run_counterfactuals.jl`
4. Add plotting functions in `plotting_utils.jl`

### Custom Constraints
Modify `solver_extensions.jl` to implement different types of constraints:
- Skill-specific constraints
- Industry-specific policies
- Progressive constraint schedules

### Alternative Decompositions
The framework can be extended to decompose other parameter groups:
- Institutional vs. technological changes
- Sector-specific vs. economy-wide changes
- Supply vs. demand factors
