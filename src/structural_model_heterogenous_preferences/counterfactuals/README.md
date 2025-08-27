# Counterfactual Experiments

This directory contains scripts for running counterfactual experiments with the structural model. Each experiment is designed to test specific economic hypotheses and policy scenarios.

## Directory Structure

```         
counterfactuals/
├── counterfactual_config.yaml           # Central configuration file
├── run_all_experiments.jl               # Master runner script
├── update_estimated_parameters.jl       # Utility to update baseline parameters
├── experiment_1_no_remote_work.jl       # Experiment 1: No remote work technology
├── experiment_2_remote_tech_levels.jl   # Experiment 2: Different tech levels
├── experiment_3_remote_mandate.jl       # Experiment 3: Remote work mandates
├── experiment_4_search_frictions.jl     # Experiment 4: Search friction variations
├── experiment_5_unemployment_benefits.jl # Experiment 5: Unemployment benefit levels
├── experiment_6_bargaining_power.jl     # Experiment 6: Worker bargaining power
├── test_infrastructure.jl              # Infrastructure testing script
└── results/                            # Output directory for results
```

## Baseline Model: Estimated Parameters

All counterfactual experiments use **estimated parameters from 2024 data** as the baseline "true" model. This ensures that counterfactuals reflect realistic policy changes from an empirically grounded starting point.

### Baseline Configuration
- **Parameter Source**: `data/results/estimated_parameters/estimated_parameters_2024.yaml`
- **Estimation Method**: Method of moments with MPI distributed search + local refinement
- **Data Year**: 2024
- **Status**: Contains latest parameter estimates from optimization runs

### Updating Estimated Parameters

When new parameter estimates become available, update the baseline using:

```bash
# Auto-update from latest optimization results
julia update_estimated_parameters.jl --auto

# Manual update from specific CSV file
julia update_estimated_parameters.jl --csv output/best_parameters_20250827_140128.csv

# With additional metadata
julia update_estimated_parameters.jl --csv output/best_parameters_20250827_140128.csv \
  --objective 0.00142 --notes "Final estimates after convergence"
```

## Experiments Overview

### Experiment 1: No Remote Work Technology

-   **Purpose**: Measures the impact of remote work technology by eliminating it
-   **Parameters**: Sets $\psi_0 = 0$ and $\nu = 0$
-   **Key Insights**: Baseline comparison to understand remote work importance

### Experiment 2: Remote Work Technology Levels

-   **Purpose**: Tests sensitivity to different remote work productivity levels
-   **Parameters**: Sweeps over $\psi_0$ and $\nu$ values
-   **Modes**: Can run as parallel (paired values) or grid (all combinations)
-   **Key Insights**: Technology adoption and productivity effects

### Experiment 3: Remote Work Mandate

-   **Purpose**: Policy experiment with mandatory minimum remote work levels
-   **Parameters**: Uses $\alpha_\text{mandated}$ constraint levels from 0% to 100%
-   **Special**: Requires `solve_model_rto` function ==(to be implemented)==
-   **Key Insights**: Policy compliance and economic distortions

### Experiment 4: Search Friction Variations

-   **Purpose**: Impact of different search market frictions
-   **Parameters**: Varies $\kappa_0$ (vacancy costs) and $\kappa_1$ (elasticity)
    -   Since $\kappa_1$ was calibrated not estimated then changing it may need careful thinking.
-   **Key Insights**: Labor market efficiency and matching

### Experiment 5: Unemployment Benefit Levels

-   **Purpose**: Welfare policy impact on labor markets
-   **Parameters**: Sweeps unemployment benefit $b$ levels
-   **Key Insights**: Moral hazard vs insurance effects

### Experiment 6: Worker Bargaining Power

-   **Purpose**: Distribution of match surplus between workers and firms
-   **Parameters**: Varies worker share $\xi$ from 10% to 90%
-   **Key Insights**: Wage inequality and market tightness

## Configuration

All experiments are controlled through `counterfactual_config.yaml`. Key sections:

-   **Base Model**: Points to main model parameters
-   **Experiment Configs**: Individual experiment settings
-   **Common Settings**: Solver options, output formats, parallel settings

### Example Configuration Modification

To test different remote work technology levels:

``` yaml
Experiment2_RemoteTechLevels:
  parameter_sweeps:
    psi_0: [0.5, 0.7, 0.9, 1.1]  # Modify these values
    nu: [0.2, 0.3, 0.4, 0.5]     # Modify these values
  sweep_mode: "parallel"          # or "grid"
```

## Usage

### Running Individual Experiments

``` bash
# Run specific experiment
julia experiment_1_no_remote_work.jl

# Run with custom config
julia experiment_2_remote_tech_levels.jl custom_config.yaml
```

### Running Multiple Experiments

``` bash
# Run all experiments
julia run_all_experiments.jl

# Run specific experiments
julia run_all_experiments.jl --experiments 1,2,3

# Run with verbose output
julia run_all_experiments.jl --experiments 2,4 --verbose

# Use custom config
julia run_all_experiments.jl --config my_config.yaml --experiments all
```

### Command Line Options

-   `--experiments, -e`: Which experiments to run (1,2,3,4,5,6 or "all")
-   `--config, -c`: Path to configuration file (default: counterfactual_config.yaml)
-   `--verbose, -v`: Verbose output for debugging
-   `--parallel, -p`: Enable parallel processing (where supported)

## Output

Results are saved in the `results/` directory with timestamps:

-   **YAML files**: Complete detailed results with all metadata
-   **CSV files**: Summary tables for analysis in R/Python/Excel

### Output Files

```         
results/
├── exp1_no_remote_20250827_143022.yaml
├── exp1_no_remote_20250827_143022.csv
├── exp2_remote_tech_20250827_143234.yaml
├── exp2_remote_tech_20250827_143234.csv
└── ...
```

## Key Features

### Parameter Sweeps

-   **Parallel Mode**: Parameters paired one-to-one
-   **Grid Mode**: All parameter combinations tested
-   Configurable through YAML without code changes

### Error Handling

-   Robust convergence checks
-   Failed runs are recorded with error messages
-   Experiments continue even if some parameter combinations fail

### Results Comparison

-   Automatic percentage change calculations
-   Baseline vs counterfactual comparisons
-   Summary statistics across parameter ranges

### Extensibility

-   Easy to add new experiments by copying existing templates
-   Configuration-driven parameter modifications
-   Modular design using shared model components

## Implementation Notes

### Experiment 3 Special Case

Experiment 3 uses a placeholder `solve_model_rto` function that needs to be implemented to handle remote work mandates. The function signature is:

``` julia
function solve_model_rto(prim, res; alpha_mandated=0.0, kwargs...)
    # TODO: Implement constrained solver
    # Should enforce alpha(h,psi) >= alpha_mandated
end
```

### Dependencies

All experiments use: - `../ModelSetup.jl`: Model structure definitions - `../ModelSolver.jl`: Solution algorithms\
- `../ModelEstimation.jl`: Moment computation and utilities

### Performance Tips

-   Use `verbose=false` for parameter sweeps to reduce output
-   Consider parallel processing for large parameter grids
-   Monitor convergence failures in complex parameter regions

## Troubleshooting

### Common Issues

1.  **Convergence Failures**: Try adjusting solver tolerance or iteration limits in config
2.  **Memory Issues**: Reduce parameter grid sizes or run experiments separately
3.  **Missing Dependencies**: Ensure all model files are in the correct relative paths

### Debugging

Use verbose mode to see detailed error messages:

``` bash
julia run_all_experiments.jl --experiments 3 --verbose
```

Check the YAML output files for complete error information and parameter values that caused failures.