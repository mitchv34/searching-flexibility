# Searching Flexibility - Experimental Framework

This repository provides a comprehensive framework for running economic experiments related to labor market flexibility, remote work, and job search dynamics. The code implements structural economic models with both basic and heterogeneous preference specifications.

## Quick Start

To run a basic experiment demonstrating the framework capabilities:

```bash
# Run a lightweight synthetic experiment (no heavy dependencies required)
julia experiment_runner.jl minimal_experiment

# Show all available experiments
julia experiment_runner.jl help
```

## Available Experiments

### 1. Minimal Experiment (`minimal_experiment`)
- **Purpose**: Demonstrates the experimental infrastructure without requiring heavy dependencies
- **Features**: Synthetic parameter sweeps, basic model file validation, configuration parsing
- **Output**: Parameter optimization results and summary statistics
- **Requirements**: Only basic Julia standard library

```bash
julia experiment_runner.jl minimal_experiment --verbose
```

### 2. Basic Model Test (`basic_model_test`)
- **Purpose**: Test full model initialization and solving capabilities
- **Features**: Real model loading, parameter initialization, basic solving
- **Requirements**: Full dependency installation (YAML, Parameters, Distributions, etc.)

```bash
julia experiment_runner.jl basic_model_test --model heterogenous
```

### 3. Parameter Estimation (`parameter_estimation`)
- **Purpose**: Run comprehensive parameter estimation using optimization algorithms
- **Features**: Distributed computing, multiple parameter optimization, moment matching
- **Models**: Both `new` and `heterogenous` model specifications

```bash
# Quick version with reduced workers and iterations
julia experiment_runner.jl parameter_estimation --model new --quick

# Full version (requires substantial computational resources)
julia experiment_runner.jl parameter_estimation --model heterogenous --verbose
```

### 4. Cross-Moment Check (`cross_moment_check`)
- **Purpose**: Validate model moments across parameter ranges
- **Features**: Parameter sweeps, moment computation, distributed processing
- **Output**: Moment sensitivity analysis and validation plots

```bash
julia experiment_runner.jl cross_moment_check --quick
```

### 5. Profile Analysis (`profile_analysis`)
- **Purpose**: Performance profiling and optimization analysis
- **Features**: Execution time analysis, memory usage profiling, bottleneck identification

```bash
julia experiment_runner.jl profile_analysis --model heterogenous
```

## Model Specifications

### Heterogeneous Preferences Model
- **Location**: `src/structural_model_heterogenous_preferences/`
- **Features**: Worker preference heterogeneity, Gamma-distributed productivity shocks
- **Key Parameters**: `k` (preference heterogeneity), `aₕ`, `bₕ` (skill distribution), `c₀` (search cost)

### New Model Specification  
- **Location**: `src/structural_model_new/`
- **Features**: Updated model structure, improved solver algorithms
- **Key Parameters**: `χ` (utility scaling), `ν` (productivity parameter), `ψ₀`, `ϕ` (remote work parameters)

## Dependencies and Installation

### Minimal Setup (for basic experiments)
```bash
# Only requires Julia standard library
julia experiment_runner.jl minimal_experiment
```

### Full Setup (for complete model experiments)
```bash
# Install all dependencies (may take 10-20 minutes)
julia --project=. -e "using Pkg; Pkg.instantiate()"

# Then run full experiments
julia experiment_runner.jl basic_model_test
```

### Key Package Dependencies
- **Core**: `Parameters.jl`, `YAML.jl`, `Distributions.jl`
- **Optimization**: `Optim.jl`, `Optimization.jl`, `OptimizationOptimJL.jl`
- **Parallel**: `Distributed.jl`, `SharedArrays.jl`
- **Plotting**: `CairoMakie.jl`, `PrettyTables.jl`
- **Advanced**: `ForwardDiff.jl`, `NonlinearSolve.jl`

## Command Line Options

```bash
julia experiment_runner.jl [experiment] [options]

Options:
  --model {new|heterogenous}    Select model specification (default: heterogenous)
  --quick                       Reduce computational requirements (fewer workers, iterations)
  --verbose                     Show detailed output during execution
  help                          Show available experiments and usage
```

## Output and Results

### Standard Output Locations
- **Logs**: `/tmp/[experiment]_output.log`
- **Results**: `/tmp/[experiment]_results.txt` 
- **Figures**: Model-specific directories under `figures/`

### Result Interpretation
- **Parameter Estimation**: Shows convergence of optimization, estimated vs. true parameter values
- **Cross-Moment Validation**: Displays moment sensitivity to parameter changes
- **Profile Analysis**: Execution time breakdowns and performance bottlenecks

## Computational Requirements

### Minimal Experiments
- **CPU**: Single core sufficient
- **Memory**: < 1GB RAM
- **Time**: < 1 minute

### Full Model Experiments  
- **CPU**: 4-10 cores recommended (distributed processing)
- **Memory**: 4-16GB RAM depending on grid sizes
- **Time**: 10 minutes to several hours for comprehensive estimation

### High-Performance Computing
For large-scale experiments, the framework supports:
- **MPI**: Distributed computing across multiple nodes
- **Slurm**: Cluster job scheduling
- **Custom configurations**: Adjustable grid sizes, iteration limits, convergence tolerances

## Troubleshooting

### Common Issues

**Dependency Installation Fails**
```bash
# Try updating registry
julia -e "using Pkg; Pkg.Registry.update()"
# Then reinstall
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

**Path Issues**
- Ensure you're running from the repository root directory
- Check that model files exist in expected locations

**Performance Issues**  
- Use `--quick` flag for testing
- Reduce worker count in distributed experiments
- Check available system memory

### Getting Help
1. Run `julia experiment_runner.jl help` for usage information
2. Check log files in `/tmp/` for detailed error messages
3. Use `--verbose` flag for debugging
4. Start with `minimal_experiment` to verify basic functionality

## Research Applications

This experimental framework supports research in:
- **Labor Economics**: Remote work adoption, job search behavior
- **Industrial Organization**: Firm flexibility, worker matching
- **Macroeconomics**: Productivity shocks, employment dynamics
- **Policy Analysis**: Remote work mandates, unemployment benefits, search subsidies

## Citation

If you use this experimental framework in research, please cite the associated research papers and acknowledge the computational methods developed here.