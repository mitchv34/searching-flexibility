# Production Ensemble Local Optimization

This directory contains a production-ready implementation of ensemble local optimization for second-stage GMM estimation. The system is designed for robust, scalable, and maintainable optimization workflows.

## Files Overview

### Core Files
- **`run_search_production.jl`** - Main production script with comprehensive error handling, logging, and result persistence
- **`run_search.jl`** - Original development script (kept for reference)
- **`run_optimization.sh`** - Bash wrapper script for easy execution
- **`local_refine_config_template.yaml`** - Configuration template with all available options

### Dependencies
- **`objective.jl`** - Objective function implementation
- **`utils.jl`** - Utility functions

## Key Production Features

### 🔧 **Robust Error Handling**
- Comprehensive exception handling at all levels
- Graceful degradation for failed objective evaluations
- Automatic retry mechanisms
- Detailed error logging with stack traces

### 📊 **Comprehensive Logging**
- Configurable log levels (DEBUG, INFO, WARN, ERROR)
- Dual output to console and file
- Performance monitoring and timing
- Progress tracking during optimization

### 💾 **Result Persistence**
- Multiple output formats (CSV, JSON)
- Timestamped result files
- Detailed optimization trajectories
- Best parameter estimates
- Summary statistics

### ⚙️ **Configuration Management**
- YAML-based configuration with validation
- Command-line interface with argument parsing
- Environment variable support
- Template configuration file

### 🚀 **Performance Optimization**
- Parallel execution with thread safety
- Memory usage monitoring
- Efficient candidate selection algorithms
- Configurable optimization parameters

## Quick Start

### 1. Setup Configuration

Copy and customize the configuration template:

```bash
cp local_refine_config_template.yaml local_refine_config.yaml
# Edit local_refine_config.yaml with your specific paths and settings
```

### 2. Basic Usage

```bash
# Simple execution with defaults
./run_optimization.sh

# With custom parameters
./run_optimization.sh --candidates 20 --log-level DEBUG

# Dry run to validate configuration
./run_optimization.sh --dry-run
```

### 3. Direct Julia Execution

```bash
julia run_search_production.jl --config local_refine_config.yaml --candidates 10
```

## Configuration

### Required Configuration Sections

1. **GlobalSearchResults** - GA results and candidate selection
2. **ModelInputs** - Data paths and moment specifications  
3. **OptimizationSettings** - Algorithm parameters and tolerances

### Key Configuration Options

```yaml
GlobalSearchResults:
  job_id: "your_ga_job_id"
  config_path: "path/to/mpi_config.yaml"
  latest_results_file: "path/to/results_{job_id}.csv"
  n_top_starts: 10

OptimizationSettings:
  optimizer: "LBFGS"  # or "BFGS", "NelderMead"
  max_iterations: 5000
  gradient_tolerance: 1.0e-7

ModelInputs:
  config_path: "config/model_config.yaml"
  data_moments_yaml: "data/moments.yaml"
  weighting_matrix_csv: "data/weights.csv"
```

## Command Line Options

```bash
Usage: ./run_optimization.sh [OPTIONS]

OPTIONS:
    -c, --config FILE       Configuration file path
    -n, --candidates N      Number of diverse candidates (default: 10)
    -l, --log-level LEVEL   Log level: DEBUG, INFO, WARN, ERROR
    -t, --threads N         Number of Julia threads
    -d, --dry-run          Validation only, no optimization
    -h, --help             Show help message
```

## Environment Variables

- **`LOCAL_GA_JOB_ID`** - Override job ID from configuration
- **`JULIA_NUM_THREADS`** - Set number of Julia threads

## Output Files

All output files are timestamped and saved to the `output/` directory:

- **`optimization_results_YYYYMMDD_HHMMSS.csv`** - Detailed trajectory results
- **`best_parameters_YYYYMMDD_HHMMSS.csv`** - Best parameter estimates
- **`optimization_summary_YYYYMMDD_HHMMSS.json`** - Summary statistics

Log files are saved to the `logs/` directory with similar naming.

## Algorithm Details

### Candidate Selection
1. **Quality Filter**: Select top quantile of GA results
2. **Diversity Selection**: Use Farthest Point Sampling for maximum separation
3. **Normalization**: Parameter space normalization using GA bounds

### Optimization Process
1. **Parallel Execution**: Multiple trajectories via `EnsembleThreads()`
2. **Gradient-Based**: LBFGS optimizer with automatic differentiation
3. **Convergence**: Multiple tolerance criteria (gradient, function, parameter)
4. **Monitoring**: Real-time progress tracking and logging

### Result Analysis
- **Convergence Statistics**: Success rates and iteration counts
- **Performance Metrics**: Timing and memory usage
- **Parameter Estimates**: Best solutions with uncertainty measures
- **Improvement Analysis**: Comparison with initial GA candidates

## Error Handling

The system includes multiple layers of error handling:

1. **Configuration Validation**: Check required sections and file paths
2. **Data Validation**: Verify GA results and parameter bounds
3. **Objective Function**: Safe evaluation with fallback values
4. **Optimization**: Graceful handling of failed trajectories
5. **I/O Operations**: Robust file operations with error recovery

## Performance Tuning

### Thread Configuration
```bash
# Auto-detect threads
./run_optimization.sh

# Explicit thread count
./run_optimization.sh --threads 8
export JULIA_NUM_THREADS=8
```

### Memory Management
- Monitor memory usage with `Performance.monitor_memory: true`
- Enable garbage collection monitoring if needed
- Use compression for large result files

### Optimizer Selection
- **LBFGS**: Fast, memory-efficient (default)
- **BFGS**: More robust, higher memory usage
- **NelderMead**: Derivative-free, slower convergence

## Troubleshooting

### Common Issues

1. **"No job ID available"**
   - Set `job_id` in config or `LOCAL_GA_JOB_ID` environment variable

2. **"GA results CSV not found"**
   - Verify `latest_results_file` path in configuration
   - Check that `{job_id}` placeholder is correct

3. **"Parameter bounds not found"**
   - Ensure MPI config path is correct
   - Verify bounds structure in MPI configuration

4. **Optimization failures**
   - Try different optimizer (NelderMead for problematic functions)
   - Reduce tolerance requirements
   - Enable debug logging for detailed diagnostics

### Debug Mode

Enable comprehensive debugging:

```bash
./run_optimization.sh --log-level DEBUG
```

This provides:
- Detailed function call traces
- Parameter values at each step
- Memory and performance metrics
- Optimizer internal state

## Integration with Workflow

### Prerequisites
1. Completed global search (GA) with results CSV
2. MPI configuration file with parameter bounds
3. Data moments and weighting matrix files
4. Simulation scaffolding data

### Typical Workflow
1. **Global Search** → GA optimization produces candidate solutions
2. **Candidate Selection** → This script selects diverse, high-quality starts
3. **Local Refinement** → Gradient-based optimization from selected starts
4. **Result Analysis** → Best estimates with convergence diagnostics
5. **Model Validation** → Use final parameters for model validation

## Development Notes

### Code Organization
- **Modular Design**: Clear separation of concerns
- **Type Safety**: Explicit type annotations where beneficial
- **Documentation**: Comprehensive docstrings for all functions
- **Testing**: Built-in validation and dry-run capabilities

### Extension Points
- **Custom Optimizers**: Easy to add new optimization algorithms
- **Output Formats**: Extensible result persistence system
- **Monitoring**: Pluggable performance monitoring
- **Validation**: Configurable validation checks

This production implementation provides a robust foundation for second-stage GMM estimation with the flexibility to adapt to different models and requirements.
