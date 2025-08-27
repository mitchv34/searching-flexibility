# Distributed MPI Parameter Search

This directory contains a scalable distributed parameter search implementation using MPI (Message Passing Interface) to leverage multiple compute nodes simultaneously for parameter optimization of the structural model.

## 🎯 Overview

This is **Strategy 3** - the most powerful and scalable approach for parameter optimization, capable of utilizing hundreds of cores across multiple nodes for a single optimization run. It uses:

- **ClusterManagers.jl** to integrate with SLURM and MPI
- **Distributed.jl** for parallel parameter evaluation
- **MPI** for inter-node communication
- **Multiple nodes** (default: 4 nodes × 32 cores = 128 total cores)

## 📁 Directory Structure

```
distributed_mpi_search/
├── mpi_search.jl              # Main MPI distributed search script
├── mpi_search_config.yaml     # Configuration file  
├── submit_mpi_search.sbatch   # SLURM submission script
├── manage_mpi_search.sh       # Management utilities
├── analyze_mpi_results.jl     # Results analysis script
├── create_mpi_sysimage.jl     # System image creation
├── MPI_GridSearch_sysimage.so # Compiled system image (created)
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Submit MPI Search Job
```bash
# Using the management script (recommended)
./manage_mpi_search.sh submit

# Or directly with sbatch
sbatch submit_mpi_search.sbatch
```

### 2. Monitor Progress
```bash
# Monitor job status
./manage_mpi_search.sh monitor

# View live logs
./manage_mpi_search.sh logs

# Check current configuration
./manage_mpi_search.sh status
```

### 3. Analyze Results
```bash
# Analyze latest results
./manage_mpi_search.sh analyze

# Or run detailed analysis script
julia analyze_mpi_results.jl
```

## ⚙️ Configuration

The search is configured via `mpi_search_config.yaml`:

### Key Configuration Sections

#### MPI Search Parameters
```yaml
MPISearchConfig:
  algorithm: "random_search"    # Search algorithm
  n_samples: 50000             # Total parameter evaluations
  
  parameters:
    names: ["a_h", "b_h", "c0", "mu", "chi", "A1", "nu", "psi_0", "phi", "kappa0"]
    bounds:                    # [min, max] for each parameter
      - [1.5, 2.5]            # a_h: Human capital shape
      - [4.0, 6.0]            # b_h: Human capital scale
      # ... etc
```

#### Cluster Configuration
```yaml
cluster_config:
  expected_nodes: 4
  expected_cores_per_node: 32
  expected_total_cores: 128
  load_balancing: "dynamic"
  fault_tolerance: true
```

#### SLURM Settings
```yaml
slurm_config:
  partition: "compute"
  nodes: 4
  ntasks_per_node: 32
  time_limit: "0-08:00:00"     # 8 hours
  memory: "0"                  # Use all available memory
```

## 🔧 Advanced Usage

### Creating System Image (Recommended)
For faster startup and improved performance:

```bash
julia create_mpi_sysimage.jl
```

This creates `MPI_GridSearch_sysimage.so` which reduces Julia worker startup latency substantially.

Environment toggles for the build script:
```bash
# Skip producing the .so (just execute warmup precompile path)
SKIP_SYSIMAGE=1 julia create_mpi_sysimage.jl

# Disable simulation moment warmup (faster build, less coverage)
PRECOMPILE_SIM=0 julia create_mpi_sysimage.jl

# Adjust lightweight solver iteration count used during warmup (default 50)
PRECOMPILE_SOLVER_STEPS=20 julia create_mpi_sysimage.jl
```

Using the system image manually:
```bash
julia --sysimage=./MPI_GridSearch_sysimage.so mpi_search.jl
```

The submission script auto-detects the file if present; no flag required.

### Custom Configuration
```bash
# Use custom config file
CONFIG_FILE=my_custom_config.yaml ./manage_mpi_search.sh submit
```

### Manual Job Management
```bash
# Submit job and capture ID
JOB_ID=$(sbatch submit_mpi_search.sbatch | grep -o '[0-9]*$')

# Monitor specific job
squeue -j $JOB_ID

# Cancel specific job  
scancel $JOB_ID

# View logs
tail -f logs/mpi_search_${JOB_ID}.out
```

## 📊 Results Analysis

The search produces comprehensive results including:

### Automatic Analysis
- **Convergence plots**: Progress over evaluations
- **Parameter distributions**: Top performers vs all samples
- **Correlation analysis**: Parameter sensitivity
- **Performance metrics**: Timing and efficiency stats

### Output Files
```
output/mpi_search_results/
├── mpi_search_results_YYYY-MM-DD_HH-MM-SS.json  # Raw results
├── convergence_analysis_YYYY-MM-DD_HH-MM-SS.json # Convergence stats
├── parameter_analysis_YYYY-MM-DD_HH-MM-SS.csv    # Parameter analysis
└── analysis_summary_YYYY-MM-DD_HH-MM-SS.txt      # Human-readable summary

figures/mpi_search_analysis/
├── mpi_convergence_YYYY-MM-DD_HH-MM-SS.png       # Convergence plots
├── parameter_correlations_YYYY-MM-DD_HH-MM-SS.png # Parameter correlations
├── parameter_distributions_YYYY-MM-DD_HH-MM-SS.png # Distribution plots
└── best_parameters_YYYY-MM-DD_HH-MM-SS.png       # Best parameter values
```

## 🏗️ Technical Architecture

### MPI Workflow
1. **Main process** loads configuration and generates parameter vectors
2. **SLURM** allocates multiple nodes with specified cores per node
3. **ClusterManagers.jl** automatically discovers and adds all allocated cores as Julia workers
4. **Parameter vectors** are distributed across all workers using `pmap()`
5. **Each worker** evaluates parameters independently (solve model + calculate objective)
6. **Results** are collected back to main process and saved

### Scalability Features
- **Dynamic load balancing**: Work is distributed automatically
- **Fault tolerance**: Failed worker evaluations are handled gracefully  
- **Memory efficiency**: Each worker only holds necessary data
- **Checkpoint saving**: Progress is saved periodically
- **Heterogeneous clusters**: Works across different node types

### Performance Optimizations
- **System image**: Pre-compiled Julia environment (~15x faster startup)
- **Minimal data transfer**: Only parameter vectors and objectives passed between nodes
- **Optimized solver settings**: Tuned for batch evaluation
- **Memory management**: Automatic cleanup of temporary data

## 🔍 Search Algorithms

### Random Search (Default)
- **Uniform sampling** within parameter bounds
- **Simple and robust** for initial exploration
- **Highly parallelizable** with no coordination needed

### Genetic Algorithm (Future)
- **Population-based** evolutionary optimization
- **Global search** with local refinement
- **More sophisticated** but requires coordination

### Bayesian Optimization (Future)
- **Model-based** optimization using Gaussian processes
- **Sample efficient** for expensive evaluations
- **Sequential** nature limits parallelization

## 🚨 Troubleshooting

### Common Issues

#### MPI Not Available
```bash
# Check if MPI module is loaded
module list

# Load MPI module
module load mpi/openmpi
```

#### System Image Issues
```bash
# Remove corrupted system image
rm MPI_GridSearch_sysimage.so

# Recreate system image
julia create_mpi_sysimage.jl
```

#### SLURM Job Failures
```bash
# Check job status
squeue -u $USER

# View error logs
tail -50 logs/mpi_search_JOBID.err

# Check node allocation
scontrol show job JOBID
```

#### Worker Connection Issues
Check the error logs for:
- Network connectivity problems
- Module loading failures
- Memory allocation issues
- File system access problems

### Performance Issues

#### Slow Convergence
- Increase `n_samples` in config
- Adjust parameter bounds to focus search
- Check if model solver is converging

#### Poor Scaling
- Verify all nodes are being used: check `n_workers` in results
- Monitor CPU usage during job execution
- Check for I/O bottlenecks

## 📈 Performance Expectations

### Typical Performance (4 nodes × 32 cores = 128 cores)
- **Startup time**: ~2 minutes (with system image)
- **Evaluations per second**: ~50-200 (depends on model complexity)
- **50,000 evaluations**: ~4-17 minutes of computation
- **Total runtime**: ~6-20 minutes including startup/cleanup

### Scaling Characteristics
- **Linear scaling** up to ~100-200 cores (depends on model)
- **Communication overhead** becomes significant beyond 200 cores
- **Memory requirements**: ~1-4 GB per worker
- **I/O bottlenecks** may limit scaling with shared file systems

## 🎛️ Management Commands

The `manage_mpi_search.sh` script provides comprehensive job management:

```bash
# Job lifecycle
./manage_mpi_search.sh submit      # Submit new job
./manage_mpi_search.sh monitor     # Check job status  
./manage_mpi_search.sh logs        # View live logs
./manage_mpi_search.sh cancel      # Cancel running job

# Analysis and maintenance
./manage_mpi_search.sh status      # Show configuration
./manage_mpi_search.sh analyze     # Analyze latest results
./manage_mpi_search.sh cleanup     # Clean old files

# Help
./manage_mpi_search.sh help        # Show usage information
```

## 🔗 Integration

This MPI search integrates with the broader parameter estimation framework:

- **Uses same model files**: ModelSetup.jl, ModelSolver.jl, ModelEstimation.jl
- **Compatible with moment filtering**: Uses moment_filtering.jl utilities
- **Consistent configuration**: Similar structure to grid_search_config.yaml
- **Unified output format**: Results compatible with other analysis tools

## 📚 References

- [ClusterManagers.jl Documentation](https://github.com/JuliaParallel/ClusterManagers.jl)
- [Distributed Computing in Julia](https://docs.julialang.org/en/v1/manual/distributed-computing/)
- [MPI.jl Documentation](https://juliaparallel.github.io/MPI.jl/stable/)
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
