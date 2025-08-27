# Structural Model Estimation - Technical Documentation

## 1. Estimation Procedure Overview
*[TODO: Add estimation methodology, moment conditions, identification strategy, and economic interpretation]*

---

## 2. MPI Distributed Search System

### 2.1 Directory Structure
```
distributed_mpi_search/
├── mpi_search.jl                    # Core MPI parameter search engine
├── launch_mpi_estimation.sh         # Comprehensive job launcher
├── submit_mpi_search.sbatch         # SLURM submission script
├── manage_mpi_search.sh             # Management interface
├── mpi_search_config.yaml           # Search configuration
├── enhanced_monitoring.jl           # Real-time progress monitoring
├── advanced_analysis.jl             # Publication-quality analysis
├── create_mpi_sysimage.jl          # System image creation
├── MPI_GridSearch_sysimage.so      # Precompiled system image (402M)
└── output/
    ├── results/                     # Search results (JSON)
    └── logs/                        # SLURM logs
```

### 2.2 Core Components

#### **mpi_search.jl**
- **Purpose**: Distributed parameter search using MPI with 16 workers
- **Key Features**:
  - Latin Hypercube Sampling for parameter space exploration
  - Automatic cleanup of intermediate snapshots and old results
  - Dual file output: `_final.json` (complete archive) and `_latest.json` (summary)
  - Robust error handling with 7e9 penalty for failed evaluations
- **Usage**: Called automatically by SLURM submission scripts

#### **launch_mpi_estimation.sh**
- **Purpose**: Production launcher with comprehensive error checking
- **Features**:
  - System image detection and fallback to compilation
  - Environment validation and dependency checking
  - Flexible worker count and time limit configuration
  - Automatic output directory creation
- **Usage**: `./launch_mpi_estimation.sh [--workers N] [--time HH:MM:SS]`

#### **submit_mpi_search.sbatch**
- **Purpose**: SLURM job submission with configurable resources
- **Configuration**:
  - Default: 16 tasks, 200GB RAM, 4-hour time limit, short partition
  - Production mode: `sbatch --export=MODE=production submit_mpi_search.sbatch`
- **Output**: Logs to `output/logs/mpi_search_JOBID.{log,err}`

### 2.3 Management Interface

#### **manage_mpi_search.sh**
Unified command-line interface for all MPI search operations:

```bash
# Job Management
./manage_mpi_search.sh submit              # Submit new search job
./manage_mpi_search.sh monitor [JOB_ID]    # Enhanced real-time monitoring
./manage_mpi_search.sh logs [JOB_ID]       # View SLURM logs
./manage_mpi_search.sh cancel [JOB_ID]     # Cancel running job

# Analysis
./manage_mpi_search.sh analyze             # Quick results summary
./manage_mpi_search.sh plots [JOB_ID]      # Generate diagnostic plots
./manage_mpi_search.sh status              # System configuration
./manage_mpi_search.sh cleanup             # Remove old files
```

**Features**:
- Automatic job ID tracking (stores last submitted job)
- Color-coded output for better readability
- Prerequisite checking before job submission
- Integration with enhanced monitoring and analysis tools

### 2.4 Monitoring and Analysis

#### **enhanced_monitoring.jl**
- **Purpose**: Real-time progress tracking with performance metrics
- **Features**:
  - Progress bar with completion percentage
  - ETA calculations and evaluation rate monitoring
  - Parallel efficiency analysis for multi-worker jobs
  - Historical progress logging
- **Usage**: `julia enhanced_monitoring.jl [JOB_ID]`
- **Output**: Comprehensive progress report with timestamps

#### **advanced_analysis.jl**
- **Purpose**: Publication-quality diagnostic analysis
- **Features**:
  - Convergence analysis with improvement tracking
  - Parameter correlation and sensitivity analysis
  - Top candidates leaderboard with performance comparison
  - Parameter and moment trajectory evolution
  - Performance and efficiency metrics
- **Usage**: `julia advanced_analysis.jl [JOB_ID]`
- **Output**: High-resolution plots saved to `figures/mpi_analysis/`

### 2.5 Configuration

#### **mpi_search_config.yaml**
```yaml
search_parameters:
  n_samples: 1000                    # Total parameter combinations
  method: "latin_hypercube"          # Sampling strategy
  
parameter_bounds:
  param1: [min_val, max_val]         # Parameter search ranges
  param2: [min_val, max_val]
  
optimization:
  objective: "gmm_criterion"         # Objective function
  max_workers: 16                    # MPI worker count
  timeout_per_eval: 300              # Seconds per evaluation
```

### 2.6 System Image Optimization

#### **create_mpi_sysimage.jl**
- **Purpose**: Creates precompiled system image to eliminate startup delays
- **Benefits**: Reduces job startup time from 2+ minutes to ~10 seconds
- **Usage**: `julia create_mpi_sysimage.jl` (creates `MPI_GridSearch_sysimage.so`)
- **Size**: ~402MB containing all compiled dependencies

### 2.7 Workflow Example

1. **Setup**: Ensure system image exists and configuration is correct
2. **Submit**: `./manage_mpi_search.sh submit`
3. **Monitor**: `./manage_mpi_search.sh monitor` (real-time progress)
4. **Analyze**: `./manage_mpi_search.sh plots` (diagnostic plots)
5. **Results**: Check `output/results/` for final parameter estimates

### 2.8 Technical Notes

#### **File Management**
- **Intermediate snapshots**: Saved every 50 evaluations, automatically cleaned (keeps only latest)
- **Final results**: `job{ID}_final.json` contains complete search history
- **Latest summary**: `job{ID}_latest.json` contains best results and search statistics
- **Log retention**: SLURM logs automatically cleaned after 7 days

#### **Error Handling**
- **7e9 penalty**: Assigned to failed model evaluations to maintain search continuity
- **Type safety**: Fixed `Vector{Symbol}` conversion for moment computation
- **Resource limits**: Automatic job cancellation on timeout or memory exhaustion

#### **Performance Optimization**
- **MPI parallelization**: 16 workers for distributed parameter evaluation
- **System image**: Precompiled dependencies eliminate compilation overhead
- **Efficient I/O**: Batched result saving reduces file system load
- **Memory management**: Automatic cleanup prevents memory leaks during long searches

### 2.9 Dependencies
- Julia 1.11.6 with MPI.jl, SlurmClusterManager.jl
- SLURM workload manager
- MPI implementation (OpenMPI/MPICH)
- CairoMakie.jl for visualization
- JSON3.jl, YAML.jl for configuration