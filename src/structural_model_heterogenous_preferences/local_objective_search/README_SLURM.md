# SLURM Ensemble Optimization Usage Guide

This directory contains a complete SLURM-based massive parallel optimization system for local refinement of optimization candidates.

## System Overview

The system consists of:

1. **Configuration**: `local_refine_config.yaml` - Centralized configuration
2. **Submission Script**: `submit_ensemble_optimization.sh` - Main submission script  
3. **SLURM Job Script**: `run_batch_optimization.slurm` - Individual batch execution
4. **Batch Processor**: `run_batch_search.jl` - Julia script for batch optimization
5. **Results Aggregator**: `aggregate_results.jl` - Combines and analyzes all results

## Quick Start

### 1. Configure the system

Edit `local_refine_config.yaml` to set your parameters:

```yaml
SlurmConfig:
  total_candidates: 128      # Total number of candidates to optimize
  batch_size: 32            # Candidates per SLURM job
  cores_per_job: 32         # CPU cores per job
  memory_per_job: "64GB"    # Memory per job
  time_limit: "6:00:00"     # Time limit per job
  output_dir: "output/mpi_results"
```

### 2. Prepare your candidates

Ensure you have a candidates file (CSV) with:
- Parameter columns specified in `param_columns`
- An `objective_value` column for quality filtering

### 3. Submit the optimization

```bash
./submit_ensemble_optimization.sh
```

This will:
- Read configuration from `local_refine_config.yaml`
- Calculate the number of batches needed (128 candidates ÷ 32 per batch = 4 batches)
- Submit a SLURM array job with 4 tasks
- Each task will optimize 32 candidates

### 4. Monitor progress

```bash
# Check job status
squeue -u $USER

# Check output logs
ls -la output/mpi_results/logs/

# Check real-time progress
tail -f output/mpi_results/logs/slurm-{job_id}_{array_id}.out
```

### 5. Aggregate results

After all jobs complete:

```bash
julia aggregate_results.jl
```

This will create:
- `aggregated_analysis.json` - Complete analysis
- `all_optimization_results.csv` - All results in tabular format
- `best_optimization_results.json` - Top 50 improvements
- `optimization_summary_report.txt` - Human-readable summary

## Configuration Details

### Key Parameters

```yaml
# Candidate selection
n_quality_filter: 1000     # Keep top N by objective value
n_diverse: 500             # Apply diversity sampling to N candidates

# Optimization
max_iters: 10000          # Maximum iterations per candidate
verbose: false            # Control optimization output

# File locations
candidates_file: "candidates.csv"
param_columns: ["param1", "param2", "param3"]

# SLURM configuration
SlurmConfig:
  partition: "compute"     # SLURM partition
  account: "your_account"  # SLURM account
  email: "user@domain.com" # Notification email
```

### Resource Scaling

For different scales:

**Small scale (32 candidates):**
```yaml
total_candidates: 32
batch_size: 8
cores_per_job: 8
memory_per_job: "16GB"
```

**Medium scale (256 candidates):**
```yaml
total_candidates: 256
batch_size: 32
cores_per_job: 32
memory_per_job: "64GB"
```

**Large scale (1000+ candidates):**
```yaml
total_candidates: 1024
batch_size: 64
cores_per_job: 32
memory_per_job: "128GB"
time_limit: "12:00:00"
```

## Output Structure

```
output/mpi_results/
├── logs/
│   ├── slurm-{job_id}_1.out          # SLURM output logs
│   ├── slurm-{job_id}_2.out
│   └── ...
├── optimization_results_batch_1.json # Detailed results per batch
├── optimization_results_batch_2.json
├── batch_1_summary.json              # Quick batch summaries
├── batch_2_summary.json
├── aggregated_analysis.json          # Complete analysis
├── all_optimization_results.csv      # All results (tabular)
├── best_optimization_results.json    # Top improvements
└── optimization_summary_report.txt   # Human-readable summary
```

## Troubleshooting

### Job fails to start
- Check SLURM partition and account settings
- Verify resource requests don't exceed limits
- Check file permissions on scripts

### Optimization errors
- Verify candidates file exists and has correct format
- Check parameter column names match config
- Ensure objective function is accessible

### Poor performance
- Increase `max_iters` for more thorough optimization
- Adjust `n_quality_filter` and `n_diverse` for better candidate selection
- Consider increasing memory allocation

### Results analysis
- Use `aggregate_results.jl` even for partial completions
- Check error files for failed batch diagnosis
- Monitor success rates and adjust parameters accordingly

## Advanced Usage

### Custom objective functions
Modify the objective function import in `run_batch_search.jl`:
```julia
include("../../../../your_custom_objective.jl")
```

### Different optimizers
Update the solver in `run_batch_search.jl`:
```julia
sol = solve(prob, BFGS(); maxiters=max_iters, solve_kwargs...)
```

### Real-time monitoring
Set up a monitoring script:
```bash
watch -n 30 'squeue -u $USER; echo "Results so far:"; ls -la output/mpi_results/*.json | wc -l'
```

## Support

For issues:
1. Check SLURM logs in `output/mpi_results/logs/`
2. Review error files: `batch_*_error.json`
3. Verify configuration in `local_refine_config.yaml`
4. Test with smaller batch sizes first
