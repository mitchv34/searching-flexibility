#!/bin/bash

# Short SLURM test job to benchmark per-evaluation time for MPI search
# Usage: sbatch submit_mpi_search_test.sbatch

#SBATCH --job-name=MPI_Search_Test
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=8            # fewer tasks for quick test
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=0-00:08:00     # 8 minutes max
#SBATCH --output=output/logs/mpi_search_test_%j.log
#SBATCH --error=output/logs/mpi_search_test_%j.err

set -euo pipefail

echo "========================================================================"
echo "MPI SEARCH TEST JOB"
echo "Started on $(hostname) at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Tasks: $SLURM_NTASKS"
echo "========================================================================"

CONFIG_FILE="src/structural_model_heterogenous_preferences/distributed_mpi_search/mpi_search_config_test.yaml"
SYSIMAGE_PATH="src/structural_model_heterogenous_preferences/distributed_mpi_search/MPI_GridSearch_sysimage.so"
JULIA_SCRIPT="src/structural_model_heterogenous_preferences/distributed_mpi_search/mpi_search.jl"

cd /project/high_tech_ind/searching-flexibility
mkdir -p output/logs output/mpi_results

echo "Julia version: $(julia --version)"
echo "Using config: $CONFIG_FILE"

if [[ -f "$SYSIMAGE_PATH" ]]; then
  echo "✓ System image found: $SYSIMAGE_PATH"
  JULIA_FLAGS="--startup-file=no --sysimage=$SYSIMAGE_PATH --project=."
else
  echo "⚠️  No system image found, running without it"
  JULIA_FLAGS="--project=."
fi

echo "Launching test run (n_samples=200 in config)" 
START_TS=$(date +%s)

# Pass config path as argument so test config is used
julia $JULIA_FLAGS $JULIA_SCRIPT $CONFIG_FILE

END_TS=$(date +%s)
ELAPSED=$((END_TS-START_TS))
echo "Test run wall time: ${ELAPSED}s" 

# Extract quick summary if latest file exists
LATEST_FILE=$(ls -1t src/structural_model_heterogenous_preferences/distributed_mpi_search/output/results/mpi_search_results_job${SLURM_JOB_ID}_latest.json 2>/dev/null | head -n1 || true)
if [[ -f "$LATEST_FILE" ]]; then
  echo "Summary file: $LATEST_FILE"
  echo "Best objective:" $(grep -m1 '"best_objective"' "$LATEST_FILE" | sed 's/.*: //')
fi

echo "========================================================================"
echo "MPI SEARCH TEST JOB COMPLETE"
echo "Finished at $(date)"
echo "========================================================================"
