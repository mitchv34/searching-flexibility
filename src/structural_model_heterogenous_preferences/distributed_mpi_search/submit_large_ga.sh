#!/bin/bash
#=============================================================================
# Slurm Submission Script: Large GA Search (32 workers)
# Usage: sbatch submit_large_ga.sh
# Environment assumptions: julia available in PATH (module load if needed)
#=============================================================================
#SBATCH --job-name=GA_Flex_single
#SBATCH --ntasks=1                  # Single master; spawns workers internally
#SBATCH --cpus-per-task=32          # Provide 32 CPUs to share among workers
#SBATCH --time=01:55:00             # Walltime (<2h limit of econ-grad-short)
#SBATCH --mem=128G                  # Total memory for all spawned workers
#SBATCH --output=src/structural_model_heterogenous_preferences/distributed_mpi_search/output/logs/ga_short_%j.out
#SBATCH --error=src/structural_model_heterogenous_preferences/distributed_mpi_search/output/logs/ga_short_%j.err
#SBATCH --partition=econ-grad-short  # Submit to econ-grad short partition per user request
#SBATCH --mail-type=END,FAIL        # (Optional) notifications
##SBATCH --mail-user=you@example.com

set -euo pipefail

ROOT_DIR="$(pwd)"
SCRIPT_DIR="${ROOT_DIR}/src/structural_model_heterogenous_preferences/distributed_mpi_search"
CONFIG_FILE="${SCRIPT_DIR}/mpi_search_config.yaml"
LOG_DIR="${SCRIPT_DIR}/output/logs"
mkdir -p "${LOG_DIR}" "${SCRIPT_DIR}/output/results"

echo "[INFO] Starting Large GA Search job on $(date)"
echo "[INFO] Working directory: ${ROOT_DIR}"
echo "[INFO] Config: ${CONFIG_FILE}"
echo "[INFO] SLURM_NTASKS=${SLURM_NTASKS:-unset} (single-task launch strategy)"

# Optional: load modules (uncomment & customize)
# module load julia/1.11.6

# Recommended: disable precompile storms on workers
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_NUM_THREADS=1   # Each worker single-threaded; master will spawn up to (CPUs-1) workers
export INCREMENTAL_WORKERS=1  # Enable incremental spawn diagnostics

echo "[INFO] Using custom sysimage if present."
unset DISABLE_CUSTOM_SYSIMAGE || true
SYSIMAGE_PATH="src/structural_model_heterogenous_preferences/distributed_mpi_search/MPI_GridSearch_sysimage.so"
if [[ -f "$SYSIMAGE_PATH" ]]; then
  SYSIMAGE_FLAG="--sysimage=$SYSIMAGE_PATH"
  echo "[INFO] Sysimage found: $SYSIMAGE_PATH"
else
  SYSIMAGE_FLAG=""
  echo "[WARN] Sysimage not found at $SYSIMAGE_PATH; proceeding with JIT."
fi

# Master process launches workers via SlurmClusterManager inside script
# srun used to allocate the tasks; only one Julia invocation needed.

# Single master process; internal addprocs will allocate workers locally (no multi-node until stable).
srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-32} julia --project=. ${SYSIMAGE_FLAG} \
  src/structural_model_heterogenous_preferences/distributed_mpi_search/mpi_search.jl \
  src/structural_model_heterogenous_preferences/distributed_mpi_search/mpi_search_config.yaml

EXIT_CODE=$?

if [[ ${EXIT_CODE} -eq 0 ]]; then
  echo "[INFO] Job completed successfully at $(date)"
else
  echo "[ERROR] Job failed with exit code ${EXIT_CODE} at $(date)" >&2
fi

exit ${EXIT_CODE}
