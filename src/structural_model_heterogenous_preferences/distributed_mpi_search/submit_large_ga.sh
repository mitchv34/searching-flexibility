#!/bin/bash
#=============================================================================
# Slurm Submission Script: Large GA Search (32 workers)
# Usage: sbatch submit_large_ga.sh
# Environment assumptions: julia available in PATH (module load if needed)
#=============================================================================
#SBATCH --job-name=GA_Flex32
#SBATCH --ntasks=32                 # MPI workers (master + 31 workers typical)
#SBATCH --cpus-per-task=1           # 1 thread per worker (adjust if using threading)
#SBATCH --time=24:00:00             # Walltime limit
#SBATCH --mem-per-cpu=4G            # Memory per task (increased after profiling ~1.4GB peak per eval)
#SBATCH --output=src/structural_model_heterogenous_preferences/distributed_mpi_search/output/logs/ga32_%j.out
#SBATCH --error=src/structural_model_heterogenous_preferences/distributed_mpi_search/output/logs/ga32_%j.err
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
echo "[INFO] SLURM_NTASKS=${SLURM_NTASKS:-unset}"

# Optional: load modules (uncomment & customize)
# module load julia/1.11.6

# Recommended: disable precompile storms on workers
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_NUM_THREADS=1   # Each worker single-threaded (tune if desired)

echo "[INFO] Forcing JIT mode (user request) – skipping custom sysimage even if present."
export DISABLE_CUSTOM_SYSIMAGE=1
SYSIMAGE_FLAG=""  # ensure unset

# Master process launches workers via SlurmClusterManager inside script
# srun used to allocate the tasks; only one Julia invocation needed.

# IMPORTANT: use a single master process; SlurmClusterManager will launch workers.
# Without --ntasks=1 here, Slurm would start 32 independent masters, each trying to
# launch 32 workers (causing the repeated "Starting Distributed MPI Parameter Search" lines).
srun --ntasks=1 julia --project=. ${SYSIMAGE_FLAG} \
  src/structural_model_heterogenous_preferences/distributed_mpi_search/mpi_search.jl \
  src/structural_model_heterogenous_preferences/distributed_mpi_search/mpi_search_config.yaml

EXIT_CODE=$?

if [[ ${EXIT_CODE} -eq 0 ]]; then
  echo "[INFO] Job completed successfully at $(date)"
else
  echo "[ERROR] Job failed with exit code ${EXIT_CODE} at $(date)" >&2
fi

exit ${EXIT_CODE}
