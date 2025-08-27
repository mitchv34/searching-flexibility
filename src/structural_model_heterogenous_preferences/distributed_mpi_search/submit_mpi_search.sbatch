#!/bin/bash

#==============================================================================
# SLURM Directives - CONFIGURABLE MPI SEARCH
#==============================================================================
#
# Usage: sbatch [--export=MODE=production] submit_mpi_search.sbatch
# MODE options: "production" (econ partition) or "standard" (short partition, default)
#
#SBATCH --job-name=MPI_Search             # Job name
#SBATCH --partition=econ                  # Use econ for longer wallclock (was short)
#
#SBATCH --nodes=1                         # Single node
#SBATCH --ntasks=96                       # Match GA population size for full search
#SBATCH --cpus-per-task=1                 # 1 CPU per task
#
#SBATCH --mem=200G                        # Default: 200GB RAM
#SBATCH --time=0-12:00:00                 # 12h target (adjust if econ max differs)
#
# Store logs within the distributed_mpi_search subtree (requested path)
# NOTE: Ensure directory exists before submission (it is created in repo).
#SBATCH --output=src/structural_model_heterogenous_preferences/distributed_mpi_search/output/logs/mpi_search_%j.log
#SBATCH --error=src/structural_model_heterogenous_preferences/distributed_mpi_search/output/logs/mpi_search_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL        # Email notifications
#SBATCH --mail-user=mitchv34@gmail.com    # Email address

#==============================================================================
# Environment Setup and Mode Configuration
#==============================================================================

# Set default mode
MODE=${MODE:-standard}

echo "========================================================================"
echo "MPI SEARCH JOB - MODE: $MODE"
echo "Started on $(hostname) at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Tasks: $SLURM_NTASKS"
echo "========================================================================"

# Configure based on mode
if [[ "$MODE" == "production" ]]; then
    echo "🏭 PRODUCTION MODE: Using optimized settings"
    # Production mode uses econ partition settings
    # Note: You should submit with: sbatch --export=MODE=production --partition=econ --constraint="amd&small" --mem=40G --time=0-02:00:00 submit_mpi_search.sbatch
    CONFIG_FILE="src/structural_model_heterogenous_preferences/distributed_mpi_search/mpi_search_config.yaml"
else
    echo "🔬 STANDARD MODE: Using flexible settings" 
    CONFIG_FILE="src/structural_model_heterogenous_preferences/distributed_mpi_search/mpi_search_config.yaml"
fi

echo "Configuration file: $CONFIG_FILE"

# Load environment
source /etc/bashrc

# Change to project directory
cd /project/high_tech_ind/searching-flexibility

# Create (legacy root) output directories for backward compatibility
mkdir -p output/logs
mkdir -p output/mpi_results
# Create colocated results/logs directories under distributed_mpi_search
mkdir -p src/structural_model_heterogenous_preferences/distributed_mpi_search/output/logs
mkdir -p src/structural_model_heterogenous_preferences/distributed_mpi_search/output/results

# Show environment info
echo "Julia version: $(julia --version)"
echo "SLURM allocation: $SLURM_NTASKS tasks"
echo "Working directory: $(pwd)"
echo ""

# Define paths
SYSIMAGE_PATH="src/structural_model_heterogenous_preferences/distributed_mpi_search/MPI_GridSearch_sysimage.so"
JULIA_SCRIPT="src/structural_model_heterogenous_preferences/distributed_mpi_search/mpi_search.jl"

#==============================================================================
# Job Execution
#==============================================================================

echo "🚀 Starting MPI parameter search with $SLURM_NTASKS tasks..."
echo "Rule-of-thumb: population size ≈ 10–15× n_params; adjust in config if needed."

if [[ "${DISABLE_SYSIMAGE:-0}" == "1" ]]; then
    echo "🚫 DISABLE_SYSIMAGE=1 -> ignoring system image even if present"
    JULIA_FLAGS="--startup-file=no --project=."
else
    # Check if system image exists
    if [[ -f "$SYSIMAGE_PATH" ]]; then
        echo "✓ System image found: $SYSIMAGE_PATH"
        echo "  Size: $(ls -lh $SYSIMAGE_PATH | awk '{print $5}')"
        JULIA_FLAGS="--startup-file=no --sysimage=$SYSIMAGE_PATH --project=."
    else
        echo "⚠️  No system image found, using standard Julia startup"
        JULIA_FLAGS="--startup-file=no --project=."
    fi
fi

echo "Julia flags: $JULIA_FLAGS"
echo "Using config: $CONFIG_FILE"
echo ""

# Run the Julia script (SlurmClusterManager handles worker spawning)
echo "Running with JULIA_FLAGS: $JULIA_FLAGS"
echo "Environment toggles: FORCE_SINGLE_PROCESS=${FORCE_SINGLE_PROCESS:-0} SKIP_STARTUP_TEST_EVAL=${SKIP_STARTUP_TEST_EVAL:-0} EARLY_EXIT_AFTER_INIT=${EARLY_EXIT_AFTER_INIT:-0}"
julia $JULIA_FLAGS $JULIA_SCRIPT

echo ""
echo "========================================================================"
echo "MPI SEARCH JOB COMPLETED - MODE: $MODE"
echo "Finished on $(hostname) at $(date)"
echo "========================================================================"
