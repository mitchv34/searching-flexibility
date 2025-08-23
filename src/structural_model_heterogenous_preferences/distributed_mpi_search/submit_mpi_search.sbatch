#!/bin/bash

#==============================================================================
# SLURM Directives - CONFIGURABLE MPI SEARCH
#==============================================================================
#
# Usage: sbatch [--export=MODE=production] submit_mpi_search.sbatch
# MODE options: "production" (econ partition) or "standard" (short partition, default)
#
#SBATCH --job-name=MPI_Search             # Job name
#SBATCH --partition=short                 # Default partition (override with MODE=production)
#
#SBATCH --nodes=1                         # Single node
#SBATCH --ntasks=16                       # 16 MPI tasks
#SBATCH --cpus-per-task=1                 # 1 CPU per task
#
#SBATCH --mem=200G                        # Default: 200GB RAM
#SBATCH --time=0-04:00:00                 # Default: 4 hours
#
#SBATCH --output=output/logs/mpi_search_%j.log    # Log files
#SBATCH --error=output/logs/mpi_search_%j.err     # Error files
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

# Create output directories
mkdir -p output/logs
mkdir -p output/mpi_results

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

# Check if system image exists
if [[ -f "$SYSIMAGE_PATH" ]]; then
    echo "✓ System image found: $SYSIMAGE_PATH"
    echo "  Size: $(ls -lh $SYSIMAGE_PATH | awk '{print $5}')"
    JULIA_FLAGS="--startup-file=no --sysimage=$SYSIMAGE_PATH --project=."
else
    echo "⚠️  No system image found, using standard Julia startup"
    JULIA_FLAGS="--project=."
fi

echo "Julia flags: $JULIA_FLAGS"
echo "Using config: $CONFIG_FILE"
echo ""

# Run the Julia script (SlurmClusterManager handles worker spawning)
julia $JULIA_FLAGS $JULIA_SCRIPT

echo ""
echo "========================================================================"
echo "MPI SEARCH JOB COMPLETED - MODE: $MODE"
echo "Finished on $(hostname) at $(date)"
echo "========================================================================"
