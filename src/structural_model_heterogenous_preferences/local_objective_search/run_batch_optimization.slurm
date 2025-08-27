#!/bin/bash
#===============================================================================
# SLURM Array Job Script for Ensemble Local Optimization
# 
# This script is submitted as a SLURM array job and runs one batch of
# local optimization starting points.
#===============================================================================

# SLURM directives are set by the submission script, but we can add defaults here
#SBATCH --job-name=local_opt_batch
#SBATCH --ntasks=32
#SBATCH --mem=64GB
#SBATCH --time=6:00:00

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/local_refine_config.yaml"

echo "==============================================================================="
echo "🔬 SLURM BATCH OPTIMIZATION - ARRAY TASK ${SLURM_ARRAY_TASK_ID}"
echo "==============================================================================="
echo "📅 Start time: $(date)"
echo "🖥️  Node: $(hostname)"
echo "🆔 Job ID: ${SLURM_JOB_ID}"
echo "📋 Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "🧮 Allocated CPUs: ${SLURM_NTASKS}"
echo "💾 Allocated Memory: ${SLURM_MEM_PER_NODE}MB"
echo "📁 Working Directory: $(pwd)"
echo "==============================================================================="

# Environment setup
echo "🔧 Setting up environment..."

# Load necessary modules (adjust for your cluster)
# module load julia/1.11
# module load gcc/11.2.0
# module load openmpi/4.1.1

# Set Julia threading
export JULIA_NUM_THREADS=${SLURM_NTASKS}
echo "   🧵 Julia threads: $JULIA_NUM_THREADS"

# Navigate to the project directory
cd "$SCRIPT_DIR"
echo "   📂 Working in: $(pwd)"

# Activate Julia environment if Manifest.toml exists
if [[ -f "../../../../Manifest.toml" ]]; then
    echo "   📦 Activating Julia environment..."
    export JULIA_PROJECT="../../../../"
fi

# Function to extract values from YAML config  
get_yaml_value() {
    local key="$1"
    local file="$2"
    grep "^[[:space:]]*${key}:" "$file" | sed 's/.*:[[:space:]]*//' | sed 's/[[:space:]]*$//' | sed 's/"//g'
}

# Read batch configuration
BATCH_SIZE=$(get_yaml_value "batch_size" "$CONFIG_FILE")
TOTAL_CANDIDATES=$(get_yaml_value "total_candidates" "$CONFIG_FILE")
OUTPUT_DIR=$(get_yaml_value "output_dir" "$CONFIG_FILE")

# Calculate batch range
BATCH_START=$(( (SLURM_ARRAY_TASK_ID - 1) * BATCH_SIZE + 1 ))
BATCH_END=$(( SLURM_ARRAY_TASK_ID * BATCH_SIZE ))
if [[ $BATCH_END -gt $TOTAL_CANDIDATES ]]; then
    BATCH_END=$TOTAL_CANDIDATES
fi
ACTUAL_BATCH_SIZE=$(( BATCH_END - BATCH_START + 1 ))

echo ""
echo "📊 Batch Configuration:"
echo "   • Batch ID: ${SLURM_ARRAY_TASK_ID}"
echo "   • Candidates: ${BATCH_START} to ${BATCH_END} (${ACTUAL_BATCH_SIZE} total)"
echo "   • Batch size: ${BATCH_SIZE}"
echo "   • Output directory: ${OUTPUT_DIR}"

# Create output directory for this batch
BATCH_OUTPUT_DIR="${OUTPUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$BATCH_OUTPUT_DIR"

# Create the Julia script arguments
JULIA_ARGS="$SLURM_ARRAY_TASK_ID $BATCH_START $BATCH_END $ACTUAL_BATCH_SIZE"

echo ""
echo "🚀 Starting optimization..."
echo "   ⏰ Start time: $(date)"

# Run the Julia optimization script
julia --project="$JULIA_PROJECT" run_batch_search.jl $JULIA_ARGS

EXIT_CODE=$?

echo ""
echo "⏱️  End time: $(date)"
echo "🎯 Exit code: $EXIT_CODE"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✅ Batch ${SLURM_ARRAY_TASK_ID} completed successfully!"
    
    # Move results to the main output directory
    if [[ -f "optimization_results_batch_${SLURM_ARRAY_TASK_ID}.json" ]]; then
        mv "optimization_results_batch_${SLURM_ARRAY_TASK_ID}.json" "$OUTPUT_DIR/"
        echo "   📊 Results saved to: ${OUTPUT_DIR}/optimization_results_batch_${SLURM_ARRAY_TASK_ID}.json"
    fi
    
    # Save summary information
    echo "{" > "${OUTPUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}_summary.json"
    echo "  \"batch_id\": ${SLURM_ARRAY_TASK_ID}," >> "${OUTPUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}_summary.json"
    echo "  \"job_id\": \"${SLURM_JOB_ID}\"," >> "${OUTPUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}_summary.json"
    echo "  \"node\": \"$(hostname)\"," >> "${OUTPUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}_summary.json"
    echo "  \"start_time\": \"$(date -Iseconds)\"," >> "${OUTPUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}_summary.json"
    echo "  \"candidates_range\": [${BATCH_START}, ${BATCH_END}]," >> "${OUTPUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}_summary.json"
    echo "  \"batch_size\": ${ACTUAL_BATCH_SIZE}," >> "${OUTPUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}_summary.json"
    echo "  \"exit_code\": ${EXIT_CODE}," >> "${OUTPUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}_summary.json"
    echo "  \"threads_used\": ${JULIA_NUM_THREADS}" >> "${OUTPUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}_summary.json"
    echo "}" >> "${OUTPUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}_summary.json"
    
else
    echo "❌ Batch ${SLURM_ARRAY_TASK_ID} failed with exit code: $EXIT_CODE"
    
    # Save error information
    echo "{" > "${OUTPUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}_error.json"
    echo "  \"batch_id\": ${SLURM_ARRAY_TASK_ID}," >> "${OUTPUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}_error.json"
    echo "  \"job_id\": \"${SLURM_JOB_ID}\"," >> "${OUTPUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}_error.json"
    echo "  \"node\": \"$(hostname)\"," >> "${OUTPUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}_error.json"
    echo "  \"error_time\": \"$(date -Iseconds)\"," >> "${OUTPUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}_error.json"
    echo "  \"exit_code\": ${EXIT_CODE}" >> "${OUTPUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}_error.json"
    echo "}" >> "${OUTPUT_DIR}/batch_${SLURM_ARRAY_TASK_ID}_error.json"
fi

echo "==============================================================================="
echo "🏁 Batch ${SLURM_ARRAY_TASK_ID} finished at $(date)"
echo "==============================================================================="

exit $EXIT_CODE
