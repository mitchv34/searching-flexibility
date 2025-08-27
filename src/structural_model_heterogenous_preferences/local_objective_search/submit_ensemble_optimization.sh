#!/bin/bash
#===============================================================================
# SLURM Ensemble Local Optimization Submission Script
# 
# This script reads configuration from local_refine_config.yaml and submits
# a SLURM array job for massive parallel local optimization.
#===============================================================================

set -e  # Exit on any error

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/local_refine_config.yaml"
SLURM_SCRIPT="${SCRIPT_DIR}/run_batch_optimization.slurm"

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ ERROR: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Function to extract values from YAML config
get_yaml_value() {
    local key="$1"
    local file="$2"
    # Simple YAML parser - works for basic key: value pairs
    grep "^[[:space:]]*${key}:" "$file" | sed 's/.*:[[:space:]]*//' | sed 's/[[:space:]]*$//' | sed 's/"//g'
}

echo "==============================================================================="
echo "🚀 SLURM ENSEMBLE LOCAL OPTIMIZATION SUBMISSION"
echo "==============================================================================="

# Read configuration from YAML
echo "📋 Reading configuration from: $CONFIG_FILE"

TOTAL_CANDIDATES=$(get_yaml_value "total_candidates" "$CONFIG_FILE")
BATCH_SIZE=$(get_yaml_value "batch_size" "$CONFIG_FILE")
JOB_NAME=$(get_yaml_value "job_name" "$CONFIG_FILE")
PARTITION=$(get_yaml_value "partition" "$CONFIG_FILE")
CORES_PER_JOB=$(get_yaml_value "cores_per_job" "$CONFIG_FILE")
MEMORY_PER_JOB=$(get_yaml_value "memory_per_job" "$CONFIG_FILE")
TIME_LIMIT=$(get_yaml_value "time_limit" "$CONFIG_FILE")
OUTPUT_DIR=$(get_yaml_value "output_dir" "$CONFIG_FILE")
LOG_PREFIX=$(get_yaml_value "log_prefix" "$CONFIG_FILE")
EMAIL=$(get_yaml_value "email" "$CONFIG_FILE")
EMAIL_TYPE=$(get_yaml_value "email_type" "$CONFIG_FILE")
ACCOUNT=$(get_yaml_value "account" "$CONFIG_FILE")
QOS=$(get_yaml_value "qos" "$CONFIG_FILE")
EXCLUSIVE=$(get_yaml_value "exclusive" "$CONFIG_FILE")
CONSTRAINT=$(get_yaml_value "constraint" "$CONFIG_FILE")

# Calculate number of batches
NUM_BATCHES=$(( (TOTAL_CANDIDATES + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "   📊 Optimization Parameters:"
echo "      • Total candidates: $TOTAL_CANDIDATES"
echo "      • Batch size: $BATCH_SIZE" 
echo "      • Number of batches: $NUM_BATCHES"
echo "      • Cores per job: $CORES_PER_JOB"
echo "      • Memory per job: $MEMORY_PER_JOB"
echo "      • Time limit: $TIME_LIMIT"

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "   📁 Output directory: $OUTPUT_DIR"

# Build SLURM options
SLURM_OPTS="--job-name=$JOB_NAME"
SLURM_OPTS="$SLURM_OPTS --array=1-$NUM_BATCHES"
SLURM_OPTS="$SLURM_OPTS --ntasks=$CORES_PER_JOB"
SLURM_OPTS="$SLURM_OPTS --mem=$MEMORY_PER_JOB"
SLURM_OPTS="$SLURM_OPTS --time=$TIME_LIMIT"
SLURM_OPTS="$SLURM_OPTS --output=${OUTPUT_DIR}/${LOG_PREFIX}_%A_%a.out"
SLURM_OPTS="$SLURM_OPTS --error=${OUTPUT_DIR}/${LOG_PREFIX}_%A_%a.err"

# Add optional parameters if specified
[[ -n "$PARTITION" && "$PARTITION" != "" ]] && SLURM_OPTS="$SLURM_OPTS --partition=$PARTITION"
[[ -n "$ACCOUNT" && "$ACCOUNT" != "" ]] && SLURM_OPTS="$SLURM_OPTS --account=$ACCOUNT"
[[ -n "$QOS" && "$QOS" != "" ]] && SLURM_OPTS="$SLURM_OPTS --qos=$QOS"
[[ -n "$CONSTRAINT" && "$CONSTRAINT" != "" ]] && SLURM_OPTS="$SLURM_OPTS --constraint=$CONSTRAINT"
[[ "$EXCLUSIVE" == "true" ]] && SLURM_OPTS="$SLURM_OPTS --exclusive"
[[ -n "$EMAIL" && "$EMAIL" != "" ]] && SLURM_OPTS="$SLURM_OPTS --mail-user=$EMAIL --mail-type=$EMAIL_TYPE"

echo ""
echo "🎯 SLURM Job Configuration:"
echo "   $SLURM_OPTS"

# Check if SLURM script exists
if [[ ! -f "$SLURM_SCRIPT" ]]; then
    echo "❌ ERROR: SLURM script not found: $SLURM_SCRIPT"
    echo "   Make sure run_batch_optimization.slurm exists in the same directory."
    exit 1
fi

# Submit the job
echo ""
echo "🚀 Submitting SLURM array job..."
JOB_ID=$(sbatch $SLURM_OPTS "$SLURM_SCRIPT" | grep -oP 'Submitted batch job \K\d+')

if [[ $? -eq 0 ]]; then
    echo "✅ Job submitted successfully!"
    echo "   📋 Job ID: $JOB_ID"
    echo "   📊 Array indices: 1-$NUM_BATCHES"
    echo "   ⏰ Estimated total runtime: $TIME_LIMIT per batch"
    echo ""
    echo "📡 Monitor your jobs with:"
    echo "   squeue -j $JOB_ID"
    echo "   squeue -u \$USER"
    echo ""
    echo "📁 Output files will be in: $OUTPUT_DIR"
    echo "   • Logs: ${LOG_PREFIX}_${JOB_ID}_*.out"
    echo "   • Errors: ${LOG_PREFIX}_${JOB_ID}_*.err"
    echo "   • Results: optimization_results_batch_*.json"
    echo ""
    echo "🔍 When complete, run the aggregation script:"
    echo "   julia aggregate_results.jl $JOB_ID"
else
    echo "❌ Job submission failed!"
    exit 1
fi

echo "==============================================================================="
