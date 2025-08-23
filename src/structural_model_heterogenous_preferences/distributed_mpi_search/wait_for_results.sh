#!/bin/bash

# Simple job monitoring script - waits for first results to appear
# Usage: ./wait_for_results.sh JOB_ID

JOB_ID=${1:-""}
if [ -z "$JOB_ID" ]; then
    echo "Usage: $0 JOB_ID"
    exit 1
fi

echo "🔍 Waiting for job $JOB_ID to start generating results..."
echo "⏰ Check started at $(date)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/output"

# Function to check if job-specific results exist
check_job_results() {
    local job_id="$1"
    ls "$RESULTS_DIR"/mpi_search_results_job${job_id}_*.json 2>/dev/null | head -1
}

# Function to check SLURM job status
check_slurm_status() {
    local job_id="$1"
    squeue -j "$job_id" --noheader 2>/dev/null | awk '{print $5}'
}

# Monitor loop
iteration=0
while true; do
    iteration=$((iteration + 1))
    
    # Check if job is still running
    job_status=$(check_slurm_status "$JOB_ID")
    if [ -z "$job_status" ]; then
        echo "❌ Job $JOB_ID is no longer in the queue (may have completed or failed)"
        break
    fi
    
    # Check for results
    results_file=$(check_job_results "$JOB_ID")
    if [ -n "$results_file" ]; then
        echo "✅ First results file found: $(basename "$results_file")"
        echo "🎉 Job $JOB_ID has started generating results!"
        echo "📊 You can now run: ./progress_monitor.sh $JOB_ID"
        break
    fi
    
    # Status update
    echo "[$(date '+%H:%M:%S')] Iteration $iteration - Job status: $job_status - No results yet..."
    
    # Wait 30 seconds before next check
    sleep 30
done

echo "🏁 Monitoring finished at $(date)"
