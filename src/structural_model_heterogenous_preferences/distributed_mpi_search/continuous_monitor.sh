#!/bin/bash

# Continuous MPI Search Monitoring Script
# Runs advanced analysis every 5 minutes
# Usage: ./continuous_monitor.sh [JOB_ID]

# Check if job ID provided
if [ -n "$1" ]; then
    JOB_ID="$1"
    echo "Monitoring specific job: $JOB_ID"
else
    JOB_ID=""
    echo "Monitoring any MPI search jobs"
fi

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_SCRIPT="$SCRIPT_DIR/advanced_analysis.jl"
LOG_FILE="$SCRIPT_DIR/../../../logs/continuous_monitor.log"
RESULTS_DIR="$SCRIPT_DIR"
INTERVAL=300  # 5 minutes in seconds

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ✅ $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ❌ $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ⚠️  $1"
}

print_info() {
    echo -e "${PURPLE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ℹ️  $1"
}

print_waiting() {
    echo -e "${CYAN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ⏳ $1"
}

# Function to check job status
check_job_status() {
    if [ -n "$JOB_ID" ]; then
        # Check specific job
        JOB_STATUS=$(squeue -j "$JOB_ID" --noheader 2>/dev/null | awk '{print $5}')
        if [ -n "$JOB_STATUS" ]; then
            echo "$JOB_STATUS"
        else
            echo "COMPLETED"
        fi
    else
        # Check any MPI search jobs
        ACTIVE_JOBS=$(squeue -u "$USER" --name=MPI_Search --noheader 2>/dev/null | wc -l)
        if [ "$ACTIVE_JOBS" -gt 0 ]; then
            echo "RUNNING"
        else
            echo "COMPLETED"
        fi
    fi
}

# Function to check if results files exist
check_results_exist() {
    if ls "$RESULTS_DIR"/mpi_search_results_*.json 1> /dev/null 2>&1; then
        return 0  # Results exist
    else
        return 1  # No results yet
    fi
}

# Function to get detailed job info
get_job_info() {
    if [ -n "$JOB_ID" ]; then
        squeue -j "$JOB_ID" --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --noheader 2>/dev/null
    else
        squeue -u "$USER" --name=MPI_Search --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --noheader 2>/dev/null
    fi
}

# Function to count and analyze results
analyze_results() {
    RESULT_COUNT=$(ls "$RESULTS_DIR"/mpi_search_results_*.json 2>/dev/null | wc -l)
    if [ "$RESULT_COUNT" -gt 0 ]; then
        # Get most recent result file
        LATEST_RESULT=$(ls -t "$RESULTS_DIR"/mpi_search_results_*.json 2>/dev/null | head -1)
        # Try to extract basic info (if jq is available)
        if command -v jq >/dev/null 2>&1 && [ -f "$LATEST_RESULT" ]; then
            N_EVALS=$(jq -r '.n_evaluations // "N/A"' "$LATEST_RESULT" 2>/dev/null)
            BEST_OBJ=$(jq -r '.best_objective // "N/A"' "$LATEST_RESULT" 2>/dev/null)
            STATUS=$(jq -r '.status // "N/A"' "$LATEST_RESULT" 2>/dev/null)
            echo "Results: $RESULT_COUNT file(s) | Evals: $N_EVALS | Best: $BEST_OBJ | Status: $STATUS"
        else
            echo "Results: $RESULT_COUNT file(s) found"
        fi
    else
        echo "No results files found"
    fi
}

# Create logs directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Trap signals for clean exit
cleanup() {
    print_status "🛑 Monitoring stopped by user"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Check if analysis script exists
if [ ! -f "$ANALYSIS_SCRIPT" ]; then
    print_error "Analysis script not found: $ANALYSIS_SCRIPT"
    exit 1
fi

print_status "🔍 Starting continuous MPI search monitoring"
if [ -n "$JOB_ID" ]; then
    print_info "Monitoring specific job ID: $JOB_ID"
fi
print_status "📊 Analysis script: $ANALYSIS_SCRIPT"
print_status "📝 Log file: $LOG_FILE"
print_status "⏰ Interval: ${INTERVAL}s (5 minutes)"
print_status "🛑 Press Ctrl+C to stop monitoring"
echo "=" >> "$LOG_FILE"
echo "Continuous monitoring started at $(date)" >> "$LOG_FILE"
if [ -n "$JOB_ID" ]; then
    echo "Monitoring job ID: $JOB_ID" >> "$LOG_FILE"
fi
echo "=" >> "$LOG_FILE"

iteration=0

while true; do
    iteration=$((iteration + 1))
    
    # Check job status
    JOB_STATUS=$(check_job_status)
    JOB_INFO=$(get_job_info)
    
    # Set status icon based on job status
    case "$JOB_STATUS" in
        "R"|"RUNNING")
            STATUS_ICON="🟢"
            JOB_STATE="Running"
            ;;
        "PD")
            STATUS_ICON="🟡"
            JOB_STATE="Pending"
            ;;
        "COMPLETED"|"")
            STATUS_ICON="🔴"
            JOB_STATE="Completed"
            ;;
        *)
            STATUS_ICON="⚪"
            JOB_STATE="$JOB_STATUS"
            ;;
    esac
    
    print_status "🔄 Analysis iteration $iteration | $STATUS_ICON Job: $JOB_STATE"
    
    if [ -n "$JOB_INFO" ]; then
        print_info "Job details: $JOB_INFO"
    fi
    
    # Check if results exist before running analysis
    if check_results_exist; then
        RESULTS_SUMMARY=$(analyze_results)
        print_info "$RESULTS_SUMMARY"
        
        print_status "📊 Running advanced analysis..."
        START_TIME=$(date +%s)
        
        # Activate Julia environment and run analysis
        cd /project/high_tech_ind/searching-flexibility
        
        if julia --project=. "$ANALYSIS_SCRIPT" >> "$LOG_FILE" 2>&1; then
            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))
            print_success "Analysis completed in ${DURATION}s"
            echo "Analysis iteration $iteration completed successfully at $(date) (${DURATION}s)" >> "$LOG_FILE"
            
            # Check plot generation
            PLOTS_DIR="/project/high_tech_ind/searching-flexibility/figures/mpi_analysis"
            if [ -d "$PLOTS_DIR" ]; then
                PLOTS_COUNT=$(find "$PLOTS_DIR" -name "*.png" -type f 2>/dev/null | wc -l)
                if [ "$PLOTS_COUNT" -gt 0 ]; then
                    print_success "Generated $PLOTS_COUNT analysis plots"
                fi
            fi
        else
            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))
            print_error "Analysis failed after ${DURATION}s (check log for details)"
            echo "Analysis iteration $iteration FAILED at $(date) (${DURATION}s)" >> "$LOG_FILE"
        fi
    else
        if [ "$JOB_STATE" = "Completed" ]; then
            print_error "Job completed but no results found - check job logs"
            print_status "Exiting monitor..."
            break
        else
            print_waiting "No results yet - waiting for job to generate output..."
            echo "Iteration $iteration: No results available yet" >> "$LOG_FILE"
        fi
    fi
    
    # Exit if job is completed and we've processed results
    if [ "$JOB_STATE" = "Completed" ] && check_results_exist; then
        print_success "Job completed and final analysis done - monitoring finished!"
        break
    fi
    
    print_status "💤 Waiting ${INTERVAL}s before next check..."
    echo "" >> "$LOG_FILE"
    
    # Sleep with countdown
    for ((i=INTERVAL; i>0; i-=30)); do
        minutes=$((i/60))
        seconds=$((i%60))
        if [ $minutes -gt 0 ]; then
            printf "\r${BLUE}[$(date '+%H:%M:%S')]${NC} Next check in: %dm %02ds   " $minutes $seconds
        else
            printf "\r${BLUE}[$(date '+%H:%M:%S')]${NC} Next check in: %ds   " $seconds
        fi
        sleep 30
    done
    printf "\r\033[K"  # Clear the countdown line
done

print_status "🏁 Continuous monitoring finished"
