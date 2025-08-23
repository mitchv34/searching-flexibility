#!/bin/bash

# Quick Continuous MPI Search Monitoring Script
# Runs advanced analysis every 5 minutes for active development
# Usage: ./quick_monitor.sh [JOB_ID]

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
ENHANCED_MONITOR="$SCRIPT_DIR/enhanced_monitoring.jl"
LOG_FILE="$SCRIPT_DIR/../../../logs/quick_monitor.log"
RESULTS_DIR="$SCRIPT_DIR/output"
INTERVAL=300  # 5 minutes in seconds

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} ✅ $1"
}

print_error() {
    echo -e "${RED}[$(date '+%H:%M:%S')]${NC} ❌ $1"
}

print_waiting() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')]${NC} ⏳ $1"
}

print_info() {
    echo -e "${PURPLE}[$(date '+%H:%M:%S')]${NC} ℹ️  $1"
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

# Function to extract job ID from latest results file
extract_job_id() {
    local latest_file=$(ls "$RESULTS_DIR"/mpi_search_results_*.json 2>/dev/null | head -1)
    if [ -n "$latest_file" ]; then
        # Extract job ID from filename pattern: mpi_search_results_job<ID>_...
        local job_id=$(basename "$latest_file" | sed -n 's/.*_job\([0-9]\+\)_.*/\1/p')
        if [ -n "$job_id" ]; then
            echo "$job_id"
        else
            echo ""
        fi
    else
        echo ""
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

# Function to count results files
count_results() {
    ls "$RESULTS_DIR"/mpi_search_results_*.json 2>/dev/null | wc -l
}

# Create logs directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Trap signals for clean exit
cleanup() {
    print_status "🛑 Quick monitoring stopped"
    exit 0
}
trap cleanup SIGINT SIGTERM

print_status "🚀 Quick monitoring (1-minute intervals) - Press Ctrl+C to stop"
if [ -n "$JOB_ID" ]; then
    print_info "Monitoring job ID: $JOB_ID"
fi
echo "Quick monitoring started at $(date)" >> "$LOG_FILE"

iteration=0

while true; do
    iteration=$((iteration + 1))
    
    # Check job status
    JOB_STATUS=$(check_job_status)
    
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
            STATUS_ICON= "✅"
            JOB_STATE="Completed"
            ;;
        *)
            STATUS_ICON="⚪"
            JOB_STATE="$JOB_STATUS"
            ;;
    esac
    
    print_status "$STATUS_ICON Job: $JOB_STATE | Iteration #$iteration"
    
    # Check if results exist
    if check_results_exist; then
        RESULT_COUNT=$(count_results)
        JOB_NUM=$(extract_job_id)
        
        # Run enhanced progress monitoring first
        print_info "📊 Checking progress for job $([ -n "$JOB_NUM" ] && echo "$JOB_NUM" || echo "any")..."
        cd /project/high_tech_ind/searching-flexibility
        
        if [ -n "$JOB_NUM" ]; then
            MONITOR_CMD="julia --project=. \"$ENHANCED_MONITOR\" $JOB_NUM"
        else
            MONITOR_CMD="julia --project=. \"$ENHANCED_MONITOR\""
        fi
        
        # Run enhanced monitoring
        if eval "$MONITOR_CMD" >> "$LOG_FILE" 2>&1; then
            print_success "Progress monitoring completed"
        else
            print_error "Progress monitoring failed - check logs"
        fi
        
        # Also run detailed analysis if requested
        if [ "$JOB_STATE" = "Completed" ] || [ $((iteration % 5)) -eq 0 ]; then
            print_info "🎨 Running detailed analysis for $RESULT_COUNT results file(s)..."
            if [ -n "$JOB_NUM" ]; then
                ANALYSIS_CMD="julia --project=. \"$ANALYSIS_SCRIPT\" $JOB_NUM"
            else
                ANALYSIS_CMD="julia --project=. \"$ANALYSIS_SCRIPT\""
            fi
            
            if eval "$ANALYSIS_CMD" >> "$LOG_FILE" 2>&1; then
                print_success "Detailed analysis completed"
            else
                print_error "Detailed analysis failed - check logs"
            fi
        fi
    else
        if [ "$JOB_STATE" = "Completed" ]; then
            print_error "Job completed but no results found - check job logs"
            print_status "Exiting monitor..."
            break
        else
            print_waiting "No results yet - waiting for job to generate output..."
        fi
    fi
    
    # Exit if job is completed and we've processed results
    if [ "$JOB_STATE" = "Completed" ] && check_results_exist; then
        print_success "Job completed and final analysis done - monitoring finished!"
        break
    fi
    
    # Quick countdown
    for ((i=INTERVAL; i>0; i-=10)); do
        printf "\r${BLUE}[$(date '+%H:%M:%S')]${NC} Next check in: %ds   " $i
        sleep 10
    done
    printf "\r\033[K"
done

print_status "🏁 Quick monitoring finished"
