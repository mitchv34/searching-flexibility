#!/bin/bash

# Standalone MPI Search Progress Monitor
# Quick progress check for MPI parameter search jobs
# Usage: ./progress_monitor.sh [JOB_ID]

# Check if job ID provided
if [ -n "$1" ]; then
    JOB_ID="$1"
    echo "🎯 Checking progress for job: $JOB_ID"
else
    JOB_ID=""
    echo "🎯 Checking progress for any MPI search jobs"
fi

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENHANCED_MONITOR="$SCRIPT_DIR/enhanced_monitoring.jl"

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

print_info() {
    echo -e "${PURPLE}[$(date '+%H:%M:%S')]${NC} ℹ️  $1"
}

print_status "🚀 Starting progress check..."

# Change to project directory
cd /project/high_tech_ind/searching-flexibility

# Run enhanced monitoring
if [ -n "$JOB_ID" ]; then
    MONITOR_CMD="julia --project=. \"$ENHANCED_MONITOR\" $JOB_ID"
else
    MONITOR_CMD="julia --project=. \"$ENHANCED_MONITOR\""
fi

print_info "Running enhanced progress monitor..."

if eval "$MONITOR_CMD"; then
    print_success "Progress check completed successfully"
    
    # Also show current SLURM job status if job ID provided
    if [ -n "$JOB_ID" ]; then
        echo ""
        print_info "SLURM job status:"
        squeue -j "$JOB_ID" 2>/dev/null || echo "Job $JOB_ID not found in queue (may be completed)"
    else
        echo ""
        print_info "Current MPI search jobs:"
        squeue -u "$USER" --name=MPI_Search 2>/dev/null || echo "No active MPI search jobs found"
    fi
else
    print_error "Progress check failed"
    exit 1
fi

print_status "🏁 Progress check finished"
