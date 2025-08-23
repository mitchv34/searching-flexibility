#!/bin/bash

# MPI Distributed Search Management Script
# Provides utilities for submitting, monitoring, and managing distributed MPI parameter search

# Set project root
PROJECT_ROOT="/project/high_tech_ind/searching-flexibility"
MPI_SEARCH_DIR="$PROJECT_ROOT/src/structural_model_heterogenous_preferences/distributed_mpi_search"
LOGS_DIR="$PROJECT_ROOT/logs"
OUTPUT_DIR="$PROJECT_ROOT/output/mpi_search_results"

# Configuration
CONFIG_FILE="${CONFIG_FILE:-mpi_search_config.yaml}"
CONFIG_PATH="$MPI_SEARCH_DIR/$CONFIG_FILE"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Create necessary directories
mkdir -p "$LOGS_DIR"
mkdir -p "$OUTPUT_DIR"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_highlight() {
    echo -e "${CYAN}$1${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check if we're in the right directory
    if [[ ! -f "$PROJECT_ROOT/Project.toml" ]]; then
        print_error "Not in the correct project directory. Expected: $PROJECT_ROOT"
        exit 1
    fi
    
    # Check if configuration file exists
    if [[ ! -f "$CONFIG_PATH" ]]; then
        print_error "Configuration file not found: $CONFIG_PATH"
        exit 1
    fi
    
    # Check if MPI search script exists
    if [[ ! -f "$MPI_SEARCH_DIR/mpi_search.jl" ]]; then
        print_error "MPI search script not found: $MPI_SEARCH_DIR/mpi_search.jl"
        exit 1
    fi
    
    # Check if SLURM submission script exists
    if [[ ! -f "$MPI_SEARCH_DIR/submit_mpi_search.sbatch" ]]; then
        print_error "SLURM submission script not found: $MPI_SEARCH_DIR/submit_mpi_search.sbatch"
        exit 1
    fi
    
    print_status "✓ All prerequisites satisfied"
}

# Function to show current configuration
show_config() {
    print_header "Current Configuration"
    print_status "Config file: $CONFIG_PATH"
    print_status "MPI search directory: $MPI_SEARCH_DIR"
    print_status "Logs directory: $LOGS_DIR"
    print_status "Output directory: $OUTPUT_DIR"
    
    if command -v julia >/dev/null 2>&1; then
        julia_version=$(julia --version)
        print_status "Julia: $julia_version"
    else
        print_warning "Julia not found in PATH"
    fi
    
    if command -v sbatch >/dev/null 2>&1; then
        print_status "SLURM: Available"
    else
        print_warning "SLURM not available"
    fi
}

# Function to submit MPI search job
submit_job() {
    print_header "Submitting MPI Distributed Search Job"
    
    check_prerequisites
    
    # Change to MPI search directory
    cd "$MPI_SEARCH_DIR" || exit 1
    
    # Submit the job
    job_id=$(sbatch submit_mpi_search.sbatch | grep -o '[0-9]*$')
    
    if [[ -n "$job_id" ]]; then
        print_status "✓ Job submitted successfully"
        print_highlight "Job ID: $job_id"
        print_status "Monitor with: squeue -j $job_id"
        print_status "View logs: tail -f $LOGS_DIR/mpi_search_${job_id}.out"
        
        # Save job ID for later reference
        echo "$job_id" > "$MPI_SEARCH_DIR/.last_job_id"
        
    else
        print_error "Failed to submit job"
        exit 1
    fi
}

# Function to monitor job status
monitor_job() {
    if [[ -n "$1" ]]; then
        job_id="$1"
    elif [[ -f "$MPI_SEARCH_DIR/.last_job_id" ]]; then
        job_id=$(cat "$MPI_SEARCH_DIR/.last_job_id")
    else
        print_error "No job ID provided and no previous job found"
        exit 1
    fi
    
    print_header "Monitoring Job $job_id"
    
    # Show job status
    squeue -j "$job_id" 2>/dev/null || {
        print_status "Job $job_id not found in queue (may have completed)"
        
        # Check if job has output files
        out_file="$LOGS_DIR/mpi_search_${job_id}.out"
        err_file="$LOGS_DIR/mpi_search_${job_id}.err"
        
        if [[ -f "$out_file" ]]; then
            print_status "Output file exists: $out_file"
            echo "Last 20 lines:"
            tail -20 "$out_file"
        fi
        
        if [[ -f "$err_file" ]] && [[ -s "$err_file" ]]; then
            print_warning "Error file has content: $err_file"
            echo "Last 10 lines:"
            tail -10 "$err_file"
        fi
    }
}

# Function to view live logs
view_logs() {
    if [[ -n "$1" ]]; then
        job_id="$1"
    elif [[ -f "$MPI_SEARCH_DIR/.last_job_id" ]]; then
        job_id=$(cat "$MPI_SEARCH_DIR/.last_job_id")
    else
        print_error "No job ID provided and no previous job found"
        exit 1
    fi
    
    out_file="$LOGS_DIR/mpi_search_${job_id}.out"
    
    if [[ -f "$out_file" ]]; then
        print_status "Following log file: $out_file"
        print_status "Press Ctrl+C to stop"
        tail -f "$out_file"
    else
        print_error "Log file not found: $out_file"
    fi
}

# Function to cancel job
cancel_job() {
    if [[ -n "$1" ]]; then
        job_id="$1"
    elif [[ -f "$MPI_SEARCH_DIR/.last_job_id" ]]; then
        job_id=$(cat "$MPI_SEARCH_DIR/.last_job_id")
    else
        print_error "No job ID provided and no previous job found"
        exit 1
    fi
    
    print_header "Cancelling Job $job_id"
    
    if scancel "$job_id"; then
        print_status "✓ Job $job_id cancelled successfully"
    else
        print_error "Failed to cancel job $job_id"
    fi
}

# Function to analyze results
analyze_results() {
    print_header "Analyzing MPI Search Results"
    
    # Find most recent results file
    latest_result=$(find "$OUTPUT_DIR" -name "mpi_search_results_*.json" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [[ -n "$latest_result" ]]; then
        print_status "Latest results file: $latest_result"
        
        # Use Julia to analyze results
        cd "$MPI_SEARCH_DIR" || exit 1
        julia --project="$PROJECT_ROOT" -e "
            using JSON3
            results = JSON3.read(read(\"$latest_result\", String))
            println(\"📊 SEARCH RESULTS SUMMARY\")
            println(\"=========================\")
            println(\"Best objective value: \", results.best_objective)
            println(\"Best parameters: \", results.best_params)
            println(\"Total evaluations: \", length(results.all_objectives))
            println(\"Workers used: \", results.n_workers)
            println(\"Elapsed time: \", round(results.elapsed_time, digits=2), \" seconds\")
            println(\"Evaluations per second: \", round(length(results.all_objectives) / results.elapsed_time, digits=2))
        "
    else
        print_warning "No results files found in $OUTPUT_DIR"
    fi
}

# Function to clean up old files
cleanup() {
    print_header "Cleaning Up Old Files"
    
    # Remove old log files (older than 7 days)
    find "$LOGS_DIR" -name "mpi_search_*.out" -mtime +7 -delete 2>/dev/null
    find "$LOGS_DIR" -name "mpi_search_*.err" -mtime +7 -delete 2>/dev/null
    
    # Remove temporary files
    rm -f "$MPI_SEARCH_DIR/.last_job_id"
    
    print_status "✓ Cleanup completed"
}

# Function to show help
show_help() {
    print_header "MPI Distributed Search Management"
    echo
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "Commands:"
    echo "  submit              Submit MPI search job to SLURM"
    echo "  monitor [JOB_ID]    Monitor job status"
    echo "  logs [JOB_ID]       View live logs"
    echo "  cancel [JOB_ID]     Cancel running job"
    echo "  status              Show current configuration and status"
    echo "  analyze             Analyze latest results"
    echo "  cleanup             Clean up old files"
    echo "  help                Show this help message"
    echo
    echo "Environment Variables:"
    echo "  CONFIG_FILE         Configuration file name (default: mpi_search_config.yaml)"
    echo
    echo "Examples:"
    echo "  $0 submit                    # Submit new MPI search job"
    echo "  $0 monitor 123456           # Monitor specific job"
    echo "  $0 logs                     # View logs of last submitted job"
    echo "  CONFIG_FILE=test.yaml $0 submit  # Use custom config file"
}

# Main script logic
main() {
    case "${1:-help}" in
        "submit")
            submit_job
            ;;
        "monitor")
            monitor_job "$2"
            ;;
        "logs")
            view_logs "$2"
            ;;
        "cancel")
            cancel_job "$2"
            ;;
        "status")
            show_config
            ;;
        "analyze")
            analyze_results
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function with all arguments
main "$@"
