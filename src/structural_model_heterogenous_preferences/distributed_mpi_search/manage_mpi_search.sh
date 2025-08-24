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
        print_status "Enhanced monitoring: $0 monitor $job_id"
        print_status "View live logs: $0 logs $job_id"
        
        # Save job ID for later reference
        echo "$job_id" > "$MPI_SEARCH_DIR/.last_job_id"
        
    else
        print_error "Failed to submit job"
        exit 1
    fi
}

# Function to monitor job status with enhanced monitoring
monitor_job() {
    if [[ -n "$1" ]]; then
        job_id="$1"
    elif [[ -f "$MPI_SEARCH_DIR/.last_job_id" ]]; then
        job_id=$(cat "$MPI_SEARCH_DIR/.last_job_id")
    else
        job_id=""  # Monitor any job if none specified
    fi
    
    print_header "Enhanced MPI Search Monitoring"
    
    # First show SLURM job status if job_id available
    if [[ -n "$job_id" ]]; then
        print_status "Checking SLURM status for job $job_id..."
        squeue -j "$job_id" 2>/dev/null || {
            print_status "Job $job_id not found in queue (may have completed)"
        }
        echo
    fi
    
    # Use enhanced monitoring script
    print_status "🚀 Launching enhanced monitoring..."
    cd "$MPI_SEARCH_DIR" || exit 1
    
    if [[ -n "$job_id" ]]; then
        julia --project="$PROJECT_ROOT" enhanced_monitoring.jl "$job_id"
    else
        julia --project="$PROJECT_ROOT" enhanced_monitoring.jl
    fi
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

# Function to run enhanced analysis
enhanced_analysis() {
    if [[ -n "$1" ]]; then
        job_id="$1"
    else
        job_id=""  # Analyze latest if none specified
    fi
    
    print_header "Enhanced MPI Search Analysis"
    
    cd "$MPI_SEARCH_DIR" || exit 1
    
    print_status "🎨 Running comprehensive analysis with publication-quality plots..."
    if [[ -n "$job_id" ]]; then
        julia --project="$PROJECT_ROOT" advanced_analysis.jl "$job_id"
    else
        julia --project="$PROJECT_ROOT" advanced_analysis.jl
    fi
}

# Function to analyze results (quick analysis)
analyze_results() {
    print_header "Quick MPI Search Results Summary"
    
    # Find most recent results file
    latest_result=$(find "$MPI_SEARCH_DIR/output/results" -name "*.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [[ -n "$latest_result" ]]; then
        print_status "Latest results file: $(basename "$latest_result")"
        
        # Use Julia to analyze results
        cd "$MPI_SEARCH_DIR" || exit 1
        julia --project="$PROJECT_ROOT" -e "
            using JSON3
            results = JSON3.read(read(\"$latest_result\", String))
            println(\"📊 SEARCH RESULTS SUMMARY\")
            println(\"=========================\")
            if haskey(results, :best_objective)
                println(\"Best objective value: \", results.best_objective)
            end
            if haskey(results, :best_params)
                println(\"Best parameters: \", results.best_params)
            end
            if haskey(results, :all_objectives)
                println(\"Total evaluations: \", length(results.all_objectives))
            end
            if haskey(results, :n_workers)
                println(\"Workers used: \", results.n_workers)
            end
            if haskey(results, :elapsed_time)
                println(\"Elapsed time: \", round(results.elapsed_time, digits=2), \" seconds\")
                if haskey(results, :all_objectives)
                    println(\"Evaluations per second: \", round(length(results.all_objectives) / results.elapsed_time, digits=2))
                end
            end
        "
    else
        print_warning "No results files found in $MPI_SEARCH_DIR/output/results"
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
    echo "  monitor [JOB_ID]    Enhanced real-time monitoring with progress tracking"
    echo "  logs [JOB_ID]       View live SLURM logs"
    echo "  cancel [JOB_ID]     Cancel running job"
    echo "  status              Show current configuration and status"
    echo "  analyze             Quick results summary"
    echo "  plots [JOB_ID]      Generate comprehensive analysis plots"
    echo "  cleanup             Clean up old files"
    echo "  help                Show this help message"
    echo
    echo "Environment Variables:"
    echo "  CONFIG_FILE         Configuration file name (default: mpi_search_config.yaml)"
    echo
    echo "Examples:"
    echo "  $0 submit                    # Submit new MPI search job"
    echo "  $0 monitor 123456           # Enhanced monitoring for specific job"
    echo "  $0 monitor                  # Monitor latest job automatically"
    echo "  $0 plots                    # Generate publication-quality analysis plots"
    echo "  $0 logs                     # View logs of last submitted job"
    echo "  CONFIG_FILE=test.yaml $0 submit  # Use custom config file"
    echo
    echo "Enhanced Features:"
    echo "  • Real-time progress tracking with ETA estimates"
    echo "  • Publication-quality diagnostic plots"
    echo "  • Automatic job ID detection"
    echo "  • Comprehensive performance analysis"
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
        "plots")
            enhanced_analysis "$2"
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
