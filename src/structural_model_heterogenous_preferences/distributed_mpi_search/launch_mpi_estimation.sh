#!/bin/bash

# Comprehensive MPI Search Launcher with Enhanced Monitoring
# Handles full estimation workflow with real-time monitoring and analysis

set -e

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="/project/high_tech_ind/searching-flexibility"
MPI_SCRIPT="$PROJECT_ROOT/src/structural_model_heterogenous_preferences/distributed_mpi_search/mpi_search.jl"
MONITORING_SCRIPT="$PROJECT_ROOT/src/structural_model_heterogenous_preferences/distributed_mpi_search/enhanced_monitoring.jl"
ANALYSIS_SCRIPT="$PROJECT_ROOT/src/structural_model_heterogenous_preferences/distributed_mpi_search/advanced_analysis.jl"
CONFIG_FILE="$PROJECT_ROOT/src/structural_model_heterogenous_preferences/distributed_mpi_search/mpi_search_config.yaml"
OUTPUT_DIR="$PROJECT_ROOT/output"
LOGS_DIR="$OUTPUT_DIR/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
N_WORKERS=16
PARTITION="short"
TIME_LIMIT="02:00:00"
CONFIG_NAME="demanding_test_config"
MEMORY_PER_CPU="4G"
RUN_MONITORING=true
RUN_ANALYSIS=true
CLEANUP_OLD_LOGS=true

# Print usage
usage() {
    echo -e "${CYAN}MPI Search Launcher - Comprehensive Estimation Workflow${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -w, --workers N         Number of MPI workers (default: 16)"
    echo "  -p, --partition NAME    SLURM partition (default: short)"
    echo "  -t, --time LIMIT        Time limit (default: 02:00:00)"
    echo "  -c, --config NAME       Config file name (default: demanding_test_config)"
    echo "  -m, --memory SIZE       Memory per CPU (default: 4G)"
    echo "  --no-monitoring         Skip enhanced monitoring"
    echo "  --no-analysis          Skip post-analysis"
    echo "  --no-cleanup           Skip old log cleanup"
    echo "  -h, --help             Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Default run"
    echo "  $0 -w 32 -t 04:00:00                # 32 workers, 4-hour limit"
    echo "  $0 -c test_search_config --no-monitoring  # Different config, no monitoring"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -w|--workers)
            N_WORKERS="$2"
            shift 2
            ;;
        -p|--partition)
            PARTITION="$2"
            shift 2
            ;;
        -t|--time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_NAME="$2"
            CONFIG_FILE="$SCRIPT_DIR/${CONFIG_NAME}.yaml"
            shift 2
            ;;
        -m|--memory)
            MEMORY_PER_CPU="$2"
            shift 2
            ;;
        --no-monitoring)
            RUN_MONITORING=false
            shift
            ;;
        --no-analysis)
            RUN_ANALYSIS=false
            shift
            ;;
        --no-cleanup)
            CLEANUP_OLD_LOGS=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Print banner
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    MPI SEARCH COMPREHENSIVE LAUNCHER                         ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Validate configuration
echo -e "${BLUE}📋 Validating configuration...${NC}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
fi

if [ ! -f "$MPI_SCRIPT" ]; then
    echo -e "${RED}❌ MPI script not found: $MPI_SCRIPT${NC}"
    exit 1
fi

if [ ! -f "$MONITORING_SCRIPT" ]; then
    echo -e "${RED}❌ Monitoring script not found: $MONITORING_SCRIPT${NC}"
    RUN_MONITORING=false
    echo -e "${YELLOW}⚠️  Monitoring disabled${NC}"
fi

if [ ! -f "$ANALYSIS_SCRIPT" ]; then
    echo -e "${RED}❌ Analysis script not found: $ANALYSIS_SCRIPT${NC}"
    RUN_ANALYSIS=false
    echo -e "${YELLOW}⚠️  Analysis disabled${NC}"
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOGS_DIR"
mkdir -p "$OUTPUT_DIR/mpi_monitoring/plots"

echo -e "${GREEN}✅ Configuration validated${NC}"

# Display configuration
echo -e "${PURPLE}🔧 Configuration Summary:${NC}"
echo -e "   Workers:      ${N_WORKERS}"
echo -e "   Partition:    ${PARTITION}"
echo -e "   Time Limit:   ${TIME_LIMIT}"
echo -e "   Config:       ${CONFIG_NAME}"
echo -e "   Memory:       ${MEMORY_PER_CPU} per CPU"
echo -e "   Monitoring:   ${RUN_MONITORING}"
echo -e "   Analysis:     ${RUN_ANALYSIS}"
echo -e "   Cleanup:      ${CLEANUP_OLD_LOGS}"

# Cleanup old logs if requested
if [ "$CLEANUP_OLD_LOGS" = true ]; then
    echo -e "${YELLOW}🧹 Cleaning old logs...${NC}"
    find "$LOGS_DIR" -name "*.out" -mtime +7 -delete 2>/dev/null || true
    find "$LOGS_DIR" -name "*.err" -mtime +7 -delete 2>/dev/null || true
    echo -e "${GREEN}✅ Old logs cleaned${NC}"
fi

# Generate unique job identifier
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="mpi_search_${CONFIG_NAME}_${TIMESTAMP}"
SLURM_OUT="$LOGS_DIR/${JOB_NAME}.out"
SLURM_ERR="$LOGS_DIR/${JOB_NAME}.err"

# Create SLURM script
SLURM_SCRIPT="$LOGS_DIR/submit_${JOB_NAME}.sh"

echo -e "${BLUE}📝 Creating SLURM script: $(basename $SLURM_SCRIPT)${NC}"

cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
#SBATCH --ntasks=$N_WORKERS
#SBATCH --mem-per-cpu=$MEMORY_PER_CPU
#SBATCH --time=$TIME_LIMIT
#SBATCH --output=$SLURM_OUT
#SBATCH --error=$SLURM_ERR
#SBATCH --export=ALL

# Print job information
echo "=============================================="
echo "SLURM JOB INFORMATION"
echo "=============================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Job Name: \$SLURM_JOB_NAME"
echo "Node List: \$SLURM_JOB_NODELIST"
echo "Number of Tasks: \$SLURM_NTASKS"
echo "CPUs per Task: \$SLURM_CPUS_PER_TASK"
echo "Memory per CPU: \$SLURM_MEM_PER_CPU"
echo "Start Time: \$(date)"
echo "=============================================="

# Set up environment
cd "$PROJECT_ROOT"
export JULIA_PROJECT="$PROJECT_ROOT"

# Print Julia environment info
echo "Julia version: \$(julia --version)"
echo "Julia project: \$JULIA_PROJECT"
echo "MPI configuration:"
module list 2>&1 | grep -i mpi || echo "No MPI modules loaded"

# Run MPI search
echo ""
echo "🚀 Starting MPI search with configuration: $CONFIG_NAME"
echo "=============================================="

julia --project="$PROJECT_ROOT" "$MPI_SCRIPT" "$CONFIG_FILE"

echo ""
echo "=============================================="
echo "🏁 MPI search completed at: \$(date)"
echo "=============================================="
EOF

chmod +x "$SLURM_SCRIPT"

# Submit job and capture job ID
echo -e "${GREEN}🚀 Submitting SLURM job...${NC}"
JOB_ID=$(sbatch --parsable "$SLURM_SCRIPT")

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Job submitted successfully!${NC}"
    echo -e "   Job ID: ${JOB_ID}"
    echo -e "   Output: $(basename $SLURM_OUT)"
    echo -e "   Error:  $(basename $SLURM_ERR)"
else
    echo -e "${RED}❌ Failed to submit job${NC}"
    exit 1
fi

# Start monitoring if requested
if [ "$RUN_MONITORING" = true ]; then
    echo -e "${BLUE}📊 Starting enhanced monitoring...${NC}"
    
    # Create monitoring script launcher
    MONITOR_LAUNCHER="$LOGS_DIR/monitor_${JOB_NAME}.sh"
    cat > "$MONITOR_LAUNCHER" << EOF
#!/bin/bash
cd "$PROJECT_ROOT"
export JULIA_PROJECT="$PROJECT_ROOT"
julia --project="$PROJECT_ROOT" "$MONITORING_SCRIPT" --job-id=$JOB_ID --config="$CONFIG_FILE"
EOF
    chmod +x "$MONITOR_LAUNCHER"
    
    # Launch monitoring in background
    nohup "$MONITOR_LAUNCHER" > "$LOGS_DIR/monitoring_${JOB_NAME}.log" 2>&1 &
    MONITOR_PID=$!
    
    echo -e "${GREEN}✅ Monitoring started (PID: $MONITOR_PID)${NC}"
    echo -e "   Monitor log: monitoring_${JOB_NAME}.log"
fi

# Print status information
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                              JOB STATUS                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${PURPLE}📊 Current job status:${NC}"
squeue -j "$JOB_ID" -o "%.10i %.20j %.8T %.10M %.6D %.20R %.8C" 2>/dev/null || echo "Job status not available"

echo ""
echo -e "${BLUE}📋 Useful commands:${NC}"
echo -e "   Check job status:     squeue -j $JOB_ID"
echo -e "   View output:          tail -f $SLURM_OUT"
echo -e "   View errors:          tail -f $SLURM_ERR"
echo -e "   Cancel job:           scancel $JOB_ID"

if [ "$RUN_MONITORING" = true ]; then
    echo -e "   Monitor progress:     tail -f $LOGS_DIR/monitoring_${JOB_NAME}.log"
fi

# Wait for job completion if requested
echo ""
read -p "$(echo -e ${YELLOW}"Wait for job completion and run analysis? (y/N): "${NC})" -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}⏳ Waiting for job completion...${NC}"
    
    # Wait for job to complete
    while true; do
        JOB_STATE=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null || echo "COMPLETED")
        
        if [ "$JOB_STATE" = "COMPLETED" ] || [ "$JOB_STATE" = "FAILED" ] || [ "$JOB_STATE" = "CANCELLED" ] || [ -z "$JOB_STATE" ]; then
            break
        fi
        
        echo -e "${BLUE}🔄 Job status: $JOB_STATE${NC}"
        sleep 30
    done
    
    echo -e "${GREEN}🏁 Job completed with status: $JOB_STATE${NC}"
    
    # Run analysis if requested and job completed successfully
    if [ "$RUN_ANALYSIS" = true ] && [ "$JOB_STATE" = "COMPLETED" ]; then
        echo -e "${BLUE}📈 Running post-analysis...${NC}"
        
        cd "$PROJECT_ROOT"
        julia --project="$PROJECT_ROOT" "$ANALYSIS_SCRIPT"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Analysis completed successfully${NC}"
        else
            echo -e "${RED}❌ Analysis failed${NC}"
        fi
    fi
    
    # Show final summary
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                           EXECUTION SUMMARY                                  ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    echo -e "${PURPLE}📊 Job Summary:${NC}"
    echo -e "   Job ID:       $JOB_ID"
    echo -e "   Final Status: $JOB_STATE"
    echo -e "   Config:       $CONFIG_NAME"
    echo -e "   Workers:      $N_WORKERS"
    
    if [ -f "$SLURM_OUT" ]; then
        echo -e "   Output size:  $(du -h $SLURM_OUT | cut -f1)"
    fi
    
    echo -e "   Completed:    $(date)"
    
else
    echo -e "${YELLOW}💡 Job is running in background. Use the commands above to monitor progress.${NC}"
fi

echo -e "${GREEN}🎯 Launcher completed successfully!${NC}"
