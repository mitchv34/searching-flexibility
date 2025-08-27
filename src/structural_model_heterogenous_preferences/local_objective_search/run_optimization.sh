#!/bin/bash

# run_optimization.sh
# Production script to run ensemble local optimization

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Default values
CONFIG_FILE="$SCRIPT_DIR/local_refine_config.yaml"
N_CANDIDATES=10
LOG_LEVEL="INFO"
DRY_RUN=false
JULIA_THREADS=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Production ensemble local optimization for GMM estimation.

OPTIONS:
    -c, --config FILE       Configuration file path (default: $CONFIG_FILE)
    -n, --candidates N      Number of diverse candidates (default: $N_CANDIDATES)
    -l, --log-level LEVEL   Log level: DEBUG, INFO, WARN, ERROR (default: $LOG_LEVEL)
    -t, --threads N         Number of Julia threads (default: auto-detect)
    -d, --dry-run          Run validation only, no optimization
    -h, --help             Show this help message

EXAMPLES:
    # Basic usage
    $0

    # With custom config and more candidates
    $0 --config my_config.yaml --candidates 20

    # Debug mode with specific thread count
    $0 --log-level DEBUG --threads 8

    # Dry run to validate configuration
    $0 --dry-run

ENVIRONMENT VARIABLES:
    LOCAL_GA_JOB_ID        Job ID for GA results (overrides config)
    JULIA_NUM_THREADS      Number of Julia threads (if not set via -t)

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -n|--candidates)
            N_CANDIDATES="$2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -t|--threads)
            JULIA_THREADS="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}" >&2
            show_help >&2
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${RED}Error: Configuration file not found: $CONFIG_FILE${NC}" >&2
    echo -e "${YELLOW}Hint: Use the template at $SCRIPT_DIR/local_refine_config_template.yaml${NC}" >&2
    exit 1
fi

if ! [[ "$N_CANDIDATES" =~ ^[0-9]+$ ]] || [[ "$N_CANDIDATES" -lt 1 ]]; then
    echo -e "${RED}Error: Number of candidates must be a positive integer${NC}" >&2
    exit 1
fi

if [[ ! "$LOG_LEVEL" =~ ^(DEBUG|INFO|WARN|ERROR)$ ]]; then
    echo -e "${RED}Error: Log level must be one of: DEBUG, INFO, WARN, ERROR${NC}" >&2
    exit 1
fi

# Set Julia thread count
if [[ "$JULIA_THREADS" -gt 0 ]]; then
    export JULIA_NUM_THREADS="$JULIA_THREADS"
elif [[ -z "${JULIA_NUM_THREADS:-}" ]]; then
    export JULIA_NUM_THREADS=$(nproc)
fi

# Create output directories
mkdir -p "$PROJECT_ROOT/output" "$PROJECT_ROOT/logs"

# Print configuration
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}PRODUCTION ENSEMBLE LOCAL OPTIMIZATION${NC}"
echo -e "${BLUE}================================================================${NC}"
echo -e "Configuration file: ${YELLOW}$CONFIG_FILE${NC}"
echo -e "Number of candidates: ${YELLOW}$N_CANDIDATES${NC}"
echo -e "Log level: ${YELLOW}$LOG_LEVEL${NC}"
echo -e "Julia threads: ${YELLOW}${JULIA_NUM_THREADS}${NC}"
echo -e "Dry run: ${YELLOW}$DRY_RUN${NC}"
echo -e "Script directory: ${YELLOW}$SCRIPT_DIR${NC}"
echo -e "Project root: ${YELLOW}$PROJECT_ROOT${NC}"
echo -e "${BLUE}================================================================${NC}"

# Check Julia availability
if ! command -v julia &> /dev/null; then
    echo -e "${RED}Error: Julia not found in PATH${NC}" >&2
    exit 1
fi

# Check Julia version
JULIA_VERSION=$(julia --version | grep -oP 'julia version \K[\d.]+')
echo -e "Julia version: ${GREEN}$JULIA_VERSION${NC}"

# Change to script directory for relative paths
cd "$SCRIPT_DIR"

# Build Julia command arguments
JULIA_ARGS=(
    "--config" "$CONFIG_FILE"
    "--candidates" "$N_CANDIDATES"
    "--log-level" "$LOG_LEVEL"
)

if [[ "$DRY_RUN" == true ]]; then
    JULIA_ARGS+=("--dry-run")
fi

# Run the optimization
echo -e "\n${GREEN}Starting optimization...${NC}"
echo "Command: julia run_search_production.jl ${JULIA_ARGS[*]}"

if julia run_search_production.jl "${JULIA_ARGS[@]}"; then
    echo -e "\n${GREEN}✅ Optimization completed successfully!${NC}"
    echo -e "📁 Results saved to: ${YELLOW}$PROJECT_ROOT/output/${NC}"
    echo -e "📋 Logs saved to: ${YELLOW}$PROJECT_ROOT/logs/${NC}"
    exit 0
else
    echo -e "\n${RED}❌ Optimization failed!${NC}" >&2
    echo -e "📋 Check log files in: ${YELLOW}$PROJECT_ROOT/logs/${NC}" >&2
    exit 1
fi
