#!/bin/bash

# Convenient wrapper to submit MPI search jobs with run numbers
# Usage: ./submit_run.sh [RUN_NUMBER] [MODE]

# Parse arguments
RUN_NUMBER=${1:-1}
MODE=${2:-standard}

echo "🚀 Submitting MPI search job"
echo "📊 Run number: $RUN_NUMBER"
echo "🔧 Mode: $MODE"

# Submit the job with the run number
sbatch --export=RUN_NUMBER=$RUN_NUMBER,MODE=$MODE submit_mpi_search.sbatch

echo ""
echo "Job submitted! Monitor with:"
echo "  ./quick_monitor.sh \$(squeue -u \$USER --noheader --format=\"%A\" | tail -1)"
echo ""
echo "Or check specific run results:"
echo "  julia --project=. advanced_analysis.jl $RUN_NUMBER"
