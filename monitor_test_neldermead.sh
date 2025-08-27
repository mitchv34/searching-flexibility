#!/bin/bash

# Monitor NelderMead Test Jobs
# Check status and show recent log outputs

echo "=== NelderMead Test Job Monitor ==="
echo "$(date)"
echo ""

# Check if any jobs are running
RUNNING_JOBS=$(squeue -u $USER --name=test_neldermead_30 --format="%.18i %.9P %.20j %.8u %.8T %.10M %.6D %R" --noheader 2>/dev/null)

if [ -n "$RUNNING_JOBS" ]; then
    echo "🏃 Currently running jobs:"
    echo "JobID            Partition   Name                 User     State    Time     Nodes Reason"
    echo "$RUNNING_JOBS"
    echo ""
else
    echo "💤 No currently running test_neldermead_30 jobs"
    echo ""
fi

# Check recent job history
echo "📈 Recent job history (last 10):"
sacct -u $USER --name=test_neldermead_30 --format="JobID,JobName%20,State,ExitCode,Start,End,Elapsed" --starttime=today 2>/dev/null | head -20
echo ""

# Check for output files
OUTPUT_DIR="src/structural_model_heterogenous_preferences/local_objective_search/slurm_output/test_run"
RESULTS_DIR="src/structural_model_heterogenous_preferences/local_objective_search/output/test_run"

if [ -d "$OUTPUT_DIR" ]; then
    echo "📄 SLURM output files:"
    ls -la "$OUTPUT_DIR" 2>/dev/null | head -10
    echo ""
    
    # Show tail of most recent log file
    LATEST_OUT=$(ls -t "$OUTPUT_DIR"/*.out 2>/dev/null | head -1)
    if [ -n "$LATEST_OUT" ]; then
        echo "📋 Latest output (last 20 lines from $(basename $LATEST_OUT)):"
        tail -20 "$LATEST_OUT" 2>/dev/null
        echo ""
    fi
fi

if [ -d "$RESULTS_DIR" ]; then
    echo "📊 Results files:"
    find "$RESULTS_DIR" -name "*.csv" -o -name "*.json" 2>/dev/null | head -10
    echo ""
fi

echo "🔄 To refresh: ./monitor_test_neldermead.sh"
echo "🛑 To cancel all: scancel -u \$USER --name=test_neldermead_30"
