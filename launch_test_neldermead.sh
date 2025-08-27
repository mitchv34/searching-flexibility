#!/bin/bash

# Submit NelderMead Test Job - 30 Independent Searches with 2 Max Iterations
# This script submits a SLURM array job to test the optimization pipeline

echo "=== NelderMead Test Job Submission ==="
echo "Submitting 30 independent NelderMead searches with 2 max iterations"
echo "Configuration: 3 SLURM array jobs, 10 candidates each"
echo "Expected runtime: < 1 hour per job"
echo ""

# Navigate to project root
cd /project/high_tech_ind/searching-flexibility

# Check if required files exist
CONFIG_FILE="src/structural_model_heterogenous_preferences/local_objective_search/test_neldermeand_config.yaml"
SLURM_SCRIPT="src/structural_model_heterogenous_preferences/local_objective_search/submit_test_neldermead.sh"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "❌ Error: SLURM script not found: $SLURM_SCRIPT"
    exit 1
fi

echo "✅ Configuration file: $CONFIG_FILE"
echo "✅ SLURM script: $SLURM_SCRIPT"
echo ""

# Submit the job
echo "Submitting SLURM array job..."
JOB_ID=$(sbatch "$SLURM_SCRIPT" | grep -o '[0-9]\+')

if [ $? -eq 0 ]; then
    echo "✅ Job submitted successfully!"
    echo "📋 Job ID: $JOB_ID"
    echo "📊 Array jobs: ${JOB_ID}_1, ${JOB_ID}_2, ${JOB_ID}_3"
    echo ""
    echo "📁 Monitor progress with:"
    echo "   squeue -u \$USER -j $JOB_ID"
    echo ""
    echo "📄 Check logs in:"
    echo "   src/structural_model_heterogenous_preferences/local_objective_search/slurm_output/test_run/"
    echo ""
    echo "📈 Results will be saved to:"
    echo "   src/structural_model_heterogenous_preferences/local_objective_search/output/test_run/"
    echo ""
    echo "⏱️  Expected completion: ~1 hour"
else
    echo "❌ Job submission failed!"
    exit 1
fi
