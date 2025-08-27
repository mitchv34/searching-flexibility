#!/bin/bash
#SBATCH --job-name=test_neldermead_30
#SBATCH --partition=econ-grad-short
#SBATCH --array=1-3                    # 3 jobs: 30 candidates / 10 per job
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_output/test_run/test_neldermead_%A_%a.out
#SBATCH --error=slurm_output/test_run/test_neldermead_%A_%a.err

# Job information
echo "=== SLURM Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID" 
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Cores: $SLURM_CPUS_PER_TASK"
echo "Memory: 32G"
echo "Started at: $(date)"
echo "=============================="

# Environment setup
module load julia/1.11.6 2>/dev/null || echo "Julia module not available, using system Julia"

# Set working directory
cd /project/high_tech_ind/searching-flexibility

# Activate conda environment
echo "Activating conda environment..."
source /software/anaconda/etc/profile.d/conda.sh
conda activate searching-flexibility

# Verify Julia installation
echo "Julia version: $(julia --version)"
echo "Working directory: $(pwd)"

# Create output directories
mkdir -p src/structural_model_heterogenous_preferences/local_objective_search/slurm_output/test_run
mkdir -p src/structural_model_heterogenous_preferences/local_objective_search/output/test_run

# Set environment variables for this array job
export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}
export LOCAL_GA_JOB_ID=3056705

# Calculate candidate range for this array task
# With 10 candidates per job and 3 jobs total:
# Job 1: candidates 1-10
# Job 2: candidates 11-20  
# Job 3: candidates 21-30
BATCH_SIZE=10
START_CANDIDATE=$(( ($SLURM_ARRAY_TASK_ID - 1) * $BATCH_SIZE + 1 ))
END_CANDIDATE=$(( $SLURM_ARRAY_TASK_ID * $BATCH_SIZE ))

# For the last job, make sure we don't exceed 30 total candidates
if [ $END_CANDIDATE -gt 30 ]; then
    END_CANDIDATE=30
fi

echo "Array task $SLURM_ARRAY_TASK_ID processing candidates $START_CANDIDATE to $END_CANDIDATE"

# Set Julia threads
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the optimization
echo "Starting Julia optimization at $(date)..."

# Check Julia environment and activate project
echo "Activating Julia project environment..."
julia --project=/project/high_tech_ind/searching-flexibility -e "using Pkg; Pkg.status()"

# Create a temporary config for this specific array task
TEMP_CONFIG="src/structural_model_heterogenous_preferences/local_objective_search/test_neldermeand_config_${SLURM_ARRAY_TASK_ID}.yaml"
cp src/structural_model_heterogenous_preferences/local_objective_search/test_neldermeand_config.yaml $TEMP_CONFIG

# Modify the temp config to use only the candidates for this array task
# We'll create a simple Julia script to do this
cat > modify_config_${SLURM_ARRAY_TASK_ID}.jl << EOF
using YAML

# Load the base config
config = YAML.load_file("$TEMP_CONFIG")

# Calculate how many candidates this job should process
total_candidates = 30
batch_size = 10
array_id = $SLURM_ARRAY_TASK_ID
start_idx = (array_id - 1) * batch_size + 1
end_idx = min(array_id * batch_size, total_candidates)
n_candidates_this_job = end_idx - start_idx + 1

# Update the config for this specific job
config["GlobalSearchResults"]["n_top_starts"] = n_candidates_this_job

# Modify output directory to include array task ID
config["Output"]["results_dir"] = "output/test_run/array_\$(array_id)"
config["Output"]["summary_file"] = "test_neldermead_summary_\$(array_id).json"
config["Output"]["detailed_prefix"] = "test_neldermead_start_\$(array_id)"

# Save the modified config
YAML.write_file("$TEMP_CONFIG", config)
println("Modified config for array task \$array_id to process \$n_candidates_this_job candidates")
EOF

# Run the config modification
julia --project=/project/high_tech_ind/searching-flexibility modify_config_${SLURM_ARRAY_TASK_ID}.jl

# Run the actual optimization with the modified config
julia --project=/project/high_tech_ind/searching-flexibility -e "
    # Set environment for this specific array task
    ENV[\"LOCAL_GA_JOB_ID\"] = \"3056705\"
    
    # Include and run the optimization
    include(\"src/structural_model_heterogenous_preferences/local_objective_search/run_search_production.jl\")
    
    # Run with modified config
    try
        results = main(Dict(
            \"config\" => \"$TEMP_CONFIG\",
            \"candidates\" => $END_CANDIDATE,  # This will be capped by config
            \"log-level\" => \"INFO\",
            \"dry-run\" => false
        ))
        
        println(\"\\n✅ Array task $SLURM_ARRAY_TASK_ID completed successfully!\")
        
        if results !== nothing
            results_df, summary = results
            println(\"📊 Results: \", nrow(results_df), \" trajectories completed\")
            println(\"🎯 Best objective: \", summary[\"best_objective\"])
            println(\"📈 Convergence rate: \", round(summary[\"convergence_rate\"] * 100, digits=1), \"%\")
        end
        
    catch e
        println(\"❌ Error in array task $SLURM_ARRAY_TASK_ID: \", e)
        rethrow(e)
    end
"

# Cleanup temporary files
rm -f $TEMP_CONFIG modify_config_${SLURM_ARRAY_TASK_ID}.jl

echo "Array task $SLURM_ARRAY_TASK_ID completed at $(date)"
echo "Output files saved to: output/test_run/array_${SLURM_ARRAY_TASK_ID}/"
