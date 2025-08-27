using YAML

# Load the base config
config = YAML.load_file("src/structural_model_heterogenous_preferences/local_objective_search/test_neldermeand_config_2.yaml")

# Calculate how many candidates this job should process
total_candidates = 30
batch_size = 10
array_id = 2
start_idx = (array_id - 1) * batch_size + 1
end_idx = min(array_id * batch_size, total_candidates)
n_candidates_this_job = end_idx - start_idx + 1

# Update the config for this specific job
config["GlobalSearchResults"]["n_top_starts"] = n_candidates_this_job

# Modify output directory to include array task ID
config["Output"]["results_dir"] = "output/test_run/array_$(array_id)"
config["Output"]["summary_file"] = "test_neldermead_summary_$(array_id).json"
config["Output"]["detailed_prefix"] = "test_neldermead_start_$(array_id)"

# Save the modified config
YAML.write_file("src/structural_model_heterogenous_preferences/local_objective_search/test_neldermeand_config_2.yaml", config)
println("Modified config for array task $array_id to process $n_candidates_this_job candidates")
