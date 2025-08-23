# Complementarity analysis (Counterfactual 4)
# src/structural_model_heterogenous_preferences/counterfactuals/complementarity_analysis.jl

using Optim
using CSV

"""
Solve the model with given phi and nu values while recalibrating kappa0 
to maintain target unemployment rate.
"""
function solve_with_recalibrated_kappa(prim_base, res_base, phi_val, nu_val, target_unemp_rate)
    # Objective function for the inner loop: distance to target unemployment
    function unemp_distance(kappa_guess)
        params = Dict(:ϕ => phi_val, :ν => nu_val, :κ₀ => kappa_guess[1])
        try
            _, res_temp = update_params_and_resolve(prim_base, res_base; params_to_update=params)
            model_unemp_rate = 1.0 - sum(res_temp.n) # Assuming total labor force is 1
            return (model_unemp_rate - target_unemp_rate)^2
        catch
            return 1e6  # Large penalty if model fails to solve
        end
    end
    
    # Use an optimizer to find the kappa0 that minimizes the distance
    initial_kappa_guess = [prim_base.κ₀]
    opt_result = optimize(unemp_distance, initial_kappa_guess, LBFGS())
    best_kappa = Optim.minimizer(opt_result)[1]
    
    # Return the final solved model with the correct kappa
    final_params = Dict(:ϕ => phi_val, :ν => nu_val, :κ₀ => best_kappa)
    return update_params_and_resolve(prim_base, res_base; params_to_update=final_params)
end

"""
Run the complementarity grid experiment.
"""
function run_complementarity_experiment(prim_2024, res_2024; 
                                       phi_grid_size=5, nu_grid_size=5, 
                                       phi_range=0.2, nu_range=0.2)
    
    println("-"^60)
    println("STEP 3: RUNNING COMPLEMENTARITY GRID EXPERIMENT")
    
    # Get baseline values and target unemployment rate
    phi_base = prim_2024.ϕ
    nu_base = prim_2024.ν
    target_unemp = 1.0 - sum(res_2024.n)
    
    # Create grids around the baseline values
    phi_grid = range(phi_base - phi_range, phi_base + phi_range, length=phi_grid_size)
    nu_grid = range(nu_base - nu_range, nu_base + nu_range, length=nu_grid_size)
    
    println("ϕ grid: $(collect(phi_grid))")
    println("ν grid: $(collect(nu_grid))")
    println("Target unemployment rate: $target_unemp")
    
    # Initialize results storage
    results = DataFrame(
        phi = Float64[],
        nu = Float64[],
        mean_alpha = Float64[],
        agg_productivity = Float64[],
        var_logwage = Float64[],
        recalibrated_kappa = Float64[]
    )
    
    # Main loop
    total_iterations = length(phi_grid) * length(nu_grid)
    current_iter = 0
    
    for (i, phi_val) in enumerate(phi_grid)
        for (j, nu_val) in enumerate(nu_grid)
            current_iter += 1
            println("Iteration $current_iter/$total_iterations: ϕ=$phi_val, ν=$nu_val")
            
            try
                # Solve with recalibrated kappa
                prim_temp, res_temp = solve_with_recalibrated_kappa(
                    prim_2024, res_2024, phi_val, nu_val, target_unemp
                )
                
                # Compute moments
                moments_temp = compute_model_moments(prim_temp, res_temp)
                
                # Store results
                push!(results, (
                    phi = phi_val,
                    nu = nu_val,
                    mean_alpha = moments_temp[:mean_alpha],
                    agg_productivity = moments_temp[:agg_productivity],
                    var_logwage = moments_temp[:var_logwage],
                    recalibrated_kappa = prim_temp.κ₀
                ))
                
                println("  ✓ Success: α̅=$(round(moments_temp[:mean_alpha], digits=4)), κ₀=$(round(prim_temp.κ₀, digits=4))")
                
            catch e
                println("  ✗ Failed: $e")
                # Store NaN values for failed cases
                push!(results, (
                    phi = phi_val,
                    nu = nu_val,
                    mean_alpha = NaN,
                    agg_productivity = NaN,
                    var_logwage = NaN,
                    recalibrated_kappa = NaN
                ))
            end
        end
    end
    
    # Save results
    output_file = joinpath(dirname(@__FILE__), "complementarity_results.csv")
    CSV.write(output_file, results)
    println("\nResults saved to: $output_file")
    
    return results
end
