# Return-to-Office (RTO) mandate analysis (Counterfactual 5)
# src/structural_model_heterogenous_preferences/counterfactuals/rto_analysis.jl

"""
Run the RTO mandate experiment by constraining maximum alpha.
This requires a modified solver that caps the integration bounds.
"""
function run_rto_experiment(prim_2024, res_2024; alpha_max_values=[0.4, 0.6, 0.8])
    
    println("-"^60)
    println("STEP 4: RUNNING RTO MANDATE EXPERIMENT")
    
    # Get baseline moments
    moments_baseline = compute_model_moments(prim_2024, res_2024)
    
    println("Baseline (no constraint): α̅ = $(round(moments_baseline[:mean_alpha], digits=4))")
    
    # Initialize results storage
    results = DataFrame(
        alpha_max = Float64[],
        mean_alpha = Float64[],
        agg_productivity = Float64[],
        var_logwage = Float64[],
        change_mean_alpha = Float64[],
        change_agg_productivity = Float64[],
        change_var_logwage = Float64[]
    )
    
    # Add baseline (unconstrained) case
    push!(results, (
        alpha_max = 1.0,
        mean_alpha = moments_baseline[:mean_alpha],
        agg_productivity = moments_baseline[:agg_productivity],
        var_logwage = moments_baseline[:var_logwage],
        change_mean_alpha = 0.0,
        change_agg_productivity = 0.0,
        change_var_logwage = 0.0
    ))
    
    # Run constrained cases
    for alpha_max in alpha_max_values
        println("\nSolving with α_max = $alpha_max...")
        
        try
            # Solve model with alpha constraint
            # Note: This requires modifying the ModelSolver to accept alpha_max parameter
            prim_constrained, res_constrained = solve_model_with_alpha_constraint(
                prim_2024, alpha_max
            )
            
            # Compute moments
            moments_constrained = compute_model_moments(prim_constrained, res_constrained)
            
            # Calculate changes relative to baseline
            change_alpha = moments_constrained[:mean_alpha] - moments_baseline[:mean_alpha]
            change_prod = moments_constrained[:agg_productivity] - moments_baseline[:agg_productivity]
            change_ineq = moments_constrained[:var_logwage] - moments_baseline[:var_logwage]
            
            # Store results
            push!(results, (
                alpha_max = alpha_max,
                mean_alpha = moments_constrained[:mean_alpha],
                agg_productivity = moments_constrained[:agg_productivity],
                var_logwage = moments_constrained[:var_logwage],
                change_mean_alpha = change_alpha,
                change_agg_productivity = change_prod,
                change_var_logwage = change_ineq
            ))
            
            println("  ✓ Success: α̅=$(round(moments_constrained[:mean_alpha], digits=4))")
            println("    Changes: Δα̅=$(round(change_alpha, digits=4)), ΔProd=$(round(change_prod, digits=4))")
            
        catch e
            println("  ✗ Failed for α_max = $alpha_max: $e")
            # Store NaN values for failed cases
            push!(results, (
                alpha_max = alpha_max,
                mean_alpha = NaN,
                agg_productivity = NaN,
                var_logwage = NaN,
                change_mean_alpha = NaN,
                change_agg_productivity = NaN,
                change_var_logwage = NaN
            ))
        end
    end
    
    # Save results
    output_file = joinpath(dirname(@__FILE__), "rto_results.csv")
    CSV.write(output_file, results)
    println("\nRTO experiment results saved to: $output_file")
    
    return results
end

"""
Solve the model with a constraint on maximum alpha.
This is a wrapper that calls the modified solver.
"""
function solve_model_with_alpha_constraint(prim, alpha_max)
    # Create a copy of primitives
    prim_constrained = deepcopy(prim)
    
    # Solve with the alpha constraint
    # Note: This function needs to be implemented in ModelSolver.jl
    res_constrained = solve_model_constrained(prim_constrained; alpha_max=alpha_max)
    
    return prim_constrained, res_constrained
end
