#==========================================================================================
Main Entry Point: main.jl
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2025-02-27
Description: Main entry point for the modular random search labor market model.
        Demonstrates how to use the separated modules for solving, plotting, and 
        running the model.
==========================================================================================#

# Import all necessary modules
include("model_functions.jl")
include("types.jl")
include("ModelSolver.jl")
include("ModelPlotting.jl") 
include("ModelRunner.jl")

using .Types, .ModelFunctions, .ModelSolver, .ModelPlotting, .ModelRunner

using Statistics

#==========================================================================================
#? Example Usage and Demonstrations
==========================================================================================#

"""
    main()

Main function demonstrating the modular usage of the random search labor market model.
Shows different ways to use the separated modules.
"""
function main()
    println("="^80)
    println("MODULAR RANDOM SEARCH LABOR MARKET MODEL")
    println("="^80)
    
    # Example 1: Quick test run using the high-level runner
    println("\n🚀 Example 1: Quick Test Run")
    prim, results = ModelRunner.run_model_test(; plot_results=true, save_plots=false)
    
    # Example 2: Step-by-step manual workflow
    println("\n🔧 Example 2: Manual Step-by-Step Workflow")
    demonstrate_manual_workflow()
    
    # Example 3: Custom plotting workflow
    println("\n📊 Example 3: Custom Plotting Workflow")
    demonstrate_custom_plotting(prim, results)
    
    # Example 4: Parameter sensitivity analysis
    println("\n📈 Example 4: Parameter Sensitivity Analysis") 
    demonstrate_sensitivity_analysis()
    
    println("\n✅ All examples completed successfully!")
end

"""
    demonstrate_manual_workflow()

Shows how to use the individual modules step-by-step for full control over the process.
"""
function demonstrate_manual_workflow()
    println("  Setting up model manually...")
    
    # Step 1: Load configuration and initialize primitives
    config_path = abspath(joinpath(@__DIR__, "parameters", "initial", "parameters_julia_model_Post_COVID.yaml"))
    prim, _ = Types.initializeModel(config_path)
    
    # Step 2: Perform calibration
    ModelRunner.calibrate_model!(prim)
    
    # Step 3: Create results structure
    results = Results(prim)
    
    # Step 4: Solve the model with custom parameters
    println("  Solving model with custom tolerance...")
    ModelSolver.solve_model(prim, results; 
                          tol=1e-8,           # Tighter tolerance
                          max_iter=1000,      # More iterations
                          verbose=false,      # Silent solve
                          λ_S=0.05,          # Slower damping
                          λ_u=0.05)          # Slower damping
    
    # Step 5: Analyze results
    println("  Analyzing results...")
    ModelRunner.print_model_results(prim, results)
    
    println("  ✓ Manual workflow completed")
end

"""
    demonstrate_custom_plotting(prim::Primitives, results::Results)

Shows how to create custom visualizations using the plotting module.
"""
function demonstrate_custom_plotting(prim::Primitives, results::Results)
    println("  Creating custom plots...")
    
    try
        # Create individual plots with custom styling
        fig1 = ModelPlotting.plot_employment_distribution(results, prim)
        fig2 = ModelPlotting.plot_surplus_function(results, prim)
        
        # Create specialized analysis plots
        n_h = prim.h_grid.n
        n_ψ = prim.ψ_grid.n
        
        # Choose specific indices for detailed analysis
        h_idx_for_analysis = round(Int, n_h * 0.75)  # High-skill worker
        ψ_indices_for_analysis = [
            round(Int, n_ψ * 0.2), 
            round(Int, n_ψ * 0.5),
            round(Int, n_ψ * 0.8)
        ]
        
        # Create the alpha derivation plot if we have required functions
        try
            fig3 = ModelPlotting.plot_alpha_derivation_and_policy(
                prim, results,
                h_idx_fixed=h_idx_for_analysis,
                ψ_indices_to_vary=ψ_indices_for_analysis
            )
            println("  ✓ Created alpha derivation plot")
        catch e
            println("  ⚠️  Alpha derivation plot skipped: $e")
        end
        
        println("  ✓ Custom plotting completed")
    catch e
        println("  ⚠️  Some custom plots failed: $e")
    end
end

"""
    demonstrate_sensitivity_analysis()

Shows how to run the model with different parameter configurations for sensitivity analysis.
"""
function demonstrate_sensitivity_analysis()
    println("  Running parameter sensitivity analysis...")
    
    # Load base configuration
    config_path = abspath(joinpath(@__DIR__, "parameters", "initial", "parameters_julia_model_Post_COVID.yaml"))
    
    try
        # Test with different bargaining power parameters
        bargaining_powers = [0.3, 0.5, 0.7]
        
        for (i, ξ) in enumerate(bargaining_powers)
            println("    Testing with bargaining power ξ = $ξ")
            
            # Initialize model
            prim, _ = Types.initializeModel(config_path)
            
            # Modify parameter
            prim.ξ = ξ
            
            # Calibrate and solve
            ModelRunner.calibrate_model!(prim)
            results = Results(prim)
            ModelSolver.solve_model(prim, results; verbose=false)
            
            # Extract key statistics
            unemployment_rate = sum(results.u) / sum(prim.h_grid.pdf)
            avg_alpha = sum(results.α_policy .* results.n) / sum(results.n)
            
            println("      → Unemployment rate: $(round(unemployment_rate*100, digits=2))%")
            println("      → Average remote share: $(round(avg_alpha, digits=3))")
        end
        
        println("  ✓ Sensitivity analysis completed")
    catch e
        println("  ⚠️  Sensitivity analysis failed: $e")
    end
end

#==========================================================================================
#? Utility Functions for Advanced Usage
==========================================================================================#

"""
    solve_model_with_config(config_dict::Dict; kwargs...)

Solves the model using a dictionary configuration instead of a YAML file.
Useful for programmatic parameter sweeps.
"""
function solve_model_with_config(config_dict::Dict; kwargs...)
    # This would require extending the Types module to accept Dict inputs
    # For now, this serves as a template for future extensions
    @warn "solve_model_with_config not yet implemented - requires extending Types.initializeModel"
    return nothing
end

"""
    compare_models(config_paths::Vector{String}; plot_comparison=true)

Compares results across multiple model configurations.
"""
function compare_models(config_paths::Vector{String}; plot_comparison=true)
    results_list = []
    
    for (i, path) in enumerate(config_paths)
        println("Solving model $i with config: $path")
        prim, results = ModelRunner.run_model(path; verbose=false)
        push!(results_list, (prim, results, path))
    end
    
    # Extract key metrics for comparison
    println("\nModel Comparison Results:")
    println("-" * "=" * "-")
    
    for (i, (prim, results, path)) in enumerate(results_list)
        unemployment_rate = sum(results.u) / sum(prim.h_grid.pdf)
        market_tightness = results.θ
        
        println("Model $i ($(basename(path))):")
        println("  Unemployment Rate: $(round(unemployment_rate*100, digits=2))%")
        println("  Market Tightness: $(round(market_tightness, digits=4))")
    end
    
    return results_list
end

#==========================================================================================
#? Execute Main Function
==========================================================================================#

# Run the main function when this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
