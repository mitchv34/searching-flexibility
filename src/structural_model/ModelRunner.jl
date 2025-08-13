#==========================================================================================
Module: ModelRunner.jl
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2025-02-27
Description: Contains functions to run and test the random search labor market model.
        Provides high-level interfaces for running the model with different configurations
        and parameters, as well as comprehensive testing and analysis workflows.
==========================================================================================#

module ModelRunner

import Term: install_term_stacktrace
using Term
install_term_stacktrace()

using Parameters, Printf, Statistics
using ..Types, ..ModelFunctions, ..ModelSolver, ..ModelPlotting

export run_model, run_model_test, calibrate_model!, print_model_results

#==========================================================================================
#? Main Model Running Functions
==========================================================================================#
"""
    run_model(config_file_path::String; verbose::Bool=true, calibrate::Bool=true)

High-level function to run the complete model workflow from configuration to results.

# Arguments
- `config_file_path::String`: Path to the YAML configuration file
- `verbose::Bool`: Whether to print detailed progress information
- `calibrate::Bool`: Whether to perform parameter calibration

# Returns
- `Tuple{Primitives, Results}`: The model primitives and converged results
"""
function run_model(config_file_path::String; verbose::Bool=true, calibrate::Bool=true)
    if verbose
        println(@bold @yellow "="^80)
        println(@bold @yellow "Running Random Search Labor Market Model")
        println(@bold @yellow "="^80)
        println(@bold @yellow "Loading configuration from: $config_file_path")
    end

    if !isfile(config_file_path)
        error("Configuration file not found at: $config_file_path")
    end

    # Initialize model primitives and results
    if verbose; println(@bold @green "Initializing model..."); end
    prim, _ = Types.initializeModel(config_file_path)
    
    # Calibrate model parameters if requested
    if calibrate
        if verbose; println(@bold @yellow "Calibrating model parameters..."); end
        calibrate_model!(prim)
        if verbose; println(@bold @green "Model calibration completed."); end
    end
    
    # Create initial results structure
    if verbose; println(@bold @yellow "Creating initial results structure..."); end
    results = Results(prim)
    if verbose; println(@bold @green "Initial results structure created."); end

    # Solve the model
    if verbose; println(@bold @yellow "\nSolving the model..."); end
    ModelSolver.solve_model(prim, results; verbose=verbose)
    if verbose; println(@bold @green "Model solved successfully."); end

    return prim, results
end

"""
    run_model(prim::Primitives; verbose::Bool=true, calibrate::Bool=false)

High-level function to run the complete model workflow using pre-initialized primitives.

# Arguments
- `prim::Primitives`: Pre-configured primitives
- `verbose::Bool`: Whether to print detailed progress information
- `calibrate::Bool`: Whether to perform parameter calibration (default is false for pre-initialized primitives)

# Returns
- `Tuple{Primitives, Results}`: The model primitives and converged results
"""
function run_model(prim::Primitives; verbose::Bool=true, calibrate::Bool=false)
    if verbose
        println(@bold @yellow "="^80)
        println(@bold @yellow "Running Random Search Labor Market Model with pre-initialized primitives")
        println(@bold @yellow "="^80)
    end

    # Calibrate model parameters if requested
    if calibrate
        if verbose; println(@bold @yellow "Calibrating model parameters..."); end
        calibrate_model!(prim)
        if verbose; println(@bold @green "Model calibration completed."); end
    end

    # Create initial results structure
    if verbose; println(@bold @yellow "Creating initial results structure..."); end
    results = Results(prim)
    if verbose; println(@bold @green "Initial results structure created."); end

    # Solve the model
    if verbose; println(@bold @yellow "\nSolving the model..."); end
    ModelSolver.solve_model(prim, results; verbose=verbose)
    if verbose; println(@bold @green "Model solved successfully."); end

    return prim, results
end

"""
    run_model_test(config_file_path::String=""; plot_results::Bool=true, save_plots::Bool=false)

Runs a comprehensive test of the model including solution and visualization.

# Arguments
- `config_file_path::String`: Path to configuration file (uses default if empty)
- `plot_results::Bool`: Whether to generate plots
- `save_plots::Bool`: Whether to save plots to disk

# Returns
- `Tuple{Primitives, Results}`: The model primitives and converged results
"""
function run_model_test(config_file_path::String=""; plot_results::Bool=true, save_plots::Bool=false)
    
    # Use default config path if not provided
    if isempty(config_file_path)
        config_file_path = abspath(joinpath(@__DIR__, "parameters", "initial", "parameters_julia_model_Post_COVID.yaml"))
    end
    
    println(@bold @yellow "="^80)
    println(@bold @yellow "Executing Model Solution Test Workflow...")
    println(@bold @yellow "="^80)
    
    # Run the model
    prim, results = run_model(config_file_path; verbose=true, calibrate=true)
    
    # Print comprehensive results
    print_model_results(prim, results)
    
    # Generate plots if requested
    if plot_results
        println(@bold @yellow "\nGenerating visualization plots...")
        generate_all_plots(prim, results; save_plots=save_plots)
        println(@bold @green "Plots generated successfully.")
    end
    
    println(@bold @green "\nTest workflow completed successfully.")
    println(@bold @yellow "="^80)
    
    return prim, results
end

#==========================================================================================
#? Model Calibration Functions
==========================================================================================#

"""
    calibrate_model!(prim::Primitives)

Calibrates model parameters based on the grid ranges and production function parameters.

This function performs the calibration described in the model documentation,
setting ψ₀ and φ parameters to ensure consistency between worker preferences
and firm technology.

# Arguments
- `prim::Primitives`: Model primitives object to be modified in-place
"""
function calibrate_model!(prim::Primitives; verbose::Bool=true)
    # Extract parameter values needed for calibration
    ψ_min = prim.ψ_grid.min
    ψ_max = prim.ψ_grid.max
    h_min = prim.h_grid.min
    h_max = prim.h_grid.max
    ν = prim.production_function.remote_efficiency.ν
    c₀ = prim.utility_function.c₀
    A₁ = prim.production_function.productivity.A₁
    
    # Calculate φ (phi) parameter
    numerator = ν * (ψ_max - ψ_min) + (c₀ / A₁) * (1/h_min - 1/h_max)
    denominator = log(h_max / h_min)
    phi_calibrated = numerator / denominator
    
    # Calculate ψ₀ parameter
    psi0_calibrated = ν * ψ_max - 1.0 + phi_calibrated * log(h_min) + c₀ / (A₁ * h_min)

    # Update the model primitives with calibrated values
    prim.production_function.remote_efficiency.ϕ = phi_calibrated
    prim.production_function.remote_efficiency.ψ₀ = psi0_calibrated
    if verbose
        @info "Model calibrated with φ = $(round(phi_calibrated, digits=4)), ψ₀ = $(round(psi0_calibrated, digits=4))"
    end
end


# """
#     calibrate_model!(prim::Primitives; verbose::Bool=true)

# Calibrates the model's technology parameters `φ` and `ψ₀` in place, based on 
# the economic restrictions defined in the `Primitives`.

# This function is specifically designed for comparative static experiments where `ν` 
# is varied. To ensure `ν` has a clear interpretation as the maximum relative 
# productivity of remote work, this calibration **enforces a normalized support 
# for `ψ` of `[0, 1]` in its calculations.**

# The function explicitly normalizes the `ψ_grid` to the [0,1] interval by 
# transforming all grid values in-place. This normalization ensures that `ν` 
# has a clear interpretation as the maximum relative productivity of remote work, 
# with `ψ_max = 1.0` being used in the formula for `ψ₀`. All calculations are 
# performed consistently within this normalized space.

# The function implements the following two-step strategy:
# 1.  **Pins `φ` (phi):** Calculates `φ` such that the slope of the hybrid 
#     threshold `underline_ψ(h)` is zero at `h_min`.
# 2.  **Pins `ψ₀` (psi0):** Given the new `φ`, calculates `ψ₀` such that the 
#     threshold curve passes through the single target point `(h_min, 1.0)`.

# # Arguments
# - `prim::Primitives`: The model's primitives struct, which will be modified in place.
# - `verbose::Bool=true`: If true, prints the calibrated values.
# """
# function calibrate_model!(
#     prim::Primitives;
#     pin_location_h::Float64=1.0,
#     pin_location_ψ::Float64=0.5,
#     verbose::Bool=true)
    
#     # --- Validate Inputs ---
#     if !(0.0 <= pin_location_h <= 1.0)
#         throw(ArgumentError("pin_location_h must be between 0.0 and 1.0."))
#     end
    
#     if !(0.0 <= pin_location_ψ <= 1.0)
#         throw(ArgumentError("pin_location_ψ must be between 0.0 and 1.0."))
#     end
    
#     # Extract parameter values needed for calibration
#     h_min = prim.h_grid.min
#     h_max = prim.h_grid.max
#     ν = prim.production_function.remote_efficiency.ν
#     c₀ = prim.utility_function.c₀
#     A₁ = prim.production_function.productivity.A₁
    
#     # Normalize ψ_grid to ensure it is on [0, 1]
#     # This is crucial for the calibration to work correctly
#     prim.ψ_grid.values = (prim.ψ_grid.values .- prim.ψ_grid.min) ./ (prim.ψ_grid.max - prim.ψ_grid.min)
#     prim.ψ_grid.min = 0.0
#     prim.ψ_grid.max = 1.0 
    
#     # Extract normalized ψ extremes
#     ψ_min = prim.ψ_grid.min
#     ψ_max = prim.ψ_grid.max
    
    
#     # Step 1: Calculate φ (phi) using the slope condition at h_min.
#     # This ensures the threshold curve is always decreasing for h > h_min.
#     # The formula is derived from setting d(underline_ψ)/dh = 0 at h = h_min.
#     phi_calibrated = c₀ / (A₁ * h_min)
    
#     # --- Step 2: Determine the target point (h_target, ψ_target) ---
#     # Linearly interpolate between the corner points based on pin_location.
#     h_target = (1.0 - pin_location_h) * h_min + pin_location_h * h_max
#     ψ_target = (1.0 - pin_location_ψ) * ψ_min + pin_location_ψ * ψ_max

#     # --- Step 3: Calculate ψ₀ using the general formula for the target point ---
#     # This is derived from setting underline_ψ(h_target) = ψ_target.
#     psi0_calibrated = ν * ψ_target - 1.0 + phi_calibrated * log(h_target) + c₀ / (A₁ * h_target)

#     # --- Update the model primitives with the newly calibrated values ---
#     prim.production_function.remote_efficiency.ϕ = phi_calibrated
#     prim.production_function.remote_efficiency.ψ₀ = psi0_calibrated

#     if verbose
#         @info "Model calibrated with ν = $(round(ν, digits=2)), φ = $(round(phi_calibrated, digits=4)), ψ₀ = $(round(psi0_calibrated, digits=4))"
#     end
# end

#==========================================================================================
#? Results Analysis and Reporting
==========================================================================================#

"""
    print_model_results(prim::Primitives, results::Results)

Prints a comprehensive summary of model results including key equilibrium objects
and consistency checks.

# Arguments
- `prim::Primitives`: Model primitives
- `results::Results`: Converged model results
"""
function print_model_results(prim::Primitives, results::Results)
    println(@bold @yellow "\n" * "="^80)
    println(@bold @yellow "           COMPREHENSIVE MODEL RESULTS           ")
    println(@bold @yellow "="^80)

    # Aggregate equilibrium variables
    println(@bold @blue "\n📊 AGGREGATE EQUILIBRIUM VARIABLES")
    println(@sprintf("  • Market Tightness (θ): %.6f", results.θ))
    println(@sprintf("  • Job Finding Rate (p): %.6f", results.p))
    println(@sprintf("  • Vacancy Filling Rate (q): %.6f", results.q))
    
    # Labor market stocks and flows
    println(@bold @blue "\n👥 LABOR MARKET COMPOSITION")
    if isdefined(results, :u) && !isempty(results.u)
        total_unemployment = sum(results.u)
        println(@sprintf("  • Total Unemployed Workers: %.6f", total_unemployment))
        
        if sum(prim.h_grid.pdf) > 0
            agg_unemployment_rate = total_unemployment / sum(prim.h_grid.pdf)
            println(@sprintf("  • Aggregate Unemployment Rate: %.4f%%", agg_unemployment_rate * 100))
        end
    end

    if isdefined(results, :v) && !isempty(results.v)
        f_ψ = prim.ψ_grid.pdf isa Function ? prim.ψ_grid.pdf.(prim.ψ_grid.values) : copy(prim.ψ_grid.pdf)
        total_vacancies = sum(results.v .* f_ψ)
        println(@sprintf("  • Total Vacancies: %.6f", total_vacancies))
    end
    
    if isdefined(results, :n) && !isempty(results.n)
        total_employment = sum(results.n)
        println(@sprintf("  • Total Employed Workers: %.6f", total_employment))
    end

    # Consistency checks
    println(@bold @blue "\n✅ CONSISTENCY CHECKS")
    if isdefined(results, :n) && !isempty(results.n) && isdefined(results, :u) && !isempty(results.u)
        total_mass = sum(results.n) + sum(results.u)
        mass_error = abs(total_mass - 1.0)
        
        if mass_error > 1e-6
            println(@bold @red @sprintf("  ⚠️  Mass Conservation Error: %.6f (should be ~0)", mass_error))
        else
            println(@bold @green @sprintf("  ✓  Mass Conservation: %.6f ≈ 1.0", total_mass))
        end
    end

    # Work arrangement distribution
    if isdefined(results, :α_policy) && !isempty(results.α_policy) && isdefined(results, :n)
        println(@bold @blue "\n🏠 WORK ARRANGEMENT DISTRIBUTION")
        
        # Calculate employment-weighted statistics
        total_emp = sum(results.n)
        if total_emp > 0
            weights = results.n ./ total_emp
            
            # Average remote work share
            avg_alpha = sum(results.α_policy .* weights)
            println(@sprintf("  • Average Remote Work Share: %.4f", avg_alpha))
            
            # Distribution by work type
            fully_remote = sum(weights[results.α_policy .≈ 1.0])
            fully_inperson = sum(weights[results.α_policy .≈ 0.0])
            hybrid = 1.0 - fully_remote - fully_inperson
            
            println(@sprintf("  • Fully Remote (α=1): %.2f%%", fully_remote * 100))
            println(@sprintf("  • Fully In-Person (α=0): %.2f%%", fully_inperson * 100))
            println(@sprintf("  • Hybrid (0<α<1): %.2f%%", hybrid * 100))
        end
    end

    # Wage and surplus statistics
    if isdefined(results, :w_policy) && isdefined(results, :S)
        println(@bold @blue "\n💰 WAGE AND SURPLUS STATISTICS")
        
        # Filter to active matches only
        active_matches = results.S .> 0
        if sum(active_matches) > 0
            active_wages = results.w_policy[active_matches]
            active_surplus = results.S[active_matches]
            
            println(@sprintf("  • Number of Active Match Types: %d", sum(active_matches)))
            println(@sprintf("  • Average Wage (Active Matches): %.4f", mean(active_wages)))
            println(@sprintf("  • Average Surplus (Active Matches): %.4f", mean(active_surplus)))
            println(@sprintf("  • Wage Range: [%.4f, %.4f]", minimum(active_wages), maximum(active_wages)))
            println(@sprintf("  • Surplus Range: [%.4f, %.4f]", minimum(active_surplus), maximum(active_surplus)))
        end
    end

    println(@bold @yellow "\n" * "="^80)
end

#==========================================================================================
#? Plotting and Visualization
==========================================================================================#

"""
    generate_all_plots(prim::Primitives, results::Results; save_plots::Bool=false)

Generates all standard model visualization plots.

# Arguments
- `prim::Primitives`: Model primitives
- `results::Results`: Model results
- `save_plots::Bool`: Whether to save plots to disk
"""
function generate_all_plots(prim::Primitives, results::Results; save_plots::Bool=false)
    
    plots_generated = []
    
    try
        # Core distribution plots
        println("  Generating employment distribution plot...")
        fig1 = ModelPlotting.plot_employment_distribution(results, prim)
        push!(plots_generated, ("employment_distribution", fig1))
        
        println("  Generating surplus function plot...")
        fig2 = ModelPlotting.plot_surplus_function(results, prim)
        push!(plots_generated, ("surplus_function", fig2))
        
        # Policy function plots
        println("  Generating alpha policy plot...")
        fig3 = ModelPlotting.plot_alpha_policy(results, prim)
        push!(plots_generated, ("alpha_policy", fig3))
        
        println("  Generating wage policy plot...")
        fig4 = ModelPlotting.plot_wage_policy(results, prim)
        push!(plots_generated, ("wage_policy", fig4))
        
        # Analysis plots
        println("  Generating outcomes by skill plot...")
        fig5 = ModelPlotting.plot_outcomes_by_skill(results, prim)
        push!(plots_generated, ("outcomes_by_skill", fig5))
        
        println("  Generating work arrangement regimes plot...")
        if isdefined(results, :ψ_bottom) && isdefined(results, :ψ_top)
            fig6 = ModelPlotting.plot_work_arrangement_regimes(results, prim)
            push!(plots_generated, ("work_arrangement_regimes", fig6))
        else
            println("    Skipping work arrangement regimes (ψ_bottom/ψ_top not available)")
        end
        
        println("  Generating alpha policy by firm type plot...")
        fig7 = ModelPlotting.plot_alpha_policy_by_firm_type(results, prim)
        push!(plots_generated, ("alpha_policy_by_firm_type", fig7))
        
        # Trade-off analysis
        println("  Generating wage-amenity trade-off plot...")
        fig8 = ModelPlotting.plot_wage_amenity_tradeoff(results, prim)
        push!(plots_generated, ("wage_amenity_tradeoff", fig8))
        
    catch e
        @warn "Error generating some plots: $e"
    end
    
    # Save plots if requested
    if save_plots && !isempty(plots_generated)
        println("  Saving plots to disk...")
        save_plots_to_disk(plots_generated)
    end
    
    println(@sprintf("  Successfully generated %d plots", length(plots_generated)))
    return plots_generated
end

"""
    save_plots_to_disk(plots_generated::Vector)

Saves generated plots to the default figures directory.

# Arguments
- `plots_generated::Vector`: Vector of (name, figure) tuples
"""
function save_plots_to_disk(plots_generated::Vector)
    figures_dir = "/Users/mitchv34/Work/WFH/src/structural/random_search_model/results/figures"
    
    # Create directory if it doesn't exist
    if !isdir(figures_dir)
        mkpath(figures_dir)
    end
    
    for (name, fig) in plots_generated
        try
            filepath = joinpath(figures_dir, "$(name).png")
            save(filepath, fig)
            println("    Saved: $(name).png")
        catch e
            @warn "Failed to save plot $name: $e"
        end
    end
end

end # module ModelRunner
