#==========================================================================================
REPL-Friendly Main Entry Point: main_repl.jl
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2025-02-27
Description: REPL-friendly version of the modular random search labor market model.
        Run sections interactively in Julia REPL line by line.
        
USAGE IN REPL:
    julia> include("main_repl.jl")
    julia> setup_environment()      # Load modules and dependencies
    julia> run_example_1()          # Quick test run
    julia> run_example_2()          # Manual workflow
    julia> run_example_3()          # Custom plotting
    julia> run_example_4()          # Sensitivity analysis
    julia> run_example_5()          # Comprehensive plotting gallery
    julia> run_example_6()          # Benchmark economy analysis (NEW!)
==========================================================================================#

#==========================================================================================
#? Module Loading and Setup
==========================================================================================#

#* Run this first in REPL to load all necessary modules and dependencies.

begin #setup_environment
    println("="^80)
    println("SETTING UP RANDOM SEARCH LABOR MARKET MODEL")
    println("="^80)
    
    # Import all necessary modules
    include("model_functions.jl")
    include("types.jl")
    include("ModelSolver.jl")
    include("ModelPlotting.jl") 
    include("ModelRunner.jl")

    # Make modules available globally for REPL
    global Types, ModelFunctions, ModelSolver, ModelPlotting, ModelRunner
    using .Types, .ModelFunctions, .ModelSolver, .ModelPlotting, .ModelRunner
    using Statistics
    using Term.Progress
    using DataFrames
    using CairoMakie
    using LaTeXStrings
    using Infiltrator
    # using Debugger

        
    
    println("✅ Environment setup complete!")
    println("📝 Available functions:")
    println("   - run_example_1(): Quick test run")
    println("   - run_example_2(): Manual step-by-step workflow") 
    println("   - run_example_3(): Custom plotting workflow")
    println("   - run_example_4(): Parameter sensitivity analysis")
    println("   - load_default_config(): Load default configuration")
    println("   - quick_solve(): Quick model solve with default params")
    return nothing
end

#==========================================================================================
#? Quick Access Functions
==========================================================================================#

"""
    load_default_config()

Load the default configuration and return primitives. Useful for interactive exploration.
"""
function load_default_config()
    config_path = "/Users/mitchv34/Work/searching-flexibility/src/structural_model/model_parameters.yaml"
    prim, _ = Types.initializeModel(config_path)
    println("✅ Default configuration loaded")
    return prim
end

"""
    quick_solve(; verbose=true)

Quick model solve with default parameters. Returns (prim, results) for interactive use.
"""
function quick_solve(; verbose=true)
    prim = load_default_config()
    ModelRunner.calibrate_model!(prim)#, pin_location_h=1.0, pin_location_ψ=0.5)
    results = Results(prim)
    
    if verbose
        println("🔄 Solving model...")
    end
    
    ModelSolver.solve_model(
        prim, 
        results;
        verbose=verbose,   
        λ_S=0.01,    # Slower damping for stability
        λ_u=0.01,    # Slower damping for stability
        )
    
    if verbose
        println("✅ Model solved!")
        ModelRunner.print_model_results(prim, results)
    end
    
    return prim, results
end

#==========================================================================================
#? Example 1: Quick Test Run (REPL Function)
==========================================================================================#

"""
    run_example_1(; plot_results=true, save_plots=false)

Example 1: Quick test run using the high-level runner.
Returns (prim, results) for further interactive exploration.
"""
function run_example_1(; plot_results=true, save_plots=false)
    println("\n🚀 EXAMPLE 1: Quick Test Run")
    println("-"^50)
    
    prim, results = ModelRunner.run_model_test(; plot_results=plot_results, save_plots=save_plots)
    
    println("✅ Example 1 completed!")
    println("💡 Tip: The returned (prim, results) are available for further analysis")
    
    return prim, results
end

#==========================================================================================
#? Example 2: Manual Step-by-Step Workflow (REPL Function)
==========================================================================================#

"""
    run_example_2()

Example 2: Manual step-by-step workflow showing individual module usage.
Returns (prim, results) for interactive exploration.
"""
function run_example_2()
    println("\n🔧 EXAMPLE 2: Manual Step-by-Step Workflow")
    println("-"^50)
    println("  Setting up model manually...")
    
    # Step 1: Load configuration and initialize primitives
    config_path = abspath(joinpath(@__DIR__, "parameters", "initial", "model_parameters.yaml"))
    prim, _ = Types.initializeModel(config_path)
    println("  ✓ Step 1: Configuration loaded")
    
    # Step 2: Perform calibration
    ModelRunner.calibrate_model!(prim)
    println("  ✓ Step 2: Model calibrated")
    
    # Step 3: Create results structure
    results = Results(prim)
    println("  ✓ Step 3: Results structure created")
    
    # Step 4: Solve the model with custom parameters
    println("  🔄 Step 4: Solving model with custom tolerance...")
    ModelSolver.solve_model(
                            prim,               # Model primitives
                            results;            # Results object to store output
                            tol=1e-8,           # Tighter tolerance
                            max_iter=1000,      # More iterations
                            verbose=false,      # Silent solve
                            λ_S=0.05,           # Slower damping
                            λ_u=0.05)           # Slower damping
    println("  ✓ Step 4: Model solved")
    
    # Step 5: Analyze results
    println("  📊 Step 5: Analyzing results...")
    ModelRunner.print_model_results(prim, results)
    
    println("✅ Example 2 completed!")
    return prim, results
end

#==========================================================================================
#? Example 3: Custom Plotting Workflow (REPL Function)  
==========================================================================================#

"""
    run_example_3(prim=nothing, results=nothing)

Example 3: Custom plotting workflow. If prim and results not provided, 
will solve a quick model first.
"""
function run_example_3(prim=nothing, results=nothing)
    println("\n📊 EXAMPLE 3: Custom Plotting Workflow")
    println("-"^50)
    
    # Use provided results or solve quickly
    if prim === nothing || results === nothing
        println("  🔄 No results provided, solving model first...")
        prim, results = quick_solve(verbose=false)
    end
    
    println("  Creating custom plots...")
    created_plots = []
    
    try
        # Create core analysis plots
        fig1 = ModelPlotting.plot_employment_distribution(results, prim)
        println("  ✓ Employment distribution plot created")
        push!(created_plots, ("employment_distribution", fig1))
        
        fig2 = ModelPlotting.plot_surplus_function(results, prim)
        println("  ✓ Surplus function plot created")
        push!(created_plots, ("surplus_function", fig2))
        
        fig3 = ModelPlotting.plot_alpha_policy(results, prim)
        println("  ✓ Alpha policy plot created")
        push!(created_plots, ("alpha_policy", fig3))
        
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
        
        # Create the alpha derivation plot
        try
            fig4 = ModelPlotting.plot_alpha_derivation_and_policy(
                prim, results,
                h_idx_fixed=h_idx_for_analysis,
                ψ_indices_to_vary=ψ_indices_for_analysis
            )
            println("  ✓ Alpha derivation plot created")
            push!(created_plots, ("alpha_derivation", fig4))
        catch e
            println("  ⚠️  Alpha derivation plot skipped: $e")
        end
        
        println("✅ Example 3 completed!")
        return created_plots
        
    catch e
        println("  ⚠️  Some custom plots failed: $e")
        return created_plots
    end
end

#==========================================================================================
#? Example 4: Parameter Sensitivity Analysis (REPL Function)
==========================================================================================#

"""
    run_example_4(; bargaining_powers=[0.3, 0.5, 0.7])

Example 4: Parameter sensitivity analysis with configurable bargaining power values.
Returns results for each parameter value tested.
"""
function run_example_4(; bargaining_powers=[0.3, 0.5, 0.7])
    println("\n📈 EXAMPLE 4: Parameter Sensitivity Analysis")
    println("-"^50)
    println("  Running parameter sensitivity analysis...")
    
    # Load base configuration
    config_path = abspath(joinpath(@__DIR__, "parameters", "initial", "model_parameters.yaml"))
    
    sensitivity_results = []
    
    try
        # Test with different bargaining power parameters
        for (i, ξ) in enumerate(bargaining_powers)
            println("    🔄 Testing with bargaining power ξ = $ξ")
            
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
            
            # Store results
            push!(sensitivity_results, (ξ=ξ, unemployment_rate=unemployment_rate, avg_alpha=avg_alpha, prim=prim, results=results))
        end
        
        println("✅ Example 4 completed!")
        return sensitivity_results
        
    catch e
        println("  ⚠️  Sensitivity analysis failed: $e")
        return sensitivity_results
    end
end

#==========================================================================================
#? Example 5: Comprehensive Plotting Gallery (NEW)
==========================================================================================#

"""
    run_example_5(prim=nothing, results=nothing)

Example 5: Comprehensive plotting gallery showcasing all available visualization functions.
Creates all available plots from ModelPlotting.jl for thorough model analysis.
"""
function run_example_5(prim=nothing, results=nothing)
    println("\n🎨 EXAMPLE 5: Comprehensive Plotting Gallery")
    println("-"^50)
    
    # Use provided results or solve quickly
    if prim === nothing || results === nothing
        println("  🔄 No results provided, solving model first...")
        prim, results = quick_solve(verbose=false)
    end
    
    println("  Creating comprehensive plot gallery...")
    all_plots = []
    
    # Core distribution plots
    try
        fig1 = ModelPlotting.plot_employment_distribution(results, prim)
        println("  ✓ Employment distribution heatmap")
        push!(all_plots, ("employment_distribution", fig1))
        
        fig2 = ModelPlotting.plot_employment_distribution_with_marginals(results, prim)
        println("  ✓ Employment distribution with marginals")
        push!(all_plots, ("employment_marginals", fig2))
    catch e
        println("  ⚠️  Employment distribution plots failed: $e")
    end
    
    # Value function plots
    try
        fig3 = ModelPlotting.plot_surplus_function(results, prim)
        println("  ✓ Match surplus function")
        push!(all_plots, ("surplus_function", fig3))
    catch e
        println("  ⚠️  Surplus function plot failed: $e")
    end
    
    # Policy function plots
    try
        fig4 = ModelPlotting.plot_alpha_policy(results, prim)
        println("  ✓ Remote work share policy")
        push!(all_plots, ("alpha_policy", fig4))
        
        fig5 = ModelPlotting.plot_wage_policy(results, prim)
        println("  ✓ Wage policy function")
        push!(all_plots, ("wage_policy", fig5))
        
        fig6 = ModelPlotting.plot_alpha_policy_by_firm_type(results, prim)
        println("  ✓ Alpha policy by firm type")
        push!(all_plots, ("alpha_by_firm_type", fig6))
    catch e
        println("  ⚠️  Policy function plots failed: $e")
    end
    
    # Trade-off and outcome analysis
    try
        fig7 = ModelPlotting.plot_wage_amenity_tradeoff(results, prim)
        println("  ✓ Wage-amenity trade-off")
        push!(all_plots, ("wage_amenity_tradeoff", fig7))
        
        fig8 = ModelPlotting.plot_outcomes_by_skill(results, prim)
        println("  ✓ Labor market outcomes by skill")
        push!(all_plots, ("outcomes_by_skill", fig8))
    catch e
        println("  ⚠️  Trade-off analysis plots failed: $e")
    end
    
    # Regime analysis plots
    try
        fig9 = ModelPlotting.plot_work_arrangement_regimes(results, prim)
        println("  ✓ Work arrangement regimes")
        push!(all_plots, ("work_regimes", fig9))
    catch e
        println("  ⚠️  Work arrangement regimes plot failed: $e")
    end
    
    # Specialized theoretical plots
    try
        n_h = prim.h_grid.n
        n_ψ = prim.ψ_grid.n
        
        h_idx_for_analysis = round(Int, n_h * 0.6)  # Medium-high skill
        ψ_indices_for_analysis = [
            round(Int, n_ψ * 0.25), 
            round(Int, n_ψ * 0.5),
            round(Int, n_ψ * 0.75)
        ]
        
        fig10 = ModelPlotting.plot_alpha_derivation_and_policy(
            prim, results,
            h_idx_fixed=h_idx_for_analysis,
            ψ_indices_to_vary=ψ_indices_for_analysis
        )
        println("  ✓ Alpha derivation and theory")
        push!(all_plots, ("alpha_derivation", fig10))
    catch e
        println("  ⚠️  Alpha derivation plot failed: $e")
    end
    
    println("\n✅ Example 5 completed!")
    println("📈 Created $(length(all_plots)) plots total")
    println("💡 Tip: Access individual plots via returned array: plots = run_example_5()")
    println("🖥️  All plots are displayed in VSCode - check the plot pane!")
    
    return all_plots
end

#==========================================================================================
#? Example 6: Benchmark Economy Analysis (NEW)
==========================================================================================#

"""
    run_example_6(prim=nothing, results=nothing; description="Baseline Economy")

Example 6: Comprehensive benchmark economy analysis including stylized facts validation.
Creates a BenchmarkEconomy structure and runs all benchmark analyses.
"""
function run_example_6(prim=nothing, results=nothing; description="Baseline Economy")
    println("\n🏛️  EXAMPLE 6: Benchmark Economy Analysis")
    println("-"^50)
    
    # Use provided results or solve quickly
    if prim === nothing || results === nothing
        println("  🔄 No results provided, solving model first...")
        prim, results = quick_solve(verbose=false)
    end
    
    println("  📊 Creating benchmark economy structure...")
    
    # Create the benchmark economy
    benchmark = ModelPlotting.BenchmarkEconomy(results, prim, description=description)
    
    println("  ✓ Benchmark economy created with timestamp: $(benchmark.timestamp)")
    println("  📈 Computing key outcomes...")
    
    # Display key results
    show_benchmark_summary(benchmark)
    
    println("  🎨 Creating benchmark visualization plots...")
    created_plots = []
    
    try
        # Create benchmark summary plot
        fig1 = ModelPlotting.plot_benchmark_summary(benchmark)
        println("  ✓ Benchmark summary dashboard created")
        push!(created_plots, ("benchmark_summary", fig1))
        
        # Create stylized facts analysis plots
        fig2 = ModelPlotting.plot_remote_wage_correlations(benchmark)
        println("  ✓ Remote wage correlation analysis created")
        push!(created_plots, ("remote_wage_correlations", fig2))
        
        fig3 = ModelPlotting.plot_within_occupation_analysis(benchmark)
        println("  ✓ Within-occupation wage premium analysis created")
        push!(created_plots, ("within_occupation_analysis", fig3))
        
        println("✅ Example 6 completed!")
        println("📈 Created $(length(created_plots)) benchmark analysis plots")
        
        return benchmark, created_plots
        
    catch e
        println("  ⚠️  Some benchmark plots failed: $e")
        return benchmark, created_plots
    end
end

#==========================================================================================
#? Benchmark Economy Helper Functions
==========================================================================================#

"""
    show_benchmark_summary(benchmark::ModelPlotting.BenchmarkEconomy)

Display a comprehensive summary of benchmark economy results.
"""
function show_benchmark_summary(benchmark::ModelPlotting.BenchmarkEconomy)
    println("\n" * "="^70)
    println("BENCHMARK ECONOMY SUMMARY")
    println("="^70)
    println("Description: $(benchmark.description)")
    println("Timestamp: $(benchmark.timestamp)")
    println()
    
    # Aggregate outcomes
    println("📊 AGGREGATE OUTCOMES:")
    println("  • Unemployment Rate: $(round(benchmark.aggregate_outcomes.unemployment_rate*100, digits=2))%")
    println("  • Market Tightness (θ): $(round(benchmark.aggregate_outcomes.market_tightness, digits=4))")
    println("  • Total Employment: $(round(benchmark.aggregate_outcomes.total_employment, digits=2))")
    println("  • Average Wage: $(round(benchmark.aggregate_outcomes.average_wage, digits=2))")
    println()
    
    # Work arrangements
    println("🏠 WORK ARRANGEMENTS:")
    println("  • In-Person Share: $(round(benchmark.work_arrangements.share_in_person*100, digits=1))%")
    println("  • Hybrid Share: $(round(benchmark.work_arrangements.share_hybrid*100, digits=1))%")
    println("  • Full Remote Share: $(round(benchmark.work_arrangements.share_full_remote*100, digits=1))%")
    println("  • Average Remote Intensity: $(round(benchmark.work_arrangements.average_alpha, digits=3))")
    println()
    
    # Sorting patterns
    println("🔄 SORTING PATTERNS:")
    println("  • Spearman Correlation (h,ψ): $(round(benchmark.sorting_measures.spearman_correlation, digits=3))")
    println("  • Mutual Information: $(round(benchmark.sorting_measures.mutual_information, digits=4))")
    println()
    
    # Wage inequality
    println("📈 WAGE INEQUALITY:")
    println("  • Gini Coefficient: $(round(benchmark.wage_inequality.gini_coefficient, digits=3))")
    println("  • 90/10 Percentile Ratio: $(round(benchmark.wage_inequality.percentile_90_10, digits=2))")
    println("  • 90/50 Percentile Ratio: $(round(benchmark.wage_inequality.percentile_90_50, digits=2))")
    println("  • Skill Premium: $(round(benchmark.wage_inequality.skill_premium, digits=2))")
    println("  • Within-Group 90/10: $(round(benchmark.wage_inequality.within_group_90_10, digits=2))")
    
    println("="^70)
end

"""
    create_benchmark(; description="Baseline Economy", verbose=true)

Quick function to create a benchmark economy from scratch.
"""
function create_benchmark(; description="Baseline Economy", verbose=true)
    println("🏛️  Creating benchmark economy: $description")
    prim, results = quick_solve(verbose=verbose)
    benchmark = ModelPlotting.BenchmarkEconomy(results, prim, description=description)
    
    if verbose
        show_benchmark_summary(benchmark)
    end
    
    return benchmark
end

"""
    compare_benchmarks(benchmark1::ModelPlotting.BenchmarkEconomy, 
                        benchmark2::ModelPlotting.BenchmarkEconomy)

Compare two benchmark economies and display key differences.
"""
function compare_benchmarks(benchmark1::ModelPlotting.BenchmarkEconomy, 
                            benchmark2::ModelPlotting.BenchmarkEconomy)
    println("\n" * "="^70)
    println("BENCHMARK COMPARISON")
    println("="^70)
    println("Benchmark 1: $(benchmark1.description)")
    println("Benchmark 2: $(benchmark2.description)")
    println()
    
    # Compare aggregate outcomes
    println("📊 AGGREGATE OUTCOMES COMPARISON:")
    
    unemp_diff = (benchmark2.aggregate_outcomes.unemployment_rate - benchmark1.aggregate_outcomes.unemployment_rate) * 100
    theta_diff = benchmark2.aggregate_outcomes.market_tightness - benchmark1.aggregate_outcomes.market_tightness
    wage_diff = benchmark2.aggregate_outcomes.average_wage - benchmark1.aggregate_outcomes.average_wage
    
    println("  • Unemployment Rate: $(print_diff(unemp_diff, " pp"))")
    println("  • Market Tightness: $(print_diff(theta_diff))")
    println("  • Average Wage: $(print_diff(wage_diff))")
    println()
    
    # Compare work arrangements
    println("🏠 WORK ARRANGEMENT CHANGES:")
    remote_diff = (benchmark2.work_arrangements.share_full_remote - benchmark1.work_arrangements.share_full_remote) * 100
    alpha_diff = benchmark2.work_arrangements.average_alpha - benchmark1.work_arrangements.average_alpha
    
    println("  • Full Remote Share: $(print_diff(remote_diff, " pp"))")
    println("  • Average Remote Intensity: $(print_diff(alpha_diff))")
    println()
    
    # Compare inequality
    println("📈 INEQUALITY CHANGES:")
    gini_diff = benchmark2.wage_inequality.gini_coefficient - benchmark1.wage_inequality.gini_coefficient
    skill_prem_diff = benchmark2.wage_inequality.skill_premium - benchmark1.wage_inequality.skill_premium
    
    println("  • Gini Coefficient: $(print_diff(gini_diff))")
    println("  • Skill Premium: $(print_diff(skill_prem_diff))")
    
    println("="^70)
end

"""
    print_diff(diff_val, suffix="")

Helper function to format difference displays with appropriate signs and colors.
"""
function print_diff(diff_val, suffix="")
    if diff_val > 0
        return "+$(round(diff_val, digits=3))$suffix ↗"
    elseif diff_val < 0
        return "$(round(diff_val, digits=3))$suffix ↘"
    else
        return "$(round(diff_val, digits=3))$suffix →"
    end
end

function explore_parameter(
                            parameter_path::Vector{<:Union{Symbol, String}},
                            values::Vector;
                            initial_index::Int = 1,
                            base_config_path=nothing,
                            verbose::Bool=true
                        )
    
    param_name = join(parameter_path, " → ")
    
    if base_config_path === nothing
        base_config_path = abspath(joinpath(@__DIR__, "parameters", "initial", "model_parameters.yaml"))
    end
    
    # Get calibrated values using initial parameter value
    prim_calib, _ = Types.initializeModel(base_config_path)
    modify_parameter!(prim_calib, parameter_path, values[initial_index])
    ModelRunner.calibrate_model!(prim_calib, verbose=false)
    calibrated_psi0 = prim_calib.production_function.remote_efficiency.ψ₀
    calibrated_phi = prim_calib.production_function.remote_efficiency.ϕ
    
    if verbose
        println("🔄 Exploring parameter: $param_name")
        println("   Values: $(length(values)) points from $(minimum(values)) to $(maximum(values))")
        println("   Calibrated values: ψ₀=$(round(calibrated_psi0, digits=4)), ϕ=$(round(calibrated_phi, digits=4))")
    end
    
    # Initialize results storage
    results_data = []
    
    pbar = ProgressBar()
    job = addjob!(pbar; N=length(values), description="Exploring parameter: $param_name")
    start!(pbar)
    
    for (i, val) in enumerate(values)
        try
            # Load fresh model for each parameter value
            prim, _ = Types.initializeModel(base_config_path)
            modify_parameter!(prim, parameter_path, val)
            modify_parameter!(prim, [:production_function, :remote_efficiency, :ψ₀], calibrated_psi0)
            modify_parameter!(prim, [:production_function, :remote_efficiency, :ϕ], calibrated_phi)
            
            # Solve model
            results = Results(prim)
            ModelSolver.solve_model(prim, results; verbose=false, λ_S=0.01, λ_u=0.01)
            
            # Extract key statistics
            unemployment_rate = sum(results.u) / sum(prim.h_grid.pdf)
            total_employment = sum(results.n)
            average_alpha = sum(results.α_policy .* results.n) / sum(results.n)
            average_wage = sum(results.w_policy .* results.n) / sum(results.n)
            
            # Calculate work arrangement shares
            n_total = sum(results.n)
            share_in_person = sum(results.n[results.α_policy .<= 0.1]) / n_total  # α ≤ 0.1
            share_hybrid = sum(results.n[(results.α_policy .> 0.1) .& (results.α_policy .< 0.9)]) / n_total  # 0.1 < α < 0.9
            share_full_remote = sum(results.n[results.α_policy .>= 0.9]) / n_total  # α ≥ 0.9
            
            # Calculate potential workers (unemployed + employed) in each regime
            # This requires looking at the match surplus to see where workers would work if employed
            potential_in_person = 0.0
            potential_hybrid = 0.0  
            potential_full_remote = 0.0
            
            for i_h in 1:prim.h_grid.n, i_ψ in 1:prim.ψ_grid.n
                worker_mass = prim.h_grid.pdf[i_h] * prim.ψ_grid.pdf[i_ψ]
                alpha_opt = results.α_policy[i_h, i_ψ]
                
                if alpha_opt <= 0.1
                    potential_in_person += worker_mass
                elseif alpha_opt >= 0.9
                    potential_full_remote += worker_mass
                else
                    potential_hybrid += worker_mass
                end
            end
            
            # Store results
            push!(results_data, (
                parameter_value = val,
                unemployment_rate = unemployment_rate,
                total_employment = total_employment,
                average_alpha = average_alpha,
                average_wage = average_wage,
                share_in_person = share_in_person,
                share_hybrid = share_hybrid,
                share_full_remote = share_full_remote,
                potential_in_person = potential_in_person,
                potential_hybrid = potential_hybrid,
                potential_full_remote = potential_full_remote,
                market_tightness = results.θ
            ))
            
        catch e
            if verbose
                println("⚠️  Failed at parameter value $(val): $e")
            end
            # Store NaN values for failed runs
            push!(results_data, (
                parameter_value = val,
                unemployment_rate = NaN,
                total_employment = NaN,
                average_alpha = NaN,
                average_wage = NaN,
                share_in_person = NaN,
                share_hybrid = NaN,
                share_full_remote = NaN,
                potential_in_person = NaN,
                potential_hybrid = NaN,
                potential_full_remote = NaN,
                market_tightness = NaN
            ))
        end
        
        # Update progress
        update!(job)
        render(pbar)
    end
    
    # Convert to DataFrame
    df = DataFrame(results_data)
    
    if verbose
        successful_runs = sum(.!isnan.(df.unemployment_rate))
        println("✅ Parameter exploration completed!")
        println("   Successful runs: $(successful_runs)/$(length(values))")
    end
    stop!(pbar)
    return df
end
function plot_parameter_exploration_results(df::DataFrame; param_name="Parameter")
    
    # Create 2x2 subplot layout using ModelPlotting helper
    fig = ModelPlotting.create_figure(type="ultra")
    
    # Filter out NaN values for plotting
    valid_idx = .!isnan.(df.unemployment_rate)
    df_clean = df[valid_idx, :]
    
    if nrow(df_clean) == 0
        println("❌ No valid data points to plot!")
        return nothing
    end
    
    # Subplot 1: Unemployment Rate
    ax1 = ModelPlotting.create_axis(fig[1, 1], 
                                   "Unemployment Rate",
                                   param_name, 
                                   "Unemployment Rate (%)")
    lines!(ax1, df_clean.parameter_value, df_clean.unemployment_rate .* 100, 
           color=ModelPlotting.COLORS[1], linewidth=3)
    
    # Subplot 2: Average Remote Share
    ax2 = ModelPlotting.create_axis(fig[1, 2], 
                                   "Average Remote Share (α)",
                                   param_name, 
                                   "Average Remote Share")
    lines!(ax2, df_clean.parameter_value, df_clean.average_alpha, 
           color=ModelPlotting.COLORS[2], linewidth=3)
    
    # Subplot 3: Employment by Work Arrangement
    ax3 = ModelPlotting.create_axis(fig[2, 1], 
                                   "Employment by Work Arrangement",
                                   param_name, 
                                   "Share of Employed Workers (%)")
    lines!(ax3, df_clean.parameter_value, df_clean.share_in_person .* 100, 
           label="In-Person", color=ModelPlotting.COLORS[1], linewidth=3)
    lines!(ax3, df_clean.parameter_value, df_clean.share_hybrid .* 100, 
           label="Hybrid", color=ModelPlotting.COLORS[2], linewidth=3)
    lines!(ax3, df_clean.parameter_value, df_clean.share_full_remote .* 100, 
           label="Full Remote", color=ModelPlotting.COLORS[3], linewidth=3)
    
    # Create a horizontal legend outside the axis at the bottom
    Legend(fig[3, 1], ax3, orientation=:horizontal, tellheight=true)
    
    # Subplot 4: Potential Workers by Work Arrangement
    ax4 = ModelPlotting.create_axis(fig[2, 2], 
                                   "Potential Workers by Work Arrangement",
                                   param_name, 
                                   "Share of All Workers (%)")
    lines!(ax4, df_clean.parameter_value, df_clean.potential_in_person .* 100, 
           label="In-Person", color=ModelPlotting.COLORS[1], linewidth=3)
    lines!(ax4, df_clean.parameter_value, df_clean.potential_hybrid .* 100, 
           label="Hybrid", color=ModelPlotting.COLORS[2], linewidth=3, )
    lines!(ax4, df_clean.parameter_value, df_clean.potential_full_remote .* 100, 
           label="Full Remote", color=ModelPlotting.COLORS[3], linewidth=3)
    
    # Add overall title using consistent font styling
    Label(fig[0, :], latexstring("Parameter Exploration", param_name), 
          fontsize=ModelPlotting.FONT_SIZE*1.2, font=:bold)
    
    return fig
end

#==========================================================================================
#? Updated Quick Access Functions
==========================================================================================#

"""
    list_available_plots()

Updated to include benchmark economy plots.
"""
function list_available_plots()
    println("\n📊 AVAILABLE PLOT TYPES FOR quick_plot():")
    println("-"^50)
    plots_info = [
        (:employment_dist, "Employment distribution heatmap"),
        (:employment_marginals, "Employment distribution with marginal densities"),
        (:surplus, "Match surplus function"),
        (:alpha_policy, "Remote work share policy function"),
        (:wage_policy, "Equilibrium wage policy"),
        (:wage_amenity, "Wage-amenity trade-off curves"),
        (:outcomes_by_skill, "Labor market outcomes by worker skill"),
        (:work_regimes, "Work arrangement regime boundaries"),
        (:alpha_by_firm, "Remote share policy by firm efficiency"),
        (:alpha_derivation, "Theoretical derivation of optimal alpha"),
        # New benchmark economy plots
        (:benchmark_summary, "Comprehensive benchmark economy dashboard"),
        (:remote_wage_correlations, "Stylized fact I: Remote work wage correlations"),
        (:within_occupation_analysis, "Stylized fact II: Within-occupation wage premiums")
    ]
    
    for (symbol, description) in plots_info
        println("  • :$(symbol) - $(description)")
    end
    
    println("\n💡 Usage: quick_plot(:employment_dist)")
    println("💡 With benchmark: quick_plot(:benchmark_summary, benchmark=my_benchmark)")
    return nothing
end

"""
    quick_plot(plot_type::Symbol; prim=nothing, results=nothing, benchmark=nothing, kwargs...)

Updated quick plotting function that supports benchmark economy plots.
"""
function quick_plot(plot_type::Symbol; prim=nothing, results=nothing, benchmark=nothing, kwargs...)
    # Handle benchmark-specific plots
    if plot_type in [:benchmark_summary, :remote_wage_correlations, :within_occupation_analysis]
        if benchmark === nothing
            if prim === nothing || results === nothing
                println("🔄 Solving model for benchmark plotting...")
                prim, results = quick_solve(verbose=false)
            end
            println("🏛️  Creating benchmark economy...")
            benchmark = ModelPlotting.BenchmarkEconomy(results, prim; description="Baseline Economy")
        end
        
        println("📊 Creating $(plot_type) plot...")
        
        try
            fig = if plot_type == :benchmark_summary
                ModelPlotting.plot_benchmark_summary(benchmark)
            elseif plot_type == :remote_wage_correlations
                ModelPlotting.plot_remote_wage_correlations(benchmark)
            elseif plot_type == :within_occupation_analysis
                ModelPlotting.plot_within_occupation_analysis(benchmark)
            else
                error("Unknown benchmark plot type: $plot_type")
            end
            
            println("✅ Benchmark plot created successfully!")
            return fig
            
        catch e
            println("❌ Benchmark plot creation failed: $e")
            return nothing
        end
    end
    
    # Handle standard plots (existing code)
    if prim === nothing || results === nothing
        println("🔄 Solving model for plotting...")
        prim, results = quick_solve(verbose=false)
    end
    
    println("📊 Creating $(plot_type) plot...")
    
    try
        fig = if plot_type == :employment_dist
            ModelPlotting.plot_employment_distribution(results, prim)
        elseif plot_type == :employment_marginals
            ModelPlotting.plot_employment_distribution_with_marginals(results, prim)
        elseif plot_type == :surplus
            ModelPlotting.plot_surplus_function(results, prim)
        elseif plot_type == :alpha_policy
            ModelPlotting.plot_alpha_policy(results, prim)
        elseif plot_type == :wage_policy
            ModelPlotting.plot_wage_policy(results, prim)
        elseif plot_type == :wage_amenity
            ModelPlotting.plot_wage_amenity_tradeoff(results, prim)
        elseif plot_type == :outcomes_by_skill
            ModelPlotting.plot_outcomes_by_skill(results, prim)
        elseif plot_type == :work_regimes
            ModelPlotting.plot_work_arrangement_regimes(results, prim)
        elseif plot_type == :work_regimes_viable
            ModelPlotting.plot_work_arrangement_regimes(results, prim, gray_nonviable=true) 
        elseif plot_type == :alpha_by_firm
            ModelPlotting.plot_alpha_policy_by_firm_type(results, prim)
        elseif plot_type == :alpha_derivation
            # Use reasonable defaults for the derivation plot
            n_h, n_ψ = prim.h_grid.n, prim.ψ_grid.n
            h_idx = get(kwargs, :h_idx, round(Int, n_h * 0.8))
            ψ_indices = get(kwargs, :ψ_indices, [round(Int, n_ψ * 0.3), round(Int, n_ψ * 0.6), round(Int, n_ψ * 0.9)])
            ModelPlotting.plot_alpha_derivation_and_policy(prim, results, h_idx_fixed=h_idx, ψ_indices_to_vary=ψ_indices)
        else
            error("Unknown plot type: $plot_type")
        end
        
        println("✅ Plot created successfully!")
        return fig
        
    catch e
        println("❌ Plot creation failed: $e")
        return nothing
    end
end

#==========================================================================================
#? Updated REPL Instructions
==========================================================================================#
# 📖 REPL USAGE INSTRUCTIONS:
# 1. First run: setup_environment block to load all necessary packages and modules.
# 2. Then try any of the examples:
#    • run_example_1()  - Quick test
#    • run_example_2()  - Manual workflow 
#    • run_example_3()  - Custom plotting
#    • run_example_4()  - Sensitivity analysis
#    • run_example_5()  - Comprehensive plotting gallery
#    • run_example_6()  - Benchmark economy analysis (NEW!)
# 3. For benchmark analysis:
#    • benchmark = create_benchmark()  - Create benchmark economy
#    • show_benchmark_summary(benchmark)  - Display summary
#    • compare_benchmarks(bench1, bench2)  - Compare two benchmarks
# 4. For parameter exploration with benchmarks:
#    • df, benchmarks = explore_parameter_with_benchmarks([:ξ], [0.3, 0.5, 0.7])
# 5. For quick plotting (including benchmark plots):
#    • list_available_plots()  - See all plot types
#    • quick_plot(:benchmark_summary)  - Create benchmark plots
#
# 💡 New benchmark features make it easy to establish baseline economies!
# 🏛️  All benchmark data is preserved for comparison and analysis!

prim, results = quick_solve();

work_regimes_plot = quick_plot(:work_regimes, prim = prim, results = results) # Work arrangement regime boundaries
# Save the plot
save("/Users/mitchv34/Work/WFH/figures/model_figures/work_regimes_plot.pdf", work_regimes_plot)
work_regimes_viable_plot = quick_plot(:work_regimes_viable, prim = prim, results = results) # Work arrangement regime boundaries
# save("/Users/mitchv34/Work/WFH/figures/model_figures/work_regimes_viable_plot.pdf", work_regimes_viable_plot)

h_idx_fixed = round(Int, prim.h_grid.n * 0.88)  # Medium-high skill

ψ_indices_to_vary = [
            round(Int, prim.ψ_grid.n * 0.25), 
            round(Int, prim.ψ_grid.n * 0.35),
            round(Int, prim.ψ_grid.n * 0.51)
        ]


alpha_derivation_plot_1 = plot_alpha_derivation_and_policy(prim, results; 
                                    h_idx_fixed = h_idx_fixed, 
                                    ψ_indices_to_vary = Int[],
                                    h_idx_secondary = nothing,
                                    plot_policy_curve = false)


alpha_derivation_plot_2 = plot_alpha_derivation_and_policy(prim, results; 
                                    h_idx_fixed = h_idx_fixed, 
                                    ψ_indices_to_vary = ψ_indices_to_vary[1:1],
                                    h_idx_secondary = nothing,
                                    plot_policy_curve = false)

alpha_derivation_plot_3 = plot_alpha_derivation_and_policy(prim, results; 
                                    h_idx_fixed = h_idx_fixed, 
                                    ψ_indices_to_vary = ψ_indices_to_vary[1:2],
                                    h_idx_secondary = nothing,
                                    plot_policy_curve = false)

alpha_derivation_plot_4 = plot_alpha_derivation_and_policy(prim, results; 
                                    h_idx_fixed = h_idx_fixed, 
                                    ψ_indices_to_vary = ψ_indices_to_vary,
                                    h_idx_secondary = nothing,
                                    plot_policy_curve = false)

alpha_derivation_plot_5 = plot_alpha_derivation_and_policy(prim, results; 
                                    h_idx_fixed = h_idx_fixed, 
                                    ψ_indices_to_vary = ψ_indices_to_vary,
                                    h_idx_secondary = nothing,
                                    plot_policy_curve = true)

alpha_derivation_plot_6 = plot_alpha_derivation_and_policy(prim, results; 
                                    h_idx_fixed = h_idx_fixed, 
                                    ψ_indices_to_vary = ψ_indices_to_vary,
                                    h_idx_secondary = round(Int, prim.h_grid.n * 0.94), # Increase skill level
                                    plot_policy_curve = true)

alpha_derivation_plot_7 = plot_alpha_derivation_and_policy(prim, results; 
                                    h_idx_fixed = h_idx_fixed, 
                                    ψ_indices_to_vary = ψ_indices_to_vary,
                                    h_idx_secondary = round(Int, prim.h_grid.n * 0.8), # Decrease skill level
                                    plot_policy_curve = true)


# Save the plots to files
save("/Users/mitchv34/Work/WFH/figures/model_figures/alpha_derivation_plot_1.pdf", alpha_derivation_plot_1)
save("/Users/mitchv34/Work/WFH/figures/model_figures/alpha_derivation_plot_2.pdf", alpha_derivation_plot_2)
save("/Users/mitchv34/Work/WFH/figures/model_figures/alpha_derivation_plot_3.pdf", alpha_derivation_plot_3)
save("/Users/mitchv34/Work/WFH/figures/model_figures/alpha_derivation_plot_4.pdf", alpha_derivation_plot_4)
save("/Users/mitchv34/Work/WFH/figures/model_figures/alpha_derivation_plot_5.pdf", alpha_derivation_plot_5)
save("/Users/mitchv34/Work/WFH/figures/model_figures/alpha_derivation_plot_6.pdf", alpha_derivation_plot_6)
save("/Users/mitchv34/Work/WFH/figures/model_figures/alpha_derivation_plot_7.pdf", alpha_derivation_plot_7)


# benchmark = create_benchmark() ;

quick_plot(:employment_dist, prim = prim, results = results) # Employment distribution heatmap
quick_plot(:employment_marginals, prim = prim, results = results) # Employment distribution with marginal densities
quick_plot(:surplus, prim = prim, results = results) # Match surplus function
quick_plot(:alpha_policy, prim = prim, results = results) # Remote work share policy function
quick_plot(:wage_policy, prim = prim, results = results) # Equilibrium wage policy
quick_plot(:wage_amenity, prim = prim, results = results) # Wage-amenity trade-off curves
quick_plot(:outcomes_by_skill, prim = prim, results = results) # Labor market outcomes by worker skill
quick_plot(:alpha_by_firm, prim = prim, results = results) # Remote share policy by firm efficiency
quick_plot(:benchmark_summary, prim = prim, results = results) # Comprehensive benchmark economy dashboard
quick_plot(:remote_wage_correlations, prim = prim, results = results) # Stylized fact I: Remote work wage correlations
quick_plot(:within_occupation_analysis, prim = prim, results = results) # Stylized fact II: Within-occupation wage premiums


# min_val_c = 0.185;
# max_val_c = 0.7;
# n = 200;
# initial_index_c = 20;
# parameter_space_c = min_val_c:((max_val_c-min_val_c)/n):max_val_c |> collect;

# exploration_results = explore_parameter(
#                                         [:utility_function, :c₀],
#                                         parameter_space_c, 
#                                         initial_index=initial_index_c,
#                                         verbose=false
#                                         );

# Create the parameter exploration plots

# param_plot = plot_parameter_exploration_results(
#                             exploration_results, 
#                             param_name=latexstring("\$c_{0}\$ (Utility Parameter)")
#                         )
    
# min_val_ν = 0.01;
# max_val_ν = 2.0;
# n = 200;
# initial_index_ν = 100;
# parameter_space_ν = min_val_ν:((max_val_ν-min_val_ν)/n):max_val_ν |> collect;

# exploration_results = explore_parameter(
#                                         [:production_function, :remote_efficiency, :ν],
#                                         parameter_space_ν, 
#                                         initial_index=initial_index_ν,
#                                         verbose=false
#                                         );

# Create the parameter exploration plots

# param_plot = plot_parameter_exploration_results(
#                             exploration_results, 
#                             param_name=latexstring(" \$\\nu\$ (Remote Efficiency Parameter)")
#                         )
    

# # Exploration of of the χ between 1.001 and 8.5
# min_val_χ = 1.001;
# max_val_χ = 8.5;
# n_χ = 200;
# initial_index_χ = 1;
# parameter_space_χ = min_val_χ:((max_val_χ-min_val_χ)/n_χ):max_val_χ |> collect;

# exploration_results_χ = explore_parameter(
#                                         [:utility_function, :χ],
#                                         parameter_space_χ, 
#                                         initial_index=initial_index_χ,
#                                         verbose=false
#                                         );

# # Create the parameter exploration plots for χ
# param_plot_χ = plot_parameter_exploration_results(
#                             exploration_results_χ, 
#                             param_name=latexstring("\$χ\$")
#                         )   

# Exploration of of the κ₀ between 0.001 and 1.5``
# min_val_κ₀ = 0.001;
# max_val_κ₀ = 1.5;
# n_κ₀ = 200;
# initial_index_κ₀ = 1;
# parameter_space_κ₀ = min_val_κ₀:((max_val_κ₀-min_val_κ₀)/n_κ₀):max_val_κ₀ |> collect;

# exploration_results_κ₀ = explore_parameter(
#                                         [:κ₀],
#                                         parameter_space_κ₀, 
#                                         initial_index=initial_index_κ₀,
#                                         verbose=false
#                                         );

# # Create the parameter exploration plots for κ₀
# param_plot_κ₀ = plot_parameter_exploration_results(
#                             exploration_results_κ₀, 
#                             param_name=latexstring("\$κ_{0}\$")
#                         )
