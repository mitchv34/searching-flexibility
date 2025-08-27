#!/usr/bin/env julia

"""
Experiment Runner for Searching Flexibility Project

This script provides a unified interface to run various experiments 
available in the searching-flexibility repository.

Usage:
    julia experiment_runner.jl [experiment_type] [options]

Available experiments:
    - basic_model_test: Basic model initialization and solving test
    - parameter_estimation: Run parameter estimation experiments
    - cross_moment_check: Run cross-moment validation
    - profile_analysis: Run performance profiling
    - counterfactuals: Run counterfactual experiments
"""

using Pkg
Pkg.activate(@__DIR__)

using Printf

function parse_commandline()
    # Simple argument parsing without ArgParse
    args = ARGS
    
    # Default values
    experiment = length(args) > 0 ? args[1] : "basic_model_test"
    model_type = "heterogenous"
    quick = false
    verbose = false
    
    # Parse simple flags
    for arg in args
        if arg == "--quick" || arg == "-q"
            quick = true
        elseif arg == "--verbose" || arg == "-v"
            verbose = true
        elseif arg == "--model=new" || arg == "-m=new"
            model_type = "new"
        elseif arg == "--model=heterogenous" || arg == "-m=heterogenous"
            model_type = "heterogenous"
        end
    end
    
    return Dict(
        "experiment" => experiment,
        "model" => model_type,
        "quick" => quick,
        "verbose" => verbose
    )
end

function run_basic_model_test(model_type::String; verbose=false, quick=false)
    println("🧪 Running basic model test for $(model_type) model...")
    
    if model_type == "new"
        model_dir = "src/structural_model_new"
        config_file = joinpath(model_dir, "model_parameters.yaml")
    elseif model_type == "heterogenous"
        model_dir = "src/structural_model_heterogenous_preferences" 
        config_file = joinpath(model_dir, "model_parameters.yaml")
    else
        error("Unknown model type: $model_type")
    end
    
    # Test script to validate basic model functionality
    test_script = """
    using Pkg
    Pkg.activate(".")
    
    # Include model files
    include("$(model_dir)/ModelSetup.jl")
    include("$(model_dir)/ModelSolver.jl") 
    include("$(model_dir)/ModelEstimation.jl")
    
    println("✅ Successfully loaded model modules")
    
    # Initialize model
    prim, res = initializeModel("$(config_file)")
    println("✅ Model initialized successfully")
    println("   - Grid size (ψ): \$(prim.n_ψ)")
    println("   - Grid size (h): \$(prim.n_h)")
    
    # Basic solve
    max_iter = $(quick ? 100 : 1000)
    if $(verbose)
        solve_model(prim, res, verbose=true, max_iter=max_iter)
    else
        solve_model(prim, res, verbose=false, max_iter=max_iter)
    end
    println("✅ Model solved successfully")
    
    # Compute moments
    moments = compute_model_moments(prim, res)
    println("✅ Computed model moments:")
    for (k, v) in pairs(moments)
        println("   - \$k: \$(round(v, digits=4))")
    end
    """
    
    # Write and execute test
    test_file = "/tmp/basic_test.jl"
    open(test_file, "w") do f
        write(f, test_script)
    end
    
    try
        run(`julia --project=. $test_file`)
        println("✅ Basic model test completed successfully!")
        return true
    catch e
        println("❌ Basic model test failed: $e")
        return false
    end
end

function run_parameter_estimation(model_type::String; verbose=false, quick=false)
    println("🔍 Running parameter estimation for $(model_type) model...")
    
    if model_type == "new"
        script_path = "src/structural_model_new/run_file.jl"
    else
        script_path = "src/structural_model_heterogenous_preferences/run_file.jl"
    end
    
    if quick
        println("   (Quick mode: reduced workers and iterations)")
        # Modify script to use fewer workers
        script_content = read(script_path, String)
        script_content = replace(script_content, "addprocs(9)" => "addprocs(2)")
        script_content = replace(script_content, "max_iter=25_000" => "max_iter=1_000")
        
        quick_script = "/tmp/quick_estimation.jl"
        open(quick_script, "w") do f
            write(f, script_content)
        end
        script_path = quick_script
    end
    
    try
        if verbose
            run(`julia --project=. $script_path`)
        else
            run(pipeline(`julia --project=. $script_path`, stdout="/tmp/estimation_output.log"))
            println("✅ Parameter estimation completed! Check /tmp/estimation_output.log for details")
        end
        return true
    catch e
        println("❌ Parameter estimation failed: $e")
        return false
    end
end

function run_cross_moment_check(; verbose=false, quick=false)
    println("📊 Running cross-moment validation...")
    
    script_path = "src/structural_model_heterogenous_preferences/tests/cross_moment_check.jl"
    
    if quick
        script_content = read(script_path, String)
        script_content = replace(script_content, "addprocs(9)" => "addprocs(2)")
        script_content = replace(script_content, "N_GRID = 41" => "N_GRID = 11")
        
        quick_script = "/tmp/quick_cross_moment.jl"
        open(quick_script, "w") do f
            write(f, script_content)
        end
        script_path = quick_script
    end
    
    try
        if verbose
            run(`julia --project=. $script_path`)
        else
            run(pipeline(`julia --project=. $script_path`, stdout="/tmp/cross_moment_output.log"))
            println("✅ Cross-moment check completed! Check /tmp/cross_moment_output.log for details")
        end
        return true
    catch e
        println("❌ Cross-moment check failed: $e")
        return false
    end
end

function run_profile_analysis(model_type::String; verbose=false, quick=false)
    println("⚡ Running performance profiling for $(model_type) model...")
    
    if model_type == "new"
        script_path = "src/structural_model_new/profile_run.jl"
    else
        script_path = "src/structural_model_heterogenous_preferences/profile_run.jl"
    end
    
    try
        if verbose
            run(`julia --project=. $script_path`)
        else
            run(pipeline(`julia --project=. $script_path`, stdout="/tmp/profile_output.log"))
            println("✅ Profile analysis completed! Check /tmp/profile_output.log for details")
        end
        return true
    catch e
        println("❌ Profile analysis failed: $e")
        return false
    end
end

function list_available_experiments()
    println("🔬 Available Experiments:")
    println("  1. basic_model_test    - Test basic model initialization and solving")
    println("  2. parameter_estimation - Run parameter estimation")
    println("  3. cross_moment_check  - Run cross-moment validation")
    println("  4. profile_analysis    - Run performance profiling")
    println("  5. counterfactuals     - Run counterfactual experiments")
    println()
    println("Usage examples:")
    println("  julia experiment_runner.jl basic_model_test")
    println("  julia experiment_runner.jl parameter_estimation --model new --quick")
    println("  julia experiment_runner.jl cross_moment_check --verbose")
end

function main()
    args = parse_commandline()
    
    experiment = args["experiment"]
    model_type = args["model"]
    verbose = args["verbose"]
    quick = args["quick"]
    
    println("🚀 Searching Flexibility Experiment Runner")
    println("=" ^ 50)
    
    if experiment == "list" || experiment == "help"
        list_available_experiments()
        return
    end
    
    success = false
    
    if experiment == "basic_model_test"
        success = run_basic_model_test(model_type; verbose=verbose, quick=quick)
    elseif experiment == "parameter_estimation"
        success = run_parameter_estimation(model_type; verbose=verbose, quick=quick)
    elseif experiment == "cross_moment_check"
        success = run_cross_moment_check(verbose=verbose, quick=quick)
    elseif experiment == "profile_analysis" 
        success = run_profile_analysis(model_type; verbose=verbose, quick=quick)
    else
        println("❌ Unknown experiment: $experiment")
        println("Run with 'help' to see available experiments")
        return
    end
    
    if success
        println("\n🎉 Experiment completed successfully!")
    else
        println("\n💥 Experiment failed. Check error messages above.")
        exit(1)
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end