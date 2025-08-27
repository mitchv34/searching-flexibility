#!/usr/bin/env julia

"""
Demo Script for Searching Flexibility Experiments

This script demonstrates the key experimental capabilities of the repository.
Run this to see what types of experiments are available and how they work.
"""

println("🚀 Searching Flexibility - Experimental Demo")
println("=" ^ 60)

println("\n📖 About This Repository:")
println("This repository implements structural economic models for studying")
println("labor market flexibility, remote work adoption, and job search dynamics.")
println("It provides a comprehensive experimental framework for economic research.")

println("\n🔬 Available Experiment Types:")

experiments = [
    ("minimal_experiment", "Lightweight synthetic experiments", "< 1 min", "No heavy deps"),
    ("basic_model_test", "Full model validation", "2-5 min", "Full installation"),
    ("parameter_estimation", "Optimize model parameters", "10-60 min", "High compute"),
    ("cross_moment_check", "Validate model moments", "15-45 min", "Distributed"),
    ("profile_analysis", "Performance profiling", "5-10 min", "Memory analysis")
]

for (i, (name, desc, time, req)) in enumerate(experiments)
    println("  $i. $name")
    println("     Description: $desc")
    println("     Time: $time | Requirements: $req")
    println()
end

println("🎯 Quick Demo - Running Minimal Experiment:")
println("This will demonstrate the experimental infrastructure without heavy dependencies.")
println()

# Run the minimal experiment as a demo
print("Press Enter to run the demo, or Ctrl+C to exit: ")
try
    readline()
catch InterruptException
    println("\nDemo cancelled. You can run experiments manually using:")
    println("julia experiment_runner.jl [experiment_name]")
    exit(0)
end

println("\n" * ("=" ^ 60))
println("🧪 DEMO: Running Minimal Experiment")
println("=" ^ 60)

# Run minimal experiment
try
    run(`julia --project=. minimal_experiment.jl`)
    
    println("\n" * ("=" ^ 60))
    println("✅ Demo Completed Successfully!")
    println("=" ^ 60)
    
    # Show the results file
    results_file = "/tmp/minimal_experiment_results.txt"
    if isfile(results_file)
        println("\n📄 Generated Results File:")
        println(read(results_file, String))
    end
    
catch e
    println("❌ Demo failed: $e")
    println("This might be due to missing dependencies or system issues.")
end

println("\n🚀 Next Steps:")
println("1. Install full dependencies: julia --project=. -e \"using Pkg; Pkg.instantiate()\"")
println("2. Run basic model test: julia experiment_runner.jl basic_model_test")
println("3. Explore parameter estimation: julia experiment_runner.jl parameter_estimation --quick")
println("4. See all options: julia experiment_runner.jl help")
println()
println("📚 Documentation: See EXPERIMENTS.md for detailed usage instructions")
println("🐛 Issues: Check /tmp/ for log files if experiments fail")

println("\n🎉 Thank you for trying the Searching Flexibility experimental framework!")