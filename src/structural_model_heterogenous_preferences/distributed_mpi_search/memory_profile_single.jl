# Memory profiling harness for a single objective evaluation / small GA sample
using Distributed
using Dates

println("[MemoryProfile] Start at $(Dates.now())")

# Force single process (no worker launch) for isolated measurement
ENV["FORCE_SINGLE_PROCESS"] = "1"
ENV["SKIP_GA"] = "1"  # ensure GA loop skipped if script branches

include("mpi_search.jl")  # will run until before GA if SKIP_GA honored

# At this point, model code should be loaded; define a representative parameter vector

if !@isdefined(evaluate_objective_function)
    println("[MemoryProfile] ERROR: evaluate_objective_function not defined after include.")
    exit(1)
end

# Construct a plausible param set near midpoints (adjust keys to match actual model's param dict expectations)
# If model expects a Dict or NamedTuple, adapt here.
params = Dict(
    "aₕ" => 5.5,
    "bₕ" => 5.5,
    "c₀" => 1.0,
    "μ"  => 0.1,
    "χ"  => 5.0,
    "ν"  => 1.5,
    "ψ₀" => 1.5,
    "ϕ"  => 1.5,
    "κ₀" => 150.5,
)

println("[MemoryProfile] Running warm-up evaluation...")
res1 = @time evaluate_objective_function(params)
println("[MemoryProfile] Warm-up objective: $(res1)")
GC.gc()

# Repeat evaluations to see steady-state allocation
N=5
allocs = Float64[]
for i in 1:N
    before = Base.gc_bytes()
    res = @time evaluate_objective_function(params)
    after = Base.gc_bytes()
    push!(allocs, max(after - before, 0))
end

mean_alloc = mean(allocs)
println("[MemoryProfile] Mean allocated bytes per evaluation (net gc bytes metric): $(mean_alloc)")

# Peak RSS (Linux) via reading /proc/self/status if available
function peak_rss_mb()
    try
        for line in eachline("/proc/self/status")
            if startswith(line, "VmHWM:")
                parts = split(strip(line))
                kb = parse(Float64, parts[end-1])  # value before unit
                return kb / 1024.0
            end
        end
    catch
    end
    return missing
end
println("[MemoryProfile] Peak resident set size (approx MB): ", peak_rss_mb())

# Rough per-worker memory: resident arrays + peak ephemeral. Use summarysize on model globals if available.
# (Optional hooks could be added here if model exposes state.)

println("[MemoryProfile] Done at $(Dates.now())")
