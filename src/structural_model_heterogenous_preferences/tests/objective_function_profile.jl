n_workers = 9 # 9 workers for parallel, 0 for serial
include("../julia_config.jl")

const FIGURES_DIR = joinpath(ROOT, "figures", "structural_model_heterogenous_preferences", "tests", "objective_function_profile")

using Random, Statistics, Term, Printf, CairoMakie, ProgressMeter

# Point to the heterogeneous model configuration file

# Initialize the model
prim, res = initializeModel(MODEL_CONFIG);
# Solve the model - FIX: Use config-based solver like run_file.jl
@time solve_model(prim, res, config=MODEL_CONFIG);
# Compute baseline moments
baseline_moments = compute_model_moments(prim, res)

# Parameters to profile 
params_to_estimate = [:aₕ, :bₕ, :c₀, :μ, :χ, :A₁, :ν, :ψ₀, :ϕ, :κ₀]
initial_values = [getfield(prim, k) for k in params_to_estimate]

# Build problem payload following structural_model_new pattern
p = (
    prim_base = prim,
    res_base = res,
    target_moments = baseline_moments,
    param_names = params_to_estimate,
    weighting_matrix = nothing,
    matrix_moment_order = nothing,
    # warm-start cache and rolling solver-state used by objective_function
    last_res = Ref(res),
    solver_state = Ref((λ_S_init = 0.01, λ_u_init = 0.01)),
    solver_config = MODEL_CONFIG
);

# Settings for the grid around the true value
n_grid = 41
rel_width = 0.001

# Create grids
θ0 = collect(float.(initial_values))
grids = [range(param * (1 - rel_width), param * (1 + rel_width), n_grid) for param in θ0]

# Ship immutable inputs to workers once
@everywhere const POBJ = $p
@everywhere const THETA0 = $θ0

# Worker-side evaluation for one parameter's grid
@everywhere function eval_param_grid(i::Int, grid)
    fvals = Vector{Float64}(undef, length(grid))
    θ = copy(THETA0)
    for j in eachindex(grid)
        θ[i] = grid[j]
        fvals[j] =  objective_function(θ, POBJ)
    end
    return fvals
end

n_params = length(params_to_estimate)
fvals_shared = SharedArray{Float64}(n_params, n_grid)
fill!(fvals_shared, NaN)

# Setup
progress = Progress( n_params )
channel = RemoteChannel(()->Channel{Bool}(1))

# Async progress updater
@async begin
    while take!(channel)
        next!(progress)
    end
end

# pmap with progress updates
results = pmap(i -> begin
    out = eval_param_grid(i, grids[i])
    put!(channel, true)  # update progress for this finished job
    out
end, 1:n_params)

put!(channel, false)  # end updating

for i in 1:n_params
    fvals_shared[i, :] = results[i]
end


# Create figures directory if it doesn't exist
mkpath(FIGURES_DIR)

fig = Figure(size = (1000, 800))
for i in eachindex(params_to_estimate)
    row = ((i - 1) ÷ 3) + 1
    col = ((i - 1) % 3) + 1
    ax = Axis(fig[row, col], title = string(params_to_estimate[i]))
    grid = grids[i]
    fvals = collect(view(fvals_shared, i, :))
    # if i == 8
    #     grid = grid[12:end]
    #     fvals = fvals[12:end]
    # end
    # if i == 7
    #     grid = grid[5:end]
    #     fvals = fvals[5:end]
    # end
    # if i == 7
    #     grid = grid[5:end]
    #     fvals = fvals[5:end]
    # end
    # if i == 10
    #     grid = grid[3:end]
    #     fvals = fvals[3:end]
    # end
    lines!(ax, grid, fvals)
    vlines!(ax, [θ0[i]], color = :red, linestyle = :dash)
end

display(fig)

# Save the figure
out_png = joinpath(FIGURES_DIR, "objective_function_profiles.png")
save(out_png, fig)
println("Objective function profiles saved to: ", out_png)

prim, res = initializeModel(MODEL_CONFIG);
@show prim.κ₀ 
prim_new, res_new = update_primitives_results(prim, res, Dict(:κ₀ => 1.3));
@show prim_new.κ₀


@time solve_model(prim, res, config=MODEL_CONFIG)
@time solve_model(prim_new, res_new, config=MODEL_CONFIG)

moments = compute_model_moments(prim, res)
moments_new = compute_model_moments(prim_new, res_new)

for (k, v) in moments
    v_new = moments_new[k]
    @printf("%-20s : original = %10.6f | new = %10.6f | diff = %10.6f\n", String(k), v, v_new, v_new - v)
end


κ_grid = range(0.5, 20.0, length=21)
A_grid = range(0.5, 20.0, length=21)  # Adjust range as needed

println("κ₀\tA₁\thybrid_in_person_share\thybrid_remote_share")
for κ₀ in κ_grid, A₁ in A_grid
    prim, res = initializeModel(MODEL_CONFIG)
    prim, res = update_primitives_results(prim, res, Dict(:κ₀ => κ₀, :A₁ => A₁))
    solve_model(prim, res, config = MODEL_CONFIG)
    # Compute the model moments
    model_moments = compute_model_moments(prim, res)
    inperson_share  = model_moments[:inperson_share]
    hybrid_share  = model_moments[:hybrid_share]
    remote_share  = model_moments[:remote_share]
    market_tightness = model_moments[:market_tightness]
    @show κ₀, A₁
    println("\t In person: $(round(inperson_share, digits=4)) \t Hybrid: $(round(hybrid_share, digits=4))\t Remote: $(round(remote_share, digits=4))\t Market Tightness: $(round(market_tightness, digits=4))")
end
