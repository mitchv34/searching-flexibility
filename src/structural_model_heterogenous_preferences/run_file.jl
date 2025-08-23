using Pkg
Pkg.activate("/project/high_tech_ind/searching-flexibility")
Pkg.instantiate()

# using Distributed
# addprocs(9)

const ROOT = @__DIR__
# @everywhere begin
#     using Random, Statistics, SharedArrays, ForwardDiff
#     using Optimization, OptimizationOptimJL, Optim
#     include(joinpath($ROOT, "ModelSetup.jl"))
#     include(joinpath($ROOT, "ModelSolver.jl"))
#     include(joinpath($ROOT, "ModelEstimation.jl"))
# end
using Random, Statistics
using Term
using Printf
using CairoMakie
using PrettyTables

include(joinpath(ROOT, "ModelPlotting.jl"))
include(joinpath(ROOT, "ModelSetup.jl"))
include(joinpath(ROOT, "ModelSolver.jl"))
include(joinpath(ROOT, "ModelEstimation.jl"))


config = "src/structural_model_heterogenous_preferences/model_parameters.yaml"
prim, res = initializeModel(config);
@time solve_model(prim, res, verbose=true, λ_S_init = 0.01, λ_u_init = 0.01, tol = 1e-8, max_iter = 25_000)

# Distribution plot moved toplot_z_distribution(prim)

s_flow = calculate_logit_flow_surplus_with_curvature(prim)

# Use plotting helpers in ModelPlotting for consistency
fig_s1, fig_s2, fig_s3, fig_s4 =plot_s_flow_diagnostics(s_flow, prim)

# Display core diagnostics
fig_s1 |> display
fig_s2 |> display
fig_s3 |> display
fig_s4 |> display


# Solution diagnostics
# # --- Generate diagnostic plots (integration with ModelPlotting) ---
# # Employment distribution heatmap
fig_emp =plot_employment_distribution(res, prim)
# # Employment distribution with marginals
fig_emp_marg =plot_employment_distribution_with_marginals(res, prim)

fig_surplus = plot_surplus_function(res, prim)

# fig_alpha =plot_alpha_policy(res, prim)
fig_alpha = plot_avg_alpha(prim, res)

# fig_wage_pol =plot_wage_policy(res, prim)
fig_wage_pol = plot_avg_wage(prim, res)

# fig_wage_amenity =plot_wage_amenity_tradeoff(res, prim)

# fig_outcome_skill =plot_outcomes_by_skill(res, prim)

# fig_work_arrangement =plot_work_arrangement_regimes(res, prim)

# fig_work_arrangement_viable =plot_work_arrangement_regimes(res, prim, gray_nonviable=true)

# fig_alpha_by_firm =plot_alpha_policy_by_firm_type(res, prim)

save(joinpath(ROOT, "temp", "fig_z.png"), fig_z)
save(joinpath(ROOT, "temp", "fig_s1.png"), fig_s1)
save(joinpath(ROOT, "temp", "fig_s2.png"), fig_s2)
save(joinpath(ROOT, "temp", "fig_s3.png"), fig_s3)
save(joinpath(ROOT, "temp", "fig_s4.png"), fig_s4)
save(joinpath(ROOT, "temp", "fig_emp.png"), fig_emp)
save(joinpath(ROOT, "temp", "fig_emp_marg.png"), fig_emp_marg)
save(joinpath(ROOT, "temp", "fig_surplus.png"), fig_surplus)
save(joinpath(ROOT, "temp", "fig_alpha.png"), fig_alpha)
save(joinpath(ROOT, "temp", "fig_wage_pol.png"), fig_wage_pol)
save(joinpath(ROOT, "temp", "fig_outcome_skill.png"), fig_outcome_skill)
# save(joinpath(ROOT, "temp", "fig_work_arrangement.png"), fig_work_arrangement)
# save(joinpath(ROOT, "temp", "fig_work_arrangement_viable.png"), fig_work_arrangement_viable)
# save(joinpath(ROOT, "temp", "fig_alpha_by_firm.png"), fig_alpha_by_firm)
