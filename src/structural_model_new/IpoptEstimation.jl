#==========================================================================================
Title: Ipopt Estimation Module
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2025-08-14
Description: This module provides functions to estimate model parameters using the Ipopt optimizer.
==========================================================================================#
module IpoptEstimation

# High-level packages we need
using JuMP, Ipopt, Printf

# The Ipopt estimator relies on helper functions/types defined elsewhere (e.g.
# `Primitives`, `Results`, `update_params_and_resolve!`, `compute_model_moments`).
# To avoid redefining those functions and replacing docstrings when this file is
# included multiple times, do NOT include the core model files here. Instead,
# the caller should `include("ModelSetup.jl")`, `include("ModelSolver.jl")`,
# and `include("ModelEstimation.jl")` once (for example in the top-level script)
# prior to `include("IpoptEstimation.jl")` or `using .IpoptEstimation`.

# Export the main entry point so other code can call it as `IpoptEstimation.estimate_with_ipopt`.
export estimate_with_ipopt

# Helper wrappers: when the user has included the Model*.jl files at the top-level (Main),
# their method implementations will live in `Main`. If we call the local functions directly
# while the objects passed in are `Main` types, dispatch will fail. These wrappers attempt
# to call `Main`'s functions first, and fall back to the module-local versions.
function call_update_params_and_resolve!(prim_arg, res_arg; kwargs...)
    if isdefined(Main, :update_params_and_resolve!)
        try
            return Main.update_params_and_resolve!(prim_arg, res_arg; kwargs...)
        catch
            # fallback to local implementation
        end
    end
    return update_params_and_resolve!(prim_arg, res_arg; kwargs...)
end

function call_compute_model_moments(prim_arg, res_arg)
    if isdefined(Main, :compute_model_moments)
        try
            return Main.compute_model_moments(prim_arg, res_arg)
        catch
            # fallback
        end
    end
    return compute_model_moments(prim_arg, res_arg)
end

"""
    estimate_with_ipopt(prim, res, target_moments::NamedTuple;
                        ipopt_opts=Dict{String,Any}(),
                        bounds=Dict{Symbol,Tuple{Float64,Float64}}(),
                        verbose::Bool=false)

Estimate parameters by minimizing the sum of squared differences between model and target moments,
using Ipopt with finite-difference derivatives.

Parameters estimated (9): A₁, ψ₀, ν, a_h, b_h, χ, c₀, ϕ, κ₀

Returns Dict(:params, :objective, :status)
"""
function estimate_with_ipopt(
    prim,                              # model primitives / baseline parameters (untyped to accept Main or module types)
    res,                               # pre-computed Results object to warm-start solves (untyped)
    target_moments::NamedTuple;        # observed/target moments we want to match
    ipopt_opts::Dict{String,Any}=Dict{String,Any}(),
    bounds::Dict{Symbol,Tuple{Float64,Float64}}=Dict{Symbol,Tuple{Float64,Float64}}(),
    verbose::Bool=false,               # toggle more Ipopt printing
)
    # ----- parameter list and starting values -----
    # The list of parameters we will estimate. Kept in a fixed order because JuMP variables
    # and the objective registration rely on a consistent ordering.
    # Include additional parameters ϕ and κ₀ per user request.
    params = [:A₁, :ψ₀, :ν, :a_h, :b_h, :χ, :c₀, :ϕ, :κ₀]

    # Parameters that must remain strictly positive; we'll optimize their logs (unconstrained)
    # and pass exp(var) into the black-box objective.
    positive_params = Set{Symbol}([:ψ₀, :ν, :a_h, :b_h, :χ, :c₀, :κ₀])

    # Extract starting values from the `prim` struct into a Dict keyed by symbol.
    # `getfield(prim, k)` reads field k from the `prim` object.
    # Build starting values defensively: if `prim` lacks a field (e.g., new params),
    # fall back to 0.0 as a safe starting guess.
    x0 = Dict{Symbol,Float64}()
    for k in params
        if hasfield(typeof(prim), k)
            x0[k] = Float64(getfield(prim, k))
        else
            x0[k] = 0.0
        end
    end

    # NOTE: per request we remove explicit bounds on the JuMP decision variables.
    # The `bounds` argument is accepted for compatibility but is ignored here.

    # ----- build the JuMP model using the Ipopt optimizer -----
    model = Model(Ipopt.Optimizer)

    # Control Ipopt's printing and numerical behavior via optimizer attributes.
    # `print_level`: 0 (quiet) up to higher integers for more verbose Ipopt logging.
    set_optimizer_attribute(model, "print_level", verbose ? 5 : 0)

    # `sb` (suppress banner) set to "yes" to avoid repeating large headers each run.
    set_optimizer_attribute(model, "sb", "yes")

    # iteration and tolerance controls
    set_optimizer_attribute(model, "max_iter", 500)
    set_optimizer_attribute(model, "tol", 1e-6)
    set_optimizer_attribute(model, "acceptable_tol", 1e-4)

    # Use L-BFGS approximation for the Hessian (common when objective is black-box)
    set_optimizer_attribute(model, "hessian_approximation", "limited-memory")

    # Tell Ipopt that Jacobians are provided by finite differences (we supply a black-box)
    set_optimizer_attribute(model, "jacobian_approximation", "finite-difference-values")

    # Merge in any user-specified Ipopt options passed via the ipopt_opts Dict
    for (k, v) in ipopt_opts
        set_optimizer_attribute(model, k, v)
    end

    # ----- declare decision variables with transformed starts where needed -----
    # For positive parameters, optimize their logs: start at log(max(x0, 1e-8)).
    @variable(model, v_A₁, start = x0[:A₁])                 # untransformed
    @variable(model, v_ψ₀, start = log(max(x0[:ψ₀], 1e-8))) # log-space
    @variable(model, v_ν,  start = log(max(x0[:ν], 1e-8)))  # log-space
    @variable(model, v_a_h, start = log(max(x0[:a_h], 1e-8)))
    @variable(model, v_b_h, start = log(max(x0[:b_h], 1e-8)))
    @variable(model, v_χ,   start = log(max(x0[:χ], 1e-8)))
    @variable(model, v_c₀,  start = log(max(x0[:c₀], 1e-8)))
    @variable(model, v_ϕ,   start = x0[:ϕ])                 # untransformed
    @variable(model, v_κ₀,  start = log(max(x0[:κ₀], 1e-8)))

    # ----- prepare fast references and moment keys -----
    # Keep a mutable reference to the latest Results to avoid allocating a new large Results
    # object on every objective evaluation. `Ref(res)` is a 1-element container we can update.
    res_ref = Ref(res)

    # Get the list of moment names from the provided NamedTuple so we can iterate deterministically.
    target_keys = collect(fieldnames(typeof(target_moments)))

    # ----- black-box objective function that JuMP will call -----
    # This function accepts the raw parameter values, updates the model primitives,
    # resolves the model, computes model moments and returns the scalar objective (SSE).
    function obj_fun(A₁, ψ₀, ν, a_h, b_h, χ, c₀, ϕ, κ₀)
        # Wrap the whole objective body so any unexpected error returns a large finite penalty
        # rather than crashing the optimizer. This lets Ipopt probe parameter space safely.
        try
            # We'll call user-provided helpers to update parameters and re-solve.
            local prim2, res2
            try
                prim2, res2 = call_update_params_and_resolve!(
                    prim, res_ref[];
                    A₁=A₁, ψ₀=ψ₀, ν=ν, a_h=a_h, b_h=b_h, χ=χ, c₀=c₀, ϕ=ϕ, κ₀=κ₀,
                )
            catch
                return 1e12
            end

            # Save the latest results so subsequent calls can warm-start from this state.
            res_ref[] = res2

            local model_moments
            try
                # Compute the vector/NamedTuple of model-implied moments for current params.
                model_moments = call_compute_model_moments(prim2, res2)
            catch
                return 1e12
            end

            # Sum of squared deviations between model moments and targets.
            s = 0.0
            @inbounds for k in target_keys
                mk = getfield(model_moments, k)   # model moment
                dk = getfield(target_moments, k)  # data/target moment

                # If either moment is not finite, bail out with a large penalty.
                if !(isfinite(mk) && isfinite(dk))
                    return 1e12
                end

                s += (mk - dk)^2
            end

            # Ensure we return a finite scalar; otherwise return the large penalty.
            return isfinite(s) ? s : 1e12
        catch
            # Any unexpected error: return large penalty so optimizer can continue.
            return 1e12
        end
    end

    # ----- register the black-box objective with JuMP -----
    # We register a function named `est_obj` with 9 arguments and tell JuMP not to use
    # automatic differentiation (autodiff = false) because the objective is a solver call.
    # JuMP requires a gradient function when autodiff=false and only the function is provided,
    # so we provide a finite-difference gradient approximator here. The gradient evaluator
    # uses deep copies of `prim`/`res` to avoid mutating the warm-start reference used by obj_fun.

    # One-time gradient debug print
    grad_debug_once = Ref(true)

    function est_obj_grad(A₁, ψ₀, ν, a_h, b_h, χ, c₀, ϕ, κ₀)
        # Pack the input into a vector for easier indexing
        x = (A₁, ψ₀, ν, a_h, b_h, χ, c₀, ϕ, κ₀)
        nx = 9
        g = zeros(Float64, nx)

        # local helper: evaluate objective at a point using fresh copies (no side-effects)
        function eval_point(xvec)
            # create local copies so update/solve won't change outer state
            prim_local = deepcopy(prim)
            res_local = deepcopy(res)
            try
                prim2, res2 = call_update_params_and_resolve!(
                    prim_local, res_local;
                    A₁=xvec[1], ψ₀=xvec[2], ν=xvec[3], a_h=xvec[4], b_h=xvec[5], χ=xvec[6], c₀=xvec[7], ϕ=xvec[8], κ₀=xvec[9],
                )
            catch
                return 1e12
            end
            try
                model_moments = call_compute_model_moments(prim2, res2)
            catch
                return 1e12
            end
            s = 0.0
            for k in target_keys
                mk = getfield(model_moments, k)
                dk = getfield(target_moments, k)
                if !(isfinite(mk) && isfinite(dk))
                    return 1e12
                end
                s += (mk - dk)^2
            end
            return isfinite(s) ? s : 1e12
        end

        # forward finite differences (n+1 evaluations) with a larger relative step to
        # avoid cancellation and zero gradients arising from solver tolerance.
        # step chosen more aggressively: relative 1e-3 or absolute floor 1e-4.
        f0 = eval_point(Float64.([x[i] for i in 1:nx]))
        for j in 1:nx
            xj = Float64(x[j])
            h = max(1e-4, abs(xj)*1e-3)
            x_plus = Float64.([x[i] for i in 1:nx])
            x_plus[j] += h
            f_plus = eval_point(x_plus)
            g[j] = (f_plus - f0) / h
        end

        # Return gradient as a tuple (JuMP accepts arrays/tuples)
        if verbose && grad_debug_once[]
            # Print a compact debug line once: f0, ||g||, min/max(|g|)
            S = sqrt(sum(abs2, g))
            gabs = abs.(g)
            gmin = minimum(gabs); gmax = maximum(gabs)
            println(@sprintf("[estimator] f0=%.4e | ||g||=%.4e | min|g|=%.4e | max|g|=%.4e", f0, S, gmin, gmax))
            grad_debug_once[] = false
        end
        return tuple(g...)
    end

        # JuMP/MathOptInterface's ReverseAD may call the gradient function with a preallocated
        # gradient storage as the first argument (an _UnsafeVectorView). Provide an in-place
        # method that fills that storage by delegating to the tuple-returning version above.
        function est_obj_grad(g_storage, A₁, ψ₀, ν, a_h, b_h, χ, c₀, ϕ, κ₀)
            # Compute gradient as tuple using existing method
            grad_tuple = est_obj_grad(A₁, ψ₀, ν, a_h, b_h, χ, c₀, ϕ, κ₀)
            # Fill the provided storage. Accept different vector-like interfaces.
            for i in 1:length(grad_tuple)
                g_storage[i] = grad_tuple[i]
            end
            return nothing
        end

    # Register the objective with an explicit gradient function (autodiff=false).
    register(model, :est_obj, 9, obj_fun, est_obj_grad; autodiff = false)

    # Use a nonlinear objective with log-transform for positive parameters via exp(...)
    @NLobjective(model, Min, est_obj(
        v_A₁,
        exp(v_ψ₀), exp(v_ν), exp(v_a_h), exp(v_b_h), exp(v_χ), exp(v_c₀),
        v_ϕ,
        exp(v_κ₀)
    ))

    # Run the optimization (this calls the registered objective many times while Ipopt probes the landscape).
    optimize!(model)

    # ----- extract results from JuMP/Ipopt -----
    status = termination_status(model)

    # Extract variable values from JuMP. Using `Ref(...)` and broadcasting `value.` keeps the code compact.
    # Extract variable values and transform back from log-space where applicable
    A₁̂ = value(v_A₁)
    ψ₀̂ = exp(value(v_ψ₀))
    ν̂  = exp(value(v_ν))
    a_ĥ = exp(value(v_a_h))
    b_ĥ = exp(value(v_b_h))
    χ̂   = exp(value(v_χ))
    c₀̂  = exp(value(v_c₀))
    ϕ̂   = value(v_ϕ)
    κ₀̂  = exp(value(v_κ₀))

    # Before reporting the objective, ensure the model's internal state is consistent with estimated params.
    prim, res_final = call_update_params_and_resolve!(
        prim, res_ref[];
        A₁=A₁̂, ψ₀=ψ₀̂, ν=ν̂, a_h=a_ĥ, b_h=b_ĥ, χ=χ̂, c₀=c₀̂, ϕ=ϕ̂, κ₀=κ₀̂,
    )

    # Compute final model moments and objective for reporting (not used by Ipopt anymore).
    final_moments = call_compute_model_moments(prim, res_final)
    final_obj = sum((getfield(final_moments, k) - getfield(target_moments, k))^2 for k in target_keys)

    # Return a compact dictionary with the estimated params, the final objective, and Ipopt's status.
    return Dict(
        :params => Dict(:A₁=>A₁̂, :ψ₀=>ψ₀̂, :ν=>ν̂, :a_h=>a_ĥ, :b_h=>b_ĥ, :χ=>χ̂, :c₀=>c₀̂, :ϕ=>ϕ̂, :κ₀=>κ₀̂),
        :objective => final_obj,
        :status => status,
    )
end

end # module
