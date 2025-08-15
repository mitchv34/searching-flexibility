#==========================================================================================
Title: MadNLP Estimation Module
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2025-08-14
Description: Estimate model parameters using the MadNLP optimizer via JuMP.
==========================================================================================#
module MadNLPEstimation

using JuMP, MadNLP

export estimate_with_madnlp

# Forwarders to call helpers defined in Main if available
function call_update_params_and_resolve!(prim_arg, res_arg; kwargs...)
    if isdefined(Main, :update_params_and_resolve!)
        try
            return Main.update_params_and_resolve!(prim_arg, res_arg; kwargs...)
        catch
        end
    end
    return update_params_and_resolve!(prim_arg, res_arg; kwargs...)
end

function call_compute_model_moments(prim_arg, res_arg)
    if isdefined(Main, :compute_model_moments)
        try
            return Main.compute_model_moments(prim_arg, res_arg)
        catch
        end
    end
    return compute_model_moments(prim_arg, res_arg)
end

"""
    estimate_with_madnlp(prim, res, target_moments::NamedTuple;
                         madnlp_opts=NamedTuple(),
                         verbose::Bool=false)

Estimate parameters by minimizing SSE between model and target moments using MadNLP.
Parameters estimated (9): A₁, ψ₀, ν, a_h, b_h, χ, c₀, ϕ, κ₀
Returns Dict(:params, :objective, :status)
"""
function estimate_with_madnlp(
    prim,
    res,
    target_moments::NamedTuple;
    madnlp_opts = (;),
    verbose::Bool=false,
)
    # Parameter symbols and starts
    params = [:A₁, :ψ₀, :ν, :a_h, :b_h, :χ, :c₀, :ϕ, :κ₀]
    x0 = Dict{Symbol,Float64}()
    for k in params
        if hasfield(typeof(prim), k)
            x0[k] = Float64(getfield(prim, k))
        else
            x0[k] = 0.0
        end
    end

    # Build model with MadNLP optimizer
    model = Model(() -> MadNLP.Optimizer())

    # Output level
    set_optimizer_attribute(model, "print_level", verbose ? MadNLP.INFO : MadNLP.WARN)
    # Typical tolerances
    set_optimizer_attribute(model, "max_iter", 500)
    set_optimizer_attribute(model, "tol", 1e-6)
    set_optimizer_attribute(model, "acceptable_tol", 1e-4)
    # Use exact or quasi-Newton Hessian depending on availability. For black-box NLobj,
    # we only provide value and gradient; MadNLP can use quasi-Newton.
    # Users can override via `madnlp_opts`.

    # Apply user-specified options
    for (k, v) in pairs(madnlp_opts)
        set_optimizer_attribute(model, String(k), v)
    end

    # Decision vars with starting values
    # Optimize logs for strictly positive parameters to keep them positive
    @variable(model, v_A₁, start = x0[:A₁])                  # untransformed
    @variable(model, v_ψ₀, start = log(max(x0[:ψ₀], 1e-8)))  # log-space
    @variable(model, v_ν,  start = log(max(x0[:ν], 1e-8)))   # log-space
    @variable(model, v_a_h, start = log(max(x0[:a_h], 1e-8)))
    @variable(model, v_b_h, start = log(max(x0[:b_h], 1e-8)))
    @variable(model, v_χ,   start = log(max(x0[:χ], 1e-8)))
    @variable(model, v_c₀,  start = log(max(x0[:c₀], 1e-8)))
    @variable(model, v_ϕ,   start = x0[:ϕ])                  # untransformed
    @variable(model, v_κ₀,  start = log(max(x0[:κ₀], 1e-8)))

    res_ref = Ref(res)
    target_keys = collect(fieldnames(typeof(target_moments)))

    function obj_fun(A₁, ψ₀, ν, a_h, b_h, χ, c₀, ϕ, κ₀)
        try
            local prim2, res2
            try
                prim2, res2 = call_update_params_and_resolve!(
                    prim, res_ref[];
                    A₁=A₁, ψ₀=ψ₀, ν=ν, a_h=a_h, b_h=b_h, χ=χ, c₀=c₀, ϕ=ϕ, κ₀=κ₀,
                )
            catch
                return 1e12
            end
            res_ref[] = res2
            local model_moments
            try
                model_moments = call_compute_model_moments(prim2, res2)
            catch
                return 1e12
            end
            s = 0.0
            @inbounds for k in target_keys
                mk = getfield(model_moments, k)
                dk = getfield(target_moments, k)
                if !(isfinite(mk) && isfinite(dk))
                    return 1e12
                end
                s += (mk - dk)^2
            end
            return isfinite(s) ? s : 1e12
        catch
            return 1e12
        end
    end

    # Finite-diff gradient (central) for black-box objective
    function est_obj_grad(A₁, ψ₀, ν, a_h, b_h, χ, c₀, ϕ, κ₀)
        x = (A₁, ψ₀, ν, a_h, b_h, χ, c₀, ϕ, κ₀)
        nx = 9
        g = zeros(Float64, nx)
        # Larger relative step to overcome inner-solve noise
        for j in 1:nx
            xj = Float64(x[j])
            h = max(1e-5, abs(xj)*1e-3)
            x_plus  = ntuple(i -> i == j ? x[i] + h : x[i], nx)
            x_minus = ntuple(i -> i == j ? x[i] - h : x[i], nx)
            f_plus  = obj_fun(x_plus...)
            f_minus = obj_fun(x_minus...)
            g[j] = (f_plus - f_minus) / (2h)
        end
        return tuple(g...)
    end

    function est_obj_grad(g_storage, A₁, ψ₀, ν, a_h, b_h, χ, c₀, ϕ, κ₀)
        grad_tuple = est_obj_grad(A₁, ψ₀, ν, a_h, b_h, χ, c₀, ϕ, κ₀)
        @inbounds for i in 1:length(grad_tuple)
            g_storage[i] = grad_tuple[i]
        end
        return nothing
    end

    register(model, :est_obj, 9, obj_fun, est_obj_grad; autodiff = false)
    # Use log-transformed variables inside the NLobjective via exp(...) to enforce positivity
    @NLobjective(model, Min, est_obj(
        v_A₁,
        exp(v_ψ₀), exp(v_ν), exp(v_a_h), exp(v_b_h), exp(v_χ), exp(v_c₀),
        v_ϕ,
        exp(v_κ₀)
    ))

    optimize!(model)

    status = termination_status(model)

    A₁̂ = value(v_A₁)
    ψ₀̂ = exp(value(v_ψ₀))
    ν̂  = exp(value(v_ν))
    a_ĥ = exp(value(v_a_h))
    b_ĥ = exp(value(v_b_h))
    χ̂   = exp(value(v_χ))
    c₀̂  = exp(value(v_c₀))
    ϕ̂   = value(v_ϕ)
    κ₀̂  = exp(value(v_κ₀))

    prim, res_final = call_update_params_and_resolve!(
        prim, res_ref[]; A₁=A₁̂[], ψ₀=ψ₀̂[], ν=ν̂[], a_h=a_ĥ[], b_h=b_ĥ[], χ=χ̂[], c₀=c₀̂[], ϕ=ϕ̂[], κ₀=κ₀̂[],
    )

    final_moments = call_compute_model_moments(prim, res_final)
    target_keys2 = collect(fieldnames(typeof(target_moments)))
    final_obj = sum((getfield(final_moments, k) - getfield(target_moments, k))^2 for k in target_keys2)

    return Dict(
    :params => Dict(:A₁=>A₁̂, :ψ₀=>ψ₀̂, :ν=>ν̂, :a_h=>a_ĥ, :b_h=>b_ĥ, :χ=>χ̂, :c₀=>c₀̂, :ϕ=>ϕ̂, :κ₀=>κ₀̂),
        :objective => final_obj,
        :status => status,
    )
end

end # module
