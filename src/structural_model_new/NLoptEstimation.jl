#==========================================================================================
Title: NLopt Estimation Module
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2025-08-14
Description: Estimate model parameters using the NLopt optimizers (derivative-free by default).
==========================================================================================#
module NLoptEstimation

using NLopt

export estimate_with_nlopt

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

# Resolve NLopt algorithm from a Symbol or fallback to default
function _algo_from_symbol(sym)
    if sym isa Symbol && hasproperty(NLopt, sym)
        return getproperty(NLopt, sym)
    else
        return NLopt.LN_BOBYQA  # robust derivative-free default with bounds
    end
end

"""
    estimate_with_nlopt(prim, res, target_moments::NamedTuple;
                        nlopt_opts = (;), verbose::Bool=false)

Estimate parameters by minimizing SSE between model and target moments using NLopt.
Parameters estimated (9): A₁, ψ₀, ν, a_h, b_h, χ, c₀, ϕ, κ₀

nlopt_opts keys (optional):
- algorithm::Symbol = :LN_BOBYQA
- maxeval::Int = 500
- maxtime::Real = Inf
- xtol_rel::Real = 1e-6
- ftol_rel::Real = 1e-8
- lower_bounds::Union{AbstractVector,Dict} = auto
- upper_bounds::Union{AbstractVector,Dict} = auto
"""
function estimate_with_nlopt(
    prim,
    res,
    target_moments::NamedTuple;
    nlopt_opts = (;),
    verbose::Bool=false,
)
    # Parameter order and starts
    params = [:A₁, :ψ₀, :ν, :a_h, :b_h, :χ, :c₀, :ϕ, :κ₀]
    n = length(params)
    x0 = similar(zeros(Float64, n))
    for (i, k) in pairs(params)
        x0[i] = hasfield(typeof(prim), k) ? Float64(getfield(prim, k)) : 0.0
    end

    # Default bounds (wide). Positive-only parameters get [1e-8, 1e3].
    pos_syms = Set([:ψ₀, :ν, :a_h, :b_h, :χ, :c₀, :κ₀])
    lb = fill(-1e3, n)
    ub = fill(+1e3, n)
    for (i, k) in pairs(params)
        if k in pos_syms
            lb[i] = 1e-8
            ub[i] = 1e3
        end
    end
    # Allow user overrides via Dict{Symbol,Float64} or vectors
    if haskey(nlopt_opts, :lower_bounds)
        B = nlopt_opts[:lower_bounds]
        if B isa AbstractVector
            lb .= Float64.(B)
        elseif B isa Dict
            for (i, k) in pairs(params); if haskey(B, k); lb[i] = Float64(B[k]); end; end
        end
    end
    if haskey(nlopt_opts, :upper_bounds)
        B = nlopt_opts[:upper_bounds]
        if B isa AbstractVector
            ub .= Float64.(B)
        elseif B isa Dict
            for (i, k) in pairs(params); if haskey(B, k); ub[i] = Float64(B[k]); end; end
        end
    end

    algo = _algo_from_symbol(get(nlopt_opts, :algorithm, :LN_BOBYQA))
    opt = NLopt.Opt(algo, n)
    NLopt.lower_bounds!(opt, lb)
    NLopt.upper_bounds!(opt, ub)

    # Tolerances and limits
    NLopt.maxeval!(opt, Int(get(nlopt_opts, :maxeval, 500)))
    NLopt.maxtime!(opt, get(nlopt_opts, :maxtime, Inf))
    NLopt.xtol_rel!(opt, get(nlopt_opts, :xtol_rel, 1e-6))
    NLopt.ftol_rel!(opt, get(nlopt_opts, :ftol_rel, 1e-8))

    # Objective: SSE of moments. Penalty on failure.
    target_keys = collect(fieldnames(typeof(target_moments)))
    res_ref = Ref(res)

    function obj!(x::Vector{Float64}, grad::Vector{Float64})
        # We use a derivative-free algorithm; grad is ignored (fill zeros if requested).
        if !isempty(grad); fill!(grad, 0.0); end
        # Map x to named parameters
        A₁, ψ₀, ν, a_h, b_h, χ, c₀, ϕ, κ₀ = x
        # Update and resolve
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
        # Moments and SSE
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
            if !(isfinite(mk) && isfinite(dk)); return 1e12; end
            s += (mk - dk)^2
        end
        return isfinite(s) ? s : 1e12
    end

    NLopt.min_objective!(opt, obj!)

    x = copy(x0)
    minf, ret = NLopt.optimize!(opt, x)  # x is updated to the best found
    status = ret

    # Build output parameter dict
    est = Dict{Symbol,Float64}()
    for (i, k) in pairs(params); est[k] = x[i]; end

    # Compute final objective and moments
    prim_final, res_final = call_update_params_and_resolve!(
        prim, res_ref[];
        A₁=est[:A₁], ψ₀=est[:ψ₀], ν=est[:ν], a_h=est[:a_h], b_h=est[:b_h],
        χ=est[:χ], c₀=est[:c₀], ϕ=est[:ϕ], κ₀=est[:κ₀],
    )
    final_moments = call_compute_model_moments(prim_final, res_final)
    final_obj = sum((getfield(final_moments, k) - getfield(target_moments, k))^2 for k in target_keys)

    return Dict(
        :params => est,
        :objective => final_obj,
        :status => status,
        :minf => minf,
        :x => x,
    )
end

end # module