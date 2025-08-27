# utils.jl
# Helper utilities for local objective search (stubs)

module LocalObjectiveUtils

using JSON3, Dates, Printf

export transform_params,
       inverse_transform_params,
       cache_last_solution!,
       load_ga_results,
       nth_best_candidate,
       top_k_candidates,
       extract_moment_keys

"""Apply simple bounds via log / logistic transforms (stub)."""
function transform_params(p::AbstractVector)
    return copy(p) # no-op for now
end

function inverse_transform_params(tp::AbstractVector)
    return copy(tp) # no-op for now
end

"""Cache last successful (prim, res) pair (stub)."""
function cache_last_solution!(cache::Dict, prim, res)
    cache[:prim] = prim
    cache[:res] = res
    return cache
end

"""Load GA latest results JSON.

Returns Dict with keys:
  :parameter_names :: Vector{Symbol}
  :all_params       :: Vector{Vector{Float64}}
  :all_objectives   :: Vector{Float64}
  :best_moments     :: Dict{Symbol,Float64} (if present)

Throws an error if required keys missing.
"""
function load_ga_results(path::AbstractString)
    !isfile(path) && error("GA results JSON not found: $path")
    data = JSON3.read(read(path, String))
    required = (:parameter_names, :all_params, :all_objectives)
    for k in required
        hasproperty(data, k) || haskey(data, String(k)) || error("Missing key $(k) in GA results JSON")
    end
    # Accommodate both property and dict-style
    pnames_any = hasproperty(data, :parameter_names) ? data.parameter_names : data["parameter_names"]
    params_any = hasproperty(data, :all_params) ? data.all_params : data["all_params"]
    objs_any   = hasproperty(data, :all_objectives) ? data.all_objectives : data["all_objectives"]
    # Convert types
    parameter_names = Symbol.(Vector{String}(pnames_any))
    all_params = [Vector{Float64}(p) for p in params_any]
    all_objectives = Float64.(objs_any)
    best_moments = Dict{Symbol,Float64}()
    if hasproperty(data, :best_moments) || haskey(data, "best_moments")
        raw = hasproperty(data, :best_moments) ? data.best_moments : data["best_moments"]
        for (k,v) in pairs(raw)
            try
                best_moments[Symbol(k)] = float(v)
            catch
            end
        end
    end
    return Dict(:parameter_names => parameter_names,
                :all_params => all_params,
                :all_objectives => all_objectives,
                :best_moments => best_moments)
end

"""Return vector of moment keys from best_moments section of GA results (may be empty)."""
function extract_moment_keys(results_dict::Dict)
    bm = get(results_dict, :best_moments, Dict{Symbol,Float64}())
    return collect(keys(bm))
end

"""Select the nth best (1-based) candidate after sorting by objective ascending.

Options:
  min_rel_gap: minimum relative difference in objective to treat a point as distinct
  dedup_exact: if true, drop exact duplicate parameter vectors

Returns (params::Vector{Float64}, objective::Float64, index_in_source::Int)
Throws error if n exceeds number of distinct candidates.
"""
function nth_best_candidate(results::Dict; n::Int=1, min_rel_gap::Float64=0.0, dedup_exact::Bool=true)
    all_params = results[:all_params]; all_objectives = results[:all_objectives]
    length(all_params) == length(all_objectives) || error("Mismatch all_params/all_objectives lengths")
    idxs = collect(eachindex(all_objectives))
    sort!(idxs, by=i->all_objectives[i])
    distinct = Int[]
    last_obj = nothing
    last_params = nothing
    for i in idxs
        obj = all_objectives[i]
        p = all_params[i]
        if isnothing(last_obj)
            push!(distinct, i); last_obj = obj; last_params = p; continue
        end
        rel_gap = abs(obj - last_obj) / max(abs(last_obj), 1e-12)
        if dedup_exact && last_params !== nothing && p == last_params
            continue
        end
        if rel_gap < min_rel_gap
            continue
        end
        push!(distinct, i); last_obj = obj; last_params = p
    end
    n > length(distinct) && error("Requested nth=$n but only $(length(distinct)) distinct candidates available")
    src_idx = distinct[n]
    return (all_params[src_idx], all_objectives[src_idx], src_idx)
end

"""Return top k distinct candidates as a vector of NamedTuples."""
function top_k_candidates(results::Dict; k::Int=5, min_rel_gap::Float64=0.0, dedup_exact::Bool=true)
    out = NamedTuple[]
    for n in 1:k
        try
            p,obj,idx = nth_best_candidate(results; n=n, min_rel_gap=min_rel_gap, dedup_exact=dedup_exact)
            push!(out, (rank=n, params=p, objective=obj, source_index=idx))
        catch
            break
        end
    end
    return out
end

end # module

