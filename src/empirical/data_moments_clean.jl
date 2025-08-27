# =============================================================================
# data_moments_clean.jl
#
# Purpose: Compute empirical moments from CPS-style data.
# Author:  Mitchell Valdes-Bobes (Refactored by AI)
# Date:    2025-08-25
#
# Notes:
#   - A clean, modular, and robust version of the original script.
#   - Mirrors the logic of the Stata counterpart, including weighted statistics
#     and fixed-effect regressions with clustered standard errors.
# =============================================================================

using DataFrames
using CSV
using Statistics
using StatsBase
using FixedEffectModels
using YAML
using Printf
using Random
using Dates

# -----------------------------------------------------------------------------
# § 1. Helper Functions
# -----------------------------------------------------------------------------

"""Map CPS education codes to approximate years of schooling."""
function educ_years(code::Union{Missing,Real})
    ismissing(code) && return missing
    c = Int(code)
    val = if c == 2; 0; elseif c == 20; 8; elseif c == 30; 9;
    elseif c == 40; 10; elseif c == 50; 11; elseif c in (60, 71, 73); 12;
    elseif c == 81; 13; elseif c in (91, 92, 10); 14; elseif c == 111; 16;
    elseif c == 123; 18; elseif c == 124; 20; elseif c == 125; 21;
    else; missing; end
    return val
end

"""Weighted quantile using StatsBase."""
function weighted_quantile(x::AbstractVector, w::AbstractVector, p::Real)
    wx = [(xi, wi) for (xi, wi) in zip(x, w) if xi !== missing && wi !== missing && wi > 0 && isfinite(xi)]
    if isempty(wx)
        return missing
    end
    xs = first.(wx)
    ws = last.(wx)
    wv = Weights(ws)
    return quantile(xs, wv, p)
end

"""Robust weighted variance that filters missing values before calculation."""
function weighted_var(x::AbstractVector, w::AbstractVector)
    finite_indices = findall((!ismissing).(x) .& (!ismissing).(w) .& (w .> 0))
    if length(finite_indices) <= 1
        return missing
    end
    X = copy(x[finite_indices])
    W = Weights(w[finite_indices])
    
    EX = mean(X, W)
    
    # Stata's `summarize [pw=]` uses the uncorrected variance
    var = mean( (X .- EX).^2, W)
    return var
end

"""Create demographic and control dummy variables required for regressions."""
function add_control_dummies!(df::DataFrame)
    # Education groups
    df.educ_less_hs = map(y -> ismissing(y) ? missing : y < 12, df.years_educ)
    df.educ_hs = map(y -> ismissing(y) ? missing : y == 12, df.years_educ)
    df.educ_some_college = map(y -> ismissing(y) ? missing : 12 < y < 16, df.years_educ)
    df.educ_ba = map(y -> ismissing(y) ? missing : y == 16, df.years_educ)
    df.educ_graduate = map(y -> ismissing(y) ? missing : y > 16, df.years_educ)

    # Sex (1=male, 2=female)
    df.sex_male = map(s -> ismissing(s) ? missing : s == 1, df.sex)
    df.sex_female = map(s -> ismissing(s) ? missing : s == 2, df.sex)

    # Race groups
    df.race_white = map(r -> ismissing(r) ? missing : r == 100, df.race)
    df.race_black = map(r -> ismissing(r) ? missing : r == 200, df.race)
    df.race_asian = map(r -> ismissing(r) ? missing : r in (651, 652), df.race)
    df.race_other = map(r -> ismissing(r) ? missing : !(r in (100, 200, 651, 652)), df.race)
    return df
end

"""Write the computed moments for a given year to a YAML file."""
function write_moments_yaml(year_moments::Dict, filepath::String)
    # Define the exact order of moments for consistent output
    moment_keys = [
        :mean_logwage, :var_logwage, :p90_p10_logwage, :mean_alpha, :var_alpha,
        :inperson_share, :hybrid_share, :remote_share, :agg_productivity,
        :diff_alpha_high_lowpsi, :market_tightness, :job_finding_rate,
        :diff_logwage_inperson_remote, :wage_premium_high_psi, :wage_slope_psi,
        :wage_alpha, :wage_alpha_curvature,
        # Diagnostic/unused moments
        :mean_logwage_inperson, :mean_logwage_remote, :mean_logwage_RH_lowpsi,
        :mean_logwage_RH_highpsi, :mean_alpha_highpsi, :mean_alpha_lowpsi,
        :var_logwage_highpsi, :var_logwage_lowpsi, :ratio_var_logwage_high_lowpsi
    ]

    open(filepath, "w") do io
        write(io, "# Data moments for year $(year_moments[:year])\n")
        write(io, "# Generated on: $(Dates.format(now(), "yyyy-mm-dd"))\n\n")
        write(io, "DataMoments:\n")

        for key in moment_keys
            val = get(year_moments, key, missing)
            # Use "null" for missing values to match YAML standard
            val_str = ismissing(val) ? "null" : @sprintf("%.6f", val)
            write(io, "  $(key): $(val_str)\n")
        end
    end
    @info "Saved moments to $filepath"
end


# -----------------------------------------------------------------------------
# § 2. SMM Weighting Matrix Calculation
# -----------------------------------------------------------------------------

"""
    compute_smm_weighting_matrix(sub_df, year_moments, regressions)

Computes the optimal SMM weighting matrix (W*) for a given year's data.
"""
function compute_smm_weighting_matrix(sub_df::DataFrame, year_moments::Dict, regressions::Dict)
    @info "Computing SMM weighting matrix for year $(year_moments[:year])..."

    # Define the canonical order of moments
    micro_moment_keys = [
        :mean_logwage, :var_logwage, :mean_alpha, :var_alpha,
        :inperson_share, :hybrid_share, :remote_share,
        :diff_logwage_inperson_remote, :wage_alpha, :wage_alpha_curvature,
        :wage_premium_high_psi, :wage_slope_psi, :diff_alpha_high_lowpsi
    ]
    K = length(micro_moment_keys)
    Omega_sum = zeros(K, K)
    N = nrow(sub_df)
    sum_weights = sum(sub_df.wtfinl)

    # --- Pre-calculate residuals for all regressions ---
    # This is crucial for calculating the moment contributions efficiently
    sub_df.resid_c0 = residuals(regressions[:reg_c0])
    sub_df.resid_chi = residuals(regressions[:reg_chi])
    # Note: psi regressions are on the rh_sample, so residuals will be missing for others
    sub_df.resid_psi0 = Vector{Union{Missing,Float64}}(missing, N)
    sub_df.resid_nu = Vector{Union{Missing,Float64}}(missing, N)
    rh_indices = findall((sub_df.is_hybrid .== true) .| (sub_df.is_remote .== true))
    sub_df.resid_psi0[rh_indices] = residuals(regressions[:reg_psi0])
    sub_df.resid_nu[rh_indices] = residuals(regressions[:reg_nu])

    # --- Loop over individuals to compute moment contributions (g_i) ---
    for i in 1:N
        row = sub_df[i, :]
        w_i = row.wtfinl
        (ismissing(w_i) || w_i <= 0) && continue

        g_i = zeros(K)
        try
            # Simple moments
            g_i[1] = row.logwage - year_moments[:mean_logwage]
            g_i[2] = (row.logwage - year_moments[:mean_logwage])^2 - year_moments[:var_logwage]
            g_i[3] = row.alpha - year_moments[:mean_alpha]
            g_i[4] = (row.alpha - year_moments[:mean_alpha])^2 - year_moments[:var_alpha]
            g_i[5] = row.is_inperson - year_moments[:inperson_share]
            g_i[6] = row.is_hybrid - year_moments[:hybrid_share]
            g_i[7] = row.is_remote - year_moments[:remote_share]

            # Regression moments (regressor * residual)
            g_i[8] = row.is_inperson * row.resid_c0
            g_i[9] = row.alpha * row.resid_chi
            g_i[10] = row.alpha_sq * row.resid_chi
            g_i[11] = row.high_psi * row.resid_psi0
            g_i[12] = row.psi * row.resid_nu
            
            # Conditional mean moment (diff_alpha_high_lowpsi)
            term_high = row.high_psi * (row.alpha - year_moments[:mean_alpha_highpsi])
            term_low = (1 - row.high_psi) * (row.alpha - year_moments[:mean_alpha_lowpsi])
            g_i[13] = term_high - term_low

            # Add weighted contribution to the sum
            Omega_sum .+= w_i * (g_i * g_i')
        catch
            # Skip observation if any required value is missing
            continue
        end
    end

    # --- Assemble the final block-diagonal matrix ---
    Omega_micro = Omega_sum / sum_weights
    
    # Handle the aggregate moment (market_tightness)
    # Use the pragmatic approach: set its variance to the average of the micro-moment variances
    var_agg = mean(diag(Omega_micro))

    # Combine into the full Omega matrix
    full_moment_keys = vcat(micro_moment_keys, :market_tightness)
    Omega_full = zeros(K + 1, K + 1)
    Omega_full[1:K, 1:K] = Omega_micro
    Omega_full[K+1, K+1] = var_agg

    # The optimal weighting matrix is the inverse of Omega
    # Use pinv for numerical stability in case the matrix is nearly singular
    W_star = pinv(Omega_full)

    return W_star, full_moment_keys
end
# -----------------------------------------------------------------------------
# § 2. Main Function
# -----------------------------------------------------------------------------

"""
    compute_data_moments(; kwargs...)

Computes empirical moments from CPS data, mirroring the logic of the Stata script.
"""
function compute_data_moments(;
    input_path::String="data/processed/cps/cps_alpha_wage_present_reweighted.csv",
    years::Vector{Int}=[2022, 2023, 2024, 2025],
    alpha_tol::Float64=0.2,
    low_psi_pct::Float64=0.25,
    high_psi_pct::Float64=0.75,
    agg_productivity::Dict{Int,Float64}=Dict(2022 => 1.0971, 2023 => 1.1179, 2024 => 1.1481, 2025 => 1.1554),
    market_tightness::Dict{Int,Float64}=Dict(2022 => 2.0, 2023 => 1.4, 2024 => 1.1, 2025 => 1.07),
    min_valid_educ_years::Int=6,
    output_dir::String="data/results/data_moments",
    write_outputs::Bool=true,
    seed::Int=12345
)

    # input_path="data/processed/cps/cps_alpha_wage_present_reweighted.csv"
    # years=[2022, 2023, 2024, 2025]
    # alpha_tol=0.2
    # low_psi_pct=0.25
    # high_psi_pct=0.75
    # agg_productivity=Dict(2022 => 1.0971, 2023 => 1.1179, 2024 => 1.1481, 2025 => 1.1554)
    # market_tightness=Dict(2022 => 2.0, 2023 => 1.4, 2024 => 1.1, 2025 => 1.07)
    # min_valid_educ_years=6
    # output_dir="data/results/data_moments"
    # write_outputs=true
    # seed=12345

    Random.seed!(seed)

    # --- 1. Data Loading and Preparation ---
    @info "Reading and preparing data from $input_path"
    required_cols = [   
                        # Observation year
                        :YEAR,
                        # Final weight
                        :WTFINL,
                        # Mincer equation controls
                        :AGE,
                        :SEX,
                        :RACE,
                        :EDUC,
                        :ind_broad,
                        :OCCSOC_broad,
                        # Outcomes (remote work and wage)
                        :ALPHA,
                        :LOG_WAGE_REAL,
                        # Teleworkability to construct ψ variable
                        :TELEWORKABLE_OCSSOC_detailed,
                        :TELEWORKABLE_OCSSOC_broad,
                        :TELEWORKABLE_OCSSOC_minor
                    ]

    df = CSV.read(input_path, DataFrame, select=required_cols)
    rename!(lowercase, df)
    # Rename columns to match expected names
    rename!(df, :ind_broad => :industry)
    rename!(df, :occsoc_broad => :occupation)
    # Ensure required columns exist and have correct types
    for col in [:log_wage_real, :alpha, :educ, :age, :sex, :race, :wtfinl]
        df[!, col] = tryparse.(Float64, string.(df[!, col]))
    end

    # Filter years and create core variables
    df = df[in.(df.year, Ref(years)), :]
    df.logwage = map(x -> (ismissing(x) || x == 0 ? missing : x), df.log_wage_real)
    df.years_educ = educ_years.(df.educ)
    df = df[.!ismissing.(df.years_educ) .& (df.years_educ .>= min_valid_educ_years), :]
    df.experience = df.age .- df.years_educ .- 6
    df.experience_sq = df.experience .^ 2

    # Construct psi using hierarchy (detailed -> broad -> minor)
    df.psi = coalesce.(
        df.teleworkable_ocssoc_detailed,
        df.teleworkable_ocssoc_broad,
        df.teleworkable_ocssoc_minor
    )
    dropmissing!(df, :psi)

    # Create work arrangement and control dummies
    df.is_inperson = map(x -> ismissing(x) ? missing : x <= alpha_tol, df.alpha)
    df.is_remote = map(x -> ismissing(x) ? missing : x >= 1 - alpha_tol, df.alpha)
    df.is_hybrid = map(x -> ismissing(x) ? missing : (alpha_tol < x < 1 - alpha_tol), df.alpha)
    df.alpha_sq = df.alpha .^ 2
    add_control_dummies!(df)

    # --- 2. Moment Calculation Loop ---
    all_moments = DataFrame()
    base_controls = term(:experience) + term(:experience_sq)
    full_controls = base_controls + fe(:educ) + fe(:sex) + fe(:race) + fe(:industry) + fe(:occupation)

    # Drop missing and nothing logwage observations
    dropmissing!(df, :logwage)
    filter!(:logwage => x -> x !== nothing, df)
    # Make sure that type is vector of floats to avoid errors
    df.logwage = convert(Vector{Float64}, df.logwage)


    for yr in years
        @info "Computing moments for year $yr..."
        sub = @view df[ (df.year .== yr) .& (.!isinf.(df.logwage)), :]

        # Calculate yearly psi quantiles for high/low groups
        psi_q_low_yr = weighted_quantile(sub.psi, sub.wtfinl, low_psi_pct)
        psi_q_high_yr = weighted_quantile(sub.psi, sub.wtfinl, high_psi_pct)
        sub_df = DataFrame(sub) # Create a mutable copy to add columns
        sub_df.high_psi = map(p -> ismissing(p) ? missing : p >= psi_q_high_yr, sub_df.psi)

        # --- Basic Moments ---
        year_moments = Dict{Symbol,Any}(:year => yr)
        year_moments[:mean_logwage] = mean(sub_df.logwage, Weights(sub_df.wtfinl))
        year_moments[:var_logwage] = weighted_var(sub_df.logwage, Weights(sub_df.wtfinl))
        year_moments[:p90_p10_logwage] = weighted_quantile(sub_df.logwage, sub_df.wtfinl, 0.9) - weighted_quantile(sub_df.logwage, sub_df.wtfinl, 0.1)
        year_moments[:mean_alpha] = mean(sub_df.alpha, Weights(sub_df.wtfinl))
        year_moments[:var_alpha] = weighted_var(sub_df.alpha, Weights(sub_df.wtfinl))
        year_moments[:inperson_share] = mean(sub_df.is_inperson, Weights(sub_df.wtfinl))
        year_moments[:hybrid_share] = mean(sub_df.is_hybrid, Weights(sub_df.wtfinl))
        year_moments[:remote_share] = mean(sub_df.is_remote, Weights(sub_df.wtfinl))

        # --- Regression-Based Moments ---
        rh_sample = @view sub_df[(sub_df.is_hybrid .== true) .| (sub_df.is_remote .== true), :]

        # Save regression objects for weighting matrix computation
        regressions = Dict{Symbol,Any}()
        try
            # Moment for c0 (dummy variable approach)
            regressions[:reg_c0] = reg(sub_df, term(:logwage) ~ term(:is_hybrid) + term(:is_inperson) + full_controls, Vcov.cluster(:industry, :occupation), weights=:wtfinl,  save = :residuals)
            year_moments[:diff_logwage_inperson_remote] = coef(regressions[:reg_c0])[2]

            # Moments for c0 and chi (continuous alpha approach)
            regressions[:reg_chi] = reg(sub_df, term(:logwage) ~ term(:alpha) + term(:alpha_sq) + full_controls, Vcov.cluster(:industry, :occupation), weights=:wtfinl,  save = :residuals)
            year_moments[:wage_alpha] = coef(regressions[:reg_chi])[1]
            year_moments[:wage_alpha_curvature] = coef(regressions[:reg_chi])[2]

            # Moment for psi0 (high_psi premium in RH sample)
            regressions[:reg_psi0] = reg(rh_sample, term(:logwage) ~ term(:high_psi) + full_controls, Vcov.cluster(:industry, :occupation), weights=:wtfinl,  save = :residuals)
            year_moments[:wage_premium_high_psi] = coef(regressions[:reg_psi0])[1]

            # Moment for nu (psi slope in RH sample)
            regressions[:reg_nu] = reg(rh_sample, term(:logwage) ~ term(:psi) + full_controls, Vcov.cluster(:industry, :occupation), weights=:wtfinl,  save = :residuals)
            year_moments[:wage_slope_psi] = coef(regressions[:reg_nu])[1]
        catch e
            @warn "A regression failed for year $yr, some moments will be missing. Error: $e"
        end

        # --- Conditional Mean Moments ---
        high_psi_sample = @view sub_df[sub_df.high_psi .== true, :]
        low_psi_sample = @view sub_df[sub_df.high_psi .== false, :]
        year_moments[:mean_alpha_highpsi] = mean(high_psi_sample.alpha, Weights(high_psi_sample.wtfinl))
        year_moments[:mean_alpha_lowpsi] = mean(low_psi_sample.alpha, Weights(low_psi_sample.wtfinl))
        if !ismissing(year_moments[:mean_alpha_highpsi]) && !ismissing(year_moments[:mean_alpha_lowpsi])
            year_moments[:diff_alpha_high_lowpsi] = year_moments[:mean_alpha_highpsi] - year_moments[:mean_alpha_lowpsi]
        end

        # --- Aggregate Moments ---
        year_moments[:agg_productivity] = get(agg_productivity, yr, missing)
        year_moments[:market_tightness] = get(market_tightness, yr, missing)
        year_moments[:job_finding_rate] = missing # Placeholder

        # --- Store Results ---
        push!(all_moments, year_moments, cols=:union)
        if write_outputs
            mkpath(output_dir)
            write_moments_yaml(year_moments, joinpath(output_dir, "data_moments_$(yr).yaml"))

            if !isempty(regressions)
                W_star, W_keys = compute_smm_weighting_matrix(sub_df, year_moments, regressions)
                W_df = DataFrame(W_star, W_keys)
                insertcols!(W_df, 1, :moment => W_keys)
                w_path = joinpath(output_dir, "smm_weighting_matrix_$(yr).csv")
                CSV.write(w_path, W_df)
                @info "Saved SMM weighting matrix for year $yr to $w_path"
            else
                @warn "Skipping weighting matrix for year $yr due to regression failures."
            end

        end
    end

    # --- 3. Final Output ---
    if write_outputs
        csv_path = joinpath(output_dir, "data_moments_by_year.csv")
        CSV.write(csv_path, all_moments)
        @info "Saved summary of all moments to $csv_path"
    end

    return all_moments
end

compute_data_moments()