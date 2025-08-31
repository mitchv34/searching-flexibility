## **Counterfactual Analysis: Decomposing the Post-Pandemic Shift**

To understand the economic forces driving the changes in the labor market between the pre-pandemic (2019) and post-pandemic (2024) steady states, we conduct a series of counterfactual experiments. The primary goal is to decompose the observed changes in work arrangements, aggregate productivity, and wage inequality into two distinct channels: (1) a **technology channel**, representing improvements in firms' ability to operate remotely, and (2) a **preference channel**, representing a shift in workers' valuation of remote work.

### **Experimental Design: The Three Core Economies**

All decomposition experiments rely on a comparison of three model specifications:

1.  **2019 Equilibrium (Baseline):** The model is solved using the full set of parameters estimated to match the 2019 data, denoted $\theta_{2019}$. This represents the pre-pandemic steady state.
2.  **2024 Equilibrium (New Reality):** The model is solved using the full set of parameters estimated to match the 2024 data, denoted $\theta_{2024}$. This represents the post-pandemic steady state.
3.  **Counterfactual Equilibrium (Hybrid World):** The model is solved using a hybrid set of parameters, $\theta_{CF}$. This hybrid combines the **production technology** from the pre-pandemic era with the **worker preferences** from the post-pandemic era.
    *   **Production Parameters:** $\{A_0, A_1, \psi_0, \nu, \phi\}_{2019}$
    *   **Preference Parameters:** $\{c_0, \chi, \mu\}_{2024}$
    *   **Market Structure Parameters:** All other parameters (e.g., search frictions `κ₀`, bargaining power `ξ`, separation rate `δ`) are held at their **2024 values**. This ensures the counterfactual experiment is conducted within the context of the modern labor market structure, isolating the historical change in preferences and technology from concurrent shifts in market dynamics.

### **Counterfactual 1: What Drove the Rise in Remote Work?**

**Objective:** To quantify how much of the increase in the average remote work share, $\mathbb{E}[\alpha]$, was caused by better technology versus a stronger collective preference for remote work.

#### **Methodology & Decomposition**

We solve each of the three models and compute the aggregate remote work share for each: $\mathbb{E}[\alpha]_{2019}$, $\mathbb{E}[\alpha]_{2024}$, and $\mathbb{E}[\alpha]_{CF}$. The total change is then decomposed as follows:

-   **Total Observed Change:**
    $$\Delta_{\text{Total}} = \mathbb{E}[\alpha]_{2024} - \mathbb{E}[\alpha]_{2019}$$

-   **Effect of 2024 Preferences with 2019 Technology (The "Preference Channel"):** The difference between the counterfactual and the 2019 baseline isolates the effect of changing worker preferences while holding production technology constant.
    $$ \Delta_{\text{Preferences}} = \mathbb{E}[\alpha]_{CF} - \mathbb{E}[\alpha]_{2019} $$

-   **Effect of 2024 Technology with 2024 Preferences (The "Technology Channel"):** The residual change isolates the effect of improved technology, given that preferences have already shifted to their new, post-pandemic state.
    $$ \Delta_{\text{Technology}} = \mathbb{E}[\alpha]_{2024} - \mathbb{E}[\alpha]_{CF} $$

The share of the total change driven by the preference channel is then $\frac{\Delta_{\text{Preferences}}}{\Delta_{\text{Total}}}$.¹

---
¹ *This decomposition assigns the interaction effect between the shift in preferences and the shift in technology entirely to the technology channel. This is a standard approach, interpreting the technology effect as its marginal contribution given that preferences have already shifted.*

---

### **Counterfactual 2: How Did Remote Work Affect Aggregate Productivity?**

**Objective:** To determine whether the shift to remote work had a net positive or negative impact on the economy's average output per worker, and to disentangle the effects of pure technological improvement from the effects of labor re-sorting driven by new preferences.

#### **Methodology & Decomposition**

We use the same three solved models but now focus on aggregate productivity as the outcome variable. The total change is decomposed similarly:

-   **Total Observed Change:**
    $$ \Delta_{\text{Total}}^{\text{Prod}} = \text{Productivity}_{2024} - \text{Productivity}_{2019} $$

-   **Effect of Preferences & Re-sorting:** This isolates the impact of the massive re-sorting of workers across firms and work arrangements caused by the shift in preferences, holding the production technology fixed at 2019 levels. This effect can be positive (if workers sort to better matches) or negative (if the preference for the remote amenity leads to a net misallocation of talent).
    $$ \Delta_{\text{Preferences}}^{\text{Prod}} = \text{Productivity}_{CF} - \text{Productivity}_{2019} $$

-   **Effect of Technology:** This isolates the pure productivity gain from firms and workers getting better at remote collaboration, given the new sorting pattern of the labor market.
    $$ \Delta_{\text{Technology}}^{\text{Prod}} = \text{Productivity}_{2024} - \text{Productivity}_{CF} $$

This decomposition allows for a nuanced conclusion about remote work's impact on productivity. For instance, it can reveal whether a net positive change was driven entirely by technology while being partially offset by a negative re-sorting effect.

### **Counterfactual 3: The Impact on Wage Inequality**

**Objective:** To understand whether the rise of remote work increased or decreased wage inequality, and to isolate the portion of the change driven by the shift in worker preferences.

#### **Methodology & Decomposition**

Using the same three economies, we now compute a measure of wage inequality, such as the variance of log wages, $\mathbb{V}\text{ar}(\log w)$, for each.

-   **Total Observed Change:**
    $$ \Delta_{\text{Total}}^{\text{Ineq}} = \mathbb{V}\text{ar}(\log w)_{2024} - \mathbb{V}\text{ar}(\log w)_{2019} $$

-   **Effect of Preferences & Re-sorting:** This isolates how the new desire for remote work changed the wage structure through re-sorting, holding technology constant.
    $$ \Delta_{\text{Preferences}}^{\text{Ineq}} = \mathbb{V}\text{ar}(\log w)_{CF} - \mathbb{V}\text{ar}(\log w)_{2019} $$

-   **Effect of Technology:** This captures how technological improvements further altered the wage distribution.
    $$ \Delta_{\text{Technology}}^{\text{Ineq}} = \mathbb{V}\text{ar}(\log w)_{2024} - \mathbb{V}\text{ar}(\log w)_{CF} $$

This experiment can reveal, for example, if the preference shift increased inequality by allowing high-skilled workers to better leverage their skills in a new sorting equilibrium, while technological advances had a more equalizing effect.

---
## **Further Experiments: Technology and Policy**

### **Experiment 4: Decomposing Skill-Firm Complementarity**

**Objective:** To disentangle the relative importance of the **worker skill channel (`ϕ`)** versus the **firm technology channel (`ν`)** in driving aggregate outcomes.

#### **Experimental Design**

This experiment is a grid-based sensitivity analysis around the estimated 2024 equilibrium.

1.  **Baseline:** We start from the estimated parameter set for the 2024 steady state, $\theta_{2024}$.
2.  **Parameter Grids:** We construct grids for the skill-complementarity parameter, `ϕ`, and the firm-complementarity parameter, `ν`, around their estimated 2024 values.
3.  **Simulation Loop:** The model is re-solved for the full steady-state equilibrium at each of the `(ϕᵢ, νⱼ)` points on the grid. **Crucially, for each point, we re-calibrate the vacancy cost scale `κ₀` to hold the aggregate unemployment rate constant at its 2024 level.** This ensures we are comparing economies with different internal complementarity structures but the same overall level of labor market slack, thus isolating the pure effect of complementarity.

#### **Analysis and Visualization**

The primary method for analyzing the results is a **contour plot**, where the x-axis is the skill channel (`ϕ`) and the y-axis is the firm channel (`ν`). The contour lines ("isoquants") will connect the `(ϕ, ν)` pairs that produce the same level of a given outcome (e.g., aggregate productivity). The slope and curvature of these isoquants will reveal the nature of the interaction between the two channels.

### **Experiment 5: Policy Simulation - "Return-to-Office" Mandates**

**Objective:** To quantify the aggregate and distributional consequences of a large-scale "return-to-office" (RTO) mandate.

#### **Experimental Design**

1.  **Start** from the estimated 2024 equilibrium.
2.  **Introduce the Policy:** An RTO mandate is modeled as a hard cap on the maximum allowable remote work share, $\alpha_{\text{max}}$. For a mandatory 3-day in-office week, for example, $\alpha_{\text{max}} = 0.4$.
3.  **Re-solve the Model:** This constraint must be incorporated directly into the calculation of the flow surplus, $s(h, \psi)$. The integral for the "inclusive value" is now taken over the restricted range $[0, \alpha_{\text{max}}]$ instead of $[0,1]$. This change in the fundamental surplus calculation will alter all subsequent choices and market outcomes.
4.  **Analyze Results:** Compare the new constrained equilibrium to the 2024 baseline. We will quantify the impact on aggregate productivity, total surplus, and the unemployment rates and wages for different skill groups `h`.
### **Counterfactual 4: The Role of Idiosyncratic Preferences**

**Objective:** To quantify the aggregate importance of unobserved worker taste heterogeneity in shaping the post-pandemic labor market. This experiment isolates the impact of individual preferences from the impact of observable productivity and cost factors. It answers the question: "How much of the observed diversity in work arrangements is due to workers being fundamentally different in their tastes, versus simply sorting into different types of jobs?"

#### **Experimental Design: A Deterministic Counterpart**

This experiment compares the main 2024 equilibrium to a counterfactual world where all idiosyncratic taste shocks are eliminated.

1.  **2024 Equilibrium (Baseline):** The model is solved using the full set of estimated 2024 parameters, $\theta_{2024}$. This is the "real world" model where worker choices are probabilistic, driven by the random utility `V(α) + με(α)`.

2.  **"No Heterogeneity" Counterfactual:** We create a deterministic version of the 2024 economy.
    *   **Parameters:** We use the *exact same* estimated 2024 parameter set, $\theta_{2024}$.
    *   **The Change:** We shut down the preference heterogeneity channel by setting the Gumbel scale parameter to its theoretical limit: **`μ → 0`**.
    *   **The Implication:** As `μ` approaches zero, the random component of utility vanishes. The worker's choice of `α` is no longer probabilistic. Instead, for each `(h, ψ)` match, the worker and firm deterministically choose the single `α*` that maximizes the deterministic flow value, `V(α) = Y(α) - c(1-α)`. This transforms the continuous logit model into its deterministic benchmark counterpart.

#### **Methodology & Analysis**

We solve both the baseline and the counterfactual models and compare their aggregate and distributional outcomes.

*   **Outcome 1: The Distribution of Work Arrangements:**
    *   **Baseline:** The model produces a smooth, continuous distribution of `α`, which we can bin to show shares of in-person, hybrid, and remote workers.
    *   **Counterfactual:** The model will produce massive "bunching." All workers within a given `(h, ψ)` region will make the exact same, discrete choice (either fully in-person, fully remote, or a specific hybrid value).
    *   **Analysis:** The difference between these two distributions provides a direct, quantitative measure of how much of the observed "smear" of hybrid work arrangements is attributable purely to taste differences, rather than productivity differences.

*   **Outcome 2: Aggregate Welfare and Productivity:**
    *   We will compute the change in aggregate output and total worker welfare (the sum of utility flows) between the two economies.
    *   **Analysis:** This allows us to quantify the "gains from variety." By forcing a "one-size-fits-all" arrangement on workers with the same observable characteristics (by setting `μ=0`), we can measure the resulting loss in worker welfare and any potential change in aggregate productivity. This speaks directly to the economic value of offering flexible work arrangements.

*   **Outcome 3: Sorting and Wage Inequality:**
    *   We will analyze how the sorting pattern of workers to firms (`Corr(h, ψ)`) and the variance of log wages change when preference heterogeneity is removed.
    *   **Analysis:** This can reveal whether taste-based sorting (workers choosing firms that offer their preferred amenity) is a significant driver of the observed wage structure, separate from the standard productivity-based sorting.

This counterfactual provides a clean and powerful way to isolate one of the most debated aspects of the new labor market: the role of individual tastes versus structural factors. It fits perfectly into your existing list and elevates the paper's contribution by tackling this fundamental question head-on.