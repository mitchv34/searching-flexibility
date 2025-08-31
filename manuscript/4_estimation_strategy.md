# Estimation Strategy


### **Calibration of Unemployment Value $b(h)$: A Summary**

This section outlines the calibration strategy for the flow value of unemployment, $b(h)$. We depart from a simple constant replacement rate of wages and instead adopt a more theoretically consistent approach that accounts for the non-pecuniary amenities central to our model.

*   **Economic Rationale:** In a model where a job's value includes both a wage and a significant amenity component (from the choice of remote work), the worker's outside option should reflect the total utility of employment, not just the wage. A worker's decision to accept a job is based on the total surplus $S(h,\psi)$, which encapsulates both pecuniary and non-pecuniary gains. Therefore, the value of non-market time (leisure, home production) should be benchmarked against this total expected gain from employment.

*   **Functional Form ("Surplus Replacement Rate"):** We model the flow value of unemployment for a worker of skill $h$ as a fraction of the expected utility gain they would receive from finding a job. This gain is their share ($\xi$) of the expected total match surplus ($E[S|h]$).
    $$ b(h) = b \cdot \xi \cdot \mathbb{E}_{\psi}[S(h, \psi)] $$

*   **Parameter Interpretation:** The single parameter to be calibrated, $b$, now has a sharp economic interpretation: it is the **value of non-market time as a fraction of the value of market time.** It represents how valuable leisure and home production are relative to the surplus generated in a formal job.

*   **Endogenous Calculation:** $b(h)$ is not a fixed primitive. It is an **endogenous object** calculated within the model's equilibrium. In each iteration of the solver, the model uses the current state of the economy (the surplus matrix $S$ and the vacancy distribution $\Gamma$) to compute the expected surplus for each worker type, $E[S|h]$, and then updates the $b(h)$ vector accordingly. This creates a realistic general equilibrium feedback loop.

*   **Calibration Strategy for $b$: ** The parameter $b$ is not directly observable. We will calibrate it externally based on standard values from the search and home production literature.
    *   **Target Value:** A standard value for the ratio of the value of non-market time to market time is approximately **0.5**. This is a common benchmark in quantitative macroeconomics.
    *   **Justification:** This value is consistent with a wide range of microeconomic estimates and is a standard choice in models that require a calibration for the value of leisure (e.g., Hall and Milgrom, 2008). We will set **$b = 0.5$** in our baseline calibration.


### **Why the Aggregate Approximation is Sufficient for Calibrating `γ₁`**

Given that `γ₁` is part of your "Group 1: Externally Calibrated Parameters," the aggregate approximation method is a great choice for the following reasons:

1.  **It's Disciplined by Data:** You are not just picking a number like `0.5` out of thin air. Your procedure uses high-quality, aggregate data from JOLTS and published research on labor flows. This makes your choice of `γ₁` transparent and grounded in empirical evidence.

2.  **It Captures the Right Magnitude:** While the approximation might miss some of the subtle cyclical or cross-sectional variation, it will get you into the correct **economic ballpark**. The literature consistently finds that the matching elasticity is somewhere between 0.3 and 0.7. Your method will almost certainly produce a value in this range, which is the primary goal of a calibration.

3.  **It's a Common Practice:** Using this type of informed approximation for calibrated parameters is a standard and well-accepted practice in quantitative macroeconomics. You are on solid methodological ground.

4.  **The Burden of Proof is Lower:** Since you are not claiming to have a new, superior estimate of the matching elasticity (it's not the main contribution of your paper), you don't need to use the absolute state-of-the-art, microdata-intensive method. You just need to demonstrate that you have chosen a reasonable value based on a transparent procedure.



### **How to Justify This in Your Paper**

This is the most important part. You need to be clear and upfront about your methodology in the calibration section of your paper.

Here is a template for how you would write this up:

**Parameter `γ₁` (Matching Elasticity):** The elasticity of the matching function, `γ₁`, is a crucial parameter governing the efficiency of the search process. While a wide range of estimates exist in the literature, we calibrate this parameter using a procedure that reflects the aggregate dynamics of the U.S. labor market. We construct a state-level panel of monthly hires, vacancies, and unemployment from 2001 to 2024. State-level vacancy and unemployment data are taken directly from the JOLTS and LAUS programs, respectively. As state-level U-to-E flows are not directly published, we approximate them by multiplying the total state-level hires from JOLTS (`H_it`) by the national, time-varying share of hires that come from unemployment, which we calculate from the aggregate CPS labor force flow data. This approach, while abstracting from state-level heterogeneity in flow shares, allows us to capture the crucial business-cycle variation in hiring sources. We then estimate the standard panel fixed-effects model of Hall and Schulhofer-Wohl (2018):
$$ \ln(f_{it}) = \delta_i + \delta_t + (1-\gamma_1) \ln(\theta_{it}) + \varepsilon_{it} $$
This procedure yields an estimate of **`γ̂₁ = [Your Value]`**, which we use in our baseline calibration. This value is consistent with recent estimates in the literature.

**Why this justification is strong:**
*   It is **transparent** about the approximation being made.
*   It **justifies** the approximation by highlighting that it still captures the important time-series variation.
*   It **cites** the state-of-the-art methodology (HSW) that it is based on.
*   It **benchmarks** the final result against the broader literature.

**Conclusion:**

Yes, for the purpose of **calibration**, your proposed aggregate method is not just sufficient; it is a good, pragmatic, and defensible choice. It provides a data-driven value for `γ₁` without requiring the massive overhead of a full microdata-based estimation for what is ultimately a background parameter in your study. You can proceed with this plan with confidence.

- [ ] **4.1. Overview:** Describe the SMM approach and the two-stage estimation strategy.
- [ ] **4.2. Identification:**
    - [ ] Insert the main identification table linking parameters to moments.
    - [ ] Write a paragraph for each parameter (or group of parameters) justifying the choice of moment, referencing the detailed analysis in Appendix B.