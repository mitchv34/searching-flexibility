### **Model Derivations**

-   [ ] **A.1. The Deterministic Benchmark Model ($\mu \to 0$)**
    -   [ ] State the joint surplus maximization problem.
    -   [ ] Derive the First-Order Condition for the interior $\alpha^*$.
    -   [ ] Derive the analytical threshold functions $\underline{\psi}(h)$ and $\overline{\psi}(h)$.
    -   [ ] Prove the properties of the optimal remote policy (monotonicity of thresholds with respect to $h$).
-   [ ] **A.2. The Full Model with Gumbel Shocks**
    -   [ ] State the choice problem with the Gumbel shock.
    -   [ ] Derive the expression for the expected flow surplus, $E[s(h,\psi)]$, showing the log-sum-exp (integral) form.
-   [ ] **A.3. Equilibrium Objects**
    -   [ ] Derive the worker's Value of Unemployment $U(h)$.
    -   [ ] Derive the recursive expression for the Match Surplus $S(h,\psi)$.
    -   [ ] Derive the full Equilibrium Wage equation $w(h,\psi)$.
    -   [ ] Derive the Free Entry Condition and the expression for Market Tightness $\theta$.
-   [ ] **A.X. Proof of Convergence to the Deterministic Limit**
    -   [ ] **TODO:** Write out the formal mathematical proof we discussed.
    -   [ ] **TODO:** Start with the log-sum-exp formula for the expected surplus.
    -   [ ] **TODO:** Use the logic of Laplace's Method to show that as $\mu \to 0$, the expected surplus converges to the maximized surplus of the deterministic model.
    -   [ ] **TODO:** Show that the probability density $P(\alpha)$ converges to a Dirac delta function at the deterministic $\alpha^*$, and therefore $E[\alpha]$ converges to $\alpha^*$.

Of course. This is the perfect way to ensure your paper is transparent and reproducible. A detailed appendix that formally defines the model's moments is a sign of high-quality research.

Here is a comprehensive summary of the moment computations, written in Markdown and suitable for direct inclusion in your paper's technical appendix. It incorporates the final, most sophisticated logic we developed, using probabilistic weighting instead of hard masks.

---

### **Appendix C: Computation of Model-Generated Moments**

This appendix provides the formal mathematical definitions for the model-generated moments used in the Simulated Method of Moments (SMM) estimation. These are the theoretical counterparts to the empirical moments described in the main text.

Given the model's structure with a continuous choice of remote work share, `α`, subject to an idiosyncratic Gumbel taste shock, the model's moments are computed as expectations. These expectations are taken over both the steady-state equilibrium distribution of employed workers, `n(h, ψ)`, and the conditional probability distribution of work arrangements, `p(α | h, ψ)`.

#### **Core Objects**

The calculation of all moments relies on three fundamental objects from the solved model:

1.  **The Employment Distribution, `n(h, ψ)`:** The steady-state mass of employed workers in a match of type `(h, ψ)`. The total mass of employed workers is `L_e = ∬ n(h, ψ) dh dψ`.

2.  **The Deterministic Value Function, `V(h, ψ; α)`:** The value of a match for a given choice of `α`, before the idiosyncratic taste shock is realized.
    $$ V(h, ψ; \alpha) = Y(h, \psi; \alpha) - c(1-\alpha) $$
    where `Y` is the output function and `c` is the baseline in-office disutility function.

3.  **The Conditional PDF of `α`, `p(α | h, ψ)`:** The probability density function for the choice of `α` in a given `(h, ψ)` match, derived from the continuous logit framework:
    $$ p(\alpha \mid h, \psi) = \frac{\exp(V(h, \psi; \alpha) / \mu)}{\int_0^1 \exp(V(h, \psi; \alpha') / \mu) d\alpha'} $$
    where `μ` is the scale of the Gumbel taste shocks.

#### **1. Unconditional Moments**

These moments are calculated over the entire population of valid employed workers.

*   **Mean of Log Wages:** The expectation of the conditional expected log wage, taken over the employment distribution.
    $$ \mathbb{E}[\log(w)] = \frac{1}{L_e} \iint \mathbb{E}[\log(w) \mid h, \psi] \cdot n(h, \psi) \,dh \,d\psi $$
    where the conditional expectation is:
    $$ \mathbb{E}[\log(w) \mid h, \psi] = \int_0^1 \log(w(h, \psi; \alpha)) \cdot p(\alpha \mid h, \psi) \,d\alpha $$

*   **Variance of Log Wages:** Calculated using the law of total variance or directly as $\mathbb{E}[(\log w)^2] - (\mathbb{E}[\log w])^2$. The expectation of the squared term is:
    $$ \mathbb{E}[(\log w)^2] = \frac{1}{L_e} \iint \left( \int_0^1 (\log(w(h, \psi; \alpha)))^2 \cdot p(\alpha \mid h, \psi) \,d\alpha \right) \cdot n(h, \psi) \,dh \,d\psi $$

*   **Mean and Variance of Alpha:** Calculated using the same logic as the wage moments, substituting `α` for `log(w)`.
    $$ \mathbb{E}[\alpha] = \frac{1}{L_e} \iint \left( \int_0^1 \alpha \cdot p(\alpha \mid h, \psi) \,d\alpha \right) \cdot n(h, \psi) \,dh \,d\psi $$
    $$ \text{Var}(\alpha) = \mathbb{E}[\alpha^2] - (\mathbb{E}[\alpha])^2 $$

#### **2. Work Arrangement Shares**

The shares are the expectation of the conditional probability of falling into a specific work arrangement category. The categories are defined by a tolerance, `α_tol` (e.g., 0.1).

*   **In-Person Share:**
    $$ \text{Share}_{\text{In-Person}} = \frac{1}{L_e} \iint \mathbb{P}(\text{In-Person} \mid h, \psi) \cdot n(h, \psi) \,dh \,d\psi $$
    where the conditional probability is the integral of the PDF over the in-person range:
    $$ \mathbb{P}(\text{In-Person} \mid h, \psi) = \int_0^{\alpha_{tol}} p(\alpha \mid h, \psi) \,d\alpha $$

*   **Remote Share:** Calculated analogously over the remote range.
    $$ \mathbb{P}(\text{Remote} \mid h, \psi) = \int_{1-\alpha_{tol}}^{1} p(\alpha \mid h, \psi) \,d\alpha $$

*   **Hybrid Share:** Calculated as the residual: `1 - Share_In-Person - Share_Remote`.

#### **3. Conditional Moments**

These moments are calculated over specific subsamples of the employed population.

*   **Difference in Average Remote Share (`diff_alpha_high_lowpsi`):** This is the difference between the conditional expectation of `α` for high-`ψ` and low-`ψ` firms. The expectation for a group `Q` (e.g., `ψ` in the top quartile) is:
    $$ \mathbb{E}[\alpha \mid \psi \in Q] = \frac{\iint_{\psi \in Q} \mathbb{E}[\alpha \mid h, \psi] \cdot n(h, \psi) \,dh \,d\psi}{\iint_{\psi \in Q} n(h, \psi) \,dh \,d\psi} $$

*   **Compensating Wage Differential (`diff_logwage_inperson_remote`):** This is the difference between the conditional expectation of log wages for in-person and remote workers. The expectation for a group `J` (e.g., In-Person) is:
    $$ \mathbb{E}[\log(w) \mid J] = \frac{\mathbb{E}[\log(w) \cdot \mathbf{I}(J)]}{\mathbb{P}(J)} $$
    where the denominator is the share of workers in group `J`, and the numerator is the total expectation of the partial expectation:
    $$ \mathbb{E}[\log(w) \cdot \mathbf{I}(J)] = \frac{1}{L_e} \iint \left( \int_{\alpha \in J} \log(w(\alpha)) \cdot p(\alpha \mid h, \psi) \,d\alpha \right) \cdot n(h, \psi) \,dh \,d\psi $$

*   **Wage Premium & Slope (`wage_premium_high_psi`, `wage_slope_psi`):** These are calculated using the same conditional expectation logic as the compensating differential, but the conditioning set is `J = {(\alpha, \psi) | \alpha > α_tol, \psi ∈ Q}`. For example:
    $$ \mathbb{E}[\log(w) \mid \text{RH}, \psi \in Q_{\text{High}}] = \frac{\mathbb{E}[\log(w) \cdot \mathbf{I}(\text{RH}) \cdot \mathbf{I}(\psi \in Q_{\text{High}})]}{\mathbb{P}(\text{RH}, \psi \in Q_{\text{High}})} $$

#### **4. Aggregate and Search Moments**

These moments are direct outputs of the solved model's equilibrium.

*   **Aggregate Productivity (`agg_productivity`):** The employment-weighted average of expected output per match.
    $$ \text{Agg. Prod.} = \frac{1}{L_e} \iint \mathbb{E}[Y(h, \psi; \alpha) \mid h, \psi] \cdot n(h, \psi) \,dh \,d\psi $$

*   **Market Tightness (`market_tightness`):** The equilibrium vacancy-to-unemployment ratio, `θ`, from the solved model.

*   **Job Finding Rate (`job_finding_rate`):** The equilibrium job finding rate, `p`, from the solved model.