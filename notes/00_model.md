Of course. I understand completely. You want to maintain the detailed, rigorous content of the continuous logit model document but present it with the structure, tone, and pedagogical clarity of your other write-up.

The goal is to create a self-contained, formal model description that is detailed enough for an appendix but clear enough for the main body of a paper. I will integrate the discrete-time derivations and detailed explanations into the continuous logit framework.

***

# **A Search and Matching Model with a Continuous Logit Choice for Remote Work**

This document details the structure of a discrete-time search and matching model designed to analyze a labor market with heterogeneity in worker skills and firm technology. The model's central feature is the incorporation of idiosyncratic worker preferences for remote work, which are modeled within a continuous logit (or "random utility") framework. This approach moves from a deterministic choice of the remote work share, `α`, to a probabilistic one, where the model predicts a full probability distribution over work arrangements for each worker-firm match.

## **1. Economic Environment**

### **1.1 Agents and Distributions**

The economy is populated by a continuum of infinitely lived, risk-neutral workers and a continuum of firms.

*   **Workers:** Workers are indexed by their skill level, $h \in \mathcal{H}$, which is a primary determinant of their productivity. Skills are exogenously distributed according to a cumulative distribution function $F_h(h)$.
*   **Firms:** Firms are indexed by their remote work efficiency, $\psi \in \Psi$. This parameter captures a firm's technological and organizational capacity to support productive remote work. Firm types are distributed according to $F_\psi(\psi)$.

### **1.2 Preferences, Technology, and Choice**

**The Random Utility Framework for Work Arrangements**
The choice of the remote work share, $\alpha \in$, is modeled as a random utility maximization problem. The total value (or utility) of choosing a specific `α` for a given match `(h, ψ)` is:
$$ U(\alpha) = V(h, \psi; \alpha) + \mu \cdot \varepsilon(\alpha) $$
where:
-   $V(h, \psi; \alpha)$ is the **deterministic component** of the match's flow value, common to all workers in an `(h, ψ)` match.
-   $\varepsilon(\alpha)$ is a **stochastic process** representing a worker's idiosyncratic taste for different work arrangements.
-   $\mu > 0$ is a **scale parameter** that governs the variance of the taste shocks. A large `μ` implies high taste heterogeneity, while as $\mu \to 0$, the choice becomes deterministic.

**Deterministic Value Component**
The deterministic part of the flow value is the sum of output net of a baseline in-office cost:
$$ V(h, \psi; \alpha) = Y(\alpha \mid h, \psi) - c(1-\alpha) $$
-   **Production Function:** Output is a linear combination of in-person and remote productivity:
    $$ Y(\alpha \mid h, \psi) = A_1 h \cdot \left((1 - \alpha) + \alpha \cdot g(h, \psi)\right) $$
    where $g(h, \psi) = \psi_0 \exp(\nu\psi + \phi h)$ is the relative productivity of remote work.
-   **Baseline In-Office Cost:** The common component of the disutility from in-office work is:
    $$ c(1-\alpha) = c_0 \frac{(1-\alpha)^{1+\chi}}{1+\chi} $$
    where $c_0$ is a scale parameter and $\chi$ governs the curvature.

**Stochastic Taste Shock Process**
We assume that $\varepsilon(\alpha)$ is a **Type I Extreme Value (Gumbel) process**, where for any $\alpha_1 \neq \alpha_2$, the shocks $\varepsilon(\alpha_1)$ and $\varepsilon(\alpha_2)$ are independent draws from a standard Gumbel distribution. This assumption, analogous to the IID assumption in discrete choice models, leads to the continuous logit framework.

### **1.3 Search and Matching**

The labor market is characterized by standard search frictions operating in discrete time.
-   Let $L$ be the mass of unemployed workers and $V$ be the mass of vacancies.
-   The number of meetings is $M(L, V) = \gamma_0 L^{\gamma_1} V^{1-\gamma_1}$.
-   Labor market tightness is $\theta = V/L$.
-   The job-finding rate for a worker is $p(\theta) = \gamma_0 \theta^{1-\gamma_1}$.
-   The vacancy-filling rate for a firm is $q(\theta) = \gamma_0 \theta^{-\gamma_1}$.

## **2. Value Functions and Equilibrium**

The key implication of the random utility framework is that the value functions $U(h)$ and $S(h, \psi)$ do not depend on the idiosyncratic shocks, as they are integrated out. The heterogeneity is captured in the definition of the flow surplus.

### **2.1 The Flow Match Surplus: The "Inclusive Value"**

The flow surplus of a match `(h, ψ)` is the expected maximized value of the match, where the expectation is taken over all possible realizations of the taste shock process $\varepsilon(\alpha)$. This expected maximum is known as the "inclusive value." As derived from the properties of the Gumbel process, this takes the form of a "log-sum" integral:
$$ \mathbb{E}[\max_{\alpha} \{V(h, \psi; \alpha) + \mu \varepsilon(\alpha)\}] = \mu \ln \left( \int_0^1 \exp\left(\frac{V(h, \psi; \alpha)}{\mu}\right) d\alpha \right) $$
*(Note: A constant term, the Euler-Mascheroni constant, is dropped as it does not affect choices).*

The **flow surplus**, $s(h, \psi)$, is this inclusive value net of the worker's flow unemployment benefit, $b(h)$:
$$ s(h, \psi) = \left[ \mu \ln \left( \int_0^1 \exp\left(\frac{Y(\alpha \mid h, \psi) - c(1-\alpha)}{\mu}\right) d\alpha \right) \right] - b(h) $$

### **2.2 The Value of Unemployment**

An unemployed worker of type $h$ receives a flow benefit $b(h)$ in the current period. At the end of the period, they search for a job. With probability $p(\theta)$, they meet a firm, and with probability $1-p(\theta)$, they remain unemployed.

If a meeting occurs with a firm of type $\psi$, the worker and firm bargain over the match surplus, $S(h, \psi)$. The worker's lifetime value from accepting this match is $W(h, \psi)$. We assume this value is determined by Nash bargaining, where the worker receives their reservation value (the value of unemployment, $U(h)$) plus a share $\xi$ of the total match surplus:
$$ W(h, \psi) = U(h) + \xi S(h, \psi) $$
A worker will only accept a job offer if it provides more value than remaining unemployed, i.e., if $W(h, \psi) > U(h)$, which is equivalent to the condition $S(h, \psi) > 0$.

The Bellman equation for an unemployed worker can therefore be written as:
$$ U(h) = b(h) + \beta \left[ (1 - p(\theta))U(h) + p(\theta) \int_{\Psi} \max\{W(h,\psi), U(h)\} d\Gamma_{v}(\psi) \right] $$
where $\beta$ is the discount factor and $\Gamma_v(\psi) = v(\psi)/V$ is the distribution of vacancies.

Substituting the expression for $W(h, \psi)$ from the bargaining rule into the Bellman equation gives:
$$ U(h) = b(h) + \beta \left[ (1 - p(\theta))U(h) + p(\theta) \int_{\Psi} \max\{U(h) + \xi S(h,\psi), U(h)\} d\Gamma_{v}(\psi) \right] $$
The `max` operator simplifies, as the worker's gain from a match is $\xi S(h, \psi)$ if the surplus is positive, and zero otherwise. This can be written as $\xi S(h, \psi)^+$, where $x^+ \equiv \max\{x, 0\}$.
$$ U(h) = b(h) + \beta \left[ (1 - p(\theta))U(h) + p(\theta) \int_{\Psi} (U(h) + \xi S(h, \psi)^+) d\Gamma_{v}(\psi) \right] $$
Distributing the terms inside the integral:
$$ U(h) = b(h) + \beta \left[ (1 - p(\theta))U(h) + p(\theta)U(h) + p(\theta) \xi \int_{\Psi} S(h, \psi)^+ d\Gamma_{v}(\psi) \right] $$
The terms involving $p(\theta)U(h)$ cancel, leading to a cleaner expression:
$$ U(h) = b(h) + \beta \left[ U(h) + p(\theta) \xi \int_{\Psi} S(h, \psi)^+ d\Gamma_{v}(\psi) \right] $$
Finally, we can solve this equation for $U(h)$ to obtain its closed-form solution in terms of the expected surplus from search:
$$ U(h) = \frac{b(h) + \beta p(\theta) \xi \int S(h, \psi)^+ d\Gamma_{v}(\psi)}{1-\beta} $$

### **2.3 The Value and Surplus of a Match**

**Value of a Match**
Let $J(h, \psi)$ be the total present value of a match between a worker of type $h$ and a firm of type $\psi$. In the current period, the match generates a joint flow value. At the end of the period, the match survives with probability $1-\delta$ and is exogenously destroyed with probability $\delta$. If destroyed, the firm receives a value of zero, and the worker enters unemployment, receiving value $U(h)$.

The total flow value generated by the match is the "inclusive value" derived from the random utility framework. This is precisely the flow surplus, $s(h, \psi)$, plus the worker's flow benefit from unemployment, $b(h)$.
$$ \text{Instant Joint Flow Value} = \mathbb{E}[\max_{\alpha} \{V(h, \psi; \alpha) + \mu \varepsilon(\alpha)\}] = s(h, \psi) + b(h) $$
The Bellman equation for the value of a match is therefore:
$$ J(h, \psi) = \left( s(h, \psi) + b(h) \right) + \beta \left[ (1 - \delta) J(h, \psi) + \delta U(h) \right] $$

**Total Surplus**
The total match surplus, $S(h, \psi)$, is defined as the difference between the value of the match and the value of unemployment, $S(h, \psi) = J(h, \psi) - U(h)$. We can derive its recursive expression by manipulating the Bellman equations for $J(h, \psi)$ and $U(h)$.

Starting with the definition of surplus:
$$
\begin{align*}
S(h, \psi) &= J(h, \psi) - U(h) \\
&= \left( s(h, \psi) + b(h) + \beta \left[ (1 - \delta) J(h, \psi) + \delta U(h) \right] \right) - \left( b(h) + \beta \left[ U(h) + p(\theta) \xi \int S(h, \psi')^+ d\Gamma_{v}(\psi') \right] \right) \\
&= s(h, \psi) + \beta \left[ (1 - \delta) J(h, \psi) + \delta U(h) - U(h) - p(\theta) \xi \int S(h, \psi')^+ d\Gamma_{v}(\psi') \right] \\
&= s(h, \psi) + \beta \left[ (1 - \delta) J(h, \psi) - (1 - \delta) U(h) - p(\theta) \xi \int S(h, \psi')^+ d\Gamma_{v}(\psi') \right]
\end{align*}
$$
Now, we can factor out $(1-\delta)$ from the first two terms inside the brackets:
$$
\begin{align*}
S(h, \psi) &= s(h, \psi) + \beta \left[ (1 - \delta) (J(h, \psi) - U(h)) - p(\theta) \xi \int S(h, \psi')^+ d\Gamma_{v}(\psi') \right] \\
&= s(h, \psi) + \beta (1 - \delta) S(h, \psi) - \beta p(\theta) \xi \int S(h, \psi')^+ d\Gamma_{v}(\psi')
\end{align*}
$$
This final expression is the recursive equation for the total surplus. We can solve for $S(h, \psi)$ to obtain the central equation of the model, which is used for the numerical solution:
$$ S(h, \psi) = \frac{s(h, \psi) - \beta p(\theta) \xi \int S(h, \psi')^+ d\Gamma_{v}(\psi')}{1 - \beta(1 - \delta)} $$
This equation shows that the total surplus of a match is the present discounted value of its flow surplus, $s(h, \psi)$, net of the worker's expected future gains from search.

## **3. Equilibrium Outcomes**

### **3.1 The Distribution of Work Arrangements**

The model no longer predicts a single optimal `α*` for a match. Instead, it yields a full probability density function for `α` for each `(h, ψ)` pair. This is the continuous logit choice probability:
$$ p(\alpha \mid h, \psi) = \frac{\exp\left(\frac{V(h, \psi; \alpha)}{\mu}\right)}{\int_0^1 \exp\left(\frac{V(h, \psi; a)}{\mu}\right) da} = \frac{\exp\left(\frac{Y(\alpha) - c(1-\alpha)}{\mu}\right)}{\int_0^1 \exp\left(\frac{Y(a) - c(1-a)}{\mu}\right) da} $$
The aggregate distribution of work arrangements in the economy is the integral of these conditional distributions, weighted by the steady-state distribution of employed workers, $n(h, \psi)$.

### **3.2 Wage Determination**

The wage is a pure transfer that divides the realized flow value of the match. The wage paid to a worker in a match `(h, ψ)` who has chosen a specific arrangement `α` is:
$$ w(h, \psi; \alpha) = \underbrace{\text{Base Wage}(h, \psi)}_{\text{From surplus split}} + \underbrace{c(1-\alpha)}_{\text{Compensating Differential}} $$
The **Base Wage** component is determined by the surplus sharing rule and is common to all workers in an `(h, ψ)` match. It delivers the worker's share of the *expected* flow value. The **Compensating Differential** is an ex-post adjustment that exactly compensates the worker for the baseline disutility of their chosen `α`.

### **3.3 Firm's Problem and Free Entry**

A firm of type $\psi$ chooses the number of vacancies $v$ to post to maximize its expected profit. The profit function is the expected benefit minus the total cost:
$$ \Pi(v(\psi)) = q(\theta) (1-\xi)S(h, \psi)^+ v - \frac{\kappa_0 v^{1 + \kappa_1}}{1 + \kappa_1} $$
The firm's first-order condition (FOC) for profit maximization is $\Pi'(v) = 0$, which implies that the marginal benefit must equal the marginal cost:
$$ q(\theta) (1-\xi)S(h, \psi)^+ = \kappa_{0} v(\psi)^{\kappa_{1}} $$
This **free-entry condition** pins down the number of vacancies posted by each firm type and, in aggregate, closes the model by determining the equilibrium labor market tightness $\theta$.

## **4. Steady-State Equilibrium Definition**

A steady-state equilibrium is a set of value functions $U(h)$, $S(h, \psi)$; a conditional choice probability density $p(\alpha \mid h, \psi)$; vacancy postings $v(\psi)$; distributions of unemployed and employed workers $u(h)$, $n(h, \psi)$; and an aggregate market tightness $\theta$ such that:
1.  The value functions solve their respective Bellman equations, with the flow surplus defined by the inclusive value integral.
2.  The choice probabilities are consistent with the continuous logit formula.
3.  The free-entry condition holds for all firm types.
4.  The distributions of workers are stationary, meaning inflows equal outflows for each state.
5.  The labor market is in flow balance.