

### ## The Logic of a Contingent Contract

The standard interpretation in this class of models is that the firm and worker do not bargain over a single wage number before the match begins. Instead, they bargain over a **contract** or **sharing rule** that specifies a wage payment for *any possible outcome*.

Here's the timing:
1.  **Bargaining (Ex-Ante)**: When the worker and firm meet, they agree on a contract. This contract promises the worker a certain lifetime value, $W(h, \psi) = U(h) + \xi S(h, \psi)$, based on the *expected* surplus of the match.
2.  **Realization (Ex-Post)**: In each period of the match, the worker's idiosyncratic taste shocks, $\{\varepsilon(\alpha)\}_{\alpha \in [0,1]}$, are realized and become known to both the worker and the firm.
3.  **Efficiency (Ex-Post)**: Given the realized shocks, they jointly choose the specific `\alpha` that maximizes the *realized* joint surplus, $J(\alpha; \varepsilon)$.
4.  **Payment (Ex-Post)**: The firm pays the worker a wage, $w(\alpha)$, that is contingent on the chosen `\alpha`, as specified by the contract they agreed to in step 1.

### ## Why the Ex-Post Differential is Correct

The wage equation you have reflects this contingent contract perfectly. It's best thought of as a wage schedule, $w(\alpha)$:
$$w(\alpha) = \text{Base Wage} + c(1-\alpha)$$
* The **Base Wage** is fixed from the start. It's the part of the payment that ensures the worker receives their share of the *expected* surplus.
* The **Compensating Differential**, $c(1-\alpha)$, is the variable part of the contract. It is an ex-post transfer that perfectly compensates the worker for the *realized* disutility of the chosen `\alpha`.

This structure is what makes the contract efficient. By guaranteeing the worker will be made whole for any in-office disutility, the contract ensures the worker is willing to agree to whichever `\alpha` maximizes the total joint surplus. It elegantly separates the efficiency decision (choosing `\alpha`) from the worker's personal cost.

