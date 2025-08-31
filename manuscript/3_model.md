# A Model of Labor Search with Heterogeneous Preferences for Workplace Flexibility

\textcolor{green}{Statement of purpose...}

We consider a labor market populated by a continuum of firms and infinitely lived, risk-neutral workers. This framework abstracts from life-cycle and precautionary savings motives to focus solely on the trade-offs inherent in job search and workplace arrangements. Heterogeneity is central to our analysis: workers differ in their skill level, $h \in \mathcal{H}$, which determines their baseline productivity, while firms differ in their remote work efficiency, $\psi \in \Psi$. This parameter primarily reflects occupation-level characteristics that determine a job's suitability for remote work, but it also encompasses firm-specific factors like technological infrastructure and organizational capacity. 

The measure of unemployed workers of skill level $h$ is denoted $u(h)$, creating an aggregate stock of job seekers $L=\int_{\mathcal{H}}u(h)dh$. Similarly, the measure of vacancies posted for type-$\psi$ jobs is $v(\psi)$, which aggregates to a total of $V=\int v(\psi)d\psi$ job opportunities. This supply of vacancies is determined by a free-entry condition, where firms can create and post jobs a cost $\kappa(v)$, doing so until the expected value of filling a position equals the cost of posting. Looking for a job is a time consuming effort, something we characterized by assuming search and matching frictions. A standard constant-returns-to-scale matching function, $M(L,V)$, governs the meeting process between unemployed workers and vacancies. This process determines the key outcomes of the search process: the job-finding rate for workers, $p(\theta)$, and the vacancy-filling rate for firms, $q(\theta)$, both of which depend on the aggregate labor market tightness, $\theta = V/L$.


**Worker preferences** over wage and remote work bundles $\{(w, \alpha)\}_{\mathbb{R}_{+}\times[0,1]}$ continuously differentiable and concave to ensure well-behaved optimization. We make two standard assumptions: utility is strictly increasing in wages ($u_w > 0$), as workers prefer more consumption, and it is also increasing in the remote work share ($u_\alpha > 0$), reflecting the direct value workers place on the flexibility and amenities associated with working from home.

A key assumption governs the trade-off between these two goods: we assume the Marginal Rate of Substitution remote work and wages: $MRS_{\alpha,w} = u_\alpha / u_w$, is *increasing* in $\alpha$. In economic terms, this means that as a worker's remote share increases, they require progressively larger wage compensation to give up an additional unit of remote work. This non-standard assumption is designed to capture real-world phenomena that can lead to a preference for corner solutions (either fully in-person or fully remote). For example, it can reflect significant lifestyle adjustments that make a fully remote setup particularly valuable once established, or the presence of high fixed costs (e.g., commuting, separate childcare arrangements) associated with even a single day of in-office presence, which can make hybrid schedules less desirable.\textcolor{green}{THERE IS DATA IN SWAA TO BACK THIS UP SHOULD HAVE A FOOTNOTE HERE.}

**The Production technology** $Y$, depends on the remote work share $\alpha$, the worker's skill $h$, and the firm/occupation's remote conduciveness $\psi$. The production function takes the form of a linear combination of output from in-person and remote work:
$$Y(\alpha \mid \psi, h) = A(h) \cdot \left((1 - \alpha) + \alpha \cdot g(h, \psi)\right)$${#eq-prod-fun} 
Here, $A(h)=A_0 + A_1 h$ is the baseline output of a worker with skill $h$ in a fully in-person setting, with $A'(h) > 0$. Notice that the only source of heterogeneity on the firm side is the parameter $\psi$, which captures the firm's remote work efficiency, therefore for full-in person arrangements $(\alpha = 0)$, the sole determinant of output is worker skill.

Remote work productivity is scaled by an efficiency adjustment factor, $g(h, \psi)$, which we specify with a flexible functional form:
$$g(h, \psi) = \psi_0 \cdot h^\phi \cdot \psi^\nu$$ {#eq-remote-prod}
n this specification, $\psi_0$​ is a baseline technology parameter for remote productivity across the economy, while $\phi$ and $\nu$ are the output elasticities with respect to worker skill and firm remote work efficiency respectively. We assume $\psi>0$ and $\nu>0$. The assumption that remote productivity increases with worker skill ($\phi>0$) is motivated by the idea that higher-skilled workers often engage in tasks requiring greater autonomy and self-direction—traits that are highly complementary to the remote work environment. Furthermore, their work may be less reliant on physical co-location for supervision and execution compared to more routine tasks.

Crucially, this functional form implies a positive cross-partial derivative $(g_{h\psi}​>0)$, meaning there is a **complementarity** between worker skill and firm remote efficiency. This complementarity is the central force that will drive assortative matching in the model, creating a tendency for high-skill workers to sort into firms that are most efficient at remote work.

## Deterministic Choice of Flexibility: A Benchmark

To build intuition for the core trade-offs governing workplace flexibility, we first analyze a simplified benchmark model where the choice of work arrangement is **fully deterministic**. This approach allows us to isolate the fundamental economic forces driven by technology and preferences before we introduce additional sources of heterogeneity.

For any given work arrangement, $\alpha$, the total flow surplus of a match is the sum of the firm's profit and the worker's utility. The firm's per-period profit is its output net of the wage paid, $\Pi(h, \psi \mid w, \alpha)=Y(h, \psi\mid \alpha)−w$. The worker's utility, given our quasi-linear specification, is their wage net of the non-pecuniary cost of in-office work, $u(w, \alpha)=w−c(1−\alpha)$. The joint surplus, $J(\alpha)$, is therefore:
$$\pi(h, \psi \mid \alpha) = \Pi(h, \psi \mid w, \alpha) +  u(w, \alpha )= \Big(Y(h, \psi \mid \alpha) - w\Big) + \Big(w - c(1-\alpha)\Big)$$ {#eq-surplus-alpha}
As the wage, $w$, is a pure intra-match transfer, it cancels out. This leaves the joint surplus dependent only on the match's total output and the worker's non-pecuniary cost.

We assume the wage is determined by generalized Nash bargaining between the firm and the worker. This bargaining framework implies that the match operates under an **efficient contract**, which separates the problem into two parts: an efficiency decision and a distributional decision. First, the remote work share, $\alpha$ is chosen jointly to maximize the total surplus generated by the match. Second, this maximized surplus is divided between the worker and the firm according to their exogenous bargaining power $\xi$. With our choice of quasi-linear utility, the wage acts as the endogenous transfer that facilitates this division. This efficient contracting structure allows us to solve for the optimal work arrangement by first focusing on the joint surplus maximization given by equation @eq-surplus-alpha:
$$\pi = \max_{\alpha \in [0,1]} \quad \Big\{ Y(\alpha \mid \psi, h) - c(\alpha)\Big\}$$ {#eq-surplus-max}

The solution to problem @eq-surplus-max reveals that the optimal work arrangement, $\alpha^∗(\psi,h)$, partitions the market into three distinct regimes based on the firm's remote efficiency $\psi$, , relative to two skill-dependent thresholds, $\underline{\psi}(h)$ and $\overline{\psi}(h)$:
$$\alpha^{*}(\psi,h) = \begin{cases}
    0 & \text{if } \psi \leq \underline{\psi}(h) \quad \text{(Full In-Person)} \\ 
    1 - \left[ \frac{A_1h(1 - g(\psi,h))}{c_0} \right]^{\frac{1}{\chi}} & \text{if } \underline{\psi}(h) < \psi < \overline{\psi}(h) \quad \text{(Hybrid)} \\
    1 & \text{if } \psi \geq \overline{\psi}(h) \quad \text{(Full Remote)}
\end{cases}$$ {#eq-optimal-alpha-deterministic}

These thresholds represent economic tipping points where the trade-off between **production efficiency and worker amenities** dictates the optimal work arrangement. The worker always values the non-pecuniary benefits of remote work, while the firm is focused on the impact on output, which may be positive or negative.

- For firms with low remote efficiency ($\psi\leq\underline{\psi}​(h)$), the **productivity loss** from remote work is too severe. Although the worker desires the remote work amenity, the firm cannot afford to grant this preference because the marginal drop in output is greater than the worker's marginal valuation for it. Thus, the match defaults to a fully in-person arrangement.
- Conversely, for firms with very high efficiency ($\psi\geq\overline{\psi}(h)$), remote work may be so productive that it generates a **"productivity premium."** In this scenario, maximizing output and satisfying the worker's desire for remote work are aligned, making a fully remote arrangement the optimal choice for the match.
- **Hybrid work** emerges for the intermediate firms where a clear trade-off exists. These firms are willing to "sell" the remote work amenity to the worker, accepting a modest productivity loss (or smaller gain) up to the point where the marginal cost in terms of output exactly equals the worker's marginal non-pecuniary benefit.
  
![Optimal Work Arrangements in the Deterministic Benchmark. The figure illustrates how the optimal remote work share, $\alpha^*$, depends on worker skill ($h$) and firm remote efficiency ($\psi$). The market is partitioned into three regimes by two skill-dependent thresholds, $\underline{\psi}(h)$ and $\overline{\psi}(h)$. Matches falling below the lower threshold are fully in-person, matches above the upper threshold are fully remote, and matches between the thresholds adopt a hybrid arrangement.](../figures/stock_image.png){#fig-deterministic-thresholds fig-align="center" width="80%"}

This deterministic model provides sharp predictions: for any given worker, only firms within a specific range of remote efficiency will offer a hybrid arrangement. However, this stark segmentation is at odds with the smooth distribution of work arrangements observed in the data. This motivates the introduction of idiosyncratic preferences, which allows for a richer and more realistic pattern of matching.

## The Full Model with Idiosyncratic Preferences

The deterministic benchmark provides sharp predictions but cannot account for the smooth distribution of work arrangements observed in the data. To capture this rich heterogeneity, we extend the model by introducing idiosyncratic worker preferences. We posit that a worker only discovers their true preference for a specific work arrangement after a match is formed. Factors such as the actual commute, the specific in-office culture, or the suitability of their home environment for work are not fully known ex-ante. This uncertainty is captured by an idiosyncratic taste shock.

For a given contract ( $w, \alpha$ ), a worker's realized utility is the sum of their deterministic utility and this stochastic taste shock:
$$u(w, \alpha ; \varepsilon)=w-c(1-\alpha)+\mu \cdot \varepsilon(\alpha)$$ {#eq-utility-random}
Here, $\varepsilon(\alpha)$ is the realization of the taste shock for a remote share $\alpha$, and $\mu$ is a scale parameter governing its importance. Analogous to the deterministic model, the total **realized joint surplus** for a given arrangement and taste shock, is the sum of the firm's profit and the worker's realized utility. The wage remains a pure transfer and cancels out, leaving the surplus dependent only on the physical aspects of the match and the worker's non-pecuniary utility. Following the discrete choice literature, we assume $\varepsilon(\alpha)$ follows a Type I Extreme Value (Gumbel) process.

Before the idiosyncratic taste shocks are realized, the firm and worker jointly maximize the expected value of their match. This value, known as the "inclusive value," accounts for the option to choose the best possible work arrangement once the shocks are revealed. The ex-ante joint maximization problem is:

$$\pi(h, \psi)  = \max_{\alpha \in [0,1]} \mathbb{E}_{\varepsilon} \left[ \underbrace {Y(\alpha \mid \psi, h) - c(\alpha)}_{V(h, \psi \mid \alpha)} + \mu \cdot \varepsilon(\alpha) \right]$$ {#eq-flow-value-expected}

Where $V(h, \psi; \alpha)$ denote the deterministic component of the joint value of the match. A well-known result from the discrete choice literature is that when the shocks $\varepsilon(\alpha)$ are drawn from a Type I Extreme Value (Gumbel) distribution, this maximization problem has a convenient closed-form solution. The maximized expected surplus of the match is given by the log-sum integral:

$$\pi(h, \psi) = \mu \ln \left( \int_0^1 \exp\left(\frac{V(h, \psi; \alpha)}{\mu}\right) d\alpha \right)$$ {#eq-flow-value-solved}

This inclusive value is the fundamental object that enters the equilibrium Bellman equations that we define next. We define the flow surplus of the match, $s(h,\psi)$, as this value net of the worker's outside option, their flow unemployment benefit $b(h)$:

$$ s(h, \psi) = \left[ \mu \ln \left( \int_0^1 \exp\left(\frac{V(h, \psi; \alpha)}{\mu}\right) d\alpha \right) \right] - b(h)$$ {#eq-flow-surplus}

This expected flow surplus, $s(h,\psi)$, is the fundamental object that determines the value of a match in the full equilibrium.

A direct consequence of this framework is that the choice of $\alpha$ becomes probabilistic. The probability that a specific arrangement $\alpha$ is chosen for a match $(h, \psi)$ follows the continuous logit formula:

$$ p(\alpha \mid h, \psi) = \frac{\exp\left({V(h, \psi; \alpha)}{\mu^{-1}}\right)}{\int_0^1 \exp\left({V(h, \psi; a)}{\mu^{-1}}\right) da} $$ {#eq-prob-alpha}

This structure provides the  link between the two versions of the model: the hard thresholds of the deterministic benchmark now become the central inflection points of the smooth choice probabilities in the full model. The scale parameter, $\mu$, governs how "blurry" or "soft" these thresholds are.

## Equilibrium

A steady-state equilibrium is characterized by a set of value functions for workers and firms, optimal vacancy posting by firms, and worker flows that are balanced. These components are mutually consistent and determine the aggregate state of the labor market.
### Value Functions and Surplus

The equilibrium is defined by the lifetime values for workers and firms in different states. We begin by defining the value of an ongoing match and the value of unemployment, and from these, we derive the Bellman equation for the match surplus, $S(h,\psi)$, which will be shown to be the central object that characterizes the equilibrium.

The joint value of a match, $J(h, \psi)$, is the present discounted value of all future returns. It is composed of the current period's expected flow surplus, $s(h,\psi)+b(h)$, plus the discounted continuation value. With probability $(1-\delta)$, the match survives and retains its value, and with probability $\delta$, it is destroyed and the worker's value reverts to that of unemployment, $U(h)$^[We assume that free entry make the value for a firm of a dicontinued match equal to zero]. This gives the Bellman equation:
$$ J(h, \psi) = \left( \pi(h,\psi) - b(h) \right) + \beta \left[ (1-\delta)J(h,\psi) + \delta U(h) \right] $$ {#eq-value-match}
The value of unemployment for a worker of type $h$, $U(h)$, consists of the current flow benefit, $b(h)$, plus the expected value from job search. With probability $p(\theta)$, the worker contacts a firm, and with probability $(1-p(\theta))$, they remain unemployed. The value of a new match to a worker, $W(h,\psi')$, is determined by our **Nash bargaining** assumption, which dictates that the worker receives their outside option, $U(h)$, plus a share $\xi$ of the total match surplus, $S(h, \psi')$. A match is only formed if the surplus is positive. This gives the Bellman equation for unemployment:
$$ U(h) = b(h) + \beta \left[ p(\theta)\mathbb{E}_{\psi'}\left[\max\{W(h,\psi'), U(h)\}\right] + (1-p(\theta))U(h) \right] $$ {#eq-value-unemployment}

Substituting $W(h, \psi') = U(h) + \xi S(h, \psi')$ and simplifying yields:
$$ U(h) = b(h) + \beta U(h) + \beta p(\theta)\xi \int \max\{0, S(h,\psi')\} d\Gamma_v(\psi') $$ {#eq-value-unemployment-simplified}
The total match surplus is the net value created by the match, defined as $S(h, \psi) \equiv J(h, \psi) - U(h)$. We can derive its Bellman equation by subtracting the equation for $U(h)$ from the one for $J(h, \psi)$, which after rearranging highlights that the worker's expected gain from search acts as an effective opportunity cost for the match:
$$ S(h, \psi) = s(h, \psi) + \beta(1-\delta)S(h,\psi) - \beta p(\theta)\xi \int \max\{0, S(h,\psi')\} d\Gamma_v(\psi') $$ {#eq-value-surplus}
Solving for $S(h, \psi)$ gives the final expression:
$$ S(h, \psi) = \frac{s(h, \psi) - \beta p(\theta) \xi \int \max\{0, S(h,\psi')\} d\Gamma_{v}(\psi')}{1 - \beta(1 - \delta)} $$ {#eq-value-surplus-solved}

This derivation highlights a key insight: the surplus equation is the only object needed to solve for the equilibrium. Because the term $\max\{0, S(h,\psi')\}$ appears inside the integral, the sign of the surplus itself determines the set of viable matches that will form. A negative surplus means the parties are better off separated, and no match is created. Therefore, $S(h,\psi)$ is a sufficient statistic that fully encodes the equilibrium matching decisions.

### Vacancy Creation and Market Tightness

The number of vacancies is determined by firms' profit maximization. Firms post vacancies for a type-$\psi$ job until the marginal cost of posting, $\kappa’(v)$, equals the expected marginal benefit. The benefit of posting depends on the probability of filling the vacancy, $q(\theta)$, and the firm's expected share of the surplus, $(1−\xi)S(h,\psi)$, averaged over the distribution of workers it might meet. This gives the vacancy creation condition:
$$c′(v(\psi))=q(\theta)(1−\xi)\int_{\mathcal{h}} \max\{0, S(h,\psi) \} \frac{u(h)​}{L}dh$$ {#eq-free-entry}
This set of decisions, aggregated across all firm types, endogenously determines the total stock of vacancies $V$ and thus the equilibrium market tightness $\theta$.

### Steady-State Flows

In a steady-state equilibrium, the flows of workers between employment and unemployment are balanced. The total number of workers who lose their jobs *(job destruction)* must equal the total number of unemployed workers who find new, acceptable jobs *(job creation)*. This condition, $\delta \cdot N_{\text{emp}}​=p(\theta)\cdot N_{\text{unemp}}​\cdot \mathbb{P}(\text{Accept})$, where $N$ denotes the mass of workers, closes the model by determining the equilibrium distributions of employed and unemployed workers for each skill type, $n(h,\psi)$ and $u(h)$.

## Wage Determination

With the expected surplus of the match, $s(h,\psi)$, determined, the wage serves as the transfer that divides the realized proceeds. It is best understood not as a single number but as a **contingent contract** agreed upon at the start of the match. In each period, after the worker's idiosyncratic taste shock $\varepsilon(\alpha)$ is realized, the specific remote work share $\alpha^*$ is chosen to maximize the realized joint surplus. The wage, $w(\alpha^*)$, is then paid according to the pre-agreed contract to ensure the division of surplus aligns with the parties' bargaining powers.

To derive the wage, we start with the Bellman equation for the worker's value in an ongoing match, $W(h, \psi)$. This value is the sum of the current period's flow utility and the discounted continuation value, accounting for the probabilities of the match surviving ($1-\delta$) or being destroyed ($\delta$):

$$ W(h, \psi) = \left( w - c(1-\alpha^*) \right) + \beta(1-\delta)W(h,\psi) + \beta\delta U(h) $$ {#eq-value-employed}

Rearranging this asset-pricing equation, we can solve for the flow utility that the wage must generate in each period to support the lifetime value $W(h, \psi)$:

$$ w - c(1-\alpha^*) = (1 - \beta(1-\delta))W(h,\psi) - \beta\delta U(h) $$ {#eq-wage-equation}

From our Nash bargaining assumption, we know the worker's equilibrium lifetime value is their outside option plus their bargained share of the total match surplus: $W(h, \psi) = U(h) + \xi S(h, \psi)$. To find the equilibrium wage, $w^*(h, \psi)$, we substitute this bargained value into the expression for the required flow utility. The quasi-linearity of preferences allows us to then simply solve for the optimal wage. This final expression clearly separates the wage into two economically distinct components:

$$ w^*(h, \psi) = \underbrace{\left( (1 - \beta(1-\delta))\left[ U(h) + \xi S(h, \psi) \right] - \beta\delta U(h) \right)}_{\text{Base Wage}} + \underbrace{c_0 \frac{(1-\alpha^*(h, \psi))^{1+\chi}}{1+\chi}}_{\text{Compensating Differential}} $$ {#eq-wage-solution}

1.  **Base Wage**: The monetary payment required to deliver the worker their bargained share of the match surplus, net of continuation values.
2.  **Compensating Differential**: An additional, separate payment that exactly reimburses the worker for the non-pecuniary disutility associated with the fraction of time, $(1-\alpha^*)$, they are required to spend in the office. This term is zero for a fully remote worker.
