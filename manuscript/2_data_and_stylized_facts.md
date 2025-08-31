# Data and Stylized Facts of the Post-Pandemic Labor Market

## **Empirical Evidence: Three Puzzles Motivating a Structural Approach**

* [cite_start]**Narrative Introduction (Revised):** _(Applying the "Punchline First" principle [cite: 55, 231])_
    Start by stating the section's main conclusion directly. For example: "This section presents three key empirical puzzles of the post-pandemic labor market. First, the adoption of flexible work is highly stratified by education and occupation. Second, wages exhibit a complex, concave relationship with an occupation's potential for remote work, even after conditioning on rich observable characteristics. Third, we document powerful sorting between highly educated workers and remote-friendly occupations. We argue that these facts, taken together, are difficult to reconcile with a simple framework and point toward a deeper sorting mechanism based on latent worker skill. This evidence forms the empirical foundation that our structural model is designed to explain. Our analysis primarily relies on data from the Current Population Survey (CPS)."

---

### **Data, Sample Construction, and Limitations**

* [cite_start]**Goal:** Describe the data, sample selection, and transparently address any limitations to build credibility[cite: 209].
* **Content:**
    * **Data Source:** Describe the Current Population Survey (CPS), the supplements used (e.g., ASEC), and the sample years (e.g., 2019, 2022-2024).
    * [cite_start]**Justification and Limitations:** _(New subsection to boost credibility )_ Briefly explain *why* the CPS is the appropriate dataset for this analysis (e.g., large sample size, detailed demographics). Then, be transparent about its limitations (e.g., "We acknowledge that the CPS questions on remote work are self-reported and may contain measurement error. Furthermore, our wage measure does not include non-wage benefits, which may be an important margin of adjustment.").
    * **Sample Selection:** Detail the filters applied to the raw data (e.g., full-time workers, age 25-64, non-military, etc.).
    * [cite_start]**Table 1: Summary Statistics of the Analysis Sample (2024):** _(Title revised to be more descriptive [cite: 247])_
        * **Content:** Present weighted means and standard deviations for key variables (age, education, log wage, shares by sex/race).
        * [cite_start]**Notes:** _(Ensure the table is self-contained [cite: 243][cite_start])_ The notes should define all variables, state the data source (CPS ASEC), and specify the sample restrictions and application of survey weights, ensuring replicability[cite: 197].

---

### **Measuring Occupational Amenability to Remote Work ($\psi$)**

* **Goal:** Introduce the key explanatory variable ($\psi$) and document its strong correlation with other key worker and job characteristics, highlighting the primary empirical challenge.
* **Narrative:** Explain that a quantitative measure of an occupation's intrinsic suitability for remote work is needed.
* **Construction of $\psi$:**
    * **Methodology:** Briefly explain the construction methodology (e.g., following Dingel and Neiman (2020)).
    * **Interpretation:** State clearly that $\psi$ is a continuous [0, 1] index.
* **Figure 1: The Skewed Distribution of Remote-Work Potential:** _(Title revised for clarity)_
    * **Content:** A density plot of the $\psi$ index across occupations, weighted by employment.
    * **Takeaway:** Emphasize the concentration, e.g., "The potential for remote work is a scarce occupational feature. Over half of all workers are in occupations with $\psi < 0.2$."
* **Table 2: Worker and Job Characteristics by $\psi$ Quantile:**
    * **Content:** Show average education, wages, and industry composition for workers in Low-$\psi$, Mid-$\psi$, and High-$\psi$ occupations.
    * [cite_start]**Takeaway and Narrative Framing:** Frame this table as demonstrating the core challenge of confounding variables[cite: 221]. State clearly: "This table reveals the central empirical challenge that motivates our structural approach. High-$\psi$ occupations are not randomly assigned; they are disproportionately held by higher-educated workers in higher-paying industries. This demonstrates that simple OLS regressions of wages on remote work status would be severely biased by worker and firm selection."

---

### **The Three Puzzles**

* **Goal:** Present the key empirical patterns as distinct puzzles that the model must explain. [cite_start]This turns the section into a compelling narrative[cite: 359].

* **Puzzle 1: The Stratification of Work Arrangements.** 🔎
    * **Figure 2: The Post-Pandemic Rise of Hybrid and Remote Work:** Time-series plot (2019-2024) of In-Person, Hybrid, and Full-Remote work shares.
    * **Figure 3: Remote and Hybrid Work are Dominated by the Highly Educated in High-$\psi$ Jobs:** The two-panel bar chart.
    * **Narrative:** "The first puzzle is the sharp stratification of who performs flexible work. It is not a widespread phenomenon but one concentrated in a specific corner of the labor market."

* **Puzzle 2: The Concave Wage Profile.** 📈
    * **Figure 4: The Non-Linear Relationship Between Wages and Occupational Flexibility:** Binned scatter plot of residual log wage against your $\psi$ or $\alpha$ index.
    * **Table 3: OLS and Fixed-Effects Estimates of the Wage-$\psi$ Profile:** The key regression table showing $\psi$ (positive) and $\psi^2$ (negative) coefficients.
    * **Narrative:** "The second puzzle lies in the wage structure. After controlling for an extensive set of worker and job characteristics, we find a positive but concave relationship between wages and an occupation's remote-work potential. This suggests a premium that diminishes at higher levels, a pattern that simple theories of compensating differentials struggle to explain."

* **Puzzle 3: Powerful Sorting on Unobservables.** 🧩
    * **Figure 5: Strong Assortative Matching on Education:** Scatter plot showing the positive correlation between worker education and their occupation's average $\psi$.
    * **Table 4: Evidence of a Sorting Premium:** _(Revised framing)_ Your key regression table.
    * **Narrative Framing:** Be precise about what this regression shows. "The final puzzle points to the importance of unobserved factors. The strong positive sorting between worker education and occupational $\psi$ is striking. More importantly, even *within the subsample of workers in flexible jobs*, we find a significant wage premium associated with being in a high-$\psi$ occupation after controlling for education (Table 4). This is not a causal estimate of a 'return to $\psi$,' but rather evidence consistent with a sorting mechanism where higher-skilled workers (both observably and unobservably) sort into high-$\psi$ jobs. This is the final piece of evidence our model must rationalize." [cite_start]This framing shows intellectual honesty and clearly defines the purpose of the regression[cite: 223].

---

### **Synthesizing the Puzzles and Motivating the Structural Model**

* **Goal:** Summarize and make the case for your model.
* **Narrative:** Your original paragraph is excellent and already aligns perfectly with the guide's principles. It masterfully summarizes the puzzles and pivots to the solution. I would only suggest a slightly more active title for the subsection.
    > "The empirical evidence points to a complex new equilibrium. Remote work is concentrated among the highly educated in specific occupations and is associated with a positive, concave wage profile that persists after controlling for these observable characteristics. The strong assortative matching suggests the presence of a deeper, unobserved factor driving these patterns. To simultaneously rationalize the distribution of work arrangements, the complex wage structure, and the powerful sorting dynamics, we develop and estimate a structural search model. The model's central mechanism is the interaction between a latent, continuously distributed worker skill, $h$, and the observable occupational technology, $\psi$, which jointly determine productivity, wages, and the choice of workplace flexibility."