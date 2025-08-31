# **Counterfactuals**

-   [ ] **Introduction:** Briefly state the purpose of this section: to use the estimated structural model to perform experiments that are impossible with reduced-form methods, allowing us to decompose the key economic forces at play.

-   [ ] **6.1. Counterfactual 1: Decomposing the Drivers of the New Equilibrium**
    -   [ ] **State the Question:** What were the relative contributions of the preference shock versus the technology shock in shaping the 2024 labor market?
    -   [ ] **Experiment A (Preference Shock Only):**
        -   [ ] **Method:** Re-solve the model using the estimated **2024 preference parameters** ($c_0$, $\mu$, $\chi$) but holding all other parameters (especially technology: $\psi_0$, $\nu$, $\phi$) at their estimated **2019 levels**.
        -   [ ] **Report Key Outcomes:** Report the model-predicted shares of in-person/hybrid/remote work, average $\alpha$, and aggregate productivity.
    -   [ ] **Experiment B (Technology Shock Only):**
        -   [ ] **Method:** Re-solve the model using the estimated **2024 technology parameters** ($\psi_0$, $\nu$, $\phi$) but holding all other parameters (especially preferences: $c_0$, $\mu$, $\chi$) at their estimated **2019 levels**.
        -   [ ] **Report Key Outcomes:** Report the same set of outcomes as in Experiment A.
    -   [ ] **Present the Results:**
        -   [ ] **Create a summary table or bar chart.** The columns should be: "Actual 2019", "Actual 2024", "CF A: Pref. Shock Only", "CF B: Tech. Shock Only". The rows should be the key outcomes.
        -   [ ] **Write the narrative:** Explain what the results show. (e.g., "As shown in Figure X, the preference shock alone can explain approximately 85% of the observed shift in the share of remote work, while the technology shock accounts for only 15%...").

-   [ ] **6.2. Counterfactual 2: Quantifying the Importance of Sorting**
    -   [ ] **State the Question:** How important is the assortative matching of high-skill workers to high-efficiency firms in generating the observed outcomes?
    -   [ ] **Method:** Take the fully estimated **2024 model** and re-solve it after "turning off" the skill-remote complementarity by setting the sorting parameter **$\phi = 0$**.
    -   [ ] **Report Key Outcomes:**
        -   [ ] How much does the average remote share (`mean_alpha`) fall?
        -   [ ] What happens to the WFH wage premium? (Calculate the conditional wage difference in the simulation).
        -   [ ] Does overall wage inequality (`var_logwage`) change?
    -   [ ] **Write the narrative:** Explain the results. (e.g., "We find that eliminating the sorting channel reduces the share of remote work by X percentage points and completely flattens the WFH wage premium, highlighting that sorting is a crucial mechanism...").