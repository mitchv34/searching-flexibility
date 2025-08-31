## **Data Appendix**

-   [ ] **C.1. Data Sources and Sample Construction**
    -   [ ] List all datasets used with sources and links (IPUMS CPS, SWAA, JOLTS, FRED, O*NET, etc.).
    -   [ ] Provide a detailed table of the sample selection criteria (age, employment status, hours worked, wage floors, etc.) and the number of observations dropped at each stage.
-   [ ] **C.2. Variable Construction**
    -   [ ] Detail the construction of key variables like real hourly wage, education categories, etc.
-   [ ] **C.3. Construction of the Teleworkability Index ($\psi$)**
    -   [ ] **This is a key methodological contribution and needs significant detail.**
    -   [ ] **Feature Set:** List or summarize the O*NET variables used as features.
    -   [ ] **Training Data:** Describe the Occupational Requirements Survey (ORS) data used for labels.
    -   [ ] **Model Specification:** Detail the two-stage Random Forest model (Stage 1: Classifier for zero vs. non-zero; Stage 2: Regressor for the intensive margin).
    -   [ ] **Validation:** Report the key performance metrics from your validation set (e.g., Accuracy, F1-score for the classifier; MSE, Correlation for the regressor).
    -   [ ] **Final Output:** Show the final distribution of the predicted $\psi$ index.