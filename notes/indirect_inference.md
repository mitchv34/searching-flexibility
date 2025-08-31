# A Detailed Guide to Implementing Indirect Inference via Simulation-Based Moment Matching

## 1. Overview and Rationale

### 1.1 The Problem
Our structural model produces theoretical moments (e.g., the average wage). However, our empirical analysis relies on regression coefficients from regressions that include control variables (e.g., education, experience). Comparing a theoretical average wage to a regression coefficient is not an apples-to-apples comparison and can lead to biased parameter estimates.

### 1.2 The Solution: Indirect Inference
To solve this, we use a technique called **indirect inference**. The core idea is to subject our model's output to the exact same statistical procedure as our real-world data. The workflow is:
1.  Solve the structural model for a given set of parameters.
2.  Use the solved model to **simulate a large dataset** of workers with realistic characteristics.
3.  Run the **exact same regression** on both the real data and the simulated data.
4.  The objective of the estimation is to find the model parameters that make the **regression coefficients** from the simulated data match the coefficients from the real data.

This guide details the implementation of this workflow, with a special focus on ensuring the process is both computationally efficient and numerically stable for the optimizer.

---
## 2. The One-Time Setup (Before Estimation)

These steps are performed only **once** at the very beginning of the project. The goal is to create a fixed "scaffolding" for the simulation to ensure consistency and eliminate random noise from the objective function.

### 2.1 Create the Panel of Observable Characteristics `X`
We need a set of realistic, observable worker traits to serve as the control variables in our simulation.

* **Action:** Sample a large number, `N` (e.g., 50,000), of individuals from the real-world microdata (e.g., the CPS or SWAA). For each sampled individual, extract the vector of control variables, `Xᵢ` (e.g., a dummy for college education, years of experience, gender, etc.).
* **Output:** A saved file, `panel_X.jld2`, containing an `N x K` matrix of observable characteristics, where `K` is the number of control variables.

### 2.2 Create the Common Random Numbers Matrix
To ensure the objective function is smooth and deterministic, we must use the same set of random numbers for every evaluation. We need three draws for each simulated agent.

* **Action:** Generate and save an `N x 3` matrix of random draws from a `Uniform(0,1)` distribution. The three columns will be used for:
    1.  **Skill (`h`):** To determine the worker's latent productivity.
    2.  **Employment Status:** To determine if the worker has a job.
    3.  **Matched Firm Type (`ψ`):** To determine which firm an employed worker is matched with.
* **Implementation (Julia):**
    ```julia
    using Random, JLD2

    N_SIM = 50000
    
    # Set a seed for reproducibility
    Random.seed!(1234)

    # Generate and save an N x 3 matrix of random numbers
    common_random_numbers = rand(N_SIM, 3)
    jldsave("common_random_numbers.jld2"; common_random_numbers)
    println("Saved $(N_SIM)x3 common random numbers.")
    ```
* **Output:** A saved file, `common_random_numbers.jld2`, containing an `N x 3` matrix of random numbers.

---
## 3. The `objective_function` (The Core Loop)

This is the main function that the optimizer will call repeatedly. It must perform the entire simulation and return a single scalar value (the loss).

### Step 3.1: Solve the Structural Model
This step is unchanged from the current setup.
* **Action:** For the current parameter guess `θ` from the optimizer, call the `solve_model` function.
* **Output:** The converged `Results` object, `res_new`, containing the equilibrium objects (`S`, `u`, `n`, etc.).

### Step 3.2: Pre-calculate the Expected Policies
This is a crucial efficiency step.
* **Action:** Call the `calculate_average_policies` function.
* **Input:** The `prim_new` and `res_new` objects from the previous step.
* **Output:** Two `n_h x n_ψ` matrices: `avg_alpha_policy` and `avg_wage_policy`.

### Step 3.3: Generate the Simulated Panel of Agents
Here, we combine the fixed observables `X` with the unobservable skill `h`.
* **Action:** Load the saved `panel_X` and `common_random_numbers` matrix. Use the current parameter guess for `a_h` and `b_h` to transform the **first column** of the random numbers into skill draws.
* **Implementation (Julia):**
    ```julia
    # Inside the objective_function...
    
    panel_X = load("panel_X.jld2")["panel_X"]
    urns = load("common_random_numbers.jld2")["common_random_numbers"]

    skill_dist = Beta(prim_new.aₕ, prim_new.bₕ)

    # Use the first column of the random matrix for skill draws
    panel_h = quantile.(skill_dist, urns[:, 1])

    # You now have N simulated agents, defined by (panel_h[i], panel_X[i, :])
    ```

### Step 3.4: Assign Employment Outcomes to the Panel
To assign an employment status and a firm type `ψ` to each agent efficiently, we use the other two random draws and the solved steady-state distributions.

#### 3.4.1 Assign Employment Status
* **Action:** For each simulated agent `i`, use the **second column** of the random numbers (`urns[:, 2]`) to determine if they are employed or unemployed.
* **Method:**
    1.  For each agent `i` with skill `hᵢ`, find their corresponding row `h_index` on your discrete `h_grid`.
    2.  Calculate the unemployment rate for that skill type from your solved model: `unemp_rate_h = res_new.u[h_index] / (res_new.u[h_index] + sum(res_new.n[h_index, :]))`.
    3.  If `urns[i, 2] < unemp_rate_h`, the agent is unemployed. Otherwise, they are employed.

#### 3.4.2 Assign Matched Firm Type
* **Action:** For all agents assigned as employed, use the **third column** of the random numbers (`urns[:, 3]`) to draw a specific firm type `ψ`.
* **Method:**
    1.  For each employed agent `i` with skill `hᵢ` (and row `h_index`), calculate the conditional probability distribution of matching with each firm type: `P(ψ | h) = res_new.n[h_index, :] / sum(res_new.n[h_index, :])`.
    2.  Use the random draw `urns[i, 3]` to sample a specific firm column `ψ_index` from this conditional distribution.

### Step 3.5: Assign Wages and Work Arrangements
This step is now a very fast lookup for all employed agents.
* **Action:** For each agent `i` who was assigned to be employed in a match `(h_index, ψ_index)`, their final outcomes are:
    * `wage = avg_wage_policy[h_index, ψ_index]`
    * `alpha = avg_alpha_policy[h_index, ψ_index]`
* **Output:** A complete simulated `DataFrame` containing only the employed agents, with their outcomes and all control variables.

### Step 3.6: Run the "Twin" Regression and Compute Distance
This is the final step.
* **Action:** Run the exact same regression specification on your simulated `DataFrame` as you did on the real-world data.
* **Implementation (Julia):**
    ```julia
    using GLM

    # Assume `data_coeffs` is a vector of the target coefficients from the real data
    
    sim_regression = lm(@formula(log(wage) ~ alpha + college_educ + experience), simulated_data_employed_only)
    sim_coeffs = coef(sim_regression)

    loss = compute_distance(sim_coeffs, data_coeffs)

    return loss
    ```