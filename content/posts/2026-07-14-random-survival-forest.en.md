---
title: "Random Survival Forests"
date: 2026-07-14
tags: ["Survival Analysis", "Random Forest", "Ensemble", "Medical AI"]
categories: ["Paper Review"]
series: ["Medical AI"]
math: true
summary: "A review of Random Survival Forests, a non-parametric survival model that extends Breiman's Random Forest to right-censored data by redesigning the splitting rule and terminal node prediction to be survival-specific, yielding an ensemble cumulative hazard function."
---

> **Paper**: Random Survival Forests — Hemant Ishwaran, Udaya B. Kogalur, Eugene H. Blackstone, Michael S. Lauer, *The Annals of Applied Statistics* 2(3), 841–860, 2008. [DOI:10.1214/08-AOAS169](https://doi.org/10.1214/08-AOAS169)

## TL;DR

Random Survival Forests (RSF) extend Breiman's Random Forest to right-censored survival data as a fully non-parametric ensemble model. Growing many trees on bootstrap samples and selecting a random subset of variables as split candidates at each node is inherited directly from standard RF. The difference lies in the outcome: it is neither a class label nor a continuous value, but censored survival data. As a consequence, the splitting rule, the prediction, and the error calculation are all made survival-specific. Each terminal node produces a cumulative hazard function (CHF) via the Nelson-Aalen estimator, and averaging over trees yields a patient-specific ensemble CHF. The derived ensemble mortality serves as an interpretable predicted outcome, and out-of-bag (OOB) data allow prediction error and the C-index to be estimated without any separate validation set.

## Background

### Survival data and right censoring

Survival analysis models the time until an event of interest occurs. The data for patient $i$ consist of three components.

- **Covariates $x_i$**: age, sex, stage, laboratory values, treatment information, and other predictors
- **Observed time $T_i$**: either the event time or the time of last follow-up
- **Event indicator $\delta_i$**: $\delta_i = 1$ if the event was observed, $\delta_i = 0$ under right censoring

Right censoring means that no event had occurred by the last follow-up, but what happened afterward is unknown. Simply discarding censored samples, or treating them as events, introduces severe bias. The splitting criterion and the error calculation of a survival model must therefore handle censoring explicitly. This is precisely why an off-the-shelf regression or classification tree cannot be used.

### The assumptions of CoxPH

The Cox proportional hazards model, the standard tool of survival analysis, factorizes the hazard into a baseline hazard and a patient-specific risk score.

$$
h(t \mid x) = h_0(t) \exp\big(\beta^\top x\big)
$$

This form carries two strong assumptions. The first is **proportional hazards**: the ratio of hazards between two patients is constant over time. The second is **linearity**: the log hazard is a linear combination of the covariates. To capture non-linear effects or higher-order interactions, the analyst must specify the relevant terms by hand, which becomes practically infeasible as the number of covariates grows.

### The need for a non-parametric ensemble

Random Forests automatically capture complex interactions through bootstrap resampling and random variable selection, and they are robust to irrelevant noise variables. Standard RF, however, cannot handle censored outcomes directly. RSF aims to preserve the ensemble machinery of RF while confronting censoring head-on, assuming neither proportional hazards nor linearity.

## Method

### Algorithm

The RSF training procedure is as follows.

1. Draw $B$ bootstrap samples from the original data. Roughly 63% of the observations are selected as in-bag in each sample; the remaining ~37% are left out-of-bag (OOB).
2. Grow one survival tree on each bootstrap sample.
3. At each node, randomly select $p$ variables out of the full set as split candidates.
4. Among the candidate variables and split points, choose the split that **maximizes survival difference**.
5. Grow the tree until each terminal node satisfies a minimum requirement on the number of events.
6. Compute the CHF at each terminal node using the Nelson-Aalen estimator.
7. Average the CHFs across trees to obtain the ensemble CHF.
8. Estimate prediction error using the OOB data.

### Binary survival trees and the splitting rule

An RSF tree is a binary tree that splits each node into two daughter nodes, exactly as in CART. Structurally it resembles an ordinary decision tree, but the splitting criterion is neither impurity nor variance reduction; it is **survival difference**.

A good split is one that makes the survival patterns of the two daughter nodes as different as possible. The canonical criterion is the **log-rank splitting rule**, which measures the separation between the survival curves of the left and right daughters for a candidate split $(v, c)$ using the log-rank statistic.

$$
L(v, c) = \frac{\sum_{k=1}^{K} \left( d_{1,k} - Y_{1,k} \dfrac{d_k}{Y_k} \right)}{\sqrt{\sum_{k=1}^{K} \dfrac{Y_{1,k}}{Y_k} \left( 1 - \dfrac{Y_{1,k}}{Y_k} \right) \left( \dfrac{Y_k - d_k}{Y_k - 1} \right) d_k}}
$$

Here $t_1 < \cdots < t_K$ are the distinct event times within the node, $d_k$ and $Y_k$ are the number of events and the size of the risk set at time $t_k$, and $d_{1,k}$ and $Y_{1,k}$ are the corresponding quantities in the left daughter node. A larger $|L(v, c)|$ indicates a sharper separation between the survival patterns of the two groups, so the split maximizing this value is adopted. The tree therefore grows in a direction that pulls apart patients with different survival risk.

Beyond the log-rank rule, the paper also examines log-rank score and conservation-of-events splitting rules. What they share is that each is a statistic that handles censoring explicitly.

### Terminal nodes and the Nelson-Aalen estimator

A terminal node is a final node that is not split further. Patients falling into the same terminal node are assumed to share a similar survival pattern, and hence a single CHF.

The CHF at terminal node $h$ is given by the Nelson-Aalen estimator.

$$
\hat{H}_h(t) = \sum_{t_{k,h} \le t} \frac{d_{k,h}}{Y_{k,h}}
$$

Here $t_{k,h}$ are the event times within node $h$, $d_{k,h}$ the number of events at that time, and $Y_{k,h}$ the size of the risk set. The CHF represents event risk accumulated over time and is monotonically increasing by construction. **The higher the CHF, the greater the cumulative risk of death or event up to that time point.**

When a new patient $x$ traverses the tree and lands in terminal node $h$, that node's CHF becomes the prediction for that patient.

$$
H(t \mid x) = \hat{H}_h(t), \quad x \in h
$$

### How RSF differs from standard RF

The underlying machinery is the same as RF, but the decisive difference is **what the model outputs**.

| | Standard Random Forest | Random Survival Forest |
|---|---|---|
| Outcome | Class label / continuous value | Right-censored survival data |
| Split criterion | Gini impurity / variance reduction | Log-rank and other survival differences |
| Terminal node prediction | Majority vote / mean | Nelson-Aalen CHF |
| Ensemble output | Predicted class / predicted value | Ensemble CHF, survival curve |
| Error estimation | OOB misclassification / MSE | OOB prediction error based on C-index |

In other words, an RSF prediction is not a scalar but a **function of time**. Each patient receives a curve.

### Missing data

The paper proposes **adaptive tree imputation** so that training and prediction remain possible when covariates, survival times, or censoring indicators are partially missing. The goal is not to recover missing values accurately but to keep splitting and prediction stable in their presence. At each node, missing entries are filled by randomly drawing from the non-missing in-bag data within that node, and missing values in test data are handled the same way.

## Prediction & Interpretation

### Ensemble CHF

Each of the $B$ trees produces a CHF $H_b(t \mid x)$ for patient $x$. The forest-level prediction is their average.

$$
H_e(t \mid x) = \frac{1}{B} \sum_{b=1}^{B} H_b(t \mid x)
$$

The OOB version averages only over the trees for which the patient was not in-bag. Letting $I_{i,b} = 1$ denote that patient $i$ is OOB for tree $b$,

$$
H_e^{**}(t \mid x_i) = \frac{\sum_{b=1}^{B} I_{i,b} \, H_b(t \mid x_i)}{\sum_{b=1}^{B} I_{i,b}}
$$

Because this OOB ensemble CHF is built exclusively from trees that never saw the patient during training, it provides the basis for an honest performance estimate without any hold-out set.

### Ensemble mortality

A CHF is a curve, which makes patient-to-patient comparison cumbersome. The paper defines **ensemble mortality** by summing the CHF over all observed event times, collapsing it into a single scalar.

$$
M_e(x_i) = \sum_{k=1}^{K} H_e(t_k \mid x_i)
$$

This quantity is interpreted as the expected total number of events if all patients had the same covariates as patient $i$. It expresses a patient's anticipated cumulative risk as one number and can be used directly for risk ranking and C-index computation. A larger value means a higher-risk patient.

### OOB-based error estimation

Roughly 37% of the data are left out of each bootstrap sample and constitute the OOB set. Each tree is trained on its in-bag data and evaluated on the OOB data it never saw. RSF computes OOB mortality from the OOB ensemble CHF and uses it to estimate the C-index and the prediction error.

$$
\text{Error} = 1 - \text{C-index}
$$

The C-index is a rank-based metric measuring whether, given two patients, the model assigned higher risk to the one who actually experienced the event sooner; 0.5 corresponds to random guessing. A meaningful model therefore needs a prediction error below 0.5. That this estimate falls out of training itself, requiring neither cross-validation nor a separate validation set, is a substantial practical advantage.

### Variable importance

VIMP (variable importance) is measured by how much the OOB prediction error degrades when a given variable is randomly permuted or its splits are randomly assigned. A large increase in error means the variable matters; a value near zero or negative suggests it behaves like noise. In an experiment injecting a large number of noise variables into breast cancer data, RSF reliably separated genuine signal variables from noise.

Across several real and simulated datasets, RSF achieved lower prediction error than Cox regression and censored RF regression, and its performance did not collapse as noise variables accumulated. In a CABG case study, it surfaced complex survival interactions involving BMI, renal function, and smoking without any of them being pre-specified.

## Takeaways

- RSF is a non-parametric survival model that retains the bootstrap ensemble structure of Random Forest while making the splitting rule and terminal node prediction survival-specific. It assumes neither proportional hazards nor linearity.
- The fact that the prediction is a CHF matters practically. Rather than a single risk score, one obtains a curve over time, which leaves ample room for clinical interpretation; summarizing it as ensemble mortality immediately enables rank comparison.
- The OOB structure allows prediction error and the C-index to be estimated without a separate validation set, a property that is especially valuable for clinical data where samples are scarce.
- That said, **the ensemble as a whole remains black-box in character.** An individual survival tree can be read and followed by a human, but there is no way to intuitively interpret the average of hundreds or thousands of trees. Interpretation relies on post hoc tools such as VIMP, partial plots, and CHF visualization, and individual feature effects are not given in closed form the way CoxPH coefficients are. Causal interpretation is harder still.
- **Computational cost in high dimensions is non-trivial.** The log-rank statistic must be computed for every candidate variable at every node while searching for the optimal split point, so cost grows quickly with the number of variables and trees.
- **Sensitivity to the choice of splitting rule** remains an open issue. Tree structure and VIMP rankings can shift depending on whether log-rank, log-rank score, or conservation-of-events is used, and no general theory establishes which rule is optimal under which conditions.
