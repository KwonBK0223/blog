---
title: "Deep-CR MTLR: A Multi-Modal Approach for Cancer Survival Prediction with Competing Risks"
date: 2026-07-15
tags: ["Survival Analysis", "Competing Risks", "Multi-Modal", "Medical AI"]
categories: ["Paper Review"]
series: ["Medical AI"]
math: true
summary: "A review of Deep-CR MTLR, a multi-modal survival model that takes pre-treatment CT volumes together with clinical variables and predicts the cumulative incidence of cancer-specific death and other-cause death jointly through an MTLR head extended to competing risks."
---

> **Paper**: Deep-CR MTLR: A Multi-Modal Approach for Cancer Survival Prediction with Competing Risks — Sejin Kim, Michal Kazmierski, Benjamin Haibe-Kains, *PMLR (Survival Prediction — Algorithms, Challenges and Applications)*, 223–231, 2021. [arXiv:2012.05765](https://arxiv.org/abs/2012.05765)

## TL;DR

Deep-CR MTLR is a multi-modal survival model that predicts a patient-specific survival distribution from a pre-treatment CT volume together with clinical EMR variables. A 3D CNN learns a prognostic representation directly from voxels, clinical variables are one-hot encoded or normalized, and the two representations are concatenated and passed through shared fully connected layers. The final head is Multi-Task Logistic Regression (MTLR) extended to competing risks: a separate parameter set per event type lets the model predict *when* an event occurs and *which* event occurs at the same time, and accumulating these probabilities over time yields event-specific cumulative incidence functions (CIF). On RADCURE, a head and neck cancer cohort, the multi-modal model outperformed single-modality models overall, with the largest gains in cancer-specific survival prediction.

## Background

### Clinical variables alone are not enough

Prognostic prediction is the starting point for treatment decisions in oncology, yet it has largely relied on traditional clinical factors such as stage, age, and performance status. These variables are easy to interpret but fail to explain much of the heterogeneity between patients. Pre-treatment CT scans, on the other hand, already contain rich quantitative prognostic information about tumor size, shape, texture, and its relationship to surrounding tissue. The difficulty is that this information is so high-dimensional that hand-crafted rules struggle to isolate the signal that actually matters.

Radiomics and CNN-based survival analysis were introduced to close this gap. The authors argue, however, that existing approaches share three limitations.

- **They use imaging only.** Clinical variables already present in the EMR go unused, discarding the possibility that the two modalities complement one another.
- **They detour through surrogate tasks.** Reformulating the problem as binary classification (for example, "death within two years") throws away timing and censoring information and never produces a survival distribution.
- **They impose strong assumptions.** Constraints such as the proportional hazards assumption of Cox-family models are baked in with no guarantee that they hold in real data.

### What competing risks means

On top of this, the authors confront the **competing risks** setting head-on. A patient may be subject to several types of event, but those events are **mutually exclusive**, so in practice only one of them is ever observed first.

For head and neck cancer patients, the setting looks like this.

- **Event 1**: cancer-specific death (death from the primary cancer)
- **Event 2**: death from other causes
- **Event 0**: censoring (end of follow-up)

If a patient experiences cancer death first, other-cause death **can never be observed** for that patient, and vice versa. The two events compete, each blocking the observation of the other. This is not an artificial construction; it is the natural structure of the problem in cancer prognosis.

Yet the conventional overall survival endpoint collapses every death into a single event, regardless of cause. Doing so dilutes exactly the information that imaging contributes — the cancer-specific risk signal arising from tumor aggressiveness — by mixing it with other-cause deaths driven by aging and comorbidity. Recovering the true contribution of imaging therefore requires predicting the event types separately, and that is the premise of this paper.

## MTLR (Multi-Task Logistic Regression)

The backbone of Deep-CR MTLR is MTLR, a model that directly predicts an individual survival distribution.

### Discretizing the time axis

MTLR partitions continuous survival time into a set of discrete time intervals. With cut points $0 = \tau_0 < \tau_1 < \cdots < \tau_m$, the $k$-th interval is $[\tau_{k-1}, \tau_k)$.

Each patient's survival time $T$ is encoded on this discrete grid as a binary vector $\mathbf{y} = (y_1, \dots, y_{m-1})$.

$$
y_j = \mathbb{1}\left[T \le \tau_j\right]
$$

If the event occurs in the $k$-th interval, $\mathbf{y}$ is a step vector that is zero up front and switches to one for every $j \ge k$.

### Dependent logistic regressors

MTLR assigns one logistic regression task per time interval, but these tasks are **not trained as a collection of independent binary classifiers**. Independent classifiers can produce predictions that contradict the ordering of time — dead at three years, alive at five. Instead, MTLR places a probability on the entire step vector $\mathbf{y}$, which structurally enforces temporal consistency. The probability that the event falls in the $k$-th interval is

$$
P\left(T \in [\tau_{k-1}, \tau_k) \mid x\right) = \frac{\exp\left(\sum_{j=k}^{m-1} \left(\theta_j^\top x + b_j\right)\right)}{\sum_{l=1}^{m} \exp\left(\sum_{j=l}^{m-1} \left(\theta_j^\top x + b_j\right)\right)}
$$

The numerator scores the corresponding step vector and the denominator normalizes over all possible event times. Because of this construction, the model's output is not a set of independent per-time-point probabilities but a **complete survival distribution for a single patient**.

### Handling censoring

For a right-censored patient the event time is unknown, but the information that "no event occurred at least up to the censoring time" is available. MTLR **marginalizes** over every time interval in which the event could still have occurred after censoring and folds that into the likelihood. Writing $c$ for the censoring interval,

$$
P\left(T \ge \tau_{c} \mid x\right) = \sum_{k > c} P\left(T \in [\tau_{k-1}, \tau_k) \mid x\right)
$$

and this quantity enters the likelihood directly. Censored samples are neither discarded nor mistaken for observed events; their partial information is used exactly as it is.

## Method

### Neural MTLR: how is a neural network still logistic regression

MTLR is originally a logistic-regression-based model built on a linear combination $\theta_j^\top x$ of the input features. Neural MTLR simply **replaces those input features with a representation learned by a neural network**. The final survival head keeps the MTLR structure intact; only the feature extractor in front of it becomes a neural network.

$$
\theta_j^\top x \;\longrightarrow\; \theta_j^\top h_\phi(x)
$$

where $h_\phi(\cdot)$ is a network with parameters $\phi$. The apparent contradiction between "neural" and "logistic regression" dissolves once one sees that the nonlinearity lives in representation learning, while the head that assembles the survival distribution remains MTLR.

### The Deep-CR MTLR architecture

The full model consists of two branches and a fusion stage.

- **Clinical branch**: categorical variables are one-hot encoded and continuous variables are normalized, producing a clinical representation $h_{\text{clin}}$.
- **Image branch**: the pre-treatment CT volume is fed to a 3D CNN, which learns a prognostic representation $h_{\text{img}}$ directly at the voxel level. Crucially, no hand-defined radiomics features are used; the representation is learned end to end.
- **Fusion**: the two representations are concatenated and passed through shared fully connected layers.

$$
h = \mathrm{FC}\left(\left[\, h_{\text{img}} \;;\; h_{\text{clin}} \,\right]\right)
$$

This shared representation $h$ is then handed to the competing risks MTLR head.

### The competing risks extension

A standard survival model predicts a single event type. Deep-CR MTLR extends MTLR to the competing risks setting by maintaining **a separate MTLR parameter set for each event type**. For event type $e \in \{1, \dots, E\}$ the model learns its own parameters $\left(\theta_j^{(e)}, b_j^{(e)}\right)$, so it predicts *when* an event occurs and *which* event occurs simultaneously.

The output is a **joint probability** over (time interval × event type).

$$
P\left(T \in [\tau_{k-1}, \tau_k),\; D = e \mid x\right) = \frac{\exp\left(\sum_{j=k}^{m-1}\left(\theta_j^{(e)\top} h + b_j^{(e)}\right)\right)}{\displaystyle\sum_{e'=1}^{E} \sum_{l=1}^{m} \exp\left(\sum_{j=l}^{m-1}\left(\theta_j^{(e')\top} h + b_j^{(e')}\right)\right)}
$$

Here $D \in \{1, \dots, E\}$ denotes the event type actually realized. Because normalization runs over all event types and all time intervals jointly, the mutual exclusivity of the events is encoded directly in the output structure.

Accumulating this joint probability over time yields the **event-specific cumulative incidence function (CIF)**.

$$
F_e(t \mid x) = P\left(T \le t,\; D = e \mid x\right) = \sum_{k \,:\, \tau_k \le t} P\left(T \in [\tau_{k-1}, \tau_k),\; D = e \mid x\right)
$$

$F_e(t \mid x)$ is the probability that event $e$ occurs for this patient before time $t$. For instance, $F_1(2\text{ years} \mid x)$ is the probability of cancer death within two years and $F_2(2\text{ years} \mid x)$ the probability of other-cause death within two years. Obtaining these two quantities separately is the practical payoff of the competing risks extension.

## Experiments

### Dataset and setup

- **Dataset**: RADCURE, consisting of pre-treatment CT images and clinical variables for 2,552 head and neck cancer patients.
- **Events**: death from the primary cancer, death from other causes, and censoring.
- **Split**: train / validation / test are split by diagnosis date rather than at random, a temporal split that gives a more realistic estimate of generalization to future patients.

### Baselines and metrics

The baselines are a clinical-only linear MTLR, an image-only convnet, and a clinical-only Neural MTLR. Together they control for (1) using a single modality and (2) using a linear model without representation learning.

Two metrics are reported.

- **2-year AUROC**: discrimination for event occurrence at the two-year mark.
- **Cause-specific C-index**: rank concordance computed per event type, measuring whether the model assigns higher risk to patients who actually experienced that event sooner.

### Results

- The multi-modal model outperformed the single-modality models overall. Imaging and clinical information complement rather than substitute for each other.
- The gains were most pronounced in **cancer-specific survival prediction**, which is consistent with the motivating hypothesis: imaging primarily reflects the aggressiveness of the tumor itself, and collapsing all deaths into overall survival dilutes that signal.
- Deep-CR MTLR achieved a 2-year AUROC of 0.774 and a C-index of 0.788 for cancer death.
- Deep models outperformed linear MTLR, evidence that learning **nonlinear interactions** between clinical variables and imaging features genuinely pays off.

## Takeaways

- Deep-CR MTLR combines CT imaging with clinical EMR data and, through an MTLR head extended to competing risks, predicts the risks of cancer-specific death and other-cause death jointly. The two design pillars are predicting the survival distribution directly rather than detouring through a surrogate task, and keeping event types separate rather than collapsing them.
- **The model may be sensitive to the choice of time discretization.** The output structure of MTLR depends on the number and placement of the cut points $\{\tau_k\}$, so this hyperparameter governs both performance and the resolution of the CIF. That is a design burden continuous-time models do not carry.
- **There is no direct comparison with Cox, DeepSurv, or DeepHit.** The authors leave this to future work, but DeepHit in particular is a canonical deep model for competing risks, so its absence is a notable gap: the current baselines function largely as ablations of the proposed model itself.
- **Interpretability remains limited.** The paper establishes that imaging contributes to prognostic prediction, but what CT findings the CNN responds to is left open, and this remains an obstacle to clinical adoption.
- **Competing risks predictions do not translate directly into treatment decisions.** What the model produces is event-specific risk estimated from observational data, not the causal effect of an intervention. Between "this patient has a high risk of cancer death within two years" and "therefore this treatment should be given" lies an additional layer of causal inference.
