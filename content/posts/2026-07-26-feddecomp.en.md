---
title: "Decoupling General and Personalized Knowledge in Federated Learning via Additive and Low-Rank Decomposition"
date: 2026-07-26
tags: ["Federated Learning", "Personalization", "Low-Rank Decomposition", "FedDecomp"]
categories: ["Paper Review"]
series: ["Federated Learning"]
math: true
summary: "A review of FedDecomp, which redesigns every layer weight as the sum of a shared full-rank branch and a personalized low-rank branch, separating general and client-specific knowledge at the parameter level."
---

> **Paper**: Decoupling General and Personalized Knowledge in Federated Learning via Additive and Low-Rank Decomposition — Xinghao Wu, Xuefeng Liu, Jianwei Niu, Haolin Wang, Shaojie Tang, Guogang Zhu, Hao Su, *ACM MM 2024*. [DOI:10.1145/3664647.3681514](https://doi.org/10.1145/3664647.3681514)

## TL;DR

- The goal of personalized FL (PFL) is not a single global model but a per-client personalized model $w_i$, which can be framed as the problem of separating general knowledge to be shared from client-specific knowledge to be kept local.
- Existing parameter-partition methods split layers or parameters into shared or personalized ones. But whichever parameter is designated as shared can still carry client-specific knowledge inside it, so enforcing a perfect split may actually hurt both general and client-specific performance.
- **FedDecomp** decomposes each layer weight additively as $\theta_i^k = \sigma_i^k + \tau_i^k$. Here $\sigma_i^k$ is a shared full-rank branch holding general knowledge, and $\tau_i^k = B_i^k A_i^k$ is a personalized low-rank branch holding client-specific knowledge.
- Within one communication round it uses alternating training: first train the low-rank branch $\tau_i$ so it absorbs local drift, then train the full-rank branch $\sigma_i$ so it focuses on general knowledge. Only $\sigma_i$ is sent to the server, while $\tau_i$ stays local to each client.

## Background

### Problem definition of personalized FL

The objective of PFL is not to hand every client the same global model, but to give each client $i$ a personalized model $w_i$ that fits its own data. This reduces to a single question: how do we separate the general knowledge that should be shared across clients from the client-specific knowledge that should stay local to each client?

A one-line distinction among related settings makes the rest clearer. **Personalized FL** aims to build a different personalized model per client; **partial FL** trains, transmits, and aggregates only a subset of parameters rather than the whole model; and **cyclic FL** cyclically rotates which parameters or blocks are shared or trained across rounds. FedDecomp targets personalized FL, but pushes the unit of separation below the layer, down to parameter components.

### Two families of prior approaches

Judged by how they handle heterogeneity (non-IID data), PFL approaches to knowledge separation fall into two families.

- **Parameter-partition-based FL** (FedPer, FedBN, FedRep, FedRoD): only a subset of parameters or layers is kept shared, while the rest is personalized and stays local. The limitation is that whichever parameter is designated as shared can still have client-specific knowledge mixed into it, and conversely, parameters kept personalized still carry general knowledge.
- **Parameter-decomposition-based FL** (FedDecomp): instead of splitting parameters into shared or personalized, it represents each parameter itself as several components. As a result, every layer holds a shared component and a personalized component **simultaneously**.

Underlying this contrast is a single observation. Even if you cleanly split layers to perfectly separate general from client-specific knowledge, each part inevitably carries some of the opposite knowledge. Forcing a perfect split can drag down both general and client-specific performance together. Starting from this observation, FedDecomp moves the separation from a layer-level dichotomy to an additive decomposition inside each parameter.

## Method

### Key idea — additive decomposition

FedDecomp defines each layer weight of the personalized model as a sum of two components.

$$\theta_i^k = \sigma_i^k + \tau_i^k$$

The crucial point is that this decomposition is **a design ahead of time, not a decomposition after the fact**. It does not take an already-trained weight $\theta$ and cut it into $\sigma$ and $\tau$ via SVD; instead it parameterizes and trains the layer as $\sigma + \tau = \sigma + BA$ from the start. Here $\theta_i^k$ is not a weight stored independently, but the **effective weight** actually used in the forward pass.

- $\sigma_i^k \in \mathbb{R}^{I \times O}$: the **shared full-rank branch**, kept as an ordinary weight matrix. Being full rank, it has enough capacity to hold general knowledge.
- $\tau_i^k = B_i^k A_i^k$: the **personalized low-rank branch**, formed as the product of two smaller matrices. For a fully-connected layer, $B_i^k \in \mathbb{R}^{I \times r}$ and $A_i^k \in \mathbb{R}^{r \times O}$, with rank $r = R_l \cdot \min\{I, O\}$. For a convolution layer, the kernel weight is unfolded, decomposed into low rank, and then reshaped back to the convolution weight shape.

Under the assumption that client-specific knowledge fits comfortably in a low capacity, the shared branch holds general knowledge in full rank while the personalized branch holds only a low-rank client-specific correction. In form this resembles LoRA: a per-client low-rank adapter added on top of a large shared weight.

### Initialization

$A$ is initialized with Gaussian random values and $B$ with zeros. Therefore, at the start of training, $B = 0$, so

$$\tau_i^k = B_i^k A_i^k = 0$$

and the initial effective weight becomes

$$\theta_i^k = \sigma_i^k + 0 = \sigma_i^k.$$

In other words, the personalized branch has no effect early in training, so the model effectively starts out like FedAvg. As local training proceeds, $\tau_i$ begins to learn the client-specific correction.

### Alternating training

FedDecomp trains the two branches alternately within a single communication round. Each client $i$ holds $\sigma_i$ and $\tau_i$, and uses $\theta_i = \sigma_i + \tau_i$ in the forward pass.

1. **Train the personalized branch**: freeze $\sigma_i$ and train $\tau_i$. In this step the low-rank branch $\tau_i$ first absorbs client-specific knowledge, i.e., the local drift.
2. **Train the shared branch**: freeze $\tau_i$ and train $\sigma_i$. Since $\tau_i$ has already absorbed much of the local drift, $\sigma_i$ is pulled less toward the non-IID local optimum and can focus more on learning general knowledge.
3. **Transmit**: the client sends only the shared full-rank matrix $\sigma_i$ to the server; $\tau_i$ is not transmitted.
4. **Aggregate**: the server averages the received full-rank matrices.
$$\sigma^{t+1} = \frac{1}{N} \sum_{i=1}^{N} \sigma_i^{t+1}$$
5. **Broadcast and initialize**: the server broadcasts the aggregated global $\sigma$, and each client reinitializes its $\sigma_i$ to the global $\sigma$ while keeping $\tau_i$ local.

The ordering — absorb drift with $\tau$ first, then train $\sigma$ — is the heart of the strategy and what sets it apart from simply attaching an adapter.

What the user sets directly is not the values of $\sigma$ or $\tau$; those are decided by training. What the user sets is the structure and the hyperparameters: the rank ratios that determine the capacity of the personalized branch — $R_l$ (FC layers) and $R_c$ (conv layers) — and the epoch ratios that split training between the branches — $E_{lora}$ (epochs training $\tau$ with $\sigma$ frozen) and $E_{global}$ (epochs training $\sigma$ with $\tau$ frozen).

## Privacy & Experiments

### Privacy — the DLG perspective

FedDecomp analyzes privacy from the perspective of Deep Leakage from Gradients (DLG). The DLG attack proceeds as follows.

1. The attacker steals the gradient each client computes on its local data.
2. Through iterative optimization, the attacker searches for an input whose gradient matches the stolen gradient as closely as possible, thereby reconstructing the original input.

In FedDecomp, only the shared full-rank branch $\sigma$ is transmitted to the server, while the personalized low-rank branch $\tau$ remains local to each client. The argument of the privacy analysis is that this structure — limiting what the server can observe to $\sigma$ — restricts the surface available to gradient-based reconstruction attacks.

### Experiments

The paper verifies across multiple datasets that FedDecomp effectively mitigates the performance degradation caused by non-IID data. The central experimental claim is that the division of labor — the shared full-rank branch handling general knowledge while the personalized low-rank branch absorbs local drift — is more robust to heterogeneity than prior PFL methods that merely split parameters into two groups.

## Takeaways

FedDecomp addresses the problem that "splitting parameters into shared or personalized leaves each part contaminated by the opposite knowledge" by redesigning the parameter itself as a sum of two components. In summary:

- **A shift in the unit of separation**: unlike prior PFL that selectively splits layers/parameters into shared and personalized, FedDecomp decomposes every parameter into a shared full-rank plus a personalized low-rank component. Its core contribution is separating general from client-specific knowledge at the level of parameter components rather than at the layer level.
- **A designed training order**: while it superficially resembles LoRA-style methods that add a low-rank adapter on top of a shared weight, its novelty lies in the ordering — "absorb local drift with $\tau$ first, then train $\sigma$" — and in formalizing this from a decomposition viewpoint. The logic of alternating training is that $\tau$ must first strip away the non-IID bias before $\sigma$ can learn clean general knowledge.
- **A reviewer's view**: performance depends on the alternating training scheme and the rank/epoch hyperparameters ($R_l, R_c, E_{lora}, E_{global}$), so sensitivity to these settings is the practical crux of applying the method. Transmitting only $\sigma$ also carries the incidental benefit of confining communication to the shared branch, reducing the privacy surface.

When client heterogeneity stems mainly from the mixing of knowledge inside individual parameters, FedDecomp offers a separation one step finer than layer-level partitioning.
