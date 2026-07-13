---
title: "Why Go Full? Elevating Federated Learning Through Partial Network Updates"
date: 2026-06-26
tags: ["Federated Learning", "Partial FL", "Layer Mismatch", "NeurIPS"]
categories: ["Paper Review"]
series: ["Federated Learning"]
math: true
summary: "A NeurIPS 2024 paper that identifies the layer mismatch problem caused by full-network aggregation and proposes FedPart, which trains and aggregates only a few layers per round in sequence, improving both convergence speed and final performance."
---

> **Paper**: Why Go Full? Elevating Federated Learning Through Partial Network Updates — H. Wang, X. Liu, J. Niu, W. Guo, S. Tang, in *Advances in Neural Information Processing Systems (NeurIPS)*, 2024. [arXiv:2410.11559](https://arxiv.org/abs/2410.11559)

## TL;DR

Conventional FL updates and averages the full set of parameters every round, and this paper identifies a problem that arises in the process: **layer mismatch**. To mitigate it, the authors propose **FedPart**, which updates only a single layer or a few layers per communication round, and theoretically analyze its convergence rate in the non-convex setting to demonstrate its advantages over full-network updates.

## Background

### What Is Layer Mismatch?

After full-network aggregation, the layers of the averaged global model are no longer sufficiently aligned with one another, causing gradient/update step sizes to spike again during the next round of local training. Indeed, a sharp increase in update step size is observed right after a global update — averaging breaks the cooperative relationships between layers. Since this mismatch recurs every time parameter averaging occurs, it slows convergence and degrades overall performance.

The paper's central observation is that switching to partial updates — updating only a subset of parameters — largely resolves this problem.

## Method

### FedPart

- Each communication round trains and aggregates **only a single layer or a small number of layers**. One communication round consists of selecting a layer (or a few layers), performing local training, then server aggregation, then broadcast.
- In implementation, parameters are split into trainable and non-trainable sets, and a mask applied to the loss ensures that only the selected portion is reflected in the global model.

### Sequential Updating and Multi-Round Cycle Training

- Layers are trained **sequentially (cyclically) from shallow to deep**. Each layer is trained for 2 rounds before moving on to the next.
- As cycles repeat, every layer becomes trainable in turn, so the entire network gets trained evenly across multiple cycles.
- In terms of ordering, performance follows **sequential (shallow→deep) > reverse > random** — stabilizing the shallow layers first is most effective.

### Theoretical Analysis

The convergence rate of FedPart is analyzed in the non-convex setting, demonstrating its advantages over full-network updates.

## Experiments

Across a wide range of experimental settings, FedPart improves over full-network updates in convergence speed, final accuracy, and communication/computation efficiency. Both theoretical justification and empirical results are provided.

## Takeaways

- The paper's biggest contribution is reinterpreting partial updates not merely as a **communication-saving mechanism** but as a **training strategy that heals the layer mismatch** induced by full aggregation.
- The method achieves its gains with a remarkably simple rule — "one layer per round, in order from shallow to deep" — which makes it highly practical.
- As a NeurIPS paper complete with convergence theory, it is well positioned to be frequently cited as the theoretical foundation for the layer-wise/partial FL line of research.
