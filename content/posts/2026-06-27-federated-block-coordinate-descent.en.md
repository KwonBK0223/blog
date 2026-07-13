---
title: "Federated Block Coordinate Descent Scheme for Learning Global and Personalized Models"
date: 2026-06-27
tags: ["Federated Learning", "Block Coordinate Descent", "Personalization", "Asynchronous FL"]
categories: ["Paper Review"]
series: ["Federated Learning"]
math: true
summary: "A theoretical paper showing that the FL objective can be solved via block coordinate descent, proposing FedBCD, which achieves personalized models and asynchronous cloud updates simultaneously over a hierarchical communication architecture."
---

> **Paper**: Federated Block Coordinate Descent Scheme for Learning Global and Personalized Models — R. Wu, A. Scaglione, H.-T. Wai, N. Karakoc, K. Hreinsson, W.-K. Ma, in *Proc. AAAI Conference on Artificial Intelligence*, vol. 35, pp. 10355–10362, 2021. [arXiv:2012.13900](https://arxiv.org/abs/2012.13900)

## TL;DR

This paper provides the **theoretical justification for updating a model in blocks**. Starting from classic block coordinate descent (BCD), it proposes FedBCD, which learns a global model and personalized models simultaneously over a hierarchical communication architecture, and provides a sublinear convergence rate along with a delay analysis for asynchronous updates.

## Background

The paper poses two fundamental questions.

1. Is it efficient to maintain the **same model** everywhere?
2. Are **synchronous** cloud model updates effective?

Existing FedAvg-style methods update a single global model synchronously; this paper presents an optimization framework that permits both personalization and asynchrony.

The main contributions are as follows.

1. A tailored **hierarchical communication architecture** for federated learning (edge device – connected cloud server – cloud-server consensus, which qualifies as HFL in the broad sense)
2. A **lightweight block coordinate descent communication protocol** achieving model personalization and asynchronous cloud server updates
3. A proof of **sublinear convergence rate**
4. A **delay analysis** demonstrating the efficacy of the asynchronous scheme: runtime is estimated from the distribution of edge device message arrival times, linking the number of clients and servers involved in each update to runtime (a novel style of analysis at the time)
5. Experiments on standard ML tasks

## Method

### Edge Device Updates

- Each edge device maintains and updates a **personalized model** $x_i$ fitted to its local data.
- Updates are performed with accelerated stochastic projected gradient steps.
- A **quadratic penalty** keeps the personalized model from drifting too far from the cloud model — personalization is allowed, but tethered to the global consensus.

### Cloud Server Updates

Cloud servers update the global model $z$ by incorporating the personalized models $x_i$ sent by edge devices. There are two modes.

- **Sync-cloud**: all cloud servers wait until they have received the edge updates, after which the coordinator gathers all the information, updates a single global model $z$, and broadcasts it to every cloud server. This resembles FedAvg.
- **Async-cloud**: only the cloud servers that are ready first perform updates. When some cloud servers have received their edge updates, the coordinator performs model mixing among only the first $B$ cloud servers to arrive, updating just those servers' $z_n$. The rest are updated at the next opportunity.

### Nature of the Aggregation

The aggregation is not exactly FedAvg. It is closer to a **penalty-based, optimization-driven cloud update** than to FedAvg-style simple averaging. The overall problem is solved from a BCD perspective, alternating optimization over blocks — the personalized-model block and the global-model block.

## Experiments

The sync/async settings are validated on standard ML benchmarks in terms of convergence and runtime, and the delay analysis is shown to match the empirical results.

## Takeaways

- This paper's significance lies less in methodological novelty than in the **theoretical foundation that "the FL objective can be solved via block coordinate descent."** It is the mathematical root of subsequent block-wise/layer-wise partial update research.
- The framework is elegant in how it absorbs personalization via a penalty term and asynchrony via BCD's block selection, all within a single unified formulation.
- The delay analysis for async-cloud is an early and meaningful attempt to connect the systems perspective (arrival time distributions) with the optimization perspective (convergence).
