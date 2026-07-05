---
title: "Is Aggregation the Only Choice? Federated Learning via Layer-wise Model Recombination"
date: 2026-06-29
tags: ["Federated Learning", "Model Recombination", "Non-IID", "KDD"]
categories: ["Paper Review"]
series: ["Partial & Layer-wise Federated Learning"]
math: true
summary: "Proposes FedMR, which replaces FedAvg's simple averaging by shuffling and recombining local models layer by layer before dispatching them to clients, achieving both escape toward flat minima and knowledge sharing in non-IID settings."
---

> **Paper**: Is Aggregation the Only Choice? Federated Learning via Layer-wise Model Recombination — M. Hu, Z. Yue, X. Xie, C. Chen, Y. Huang, X. Wei, X. Lian, Y. Liu, M. Chen, in *Proc. 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)*, 2024. [doi:10.1145/3637528.3671722](https://doi.org/10.1145/3637528.3671722) / [arXiv:2305.10730](https://arxiv.org/abs/2305.10730)

## TL;DR

True to its title — "Is aggregation the only choice?" — this paper proposes FedMR, which replaces FedAvg's averaging-based aggregation with **layer-wise model recombination**. The cloud decomposes the uploaded local models layer by layer, shuffles layers sharing the same index to assemble new models, and dispatches them to clients, improving generalization in non-IID settings.

## Background

Overcoming the drawbacks of the FedAvg-based paradigm and improving FL performance in non-IID scenarios is an important challenge. The authors summarize FedAvg's problems as follows.

1. In FL, data distributions differ across clients.
2. As a result, local models move in different optimization directions.
3. FedAvg simply averages these local models.
4. Simple averaging can dilute or discard the specific information each local model carries.
5. Moreover, since every client restarts local training from the same global model, the local search can get trapped in a particular **sharp region**.
6. Solutions with good generalization are more likely to lie in a **flat area** than in a sharp one.
7. Therefore, replacing simple averaging with layer-wise recombination — which perturbs models in the solution space — increases the chance of escaping sharp areas and reaching flat ones.

The authors claim the following contributions.

1. **FedMR**, a novel layer-wise model recombination method that replaces FedAvg-based model aggregation
2. Accelerating the entire FL training process by combining the strengths of recombination and aggregation
3. A convergence proof in the convex setting and experiments in non-convex scenarios
4. Extensive experiments demonstrating effectiveness and compatibility

## Method

### The Three Steps of FedMR

- **Step 1 — Model dispatching**: the cloud sends each of $K$ intermediate/recombined models to one of $K$ selected clients. (This differs from FedAvg, which sends the same global model to everyone.)
- **Step 2 — Model upload**: each client trains the received model on its local data, then uploads the updated local model to the cloud.
- **Step 3 — Model recombination**: the cloud decomposes the uploaded models layer by layer and **shuffles layers with the same index** to generate $K$ new recombined models.

Because shuffling occurs only among layers with the same index, the recombined models retain a homogeneous architecture. A client receives a model containing layers from other clients and continues local training on it; knowledge from other clients is transferred in the process, producing a **knowledge-sharing effect analogous to aggregation**. Information thus propagates through the network via shuffle-and-retrain, without any explicit averaging.

### Two-Stage Training Scheme

Early in training, the layers are not yet aligned with one another, and applying recombination in this state can further break layer dependencies and slow early convergence. To mitigate this, training proceeds in two stages.

- **Stage 1 — Aggregation-based pre-training**: train a rough global model with FedAvg for a fixed number of rounds.
- **Stage 2 — Model recombination stage**: apply FedMR recombination starting from the pre-trained models.

## Experiments

Convergence is proven theoretically in the convex setting, and extensive experiments are conducted in non-convex scenarios. Compared to existing SOTA FL methods, FedMR significantly improves inference accuracy without exposing client privacy.

## Takeaways

- The core insight is the paradigm shift that FL aggregation can be **"recombination" rather than "averaging."** Retraining on shuffled models is itself an implicit knowledge-sharing mechanism.
- The flat-minima motivation is intriguing: recombination jostles models around the solution space, helping them escape sharp minima.
- That said, the need for FedAvg pre-training — due to broken layer dependencies — suggests recombination is not a cure-all but rather a **tool suited to fine-grained exploration in the later phase of training**.
