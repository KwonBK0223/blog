---
title: "Adaptive Multi-Model Hierarchical Federated Learning for Robust IoT Intrusion Detection"
date: 2026-06-19
tags: ["Federated Learning", "Hierarchical FL", "Intrusion Detection", "IoT", "Clustering"]
categories: ["Paper Review"]
series: ["Federated Learning Applications"]
math: true
summary: "An adaptive multi-model HFL (A-MM-HFL) framework that combines similarity-based clustering, multi-model meta-aggregation, and dynamic client-side model selection across a client–edge–cloud hierarchy."
---

> **Paper**: Adaptive Multi-Model Hierarchical Federated Learning for Robust IoT Intrusion Detection — S. Latif, D. Djenouri, *Sensors*, 2026.

## TL;DR

This paper proposes A-MM-HFL, an adaptive multi-model HFL framework that integrates **similarity-based clustering, multi-model aggregation, and dynamic model selection** within a client–edge–cloud hierarchy. Edge servers cluster incoming updates to filter out anomalous ones, the cloud meta-aggregates multiple intermediate models to maintain a diverse pool of global models, and each client picks the model that best fits its local performance.

## Background

The paper points to the limitations of existing FL-based IoT intrusion detection: **communication bottlenecks, scalability limitations, and vulnerability to adversarial manipulation**. In particular, under extreme non-IID conditions, the FedAvg approach of forcing a single global model on every client inevitably degrades performance.

The authors claim the following contributions.

1. **Adaptive HFL**: an enhanced HFL architecture incorporating similarity-aware aggregation to improve robustness and scalability under extreme non-IID conditions
2. **Multi-model learning with dynamic client adaptation**: a multi-model aggregation strategy that generates multiple candidate models and enables dynamic client-side model selection based on local performance
3. **A robust and secure aggregation mechanism**: combining lightweight encryption with cluster-based statistical anomaly detection to mitigate attacks while keeping computation and communication overhead low

## Method

### Layer-by-Layer Design

- **Client layer**: each IoT client **selects the model with the lowest local loss** from the multiple global models it receives from the cloud. Rather than being forced to use a single global model, each client chooses the candidate that best fits its local data.
- **Edge layer**: receives client updates, performs **cosine-similarity-based clustering**, and excludes or isolates anomalous updates.
- **Cloud layer**: clusters and meta-aggregates the intermediate models coming from the edge, producing a **pool of multiple global models**.

### Aggregation Procedure

1. Client updates that selected the same base model are grouped together, and update deviations are clustered via cosine similarity.
2. For each valid cluster, the mean update is computed to produce an intermediate model (edge aggregation).
3. The cloud again applies similarity-based clustering to the edge intermediate models and averages within each cluster to refresh the global multi-model pool (cloud aggregation).

The approach therefore belongs to the family of **clustering-based multi-model weight/mean aggregation**, rather than FedAvg's single average. Because clustering doubles as a filter for anomalous updates, adversarial updates can be screened out to a fair degree without any heavyweight dedicated defense mechanism.

### Backbone

The backbone model is a customized multilayer perceptron (MLP).

## Experiments

- **Dataset**: IDSoT2024
- Evaluates improvements in robustness and scalability under extreme non-IID settings.

## Takeaways

- The key idea is abandoning FL's default assumption of "one global model" and tackling the non-IID problem with a **pool of multiple global models plus client-side selection**.
- Similarity-based clustering simultaneously handles the formation of aggregation groups and anomaly detection — a design that addresses personalization and robustness with a single mechanism.
- By keeping the three-tier client–edge–cloud structure of HFL intact while assigning each tier a distinct role (selection – filtering – meta-aggregation), the aggregation strategy is noticeably more sophisticated than in earlier HFL IDS papers.
