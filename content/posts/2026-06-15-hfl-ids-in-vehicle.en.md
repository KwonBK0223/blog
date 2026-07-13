---
title: "Hierarchical Federated Learning-Based Intrusion Detection for In-Vehicle Networks"
date: 2026-06-15
tags: ["Federated Learning", "Hierarchical FL", "Intrusion Detection", "Vehicular Networks"]
categories: ["Paper Review"]
series: ["Federated Learning Applications"]
math: true
summary: "The first paper to apply hierarchical federated learning (HFL) to in-vehicle network (CAN) intrusion detection, using multiple edge aggregators to mitigate the single point of failure and improve scalability."
---

> **Paper**: Hierarchical Federated Learning-Based Intrusion Detection for In-Vehicle Networks — M. Althunayyan, A. Javed, O. Rana, T. Spyridopoulos, *Future Internet*, vol. 16, no. 12, art. 451, 2024. [doi:10.3390/fi16120451](https://doi.org/10.3390/fi16120451)

## TL;DR

This work applies Hierarchical Federated Learning (HFL) to an intrusion detection system (IDS) for in-vehicle networks (CAN). The authors claim it is the first study to apply HFL to an IDS. The core idea is applying FedAvg in two stages — client → edge and edge → cloud — while the IDS backbone reuses an existing multi-stage ANN + LSTM AE architecture.

## Background

In conventional flat FL, every vehicle client communicates directly with a single central server. This creates two problems.

- As the number of clients grows, the central server becomes a **performance bottleneck**.
- The central server is a **single point of failure**, undermining robustness and scalability.

To address this, the authors introduce **multiple edge aggregators** alongside the central aggregator. Edge aggregators are edge servers deployed across regions or client groups that first aggregate the local model updates of their attached vehicle clients; the central aggregator is a cloud aggregator that re-aggregates the edge models into a global model for the whole network. This structure mitigates the single point of failure, improves scalability, and distributes the computational load. It also allows more clients to join the framework, with cost benefits as a bonus.

For reference, prior vehicular IDS work has used GRU, LSTM, GCN, ConvLSTM, and CNN backbones.

## Method

The experimental setup consists of 1 main server, 2 edge servers, and 5 clients per edge. HFL operates as follows.

1. The main server sends the global model to the edges.
2. Each edge selects suitable vehicles. To handle the variability of CAN bus data across vehicle makes and models, clients are selected based on CAN bus data similarity, such as manufacturer or model.
3. The selected vehicles receive the global model.
4. Each vehicle performs a local update.
5. Local updates are sent to the edge.
6. The edge aggregates them.
7. The aggregated model is sent to the main server.

FedAvg is used for aggregation at every level of the hierarchy.

### IDS Classifier Architecture

The IDS classifier deployed on each vehicle operates in two stages.

- **Stage 1**: an Artificial Neural Network (ANN) detects known attacks.
- **Stage 2**: only the traffic classified as normal in Stage 1 goes through an LSTM Autoencoder for anomaly detection, catching unknown attacks.

The parameters shared through FL cover both the ANN and the LSTM AE, and the entire classifier resides on each vehicle.

### Initial Global Model

The initial global model is prepared by pre-training on 0.8% (under 1%) of the original dataset, selected via cluster-based sampling; this data is never distributed to clients afterward. Class imbalance is mitigated with SMOTE before pre-training.

## Experiments

- **Datasets**: the benchmark car hacking dataset (main) and the CAR Hacking: Attack & Defense Challenge 2020 dataset (external).
- **Non-IID setup**: data is distributed using a Dirichlet distribution — the standard way to construct non-IID conditions in FL experiments. Other options include quantity skew (unequal data volumes per client), vehicle/scenario splits (partitioning by vehicle, attack scenario, or dataset source), and mobility/RSU splits (partitioning by location, connectivity, or RSU).
- **Scenarios**: 2 datasets × 3 non-IID levels × 2 model types × 2 edge configurations, for 24 evaluation scenarios in total.
- **Evaluation**: performed on the client side. This allows the model to be evaluated on a larger pool of data, yielding more realistic results.

One caveat: the original datasets are not inherently hierarchical — the hierarchy is an experimental assumption imposed by the authors.

## Takeaways

- As an early application of HFL to vehicular IDS, the contribution is **architectural** rather than algorithmic novelty.
- Methodologically, it is FedAvg applied in two stages (client → edge, edge → cloud), with the existing multi-stage ANN + LSTM AE reused as the IDS backbone.
- It demonstrates that the multi-edge-aggregator structure delivers practical benefits: mitigating the single point of failure, improving scalability, and distributing computational load.
