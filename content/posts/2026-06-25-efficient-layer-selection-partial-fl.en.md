---
title: "An Efficient Layer Selection Algorithm for Partial Federated Learning"
date: 2026-06-25
tags: ["Federated Learning", "Partial FL", "Layer Selection", "V2V"]
categories: ["Paper Review"]
series: ["Partial & Layer-wise Federated Learning"]
math: true
summary: "Proposes Partial Federated Learning for vehicular environments, exchanging only the most important layers over V2V links instead of the full model, combining Kalman-filter-based contact time prediction with CKA-similarity-based aggregation."
---

> **Paper**: An Efficient Layer Selection Algorithm for Partial Federated Learning — L. Pacheco, T. Braun, D. Rosário, E. Cerqueira, in *Proc. IEEE International Conference on Pervasive Computing and Communications Workshops (PerCom Workshops)*, pp. 172–177, 2024.

## TL;DR

A **Partial Federated Learning (PFL)** paper built on the observation that in vehicular environments, sharing the full model every round is unnecessary. Each vehicle selectively exchanges only the layers that mattered most during training, reducing communication cost without substantially degrading convergence time or model accuracy. Notably, aggregation is fully decentralized over V2V communication, with no central server.

## Background

The rationale for splitting the model into layers for exchange is clear: vehicles are constantly moving, communication windows are short, and the amount of information to exchange (the full model parameters) is simply too large. PFL mitigates this by transmitting only selected layers of the local model.

The main contributions are as follows.

1. An **in-vehicle local model aggregation mechanism** that removes the dependence on a centralized server
2. **V2V link contact time prediction** to optimize model transmission within the available communication window
3. Application of the **Centered Kernel Alignment (CKA) similarity metric** to preserve model personalization on heterogeneous datasets

## Method

### Important Layer Selection (Perturbation-based)

The most important layers are selected via perturbation.

1. Add small noise to layer $l$.
2. Measure the accuracy of the perturbed model.
3. Compare it against the original accuracy.
4. Layers with larger accuracy drops are judged more important.

Selected layers may additionally undergo pruning or quantization before transmission.

### Communication Window Prediction

- A **Kalman filter** predicts the V2V contact time. Note that the Kalman filter here is not a learning model but a tool for **communication window prediction** — what gets exchanged are the layers of the ML model, not the Kalman filter's transition matrix or gain.
- **Shannon capacity** is used to estimate link capacity, determining which subset of layers can be transmitted within the available contact time.

Together, these predictions reduce both communication cost and latency.

### Decentralized V2V Aggregation (CKA-based)

Each vehicle transmits layers to other vehicles rather than to a central server. The aggregation procedure is as follows.

1. Vehicle A comes into contact with vehicle B.
2. They exchange only the layer subsets that can be transmitted.
3. Each vehicle computes the **CKA similarity** between the received layers and its own local layers.
4. If the similarity exceeds a threshold, aggregation is performed.
5. Older (higher-age) models receive lower weights, while models with higher CKA similarity receive higher weights.
6. The final update takes the following form.

$$W_{\text{new}} = \alpha \cdot W_{\text{local}} + (1 - \alpha) \cdot W_{\text{received}}^{\text{weighted}}$$

Thanks to the CKA filtering, even in heterogeneous data environments, a vehicle only accepts models trained in a direction similar to its own, preserving personalization.

## Experiments

- **Datasets**: CIFAR-10, CIFAR-100, MNIST
- It is somewhat unfortunate that a paper about vehicular scenarios evaluates on image classification benchmarks (likely a limitation of a workshop paper).

## Takeaways

- The core framing of this paper is jointly optimizing "what to exchange" (important layers) and "when/how much can be exchanged" (contact time and link capacity).
- Serverless V2V decentralized aggregation plus CKA-based selective acceptance is a design well suited to vehicular environments with unreliable infrastructure.
- Measuring layer importance via perturbation is simple, but comes with the trade-off of extra local evaluation cost.
