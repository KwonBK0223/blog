---
title: "Layer-Wise Adaptive Model Aggregation for Scalable Federated Learning"
date: 2026-06-28
tags: ["Federated Learning", "Layer-wise Aggregation", "Communication Efficiency"]
categories: ["Paper Review"]
series: ["Partial & Layer-wise Federated Learning"]
math: true
summary: "An AAAI 2023 paper proposing FedLAMA, which adaptively adjusts the aggregation interval per layer, based on the observation that inter-client model discrepancy grows at different rates across layers."
---

> **Paper**: Layer-Wise Adaptive Model Aggregation for Scalable Federated Learning — S. Lee, T. Zhang, A. S. Avestimehr, in *Proc. AAAI Conference on Artificial Intelligence*, vol. 37, no. 7, pp. 8491–8499, 2023. [doi:10.1609/aaai.v37i7.26023](https://doi.org/10.1609/aaai.v37i7.26023) / [arXiv:2110.10302](https://arxiv.org/abs/2110.10302)

## TL;DR

The paper starts from the observation that different layers matter to different clients, and that inter-client drift accumulates at different rates across layers. Instead of synchronizing every layer at the same interval, it proposes FedLAMA, a layer-wise adaptive aggregation scheme that **adaptively adjusts each layer's aggregation interval by jointly considering model discrepancy and communication cost**.

## Background

### What Is Model Discrepancy?

It measures how far each client's local model has drifted from the global model. In standard FedAvg, clients receive the global model and then run local SGD independently; with non-IID data or stochastic gradient noise, the client models move in different directions.

The key observation is that this discrepancy **grows at different rates across layers**. Some layers diverge quickly across clients, while others barely change. Synchronizing all layers at the same interval is therefore a waste of network bandwidth.

## Method

### Layer Prioritization

The method prioritizes which layers should be aggregated frequently. For each layer $l$, it computes the **layer-wise unit model discrepancy** $d_l$.

- $d_l$ jointly reflects the difference between client models and the global model at layer $l$, the aggregation interval, and the layer size; it is defined as **the model discrepancy reduction achievable by synchronizing that layer per unit communication cost**.
- Large $d_l$: the layer drifts substantially across clients and synchronization pays off → aggregate frequently.
- Small $d_l$: the layer drifts little, so synchronizing it gains little → aggregate less frequently.

### Layer-wise Aggregation Interval Adjustment

Each layer gets its own aggregation interval. Important layers are aggregated at the base interval; less important layers less often. Layers are sorted by $d_l$, and the increase in model discrepancy is weighed against the reduction in communication cost to decide which layers' intervals to extend.

### Full Algorithm

1. Initialize the aggregation interval of every layer.
2. Clients perform local SGD.
3. At each local step, check whether layer $l$ is due for synchronization.
4. If so, aggregate only that layer.
5. Compute the global average for that layer.
6. Replace the corresponding layer of each client's local model with the aggregated layer.
7. Update $d_l$ whenever a layer is synchronized.
8. Once the whole model has been synchronized at least once, re-adjust the aggregation intervals.
9. Return to step 2 and repeat.

### Interval Increase Factor

A hyperparameter controlling how much to extend the aggregation interval of less important layers, tuning the trade-off between communication savings and accuracy.

## Takeaways

- This paper translates the observation that "layers drift at different rates" into a **communication scheduling problem**. Where partial FL asks "what to send," FedLAMA optimizes "how often to send" at the layer level.
- Defining $d_l$ as discrepancy reduction per unit communication cost is the heart of the design — it explicitly formalizes a benefit-per-cost criterion.
- Because it keeps full-model synchronization and only differentiates the intervals, it is a practical method that can be layered onto an existing FedAvg pipeline with relatively little effort.
