---
title: "Secure Hierarchical Federated Learning in Vehicular Networks Using Dynamic Client Selection and Anomaly Detection"
date: 2026-06-17
tags: ["Federated Learning", "Hierarchical FL", "Client Selection", "Anomaly Detection"]
categories: ["Paper Review"]
series: ["Hierarchical Federated Learning"]
math: true
summary: "This work combines reliability-score-based dynamic client selection with cosine-similarity-based anomalous update detection in cluster-based vehicular HFL, aiming for secure training even when malicious vehicles are present."
---

> **Paper**: Secure Hierarchical Federated Learning in Vehicular Networks Using Dynamic Client Selection and Anomaly Detection — M. S. HaghighiFard, S. Coleri, arXiv preprint, May 2024. [arXiv:2405.17497](https://arxiv.org/abs/2405.17497)

## TL;DR

In a Cluster-based HFL (CbHFL) setup where vehicles form clusters to run hierarchical federated learning, this paper combines **dynamic client selection based on a reliability score** with **cosine-similarity-based anomaly detection**. Notably, anomaly detection here does not mean detecting malicious network packets — it means flagging suspicious model updates within the FL training pipeline.

## Background

As many HFL papers repeatedly point out, the **limited communication resources** of vehicular networks are the primary motivation for using HFL. On top of that, this paper focuses on how to guarantee the integrity of the training process when malicious participants may be present, and introduces **client selection** as the mechanism for doing so.

The authors claim the following contributions.

- Vehicles form clusters to implement HFL, and a reliability score is used to identify and prioritize well-behaving vehicles. Vehicles exhibiting malicious or suspicious behavior are temporarily excluded, with their behavior re-evaluated after a set period.
- A novel dynamic client selection methodology that integrates historical accuracy, contribution frequency, and anomaly records as its main metrics.

## Method

### Network Architecture and Cluster-based HFL

The network consists of cluster heads (CH), cluster members (CM), and the evolving packet core (EPC).

- CH and CM roles change over time.
- Clustering is performed based on the similarity of local model parameters and the average relative speed.
- CMs act as clients and send their parameters to the CH; the CH acts as an edge node, aggregating the parameters it receives and forwarding them to the EPC.
- Local updates are performed with SGD.

### Reliability Score

Each vehicle's reliability score is a **weighted sum** of three metrics.

- **Historical Accuracy**: the average performance on a validation set of the updates the vehicle submitted in past rounds. Vehicles that performed well are assumed to remain trustworthy in subsequent rounds.
- **Contribution Frequency**: how often the vehicle has participated in FL rounds so far. Vehicles that participate frequently and consistently likely have good communication links and hardware, which helps reduce packet loss.
- **Anomaly Record**: how many times the vehicle has been flagged for anomalous updates. A high value suggests the vehicle may be malicious or unstable, and counts against its score.

### Dynamic Client Selection Algorithm

1. The CH inspects the VIB (vehicle information base) of each CM.
2. Each CM's reliability score is initialized/updated.
3. CMs are sorted in descending order of reliability score.
4. A predefined fraction of top clients is selected (75% in the experiments).
5. Vehicles whose block flag is set and whose unblock time has not yet elapsed are excluded (unblock time 5).
6. The updates of selected vehicles are checked with cosine similarity.
7. If the update is normal, the vehicle's participation and accuracy records are updated.
8. If the update is anomalous, the block flag is set and the anomaly count is incremented.
9. The new reliability score is stored in the VIB.

### Anomaly Detection

The cosine similarity between the current update and the previous update is compared; if the change is excessively large, the update is judged to be malicious or anomalous. In other words, the target of anomaly detection here is not attack traffic in the vehicular network but **poisoned model updates inside the FL pipeline**.

## Experiments

- **Dataset**: MNIST (assuming a non-IID distribution)
- **Implementation**: PyTorch, with SUMO as the mobility simulator and Kafka for streaming
- **Environment**: 1 km² area, 25 vehicles, speeds of 10–35 m/s, 100 m transmission range
- **Communication**: IEEE 802.11p for V2V, 5G NR for V2I
- **Attack setup**: 20% of all vehicles are attackers. Attacks inject fake additive Gaussian noise into model updates; two scenarios are evaluated — a single attack at round 10, and a sustained attack from round 10 onward.
- **Evaluation**: accuracy and convergence time, measured at the EPC.

However, the paper does not specify whether the deep learning model is an MLP or a CNN, the exact aggregation scheme, or how the validation set is constructed.

## Takeaways

- This work brings a security perspective into HFL; its core idea is managing "who gets to participate in training" through an explicit reliability score.
- Because anomaly detection operates at the model-update level rather than the traffic level, the defense target is fundamentally different from that of IDS papers.
- Simulating a vehicular environment while using MNIST as the data, and omitting the model architecture and aggregation details, are notable weaknesses.
