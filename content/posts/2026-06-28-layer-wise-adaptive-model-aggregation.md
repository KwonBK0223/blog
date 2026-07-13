---
title: "Layer-Wise Adaptive Model Aggregation for Scalable Federated Learning"
date: 2026-06-28
tags: ["Federated Learning", "Layer-wise Aggregation", "Communication Efficiency"]
categories: ["Paper Review"]
series: ["Federated Learning"]
math: true
summary: "레이어마다 client 간 model discrepancy가 다르게 증가한다는 관찰에 기반해, 레이어별로 aggregation 주기를 적응적으로 조절하는 FedLAMA를 제안한 AAAI 2023 논문이다."
---

> **Paper**: Layer-Wise Adaptive Model Aggregation for Scalable Federated Learning — S. Lee, T. Zhang, A. S. Avestimehr, in *Proc. AAAI Conference on Artificial Intelligence*, vol. 37, no. 7, pp. 8491–8499, 2023. [doi:10.1609/aaai.v37i7.26023](https://doi.org/10.1609/aaai.v37i7.26023) / [arXiv:2110.10302](https://arxiv.org/abs/2110.10302)

## TL;DR

클라이언트별로 중요한 레이어가 다르고, 레이어마다 client 간 drift 속도가 다르다는 관찰에서 출발한다. 모든 레이어를 같은 주기로 동기화하는 대신, **model discrepancy와 통신 비용을 함께 고려해 레이어별 aggregation 주기를 적응적으로 조절**하는 layer-wise adaptive aggregation(FedLAMA)을 제안한다.

## Background

### Model Discrepancy란?

각 클라이언트의 local model이 global model로부터 얼마나 멀어졌는지를 의미한다. 일반적인 FedAvg에서는 클라이언트들이 global model을 받은 뒤 각자 local SGD를 수행하는데, 데이터가 non-IID이거나 stochastic gradient noise가 있으면 client model들이 서로 다른 방향으로 이동한다.

핵심 관찰은 이 discrepancy가 **레이어마다 다르게 증가한다**는 것이다. 어떤 레이어는 클라이언트들 사이에서 빠르게 달라지고, 어떤 레이어는 별로 달라지지 않는다. 따라서 모든 레이어를 같은 주기로 동기화하는 것은 네트워크 대역폭 낭비다.

## Method

### Layer Prioritization

어떤 레이어를 자주 aggregate해야 하는지 우선순위를 정한다. 각 레이어 $l$에 대해 **layer-wise unit model discrepancy** $d_l$을 계산한다.

- $d_l$은 레이어 $l$에서 client model들과 global model의 차이, aggregation interval, 레이어 크기를 함께 반영하며, **unit communication cost당 해당 레이어를 동기화했을 때 줄일 수 있는 model discrepancy**로 정의된다.
- $d_l$이 크다: client 간 layer drift가 크고 동기화 효과가 크다 → 자주 aggregation한다.
- $d_l$이 작다: layer drift가 작아 동기화해도 얻는 것이 적다 → 덜 자주 aggregation한다.

### Layer-wise Aggregation Interval Adjustment

레이어마다 aggregation interval을 다르게 설정한다. 중요한 레이어는 기본 주기마다 자주 aggregate하고, 덜 중요한 레이어는 덜 자주 aggregate한다. $d_l$ 기준으로 레이어를 정렬한 뒤, model discrepancy 증가량과 communication cost 감소량을 비교하여 어떤 레이어의 interval을 늘릴지 결정한다.

### 전체 알고리즘

1. 모든 레이어의 aggregation interval을 초기화한다.
2. 클라이언트들이 local SGD를 수행한다.
3. 각 local step에서 레이어 $l$이 동기화 시점인지 확인한다.
4. 동기화 시점이면 해당 레이어만 aggregation한다.
5. 해당 레이어의 global average를 계산한다.
6. 클라이언트 local model의 해당 레이어를 aggregated layer로 교체한다.
7. 레이어가 동기화될 때 $d_l$을 업데이트한다.
8. 전체 모델이 한 번 이상 동기화되면 aggregation interval을 재조정한다.
9. 2번으로 돌아가 반복한다.

### Interval Increase Factor

덜 중요한 레이어의 aggregation interval을 얼마나 늘릴지 정하는 하이퍼파라미터로, 통신 절감과 정확도 사이의 트레이드오프를 조절한다.

## Takeaways

- "레이어마다 drift 속도가 다르다"는 관찰을 **통신 스케줄링 문제**로 번역한 논문이다. Partial FL이 '무엇을 보낼까'를 고민한다면, FedLAMA는 '얼마나 자주 보낼까'를 레이어 단위로 최적화한다.
- $d_l$을 unit communication cost당 discrepancy 감소량으로 정의한 것이 설계의 핵심으로, 효과 대비 비용이라는 기준을 명시적으로 수식화했다.
- 전체 모델 동기화를 유지하면서도 주기만 차등화하므로, 기존 FedAvg 파이프라인에 비교적 쉽게 얹을 수 있는 실용적인 방법이다.
