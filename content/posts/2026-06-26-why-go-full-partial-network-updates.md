---
title: "Why Go Full? Elevating Federated Learning Through Partial Network Updates"
date: 2026-06-26
tags: ["Federated Learning", "Partial FL", "Layer Mismatch", "NeurIPS"]
categories: ["Paper Review"]
series: ["Federated Learning"]
math: true
summary: "Full-network aggregation이 유발하는 layer mismatch 문제를 규명하고, 라운드마다 일부 레이어만 순차적으로 학습·집계하는 FedPart로 수렴 속도와 성능을 동시에 개선한 NeurIPS 2024 논문이다."
---

> **Paper**: Why Go Full? Elevating Federated Learning Through Partial Network Updates — H. Wang, X. Liu, J. Niu, W. Guo, S. Tang, in *Advances in Neural Information Processing Systems (NeurIPS)*, 2024. [arXiv:2410.11559](https://arxiv.org/abs/2410.11559)

## TL;DR

기존 FL은 매 라운드 전체 파라미터를 업데이트하고 평균하는데, 이 과정에서 **layer mismatch**라는 문제가 발생함을 규명한 논문이다. 이를 완화하기 위해 각 communication round마다 단일 레이어 또는 소수의 레이어만 업데이트하는 **FedPart**를 제안하고, non-convex setting에서의 수렴 속도를 이론적으로 분석해 full-network update 대비 장점을 입증했다.

## Background

### Layer Mismatch란?

Full-network aggregation 이후, 평균된 global model의 레이어들이 서로 충분히 align되지 않아 다음 local training에서 gradient/update step size가 다시 커지는 현상이다. 실제로 global update가 이루어진 직후 update step size가 급증하는 것이 관찰되는데, 이는 평균화가 각 레이어 간의 협력 관계를 깨뜨리기 때문이다. 이 mismatch는 파라미터 평균화가 일어날 때마다 반복되므로, 모델의 수렴 속도를 늦추고 전반적인 성능을 저하시킨다.

파라미터 일부만 업데이트하는 partial update로 가져가면 이 문제를 상당 부분 해결할 수 있다는 것이 이 논문의 핵심 관찰이다.

## Method

### FedPart

- 각 communication round마다 **단일 레이어 또는 소수의 레이어만** 학습·집계한다. 한 레이어(또는 few layers)를 선택해 local training → server aggregation → broadcast를 수행하는 것이 하나의 communication round다.
- 구현상으로는 trainable/non-trainable 파라미터를 구분하고, loss에 mask를 적용해 선택된 부분만 global model에 반영한다.

### Sequential Updating과 Multi-Round Cycle Training

- 레이어는 **shallow에서 deep 순서로 순차적으로(cyclically)** 학습된다. 하나의 레이어를 2 round 동안 학습한 뒤 다음 레이어로 넘어간다.
- Cycle이 반복되면서 각 레이어가 순서대로 trainable 상태가 되며, 전체 네트워크가 여러 cycle에 걸쳐 고루 학습된다.
- 순서에 따른 성능은 **sequential(shallow→deep) > reverse > random**으로, 얕은 레이어부터 안정화시키는 것이 가장 효과적이다.

### 이론 분석

Non-convex setting에서 FedPart의 수렴 속도를 이론적으로 분석하여, full-network update 대비 장점을 보였다.

## Experiments

광범위한 실험 설정에서 FedPart가 full-network update 대비 수렴 속도, 최종 정확도, 통신·계산 효율 측면에서 개선을 보임을 확인했다. 이론적 근거와 실험 결과가 함께 제시된다.

## Takeaways

- Partial update를 단순히 **통신량 절감 수단**이 아니라, full aggregation이 유발하는 **layer mismatch를 치유하는 학습 전략**으로 재해석했다는 점이 이 논문의 가장 큰 기여다.
- "매 라운드 한 레이어씩, 얕은 층부터 순서대로"라는 매우 단순한 규칙으로 효과를 얻는다는 점에서 실용성이 높다.
- 수렴 이론까지 갖춘 NeurIPS 논문으로, layer-wise/partial FL 계열 연구의 이론적 근거로 자주 인용될 만한 위치에 있다.
