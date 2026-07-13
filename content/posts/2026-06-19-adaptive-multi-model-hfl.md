---
title: "Adaptive Multi-Model Hierarchical Federated Learning for Robust IoT Intrusion Detection"
date: 2026-06-19
tags: ["Federated Learning", "Hierarchical FL", "Intrusion Detection", "IoT", "Clustering"]
categories: ["Paper Review"]
series: ["Federated Learning Applications"]
math: true
summary: "클라이언트–엣지–클라우드 3계층에서 유사도 기반 클러스터링, 다중 모델 메타 집계, 클라이언트 측 동적 모델 선택을 결합한 적응형 다중 모델 HFL(A-MM-HFL) 프레임워크다."
---

> **Paper**: Adaptive Multi-Model Hierarchical Federated Learning for Robust IoT Intrusion Detection — S. Latif, D. Djenouri, *Sensors*, 2026.

## TL;DR

클라이언트–엣지–클라우드 3계층 구조에서 **유사도 기반 클러스터링, 다중 모델 집계, 동적 모델 선택**을 통합한 적응형 다중 모델 HFL(A-MM-HFL) 프레임워크를 제안한다. 엣지 서버에서 업데이트를 클러스터링해 이상 데이터를 제외하고, 클라우드에서 여러 중간 모델을 메타 집계해 다양한 전역 모델 풀을 유지한 뒤, 클라이언트가 로컬 성능에 따라 가장 적합한 모델을 선택한다.

## Background

기존 FL 기반 IoT 침입탐지의 한계로 **communication bottleneck, scalability limitation, adversarial manipulation에 대한 취약성**이 지적된다. 특히 극한의 non-IID 환경에서 단일 global model을 모든 클라이언트에게 강제하는 FedAvg 방식은 성능 저하를 피하기 어렵다.

저자들이 주장하는 기여는 다음과 같다.

1. **적응형 HFL**: 극한의 non-IID 환경에서 강건성과 확장성을 개선하기 위해 유사성 인식(similarity-aware) 집계를 통합한 향상된 HFL 구조 개발
2. **동적 클라이언트 적응을 통한 다중 모델 학습**: 여러 후보 모델을 생성하고 로컬 성능에 기반한 동적 클라이언트 측 모델 선택을 가능하게 하는 다중 모델 집계 전략 도입
3. **강건하고 안전한 집계 메커니즘**: 경량 암호화와 클러스터 기반 통계적 이상 탐지를 결합하여, 낮은 계산·통신 오버헤드를 유지하면서 공격을 완화

## Method

### 계층별 구조 설계

- **Client layer**: 각 IoT 클라이언트가 cloud에서 받은 여러 global model 중 **local loss가 가장 낮은 모델을 선택**한다. 하나의 global model을 무조건 쓰는 것이 아니라, 여러 후보 중 자신의 로컬 데이터에 가장 잘 맞는 모델을 고른다.
- **Edge layer**: 클라이언트 update를 받아 **cosine similarity 기반 clustering**을 수행하고, 이상 update를 제외하거나 분리한다.
- **Cloud layer**: 엣지에서 올라온 intermediate model들을 다시 clustering/meta-aggregation하여 **multiple global model pool**을 만든다.

### 집계 절차

1. 같은 base model을 선택한 클라이언트 update끼리 모은 뒤, update deviation을 cosine similarity로 clustering한다.
2. Valid cluster별로 평균 update를 계산해 intermediate model을 만든다 (edge aggregation).
3. Cloud는 edge intermediate model들을 다시 similarity-based clustering하고, cluster별 평균을 내서 global multi-model pool을 갱신한다 (cloud aggregation).

따라서 이 방식은 FedAvg의 단일 평균이 아니라 **clustering 기반 multi-model weight/mean aggregation** 계열이다. 클러스터링이 이상 update 필터 역할을 겸하기 때문에, 별도의 무거운 방어 메커니즘 없이도 adversarial update를 어느 정도 걸러낼 수 있다.

### Backbone

Backbone 모델은 customized multilayer perceptron(MLP)이다.

## Experiments

- **데이터셋**: IDSoT2024
- 극한 non-IID 설정에서 강건성과 확장성 개선을 평가한다.

## Takeaways

- "하나의 global model"이라는 FL의 기본 가정을 버리고, **다중 global model pool + 클라이언트 측 선택**으로 non-IID 문제에 접근한 점이 핵심이다.
- 유사도 기반 클러스터링이 집계 단위 구성과 이상 탐지를 동시에 담당하여, personalization과 robustness를 한 메커니즘으로 해결하려는 설계다.
- HFL의 3계층 구조(client–edge–cloud)를 그대로 유지하면서 각 계층에 서로 다른 역할(선택–필터링–메타 집계)을 부여했다는 점에서, 앞선 HFL IDS 논문들보다 집계 전략이 한층 정교하다.
