---
title: "Federated Block Coordinate Descent Scheme for Learning Global and Personalized Models"
date: 2026-06-27
tags: ["Federated Learning", "Block Coordinate Descent", "Personalization", "Asynchronous FL"]
categories: ["Paper Review"]
series: ["Partial & Layer-wise FL"]
math: true
summary: "FL objective를 block coordinate descent로 풀 수 있음을 보인 이론 논문으로, 계층적 통신 구조에서 개인화 모델과 비동기 cloud 업데이트를 동시에 달성하는 FedBCD를 제안한다."
---

> **Paper**: Federated Block Coordinate Descent Scheme for Learning Global and Personalized Models — R. Wu, A. Scaglione, H.-T. Wai, N. Karakoc, K. Hreinsson, W.-K. Ma, in *Proc. AAAI Conference on Artificial Intelligence*, vol. 35, pp. 10355–10362, 2021. [arXiv:2012.13900](https://arxiv.org/abs/2012.13900)

## TL;DR

모델을 **block 단위로 쪼개서 업데이트할 수 있다는 이론적 근거**를 제공하는 논문이다. Classic block coordinate descent(BCD)에서 출발하여, 계층적 통신 아키텍처 위에서 global model과 personalized model을 동시에 학습하는 FedBCD를 제안하고, sublinear 수렴 속도와 비동기 업데이트의 지연 분석까지 제공한다.

## Background

논문은 두 가지 근본적인 질문을 던진다.

1. 모든 곳에서 **동일한 모델**을 유지하는 것이 효율적인가?
2. **Synchronous** cloud 모델 업데이트가 효과적인가?

기존 FedAvg류는 단일 global model을 동기식으로 갱신하는데, 이 논문은 개인화와 비동기성을 모두 허용하는 최적화 관점의 프레임워크를 제시한다.

주요 기여는 다음과 같다.

1. 연합학습을 위한 맞춤형 **계층적 통신 아키텍처** (edge device – connected cloud server – cloud-server consensus 구조로, 넓은 의미의 HFL에 해당)
2. **Lightweight block coordinate descent 통신 프로토콜**: 모델 개인화와 cloud server 비동기 업데이트를 달성
3. **Sublinear 수렴 속도** 증명
4. 비동기 방식의 효능을 입증하기 위한 **지연(delay) 분석**: edge device 메시지 도착 시간의 분포로부터 런타임을 추정하고, 각 업데이트에 관여하는 클라이언트·서버 수와 런타임을 연결 (당시 기준 새로운 분석 방식)
5. 표준 ML 태스크에 대한 실험

## Method

### Edge Device 업데이트

- 각 edge device는 로컬 데이터에 맞는 **personalized model** $x_i$를 유지·업데이트한다.
- 업데이트는 accelerated stochastic projected gradient로 수행한다.
- Cloud model과 너무 멀어지지 않도록 **quadratic penalty**가 적용된다. 즉 개인화를 허용하되 global consensus에 묶어두는 구조다.

### Cloud Server 업데이트

Cloud server들은 edge device들이 보낸 personalized model $x_i$들을 반영해 global model $z$를 갱신한다. 두 가지 모드가 있다.

- **Sync-cloud**: 모든 cloud server가 edge update를 수신할 때까지 기다린 뒤, coordinator가 전체 정보를 모아 하나의 global model $z$를 업데이트하고 모든 cloud server에 broadcast한다. FedAvg와 유사한 구조다.
- **Async-cloud**: 먼저 준비된 일부 cloud server만 먼저 업데이트한다. 일부 cloud server가 edge update를 수신하면, coordinator가 먼저 도착한 $B$개의 cloud server만 사용해 이들 간 model mixing을 수행하고 해당 server들의 $z_n$만 업데이트한다. 나머지는 다음 기회에 갱신된다.

### Aggregation의 성격

집계는 완전한 FedAvg가 아니다. FedAvg식 단순 평균보다는 **penalty 기반 optimization-based cloud update**에 가깝다. 전체 문제를 BCD 관점에서 블록(개인화 모델 블록, global 모델 블록)별로 번갈아 최적화하는 구조다.

## Experiments

표준 ML 벤치마크에서 sync/async 설정의 수렴과 런타임을 검증하고, 지연 분석과 실험 결과가 일치함을 보인다.

## Takeaways

- 이 논문의 위치는 방법론적 신기법보다 **"FL objective를 block coordinate descent로 풀 수 있다"는 이론적 기반**에 있다. 이후의 block-wise/layer-wise partial update 연구들의 수학적 뿌리에 해당한다.
- Personalization을 penalty로, 비동기성을 BCD의 블록 선택으로 자연스럽게 흡수한 통합 프레임워크라는 점이 우아하다.
- Async-cloud의 지연 분석은 시스템 관점(도착 시간 분포)과 최적화 관점(수렴)을 연결한 초기 시도로 의미가 있다.
