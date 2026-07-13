---
title: "An Efficient Layer Selection Algorithm for Partial Federated Learning"
date: 2026-06-25
tags: ["Federated Learning", "Partial FL", "Layer Selection", "V2V"]
categories: ["Paper Review"]
series: ["Federated Learning"]
math: true
summary: "차량 환경에서 전체 모델 대신 중요한 레이어만 선택적으로 V2V로 교환하는 Partial Federated Learning을 제안하며, 칼만 필터 기반 접촉 시간 예측과 CKA 유사도 기반 집계를 결합한다."
---

> **Paper**: An Efficient Layer Selection Algorithm for Partial Federated Learning — L. Pacheco, T. Braun, D. Rosário, E. Cerqueira, in *Proc. IEEE International Conference on Pervasive Computing and Communications Workshops (PerCom Workshops)*, pp. 172–177, 2024.

## TL;DR

차량 환경에서는 전체 모델을 매번 공유할 필요가 없다는 관찰에서 출발한 **Partial Federated Learning(PFL)** 연구다. 각 차량이 훈련 중 가장 중요했던 레이어만 선택적으로 교환하여, 수렴 시간과 모델 정확도를 크게 저하시키지 않으면서 통신 비용을 줄인다. 특히 중앙 서버 없이 V2V 통신만으로 분산 집계를 수행한다는 점이 특징이다.

## Background

모델을 레이어 단위로 쪼개서 교환하는 이유는 명확하다. 차량은 계속 이동하고 통신 윈도우가 짧은데, 교환할 정보(전체 모델 파라미터)가 너무 많아 비용이 크기 때문이다. PFL은 로컬 모델의 선택된 레이어만 전송함으로써 이 문제를 완화한다.

주요 기여는 다음과 같다.

1. 중앙 집중식 서버에 대한 의존성을 없애는 **차량 내 로컬 모델 집계 메커니즘** 구현
2. 가용 통신 윈도우 내에서 모델 전송을 최적화하기 위한 **V2V 링크 접촉 시간(contact time) 예측** 도입
3. Heterogeneous dataset에서 모델 개인화를 보장하기 위한 **Centered Kernel Alignment(CKA) 유사성 메트릭** 적용

## Method

### 중요 레이어 선택 (Perturbation 기반)

가장 중요한 레이어를 고르는 방법은 perturbation 기반이다.

1. 레이어 $l$에 작은 noise를 추가한다.
2. Perturbed model의 accuracy를 측정한다.
3. 원래 accuracy와 비교한다.
4. Accuracy drop이 큰 레이어일수록 중요한 레이어로 판정한다.

선택된 레이어는 전송 전에 pruning이나 양자화를 거치기도 한다.

### 통신 윈도우 예측

- **칼만 필터(Kalman Filter)**로 V2V contact time을 예측한다. 여기서 칼만 필터는 학습 모델이 아니라 **통신 윈도우 예측용**이다. 즉 교환 대상은 ML 모델의 레이어이며, 칼만 필터의 transition matrix나 gain을 교환하는 것이 아니다.
- **Shannon capacity**로 링크 용량을 추정하여, 주어진 접촉 시간 내에 전송 가능한 레이어 subset을 결정한다.

이 예측을 통해 통신 비용과 latency를 함께 줄인다.

### V2V 분산 집계 (CKA 기반)

각 차량은 중앙 서버가 아니라 다른 차량에게 레이어를 전송한다. 집계 절차는 다음과 같다.

1. 차량 A가 차량 B와 contact한다.
2. 서로 전송 가능한 layer subset만 교환한다.
3. 수신한 레이어와 자신의 로컬 레이어 간 **CKA similarity**를 계산한다.
4. Similarity가 threshold보다 높으면 aggregation을 수행한다.
5. 오래된(age가 큰) 모델에는 낮은 weight를, CKA similarity가 높은 모델에는 높은 weight를 부여한다.
6. 최종 갱신은 다음과 같은 형태다.

$$W_{\text{new}} = \alpha \cdot W_{\text{local}} + (1 - \alpha) \cdot W_{\text{received}}^{\text{weighted}}$$

CKA 필터링 덕분에 heterogeneous한 데이터 환경에서도 자신과 유사한 방향으로 학습된 모델만 받아들여 개인화를 유지할 수 있다.

## Experiments

- **데이터셋**: CIFAR-10, CIFAR-100, MNIST
- 차량 시나리오를 다루는 논문임에도 평가 데이터가 이미지 분류 벤치마크라는 점은 다소 아쉬운 부분이다 (워크샵 논문의 한계로 보인다).

## Takeaways

- "무엇을 교환할 것인가(중요 레이어)"와 "언제/얼마나 교환할 수 있는가(접촉 시간·링크 용량)"를 함께 최적화한다는 프레이밍이 이 논문의 핵심이다.
- 중앙 서버 없는 V2V 분산 집계 + CKA 기반 선택적 수용은 인프라가 불안정한 차량 환경에 잘 맞는 설계다.
- 레이어 중요도를 perturbation으로 측정하는 방식은 단순하지만, 로컬에서 추가 평가 비용이 발생한다는 트레이드오프가 있다.
