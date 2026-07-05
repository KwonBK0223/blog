---
title: "Hierarchical Federated Learning-Based Intrusion Detection for In-Vehicle Networks"
date: 2026-06-15
tags: ["Federated Learning", "Hierarchical FL", "Intrusion Detection", "Vehicular Networks"]
categories: ["Paper Review"]
series: ["HFL 기반 차량 침입탐지"]
math: true
summary: "차량 내 네트워크(CAN) 침입탐지에 계층형 연합학습(HFL)을 처음 적용한 논문으로, 다중 edge aggregator를 두어 단일 실패 지점을 완화하고 확장성을 개선한다."
---

> **Paper**: Hierarchical Federated Learning-Based Intrusion Detection for In-Vehicle Networks — M. Althunayyan, A. Javed, O. Rana, T. Spyridopoulos, *Future Internet*, vol. 16, no. 12, art. 451, 2024. [doi:10.3390/fi16120451](https://doi.org/10.3390/fi16120451)

## TL;DR

차량 내 네트워크(CAN) 침입탐지 시스템(IDS)에 계층형 연합학습(Hierarchical Federated Learning, HFL)을 적용한 연구다. 저자들은 이것이 IDS에 HFL을 적용한 최초의 연구라고 주장한다. 핵심은 FedAvg를 client → edge, edge → cloud의 두 단계로 적용한 것이며, IDS backbone은 기존의 multi-stage ANN + LSTM AE 구조를 가져왔다.

## Background

기존의 flat FL에서는 모든 차량 클라이언트가 하나의 중앙 서버와 직접 통신한다. 이 구조는 두 가지 문제를 야기한다.

- 클라이언트 수가 늘어나면 중앙 서버에 **성능 병목**이 발생한다.
- 중앙 서버가 **단일 실패 지점(single point of failure)**이 되어 강건성과 확장성을 저해한다.

이를 해결하기 위해 저자들은 중앙 aggregator와 함께 **다중 edge aggregator**를 도입한다. Edge aggregator는 여러 지역 또는 클라이언트 그룹에 배치된 edge server들이 각자 연결된 차량 클라이언트들의 로컬 모델 업데이트를 먼저 집계하는 중간 aggregator이고, central aggregator는 edge aggregator들이 만든 edge model을 다시 집계해 전체 네트워크의 global model을 만드는 cloud aggregator다. 이 구조는 단일 실패 지점의 위험을 완화하고, 확장성을 개선하며, 계산 부하를 분산시킨다. 또한 더 많은 클라이언트가 프레임워크에 참여할 수 있게 하여 비용 측면의 이점도 있다.

참고로 기존 차량 IDS 연구들의 backbone 모델로는 GRU, LSTM, GCN, ConvLSTM, CNN 등이 사용되어 왔다.

## Method

실험 구성은 1개의 main server, 2개의 edge server, edge당 5개의 클라이언트다. HFL의 동작 방식은 다음과 같다.

1. Main server가 global model을 edge로 전송한다.
2. Edge에서 적합한 차량을 선택한다. 다양한 차량 제조사와 모델에 걸친 CAN 버스 데이터의 변동성을 처리하기 위해, 제조사나 모델과 같은 CAN 버스 데이터의 유사성을 기반으로 클라이언트를 선택한다.
3. 선택된 차량이 global model을 받는다.
4. 로컬 업데이트를 수행한다.
5. 로컬 업데이트를 edge로 전송한다.
6. Edge에서 집계한다.
7. 집계된 모델을 main server로 전송한다.

집계 방식은 계층에 상관없이 FedAvg를 사용한다.

### IDS Classifier 구조

각 차량에 탑재되는 IDS classifier는 두 단계로 동작한다.

- **Stage 1**: Artificial Neural Network(ANN)으로 이미 알려진 공격을 탐지한다.
- **Stage 2**: Stage 1에서 normal로 분류된 트래픽만 LSTM Autoencoder로 anomaly detection을 수행하여, 알려지지 않은 공격을 잡는다.

FL에서 공유되는 파라미터는 ANN과 LSTM AE 모두를 포함하며, 이 classifier 전체가 각 차량에 들어간다.

### Initial Global Model

초기 global model은 원본 데이터셋의 0.8%(1% 미만)를 사전 훈련에 사용해 준비한다. 군집 기반 샘플링(cluster-based sampling)으로 이 데이터를 선택하며, 이 데이터는 이후 클라이언트에 배분되지 않는다. 클래스 불균형은 SMOTE로 완화한 뒤 pre-training을 수행한다.

## Experiments

- **데이터셋**: benchmark car hacking dataset(main)과 CAR Hacking: Attack & Defense Challenge 2020 dataset(external) 두 가지를 사용한다.
- **Non-IID 설정**: 디리클레(Dirichlet) 분포 기반 데이터 분배를 사용한다. 이는 FL 실험에서 non-IID 환경을 구성하는 표준적인 방식으로, 이외에도 quantity-skew(클라이언트별 데이터 양 차등 분배), vehicle/scenario split(차량·공격 시나리오·데이터셋 소스 기준 분리), mobility/RSU split(위치·연결성·RSU 기준 분리) 등의 방식이 있다.
- **시나리오**: 2개 데이터셋 × 3가지 non-IID 수준 × 2가지 모델 타입 × 2개 edge 구성으로 총 24개 시나리오를 평가한다.
- **평가 방식**: 평가는 client side에서 진행한다. 더 큰 데이터 집합에서 모델을 평가할 수 있게 하여 더 현실적인 평가 결과를 도출할 수 있다는 이점이 있다.

한 가지 유의할 점은 원본 데이터셋 자체는 계층적이지 않으며, 계층 구조는 저자들이 실험을 위해 설정한 가정이라는 것이다.

## Takeaways

- HFL을 차량 IDS에 적용한 초기 연구로서, 알고리즘 자체의 새로움보다는 **구조적 기여**가 핵심이다.
- 방법론적으로는 FedAvg를 두 단계(client → edge, edge → cloud)로 적용한 것이고, IDS backbone은 기존의 multi-stage ANN + LSTM AE를 그대로 활용했다.
- 다중 edge aggregator 구조가 단일 실패 지점 완화, 확장성 개선, 계산 부하 분산이라는 실용적 이점을 제공함을 보여준다.
