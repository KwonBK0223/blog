---
title: "Secure Hierarchical Federated Learning in Vehicular Networks Using Dynamic Client Selection and Anomaly Detection"
date: 2026-06-17
tags: ["Federated Learning", "Hierarchical FL", "Client Selection", "Anomaly Detection"]
categories: ["Paper Review"]
series: ["Hierarchical Federated Learning"]
math: true
summary: "차량 클러스터 기반 HFL에서 신뢰성 점수 기반 동적 클라이언트 선택과 cosine similarity 기반 이상 업데이트 탐지를 결합하여, 악성 차량이 섞인 환경에서도 안전한 학습을 달성하려는 연구다."
---

> **Paper**: Secure Hierarchical Federated Learning in Vehicular Networks Using Dynamic Client Selection and Anomaly Detection — M. S. HaghighiFard, S. Coleri, arXiv preprint, May 2024. [arXiv:2405.17497](https://arxiv.org/abs/2405.17497)

## TL;DR

차량들이 클러스터를 형성해 HFL을 수행하는 Cluster-based HFL(CbHFL) 구조에서, **신뢰성 점수(reliability score) 기반 동적 클라이언트 선택**과 **cosine similarity 기반 anomaly detection**을 결합한 연구다. 여기서 anomaly detection은 네트워크 공격 패킷을 탐지하는 것이 아니라, FL 학습 과정에서 이상한 model update를 탐지하는 것이라는 점이 특징이다.

## Background

여러 HFL 논문에서 반복적으로 언급되듯, 차량 네트워크의 **제한된 통신 자원(limited communication resources)**이 HFL을 사용하는 핵심 이유다. 여기에 더해 이 논문은 악의적인 참여자가 존재할 수 있는 환경에서 학습 과정의 무결성을 어떻게 보장할 것인가에 초점을 맞추고, 그 수단으로 **client selection** 개념을 도입한다.

저자들이 주장하는 기여는 다음과 같다.

- 차량들이 클러스터를 형성하여 HFL을 구현하고, 신뢰성 점수를 사용해 잘 행동하는(well-behaving) 차량을 식별·우선순위화한다. 악의적이거나 의심스러운 행동을 보이는 차량은 일시적으로 제외되고, 일정 시간 후 행동을 재평가한다.
- 역사적 정확성(historical accuracy), 기여 빈도(contribution frequency), 이상 기록(anomaly record)을 주요 지표로 통합한 새로운 동적 클라이언트 선택 방법론을 제시한다.

## Method

### 네트워크 구성과 Cluster-based HFL

네트워크는 cluster head(CH), cluster member(CM), evolving packet core(EPC)로 구성된다.

- CH와 CM 역할은 시간에 따라 변한다.
- 클러스터링은 로컬 모델 파라미터의 유사도와 평균 상대 속력을 기준으로 수행된다.
- CM이 클라이언트 역할로 CH에게 파라미터를 전달하고, CH가 edge node 역할로 CH들이 받은 파라미터를 집계해 EPC에 전달한다.
- 로컬 업데이트는 SGD로 수행한다.

### Reliability Score

각 차량의 신뢰성 점수는 세 지표의 **가중합**이다.

- **Historical Accuracy**: 해당 차량이 과거 round에 보낸 update가 validation set에서 얼마나 좋은 성능을 냈는지의 평균. 성능이 좋았던 차량은 다음 round에서도 신뢰할 수 있다고 본다.
- **Contribution Frequency**: 해당 차량이 지금까지 FL round에 얼마나 자주 참여했는지. 자주 안정적으로 참여한 차량은 통신 링크와 하드웨어 상태가 좋을 가능성이 높아 패킷 손실을 줄이는 데 기여한다.
- **Anomaly Record**: 해당 차량이 지금까지 몇 번이나 이상 update로 flag되었는지. 값이 높으면 악성 또는 불안정 차량일 가능성이 있어 감점 요인이 된다.

### Dynamic Client Selection 알고리즘

1. CH가 각 CM의 VIB(vehicle information base) 정보를 확인한다.
2. 각 CM의 reliability score를 초기화/업데이트한다.
3. Reliability score 기준으로 CM을 내림차순 정렬한다.
4. 미리 정한 비율만큼 상위 클라이언트를 선택한다 (실험에서는 75%).
5. Block flag가 켜져 있고 unblock time이 지나지 않은 차량은 제외한다 (unblock time 5).
6. 선택된 차량의 update를 cosine similarity로 검사한다.
7. 정상 update면 학습 참여 기록과 정확도 기록을 갱신한다.
8. 이상 update면 block flag를 켜고 anomaly count를 증가시킨다.
9. 새 reliability score를 VIB에 저장한다.

### Anomaly Detection

현재 update와 이전 update의 cosine similarity를 비교하여, 변화가 지나치게 크면 악성 또는 이상 update로 판단한다. 즉, 여기서의 anomaly detection 대상은 차량 네트워크의 공격 트래픽이 아니라 **FL 파이프라인 안의 오염된 model update**다.

## Experiments

- **Dataset**: MNIST (non-IID 분포 가정)
- **구현**: PyTorch, mobility simulator는 SUMO, 스트리밍은 Kafka
- **환경**: 1 km² 영역, 차량 25대, 속도 10–35 m/s, 전송 범위 100 m
- **통신**: V2V는 IEEE 802.11p, V2I는 5G NR
- **공격 설정**: 전체 차량의 20%가 공격자. Model update에 fake additive Gaussian noise를 삽입하는 방식이며, 10번째 round에 1회 공격하는 시나리오와 10번째 round부터 지속 공격하는 시나리오를 평가한다.
- **평가**: EPC에서 accuracy와 convergence time으로 수행한다.

다만 딥러닝 모델이 MLP인지 CNN인지에 대한 정보, aggregation의 구체적 방식, validation set 구성 방법은 논문에서 명시되지 않았다.

## Takeaways

- HFL에 보안 관점을 결합한 연구로, "누구를 학습에 참여시킬 것인가"를 reliability score라는 명시적 지표로 관리한다는 점이 핵심이다.
- Anomaly detection이 트래픽 수준이 아니라 model update 수준에서 이루어진다는 점에서, IDS 논문들과는 방어 대상 자체가 다르다.
- 차량 환경을 시뮬레이션했지만 데이터는 MNIST를 사용했다는 점, 모델 구조와 aggregation 세부가 빠져 있다는 점은 아쉬운 부분이다.
