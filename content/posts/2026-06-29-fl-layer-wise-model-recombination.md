---
title: "Is Aggregation the Only Choice? Federated Learning via Layer-wise Model Recombination"
date: 2026-06-29
tags: ["Federated Learning", "Model Recombination", "Non-IID", "KDD"]
categories: ["Paper Review"]
series: ["Partial & Layer-wise FL"]
math: true
summary: "FedAvg의 단순 평균 대신 로컬 모델들을 레이어 단위로 셔플·재조합해 클라이언트에 배포하는 FedMR을 제안하여, non-IID 환경에서 flat minima로의 탈출과 지식 공유를 동시에 달성한다."
---

> **Paper**: Is Aggregation the Only Choice? Federated Learning via Layer-wise Model Recombination — M. Hu, Z. Yue, X. Xie, C. Chen, Y. Huang, X. Wei, X. Lian, Y. Liu, M. Chen, in *Proc. 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)*, 2024. [doi:10.1145/3637528.3671722](https://doi.org/10.1145/3637528.3671722) / [arXiv:2305.10730](https://arxiv.org/abs/2305.10730)

## TL;DR

"Aggregation이 유일한 선택인가?"라는 제목 그대로, FedAvg의 평균 기반 집계를 **레이어 단위 model recombination**으로 대체한 FedMR을 제안한다. 클라우드가 업로드된 로컬 모델들을 레이어별로 분해하고 같은 index의 레이어끼리 셔플해 새로운 모델들을 만들어 배포함으로써, non-IID 환경에서의 일반화 성능을 개선한다.

## Background

FedAvg 기반 패러다임의 단점을 극복하고 non-IID 시나리오에서 FL 성능을 높이는 것은 중요한 과제다. 저자들이 정리하는 FedAvg의 문제는 다음과 같다.

1. FL에서는 클라이언트마다 data distribution이 다르다.
2. 그래서 로컬 모델들이 서로 다른 optimization direction으로 이동한다.
3. FedAvg는 이 로컬 모델들을 단순 평균 낸다.
4. 단순 평균은 각 로컬 모델이 가진 특정 정보를 희석하거나 무시할 수 있다.
5. 또한 모든 클라이언트가 같은 global model에서 다시 local training을 시작하므로, local search가 특정 **sharp region**에 갇힐 수 있다.
6. 좋은 generalization solution은 sharp area보다 **flat area**에 있을 가능성이 높다.
7. 따라서 단순 평균 대신 layer-wise recombination으로 모델을 solution space에서 흔들어 주면 sharp area를 벗어나 flat area로 갈 가능성이 커진다.

저자들이 주장하는 기여는 다음과 같다.

1. FedAvg 기반 model aggregation을 대체하는 새로운 층별 model recombination 방법 **FedMR** 제안
2. Recombination과 aggregation의 장점을 결합하여 전체 FL 훈련 과정 가속화
3. Convex 시나리오에서의 수렴성 증명과 non-convex 시나리오에서의 실험
4. 효과성과 호환성을 보여주는 다양한 실험

## Method

### FedMR의 3단계

- **Step 1 — Model dispatching**: Cloud가 $K$개의 intermediate/recombined model을 $K$명의 selected client에 각각 하나씩 보낸다. (모두에게 같은 global model을 보내는 FedAvg와 다르다.)
- **Step 2 — Model upload**: 각 클라이언트가 받은 모델을 자기 로컬 데이터로 학습한 뒤, 업데이트된 로컬 모델을 cloud에 업로드한다.
- **Step 3 — Model recombination**: Cloud가 업로드된 모델들을 레이어별로 분해하고, **같은 index의 레이어끼리 shuffle**해서 $K$개의 새로운 recombined model을 생성한다.

같은 layer index끼리 셔플하므로 재조합된 모델들은 여전히 homogeneous한 구조를 유지한다. 클라이언트는 다른 클라이언트의 레이어가 섞인 모델을 받아 local training을 진행하게 되고, 이 과정에서 다른 클라이언트의 지식이 전달되므로 **aggregation과 유사한 knowledge sharing 효과**가 발생한다. 즉 명시적 평균 없이도 셔플-재학습을 통해 정보가 네트워크 전체로 퍼진다.

### Two-Stage Training Scheme

학습 초기에는 레이어들이 서로 align되어 있지 않은데, 이 상태에서 recombination까지 하면 layer dependency가 더 깨져 초기 수렴이 느려질 수 있다. 이를 완화하기 위해 두 단계로 학습한다.

- **Stage 1 — Aggregation-based pre-training**: FedAvg 방식으로 일정 round 동안 global model을 대략적으로 학습한다.
- **Stage 2 — Model recombination stage**: pre-trained 모델들을 기반으로 FedMR recombination을 수행한다.

## Experiments

Convex 시나리오에서 수렴성을 이론적으로 증명하고, non-convex 시나리오에서 다양한 실험을 수행했다. 기존 SOTA FL 방법들과 비교해 클라이언트의 프라이버시를 노출하지 않으면서 추론 정확도를 유의미하게 개선함을 보였다.

## Takeaways

- FL의 집계를 "평균"이 아니라 **"재조합"**으로 바꿀 수 있다는 발상의 전환이 핵심이다. 셔플된 모델로 재학습하는 것 자체가 암묵적인 지식 공유 메커니즘이 된다.
- Flat minima 관점의 동기 부여가 흥미롭다. Recombination이 모델을 solution space에서 흔들어 sharp minima 탈출을 돕는다는 해석이다.
- 다만 layer dependency가 깨지는 문제 때문에 FedAvg pre-training이 필요하다는 점은, recombination이 만능이 아니라 **학습 후반부의 fine-grained 탐색에 적합한 도구**임을 시사한다.
