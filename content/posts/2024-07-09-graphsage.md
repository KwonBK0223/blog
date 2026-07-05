---
title: "Inductive Representation Learning on Large Graphs (GraphSAGE)"
date: 2024-07-09
tags: ["GNN", "GraphSAGE", "Node Embedding", "Inductive Learning"]
categories: ["Paper Review"]
series: ["Classic Papers"]
math: true
summary: "노드별 임베딩을 직접 학습하는 대신 이웃의 feature를 샘플링·집계하는 함수를 학습하여, 학습 중 보지 못한 노드와 그래프에도 임베딩을 일반화하는 inductive 프레임워크 GraphSAGE를 리뷰한다."
---

> **Paper**: Inductive Representation Learning on Large Graphs — William L. Hamilton, Rex Ying, Jure Leskovec, *NeurIPS 2017*. [arXiv:1706.02216](https://arxiv.org/abs/1706.02216)

## TL;DR

- 기존 노드 임베딩 기법(DeepWalk, node2vec 등)과 초기 GCN은 본질적으로 **transductive**하다. 고정된 하나의 그래프 위에서 모든 노드의 임베딩을 함께 최적화하므로, 새로운 노드가 나타나면 재학습이 필요하다.
- GraphSAGE(SAmple and aggreGatE)는 노드별 임베딩 벡터를 학습하는 대신, **이웃 노드의 feature를 샘플링하고 집계(aggregate)하는 함수를 학습**한다. 학습된 집계 함수는 처음 보는 노드, 심지어 완전히 새로운 그래프에도 그대로 적용된다.
- 이웃을 고정 크기로 **균일 샘플링**하여 배치당 계산량을 통제하므로 대규모 그래프에 확장 가능하다.
- Mean, LSTM, Pooling 등 여러 aggregator를 제안하며, citation·Reddit·단백질 상호작용(PPI) 그래프의 inductive node classification에서 기존 기법을 큰 폭으로 앞섰다.

## Background

### Transductive vs Inductive

노드 임베딩의 목표는 그래프의 구조 정보와 노드 feature를 저차원 벡터로 압축하는 것이다. 행렬 분해 기반 방법이나 random walk 기반 방법(DeepWalk, node2vec)은 **각 노드마다 고유한 임베딩 벡터를 lookup table처럼 직접 최적화**한다. 이 방식은 학습에 사용된 그래프의 노드에 대해서만 임베딩을 제공하므로 transductive하다.

그러나 현실의 그래프는 계속 변한다. 새로운 사용자, 새로운 게시글, 새로운 측정 지점이 끊임없이 추가되고, 같은 형태의 완전히 새로운 그래프에 모델을 적용해야 하는 경우도 많다. 매번 전체 재학습 없이 **보지 못한 노드에 대해 임베딩을 즉시 생성**하려면, 노드 개별 벡터가 아니라 임베딩을 만들어내는 **함수**를 학습해야 한다. 이것이 inductive 학습이며 GraphSAGE의 핵심 문제의식이다.

## Method

### Sample and Aggregate

GraphSAGE의 임베딩 생성(forward propagation)은 $K$개의 층(탐색 깊이)으로 구성된다. 초기값은 노드 feature $h_v^0 = x_v$이고, 각 층 $k$에서 모든 노드 $v$에 대해 다음을 수행한다.

$$h_{\mathcal{N}(v)}^{k} = \text{AGGREGATE}_k\!\left( \{ h_u^{k-1} : u \in \mathcal{N}(v) \} \right)$$

$$h_v^{k} = \sigma\!\left( W^k \cdot \text{CONCAT}\!\left( h_v^{k-1},\; h_{\mathcal{N}(v)}^{k} \right) \right)$$

즉 (1) 이웃들의 이전 층 표현을 하나의 벡터로 집계하고, (2) 자신의 이전 표현과 concat한 뒤 선형 변환과 비선형성을 적용한다. $K$번 반복하면 각 노드는 $K$-hop 이웃의 구조·feature 정보를 흡수한 표현 $z_v = h_v^K$를 갖는다. 학습되는 것은 노드별 벡터가 아니라 **각 층의 가중치 $W^k$와 aggregator 파라미터**뿐이므로, 새로운 노드가 와도 같은 절차를 실행하기만 하면 임베딩이 생성된다.

### 이웃 샘플링

전체 이웃을 쓰면 노드 차수에 따라 계산량이 폭발한다. GraphSAGE는 각 층에서 이웃을 고정 크기 $S_k$로 **균일 샘플링**해, 배치당 계산·메모리 비용을 $\mathcal{O}(\prod_k S_k)$로 고정한다. 이 설계가 수억 엣지 규모의 그래프에 미니배치 학습을 가능하게 하는 실용적 장치다.

### Aggregator 함수들

집계 함수는 이웃 순서에 불변(permutation invariant)인 것이 이상적이며, 논문은 세 가지를 제안한다.

- **Mean aggregator**: 이웃 표현의 원소별 평균. $h_v^{k} \leftarrow \sigma(W \cdot \text{MEAN}(\{h_v^{k-1}\} \cup \{h_u^{k-1}, u \in \mathcal{N}(v)\}))$ 형태로 쓰면 GCN의 convolution 규칙의 inductive 근사가 된다. 단순하지만 강력하다.
- **LSTM aggregator**: 표현력은 크지만 순서에 민감하므로, 이웃을 무작위 순열로 섞어 적용한다. 순서가 있는 관계(예: 시간적 연결)를 다룰 때 자연스러운 선택이 되기도 한다.
- **Pooling aggregator**: 각 이웃 표현을 독립적인 fully-connected layer에 통과시킨 뒤 원소별 max-pooling을 적용한다. 실험에서 가장 좋은 성능을 보인 경우가 많다.

실무적으로는 그래프에 서로 다른 의미의 엣지가 섞여 있을 때 엣지 유형별로 다른 aggregator(예: 대칭적 관계에는 mean, 순서가 있는 관계에는 LSTM)를 조합하는 식의 활용도 가능하다.

### 학습: 지도 및 비지도

GraphSAGE는 downstream task의 지도 손실로 end-to-end 학습할 수도 있고, 레이블 없이 그래프 구조만으로 학습할 수도 있다. 비지도 손실은 가까운 노드는 유사한 임베딩을, 무관한 노드는 상이한 임베딩을 갖도록 하는 negative sampling 기반 목적 함수다.

$$J(z_u) = -\log\!\left( \sigma(z_u^{\top} z_v) \right) - Q \cdot \mathbb{E}_{v_n \sim P_n(v)} \log\!\left( \sigma(-z_u^{\top} z_{v_n}) \right)$$

여기서 $v$는 $u$ 근처에서 random walk로 함께 등장한 노드, $P_n$은 negative sampling 분포다.

## Experiments

세 가지 inductive 벤치마크에서 평가했다.

1. **Citation 그래프 (Web of Science)**: 새로 등장한 논문의 주제 분류
2. **Reddit**: 새 게시글이 속한 커뮤니티 분류 (23만 노드 규모)
3. **PPI (단백질 상호작용)**: 학습에 전혀 등장하지 않은 **완전히 새로운 그래프들**에 대한 단백질 기능 예측 (multi-label)

주요 결과는 다음과 같다.

- GraphSAGE는 raw feature 기반 분류기, DeepWalk, DeepWalk+feature 결합 대비 F1 기준 큰 폭의 향상을 보였다. Citation과 Reddit에서 완전 비지도 임베딩만으로도 기존 방법을 앞섰고, 지도 학습 버전은 추가 이득을 얻었다.
- 특히 PPI처럼 **학습 그래프와 테스트 그래프가 분리된 설정**에서 transductive 방법은 원천적으로 적용이 불가능하거나 재학습이 필요하지만, GraphSAGE는 학습된 집계 함수를 그대로 적용해 강한 성능을 냈다.
- Aggregator 간에는 pooling과 LSTM이 mean보다 다소 우수한 경향이 있었으나, mean은 가장 가볍고 안정적이었다.
- 이웃 샘플 크기 $S$를 키우면 성능이 오르지만 수확 체감이 뚜렷해, 작은 샘플로도 충분한 성능-효율 트레이드오프를 얻을 수 있다.
- DeepWalk는 새 노드가 추가될 때 임베딩 재최적화에 수백 배의 시간이 필요한 반면, GraphSAGE는 forward pass 한 번이면 된다.

## Takeaways

- GraphSAGE의 본질적 기여는 "임베딩을 학습한다"에서 "**임베딩을 만드는 함수를 학습한다**"로의 관점 전환이다. 이 전환 덕분에 진화하는 그래프와 미지의 그래프에 대한 일반화가 가능해졌다.
- 이웃 샘플링에 의한 계산량 통제는 GNN을 실서비스 규모로 확장하는 표준 기법이 되었고, 이후 PinSAGE 같은 산업 규모 추천 시스템의 토대가 되었다.
- Aggregator를 교체 가능한 모듈로 추상화한 설계는 이후 GAT(attention aggregator) 등 다양한 GNN 아키텍처를 하나의 message passing 프레임워크로 이해하게 하는 데 기여했다.
- 한계로는 균일 샘플링이 중요한 이웃과 그렇지 않은 이웃을 구별하지 못한다는 점, 깊이 $K$가 커질수록 이웃 폭발과 over-smoothing이 문제가 된다는 점이 있으며, 이는 각각 중요도 기반 샘플링과 attention, 깊은 GNN 연구로 이어졌다.

노드 feature가 존재하는 대규모·동적 그래프라면 지금도 가장 먼저 고려되는 baseline이며, GNN의 실용화를 연 고전이라 할 수 있다.
