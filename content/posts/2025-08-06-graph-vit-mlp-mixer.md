---
title: "A Generalization of ViT/MLP-Mixer to Graphs"
date: 2025-08-06
tags: [GNN, MLP-Mixer, Graph Transformer, ICML]
categories: ["Paper Review"]
series: ["AI Algorithms"]
math: true
summary: "METIS 그래프 분할로 '그래프 패치'를 정의하고 ViT/MLP-Mixer 문법을 그래프에 이식하여, 선형 복잡도로 long-range dependency와 3-WL급 표현력을 동시에 잡은 Graph ViT/MLP-Mixer를 리뷰한다."
---

> **Paper**: A Generalization of ViT/MLP-Mixer to Graphs — Xiaoxin He, Bryan Hooi, Thomas Laurent, Adam Perold, Yann LeCun, Xavier Bresson, ICML 2023 (PMLR 202:12724–12745). [arXiv:2212.13350](https://arxiv.org/abs/2212.13350)

## TL;DR

Message-passing GNN은 빠르지만 표현력이 1-WL에 묶여 있고 long-range dependency에 약하다. Graph Transformer는 이 둘을 해결하지만 global attention의 복잡도가 부담이다. 이 논문은 computer vision의 ViT/MLP-Mixer 문법 — "패치로 자르고, 인코딩하고, mixing한다" — 을 그래프에 이식한 **Graph ViT/MLP-Mixer**를 제안한다. METIS 분할로 그래프 패치를 추출하고, 각 패치를 작은 GNN으로 인코딩한 뒤, 패치 토큰들을 MLP-Mixer(또는 graph MHA)로 섞는다. 그 결과 노드·엣지 수에 선형인 비용으로 over-squashing을 완화하고, SR25에서 3-WL 수준의 표현력을 입증하며, ZINC MAE 0.073 등 경쟁력 있는 성능을 달성했다.

## Background

### MP-GNN의 두 가지 근본 한계

대부분의 GNN은 1-hop message passing(GCN, GAT 등)을 여러 layer 쌓아 $l$-hop 정보까지 전달하는 구조다. 여기에는 두 가지 구조적 약점이 있다.

**한계 1 — 제한된 표현력.** MP-GNN의 판별력은 1-WL(Weisfeiler-Leman) 테스트와 동등하다. WL 테스트는 각 노드가 이웃 라벨을 집계해 자신의 라벨을 반복 갱신하고, 두 그래프의 라벨 분포가 달라지면 non-isomorphic으로 판정하는 절차다. 1-WL과 동등하다는 것은, 실제로는 동형이 아닌 그래프를 MP-GNN이 구분하지 못하는 경우가 있다는 뜻이다. 이를 보완하는 $k$-WL 기반 GNN은 노드 $k$-tuple 단위로 라벨링하므로 표현력은 올라가지만 복잡도가 $O(N^2)$, $O(N^3)$으로 급증한다. 또 다른 보완책인 그래프 위치 인코딩(Laplacian eigenvector, random walk 등)은 $O(E)$ 복잡도로 1-WL보다 강한 표현력을 주지만, 그래프에는 고정된 방향성이 없어 전역 위치를 정확히 표현하지 못한다. 예컨대 Laplacian eigenvector는 부호를 뒤집어도 동일해서(sign ambiguity) 모델이 불필요한 대칭성을 학습하게 된다.

**한계 2 — 정보 손실과 over-squashing.** $L$-hop 떨어진 정보를 받으려면 $L$개 layer가 필요한데, 각 layer는 지수적으로 늘어나는 이웃 정보를 고정 차원 벡터에 압축한다. 멀리 있는 유용한 정보가 짓눌리는 over-squashing과 vanishing gradient가 복합적으로 발생하며, 이는 RNN의 장기 의존성 문제와 유사하다. Graph Transformer는 attention으로 이를 해결하지만 계산 복잡도가 매우 커진다.

### 왜 ViT/MLP-Mixer인가

Vision에서 ViT와 MLP-Mixer는 이미지를 고정 패치로 분할하고 patch embedding → token/channel mixing이라는 일관된 구조로 long-range dependency를 처리한다. 이 문법을 그래프로 가져오면 Transformer 수준의 표현력과 MP-GNN 수준의 효율을 동시에 노릴 수 있다는 것이 논문의 핵심 발상이다.

## Challenges: 격자에서 그래프로 갈 때 부딪히는 문제들

이미지와 그래프의 차이 때문에 단순 이식은 불가능하다. 논문은 네 가지 과제를 정리한다.

1. **패치 정의/추출**: 이미지는 고정 격자라 균일 분할이 쉽지만, 그래프는 노드·엣지 수와 구조가 제각각이라 일관된 패치 정의가 불가능하다. 픽셀 재배열 대신 그래프 클러스터링이 필요하다.
2. **패치 인코딩**: 이미지 패치는 크기가 같아 MLP로 바로 인코딩되지만, 그래프 패치는 크기·구조가 다르고 노드 순서도 없어 순열 불변(permutation invariant)한 고정 길이 인코딩이 필요하다.
3. **위치 정보**: 이미지는 상하좌우가 암묵적으로 보존되지만 그래프에는 위치 개념 자체가 없다. 노드 수준과 패치 수준 모두에서 위치 인코딩을 설계해야 한다.
4. **과적합 방지**: ViT/MLP-Mixer 계열은 학습력이 강해 과적합하기 쉬운데, vision과 달리 그래프에는 표준화된 데이터 증강(crop, flip 등에 해당하는 것)이 없다. Node dropping, edge perturbation, subgraph sampling, feature masking, mixup 등이 제안되어 있지만 성능 일관성이 떨어진다.

## Method: Graph ViT/MLP-Mixer

전체 파이프라인은 Patch Extraction → Patch Encoder(GNN) → Mixer Layers → Global Average Pooling → 예측으로 구성된다.

### 1) Patch Extraction: METIS 분할 + 1-hop overlap

**METIS**는 각 클러스터의 노드 수가 비슷하게 유지되면서(balance) 클러스터 간 edge cut이 최소화되도록 그래프를 분할하는 고전적 알고리즘이다. Coarsening(인접 노드를 매칭·압축해 그래프 축소) → Initial Partitioning(가장 축소된 그래프에서 분할) → Uncoarsening & Refinement(Kernighan-Lin 등으로 품질 개선)의 3단계로 동작하며, 내부 연결성은 높고 외부 연결성은 낮은 클러스터를 찾는다.

METIS로 그래프를 $P$개의 클러스터로 나누고 각 클러스터를 subgraph $G_p = (V_p, E_p)$로 처리하되, 분할 과정에서 잘려나간 엣지를 살리기 위해 **1-hop overlapping**을 적용해 모든 엣지가 최소 하나의 패치에 포함되도록 한다. 중요한 점은 이 패치 추출이 매 epoch 새로 수행되어 확률적(stochastic)이라는 것이다(아래 Data Augmentation 참조).

### 2) Patch Encoder: 패치별 GNN

먼저 노드·엣지 feature를 $d$차원으로 선형 임베딩한다.

$$h_i^0 = U^0 \alpha_i + u^0 \in \mathbb{R}^d, \qquad e_{ij}^0 = V^0 \beta_{ij} + v^0 \in \mathbb{R}^d$$

각 패치에 MP-GNN(GCN, GAT 등 무엇이든 가능)을 적용해 노드·엣지 임베딩을 갱신하는데, 패치 전체의 평균 정보를 반영하는 항이 함께 더해진다.

$$h_{i,p}^{\ell+1} = f_{\text{node}}\left(h_{i,p}^{\ell}, \{h_{j,p}^{\ell} \mid j \in \mathcal{N}(i)\}, e_{ij,p}^{\ell}\right) + g_{\text{patch-node}}(h_p^{\ell})$$

여기서 $h_p^{\ell} = \frac{1}{|V_p|}\sum_{i \in V_p} h_{i,p}^{\ell}$는 패치 내 노드 평균으로, 패치 수준의 글로벌 context 역할을 한다. 하나의 노드가 여러 패치에 중복 포함되는 경우(1-hop overlap 때문) 겹치는 임베딩을 평균해 일관성을 유지한다.

$$h_i^{\ell+1} = \mathrm{Mean}_{k \mid i \in V_k}\, h_{i,k}^{\ell+1}$$

마지막으로 패치별 평균 풀링과 MLP로 고정 길이 패치 임베딩 $x_{G_p} = \mathrm{MLP}(h_p) \in \mathbb{R}^d$를 만든다. GNN 인코더 자체가 순열 불변이므로 과제 2가 해결된다.

### 3) Positional Encoding: 노드 PE + 패치 PE

- **Node PE**: random walk 또는 Laplacian 기반 절대 위치 인코딩 $p_i$를 초기 노드 임베딩에 더한다($h_i^0 = T_0 p_i + U_0 \alpha_i + u_0$). Laplacian eigenvector의 sign ambiguity는 학습 중 랜덤 부호 flip으로 대응한다.
- **Patch PE**: 클러스터 간 연결 수로 coarse adjacency matrix를 정의하고,

$$A_P(i,j) = |\mathrm{Cut}(V_i, V_j)| = \sum_{k \in V_i}\sum_{l \in V_j} A_{kl}$$

이를 기반으로 한 패치 위치 인코딩 $\hat{p}_i$를 패치 임베딩에 더해($x_i^0 = \hat{T}_0 \hat{p}_i + \hat{U}_0 x_i + \hat{u}_0$) 패치 간 상대적 위치 관계를 보존한다.

### 4) Mixer Layer: 토큰=패치

패치 임베딩 행렬 $X \in \mathbb{R}^{P \times d}$에 MLP-Mixer 문법을 그대로 적용한다.

$$U = X + \left(W_2\,\sigma(W_1\, \mathrm{LayerNorm}(X))\right) \quad \text{(Token mixer)}$$
$$Y = U + \left(W_4\,\sigma(W_3\, \mathrm{LayerNorm}(U)^T)\right)^T \quad \text{(Channel mixer)}$$

Token mixer에는 coarse adjacency $A_D^P \in \mathbb{R}^{P \times P}$가 곱해져 패치 연결 구조가 반영된다. ViT 변형(Graph ViT)에서는 token mixer 자리에 graph 기반 multi-head attention(gMHA)을 쓴다.

$$U = X + \mathrm{gMHA}(\mathrm{LayerNorm}(X)), \qquad Y = U + \mathrm{MLP}(\mathrm{LayerNorm}(U))$$

마지막으로 Global Average Pooling으로 그래프 임베딩 $h_G$를 얻고 $y_G = \mathrm{MLP}(h_G)$로 회귀 또는 분류를 수행한다.

### 5) 암묵적 Data Augmentation

METIS + 1-hop overlap 기반 패치 추출은 확률적이어서 같은 그래프라도 epoch마다 다른 patch set이 나온다. 이것이 곧 간접적인 데이터 증강으로 작동해 데이터 다양성을 높이고, 모델이 한정된 분할 구조에만 과적합하지 않도록 유도한다. 특히 작은 그래프 데이터셋에서 유효 데이터 수를 늘리는 효과가 있다. 과제 4에 대한 답이 별도의 증강 기법이 아니라 파이프라인 자체에 내장되어 있는 셈이다.

## Experiments

- **분자 벤치마크**: ZINC에서 MAE 0.0730, MolHIV에서 ROC-AUC 0.7997로 Benchmarking-GNNs 및 OGB 기준에서 경쟁력 있는 성능을 보였다.
- **Long-range 모델링**: LRGB(Long Range Graph Benchmark)에서 long-range dependency 모델링 성능을 검증했고, TreeNeighbourMatch 실험에서 over-squashing 완화를 확인했다.
- **표현력**: 1-WL로 구분 불가능한 strongly regular graph 데이터셋 SR25에서 3-WL 수준의 판별력을 입증했다. METIS 패치가 만들어내는 부분구조 정보와 패치 PE의 조합이 순수 message passing보다 강한 구조 식별 능력을 주는 것이다.
- **효율**: 모든 구성 요소(METIS, 패치별 GNN, Mixer)가 노드·엣지 수에 선형이라 global attention 대비 확장성이 좋다.

## Takeaways

- "패치 → 인코딩 → mixing"이라는 vision의 설계 문법이 불규칙한 그래프 도메인에서도 성립함을 보인 논문이다. CV, NLP, Graph를 하나의 아키텍처 언어로 잇는 통합적 관점을 제시한다.
- MP-GNN의 두 고질병(표현력, over-squashing)을 attention 없이도 동시에 다룰 수 있다. 패치 단위 토큰화가 정보 전달 경로를 근본적으로 짧게 만들기 때문이다.
- METIS의 확률적 분할이 증강 역할까지 겸하는 설계는 우아하다. 구조적 선택 하나가 효율·표현력·일반화라는 세 마리 토끼에 모두 기여한다.
- 그래프 수준 예측(분자 물성, 재료 물성 등)처럼 전역 구조가 중요한 문제에서, GNN으로 로컬 구조를 인코딩하고 Mixer로 전역 정보를 섞는 하이브리드는 실용적인 기본기로 삼을 만하다.
