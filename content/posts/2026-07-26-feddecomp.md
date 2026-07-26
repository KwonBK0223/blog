---
title: "Decoupling General and Personalized Knowledge in Federated Learning via Additive and Low-Rank Decomposition"
date: 2026-07-26
tags: ["Federated Learning", "Personalization", "Low-Rank Decomposition", "FedDecomp"]
categories: ["Paper Review"]
series: ["Federated Learning"]
math: true
summary: "모든 layer의 weight를 shared full-rank branch와 personalized low-rank branch의 합으로 재설계해, general knowledge와 client-specific knowledge를 parameter 단위로 분리하는 FedDecomp를 리뷰한다."
---

> **Paper**: Decoupling General and Personalized Knowledge in Federated Learning via Additive and Low-Rank Decomposition — Xinghao Wu, Xuefeng Liu, Jianwei Niu, Haolin Wang, Shaojie Tang, Guogang Zhu, Hao Su, *ACM MM 2024*. [DOI:10.1145/3664647.3681514](https://doi.org/10.1145/3664647.3681514)

## TL;DR

- Personalized FL(PFL)의 목표는 하나의 global model이 아니라 client마다 다른 personalized model $w_i$를 학습하는 것이며, 이는 "공유할 general knowledge와 local에 남길 client-specific knowledge를 분리하는 문제"로 볼 수 있다.
- 기존 parameter-partition 방식은 layer나 parameter를 shared 또는 personalized로 이분한다. 그러나 어떤 parameter를 shared로 지정해도 그 안에 client-specific knowledge가 섞일 수 있어, 완벽한 분리는 오히려 general/client-specific 양쪽 성능을 모두 떨어뜨릴 수 있다.
- **FedDecomp**는 각 layer의 weight를 additive하게 $\theta_i^k = \sigma_i^k + \tau_i^k$로 분해한다. $\sigma_i^k$는 general knowledge를 담는 shared full-rank branch, $\tau_i^k = B_i^k A_i^k$는 client-specific knowledge를 담는 personalized low-rank branch다.
- 한 communication round 안에서 먼저 low-rank branch $\tau_i$를 학습해 local drift를 흡수하게 하고, 그다음 full-rank branch $\sigma_i$를 학습해 general knowledge에 집중하게 하는 alternating training을 사용한다. 서버에는 $\sigma_i$만 전송되고 $\tau_i$는 client local에 남는다.

## Background

### Personalized FL의 문제 정의

PFL의 목표는 모든 client에게 동일한 global model을 주는 것이 아니라, client $i$마다 자기 데이터에 잘 맞는 personalized model $w_i$를 갖게 하는 것이다. 이 문제는 결국 "client 간에 공유해야 할 general knowledge와 각 client에 특화되어 local에 남겨야 할 client-specific knowledge를 어떻게 분리할 것인가"라는 물음으로 환원된다.

관련 개념을 한 줄씩 구분해 두면 이후 설명이 명확해진다. **personalized FL**은 client마다 다른 personalized model을 만드는 것이 목적이고, **partial FL**은 전체 parameter가 아니라 일부 parameter만 학습/전송/집계하며, **cyclic FL**은 round마다 공유하거나 학습하는 parameter/block을 순환적으로 바꾼다. FedDecomp는 이 중 personalized FL을 겨냥하되, 분리의 단위를 layer가 아니라 parameter 성분으로 내린다.

### 두 갈래의 기존 접근

heterogeneity(non-IID)를 어떻게 다루느냐를 기준으로, PFL의 knowledge 분리 방식은 크게 두 갈래로 나뉜다.

- **parameter-partition 기반 FL** (FedPer, FedBN, FedRep, FedRoD): 모델의 일부 parameter나 layer만 shared로 두고 나머지는 personalized로 local에 둔다. 한계는, 어떤 parameter를 shared로 지정하더라도 그 parameter 안에 client-specific knowledge가 섞여 들어갈 수 있다는 점이다. 반대로 personalized로 둔 parameter에도 general knowledge가 섞인다.
- **parameter-decomposition 기반 FL** (FedDecomp): parameter를 shared/personalized로 이분하는 대신, parameter 자체를 여러 성분으로 표현한다. 그 결과 모든 layer가 shared component와 personalized component를 **동시에** 갖는다.

이 대비의 밑에는 하나의 문제의식이 있다. general knowledge와 client-specific knowledge를 완벽히 분리하려고 layer를 명백하게 갈라도, 각 부분에는 그 반대편 지식이 어느 정도 섞일 수밖에 없다. 오히려 완벽한 분리를 강제하면 general/client-specific 양쪽 성능이 함께 떨어질 수 있다. FedDecomp는 이 관찰에서 출발해, 분리를 layer 단위의 이분이 아니라 parameter 내부의 additive 분해로 옮긴다.

## Method

### 핵심 아이디어 — Additive Decomposition

FedDecomp는 personalized model의 각 layer weight를 두 성분의 합으로 정의한다.

$$\theta_i^k = \sigma_i^k + \tau_i^k$$

여기서 중요한 점은 이 분해가 **사후 분해가 아니라 사전 설계**라는 것이다. 이미 학습된 weight $\theta$를 SVD로 잘라 $\sigma$와 $\tau$로 나누는 것이 아니라, 처음부터 layer를 $\sigma + \tau = \sigma + BA$ 구조로 parameterization해 학습한다. $\theta_i^k$는 독립적으로 저장되는 weight가 아니라, forward에서 실제로 쓰이는 **effective weight**다.

- $\sigma_i^k \in \mathbb{R}^{I \times O}$: 일반 weight matrix처럼 두는 **shared full-rank branch**. full rank이므로 general knowledge를 담기에 충분한 capacity를 갖는다.
- $\tau_i^k = B_i^k A_i^k$: 두 개의 작은 matrix 곱으로 만드는 **personalized low-rank branch**. fully-connected layer 기준으로 $B_i^k \in \mathbb{R}^{I \times r}$, $A_i^k \in \mathbb{R}^{r \times O}$이며 rank는 $r = R_l \cdot \min\{I, O\}$이다. convolution layer는 kernel weight를 펼쳐 low-rank 분해를 한 뒤 다시 convolution weight shape로 reshape한다.

client-specific knowledge는 낮은 capacity로도 충분하다는 가정 아래, shared branch는 full-rank로 general knowledge를 담고 personalized branch는 low-rank로 client-specific 보정만 담게 하는 구조다. 큰 shared weight 위에 client별 low-rank adapter를 더한다는 점에서 LoRA와 형태가 유사하다.

### 초기화

$A$는 Gaussian random으로, $B$는 zero로 초기화한다. 따라서 학습 시작 시점에는 $B = 0$이므로

$$\tau_i^k = B_i^k A_i^k = 0$$

이 되고, 초기 effective weight는

$$\theta_i^k = \sigma_i^k + 0 = \sigma_i^k$$

가 된다. 즉 학습 초반에는 personalized branch가 아무 영향을 주지 않아 사실상 FedAvg 모델처럼 시작하고, 이후 local training이 진행되면서 $\tau_i$가 client-specific correction을 배우기 시작한다.

### Alternating Training

FedDecomp는 한 communication round 안에서 두 branch를 번갈아 학습하는 전략을 쓴다. 각 client $i$는 $\sigma_i$와 $\tau_i$를 가지고 있고 forward에는 $\theta_i = \sigma_i + \tau_i$를 사용한다.

1. **Personalized branch 학습**: $\sigma_i$를 freeze하고 $\tau_i$를 학습한다. 이 단계에서 low-rank branch $\tau_i$가 client-specific knowledge, 즉 local drift를 먼저 흡수한다.
2. **Shared branch 학습**: $\tau_i$를 freeze하고 $\sigma_i$를 학습한다. $\tau_i$가 이미 local drift를 어느 정도 흡수한 상태이므로, $\sigma_i$는 non-IID local optimum에 덜 끌려가고 general knowledge 학습에 더 집중할 수 있다.
3. **전송**: client는 shared full-rank matrix $\sigma_i$만 서버로 보낸다. $\tau_i$는 전송하지 않는다.
4. **집계**: 서버는 받은 full-rank matrix들을 평균낸다.
$$\sigma^{t+1} = \frac{1}{N} \sum_{i=1}^{N} \sigma_i^{t+1}$$
5. **Broadcast 및 초기화**: 서버가 aggregated global $\sigma$를 다시 broadcast하면, 각 client는 자기 $\sigma_i$를 global $\sigma$로 초기화하고 $\tau_i$는 local에 그대로 유지한다.

먼저 $\tau$로 drift를 흡수한 뒤 $\sigma$를 학습한다는 순서가 이 전략의 핵심으로, 단순히 adapter를 붙이는 것과 구분되는 지점이다.

유저가 직접 정하는 것은 $\sigma$나 $\tau$의 값이 아니다. 그 값은 학습으로 결정된다. 유저가 정하는 것은 구조와 hyperparameter로, personalized branch의 capacity를 정하는 rank 비율 $R_l$(FC layer), $R_c$(conv layer)와, 두 branch에 배분하는 학습 epoch 비율 $E_{lora}$($\sigma$를 freeze하고 $\tau$를 학습하는 epoch 수), $E_{global}$($\tau$를 freeze하고 $\sigma$를 학습하는 epoch 수)이다.

## Privacy & Experiments

### Privacy — DLG 관점

FedDecomp는 privacy를 Deep Leakage from Gradients(DLG) 관점에서 분석한다. DLG 공격은 다음과 같이 진행된다.

1. 공격자는 각 client가 local data로 계산한 gradient를 훔친다.
2. 공격자는 반복 최적화를 통해 입력값을 찾아내되, 그 입력으로 계산한 gradient가 실제 gradient에 최대한 가까워지도록 만들어 원본 입력을 복원한다.

FedDecomp는 서버로 shared full-rank branch $\sigma$만 전송되고 personalized low-rank branch $\tau$는 client local에 남는다. 서버가 관측할 수 있는 정보가 $\sigma$에 국한된다는 이 구조가 gradient 기반 복원 공격의 표면을 제한한다는 것이 privacy 분석의 논지다.

### Experiments

논문은 여러 데이터셋에 걸쳐 FedDecomp가 non-IID로 인한 성능 저하를 효과적으로 완화함을 검증한다. shared full-rank branch가 general knowledge를 담당하고 personalized low-rank branch가 local drift를 흡수하는 분업이, parameter를 단순히 이분하는 기존 PFL 대비 heterogeneity에 강건하다는 것이 실험의 주된 주장이다.

## Takeaways

FedDecomp는 "parameter를 shared 또는 personalized로 이분하면 각 부분에 반대편 지식이 섞인다"는 문제를, parameter 자체를 두 성분의 합으로 재설계하는 것으로 푼다. 정리하면 다음과 같다.

- **분리 단위의 전환**: 기존 PFL이 layer/parameter를 선택적으로 shared와 personalized로 나눈 것과 달리, FedDecomp는 모든 parameter를 shared full-rank + personalized low-rank로 분해한다. general knowledge와 client-specific knowledge를 layer가 아니라 parameter 성분 수준에서 더 세밀하게 분리한다는 것이 핵심 기여다.
- **훈련 순서의 설계**: LoRA류가 shared weight 위에 low-rank adapter를 더하는 형태와 겉모습은 비슷하지만, novelty는 "먼저 $\tau$로 local drift를 흡수한 뒤 $\sigma$를 학습한다"는 순서와, 이를 decomposition 관점으로 정식화한 데 있다. $\tau$가 non-IID 편향을 먼저 걷어내 주어야 $\sigma$가 깨끗한 general knowledge를 학습할 수 있다는 것이 alternating training의 논리다.
- **리뷰 관점**: 성능은 alternating training과 rank/epoch hyperparameter($R_l, R_c, E_{lora}, E_{global}$)에 의존하므로, 이들 설정에 대한 민감도가 실무 적용에서의 관건이 된다. 또한 $\sigma$만 전송하는 구조는 통신 대상을 shared branch로 한정해 privacy 표면을 줄이는 부수적 이점도 갖는다.

client 간 이질성이 주로 parameter 내부의 지식 혼재에서 오는 상황이라면, FedDecomp는 layer 단위 분리보다 한 단계 더 미세한 분리를 제공하는 방법이 된다.
