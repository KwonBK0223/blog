---
title: "HeroFilter: Adaptive Spectral Graph Filter for Varying Heterophilic Relations"
date: 2026-08-31
tags: ["Graph Neural Networks", "Spectral Filter", "Heterophily", "MLP-Mixer"]
categories: ["Paper Review"]
series: ["AI Algorithms"]
math: true
summary: "heterophily와 최적 스펙트럼 필터의 관계가 단조롭지 않다는 이론적 관찰에서 출발해, 고정된 low-/high-pass 대신 노드마다 적응형 다항 스펙트럼 필터를 학습하는 MLP-Mixer 기반 GNN인 HeroFilter를 리뷰한다."
---

> **Paper**: HeroFilter: Adaptive Spectral Graph Filter for Varying Heterophilic Relations — Shuaicheng Zhang, Haohui Wang, Junhong Lin, Xudong Guo, Yifan Zhu, Si Zhang, ..., Dawei Zhou, *NeurIPS 2025*.

## TL;DR

heterophily가 강한 그래프에서는 이웃을 무조건 평균 내는 기존 GNN이 오히려 정보를 오염시킨다. 이를 스펙트럼 관점에서 다루려는 filter-based GNN은 "homophily면 low-pass, heterophily면 high-pass"라는 단순한 가정을 써 왔다. HeroFilter는 이 가정이 틀렸다는 데서 출발한다. 저자들은 heterophily가 커진다고 최적 스펙트럼 응답이 단순히 저주파에서 고주파로 이동하지 않고 전 대역에 걸쳐 활성화됨을 관찰하고, 이를 뒷받침하는 이론을 세운 뒤, 고정 필터 대신 노드마다 적응형 다항 스펙트럼 필터를 학습하는 MLP-Mixer 기반 아키텍처를 제안한다. 핵심 메시지는 하나다. **필터는 heterophily의 "크기"가 아니라 판별 정보와 heterophily가 어느 "주파수"에 놓여 있는지에 적응해야 한다.**

## Background

HeroFilter를 이해하려면 두 개의 축을 먼저 정리해야 한다. 하나는 heterophily라는 그래프의 성질이고, 다른 하나는 그래프를 주파수 관점에서 보는 spectral 시각이다.

### Heterophily란 무엇인가

**Heterophily(이종친화성)** 는 연결된 노드끼리 서로 다른 라벨을 가질 확률이 높은 상태를 뜻한다. "친구끼리 취향이 비슷하다"는 **Homophily(동질친화성, 유유상종)** 의 반대 개념이다. Heterophilic 그래프는 "연결된 존재끼리 서로 다르다"는 규칙을 따른다. 대표적인 예가 몇 가지 있다.

- **사기 거래 탐지**: 사기꾼 계정은 주로 정상 사용자 계정과 거래를 트기 때문에, 서로 다른 라벨의 노드가 연결될 확률이 높다.
- **데이팅 네트워크**: 매칭 그래프에서는 주로 남성 노드와 여성 노드가 연결된다.
- **분자 구조**: 화학 결합에서는 같은 원자끼리보다 서로 다른 종류의 원자·분자가 결합하는 경우가 많다.

기존의 GCN, GAT 같은 GNN은 "이웃은 나와 비슷할 것"이라는 homophily 가정 위에 설계되었다. 이웃 정보를 더하거나 평균 내어(aggregation) 자기 특징을 갱신하는 방식이다. 그러나 heterophilic 그래프에 이를 그대로 적용하면 문제가 생긴다. 첫째, **정보의 오염** — 나와 완전히 다른 클래스의 이웃 정보가 섞여 내 정체성이 흐려진다. 둘째, **over-smoothing** — 이웃을 섞을수록 노드 간 구분이 사라져 예측 정확도가 급격히 떨어지고, 때로는 그래프 구조를 아예 무시하는 MLP보다도 못한 성능을 낸다.

여기서 흔히 혼동되는 개념 하나를 짚고 넘어가는 편이 좋다. **Heterophilic(연결의 경향성)** 은 "서로 다른 라벨을 가진 노드끼리 연결되는가"라는 관계의 성질을 말하고, **Heterogeneous(구성의 다양성)** 는 "그래프 안에 여러 종류의 노드·엣지 타입이 섞여 있는가"라는 구조의 성질을 말한다. 노드 타입이 하나뿐인(homogeneous) 그래프도 얼마든지 heterophilic일 수 있으므로 둘은 독립된 개념이다.

### 그래프에서의 frequency

두 번째 축은 그래프를 주파수로 보는 관점이다. 여기서 **frequency는 시간에 따른 진동 횟수가 아니다.** 그래프에서 frequency는 "연결된 이웃끼리 신호값이 얼마나 빠르게 변하는가"를 뜻한다.

간단한 사슬 그래프 $A-B-C-D-E$를 두고 각 노드에 신호 $x$(온도든 임베딩이든 라벨이든)를 얹어 보자. 노드값이 $(10, 10, 11, 11, 10)$처럼 연결된 이웃끼리 비슷하면 이는 **low-frequency** 신호다. 반대로 $(10, -10, 10, -10, 10)$처럼 엣지를 하나 건널 때마다 값이 확확 뒤집히면 **high-frequency** 신호다. 즉 그래프에서 주파수의 기준은 connectivity, 이웃 간 변화량이다.

이 직관은 Graph Laplacian으로 정확히 형식화된다. $L = D - A$로 정의된 Laplacian에 대해 다음이 성립한다.

$$x^\top L x = \sum_{(i,j)\in E}(x_i - x_j)^2$$

즉 Laplacian은 본질적으로 "이 신호가 이웃끼리 얼마나 다른가"를 재는 연산자다. 이제 Laplacian을 고유분해하면 $L u_k = \lambda_k u_k$이고, 고유벡터 $u_k$를 norm 1로 잡으면 $u_k^\top L u_k = \lambda_k$가 된다. 그런데 좌변은 방금 본 대로 $\sum_{(i,j)\in E}(u_k(i) - u_k(j))^2$, 곧 그 고유벡터가 edge를 따라 얼마나 많이 변하는지를 재는 값이다. 따라서 **eigenvalue $\lambda_k$가 클수록 그 mode는 edge를 따라 많이 변하는 high-frequency에 해당한다.** $u_k$는 그래프의 spectral mode, $\lambda_k$는 그 mode의 주파수 정도로 읽으면 된다.

이 언어로 보면 GCN은 1차 Chebyshev 다항식을 쓰는 스펙트럴 GNN, 곧 **low-pass filter**로 해석된다. 저주파를 증폭하고 고주파를 억제해 그래프 전반에 걸친 매끄러운 신호를 유지한다. 저주파에 유용한 신호가 몰려 있는 homophilic 구조에는 잘 맞지만, 유용한 신호가 더 높은 주파수 대역에 있을 수 있는 heterophilic 구조에는 부적합하다.

## Motivation: Non-Monotonic Filter–Heterophily Relationship

Spectral 관점을 받아들이면 자연스럽게 filter-based GNN이 등장한다. 고주파 정보를 명시적으로 통합하거나 high-pass 혹은 혼합 필터를 써서 heterophily에 대응하려는 흐름이다. 그런데 이들 대부분은 매우 단순한 가정을 깔고 있다.

$$\text{Homophily} \rightarrow \text{Low-frequency} \rightarrow \text{Low-pass}$$
$$\text{Heterophily} \rightarrow \text{High-frequency} \rightarrow \text{High-pass}$$

HeroFilter의 문제 제기는 이 대응이 **너무 단조롭고 현실과 어긋날 가능성이 높다**는 것이다. 저자들은 heterophily 수준을 조절할 수 있는 synthetic graph를 만들어 각 그래프에서 학습된 최적 필터의 스펙트럼 응답을 관찰했다. 결과는 단순한 low→high 이동과 거리가 멀었다. heterophily가 낮은 그래프에서 나온 필터도 저주파에만 머물지 않고 전 대역에 걸쳐 활성화되었고, heterophily가 높은 그래프에서 나온 필터 역시 고주파뿐 아니라 저주파 성분까지 함께 살렸다. 최적 스펙트럼 응답은 heterophily 수준 하나로 예측되는 매끄러운 함수가 아니었던 것이다.

이 관찰에서 두 개의 질문이 나온다.

- **Q1.** 그래프의 heterophily, 스펙트럼 필터, GNN의 예측 성능 사이의 관계를 어떻게 형식적으로 설명할 수 있는가?
- **Q2.** 다양한 heterophily 패턴을 가진 그래프에서 robust하게 작동하는 adaptive filter GNN을 어떻게 설계할 수 있는가?

이 글의 나머지 절반은 Q1에 답하는 이론(Theory)과 Q2에 답하는 아키텍처(HeroFilter)로 이어진다.

## Theory

이론 파트의 목표는 "필터는 고정된 low-/high-pass가 아니라 적응형이어야 한다"를 empirical observation이 아니라 논리로 정당화하는 것이다. 전체 흐름은 네 단계로 정리된다. Definition 2가 heterophily를 필터와 같은 domain으로 옮기고, Proposition 1이 고정 스펙트럼 직관이 충분하지 않음을 보이며, Proposition 2가 적응형 필터의 표현력을 확보하고, Theorem 1이 이 모든 것을 실제 예측 오차와 연결한다. 아래에서는 대수 전개를 그대로 옮기기보다 각 결과가 다음 결과를 위해 하는 **역할**과 **결론**에 집중한다.

### Definition 2: Spectral Heterophily

문제의 출발점은 domain mismatch다. Heterophily는 보통 spatial domain에서 정의된다. 노드 $v_i$의 spatial heterophily는

$$h_i = \frac{|\{v_j \in N(v_i) : y_j \neq y_i\}|}{|N(v_i)|}$$

로, "내 이웃 중 나와 다른 라벨이 얼마나 있는가"를 재는 스칼라다. 반면 그래프 필터 $g(\Lambda)$는 본질적으로 spectral domain에서 정의된다. 서로 다른 domain에 사는 두 대상을 직접 연결하기는 어렵다.

Definition 2는 이 격차를 graph Fourier transform으로 메운다. spatial heterophily 벡터 $h$에 Fourier 기저 $U^\top$를 곱해

$$\hat h = U^\top h$$

를 정의한다. 이렇게 하면 heterophily를 더 이상 하나의 스칼라가 아니라 **주파수별 분포**로 볼 수 있게 된다. $\hat h_i$는 spectral component $i$에 담긴 heterophily의 양을 뜻한다. 이 한 걸음 덕분에 "heterophily가 얼마나 큰가"라는 질문을 "heterophily가 어느 주파수에 위치하는가"라는 훨씬 풍부한 질문으로 바꿀 수 있고, 이것이 이후 모든 논의의 토대가 된다.

### Proposition 1: 필터 응답과 heterophily는 단조 관계가 아니다

Proposition 1은 필터의 평균 스펙트럼 응답과 그래프 heterophily 수준이 단순한 단조 관계를 따르지 않음을 증명한다. 증명의 핵심 도구는 weighted AM-GM 부등식(가중 산술평균 ≥ 가중 기하평균)이다.

증명은 heterophily 값 $x_i = |\hat h_i|$에 필터 응답을 확률처럼 정규화한 가중치

$$w_i = \frac{g(\lambda_i)}{\sum_k g(\lambda_k)}$$

를 부여하는 데서 시작한다. $w_i$는 전체 필터 응답 중 주파수 $i$가 차지하는 상대적 비중으로, 필터가 어느 주파수를 중요하게 보는지를 나타낸다. 여기에 AM-GM을 적용하면 "필터로 가중한 heterophily의 산술평균 ≥ 기하평균"이라는 부등식이 나오고, 양변에 로그를 취해 곱을 합으로 풀면 평균 필터 응답을 heterophily 스펙트럼과 필터-heterophily 상호작용 항으로 분리한 형태로 재배열된다.

결론이 중요하다. 평균 필터 응답은 하나의 스칼라 heterophily 수준만으로 결정되지 않는다. **heterophily가 "크다/작다"가 아니라 어느 주파수에 놓여 있는지, 그리고 그 주파수에서 필터 $g(\lambda_i)$가 어떻게 반응하는지가 함께 중요하다.** 같은 평균 heterophily라도 그것이 저주파에 몰려 있는지 고주파에 몰려 있는지에 따라 필요한 필터가 달라진다. 고정된 low-/high-pass 하나로는 이 다양성을 담을 수 없으므로, adaptive filter가 필요하다.

### Proposition 2: Adaptive filter는 충분한 표현력을 가진다

Proposition 1이 "고정 필터는 충분하지 않다"를 말했다면, 곧바로 이어지는 질문은 "그럼 적응형 필터는 필요한 스펙트럼 응답을 실제로 표현할 수 있는가"이다. Proposition 2가 여기에 답한다.

증명의 아이디어는 단순하다. 필터의 스펙트럼 응답 $g(\Lambda)$와 target 라벨 분포 $Y$의 방향을 완전히 일치시키려면(코사인 유사도 1), 둘을 스칼라 배수 관계 $g(\Lambda) = cY$로 만들면 된다. 주파수별로 학습 가능한 다항 필터는 각 주파수 $i$에서 여러 항의 합으로 $c y_i$를 만들 수 있고, activation이 유계든 무계든 각 항이 균등하게 기여하도록 파라미터 $\{w_k\}$를 잡으면 언제나 이 등식을 맞출 수 있다. 따라서 목표 방향과 정렬되는 필터 파라미터가 존재한다.

여기서 반드시 구분해야 할 것이 있다. Proposition 2가 증명하는 것은 **존재성(expressivity)** 이지 학습 가능성이 아니다. "이런 파라미터가 존재한다($\exists\{w_k\}$ s.t. $g(\Lambda) \parallel Y$)"는 것과 "최적화가 그 파라미터를 실제로 찾아낸다"는 것은 전혀 다른 이야기다. 적응형 필터에 충분한 자유도가 있어 원하는 target spectral pattern을 표현할 수 있다는 것이 결론이며, 학습이 그 최적점에 도달한다거나 일반화가 보장된다는 주장은 아니다. **"표현할 수 있다(can represent)"와 "학습할 수 있다(can learn)"는 다르다.**

### Theorem 1: 예측 오차를 주파수별 정렬로 분해하다

Theorem 1은 이 이론의 정점으로, heterophily와 필터 설계가 예측 성능에 미치는 영향을 정량화하는 오차 상한을 도출한다. 결과의 형태는 다음과 같다.

$$Er(X, Y) \le c_1 - (\text{spectral alignment term})$$

여기서 spectral alignment term은 단일 heterophily 스칼라가 아니라 **주파수별 네 성분의 정렬**로 결정된다.

- $\eta_i$: 주파수 $i$에서 두 클래스의 **feature difference**
- $\delta_i$: 주파수 $i$에서 두 클래스의 **label difference**
- $g(1 - \lambda_i)$: 그 주파수에서의 **filter response**
- $|\hat h_i|$: 그 주파수의 **heterophily spectrum**

증명의 큰 줄기만 따라가 보자. 이진 분류의 sigmoid 제곱 오차는 비선형이라 다루기 어렵기 때문에, 먼저 점수 $x$를 clamp 함수 $\psi(x) \in [-1, 1]$로 유계 구간에 가둔다. 이렇게 잘라내도 손실이 무한정 달라지지 않음을 상수로 상한 지어 보이고($x > 1$이거나 $x < -1$인 극단에서도 clamped/unclamped 손실 차이가 일정 상수 이하), 이후 $x=0$ 근방에서 1차 Taylor 전개 $\frac{1}{1+e^x} \approx \frac{1}{2} - \frac{1}{4}x$를 쓴다. 나머지 항은 $|R(x)| \le \frac{|x|^3}{48}$로 통제된다. 이 과정을 거치면 비선형 sigmoid 오차가 분석 가능한 다항식 상한으로 바뀐다.

전개에서 핵심으로 살아남는 항은 label-prediction alignment 항 $-\frac{(1-2y)\psi(x)}{4}$이다. 이진 라벨에서 $1-2y$는 $\pm 1$이고 sigmoid가 감소함수이므로, 예측 점수가 정답 방향을 가리킬수록 이 항이 커지고 오차 상한이 작아진다. 그다음 스칼라 점수 $x$ 자리에 그래프 필터를 통과한 클래스 차이 $z = g(I - \tilde A)(X_1 - X_0)$를 넣으면, 핵심 항은

$$-\frac{1}{4}(y_1 - y_0)^\top \psi\!\left(g(I - \tilde A)(X_1 - X_0)\right)$$

가 된다. 여기서 $y_1 - y_0$는 label separation, $g(I - \tilde A)(X_1 - X_0)$는 필터를 거친 feature separation이다. 즉 **필터링된 feature 차이가 label 차이와 잘 정렬될수록 오차 상한이 작아진다.** 참고로 필터 응답이 $g(1 - \lambda_i)$ 형태로 나타나는 이유는 증명에서 필터를 $g(I - \tilde A)$로 두기 때문이다. $\tilde A$의 고유값이 $\lambda_i$이면 $I - \tilde A$의 고유값은 $1 - \lambda_i$이므로 스펙트럼 응답이 $g(1 - \lambda_i)$가 된다.

마지막으로 이 표현을 graph Fourier 기저에서 다시 쓰면 $y_1 - y_0 \to \delta_i$, $X_1 - X_0 \to \eta_i$로 바뀌어 주파수별 곱 $\eta_i\, g(1 - \lambda_i)\, \delta_i$가 등장한다. 이는 "주파수 $i$에서 필터가 두 클래스를 구분하는 feature와 label 신호를 얼마나 잘 연결하는가"를 뜻한다. 그리고 이 alignment 합을 총 필터 응답 $\sum_i g_i$의 하한으로 묶은 뒤 Proposition 1을 대입하면, 총 필터 응답이 heterophily 스펙트럼에 의존하는 하한으로 바뀌면서 **heterophily가 비로소 오차 상한 안으로 들어온다.** 이것이 Theorem 1 전체에서 가장 중요한 개념적 다리다.

Theorem 1의 메시지는 명확하다. 기존의 단순한 사고는 "heterophily↑ ⇒ high-pass"였지만, 오차는 heterophily 스칼라 하나가 아니라 $|\hat h_i|$, $g(1 - \lambda_i)$, $\delta_i$, $\eta_i$라는 주파수별 성분들의 정렬로 결정된다. 따라서 **어느 주파수에 class-discriminative feature와 label 정보가 있고 heterophily가 어떻게 분포하는지에 필터가 적응해야 한다.** 성능은 주파수 의존적이며, 단일 heterophily 수준으로 결정되지 않는다.

## HeroFilter Architecture

이론이 말하는 바는 두 가지로 요약된다. 첫째, heterophily를 지닌 그래프의 스펙트럼 거동은 복잡하고 비단조적이어서 균일한 low-/high-pass 필터가 효과적이지 않다. 둘째, 서로 다른 주파수 대역은 그래프의 heterophily 구조에 따라 예측 성능에 서로 다르게 기여한다. HeroFilter는 이 결론을 그대로 설계로 옮긴 아키텍처이며, MLP-Mixer에서 영감을 받아 두 모듈로 구성된다.

### Patcher: 학습된 필터로 이웃을 다시 고르다

Patcher의 핵심 목표는 한 문장으로 요약된다. **그래프의 원래 이웃을 그대로 쓰지 말고, 학습된 스펙트럼 필터로 유용한 이웃을 다시 고르자.** 각 노드에 대해 적응형 다항 스펙트럼 필터를 학습하고, 이를 통해 topology상 바로 옆에 있는 이웃이 아니라 스펙트럼적으로 관련된 노드를 동적으로 선택한다. 이렇게 하면 단순한 local neighbor를 넘어 여러 spectral band에 걸쳐 문맥적으로 중요한 노드에 주목할 수 있다. 계산된 relevance로 노드를 정렬(ranking)한 뒤 상위 Top-$p$ 노드를 모아 각 노드의 그래프 패치를 구성한다.

### Mixer: 두 축에서 섞는 dual-axis MLP

Mixer는 사실상 MLP-Mixer와 같은 구조로, 선택된 패치를 spatial(patch) 축과 feature 축 양쪽에서 섞는 dual-axis MLP다. 두 축을 모두 섞는 이유가 설계의 핵심이다.

- **Patch-mixing**은 다른 노드가 특정 노드의 표현에 어떤 영향을 미치는지를 모델링해 노드 문맥의 heterophily를 포착한다.
- **Feature-mixing**은 raw input 특징을 추상적으로 변환해 노드 내부의 복잡성을 포착한다.

둘의 결합은 선택이 아니라 필수다. patch-mixing은 heterophily 영역에서 발생하는 구조적 불규칙성에 대한 adaptivity를 부여하고, feature-mixing은 expressive한 node-level 추론을 가능하게 한다. 하나만으로는 이론이 요구하는 "구조 적응 + 표현적 추론"을 동시에 충족할 수 없다.

### Fast-HeroFilter: 대규모 그래프를 위한 근사

Patcher가 스펙트럼 relevance를 직접 얻으려면 eigen-decomposition이 필요한데, 이는 대규모 그래프에서 비용이 지나치게 크다. Fast-HeroFilter는 이 값을 직접 계산하는 대신 personalized PageRank(PPR)와 유사한 multi-hop diffusion으로 근사한다. Section 4.1의 표현력 있는 eigenvalue-decomposition 기반 스펙트럼 patcher를 PPR-like diffusion으로 대체함으로써, Transformer 없이도 경량하고 scalable하게 대규모 그래프를 처리할 수 있게 한다.

## Experiments

HeroFilter는 homophily와 heterophily를 모두 포함하는 16개 벤치마크 데이터셋에서 광범위하게 검증되었다. 구체적 수치보다 claim 수준에서 정리하면 다음과 같다.

- **광범위한 검증**: 다양한 heterophily 수준을 아우르는 16개 데이터셋에서 경쟁력 있는 성능을 주장한다.
- **Ablation study**: (1) adaptive filter가 fixed/shared filter보다 유효한지, (2) relevance로 정렬한 패치가 무작위 순서보다 나은지, (3) patch-mixing과 feature-mixing이 각각 얼마나 기여하는지를 확인해 설계 선택을 뒷받침한다.
- **경량성**: Transformer 기반 graph model과 비교해 attention 없이 경량이면서도 경쟁력 있는 성능을 낸다고 주장한다.

## Takeaways

HeroFilter의 기여는 세 가지로 요약된다.

1. graph heterophily, 스펙트럼 필터 응답, 예측 오차를 처음으로 형식적으로 연결하는 이론적 분석을 제시하고, 단조로운 필터-heterophily 상관관계라는 기존 가정에 도전했다.
2. 경량 MLP-Mixer backbone과 adaptive 다항 필터를 결합한 모듈형 아키텍처로, 다양한 그래프 구조에서 효율적인 스펙트럼 추론을 가능하게 했다.
3. 16개 벤치마크에서 광범위한 실증 검증을 수행했다.

이론적 직관은 설득력이 있다. spatial heterophily와 spectral filtering 사이에 개념적 다리를 놓고, Definition 2 → Proposition 1 → Proposition 2 → Theorem 1로 이어지는 서사가 아키텍처 설계와 자연스럽게 맞물린다. "heterophily = 고주파"라는 이분법적 직관을 넘어 훨씬 세밀한 주파수별 관점을 제시한 점, 그리고 Patcher의 adaptive filter를 단순한 engineering trick이 아니라 이론적 동기와 연결한 점은 분명한 강점이다.

다만 객관적으로 보면 몇 가지 정밀화가 필요한 지점이 남는다.

- **Proposition 1의 가정.** 부등식 전개 과정에서 필터 가중치를 $\frac{g_i}{\sum_k g_k} \to \frac{1}{\sum_k g_k}$로 바꾸는 단계는 $0 < |\hat h_i| \le 1$ 같은 범위 조건에 기댄다. 그런데 $\hat h = U^\top h$이므로 spatial한 $h_i \in [0,1]$이라 해도 Fourier 변환을 거친 $|\hat h_i| \le 1$이 자동으로 보장되지는 않는다. 또한 최종 형태에서 분모로 쓰이는 항이 양수라는 부호 조건도 명시적으로 확보될 필요가 있다.
- **Proposition 2의 성격.** 이 명제는 "표현 가능(can represent)"을 보인 존재성 결과이지 "학습·일반화 보장"이 아니다. 적응형 필터가 임의의 target에 정렬될 파라미터를 가진다는 사실이, 실제 최적화가 그 파라미터를 찾거나 일반화한다는 것을 함의하지는 않는다. 또한 $g(\Lambda)$의 인덱스는 spectral index인 반면 $Y$는 raw node label vector처럼 정의되어 있어, 둘의 대응이 충분히 규정되었는지는 개념적으로 더 다듬을 여지가 있다.
- **Interpretability의 위상.** 저자들은 어떤 주파수를 강조했고 어떤 노드가 패치에 선택되었는지를 원칙적으로 들여다볼 수 있다는 의미에서 모델이 interpretable하다고 주장한다. 그러나 이는 **설계상 해석 가능(interpretability-by-design)** 이지 정량적으로 검증된 해석 가능성(validated interpretability)은 아니다.

종합하면 HeroFilter는 heterophily를 다루는 GNN 연구에서 "고정 스펙트럼 편향을 적응형 필터로 대체해야 한다"는 방향을 이론과 아키텍처 양면에서 설득력 있게 제시한 연구다. 이론적 서사는 매력적이며, 몇몇 증명 가정에 대한 정밀화가 뒤따른다면 그 기여는 한층 견고해질 것이다.
