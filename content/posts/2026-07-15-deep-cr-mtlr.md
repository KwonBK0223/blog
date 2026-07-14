---
title: "Deep-CR MTLR: A Multi-Modal Approach for Cancer Survival Prediction with Competing Risks"
date: 2026-07-15
tags: ["Survival Analysis", "Competing Risks", "Multi-Modal", "Medical AI"]
categories: ["Paper Review"]
series: ["Medical AI"]
math: true
summary: "치료 전 CT 영상과 임상 변수를 함께 입력받아, MTLR을 competing risks로 확장한 head로 cancer-specific death와 other-cause death의 cumulative incidence를 동시에 예측하는 multi-modal 생존 모델 Deep-CR MTLR을 리뷰한다."
---

> **Paper**: Deep-CR MTLR: A Multi-Modal Approach for Cancer Survival Prediction with Competing Risks — Sejin Kim, Michal Kazmierski, Benjamin Haibe-Kains, *PMLR (Survival Prediction — Algorithms, Challenges and Applications)*, 223–231, 2021. [arXiv:2012.05765](https://arxiv.org/abs/2012.05765)

## TL;DR

Deep-CR MTLR은 치료 전 CT volume과 임상 EMR 변수를 함께 입력으로 받아 환자별 생존 분포를 예측하는 multi-modal 생존 모델이다. 영상은 3D CNN이 voxel 단계에서 예후 표현을 학습하고, 임상 변수는 one-hot encoding과 normalization을 거친 뒤, 두 표현을 concat해 shared fully connected layer를 통과시킨다. 최종 head는 Multi-Task Logistic Regression(MTLR)을 competing risks로 확장한 것으로, event type마다 별도의 parameter set을 두어 "언제 사건이 발생하는가"와 "어떤 사건이 발생하는가"를 함께 예측하고, 그 결과를 시간 방향으로 누적해 event-specific cumulative incidence function(CIF)을 산출한다. 두경부암 코호트 RADCURE에서 multi-modal 모델은 단일 modality 모델을 전반적으로 능가했으며, cancer-specific death 예측에서 특히 이득이 컸다.

## Background

### 임상 변수만으로는 부족하다

암 환자의 예후 예측은 치료 방침 결정의 출발점이지만, 지금까지의 예측은 대체로 병기(stage), 나이, 수행 상태 같은 전통적인 임상 변수에 의존해 왔다. 이런 변수들은 해석은 쉽지만 환자 간 이질성을 충분히 설명하지 못한다. 반면 치료 전에 이미 촬영된 CT 영상에는 종양의 크기, 모양, 질감, 주변 조직과의 관계 같은 정량적 예후 정보가 풍부하게 들어 있다. 문제는 그 정보가 지나치게 고차원이라 사람이 직접 정의한 규칙으로는 필요한 신호만 골라내기 어렵다는 점이다.

이 간극을 메우기 위한 시도가 radiomics와 CNN 기반 생존 분석이다. 그러나 저자들은 기존 접근이 세 가지 한계를 공유한다고 지적한다.

- **영상만 사용한다.** 이미 존재하는 EMR의 임상 변수를 활용하지 않아, 두 modality가 상호 보완할 여지를 스스로 버린다.
- **대리(surrogate) task로 우회한다.** "2년 내 사망 여부"와 같은 이진 분류로 문제를 바꿔 풀면 시간 정보와 검열 정보가 소실되고, 생존 분포 자체를 예측하지 못한다.
- **강한 가정을 둔다.** Cox 계열의 비례 위험 가정처럼, 실제 데이터에서 성립한다는 보장이 없는 제약을 깔고 간다.

### Competing risks란 무엇인가

여기에 더해 저자들이 정면으로 다루는 문제 설정이 **competing risks**다. 한 환자에게 여러 종류의 사건이 발생할 수 있지만, 그 사건들이 서로 **배타적(mutually exclusive)**이어서 실제로는 그중 하나만 먼저 관찰되는 상황을 말한다.

두경부암 환자를 예로 들면 다음과 같다.

- **Event 1**: cancer-specific death (원발암으로 인한 사망)
- **Event 2**: death from other causes (다른 원인으로 인한 사망)
- **Event 0**: censoring (추적 관찰 종료)

한 환자가 cancer death를 먼저 경험하면 그 이후의 other-cause death는 **관찰 자체가 불가능**하다. 반대도 마찬가지다. 두 사건은 서로의 관찰을 가로막으며 경쟁한다. 이는 암 예후 예측에서 인위적으로 만들어낸 설정이 아니라 지극히 자연스러운 문제 구조다.

그런데 관행적으로 많이 쓰이는 overall survival은 사망의 원인을 구분하지 않고 모든 사망을 하나의 사건으로 합친다. 이렇게 하면 CT 영상이 실제로 기여하는 정보, 즉 종양 자체의 공격성에서 비롯되는 cancer-specific risk 신호가 노화나 동반 질환에서 비롯된 other-cause death와 뒤섞이며 희석된다. 영상의 기여를 제대로 보려면 사건 유형을 분리해 예측해야 한다는 것이 이 논문의 출발점이다.

## MTLR (Multi-Task Logistic Regression)

Deep-CR MTLR의 뼈대는 개인별 생존 분포를 직접 예측하는 모델인 MTLR이다.

### 시간축의 이산화

MTLR은 연속적인 생존 시간을 여러 개의 discrete time interval로 나눈다. 절단점(cut point)을 $0 = \tau_0 < \tau_1 < \cdots < \tau_m$이라 하면, $k$번째 구간은 $[\tau_{k-1}, \tau_k)$가 된다.

각 환자의 생존 시각 $T$는 이 이산 격자 위에서 이진 벡터 $\mathbf{y} = (y_1, \dots, y_{m-1})$로 인코딩된다.

$$
y_j = \mathbb{1}\left[T \le \tau_j\right]
$$

즉 사건이 $k$번째 구간에서 발생했다면 $\mathbf{y}$는 앞쪽이 0이고 $j \ge k$부터 1이 되는 계단 형태의 벡터다.

### Dependent logistic regressors

각 time interval마다 하나의 logistic regression task를 두지만, 이들을 **독립적인 이진 분류기의 나열로 학습하지 않는다**. 독립적으로 학습하면 "3년 시점에는 사망했는데 5년 시점에는 생존"처럼 시간 순서와 모순되는 예측이 나올 수 있다. MTLR은 대신 계단 형태의 벡터 $\mathbf{y}$ 전체에 확률을 부여하는 방식으로 시간 방향의 일관성을 구조적으로 강제한다. 사건이 $k$번째 구간에서 발생할 확률은 다음과 같이 쓰인다.

$$
P\left(T \in [\tau_{k-1}, \tau_k) \mid x\right) = \frac{\exp\left(\sum_{j=k}^{m-1} \left(\theta_j^\top x + b_j\right)\right)}{\sum_{l=1}^{m} \exp\left(\sum_{j=l}^{m-1} \left(\theta_j^\top x + b_j\right)\right)}$$

분자는 해당 계단 벡터의 점수이고, 분모는 가능한 모든 사건 시점에 대한 정규화 항이다. 이 구조 덕분에 모델의 출력은 개별 시점의 독립적 확률이 아니라 **환자 하나에 대한 완전한 생존 분포**가 된다.

### 검열의 처리

우측 검열된 환자는 사건이 언제 발생했는지 모르지만, "적어도 검열 시각까지는 사건이 없었다"는 정보는 가지고 있다. MTLR은 검열 시각 이후에 사건이 발생할 수 있는 모든 time interval을 **marginalize**해 우도(likelihood)에 반영한다. 검열 구간을 $c$라 하면

$$
P\left(T \ge \tau_{c} \mid x\right) = \sum_{k > c} P\left(T \in [\tau_{k-1}, \tau_k) \mid x\right)
$$

이 값을 그대로 우도에 넣는다. 검열된 샘플을 버리거나 관찰 시각을 사건 시각으로 오인하지 않고, 부분적인 정보를 있는 그대로 학습에 사용하는 지점이다.

## Method

### Neural MTLR: 신경망인데 왜 logistic regression인가

MTLR은 본래 입력 $x$에 대한 선형 결합 $\theta_j^\top x$를 사용하는 logistic regression 기반 모델이다. Neural MTLR은 이 **입력 feature를 신경망이 학습한 representation으로 교체**한 것이다. 즉 마지막 생존 예측 head는 MTLR 구조를 그대로 유지하되, 그 앞단의 feature extractor를 신경망으로 바꾼다.

$$
\theta_j^\top x \;\longrightarrow\; \theta_j^\top h_\phi(x)
$$

여기서 $h_\phi(\cdot)$는 파라미터 $\phi$를 가진 신경망이다. "Neural"과 "logistic regression"이 모순되어 보이지만, 비선형화되는 것은 표현 학습 단계이고 생존 분포를 구성하는 head는 여전히 MTLR이라는 뜻이다.

### Deep-CR MTLR 아키텍처

전체 구조는 두 개의 branch와 하나의 fusion 단계로 이루어진다.

- **Clinical branch**: categorical 변수는 one-hot encoding, continuous 변수는 normalization을 거쳐 임상 표현 $h_{\text{clin}}$을 만든다.
- **Image branch**: 치료 전 CT volume을 3D CNN에 입력해 voxel 단계에서 곧바로 예후 표현 $h_{\text{img}}$를 학습한다. 사람이 미리 정의한 radiomics feature를 사용하지 않고 end-to-end로 학습한다는 점이 핵심이다.
- **Fusion**: 두 표현을 concat한 뒤 shared fully connected layer를 통과시킨다.

$$
h = \mathrm{FC}\left(\left[\, h_{\text{img}} \;;\; h_{\text{clin}} \,\right]\right)
$$

이렇게 얻은 공유 표현 $h$가 competing risks MTLR head로 전달된다.

### Competing risks 확장

일반적인 생존 모델은 하나의 사건만 예측한다. Deep-CR MTLR은 MTLR을 competing risks 설정으로 확장해, **event type마다 별도의 MTLR parameter set**을 둔다. 사건 유형 $e \in \{1, \dots, E\}$에 대해 파라미터 $\left(\theta_j^{(e)}, b_j^{(e)}\right)$를 각각 학습하며, 모델은 "언제 사건이 발생하는가"와 "어떤 사건이 발생하는가"를 동시에 예측한다.

출력은 (time interval × event type)의 **joint probability**다.

$$
P\left(T \in [\tau_{k-1}, \tau_k),\; D = e \mid x\right) = \frac{\exp\left(\sum_{j=k}^{m-1}\left(\theta_j^{(e)\top} h + b_j^{(e)}\right)\right)}{\displaystyle\sum_{e'=1}^{E} \sum_{l=1}^{m} \exp\left(\sum_{j=l}^{m-1}\left(\theta_j^{(e')\top} h + b_j^{(e')}\right)\right)}
$$

여기서 $D \in \{1, \dots, E\}$는 실제로 발생한 사건의 유형이다. 정규화가 모든 event type과 모든 time interval에 걸쳐 이루어지므로, 사건들의 배타성이 출력 구조에 직접 반영된다.

이 joint probability를 시간 방향으로 누적하면 **event-specific cumulative incidence function(CIF)**이 된다.

$$
F_e(t \mid x) = P\left(T \le t,\; D = e \mid x\right) = \sum_{k \,:\, \tau_k \le t} P\left(T \in [\tau_{k-1}, \tau_k),\; D = e \mid x\right)
$$

$F_e(t \mid x)$는 "이 환자에게 시각 $t$ 이전에 사건 $e$가 발생할 확률"이다. 예를 들어 $F_1(2\text{년} \mid x)$는 2년 안에 cancer death가 발생할 확률, $F_2(2\text{년} \mid x)$는 2년 안에 other-cause death가 발생할 확률이 되며, 두 값을 분리해 얻을 수 있다는 것이 competing risks 확장의 실질적 산출물이다.

## Experiments

### 데이터셋과 설정

- **Dataset**: RADCURE. 두경부암 환자 2,552명의 치료 전 CT 영상과 임상 변수로 구성된다.
- **Events**: death from primary cancer, death from other causes, censoring의 세 가지로 정의된다.
- **Split**: 진단 일자(diagnosis date)를 기준으로 train / validation / test를 나눈다. 무작위 분할이 아니라 시간 기준 분할이므로, 향후 환자에 대한 일반화를 좀 더 현실적으로 평가하는 설정이다.

### Baseline과 평가 지표

Baseline으로는 임상 변수만 사용하는 linear MTLR, 영상만 사용하는 convnet, 그리고 임상 변수만 사용하는 Neural MTLR이 비교된다. 즉 (1) modality를 하나만 쓰는 경우와 (2) 표현 학습 없이 선형 모델을 쓰는 경우를 각각 통제한다.

평가 지표는 두 가지다.

- **2-year AUROC**: 2년 시점의 사건 발생 여부에 대한 판별력
- **Cause-specific C-index**: 사건 유형별로 계산한 순위 일치도. 실제로 더 빨리 해당 사건을 겪은 환자에게 모델이 더 높은 위험을 부여했는지를 측정한다.

### 결과

- Multi-modal 모델이 단일 modality 모델들보다 전반적으로 우수했다. 영상과 임상 정보는 서로를 대체하지 않고 보완한다.
- 성능 향상은 특히 **cancer-specific survival 예측**에서 두드러졌다. 이는 배경에서 제기한 가설, 즉 영상 정보가 주로 암 자체의 공격성을 반영하며 overall survival로 모든 사망을 합치면 그 신호가 희석된다는 주장과 일관된 결과다.
- Deep-CR MTLR은 cancer death에 대해 2-year AUROC 0.774, C-index 0.788을 기록했다.
- Deep 모델이 linear MTLR을 능가했다는 점은 임상 변수와 영상 feature 사이의 **비선형 상호작용**을 학습하는 것이 실제로 이득이 된다는 근거가 된다.

## Takeaways

- Deep-CR MTLR은 CT 영상과 임상 EMR을 결합하고, competing risks로 확장한 MTLR head를 통해 cancer-specific death와 other-cause death의 위험을 동시에 예측하는 multi-modal 생존 모델이다. 대리 task로 우회하지 않고 생존 분포를 직접 예측한다는 점, 그리고 사건 유형을 합치지 않고 분리한다는 점이 설계의 두 축이다.
- **시간 이산화 구간의 선택에 민감할 수 있다.** MTLR의 출력 구조 자체가 절단점 $\{\tau_k\}$의 개수와 위치에 의존하므로, 이 하이퍼파라미터가 성능과 CIF의 해상도를 함께 좌우한다. 연속 시간 모델에는 없는 추가적인 설계 부담이다.
- **Cox, DeepSurv, DeepHit과의 직접 비교가 없다.** 저자들은 이를 future work로 남겼는데, 특히 DeepHit은 competing risks를 다루는 대표적인 딥러닝 모델이므로 비교 부재는 아쉬운 지점이다. 현재의 baseline은 주로 자기 자신의 ablation에 가깝다.
- **해석 가능성이 제한적이다.** 영상이 예후 예측에 기여한다는 사실은 보였지만, CNN이 CT의 어떤 소견에 반응하는지는 열려 있다. 임상 채택에는 이 부분이 걸림돌로 남는다.
- **Competing risks 예측이 곧바로 치료 결정으로 이어지지는 않는다.** 모델이 산출하는 것은 관찰 데이터에 기반한 event-specific risk이지 개입의 인과 효과가 아니다. "이 환자는 2년 내 cancer death 위험이 높다"는 예측과 "그러므로 어떤 치료를 해야 한다"는 결정 사이에는 인과 추론의 층위가 하나 더 필요하다.
