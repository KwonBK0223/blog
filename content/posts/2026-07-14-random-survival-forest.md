---
title: "Random Survival Forests"
date: 2026-07-14
tags: ["Survival Analysis", "Random Forest", "Ensemble", "Medical AI"]
categories: ["Paper Review"]
series: ["Medical AI"]
math: true
summary: "Breiman의 Random Forest를 우측 검열된 생존 데이터로 확장해, split rule과 terminal node prediction을 survival-specific하게 재설계하고 앙상블 cumulative hazard function을 예측하는 비모수 생존 모델을 리뷰한다."
---

> **Paper**: Random Survival Forests — Hemant Ishwaran, Udaya B. Kogalur, Eugene H. Blackstone, Michael S. Lauer, *The Annals of Applied Statistics* 2(3), 841–860, 2008. [DOI:10.1214/08-AOAS169](https://doi.org/10.1214/08-AOAS169)

## TL;DR

Random Survival Forests(RSF)는 Breiman의 Random Forest를 우측 검열(right-censored) 생존 데이터에 맞게 확장한 비모수 앙상블 모델이다. Bootstrap 샘플로 여러 개의 survival tree를 키우고, 각 노드에서 변수의 무작위 부분집합만 후보로 삼는다는 점은 일반 RF와 동일하다. 차이는 outcome이 class label이나 연속값이 아니라 생존 데이터라는 데 있으며, 그 결과 split rule, prediction, error 계산이 모두 survival-specific하게 바뀐다. 각 terminal node에서 Nelson-Aalen 추정량으로 cumulative hazard function(CHF)을 계산하고, 트리별 CHF를 평균해 환자별 ensemble CHF를 얻는다. 여기서 파생되는 ensemble mortality는 해석 가능한 예측량이며, out-of-bag(OOB) 데이터를 이용하면 별도의 validation set 없이 prediction error와 C-index를 추정할 수 있다.

## Background

### Survival data와 우측 검열

생존 분석은 "사건이 발생하기까지 걸리는 시간"을 모델링한다. 환자 $i$의 데이터는 세 가지 요소로 구성된다.

- **Covariates $x_i$**: 나이, 성별, 병기, 검사 수치, 치료 정보 등 예측 변수
- **관찰 시간 $T_i$**: 사건 발생 시각 또는 마지막 추적 시각
- **사건 지표 $\delta_i$**: $\delta_i = 1$이면 사건이 관찰된 경우, $\delta_i = 0$이면 우측 검열

우측 검열이란 마지막 추적 시점까지는 사건이 발생하지 않았지만 그 이후 어떻게 되었는지 알 수 없는 경우를 말한다. 검열된 샘플을 단순히 버리거나 사건 발생으로 취급하면 심각한 편향이 생기므로, 생존 모델의 split 기준과 error 계산은 검열을 명시적으로 다뤄야 한다. 표준적인 회귀 트리나 분류 트리를 그대로 쓸 수 없는 이유가 여기에 있다.

### CoxPH의 가정

생존 분석의 표준 도구인 Cox proportional hazards 모델은 hazard를 baseline hazard와 환자별 risk score의 곱으로 분해한다.

$$
h(t \mid x) = h_0(t) \exp\big(\beta^\top x\big)
$$

이 형태는 두 가지 강한 가정을 깔고 있다. 첫째는 **proportional hazards** 가정으로, 두 환자의 hazard 비가 시간에 관계없이 일정하다는 것이다. 둘째는 **linearity** 가정으로, log hazard가 covariate의 선형 결합이라는 것이다. 비선형 효과나 고차 interaction을 반영하려면 연구자가 직접 항을 지정해야 하는데, covariate가 많아질수록 이는 사실상 불가능해진다.

### 비모수적 앙상블의 필요성

Random Forest는 bootstrap 샘플과 무작위 변수 선택을 통해 복잡한 interaction을 자동으로 포착하며, 무관한 noise 변수에도 강건하다. 그러나 표준 RF는 검열된 outcome을 직접 다루지 못한다. RSF는 RF의 앙상블 구조를 유지하면서 검열을 정면으로 다루는 비모수 생존 모델을 목표로 한다. Proportional hazards도, linearity도 가정하지 않는다.

## Method

### 알고리즘 절차

RSF의 학습 절차는 다음과 같다.

1. 원본 데이터에서 bootstrap 샘플을 $B$번 뽑는다. 각 샘플에서 약 63%의 관측치가 in-bag으로 선택되고, 나머지 약 37%는 out-of-bag(OOB)으로 남는다.
2. 각 bootstrap 샘플로 survival tree를 하나씩 키운다.
3. 각 노드에서 전체 변수 중 무작위로 $p$개만 split 후보로 뽑는다.
4. 후보 변수와 split point 중에서 **survival difference를 최대화하는** split을 선택한다.
5. Terminal node가 사건 수에 대한 최소 조건을 만족할 때까지 트리를 성장시킨다.
6. 각 terminal node에서 Nelson-Aalen 추정량으로 CHF를 계산한다.
7. 트리별 CHF를 평균해 ensemble CHF를 만든다.
8. OOB 데이터로 prediction error를 추정한다.

### Binary survival tree와 splitting rule

RSF의 트리는 CART처럼 노드를 두 개의 daughter node로 나누는 binary tree다. 일반 decision tree와 구조는 같지만, split 기준이 impurity나 variance reduction이 아니라 **survival difference**라는 점이 다르다.

좋은 split은 두 daughter node의 생존 양상을 최대한 다르게 만드는 split이다. 대표적인 기준이 **log-rank splitting rule**로, 후보 split $(v, c)$에 대해 왼쪽·오른쪽 daughter node의 생존 곡선 차이를 log-rank 통계량으로 측정한다.

$$
L(v, c) = \frac{\sum_{k=1}^{K} \left( d_{1,k} - Y_{1,k} \dfrac{d_k}{Y_k} \right)}{\sqrt{\sum_{k=1}^{K} \dfrac{Y_{1,k}}{Y_k} \left( 1 - \dfrac{Y_{1,k}}{Y_k} \right) \left( \dfrac{Y_k - d_k}{Y_k - 1} \right) d_k}}
$$

여기서 $t_1 < \cdots < t_K$는 노드 내의 서로 다른 사건 시각이고, $d_k$와 $Y_k$는 시각 $t_k$에서의 사건 수와 risk set 크기, $d_{1,k}$와 $Y_{1,k}$는 왼쪽 daughter node에서의 대응값이다. $|L(v, c)|$가 클수록 두 그룹의 생존 양상이 크게 갈린다는 뜻이므로, 이 값을 최대화하는 split을 채택한다. 결과적으로 트리는 survival risk가 다른 환자들을 잘 분리하는 방향으로 성장한다.

논문은 log-rank 외에 log-rank score, conservation-of-events 등 여러 splitting rule을 함께 검토한다. 검열을 명시적으로 다루는 통계량이라는 점이 공통이다.

### Terminal node와 Nelson-Aalen 추정량

Terminal node는 더 이상 split하지 않는 최종 노드다. 같은 terminal node에 들어간 환자들은 비슷한 survival pattern을 가진다고 보고, 하나의 CHF를 공유한다.

Terminal node $h$에서의 CHF는 Nelson-Aalen 추정량으로 계산된다.

$$
\hat{H}_h(t) = \sum_{t_{k,h} \le t} \frac{d_{k,h}}{Y_{k,h}}
$$

여기서 $t_{k,h}$는 노드 $h$ 내의 사건 시각, $d_{k,h}$는 그 시각의 사건 수, $Y_{k,h}$는 그 시각의 risk set 크기다. CHF는 시간에 따라 누적된 event risk를 나타내며, 정의상 단조 증가한다. **CHF가 높을수록 그 시점까지 누적된 사망·사건 위험이 크다.**

새 환자 $x$가 트리를 타고 내려가 terminal node $h$에 도달하면, 그 노드의 CHF가 그 환자의 예측값이 된다.

$$
H(t \mid x) = \hat{H}_h(t), \quad x \in h
$$

### 일반 RF와의 차이

기본 구조는 RF와 같지만, 결정적인 차이는 **출력이 무엇인가**에 있다.

| | 일반 Random Forest | Random Survival Forest |
|---|---|---|
| Outcome | Class label / 연속값 | Right-censored survival data |
| Split 기준 | Gini impurity / variance reduction | Log-rank 등 survival difference |
| Terminal node 예측 | 다수결 / 평균 | Nelson-Aalen CHF |
| 앙상블 결과 | 예측 class / 예측값 | Ensemble CHF, survival curve |
| Error 추정 | OOB misclassification / MSE | OOB C-index 기반 prediction error |

즉 RSF의 예측은 하나의 스칼라가 아니라 **시간에 대한 함수**다. 환자마다 하나의 곡선이 나온다.

### Missing data

논문은 covariate, 생존 시간, 검열 지표에 결측이 있어도 학습과 예측이 가능하도록 **adaptive tree imputation**을 제안한다. 목적은 결측값을 정확히 복원하는 것이 아니라, 결측이 있어도 split과 prediction을 안정적으로 수행하는 것이다. 각 노드에서 결측이 없는 in-bag 데이터로부터 값을 무작위로 뽑아 결측을 채우며, test data의 결측도 같은 방식으로 처리한다.

## Prediction & Interpretation

### Ensemble CHF

$B$개의 트리 각각이 환자 $x$에 대해 CHF $H_b(t \mid x)$를 내놓는다. Forest 수준의 예측은 이들의 평균이다.

$$
H_e(t \mid x) = \frac{1}{B} \sum_{b=1}^{B} H_b(t \mid x)
$$

OOB 버전은 해당 환자를 in-bag으로 쓰지 않은 트리들만 평균한다. $I_{i,b} = 1$을 "환자 $i$가 트리 $b$의 OOB에 속함"이라 하면,

$$
H_e^{**}(t \mid x_i) = \frac{\sum_{b=1}^{B} I_{i,b} \, H_b(t \mid x_i)}{\sum_{b=1}^{B} I_{i,b}}
$$

이 OOB ensemble CHF는 해당 환자를 한 번도 학습에 쓰지 않은 트리들로만 만들어진 예측이므로, 별도의 hold-out set 없이도 정직한 성능 추정의 근거가 된다.

### Ensemble mortality

CHF는 곡선이라 환자 간 비교가 번거롭다. 논문은 CHF를 관측된 사건 시각 전체에 걸쳐 합산해 하나의 스칼라로 요약한 **ensemble mortality**를 정의한다.

$$
M_e(x_i) = \sum_{k=1}^{K} H_e(t_k \mid x_i)
$$

이 값은 "모든 환자가 환자 $i$와 같은 covariate를 가진다고 가정했을 때 기대되는 총 사건 수"로 해석된다. 즉 환자별 예상 누적 사망 위험을 하나의 수치로 표현한 것이며, risk ranking과 C-index 계산에 그대로 쓸 수 있다. 값이 클수록 위험이 높은 환자다.

### OOB 기반 error 추정

Bootstrap 샘플에 포함되지 않은 약 37%의 데이터가 OOB다. 각 트리는 in-bag 데이터로 학습되고, 그 트리에 들어가지 않은 OOB 데이터로 평가된다. RSF는 OOB ensemble CHF로부터 OOB mortality를 계산하고, 이를 이용해 C-index와 prediction error를 추정한다.

$$
\text{Error} = 1 - \text{C-index}
$$

C-index는 두 환자를 비교했을 때 실제로 더 빨리 사건이 발생한 환자에게 모델이 더 높은 risk를 부여했는지를 보는 순위 기반 지표로, 0.5는 무작위 수준이다. 따라서 prediction error가 0.5보다 작아야 의미 있는 모델이며, cross-validation이나 별도 validation set 없이 학습 과정에서 곧바로 얻어진다는 점이 실무적으로 큰 장점이다.

### Variable importance

VIMP(variable importance)는 특정 변수의 값을 무작위로 섞거나 split을 무작위로 보낸 뒤 OOB prediction error가 얼마나 악화되는지로 측정한다. Error 증가가 크면 그 변수는 중요하다는 뜻이고, 0에 가깝거나 음수면 noise에 가깝다. Noise 변수를 대량으로 주입한 breast cancer 데이터 실험에서 RSF는 실제 신호 변수와 noise 변수를 안정적으로 구분해냈다.

실험에서 RSF는 여러 실제·시뮬레이션 데이터셋에서 Cox regression이나 censored RF regression보다 낮은 prediction error를 보였고, noise 변수가 늘어나도 성능이 크게 무너지지 않았다. CABG case study에서는 BMI, 신기능, 흡연이 얽힌 복잡한 생존 interaction을 사전 지정 없이 드러냈다.

## Takeaways

- RSF는 Random Forest의 bootstrap 앙상블 구조를 유지하면서 splitting rule과 terminal node prediction을 survival-specific하게 바꾼 비모수 생존 모델이다. Proportional hazards도 linearity도 가정하지 않는다.
- 예측 산출물이 CHF라는 점이 실용적으로 중요하다. Risk score 하나가 아니라 시간에 대한 곡선을 얻으므로 임상적 해석의 여지가 넓고, ensemble mortality로 요약하면 곧바로 순위 비교가 가능하다.
- OOB 구조 덕분에 별도의 validation set 없이 prediction error와 C-index를 추정할 수 있다. 표본이 귀한 임상 데이터에서 특히 유용한 성질이다.
- 다만 **앙상블 전체는 여전히 blackbox 성격을 벗어나지 못한다.** 개별 survival tree는 사람이 따라 읽을 수 있지만, 수백~수천 개 트리의 평균을 사람이 직관적으로 해석할 방법은 없다. 해석은 VIMP, partial plot, CHF 시각화 같은 사후적 도구에 의존하며, 개별 feature effect가 CoxPH의 계수처럼 닫힌 수식으로 주어지지 않는다. Causal interpretation은 더욱 어렵다.
- **고차원에서의 계산 비용이 만만치 않다.** 모든 노드에서 후보 변수마다 log-rank 통계량을 계산하며 최적 split point를 탐색해야 하므로, 변수 수와 트리 수가 늘어나면 비용이 빠르게 증가한다.
- **Splitting rule 선택에 대한 민감도**도 남는 문제다. Log-rank, log-rank score, conservation-of-events 중 무엇을 쓰느냐에 따라 트리 구조와 VIMP 순위가 달라질 수 있으며, 어떤 상황에서 어떤 규칙이 최적인지에 대한 일반론은 확립되어 있지 않다.
