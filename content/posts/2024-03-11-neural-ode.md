---
title: "Neural Ordinary Differential Equations"
date: 2024-03-11
tags: ["Neural ODE", "Continuous-depth Models", "Normalizing Flow"]
categories: ["Paper Review"]
series: ["Classic Papers"]
math: true
summary: "이산적인 hidden layer의 시퀀스 대신 hidden state의 도함수를 신경망으로 매개변수화하고 ODE solver로 출력을 계산하는 continuous-depth 모델과, 상수 메모리로 역전파를 가능하게 하는 adjoint sensitivity method를 리뷰한다."
---

> **Paper**: Neural Ordinary Differential Equations — Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud, *NeurIPS 2018* (Best Paper). [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)

## TL;DR

- ResNet의 residual 업데이트 $h_{t+1} = h_t + f(h_t, \theta_t)$는 연속적인 변환의 **Euler discretization**으로 볼 수 있다. 레이어를 무한히 잘게 나눈 극한에서, hidden state의 변화를 ODE $\frac{dh(t)}{dt} = f(h(t), t, \theta)$로 매개변수화한 것이 Neural ODE다.
- 출력은 블랙박스 ODE solver로 적분해서 얻고, 학습은 **adjoint sensitivity method**로 또 하나의 ODE를 역방향으로 풀어 수행한다. 덕분에 forward 연산을 저장할 필요가 없어 **메모리 비용이 $\mathcal{O}(1)$** 이다.
- 정확도와 속도를 solver의 오차 허용치로 명시적으로 교환할 수 있고(적응형 계산), Jacobian determinant 계산이 trace 연산으로 단순해지는 **Continuous Normalizing Flow(CNF)**, 불규칙 샘플링 시계열을 다루는 **latent ODE** 등의 응용이 열린다.

## Background

### 미분방정식과 초기값 문제

미분방정식(DE)은 종속변수를 독립변수에 대해 미분한 도함수를 포함하는 방정식이다. 단변수면 상미분방정식(ODE), 다변수면 편미분방정식(PDE)이다. "ODE를 푼다"는 것은 도함수 관계를 만족하는 원래 함수(해)를 찾는 것, 즉 적분하는 것과 같다. 예를 들어 $f'(x) - 2x = 0$의 해는 $f(x) = x^2 + C$다.

초기 조건이 주어졌을 때 해를 구하는 문제를 **초기값 문제(IVP)** 라고 하며, 해석적으로 풀 수 없을 때는 수치적 방법을 쓴다. 가장 단순한 것이 **Euler method**로, 구간을 작은 step $h$로 나누어 각 지점의 기울기로 다음 값을 예측한다.

$$y_{n} = y_{n-1} + h\,\frac{\partial y_{n-1}}{\partial x_{n-1}}$$

step을 펼쳐 쓰면 $y_n = y_1 + \sum_{i=1}^{n-1} h \frac{\partial y_i}{\partial x_i}$가 되는데, step 크기 $\Delta x$가 작을수록 수치해가 실제 해에 가까워진다. 이 형태가 ResNet의 skip connection과 정확히 같은 구조라는 것이 이 논문의 출발점이다.

## Method

### Residual Network에서 ODE Network로

ResNet, RNN decoder, normalizing flow 같은 모델들은 변환의 시퀀스를 hidden state로 쌓아 복잡한 변환을 구축한다.

$$h_{t+1} = h_t + f(h_t, \theta_t), \quad t \in \{0, \dots, T\},\; h_t \in \mathbb{R}^D$$

이 반복적 업데이트는 연속 변환의 Euler discretization이다. 레이어를 더 많이 추가하고 step을 더 작게 하는 극한에서, hidden unit의 연속적인 역학을 신경망으로 매개변수화할 수 있다.

$$\frac{dh(t)}{dt} = f(h(t), t, \theta)$$

이제 이산적인 레이어들의 스택은 하나의 연속적인 "ODE 구간"으로 대체된다. 입력에 해당하는 초기 상태 $z(0)$에서 출력에 해당하는 최종 상태 $z(1)$까지의 변환은 적분으로 계산된다.

$$z(1) = z(0) + \int_0^1 f(z(t), t, \theta)\, dt$$

이 적분은 Euler method, Runge–Kutta 4 등 어떤 ODE solver로도 풀 수 있으며, 현대적 solver는 오차를 모니터링하며 평가 지점을 적응적으로 조절한다. 즉 **모델의 "깊이"가 고정된 하이퍼파라미터가 아니라 solver가 입력마다 결정하는 양**이 된다.

### Adjoint Sensitivity Method: 상수 메모리 역전파

ODE solver의 내부 연산을 그대로 자동미분하면 forward의 모든 중간값을 저장해야 해서 메모리가 크게 든다. 논문은 대신 **adjoint sensitivity method**를 사용한다.

각 시점의 gradient를 adjoint state로 정의한다.

$$a(t) = \frac{\partial \text{Loss}}{\partial z(t)}$$

이 adjoint state 자체가 또 하나의 ODE를 따른다.

$$\frac{da(t)}{dt} = -a(t)^{\top} \frac{\partial f(z(t), t, \theta)}{\partial z}$$

따라서 $a(1)$에서 출발해 시간을 거꾸로 적분하면 $a(0)$, 즉 입력에 대한 gradient를 얻는다.

$$a(0) = a(1) - \int_1^0 \frac{\partial a(t)}{\partial t}\, dt \;\Longleftrightarrow\; \frac{\partial \text{Loss}}{\partial z(0)} = \frac{\partial \text{Loss}}{\partial z(1)} - \int_1^0 \frac{\partial a(t)}{\partial t}\, dt$$

파라미터 $\theta$에 대한 gradient도 같은 방식으로 $-a(t)^{\top}\frac{\partial f}{\partial \theta}$를 함께 적분해 얻는다. 요컨대 **forward도 ODE 풀기, backward도 ODE 풀기**다. 중간 활성값을 저장할 필요가 없으므로 메모리 비용이 깊이에 대해 $\mathcal{O}(1)$이고, gradient가 ODE 구간 앞단의 conv layer까지 자연스럽게 흘러가 전체 모델을 end-to-end로 학습할 수 있다.

## Experiments

### 지도학습: MNIST

ResNet을 ODE 블록으로 대체한 ODE-Net은 MNIST에서 test error 0.42%로 ResNet(0.41%)과 동등한 성능을 내면서, 파라미터는 약 1/3(0.22M vs 0.60M), 메모리는 $\mathcal{O}(\tilde{L})$이 아닌 $\mathcal{O}(1)$을 달성했다. 여기서 $\tilde{L}$은 ODE solver가 한 번의 forward pass에서 요청한 함수 평가 횟수(NFE)다.

학습된 ODE-Net의 통계도 흥미롭다. (a) NFE가 늘수록 수치 오차는 줄고(정밀도-속도의 명시적 교환), (b) NFE와 계산 시간은 비례하며, (c) backward의 NFE는 forward의 약 절반이고, (d) 학습이 진행될수록 NFE가 증가한다 — 모델이 점점 복잡한 역학을 학습한다는 뜻이다.

별도로 영어 대문자 손글씨 데이터에 대한 재현 실험에서도 Neural ODE(Euler method 직접 구현/자동미분 버전)는 CNN·ResNet과 동등한 검증 정확도(약 99.2~99.3%)를 보였다.

### Continuous Normalizing Flow

Normalizing flow에서 변수 변환 공식은 Jacobian determinant를 요구한다.

$$z_1 = f(z_0) \;\Rightarrow\; \log p(z_1) = \log p(z_0) - \log \left| \det \frac{\partial f}{\partial z_0} \right|$$

이 determinant 계산은 hidden unit 수의 세제곱에 비례하는 비용이 든다. 그런데 변환을 연속 시간으로 옮기면 로그 확률의 변화가 **trace 연산만으로** 계산된다.

$$\frac{\partial \log p(z(t))}{\partial t} = -\text{tr}\left( \frac{df}{dz(t)} \right)$$

이산적인 레이어의 집합에서 연속적인 변환으로 이동하면 normalizing 상수의 변화 계산이 단순해지는 것이다. 여기에 hidden unit별 gating $\frac{dz}{dt} = \sum_n \sigma_n(t) f_n(z)$을 도입한 것이 CNF이며, 같은 깊이/폭 예산에서 NF보다 낮은 loss로 목표 분포를 근사했다.

### Latent ODE: 불규칙 시계열

기존 RNN은 불규칙하게 샘플링된 시계열을 다루기 어렵다. 논문은 RNN encoder로 관측치들을 잠재 초기 상태 $z_{t_0}$의 분포로 인코딩하고, latent trajectory를 ODE로 전개해 임의의 시점에서 디코딩하는 생성 모델(latent ODE)을 제안한다. 나선형 궤적 실험에서 관측 비율이 30/100, 50/100, 100/100일 때 예측 RMSE가 RNN(0.3937/0.3202/0.1813)보다 latent ODE(0.1642/0.1502/0.1346)가 일관되게 낮았고, 특히 관측이 희박할수록 격차가 컸다.

## Takeaways

**장점**
1. **메모리 효율성** — adjoint method 덕분에 깊이와 무관한 상수 메모리.
2. **적응형 계산** — 입력과 요구 정밀도에 따라 solver가 평가 전략을 조절.
3. **확장 가능하고 가역적인 normalizing flow** — determinant가 trace로 단순화.
4. **연속 시간 시계열 모델** — 불규칙 샘플링 데이터의 자연스러운 처리.

**한계**
1. **미니배치** — 일반 신경망보다 사용이 복잡하다.
2. **해의 유일성** — 신경망 가중치가 유한하고 tanh, ReLU 같은 Lipschitz 비선형성을 쓸 때 보장된다(Picard 정리).
3. **오차 허용치 설정** — forward와 backward 모두에서 tolerance를 선택해야 한다.
4. **Forward 궤적 재구성** — 역방향 적분으로 궤적을 재구성할 때 수치 오차가 누적될 수 있으며, 체크포인트로 완화할 수 있다.

"깊이"라는 이산적 개념을 연속 역학으로 재해석한 이 논문은 이후 continuous-depth 모델, neural SDE/CDE, flow 기반 생성 모델과 시계열 모델링 연구의 기반이 된 현대의 고전이다.
