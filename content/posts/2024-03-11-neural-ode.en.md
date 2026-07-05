---
title: "Neural Ordinary Differential Equations"
date: 2024-03-11
tags: ["Neural ODE", "Continuous-depth Models", "Normalizing Flow"]
categories: ["Paper Review"]
series: ["AI Algorithms"]
math: true
summary: "A review of continuous-depth models that replace discrete sequences of hidden layers by parameterizing the derivative of the hidden state with a neural network and computing outputs with an ODE solver, along with the adjoint sensitivity method that enables backpropagation with constant memory."
---

> **Paper**: Neural Ordinary Differential Equations — Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud, *NeurIPS 2018* (Best Paper). [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)

## TL;DR

- The residual update in ResNet, $h_{t+1} = h_t + f(h_t, \theta_t)$, can be viewed as an **Euler discretization** of a continuous transformation. In the limit of infinitely fine layers, parameterizing the dynamics of the hidden state as the ODE $\frac{dh(t)}{dt} = f(h(t), t, \theta)$ gives us a Neural ODE.
- Outputs are computed by integrating with a black-box ODE solver, and training is done via the **adjoint sensitivity method**, which solves a second ODE backwards in time. Since no forward activations need to be stored, the **memory cost is $\mathcal{O}(1)$**.
- Accuracy can be traded off explicitly against speed through the solver's error tolerance (adaptive computation), and the framework opens up applications such as **Continuous Normalizing Flows (CNF)**, where the Jacobian determinant computation reduces to a trace operation, and **latent ODEs** for irregularly-sampled time series.

## Background

### Differential Equations and Initial Value Problems

A differential equation (DE) is an equation involving derivatives of a dependent variable with respect to one or more independent variables. With a single independent variable it is an ordinary differential equation (ODE); with several, a partial differential equation (PDE). "Solving an ODE" means finding the original function (the solution) that satisfies the derivative relation — that is, integrating. For example, the solution of $f'(x) - 2x = 0$ is $f(x) = x^2 + C$.

Finding the solution given an initial condition is called an **initial value problem (IVP)**; when no analytic solution exists, we resort to numerical methods. The simplest is the **Euler method**, which divides the interval into small steps $h$ and predicts the next value using the slope at each point.

$$y_{n} = y_{n-1} + h\,\frac{\partial y_{n-1}}{\partial x_{n-1}}$$

Unrolling the steps gives $y_n = y_1 + \sum_{i=1}^{n-1} h \frac{\partial y_i}{\partial x_i}$, and the smaller the step size $\Delta x$, the closer the numerical solution gets to the true one. The observation that this form is structurally identical to ResNet's skip connection is the starting point of the paper.

## Method

### From Residual Networks to ODE Networks

Models such as ResNets, RNN decoders, and normalizing flows build complicated transformations by composing a sequence of transformations to a hidden state.

$$h_{t+1} = h_t + f(h_t, \theta_t), \quad t \in \{0, \dots, T\},\; h_t \in \mathbb{R}^D$$

This iterative update is an Euler discretization of a continuous transformation. In the limit of adding more layers and taking smaller steps, we can parameterize the continuous dynamics of the hidden units with a neural network.

$$\frac{dh(t)}{dt} = f(h(t), t, \theta)$$

The stack of discrete layers is now replaced by a single continuous "ODE block." The transformation from the initial state $z(0)$, corresponding to the input, to the final state $z(1)$, corresponding to the output, is computed by integration.

$$z(1) = z(0) + \int_0^1 f(z(t), t, \theta)\, dt$$

This integral can be evaluated with any ODE solver — Euler's method, Runge–Kutta 4, and so on — and modern solvers adaptively adjust their evaluation points while monitoring the error. In other words, the **"depth" of the model is no longer a fixed hyperparameter but a quantity the solver determines per input**.

### Adjoint Sensitivity Method: Constant-Memory Backpropagation

Differentiating through the internal operations of the ODE solver directly would require storing every intermediate value from the forward pass, incurring a large memory cost. Instead, the paper uses the **adjoint sensitivity method**.

Define the gradient at each time point as the adjoint state.

$$a(t) = \frac{\partial \text{Loss}}{\partial z(t)}$$

The adjoint state itself follows another ODE.

$$\frac{da(t)}{dt} = -a(t)^{\top} \frac{\partial f(z(t), t, \theta)}{\partial z}$$

Starting from $a(1)$ and integrating backwards in time therefore yields $a(0)$, the gradient with respect to the input.

$$a(0) = a(1) - \int_1^0 \frac{\partial a(t)}{\partial t}\, dt \;\Longleftrightarrow\; \frac{\partial \text{Loss}}{\partial z(0)} = \frac{\partial \text{Loss}}{\partial z(1)} - \int_1^0 \frac{\partial a(t)}{\partial t}\, dt$$

The gradient with respect to the parameters $\theta$ is obtained the same way, by jointly integrating $-a(t)^{\top}\frac{\partial f}{\partial \theta}$. In short, **the forward pass solves an ODE, and so does the backward pass**. Since no intermediate activations need to be stored, the memory cost is $\mathcal{O}(1)$ in depth, and gradients flow naturally through to conv layers preceding the ODE block, allowing the whole model to be trained end-to-end.

## Experiments

### Supervised Learning: MNIST

ODE-Net, which replaces ResNet's residual blocks with an ODE block, matches ResNet on MNIST (0.42% vs. 0.41% test error) while using roughly a third of the parameters (0.22M vs. 0.60M) and achieving $\mathcal{O}(1)$ memory instead of $\mathcal{O}(\tilde{L})$, where $\tilde{L}$ is the number of function evaluations (NFE) the ODE solver requests in a single forward pass.

The statistics of the trained ODE-Net are also interesting: (a) numerical error decreases as NFE increases (an explicit precision–speed trade-off), (b) NFE is proportional to wall-clock time, (c) the backward pass requires roughly half the NFE of the forward pass, and (d) NFE increases over the course of training — meaning the model learns increasingly complex dynamics.

In a separate reproduction experiment on handwritten uppercase English letters, Neural ODE (both a hand-implemented Euler method version and an autodiff version) matched CNN and ResNet baselines with about 99.2–99.3% validation accuracy.

### Continuous Normalizing Flow

In normalizing flows, the change-of-variables formula requires a Jacobian determinant.

$$z_1 = f(z_0) \;\Rightarrow\; \log p(z_1) = \log p(z_0) - \log \left| \det \frac{\partial f}{\partial z_0} \right|$$

Computing this determinant costs cubic in the number of hidden units. But if the transformation is moved to continuous time, the change in log-probability can be computed with **only a trace operation**.

$$\frac{\partial \log p(z(t))}{\partial t} = -\text{tr}\left( \frac{df}{dz(t)} \right)$$

Moving from a discrete set of layers to a continuous transformation simplifies the computation of the change in the normalizing constant. Adding per-hidden-unit gating $\frac{dz}{dt} = \sum_n \sigma_n(t) f_n(z)$ gives the CNF, which approximated target distributions with lower loss than NF under the same depth/width budget.

### Latent ODE: Irregular Time Series

Standard RNNs struggle with irregularly-sampled time series. The paper proposes a generative model (the latent ODE) that encodes observations into a distribution over a latent initial state $z_{t_0}$ with an RNN encoder, unrolls the latent trajectory with an ODE, and decodes at arbitrary time points. In spiral trajectory experiments with observation ratios of 30/100, 50/100, and 100/100, the latent ODE achieved consistently lower predictive RMSE (0.1642/0.1502/0.1346) than the RNN (0.3937/0.3202/0.1813), with the gap widening as observations became sparser.

## Takeaways

**Strengths**
1. **Memory efficiency** — constant memory independent of depth, thanks to the adjoint method.
2. **Adaptive computation** — the solver adjusts its evaluation strategy per input and required precision.
3. **Scalable, invertible normalizing flows** — the determinant simplifies to a trace.
4. **Continuous-time time-series models** — irregularly-sampled data is handled naturally.

**Limitations**
1. **Minibatching** — more complicated to use than in ordinary neural networks.
2. **Uniqueness of solutions** — guaranteed when the network has finite weights and Lipschitz nonlinearities such as tanh or ReLU (Picard's theorem).
3. **Setting error tolerances** — tolerances must be chosen for both the forward and backward passes.
4. **Reconstructing forward trajectories** — numerical error can accumulate when reconstructing the trajectory via backward integration, which checkpointing can mitigate.

By reinterpreting the discrete notion of "depth" as continuous dynamics, this paper became a modern classic that laid the groundwork for subsequent research on continuous-depth models, neural SDEs/CDEs, flow-based generative models, and time-series modeling.
