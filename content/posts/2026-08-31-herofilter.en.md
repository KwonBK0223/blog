---
title: "HeroFilter: Adaptive Spectral Graph Filter for Varying Heterophilic Relations"
date: 2026-08-31
tags: ["Graph Neural Networks", "Spectral Filter", "Heterophily", "MLP-Mixer"]
categories: ["Paper Review"]
series: ["AI Algorithms"]
math: true
summary: "A review of HeroFilter, an MLP-Mixer-based GNN that starts from the theoretical observation that the relationship between heterophily and the optimal spectral filter is non-monotonic, and learns an adaptive polynomial spectral filter per node instead of a fixed low-/high-pass filter."
---

> **Paper**: HeroFilter: Adaptive Spectral Graph Filter for Varying Heterophilic Relations — Shuaicheng Zhang, Haohui Wang, Junhong Lin, Xudong Guo, Yifan Zhu, Si Zhang, ..., Dawei Zhou, *NeurIPS 2025*.

## TL;DR

On strongly heterophilic graphs, conventional GNNs that blindly average neighbors end up contaminating information. Filter-based GNNs that try to address this from a spectral viewpoint have relied on a simple assumption: "low-pass for homophily, high-pass for heterophily." HeroFilter starts precisely by challenging this assumption. The authors observe that as heterophily increases, the optimal spectral response does not simply shift from low to high frequency but activates across the entire band. They build a theory to support this observation, then propose an MLP-Mixer-based architecture that learns an adaptive polynomial spectral filter per node instead of a fixed filter. The core message is singular: **a filter must adapt not to the "magnitude" of heterophily, but to the "frequency" at which discriminative information and heterophily reside.**

## Background

To understand HeroFilter, two axes need to be laid out first. One is heterophily as a graph property, and the other is the spectral view that treats a graph in terms of frequency.

### What is heterophily?

**Heterophily** refers to a state in which connected nodes are likely to carry different labels. It is the opposite of **homophily** ("birds of a feather flock together"). A heterophilic graph follows the rule "connected entities tend to differ." A few representative examples make this concrete.

- **Fraud detection**: fraudulent accounts mainly transact with normal user accounts, so nodes of different labels are likely to be connected.
- **Dating networks**: in matching graphs, male nodes are mostly connected to female nodes.
- **Molecular structures**: in chemical bonds, different kinds of atoms and molecules often bond together rather than identical ones.

Conventional GNNs such as GCN and GAT are designed on top of the homophily assumption that "neighbors are similar to me." They update a node's features by summing or averaging neighbor information (aggregation). Applying this directly to heterophilic graphs causes problems. First, **information contamination** — neighbor information from an entirely different class gets mixed in, blurring the node's own identity. Second, **over-smoothing** — the more neighbors are mixed, the more nodes become indistinguishable, sharply degrading prediction accuracy, sometimes performing even worse than an MLP that ignores graph structure altogether.

It is worth clarifying a commonly confused distinction here. **Heterophilic** (a property of connectivity) concerns "whether nodes with different labels tend to connect," whereas **heterogeneous** (a property of composition) concerns "whether multiple types of nodes and edges are mixed within the graph." A graph with only one node type (homogeneous) can still be perfectly heterophilic, so the two concepts are independent.

### Frequency on a graph

The second axis is the view that treats a graph in terms of frequency. Here, **frequency is not the number of oscillations over time.** On a graph, frequency means "how rapidly signal values change between connected neighbors."

Consider a simple chain graph $A-B-C-D-E$ with a signal $x$ (temperature, embedding, or label) placed on each node. If node values are similar between connected neighbors, such as $(10, 10, 11, 11, 10)$, this is a **low-frequency** signal. Conversely, if the value flips sharply each time an edge is crossed, such as $(10, -10, 10, -10, 10)$, it is a **high-frequency** signal. In other words, the reference for frequency on a graph is connectivity, the amount of change between neighbors.

This intuition is formalized precisely by the Graph Laplacian. For the Laplacian defined as $L = D - A$, the following holds.

$$x^\top L x = \sum_{(i,j)\in E}(x_i - x_j)^2$$

That is, the Laplacian is essentially an operator that measures "how different a signal is between neighbors." Now, eigendecomposing the Laplacian gives $L u_k = \lambda_k u_k$, and taking eigenvector $u_k$ with norm 1 yields $u_k^\top L u_k = \lambda_k$. But the left-hand side, as just seen, is $\sum_{(i,j)\in E}(u_k(i) - u_k(j))^2$, precisely a measure of how much that eigenvector changes along edges. Therefore, **the larger the eigenvalue $\lambda_k$, the more that mode changes along edges, corresponding to high frequency.** We can read $u_k$ as a spectral mode of the graph and $\lambda_k$ as the degree of frequency of that mode.

In this language, GCN can be interpreted as a spectral GNN using a first-order Chebyshev polynomial, that is, a **low-pass filter**. It amplifies low frequencies and suppresses high frequencies, maintaining smooth signals across the graph. This fits homophilic structures well, where useful signals cluster in low frequencies, but is ill-suited for heterophilic structures, where useful signals may reside in higher-frequency bands.

## Motivation: Non-Monotonic Filter–Heterophily Relationship

Once the spectral view is adopted, filter-based GNNs naturally emerge, a line of work that explicitly incorporates high-frequency information or uses high-pass or mixed filters to handle heterophily. Yet most of these rest on a very simple assumption.

$$\text{Homophily} \rightarrow \text{Low-frequency} \rightarrow \text{Low-pass}$$
$$\text{Heterophily} \rightarrow \text{High-frequency} \rightarrow \text{High-pass}$$

HeroFilter's central objection is that this correspondence is **too monotonic and likely at odds with reality**. The authors construct synthetic graphs with tunable heterophily levels and observe the spectral response of the optimal filter learned on each graph. The result was far from a simple low-to-high shift. Filters learned on low-heterophily graphs did not stay confined to low frequencies but activated across the entire band, and filters learned on high-heterophily graphs preserved not only high-frequency but also low-frequency components. The optimal spectral response was not a smooth function predictable from a single heterophily level.

This observation gives rise to two questions.

- **Q1.** How can the relationship among a graph's heterophily, the spectral filter, and a GNN's prediction performance be formally explained?
- **Q2.** How can an adaptive-filter GNN be designed to operate robustly on graphs with diverse heterophily patterns?

The remaining half of this post follows the Theory that answers Q1 and the HeroFilter architecture that answers Q2.

## Theory

The goal of the theory part is to justify "a filter should be adaptive rather than a fixed low-/high-pass" through logic rather than empirical observation. The overall flow is organized into four stages. Definition 2 moves heterophily into the same domain as the filter, Proposition 1 shows that a fixed spectral intuition is insufficient, Proposition 2 secures the expressivity of an adaptive filter, and Theorem 1 connects all of this to the actual prediction error. Below, rather than transcribing the algebra, the focus is on the **role** each result plays for the next and its **conclusion**.

### Definition 2: Spectral Heterophily

The starting point of the problem is a domain mismatch. Heterophily is usually defined in the spatial domain. The spatial heterophily of node $v_i$ is

$$h_i = \frac{|\{v_j \in N(v_i) : y_j \neq y_i\}|}{|N(v_i)|}$$

a scalar measuring "how many of my neighbors carry a label different from mine." In contrast, a graph filter $g(\Lambda)$ is essentially defined in the spectral domain. Connecting two objects that live in different domains directly is difficult.

Definition 2 bridges this gap with the graph Fourier transform. Multiplying the spatial heterophily vector $h$ by the Fourier basis $U^\top$ defines

$$\hat h = U^\top h$$

This makes it possible to view heterophily no longer as a single scalar but as a **per-frequency distribution**. Here $\hat h_i$ denotes the amount of heterophily contained in spectral component $i$. Thanks to this single step, the question "how large is heterophily" can be transformed into the much richer question "at which frequency does heterophily reside," and this becomes the foundation for all subsequent discussion.

### Proposition 1: Filter response and heterophily are not monotonically related

Proposition 1 proves that the average spectral response of a filter does not follow a simple monotonic relationship with the graph's heterophily level. The key tool of the proof is the weighted AM-GM inequality (weighted arithmetic mean ≥ weighted geometric mean).

The proof begins by assigning to the heterophily values $x_i = |\hat h_i|$ a weight that normalizes the filter response like a probability,

$$w_i = \frac{g(\lambda_i)}{\sum_k g(\lambda_k)}$$

where $w_i$ represents the relative share of frequency $i$ within the total filter response, indicating which frequencies the filter deems important. Applying AM-GM here yields an inequality of the form "filter-weighted arithmetic mean of heterophily ≥ geometric mean," and taking the logarithm of both sides to turn the product into a sum rearranges the average filter response into a form separated into a heterophily-spectrum term and a filter-heterophily interaction term.

The conclusion is what matters. The average filter response is not determined by a single scalar heterophily level. **What matters is not whether heterophily is "large" or "small," but at which frequency it resides and how the filter $g(\lambda_i)$ responds at that frequency.** Even for the same average heterophily, the required filter differs depending on whether it is concentrated in low or high frequencies. A single fixed low-/high-pass filter cannot capture this diversity, so an adaptive filter is needed.

### Proposition 2: An adaptive filter has sufficient expressivity

If Proposition 1 said "a fixed filter is insufficient," the immediate follow-up question is "then can an adaptive filter actually represent the required spectral response?" Proposition 2 answers this.

The idea of the proof is simple. To align the filter's spectral response $g(\Lambda)$ exactly with the target label distribution $Y$ (cosine similarity of 1), it suffices to make the two a scalar multiple, $g(\Lambda) = cY$. A polynomial filter learnable per frequency can produce $c y_i$ at each frequency $i$ as a sum of several terms, and whether the activation is bounded or unbounded, choosing parameters $\{w_k\}$ so that each term contributes evenly always makes this equality hold. Therefore, filter parameters aligned with the target direction exist.

There is a distinction that must be made here. What Proposition 2 proves is **existence (expressivity)**, not learnability. "Such parameters exist ($\exists\{w_k\}$ s.t. $g(\Lambda) \parallel Y$)" is an entirely different statement from "optimization actually finds those parameters." The conclusion is that an adaptive filter has enough degrees of freedom to represent any desired target spectral pattern; it is not a claim that training reaches that optimum or that generalization is guaranteed. **"Can represent" is different from "can learn."**

### Theorem 1: Decomposing prediction error into per-frequency alignment

Theorem 1 is the apex of this theory, deriving an error upper bound that quantifies the effect of heterophily and filter design on prediction performance. The result takes the following form.

$$Er(X, Y) \le c_1 - (\text{spectral alignment term})$$

Here the spectral alignment term is determined not by a single heterophily scalar but by the **alignment of four per-frequency components**.

- $\eta_i$: the **feature difference** between the two classes at frequency $i$
- $\delta_i$: the **label difference** between the two classes at frequency $i$
- $g(1 - \lambda_i)$: the **filter response** at that frequency
- $|\hat h_i|$: the **heterophily spectrum** at that frequency

Let us follow only the main thread of the proof. Since the sigmoid squared error of binary classification is nonlinear and hard to handle, the score $x$ is first confined to a bounded interval by a clamp function $\psi(x) \in [-1, 1]$. It is shown that this clamping does not change the loss unboundedly, bounding the difference by a constant (even in the extremes where $x > 1$ or $x < -1$, the clamped/unclamped loss difference stays below a fixed constant), after which a first-order Taylor expansion $\frac{1}{1+e^x} \approx \frac{1}{2} - \frac{1}{4}x$ is used around $x=0$. The remainder is controlled by $|R(x)| \le \frac{|x|^3}{48}$. Through this process, the nonlinear sigmoid error is turned into an analyzable polynomial bound.

The term that survives as the crux of the expansion is the label-prediction alignment term $-\frac{(1-2y)\psi(x)}{4}$. For binary labels, $1-2y$ is $\pm 1$, and since the sigmoid is a decreasing function, the more the prediction score points in the correct direction, the larger this term and the smaller the error bound. Next, substituting the graph-filtered class difference $z = g(I - \tilde A)(X_1 - X_0)$ in place of the scalar score $x$ yields the key term

$$-\frac{1}{4}(y_1 - y_0)^\top \psi\!\left(g(I - \tilde A)(X_1 - X_0)\right)$$

Here $y_1 - y_0$ is the label separation and $g(I - \tilde A)(X_1 - X_0)$ is the feature separation after filtering. That is, **the better the filtered feature difference aligns with the label difference, the smaller the error bound.** As an aside, the filter response appears in the form $g(1 - \lambda_i)$ because the proof sets the filter as $g(I - \tilde A)$. If the eigenvalue of $\tilde A$ is $\lambda_i$, the eigenvalue of $I - \tilde A$ is $1 - \lambda_i$, so the spectral response becomes $g(1 - \lambda_i)$.

Finally, rewriting this expression in the graph Fourier basis turns $y_1 - y_0 \to \delta_i$ and $X_1 - X_0 \to \eta_i$, giving rise to the per-frequency product $\eta_i\, g(1 - \lambda_i)\, \delta_i$. This denotes "how well the filter connects the feature and label signals that discriminate the two classes at frequency $i$." Then, lower-bounding this alignment sum by the total filter response $\sum_i g_i$ and substituting Proposition 1 turns the total filter response into a lower bound that depends on the heterophily spectrum, at which point **heterophily finally enters the error bound.** This is the most important conceptual bridge in all of Theorem 1.

The message of Theorem 1 is clear. The conventional simple thinking was "heterophily↑ ⇒ high-pass," but the error is determined not by a single heterophily scalar but by the alignment of the per-frequency components $|\hat h_i|$, $g(1 - \lambda_i)$, $\delta_i$, and $\eta_i$. Therefore, **the filter must adapt to which frequencies carry class-discriminative features and label information and how heterophily is distributed.** Performance is frequency-dependent and is not determined by a single heterophily level.

## HeroFilter Architecture

What the theory says can be summarized in two points. First, the spectral behavior of a heterophilic graph is complex and non-monotonic, so a uniform low-/high-pass filter is not effective. Second, different frequency bands contribute differently to prediction performance depending on the graph's heterophily structure. HeroFilter is an architecture that translates these conclusions directly into design, and inspired by MLP-Mixer, it consists of two modules.

### Patcher: reselecting neighbors with a learned filter

The core goal of the Patcher can be summed up in one sentence. **Rather than using the graph's original neighbors as is, reselect useful neighbors with a learned spectral filter.** For each node, an adaptive polynomial spectral filter is learned, and through it, spectrally relevant nodes are dynamically selected rather than the neighbors immediately adjacent in topology. This allows the model to attend to contextually important nodes across multiple spectral bands, beyond simple local neighbors. After ranking nodes by the computed relevance, the top-$p$ nodes are gathered to form each node's graph patch.

### Mixer: a dual-axis MLP mixing along two axes

The Mixer is essentially the same structure as MLP-Mixer, a dual-axis MLP that mixes the selected patches along both the spatial (patch) axis and the feature axis. The reason for mixing both axes is the crux of the design.

- **Patch-mixing** models how other nodes influence a particular node's representation, capturing the heterophily of the node's context.
- **Feature-mixing** abstractly transforms raw input features, capturing the complexity within a node.

Combining the two is not optional but essential. Patch-mixing grants adaptivity to the structural irregularities arising in heterophilic regions, while feature-mixing enables expressive node-level reasoning. Either one alone cannot simultaneously satisfy the "structural adaptation + expressive reasoning" that the theory demands.

### Fast-HeroFilter: an approximation for large-scale graphs

For the Patcher to obtain spectral relevance directly, eigen-decomposition is required, which is prohibitively expensive on large-scale graphs. Instead of computing this directly, Fast-HeroFilter approximates it with a personalized PageRank (PPR)-like multi-hop diffusion. By replacing the expressive eigenvalue-decomposition-based spectral patcher of Section 4.1 with PPR-like diffusion, it processes large-scale graphs in a lightweight and scalable manner without any Transformer.

## Experiments

HeroFilter is extensively validated on 16 benchmark datasets spanning both homophily and heterophily. Rather than specific numbers, the claims can be summarized as follows.

- **Extensive validation**: it claims competitive performance across 16 datasets covering diverse heterophily levels.
- **Ablation study**: it verifies (1) whether the adaptive filter is more effective than a fixed/shared filter, (2) whether relevance-ranked patches outperform random ordering, and (3) how much patch-mixing and feature-mixing each contribute, supporting the design choices.
- **Lightweight**: compared to Transformer-based graph models, it claims competitive performance while being lightweight and free of attention.

## Takeaways

HeroFilter's contributions can be summarized in three points.

1. It presents the first theoretical analysis that formally connects graph heterophily, spectral filter response, and prediction error, challenging the conventional assumption of a monotonic filter-heterophily correlation.
2. It offers a modular architecture combining a lightweight MLP-Mixer backbone with an adaptive polynomial filter, enabling efficient spectral inference across diverse graph structures.
3. It conducts extensive empirical validation on 16 benchmarks.

The theoretical intuition is persuasive. It builds a conceptual bridge between spatial heterophily and spectral filtering, and the narrative running from Definition 2 → Proposition 1 → Proposition 2 → Theorem 1 dovetails naturally with the architecture design. Moving beyond the binary intuition of "heterophily = high frequency" to present a much finer per-frequency perspective, and connecting the Patcher's adaptive filter to a theoretical motivation rather than a mere engineering trick, are clear strengths.

That said, viewed objectively, a few points still require refinement.

- **Assumptions in Proposition 1.** In the inequality derivation, the step that changes the filter weight from $\frac{g_i}{\sum_k g_k} \to \frac{1}{\sum_k g_k}$ relies on a range condition such as $0 < |\hat h_i| \le 1$. Yet since $\hat h = U^\top h$, even if the spatial $h_i \in [0,1]$, $|\hat h_i| \le 1$ after the Fourier transform is not automatically guaranteed. Furthermore, the sign condition that the term used as the denominator in the final form is positive also needs to be secured explicitly.
- **The nature of Proposition 2.** This proposition is an existence result showing "can represent," not a "guarantee of learning/generalization." The fact that an adaptive filter has parameters that would align with an arbitrary target does not imply that actual optimization finds those parameters or generalizes. Moreover, the index of $g(\Lambda)$ is a spectral index while $Y$ is defined like a raw node label vector, so whether the correspondence between the two is sufficiently defined leaves conceptual room for further refinement.
- **The standing of interpretability.** The authors argue that the model is interpretable in the sense that one can, in principle, inspect which frequencies were emphasized and which nodes were selected into a patch. However, this is **interpretability-by-design**, not quantitatively validated interpretability.

Taken together, HeroFilter is a study that persuasively presents, on both the theoretical and architectural fronts, the direction that "fixed spectral bias should be replaced with adaptive filtering" in GNN research on heterophily. The theoretical narrative is compelling, and if refinement of a few proof assumptions follows, its contribution will only become more solid.
