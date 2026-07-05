---
title: "A Generalization of ViT/MLP-Mixer to Graphs"
date: 2025-08-06
tags: [GNN, MLP-Mixer, Graph Transformer, ICML]
categories: ["Paper Review"]
series: ["AI Algorithms"]
math: true
summary: "A review of Graph ViT/MLP-Mixer, which defines 'graph patches' via METIS partitioning and ports the ViT/MLP-Mixer grammar to graphs — capturing long-range dependencies and 3-WL-level expressiveness at linear complexity."
---

> **Paper**: A Generalization of ViT/MLP-Mixer to Graphs — Xiaoxin He, Bryan Hooi, Thomas Laurent, Adam Perold, Yann LeCun, Xavier Bresson, ICML 2023 (PMLR 202:12724–12745). [arXiv:2212.13350](https://arxiv.org/abs/2212.13350)

## TL;DR

Message-passing GNNs are fast but capped at 1-WL expressiveness and weak on long-range dependencies. Graph Transformers fix both, but the cost of global attention is a burden. This paper ports the ViT/MLP-Mixer grammar from computer vision — "cut into patches, encode, mix" — to graphs, proposing **Graph ViT/MLP-Mixer**. Graph patches are extracted with METIS partitioning, each patch is encoded by a small GNN, and the patch tokens are mixed by an MLP-Mixer (or graph MHA). The result: over-squashing is mitigated at a cost linear in the number of nodes and edges, 3-WL-level expressiveness is demonstrated on SR25, and competitive performance is achieved, including a ZINC MAE of 0.073.

## Background

### Two fundamental limits of MP-GNNs

Most GNNs stack 1-hop message-passing layers (GCN, GAT, etc.) so that $l$ layers propagate information up to $l$ hops. This design carries two structural weaknesses.

**Limit 1 — bounded expressiveness.** The discriminative power of MP-GNNs is equivalent to the 1-WL (Weisfeiler–Leman) test. In the WL test, each node repeatedly updates its label by aggregating its neighbours' labels, and two graphs are declared non-isomorphic if their label distributions diverge. Equivalence to 1-WL means there are non-isomorphic graphs that MP-GNNs cannot tell apart. $k$-WL-based GNNs address this by labelling $k$-tuples of nodes, raising expressiveness but exploding complexity to $O(N^2)$ or $O(N^3)$. Another remedy, graph positional encodings (Laplacian eigenvectors, random walks, etc.), offers expressiveness beyond 1-WL at $O(E)$ cost, but graphs lack a canonical orientation, so global positions cannot be represented exactly. Laplacian eigenvectors, for instance, are identical up to sign flips (sign ambiguity), forcing the model to learn spurious symmetries.

**Limit 2 — information loss and over-squashing.** Receiving information from $L$ hops away requires $L$ layers, and each layer compresses an exponentially growing neighbourhood into a fixed-dimensional vector. Useful distant information gets crushed — over-squashing — compounded by vanishing gradients, much like the long-term dependency problem in RNNs. Graph Transformers solve this with attention, but at a steep computational cost.

### Why ViT/MLP-Mixer?

In vision, ViT and MLP-Mixer split an image into fixed patches and handle long-range dependencies through a uniform pipeline of patch embedding → token/channel mixing. The paper's central idea is that bringing this grammar to graphs could deliver Transformer-level expressiveness at MP-GNN-level efficiency.

## Challenges: What Breaks When Going from Grids to Graphs

The differences between images and graphs rule out a naive port. The paper identifies four challenges.

1. **Patch definition/extraction**: images sit on a fixed grid, so uniform splitting is easy; graphs vary in node/edge counts and structure, so no consistent patch definition exists. Graph clustering must replace pixel rearrangement.
2. **Patch encoding**: image patches share a common size and can be encoded directly by an MLP, but graph patches differ in size and structure and have no node ordering, requiring a permutation-invariant, fixed-length encoding.
3. **Positional information**: images implicitly preserve up/down/left/right, but graphs have no notion of position at all. Positional encodings must be designed at both the node level and the patch level.
4. **Overfitting**: ViT/MLP-Mixer models are strong learners prone to overfitting, and unlike vision, graphs have no standardized data augmentation (no equivalent of crop or flip). Node dropping, edge perturbation, subgraph sampling, feature masking, and mixup have been proposed, but their performance is inconsistent.

## Method: Graph ViT/MLP-Mixer

The full pipeline is Patch Extraction → Patch Encoder (GNN) → Mixer Layers → Global Average Pooling → prediction.

### 1) Patch Extraction: METIS partitioning + 1-hop overlap

**METIS** is a classical algorithm that partitions a graph so that clusters stay balanced in node count while minimizing the edge cut between them. It operates in three phases — Coarsening (matching and collapsing adjacent nodes to shrink the graph) → Initial Partitioning (partitioning the coarsest graph) → Uncoarsening & Refinement (quality improvement via Kernighan–Lin and similar) — and finds clusters with high internal and low external connectivity.

METIS splits the graph into $P$ clusters, each treated as a subgraph $G_p = (V_p, E_p)$; to recover edges severed by the partition, a **1-hop overlap** is applied so that every edge belongs to at least one patch. Crucially, patch extraction is re-run every epoch, making it stochastic (see Data Augmentation below).

### 2) Patch Encoder: a GNN per patch

Node and edge features are first linearly embedded into $d$ dimensions.

$$h_i^0 = U^0 \alpha_i + u^0 \in \mathbb{R}^d, \qquad e_{ij}^0 = V^0 \beta_{ij} + v^0 \in \mathbb{R}^d$$

An MP-GNN (any of GCN, GAT, etc.) is applied within each patch to update node and edge embeddings, with an added term injecting the patch's mean information.

$$h_{i,p}^{\ell+1} = f_{\text{node}}\left(h_{i,p}^{\ell}, \{h_{j,p}^{\ell} \mid j \in \mathcal{N}(i)\}, e_{ij,p}^{\ell}\right) + g_{\text{patch-node}}(h_p^{\ell})$$

Here $h_p^{\ell} = \frac{1}{|V_p|}\sum_{i \in V_p} h_{i,p}^{\ell}$ is the mean over the patch's nodes, serving as patch-level global context. When a node belongs to multiple patches (due to the 1-hop overlap), the overlapping embeddings are averaged for consistency.

$$h_i^{\ell+1} = \mathrm{Mean}_{k \mid i \in V_k}\, h_{i,k}^{\ell+1}$$

Finally, per-patch mean pooling followed by an MLP yields a fixed-length patch embedding $x_{G_p} = \mathrm{MLP}(h_p) \in \mathbb{R}^d$. Since the GNN encoder is permutation-invariant by construction, challenge 2 is resolved.

### 3) Positional Encoding: node PE + patch PE

- **Node PE**: a random-walk or Laplacian-based absolute positional encoding $p_i$ is added to the initial node embedding ($h_i^0 = T_0 p_i + U_0 \alpha_i + u_0$). The sign ambiguity of Laplacian eigenvectors is handled by random sign flips during training.
- **Patch PE**: a coarse adjacency matrix is defined from the number of connections between clusters,

$$A_P(i,j) = |\mathrm{Cut}(V_i, V_j)| = \sum_{k \in V_i}\sum_{l \in V_j} A_{kl}$$

and a patch positional encoding $\hat{p}_i$ derived from it is added to the patch embedding ($x_i^0 = \hat{T}_0 \hat{p}_i + \hat{U}_0 x_i + \hat{u}_0$), preserving the relative positions among patches.

### 4) Mixer Layer: token = patch

The MLP-Mixer grammar is applied verbatim to the patch embedding matrix $X \in \mathbb{R}^{P \times d}$.

$$U = X + \left(W_2\,\sigma(W_1\, \mathrm{LayerNorm}(X))\right) \quad \text{(Token mixer)}$$
$$Y = U + \left(W_4\,\sigma(W_3\, \mathrm{LayerNorm}(U)^T)\right)^T \quad \text{(Channel mixer)}$$

The token mixer is multiplied by the coarse adjacency $A_D^P \in \mathbb{R}^{P \times P}$ so that the patch connectivity structure is respected. In the ViT variant (Graph ViT), graph-based multi-head attention (gMHA) replaces the token mixer.

$$U = X + \mathrm{gMHA}(\mathrm{LayerNorm}(X)), \qquad Y = U + \mathrm{MLP}(\mathrm{LayerNorm}(U))$$

Finally, Global Average Pooling produces the graph embedding $h_G$, and $y_G = \mathrm{MLP}(h_G)$ performs regression or classification.

### 5) Implicit Data Augmentation

Patch extraction with METIS + 1-hop overlap is stochastic, so the same graph yields a different patch set each epoch. This acts as implicit data augmentation, increasing data diversity and keeping the model from overfitting to any single partition. It is especially effective at inflating the effective sample count on small graph datasets. The answer to challenge 4 is thus built into the pipeline itself, rather than bolted on as a separate augmentation scheme.

## Experiments

- **Molecular benchmarks**: MAE 0.0730 on ZINC and ROC-AUC 0.7997 on MolHIV — competitive under both the Benchmarking-GNNs and OGB protocols.
- **Long-range modelling**: validated on LRGB (Long Range Graph Benchmark) for long-range dependency modelling, with over-squashing mitigation confirmed on the TreeNeighbourMatch experiment.
- **Expressiveness**: on SR25, a dataset of strongly regular graphs indistinguishable by 1-WL, the model demonstrates 3-WL-level discriminative power. The substructure information carried by METIS patches combined with the patch PE gives structural identification well beyond pure message passing.
- **Efficiency**: every component (METIS, per-patch GNN, Mixer) is linear in the number of nodes and edges, scaling far better than global attention.

## Takeaways

- The paper shows that vision's design grammar of "patch → encode → mix" holds even in the irregular graph domain, offering a unifying architectural language across CV, NLP, and graphs.
- The two chronic ailments of MP-GNNs (expressiveness and over-squashing) can be treated simultaneously without attention, because patch-level tokenization fundamentally shortens information pathways.
- The design in which METIS's stochastic partitioning doubles as augmentation is elegant: one structural choice contributes to efficiency, expressiveness, and generalization all at once.
- For graph-level prediction problems where global structure matters (molecular properties, materials properties), the hybrid of GNNs encoding local structure and a Mixer blending global information is a practical default worth adopting.
