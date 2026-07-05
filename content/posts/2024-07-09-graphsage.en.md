---
title: "Inductive Representation Learning on Large Graphs (GraphSAGE)"
date: 2024-07-09
tags: ["GNN", "GraphSAGE", "Node Embedding", "Inductive Learning"]
categories: ["Paper Review"]
series: ["AI Algorithms"]
math: true
summary: "A review of GraphSAGE, an inductive framework that learns a function for sampling and aggregating neighbor features instead of learning per-node embeddings directly, generalizing embeddings to nodes and graphs unseen during training."
---

> **Paper**: Inductive Representation Learning on Large Graphs — William L. Hamilton, Rex Ying, Jure Leskovec, *NeurIPS 2017*. [arXiv:1706.02216](https://arxiv.org/abs/1706.02216)

## TL;DR

- Prior node embedding methods (DeepWalk, node2vec, etc.) and early GCNs are inherently **transductive**: they jointly optimize the embeddings of all nodes on a single fixed graph, so new nodes require retraining.
- GraphSAGE (SAmple and aggreGatE) learns, instead of per-node embedding vectors, **a function that samples and aggregates the features of neighboring nodes**. The learned aggregation function applies directly to unseen nodes — even to entirely new graphs.
- Neighbors are **uniformly sampled** at a fixed size, keeping per-batch computation bounded and making the method scalable to large graphs.
- The paper proposes several aggregators — Mean, LSTM, and Pooling — and outperforms prior methods by a wide margin on inductive node classification over citation, Reddit, and protein-protein interaction (PPI) graphs.

## Background

### Transductive vs Inductive

The goal of node embedding is to compress a graph's structural information and node features into low-dimensional vectors. Matrix-factorization-based methods and random-walk-based methods (DeepWalk, node2vec) **directly optimize a unique embedding vector per node, like a lookup table**. This approach provides embeddings only for the nodes in the training graph, making it transductive.

Real-world graphs, however, keep changing. New users, new posts, and new measurement points are added constantly, and models often need to be applied to entirely new graphs of the same form. To **generate embeddings immediately for unseen nodes** without full retraining each time, one must learn not per-node vectors but a **function** that produces embeddings. This is inductive learning, and it is the core motivation behind GraphSAGE.

## Method

### Sample and Aggregate

GraphSAGE's embedding generation (forward propagation) consists of $K$ layers (search depth). The initial values are the node features $h_v^0 = x_v$, and at each layer $k$, the following is performed for every node $v$.

$$h_{\mathcal{N}(v)}^{k} = \text{AGGREGATE}_k\!\left( \{ h_u^{k-1} : u \in \mathcal{N}(v) \} \right)$$

$$h_v^{k} = \sigma\!\left( W^k \cdot \text{CONCAT}\!\left( h_v^{k-1},\; h_{\mathcal{N}(v)}^{k} \right) \right)$$

That is, (1) aggregate the neighbors' previous-layer representations into a single vector, then (2) concatenate it with the node's own previous representation and apply a linear transformation and a nonlinearity. After $K$ iterations, each node has a representation $z_v = h_v^K$ that absorbs the structural and feature information of its $K$-hop neighborhood. What is learned is not per-node vectors but only **the per-layer weights $W^k$ and the aggregator parameters**, so when a new node arrives, running the same procedure is all it takes to produce its embedding.

### Neighbor Sampling

Using all neighbors makes computation explode with node degree. GraphSAGE **uniformly samples** a fixed number $S_k$ of neighbors at each layer, fixing the per-batch computation and memory cost at $\mathcal{O}(\prod_k S_k)$. This design is the practical device that makes minibatch training feasible on graphs with hundreds of millions of edges.

### Aggregator Functions

Ideally the aggregation function is permutation invariant with respect to neighbor order; the paper proposes three.

- **Mean aggregator**: the element-wise mean of neighbor representations. Written as $h_v^{k} \leftarrow \sigma(W \cdot \text{MEAN}(\{h_v^{k-1}\} \cup \{h_u^{k-1}, u \in \mathcal{N}(v)\}))$, it becomes an inductive approximation of the GCN convolution rule. Simple but powerful.
- **LSTM aggregator**: more expressive, but order-sensitive, so it is applied to random permutations of the neighbors. It can also be a natural choice when handling ordered relationships (e.g., temporal connections).
- **Pooling aggregator**: each neighbor representation is passed through an independent fully-connected layer, followed by element-wise max-pooling. It often performed best in the experiments.

In practice, when a graph mixes edges with different semantics, one can combine different aggregators per edge type (e.g., mean for symmetric relationships, LSTM for ordered ones).

### Training: Supervised and Unsupervised

GraphSAGE can be trained end-to-end with a supervised loss for a downstream task, or without labels using only the graph structure. The unsupervised loss is a negative-sampling-based objective that pushes nearby nodes toward similar embeddings and unrelated nodes toward dissimilar ones.

$$J(z_u) = -\log\!\left( \sigma(z_u^{\top} z_v) \right) - Q \cdot \mathbb{E}_{v_n \sim P_n(v)} \log\!\left( \sigma(-z_u^{\top} z_{v_n}) \right)$$

Here $v$ is a node that co-occurs near $u$ on a random walk, and $P_n$ is the negative sampling distribution.

## Experiments

Evaluation covers three inductive benchmarks.

1. **Citation graph (Web of Science)**: classifying the subject of newly appearing papers
2. **Reddit**: classifying the community a new post belongs to (about 230K nodes)
3. **PPI (protein-protein interaction)**: predicting protein functions on **entirely new graphs** never seen during training (multi-label)

The main results are as follows.

- GraphSAGE showed large F1 gains over raw-feature classifiers, DeepWalk, and DeepWalk+feature combinations. On Citation and Reddit, even fully unsupervised embeddings beat prior methods, and the supervised version gained further.
- In particular, in settings like PPI where **training and test graphs are disjoint**, transductive methods are fundamentally inapplicable or require retraining, whereas GraphSAGE applied its learned aggregation function directly and performed strongly.
- Among aggregators, pooling and LSTM tended to be somewhat better than mean, but mean was the lightest and most stable.
- Increasing the neighbor sample size $S$ improves performance, but with clear diminishing returns, so a small sample already offers a good performance-efficiency trade-off.
- When new nodes are added, DeepWalk needs hundreds of times longer to re-optimize embeddings, whereas GraphSAGE needs only a single forward pass.

## Takeaways

- GraphSAGE's essential contribution is the shift in perspective from "learning embeddings" to "**learning a function that produces embeddings**." This shift enables generalization to evolving graphs and unseen graphs.
- Controlling computation via neighbor sampling became the standard technique for scaling GNNs to production, later serving as the foundation for industrial-scale recommender systems like PinSAGE.
- Abstracting the aggregator as a swappable module helped frame subsequent GNN architectures — such as GAT (an attention aggregator) — within a single message passing framework.
- Its limitations — uniform sampling cannot distinguish important neighbors from unimportant ones, and larger depths $K$ suffer from neighborhood explosion and over-smoothing — led respectively to importance-based sampling and attention, and to research on deep GNNs.

For large, dynamic graphs with node features, it remains the first baseline to consider even today — a classic that opened up the practical use of GNNs.
