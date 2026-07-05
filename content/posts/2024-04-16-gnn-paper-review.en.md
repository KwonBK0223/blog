---
title: "HGHAN: Hacker Group Identification Based on Heterogeneous Graph Attention Network"
date: 2024-04-16
tags: ["GNN", "Heterogeneous Graph", "Attention", "Cyber Security"]
categories: ["Paper Review"]
series: ["AI Algorithms"]
math: true
summary: "A review of HGHAN, a heterogeneous graph neural network that builds a Web Attack Heterogeneous Information Network (WAHIN) and combines LSTM attribute encoding, Metapath2Vec structural encoding, and HAN attention to identify hacker groups."
---

> **Paper**: HGHAN: Hacker group identification based on heterogeneous graph attention network — Y. Xu, Y. Fang, C. Huang, Z. Liu, *Information Sciences*, vol. 612, pp. 848–863, 2022. [DOI:10.1016/j.ins.2022.08.097](https://doi.org/10.1016/j.ins.2022.08.097)

## TL;DR

HGHAN represents web attack data as a **Heterogeneous Information Network**, encodes the heterogeneous attribute information and the structural information of nodes separately, and fuses them with a Heterogeneous Graph Attention Network (HAN) to identify hacker groups. An LSTM maps heterogeneous node attributes into vectors of a unified dimension, Metapath2Vec extracts metapath-based structural features, and the fused representation is fed into HAN to obtain node embeddings. Finally, a BiLSTM classifier distinguishes hacker groups from individual hackers.

## Background

### Problem Definition

Cybercrime has surged since the global pandemic, raising the stakes for cyber security. In particular, **organized hacker groups** are far more destructive and threatening than individual hackers, making hacker group identification an important task. Existing methods rely on static fingerprints or cloud-based collection, which are ill-suited to complex, hierarchical cyber attacks. The goal of this paper is to identify hacker groups effectively using HGHAN.

Hacker groups employ sophisticated, collaborative strategies; pursue concrete objectives such as financial gain, political influence, or corporate espionage; operate on long-term projects; and are careful about evading detection. Individual hackers, by contrast, are typically driven by personal motives such as curiosity, revenge, or personal interest, are relatively less sophisticated, and often work on short-term projects.

### Heterogeneous Graphs and Metapaths

Cyber security data intertwines many types of entities and relations, so a **heterogeneous graph with multiple node and edge types** is more effective at representing complex attack structures than a homogeneous graph that handles only a single type.

- **Heterogeneous graph**: defined as $G = (V, E, T, X)$, where $V$ is the node set and $E$ the edge set; the nodes and edges of a heterogeneous graph carry different types and attributes. $T = (T_v, T_e)$ records the type mappings and $X = (X_v, X_e)$ the feature mappings.
- **Meta-path**: a composite path made up of multiple relations, connecting nodes of different types. By capturing **indirect relationships** between nodes of different types, it enables the analysis and inference of latent threats that appear unrelated on the surface.

The paper models web attack data as a **Web Attack Heterogeneous Information Network (WAHIN)** with five node types — Hacker, IP, Domain, Webserver, Page — and edge types including Attack, Infect, Permeate, Control, Map, Belong, Relate, and Link. It then defines nine metapaths (MP1–MP9) — "attacked the same IP ($Hacker \to IP \to Hacker$)," "infected the same page," "permeated the same domain," "the complete web attack chain," and so on — to capture the various associations among hackers.

## Method

HGHAN's node embedding process consists of five stages.

### Stage 1: Node and Attribute Extraction

Extract the node set $V$ and the corresponding attribute set $X_v$ from the dataset (WAHIN).

### Stage 2: LSTM-Based Attribute Feature Encoding

Nodes of different types have attributes of different dimensions and semantics. Since these heterogeneous attributes cannot be used as-is, an **LSTM** converts each node's heterogeneous attributes into a vector of unified dimension. First, the input vector length is computed from the number of node types.

$$
\mathrm{Len}(\mathrm{Input}) = N = \sum_{i=0}^{K-1} |T_{v_i}|, \qquad K = |T_v|
$$

Here $|T_{v_i}|$ is the dimension of each node and $K$ is the number of node types. Each node's original attributes are mapped into a $K$-dimensional zero-matrix space and then fed into the LSTM. The LSTM cell updates its state as follows.

$$
h_t = o_t \ast \tanh(c_t), \qquad c_t = f_t \ast c_{t-1} + i_t \ast c_t'
$$

Here $f_t$ is the forget gate and $i_t$ the input gate. The cell state $c_t$ updates itself based on the previous state, and the output of a fully connected layer on top of the LSTM output layer becomes the **set of attribute feature vectors $A = \{A_1, A_2, \dots, A_n\}$**.

### Stage 3: Metapath2Vec-Based Structural Feature Encoding

WAHIN and the metapath set are fed into the **Metapath2Vec** algorithm. Metapath-guided random walks generate a large number of paths carrying graph semantics, and skip-gram is applied over these paths to compute node vectors. The objective is to maximize the following conditional probability.

$$
\arg\max_{\theta} \sum_{v \in V} \sum_{t \in T} \sum_{c_t \in N_t(v)} \log p(c_t \mid v; \theta)
$$

Here $N_t(v)$ is the set of neighbors of vertex $v$ that are of type $t$, and $p(c_t \mid v; \theta)$ is a softmax function. This stage ignores the original node attributes and focuses solely on the structural information within the heterogeneous graph, producing the **structural feature set $S = \{S_1, S_2, \dots, S_n\}$**, which allows similarity between hacker groups to be computed.

### Stage 4: Feature Fusion

The attribute features $A$ and structural features $S$ are fused into the final features $F$ by averaging each dimensional component.

$$
F_{i_k} = \frac{S_{i_k} + A_{i_k}}{2}, \qquad 0 < k < K
$$

Here $F_i$ is the mapped feature of node $v_i$, and $F_{i_k}$ is a component of the $K$-dimensional final feature vector.

### Stage 5: HAN Attention and Node Embedding

The fused features $F$ are fed into the **HAN (Heterogeneous Graph Attention Network)** algorithm. HAN uses two levels of attention.

- **Node-level attention**: learns the importance of neighboring nodes under each metapath.

$$
z_i^{\Phi} = \sigma\!\left(\sum_{j \in \mathcal{N}_i^{\Phi}} \alpha_{ij}^{\Phi} \cdot h_j'\right)
$$

Here $\alpha_{ij}^{\Phi}$ is the node-pair weight coefficient, $h_j'$ is the unified mapped feature space, and $\mathcal{N}_i^{\Phi}$ is the metapath-based neighbor set.

- **Semantic-level attention**: aggregates the node-level attention results across metapaths and learns the importance of each metapath.

$$
Z = \sum_{p=1}^{P} \beta_{\Phi_p} \cdot Z_{\Phi_p}
$$

Here $\beta_{\Phi_p}$ is the semantic weight, $Z_{\Phi_p}$ the node-level attention weight, and $P$ the metapath set. Node embedding proceeds in this manner, producing the final embedding vectors $E = \{E_1, E_2, \dots, E_n\}$.

### Classification: BiLSTM

The resulting node embeddings are used to classify hacker groups versus individual hackers. A **BiLSTM (bidirectional LSTM)** is used: the forward and backward vectors are each fed into an LSTM, and the two outputs are combined to capture bidirectional relational information. Since HGHAN operates on an undirected graph, BiLSTM is a better fit for the classification task.

## Experiments

### Dataset

- The web hacking dataset provided by HCRL (Hacking and Countermeasure Research Lab) is used.
- It comprises 183,326 complete web attack records, 374,397 nodes, and 936,808 edges.
- It contains 6,580 hacker nodes and 499 hacker groups; the main objective is to classify the 6,580 hacker nodes.

### Results

- **Node embedding (LSTM attributes)**: training stabilized after 10 epochs, maintaining an accuracy of 99.94% and a loss of 0.0017. This shows that node types can be distinguished well even when heterogeneous attributes of different node types are mapped into a space of the same dimension.
- **Structural features**: path generation over the longest metapath (MP11, the complete web attack chain) yielded 761,946 pieces of semantic information, and t-SNE visualization showed the various node types clustering together, confirming that the structural features are well separated in space.
- **Classification performance**: BiLSTM achieved an accuracy of 0.9961, outperforming LSTM. HAN's node embeddings contributed substantially to accuracy, and the heterogeneous graph's flexibility over homogeneous graphs — in terms of information propagation and data volume — improved classification. The full HGHAN model reached an AUC of 0.9955, far ahead of comparison models such as LSTM+HAN (0.9418), GAT (0.8528), and LSTM (0.8590).

## Takeaways

- The core contribution is modeling web attack data as a **heterogeneous information network (WAHIN)** with five node types and diverse relations, and capturing even indirect associations among hackers through nine metapaths.
- The design of separately encoding **heterogeneous node attributes (LSTM)** and **graph structure (Metapath2Vec)**, fusing them, and integrating them via HAN's dual attention handles heterogeneous information effectively.
- Matching the undirected nature of the graph, BiLSTM is chosen as the classifier to exploit bidirectional relational information.

### Limitations

- HAN requires computing an adjacency matrix of size $N^2$ for $N$ nodes, so computational cost grows with the number of nodes. Subgraph training reduces this cost, but whether it constitutes better training is hard to prove since a direct comparison against full-graph training is infeasible.
- A model limited to attack detection cannot explore relationships among hackers.
- Adding new nodes requires retraining. The authors suggest future work on realizing propagation updates over the heterogeneous graph instead of retraining.
