---
title: "Attention Is All You Need"
date: 2024-05-03
tags: ["Transformer", "Attention", "NLP", "Deep Learning"]
categories: ["Paper Review"]
series: ["AI Algorithms"]
math: true
summary: "A review of the paper that introduced the Transformer, which processes sequences with attention alone — no recurrence or convolution — achieving both parallelization and long-range dependency learning, and setting a new SOTA in machine translation."
---

> **Paper**: Attention Is All You Need — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin, NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

## TL;DR

The Transformer is a model that encodes and decodes sequences using **only attention mechanisms**, with no recurrent (RNN) or convolutional (CNN) structure. By eliminating sequential computation, it makes training fully parallelizable, and by connecting any two positions with a constant number of operations, it makes learning long-range dependencies easy. It achieved 28.4 BLEU on WMT 2014 English-to-German and 41.8 BLEU on English-to-French translation — a new machine translation SOTA at the time — while substantially reducing training cost.

## Background

### Limitations of Prior Sequence Models

Recurrent models in the RNN/LSTM/GRU family process inputs sequentially. The hidden state $h_t$ at time step $t$ depends on the previous state $h_{t-1}$ and the current input, so this sequential dependency forces **as many serial operations as the sequence length**. Training therefore cannot be parallelized along the sequence dimension, computation becomes inefficient on long sentences, and information between distant tokens must traverse just as many steps, making long-range dependencies hard to learn.

Attention mechanisms were already widely used in encoder-decoder translation models, but mostly as an **add-on** to recurrent architectures. This paper strips out the recurrence entirely and builds the whole model from attention alone.

### Evaluating Generated Sentences: BLEU and ROUGE

Machine translation performance is measured primarily with BLEU. BLEU is a precision-oriented metric that computes, via n-gram matching, **how much of the generated sentence is contained in the reference sentence**. Conversely, ROUGE is a recall-oriented metric that measures **how much of the reference sentence is contained in the generated sentence**; the two metrics offer complementary perspectives. BLEU is the standard for reporting translation quality.

## Method

The Transformer follows an encoder-decoder architecture composed of an encoder stack and a decoder stack. In the paper's base configuration, the encoder and decoder each stack $N=6$ identical layers.

### Scaled Dot-Product Attention

At the heart of attention is the interaction of three vectors: Query, Key, and Value.

- **Query**: a vector representing what we currently want to attend to.
- **Key**: the element compared against the Query, associated with each part of the input sequence. It provides the basis for computing how much attention each position in the input sequence should receive relative to the current Query — i.e., a compatibility score with the Query.
- **Value**: the information that is actually weighted, summed, and passed to the output.

Similarities are computed as dot products between the Query and all Keys, scaled by $\sqrt{d_k}$, passed through a softmax to form weights, and used to take a weighted sum of the Values.

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
$$

The division by $\sqrt{d_k}$ exists because as $d_k$ grows, the variance of the dot products grows, pushing the softmax into saturated regions where gradients vanish.

### Multi-Head Attention

Instead of a single attention function, the Queries, Keys, and Values are projected into $h$ subspaces with different learned linear transformations, attention is computed in parallel in each, and the results are concatenated and projected again.

$$
\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\,W^O
$$

$$
\mathrm{head}_i = \mathrm{Attention}(QW_i^Q,\, KW_i^K,\, VW_i^V)
$$

By letting each head attend to different positional relationships in different representation subspaces, this preserves information that a single attention function would lose through averaging. The paper uses $h=8$ heads.

### Three Uses of Attention

1. **Encoder self-attention**: Queries, Keys, and Values all come from the output of the previous encoder layer, and each position attends to every position in the input sequence.
2. **Decoder masked self-attention**: future tokens beyond the current position are masked out to preserve autoregressive generation.
3. **Encoder-decoder attention**: Queries come from the decoder, while Keys and Values come from the encoder output. This is the traditional encoder-decoder attention, letting every position in the decoder attend to the entire input sequence.

### Positional Encoding

RNNs and LSTMs process data sequentially, so order information is built into the architecture. The Transformer, by contrast, processes sequences in parallel, so token position information must be injected separately. To this end, positional encodings are added to the input embeddings at the bottom of the encoder and decoder stacks. The positional encoding has the same dimension $d_{model}$ as the embeddings, so it can be summed directly with the token embeddings; position information is thereby embedded alongside token information, allowing attention to take position into account when processing each token. The paper uses sine and cosine functions of different frequencies.

$$
PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right), \qquad
PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

### Other Components

Each encoder and decoder layer consists of a multi-head attention sublayer and a position-wise fully connected feed-forward sublayer, with residual connections and layer normalization applied around each sublayer. This residual structure keeps training stable even in deep stacks.

## Experiments

### Training Setup

The Adam optimizer is used with a learning rate schedule that warms up and then decays, and residual dropout and label smoothing are applied for regularization. Training uses the WMT 2014 English-German and English-French datasets.

### Results

- **WMT 2014 English-to-German**: 28.4 BLEU, surpassing the previous best results including ensembles.
- **WMT 2014 English-to-French**: 41.8 BLEU, a new single-model SOTA.
- These results were achieved at a **fraction of the training cost** of the previous best models — the parallelization enabled by removing sequential computation dramatically improved training efficiency.

The model also performed well when applied to English constituency parsing, demonstrating generality beyond translation.

## Takeaways

- The paper showed that a sequence model can be built from **attention alone**, without recurrence or convolution — an architecture that subsequently became the de facto standard for large language models.
- Removing sequential dependencies parallelizes training, and connecting any two positions with a constant path length fundamentally eases the learning of long-range dependencies.
- Multi-head attention increases expressiveness by capturing different relationships simultaneously in multiple representation subspaces.
- The idea of compensating for the order information lost to parallel processing via positional encoding became the starting point for a wide range of subsequent positional encoding research.
