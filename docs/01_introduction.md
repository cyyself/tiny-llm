# Chapter 1: Introduction to Large Language Models

## What Are Large Language Models?

A **Large Language Model (LLM)** is a type of artificial intelligence model trained to understand and generate human language. At their core, LLMs are statistical models that learn the probability distribution of sequences of words (or tokens) from massive amounts of text data.

Given a sequence of tokens $x_1, x_2, \ldots, x_{t-1}$, an LLM learns to predict the next token $x_t$ by modeling:

$$P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

This is called **autoregressive language modeling** — the model generates text one token at a time, each time conditioning on all previously generated tokens.

### Examples of LLMs

| Model | Organization | Parameters | Year |
|-------|-------------|-----------|------|
| GPT-2 | OpenAI | 1.5B | 2019 |
| GPT-3 | OpenAI | 175B | 2020 |
| LLaMA | Meta | 7B–65B | 2023 |
| GPT-4 | OpenAI | ~1.8T (est.) | 2023 |

Even though these models have billions or trillions of parameters, the underlying architecture is surprisingly elegant. In this tutorial, we'll build a **tiny version** (a few million parameters) to learn the fundamentals.

## Why Are LLMs Important?

LLMs have revolutionized natural language processing (NLP) and AI more broadly:

- **Text Generation**: Writing essays, code, poetry, and more
- **Question Answering**: Understanding and answering questions about documents
- **Translation**: Converting text between languages
- **Summarization**: Condensing long documents into short summaries
- **Reasoning**: Solving math problems, logical puzzles, and coding challenges

The key insight is that a single model, trained on diverse text data, can perform all of these tasks without task-specific training — a property called **emergent behavior**.

## The Transformer Architecture: A High-Level Overview

Almost all modern LLMs are based on the **Transformer** architecture, introduced in the landmark paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

### The Big Picture

A Transformer-based language model consists of the following key components stacked together:

```
Input Text
    │
    ▼
┌──────────────────┐
│  Tokenization    │  Convert text → token IDs
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Token Embedding │  Token IDs → dense vectors
│  + Positional    │  Add position information
│    Encoding      │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Transformer     │  ×N layers
│  Block           │
│  ┌─────────────┐ │
│  │ Multi-Head  │ │
│  │ Self-Attn   │ │
│  └─────────────┘ │
│  ┌─────────────┐ │
│  │ Feed-Forward│ │
│  │ Network     │ │
│  └─────────────┘ │
│  (+ LayerNorm    │
│   + Residual     │
│   connections)   │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Output Layer    │  Project → vocabulary logits
│  (Linear + SM)   │  Apply softmax → probabilities
└──────────────────┘
    │
    ▼
Next Token Prediction
```

Let's briefly describe each component. We'll implement all of them from scratch in Chapter 3.

### 1. Tokenization

Before any processing, raw text must be converted into numerical representations. A **tokenizer** splits text into smaller units called **tokens** — these could be words, subwords, or even individual characters.

For example, using Byte-Pair Encoding (BPE):

```
"The cat sat" → ["The", " cat", " sat"] → [1024, 5765, 3290]
```

### 2. Token Embedding + Positional Encoding

Each token ID is mapped to a dense vector of dimension $d_{model}$ via a learned **embedding matrix** $E \in \mathbb{R}^{V \times d_{model}}$, where $V$ is the vocabulary size.

Since Transformers process all tokens in parallel (unlike RNNs), they have no inherent notion of order. **Positional encodings** are added to inject position information:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

### 3. Self-Attention Mechanism

The **self-attention** mechanism is the heart of the Transformer. It allows each token to "attend to" (i.e., gather information from) every other token in the sequence.

Given input matrix $X$, we compute three matrices:

- **Query**: $Q = XW_Q$
- **Key**: $K = XW_K$  
- **Value**: $V = XW_V$

The attention output is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

The $\frac{1}{\sqrt{d_k}}$ scaling prevents the dot products from growing too large.

**Multi-Head Attention** runs multiple attention operations in parallel (with different learned projections), then concatenates the results:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

For **decoder-only** models (like GPT), we also apply a **causal mask** so that each token can only attend to itself and previous tokens — not future ones.

### 4. Feed-Forward Network

After attention, each position passes through a simple two-layer fully connected network:

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

This operates independently on each position, adding non-linear transformation capacity.

### 5. Layer Normalization and Residual Connections

Each sub-layer (attention and FFN) is wrapped with:

- **Residual connection**: $\text{output} = x + \text{SubLayer}(x)$
- **Layer normalization**: normalizes across features to stabilize training

### 6. Output Layer

The final hidden states are projected back to vocabulary size via a linear layer, and a **softmax** function converts these logits into a probability distribution over the next token:

$$P(x_t | x_{<t}) = \text{softmax}(h_t W_{vocab} + b)$$

## Decoder-Only vs Encoder-Decoder

There are two main Transformer variants:

| | Encoder-Decoder | Decoder-Only |
|---|---|---|
| **Examples** | T5, BART | GPT, LLaMA |
| **Use case** | Seq-to-seq (translation) | Text generation |
| **Attention** | Cross-attention + self | Causal self-attention |
| **Our focus** | ✗ | ✓ |

We'll build a **decoder-only** Transformer, which is simpler and the basis for most modern LLMs.

## Our Tiny LLM: Design Choices

For this tutorial, we'll build a model with these specifications:

| Hyperparameter | Value |
|---------------|-------|
| Vocabulary size ($V$) | 400 |
| Embedding dimension ($d_{model}$) | 64 |
| Number of layers ($N$) | 2 |
| Number of attention heads ($h$) | 4 |
| Head dimension ($d_k$) | 16 |
| FFN hidden dimension | 256 |
| Max sequence length | 64 |
| Dropout rate | 0.05 |
| **Total parameters** | **~121K** |

This is deliberately small relative to real LLMs, but well-matched to our Q&A training corpus. With 120 epochs of training, overlapping sequence windows, and dropout regularization, the model achieves test perplexity of ~4 and correctly answers question-answer pairs it has been trained on.

!!! tip "Scaling Up"
    If you have a larger dataset (1MB+), increase `vocab_size` to 2000–5000, `d_model` to 128, `n_layers` to 4, and reduce epochs. The `run_pipeline.py` script accepts command-line arguments for all these.

## What's Next?

In the next chapter, we'll learn how to collect and preprocess text data to feed into our model. We'll implement a Byte-Pair Encoding (BPE) tokenizer from scratch.

[Continue to Chapter 2: Data Collection & Preprocessing →](02_data_preprocessing.md)
