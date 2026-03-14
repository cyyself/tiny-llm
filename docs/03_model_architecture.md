# Chapter 3: Model Architecture

In this chapter we implement the full **decoder-only Transformer** from scratch using Python and NumPy.

!!! info "No PyTorch, No TensorFlow"
    Every operation — embedding lookup, attention, layer norm, softmax — is implemented by hand so you can see exactly what happens inside an LLM.

## Overview

Recall the component stack from Chapter 1:

```
Token IDs ──► Embedding + PosEnc ──► N × TransformerBlock ──► Linear ──► Softmax
```

We'll implement each piece as a Python class in `src/model.py` and wire them together into a `TinyLLM` model.

## 1. Utility Functions

Before building layers, we need a handful of numerical primitives.

### Softmax

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

We subtract $\max(x)$ for numerical stability.

```python
def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
```

### ReLU / GELU

We provide both activations. GELU is used in many modern LLMs:

$$\text{ReLU}(x) = \max(0, x)$$

$$\text{GELU}(x) \approx 0.5\,x\left[1 + \tanh\!\left(\sqrt{2/\pi}\,(x + 0.044715\,x^3)\right)\right]$$

```python
def relu(x):
    return np.maximum(0, x)

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    ))
```

### Xavier Initialization

We initialize weight matrices using **Xavier (Glorot) uniform** initialization to keep variance stable across layers:

$$W_{ij} \sim \mathcal{U}\!\left[-\sqrt{\frac{6}{n_{in}+n_{out}}},\;\sqrt{\frac{6}{n_{in}+n_{out}}}\right]$$

```python
def xavier_init(shape, rng=None):
    rng = rng or np.random.default_rng()
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)
```

## 2. Token Embedding + Positional Encoding

### Token Embedding

Each token ID $t$ is mapped to a $d_{model}$-dimensional vector via a lookup table (matrix row):

$$\mathbf{e}_t = E[t]$$

```python
class Embedding:
    def __init__(self, vocab_size, d_model, rng=None):
        self.weight = xavier_init((vocab_size, d_model), rng)

    def forward(self, token_ids):
        # token_ids: (batch, seq_len) -> (batch, seq_len, d_model)
        return self.weight[token_ids]
```

### Sinusoidal Positional Encoding

We use the fixed sinusoidal encoding from the original Transformer paper:

```python
def sinusoidal_positional_encoding(max_len, d_model):
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
    )
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe  # (max_len, d_model)
```

## 3. Layer Normalization

Layer normalization normalizes across the feature dimension:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

```python
class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.beta = np.zeros(d_model, dtype=np.float32)
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

## 4. Multi-Head Self-Attention

This is the core of the Transformer. For each head $i$:

$$\text{head}_i = \text{softmax}\!\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

We concatenate all heads and project:

$$\text{MultiHead} = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,W_O$$

### Causal Mask

For autoregressive (decoder-only) models, we mask out future positions so position $t$ can only attend to positions $\le t$:

```python
def causal_mask(seq_len):
    # Upper-triangular matrix of -inf (positions to mask out)
    mask = np.full((seq_len, seq_len), -np.inf, dtype=np.float32)
    mask = np.triu(mask, k=1)  # zero on and below diagonal
    return mask
```

### Full Attention Implementation

```python
class MultiHeadAttention:
    def __init__(self, d_model, n_heads, rng=None):
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = xavier_init((d_model, d_model), rng)
        self.W_k = xavier_init((d_model, d_model), rng)
        self.W_v = xavier_init((d_model, d_model), rng)
        self.W_o = xavier_init((d_model, d_model), rng)

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        # Linear projections
        Q = x @ self.W_q  # (batch, seq, d_model)
        K = x @ self.W_k
        V = x @ self.W_v

        # Reshape into heads: (batch, n_heads, seq, d_k)
        Q = Q.reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)

        # Apply causal mask
        mask = causal_mask(seq_len)
        scores = scores + mask  # broadcasting over batch & heads

        attn_weights = softmax(scores, axis=-1)

        # Weighted sum of values
        attn_output = attn_weights @ V  # (batch, n_heads, seq, d_k)

        # Concatenate heads
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)

        # Output projection
        return attn_output @ self.W_o
```

## 5. Feed-Forward Network

A simple two-layer MLP applied independently to each position:

$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1)\,W_2 + b_2$$

```python
class FeedForward:
    def __init__(self, d_model, d_ff, rng=None):
        self.W1 = xavier_init((d_model, d_ff), rng)
        self.b1 = np.zeros(d_ff, dtype=np.float32)
        self.W2 = xavier_init((d_ff, d_model), rng)
        self.b2 = np.zeros(d_model, dtype=np.float32)

    def forward(self, x):
        h = gelu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2
```

## 6. Transformer Block

One block = attention → residual → layernorm → FFN → residual → layernorm.

We use **Pre-LayerNorm** ordering (norm *before* the sub-layer), which is the default in modern LLMs:

```python
class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff, rng=None):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, rng)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, rng)

    def forward(self, x):
        # Pre-norm attention
        h = self.ln1.forward(x)
        x = x + self.attn.forward(h)

        # Pre-norm FFN
        h = self.ln2.forward(x)
        x = x + self.ffn.forward(h)

        return x
```

## 7. The Complete TinyLLM Model

```python
class TinyLLM:
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 max_seq_len, rng=None):
        rng = rng or np.random.default_rng(42)

        self.embedding = Embedding(vocab_size, d_model, rng)
        self.pos_enc = sinusoidal_positional_encoding(max_seq_len, d_model)

        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, rng)
            for _ in range(n_layers)
        ]
        self.ln_final = LayerNorm(d_model)

        # Output projection (tied with embedding weights)
        self.output_proj = self.embedding.weight  # weight tying

    def forward(self, token_ids):
        batch, seq_len = token_ids.shape

        # Embedding + positional encoding
        x = self.embedding.forward(token_ids)
        x = x + self.pos_enc[:seq_len]

        # Transformer blocks
        for block in self.blocks:
            x = block.forward(x)

        # Final layer norm
        x = self.ln_final.forward(x)

        # Project to vocabulary logits
        logits = x @ self.output_proj.T  # (batch, seq, vocab_size)

        return logits

    def count_parameters(self):
        total = 0
        # Embedding
        total += self.embedding.weight.size
        # Transformer blocks
        for b in self.blocks:
            total += b.attn.W_q.size + b.attn.W_k.size
            total += b.attn.W_v.size + b.attn.W_o.size
            total += b.ffn.W1.size + b.ffn.b1.size
            total += b.ffn.W2.size + b.ffn.b2.size
            total += b.ln1.gamma.size + b.ln1.beta.size
            total += b.ln2.gamma.size + b.ln2.beta.size
        total += self.ln_final.gamma.size + self.ln_final.beta.size
        return total
```

### Instantiating Our Model

```python
model = TinyLLM(
    vocab_size=5000,
    d_model=128,
    n_heads=4,
    n_layers=4,
    d_ff=512,
    max_seq_len=128,
)
print(f"Total parameters: {model.count_parameters():,}")
# ≈ 3,000,000 parameters
```

## 8. Bringing It All Together

Here's a quick forward pass test:

```python
import numpy as np
from model import TinyLLM

model = TinyLLM(
    vocab_size=5000, d_model=128, n_heads=4,
    n_layers=4, d_ff=512, max_seq_len=128,
)

# Fake input: batch of 2 sequences, each 16 tokens
x = np.random.randint(0, 5000, size=(2, 16))
logits = model.forward(x)

print(f"Input shape:  {x.shape}")       # (2, 16)
print(f"Output shape: {logits.shape}")   # (2, 16, 5000)
```

## Summary

In this chapter we built every component of a GPT-style Transformer from scratch:

| Component | Parameters | Purpose |
|-----------|-----------|---------|
| `Embedding` | $V \times d$ | Token → vector |
| Positional Encoding | — (fixed) | Inject position info |
| `MultiHeadAttention` | $4 \times d^2$ | Token interactions |
| `FeedForward` | $2 \times d \times d_{ff}$ | Non-linear transform |
| `LayerNorm` | $2 \times d$ | Stabilize training |
| Output projection | (tied) | Vector → vocab logits |

We also wrote a **C++ AVX-512 matrix multiply** kernel for acceleration.

[Continue to Chapter 4: Training the Model →](04_training.md)
