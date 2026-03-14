"""
Tiny LLM — A decoder-only Transformer implemented from scratch with NumPy.

No PyTorch, no TensorFlow — just NumPy and math.

Architecture:
    Token Embedding + Sinusoidal Positional Encoding
    → N × (Pre-LayerNorm → MultiHeadAttention → Residual
            → Pre-LayerNorm → FeedForward → Residual)
    → Final LayerNorm
    → Linear output projection (weight-tied with embedding)
"""

import numpy as np


# ======================================================================
# Utility functions
# ======================================================================

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def relu(x):
    return np.maximum(0, x)


def gelu(x):
    """Gaussian Error Linear Unit (approximate)."""
    return 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    ))


def xavier_init(shape, rng=None):
    """Xavier / Glorot uniform initialization."""
    rng = rng or np.random.default_rng()
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)


def causal_mask(seq_len):
    """Upper-triangular mask filled with -inf for future positions."""
    mask = np.full((seq_len, seq_len), -np.inf, dtype=np.float32)
    mask = np.triu(mask, k=1)
    return mask


def sinusoidal_positional_encoding(max_len, d_model):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
    )
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


# ======================================================================
# Layers
# ======================================================================

class Embedding:
    """Lookup-table embedding layer."""

    def __init__(self, vocab_size, d_model, rng=None):
        self.weight = xavier_init((vocab_size, d_model), rng)

    def forward(self, token_ids):
        """token_ids: (batch, seq_len) → (batch, seq_len, d_model)"""
        return self.weight[token_ids]


class LayerNorm:
    """Layer normalization across the last dimension."""

    def __init__(self, d_model, eps=1e-5):
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.beta = np.zeros(d_model, dtype=np.float32)
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class MultiHeadAttention:
    """Multi-head causal self-attention."""

    def __init__(self, d_model, n_heads, rng=None):
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = xavier_init((d_model, d_model), rng)
        self.W_k = xavier_init((d_model, d_model), rng)
        self.W_v = xavier_init((d_model, d_model), rng)
        self.W_o = xavier_init((d_model, d_model), rng)

    def forward(self, x):
        """x: (batch, seq_len, d_model) → (batch, seq_len, d_model)"""
        batch, seq_len, d_model = x.shape

        # Project to Q, K, V
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Split into heads: (batch, n_heads, seq_len, d_k)
        Q = Q.reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        scores = scores + causal_mask(seq_len)
        attn_weights = softmax(scores, axis=-1)
        attn_output = attn_weights @ V  # (batch, n_heads, seq_len, d_k)

        # Concatenate heads and project
        attn_output = (attn_output
                       .transpose(0, 2, 1, 3)
                       .reshape(batch, seq_len, d_model))
        return attn_output @ self.W_o


class FeedForward:
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, d_model, d_ff, rng=None):
        self.W1 = xavier_init((d_model, d_ff), rng)
        self.b1 = np.zeros(d_ff, dtype=np.float32)
        self.W2 = xavier_init((d_ff, d_model), rng)
        self.b2 = np.zeros(d_model, dtype=np.float32)

    def forward(self, x):
        h = gelu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2


class TransformerBlock:
    """Single Transformer decoder block (Pre-LayerNorm variant)."""

    def __init__(self, d_model, n_heads, d_ff, rng=None):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, rng)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, rng)

    def forward(self, x):
        # Pre-norm self-attention + residual
        h = self.ln1.forward(x)
        x = x + self.attn.forward(h)
        # Pre-norm FFN + residual
        h = self.ln2.forward(x)
        x = x + self.ffn.forward(h)
        return x


# ======================================================================
# Full model
# ======================================================================

class TinyLLM:
    """A tiny GPT-style decoder-only Transformer."""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 max_seq_len, rng=None):
        rng = rng or np.random.default_rng(42)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Layers
        self.embedding = Embedding(vocab_size, d_model, rng)
        self.pos_enc = sinusoidal_positional_encoding(max_seq_len, d_model)

        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, rng)
            for _ in range(n_layers)
        ]
        self.ln_final = LayerNorm(d_model)

        # Output projection — weight-tied with embedding
        self.output_proj = self.embedding.weight

    def forward(self, token_ids):
        """
        Forward pass.

        Args:
            token_ids: np.ndarray of shape (batch, seq_len) with int dtype.

        Returns:
            logits: np.ndarray of shape (batch, seq_len, vocab_size).
        """
        batch, seq_len = token_ids.shape

        # Embedding + positional encoding
        x = self.embedding.forward(token_ids)
        x = x + self.pos_enc[:seq_len]

        # Transformer blocks
        for block in self.blocks:
            x = block.forward(x)

        # Final layer norm
        x = self.ln_final.forward(x)

        # Project to logits
        logits = x @ self.output_proj.T
        return logits

    def count_parameters(self):
        """Count total number of trainable scalar parameters."""
        total = self.embedding.weight.size
        for b in self.blocks:
            total += (b.attn.W_q.size + b.attn.W_k.size +
                      b.attn.W_v.size + b.attn.W_o.size)
            total += (b.ffn.W1.size + b.ffn.b1.size +
                      b.ffn.W2.size + b.ffn.b2.size)
            total += (b.ln1.gamma.size + b.ln1.beta.size +
                      b.ln2.gamma.size + b.ln2.beta.size)
        total += self.ln_final.gamma.size + self.ln_final.beta.size
        return total

    def get_all_parameters(self):
        """Return a flat list of all parameter arrays (for the optimizer)."""
        params = [self.embedding.weight]
        for b in self.blocks:
            params.extend([
                b.ln1.gamma, b.ln1.beta,
                b.attn.W_q, b.attn.W_k, b.attn.W_v, b.attn.W_o,
                b.ln2.gamma, b.ln2.beta,
                b.ffn.W1, b.ffn.b1, b.ffn.W2, b.ffn.b2,
            ])
        params.extend([self.ln_final.gamma, self.ln_final.beta])
        return params


# ======================================================================
# Quick smoke test
# ======================================================================
if __name__ == '__main__':
    model = TinyLLM(
        vocab_size=5000,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=128,
    )
    print(f"Total parameters: {model.count_parameters():,}")

    x = np.random.randint(0, 5000, size=(2, 16))
    logits = model.forward(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {logits.shape}")

    # Verify softmax output sums to 1
    probs = softmax(logits, axis=-1)
    print(f"Prob sums (should be ≈1): {probs.sum(axis=-1)[0, :3]}")
