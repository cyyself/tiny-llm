"""
Training pipeline for the tiny LLM — from-scratch backpropagation and Adam.

This module implements:
  - Cross-entropy loss (forward + backward)
  - Full backward pass through every layer of the Transformer
  - Adam optimizer
  - Cosine learning-rate schedule with warmup
  - Gradient clipping
  - The main training loop
"""

import numpy as np

from model import TinyLLM, softmax, gelu, causal_mask
from data_preprocessing import create_batches


# ======================================================================
# Loss
# ======================================================================

def cross_entropy_loss(logits, targets):
    """
    Cross-entropy loss for next-token prediction.

    Args:
        logits:  (batch, seq_len, vocab_size)
        targets: (batch, seq_len)

    Returns:
        loss:     scalar
        d_logits: gradient w.r.t. logits, same shape as logits
    """
    batch, seq_len, vocab_size = logits.shape

    probs = softmax(logits)

    b_idx = np.arange(batch)[:, None]
    t_idx = np.arange(seq_len)[None, :]
    correct_probs = probs[b_idx, t_idx, targets]

    loss = -np.mean(np.log(np.clip(correct_probs, 1e-9, None)))

    # Gradient: p - one_hot(y), averaged over batch*seq_len
    d_logits = probs.copy()
    d_logits[b_idx, t_idx, targets] -= 1.0
    d_logits /= (batch * seq_len)

    return loss, d_logits


# ======================================================================
# Backward helpers
# ======================================================================

def gelu_backward(x):
    """Derivative of the GELU approximation."""
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x ** 3)
    tanh_inner = np.tanh(inner)
    sech2 = 1.0 - tanh_inner ** 2
    d_inner = c * (1.0 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner


def layernorm_backward(d_out, x, gamma, eps=1e-5):
    """Backward pass for layer normalization.

    Returns:
        d_x:     gradient w.r.t. input x
        d_gamma: gradient w.r.t. gamma
        d_beta:  gradient w.r.t. beta
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    std_inv = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mean) * std_inv
    d_model = x.shape[-1]

    d_gamma = np.sum(d_out * x_hat, axis=tuple(range(d_out.ndim - 1)))
    d_beta = np.sum(d_out, axis=tuple(range(d_out.ndim - 1)))

    dx_hat = d_out * gamma
    d_var = np.sum(dx_hat * (x - mean) * (-0.5) * std_inv ** 3,
                   axis=-1, keepdims=True)
    d_mean = (np.sum(dx_hat * (-std_inv), axis=-1, keepdims=True)
              + d_var * np.mean(-2.0 * (x - mean), axis=-1, keepdims=True))
    d_x = dx_hat * std_inv + d_var * 2.0 * (x - mean) / d_model + d_mean / d_model

    return d_x, d_gamma, d_beta


# ======================================================================
# Full backward pass through the model
# ======================================================================

class ForwardCache:
    """Stores intermediate activations needed for the backward pass."""
    pass


def forward_with_cache(model, token_ids, dropout_rate=0.0, rng=None):
    """Forward pass that stores all intermediate values for backprop."""
    cache = ForwardCache()
    cache.token_ids = token_ids
    cache.dropout_rate = dropout_rate
    batch, seq_len = token_ids.shape

    # Embedding + positional encoding
    x = model.embedding.forward(token_ids)
    x = x + model.pos_enc[:seq_len]

    # Dropout after embedding
    if dropout_rate > 0 and rng is not None:
        mask = (rng.random(x.shape) >= dropout_rate).astype(np.float32) / (1 - dropout_rate)
        x = x * mask
        cache.dropout_mask_emb = mask
    else:
        cache.dropout_mask_emb = None

    cache.embedding_out = x.copy()

    # Transformer blocks
    cache.block_inputs = []
    cache.ln1_inputs = []
    cache.attn_inputs = []
    cache.attn_Q = []
    cache.attn_K = []
    cache.attn_V = []
    cache.attn_weights = []
    cache.attn_out_preproj = []
    cache.ln2_inputs = []
    cache.ffn_inputs = []
    cache.ffn_hidden_pre_act = []
    cache.dropout_masks_attn = []
    cache.dropout_masks_ffn = []

    for block in model.blocks:
        cache.block_inputs.append(x.copy())

        # Pre-norm attention
        ln1_out = block.ln1.forward(x)
        cache.ln1_inputs.append(x.copy())
        cache.attn_inputs.append(ln1_out.copy())

        # Attention forward (detailed)
        h = ln1_out
        d_model = h.shape[-1]
        Q = h @ block.attn.W_q
        K = h @ block.attn.W_k
        V = h @ block.attn.W_v

        n_heads = block.attn.n_heads
        d_k = block.attn.d_k

        Q = Q.reshape(batch, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)

        scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        scores = scores + causal_mask(seq_len)
        attn_w = softmax(scores, axis=-1)
        attn_out = attn_w @ V
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
        attn_out = attn_out @ block.attn.W_o

        cache.attn_Q.append(Q)
        cache.attn_K.append(K)
        cache.attn_V.append(V)
        cache.attn_weights.append(attn_w)
        cache.attn_out_preproj.append(
            (attn_w @ V).transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
        )

        # Dropout after attention
        if dropout_rate > 0 and rng is not None:
            mask = (rng.random(attn_out.shape) >= dropout_rate).astype(np.float32) / (1 - dropout_rate)
            attn_out = attn_out * mask
            cache.dropout_masks_attn.append(mask)
        else:
            cache.dropout_masks_attn.append(None)

        x = x + attn_out

        # Pre-norm FFN
        cache.ln2_inputs.append(x.copy())
        ln2_out = block.ln2.forward(x)
        cache.ffn_inputs.append(ln2_out.copy())

        h_pre = ln2_out @ block.ffn.W1 + block.ffn.b1
        cache.ffn_hidden_pre_act.append(h_pre.copy())
        h_act = gelu(h_pre)
        ffn_out = h_act @ block.ffn.W2 + block.ffn.b2

        # Dropout after FFN
        if dropout_rate > 0 and rng is not None:
            mask = (rng.random(ffn_out.shape) >= dropout_rate).astype(np.float32) / (1 - dropout_rate)
            ffn_out = ffn_out * mask
            cache.dropout_masks_ffn.append(mask)
        else:
            cache.dropout_masks_ffn.append(None)

        x = x + ffn_out

    # Final layer norm
    cache.final_ln_input = x.copy()
    x = model.ln_final.forward(x)
    cache.final_ln_output = x.copy()

    # Output projection (weight-tied)
    logits = x @ model.output_proj.T

    return logits, cache


def backward_pass(model, d_logits, cache):
    """
    Full backward pass through the model.

    Returns a list of gradients aligned with model.get_all_parameters().
    """
    batch, seq_len, vocab_size = d_logits.shape
    d_model = model.d_model
    n_layers = len(model.blocks)

    # --- Output projection (weight-tied with embedding) ---
    # logits = x @ W_emb.T  =>  d_x = d_logits @ W_emb,  d_W_emb += (d_logits^T @ x)^T
    d_x = d_logits @ model.output_proj  # (batch, seq, d_model)
    # Gradient for embedding weight (from output projection)
    d_emb_from_output = np.einsum('bsv,bsd->vd', d_logits, cache.final_ln_output)

    # --- Final LayerNorm ---
    d_x, d_ln_final_gamma, d_ln_final_beta = layernorm_backward(
        d_x, cache.final_ln_input, model.ln_final.gamma
    )

    # Accumulate gradients for each block (in reverse)
    block_grads = []

    for layer in reversed(range(n_layers)):
        block = model.blocks[layer]

        # --- FFN residual: x = x_prev + ffn_out ---
        d_ffn_out = d_x.copy()
        # d_x_prev passes through (residual)

        # Apply dropout mask for FFN (if used)
        if cache.dropout_masks_ffn[layer] is not None:
            d_ffn_out = d_ffn_out * cache.dropout_masks_ffn[layer]

        # --- FFN backward ---
        ffn_input = cache.ffn_inputs[layer]
        h_pre = cache.ffn_hidden_pre_act[layer]
        h_act = gelu(h_pre)

        # ffn_out = h_act @ W2 + b2
        d_h_act = d_ffn_out @ block.ffn.W2.T
        d_W2 = np.einsum('bsd,bsm->dm', h_act, d_ffn_out)
        d_b2 = np.sum(d_ffn_out, axis=(0, 1))

        # h_act = gelu(h_pre)
        d_h_pre = d_h_act * gelu_backward(h_pre)

        # h_pre = ffn_input @ W1 + b1
        d_ffn_input = d_h_pre @ block.ffn.W1.T
        d_W1 = np.einsum('bsd,bsm->dm', ffn_input, d_h_pre)
        d_b1 = np.sum(d_h_pre, axis=(0, 1))

        # --- LayerNorm 2 backward ---
        d_ln2, d_ln2_gamma, d_ln2_beta = layernorm_backward(
            d_ffn_input, cache.ln2_inputs[layer], block.ln2.gamma
        )
        d_x = d_x + d_ln2  # add gradient from residual

        # --- Attention residual: x_prev = x_block_input + attn_out ---
        d_attn_out = d_x.copy()

        # Apply dropout mask for attention (if used)
        if cache.dropout_masks_attn[layer] is not None:
            d_attn_out = d_attn_out * cache.dropout_masks_attn[layer]

        # --- Attention backward ---
        Q = cache.attn_Q[layer]
        K = cache.attn_K[layer]
        V = cache.attn_V[layer]
        attn_w = cache.attn_weights[layer]
        attn_input = cache.attn_inputs[layer]
        n_heads = block.attn.n_heads
        d_k = block.attn.d_k

        # attn_out_final = concat_heads @ W_o
        concat = cache.attn_out_preproj[layer]
        d_concat = d_attn_out @ block.attn.W_o.T
        d_Wo = np.einsum('bsd,bsm->dm', concat, d_attn_out)

        # Un-concatenate heads: (batch, seq, d_model) -> (batch, n_heads, seq, d_k)
        d_attn_out_heads = d_concat.reshape(
            batch, seq_len, n_heads, d_k
        ).transpose(0, 2, 1, 3)

        # attn_out = attn_w @ V
        d_attn_w = d_attn_out_heads @ V.transpose(0, 1, 3, 2)
        d_V = attn_w.transpose(0, 1, 3, 2) @ d_attn_out_heads

        # softmax backward: d_scores
        # For softmax(s) = p, d_s = p * (d_p - sum(d_p * p, axis=-1, keepdims))
        d_scores = attn_w * (d_attn_w - np.sum(d_attn_w * attn_w,
                                                 axis=-1, keepdims=True))
        d_scores /= np.sqrt(d_k)

        # scores = Q @ K^T / sqrt(d_k)
        d_Q = d_scores @ K
        d_K = d_scores.transpose(0, 1, 3, 2) @ Q

        # Reshape back: (batch, n_heads, seq, d_k) -> (batch, seq, d_model)
        d_Q = d_Q.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
        d_K = d_K.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
        d_V = d_V.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)

        # Q = h @ W_q, etc.
        d_h_from_q = d_Q @ block.attn.W_q.T
        d_Wq = np.einsum('bsd,bsm->dm', attn_input, d_Q)

        d_h_from_k = d_K @ block.attn.W_k.T
        d_Wk = np.einsum('bsd,bsm->dm', attn_input, d_K)

        d_h_from_v = d_V @ block.attn.W_v.T
        d_Wv = np.einsum('bsd,bsm->dm', attn_input, d_V)

        d_attn_input = d_h_from_q + d_h_from_k + d_h_from_v

        # --- LayerNorm 1 backward ---
        d_ln1, d_ln1_gamma, d_ln1_beta = layernorm_backward(
            d_attn_input, cache.ln1_inputs[layer], block.ln1.gamma
        )
        d_x = d_x + d_ln1  # add gradient from residual

        # Collect gradients for this block (same order as get_all_parameters)
        block_grads.append({
            'ln1_gamma': d_ln1_gamma, 'ln1_beta': d_ln1_beta,
            'W_q': d_Wq, 'W_k': d_Wk, 'W_v': d_Wv, 'W_o': d_Wo,
            'ln2_gamma': d_ln2_gamma, 'ln2_beta': d_ln2_beta,
            'W1': d_W1, 'b1': d_b1, 'W2': d_W2, 'b2': d_b2,
        })

    block_grads.reverse()  # restore forward order

    # --- Embedding gradient ---
    # d_x flows back into embedding lookup
    # Apply embedding dropout mask if used
    if cache.dropout_mask_emb is not None:
        d_x = d_x * cache.dropout_mask_emb
    d_emb = np.zeros_like(model.embedding.weight)
    np.add.at(d_emb, cache.token_ids, d_x)
    d_emb += d_emb_from_output  # add output projection gradient (weight tying)

    # --- Assemble final gradient list ---
    all_grads = [d_emb]
    for bg in block_grads:
        all_grads.extend([
            bg['ln1_gamma'], bg['ln1_beta'],
            bg['W_q'], bg['W_k'], bg['W_v'], bg['W_o'],
            bg['ln2_gamma'], bg['ln2_beta'],
            bg['W1'], bg['b1'], bg['W2'], bg['b2'],
        ])
    all_grads.extend([d_ln_final_gamma, d_ln_final_beta])

    return all_grads


# ======================================================================
# Optimizer
# ======================================================================

class AdamOptimizer:
    """Adam optimizer (Kingma & Ba, 2015)."""

    def __init__(self, params, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        """Perform one optimization step."""
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ======================================================================
# Learning rate schedule
# ======================================================================

def cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=1e-6):
    """Warmup + cosine decay learning rate schedule."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))


# ======================================================================
# Gradient clipping
# ======================================================================

def clip_grad_norm(grads, max_norm=1.0):
    """Clip gradients by global L2 norm."""
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        grads = [g * scale for g in grads]
    return grads, total_norm


# ======================================================================
# Training loop
# ======================================================================

def train(model, inputs, targets, epochs=10, batch_size=32,
          max_lr=3e-4, warmup_steps=100, max_grad_norm=1.0,
          log_every=10, dropout_rate=0.1):
    """
    Train the tiny LLM.

    Args:
        model:       TinyLLM instance
        inputs:      np.ndarray (N, seq_len)
        targets:     np.ndarray (N, seq_len)
        epochs:      number of training epochs
        batch_size:  mini-batch size
        max_lr:      peak learning rate
        warmup_steps: number of LR warmup steps
        max_grad_norm: gradient clipping threshold
        log_every:   print loss every N steps
        dropout_rate: dropout probability (0.0 = no dropout)
    """
    params = model.get_all_parameters()
    optimizer = AdamOptimizer(params, lr=max_lr)
    rng = np.random.default_rng(42)

    n_batches = max(len(inputs) // batch_size, 1)
    total_steps = epochs * n_batches

    step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        n = 0

        for batch_input, batch_target in create_batches(
                inputs, targets, batch_size):

            # 1. Forward pass (with cache for backprop)
            logits, cache = forward_with_cache(model, batch_input,
                                               dropout_rate=dropout_rate,
                                               rng=rng)

            # 2. Compute loss
            loss, d_logits = cross_entropy_loss(logits, batch_target)

            # 3. Backward pass
            grads = backward_pass(model, d_logits, cache)

            # 4. Gradient clipping
            grads, grad_norm = clip_grad_norm(grads, max_grad_norm)

            # 5. Update learning rate
            lr = cosine_lr_schedule(step, total_steps, warmup_steps, max_lr)
            optimizer.lr = lr

            # 6. Update parameters
            optimizer.step(grads)

            epoch_loss += loss
            n += 1
            step += 1

            if step % log_every == 0:
                print(f"  Step {step:>5d} | Loss: {loss:.4f} | "
                      f"Grad norm: {grad_norm:.4f} | LR: {lr:.6f}")

        avg_loss = epoch_loss / max(n, 1)
        print(f"Epoch {epoch + 1}/{epochs} | Avg loss: {avg_loss:.4f}")

    print("Training complete.")
    return model


# ======================================================================
# Demo / smoke test
# ======================================================================
if __name__ == '__main__':
    from model import TinyLLM

    rng = np.random.default_rng(42)

    # Small model for testing
    model = TinyLLM(
        vocab_size=200, d_model=64, n_heads=4,
        n_layers=2, d_ff=256, max_seq_len=32, rng=rng,
    )
    print(f"Parameters: {model.count_parameters():,}")

    # Fake training data
    N = 64
    seq_len = 16
    inputs = rng.integers(0, 200, size=(N, seq_len))
    targets = rng.integers(0, 200, size=(N, seq_len))

    train(model, inputs, targets, epochs=3, batch_size=16,
          max_lr=1e-3, warmup_steps=5, log_every=2)
