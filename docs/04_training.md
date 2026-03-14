# Chapter 4: Training the Model

We now have data (Chapter 2) and a model (Chapter 3). In this chapter we'll implement the full training pipeline: the **loss function**, a **backpropagation** engine, an **optimizer**, and the **training loop** — all from scratch.

## 1. Cross-Entropy Loss

Language modeling is a classification problem: at each position the model predicts a distribution over the vocabulary, and the true label is the next token. The standard loss is **cross-entropy**:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\log P(y_i \mid x_{<i})$$

where $P(y_i \mid x_{<i})$ is the softmax probability assigned to the correct token.

### Implementation

```python
def cross_entropy_loss(logits, targets):
    """
    Compute mean cross-entropy loss.

    Args:
        logits:  (batch, seq_len, vocab_size) — raw scores from the model
        targets: (batch, seq_len) — ground truth token IDs

    Returns:
        loss:     scalar
        d_logits: (batch, seq_len, vocab_size) — gradient of loss w.r.t. logits
    """
    batch, seq_len, vocab_size = logits.shape

    # Softmax
    probs = softmax(logits)

    # Gather probability of correct class
    # Advanced indexing: probs[b, t, targets[b, t]]
    b_idx = np.arange(batch)[:, None]
    t_idx = np.arange(seq_len)[None, :]
    correct_probs = probs[b_idx, t_idx, targets]

    # Negative log-likelihood (clipped for numerical safety)
    loss = -np.mean(np.log(np.clip(correct_probs, 1e-9, None)))

    # Gradient of softmax cross-entropy: probs - one_hot(target)
    d_logits = probs.copy()
    d_logits[b_idx, t_idx, targets] -= 1.0
    d_logits /= (batch * seq_len)

    return loss, d_logits
```

The gradient formula is elegantly simple: $\frac{\partial \mathcal{L}}{\partial z_i} = p_i - \mathbb{1}[i = y]$, where $p_i$ is the softmax probability.

## 2. Backpropagation

To train, we need gradients of the loss with respect to every parameter. We implement backward passes for each layer.

!!! note "Notation"
    We store both the forward output and the gradient computation together. Each layer's `forward()` now caches intermediate values, and `backward()` computes parameter gradients and returns the input gradient.

### Layer-by-Layer Backward Passes

The full backward pass flows through the model in reverse order:

```
d_logits → Output projection
         → Final LayerNorm
         → Transformer Block N (backward)
         → ...
         → Transformer Block 1 (backward)
         → Embedding (backward)
```

Each backward function follows the same pattern:

1. Receive the gradient of the loss w.r.t. the layer's output ($\frac{\partial \mathcal{L}}{\partial \mathbf{y}}$)
2. Compute gradients w.r.t. parameters ($\frac{\partial \mathcal{L}}{\partial W}$)
3. Return gradient w.r.t. the layer's input ($\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$)

### Linear Layer Backward

For $y = xW + b$:

$$\frac{\partial \mathcal{L}}{\partial W} = x^T \cdot \frac{\partial \mathcal{L}}{\partial y}$$

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot W^T$$

$$\frac{\partial \mathcal{L}}{\partial b} = \sum \frac{\partial \mathcal{L}}{\partial y}$$

### GELU Backward

$$\frac{\partial}{\partial x}\text{GELU}(x) \approx 0.5 \tanh(u) + 0.5 + \frac{0.5 x}{\cosh^2(u)} \cdot \sqrt{\frac{2}{\pi}}(1 + 3 \times 0.044715 \, x^2)$$

where $u = \sqrt{2/\pi}\,(x + 0.044715\,x^3)$.

### Full Training Code

The complete implementation with backward passes is in `src/train.py`. See the source code below for the full working version.

## 3. Optimizer: Adam

We use the **Adam** optimizer, the de facto standard for training Transformers:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

```python
class AdamOptimizer:
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
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

## 4. Learning Rate Schedule

We use a **warmup + cosine decay** schedule:

- **Warmup** (first $w$ steps): linearly increase LR from 0 to $\eta_{max}$
- **Cosine decay** (remaining steps): smoothly decrease LR to $\eta_{min}$

$$\eta_t = \begin{cases}
\eta_{max} \cdot \frac{t}{w} & \text{if } t < w \\
\eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{\pi(t - w)}{T - w}\right)\right) & \text{otherwise}
\end{cases}$$

```python
def cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=1e-6):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))
```

## 5. Gradient Clipping

Large gradients can destabilize training. We clip the **global gradient norm**:

$$\text{if } \|\mathbf{g}\|_2 > c, \quad \mathbf{g} \leftarrow c \cdot \frac{\mathbf{g}}{\|\mathbf{g}\|_2}$$

```python
def clip_grad_norm(grads, max_norm=1.0):
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        grads = [g * scale for g in grads]
    return grads, total_norm
```

## 6. Dropout Regularization

With a small training corpus, the model can easily memorize the data without learning general patterns. **Dropout** randomly sets a fraction of activations to zero during training, forcing the model to be more robust:

$$\text{dropout}(x, p) = \frac{x \cdot \text{mask}}{1 - p}, \quad \text{mask}_i \sim \text{Bernoulli}(1 - p)$$

The scaling by $\frac{1}{1-p}$ ensures the expected value stays the same. In our implementation, dropout is applied after the embedding layer, after attention output, and after the feed-forward output in each block. Dropout is **disabled during inference** (generation uses the model's regular forward pass without dropout).

## 7. The Training Loop

Here's the complete training loop that ties everything together:

```python
def train(model, inputs, targets, epochs=10, batch_size=32,
          max_lr=3e-4, warmup_steps=100):
    params = model.get_all_parameters()
    optimizer = AdamOptimizer(params, lr=max_lr)

    n_batches = len(inputs) // batch_size
    total_steps = epochs * n_batches

    step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        n = 0

        for batch_input, batch_target in create_batches(
                inputs, targets, batch_size):
            # 1. Forward pass
            logits = model.forward(batch_input)

            # 2. Compute loss
            loss, d_logits = cross_entropy_loss(logits, batch_target)

            # 3. Backward pass (see src/train.py for full implementation)
            grads = backward(model, d_logits, ...)

            # 4. Gradient clipping
            grads, grad_norm = clip_grad_norm(grads, max_norm=1.0)

            # 5. Update learning rate
            lr = cosine_lr_schedule(step, total_steps, warmup_steps, max_lr)
            optimizer.lr = lr

            # 6. Update parameters
            optimizer.step(grads)

            epoch_loss += loss
            n += 1
            step += 1

        avg_loss = epoch_loss / n
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | "
              f"LR: {lr:.6f}")
```

## 8. Numerical Gradient Checking

To verify our backward pass is correct, we compare analytical gradients with **finite-difference** numerical gradients:

$$\frac{\partial \mathcal{L}}{\partial \theta_i} \approx \frac{\mathcal{L}(\theta_i + \epsilon) - \mathcal{L}(\theta_i - \epsilon)}{2\epsilon}$$

```python
def gradient_check(model, x, targets, param, eps=1e-5):
    """Check analytical gradient vs numerical gradient for one parameter."""
    # Analytical gradient (from backprop)
    logits = model.forward(x)
    loss, d_logits = cross_entropy_loss(logits, targets)
    grads = backward(model, d_logits, ...)  # get gradient for param

    # Numerical gradient
    num_grad = np.zeros_like(param)
    flat = param.ravel()
    for i in range(min(flat.size, 20)):  # check first 20 elements
        old_val = flat[i]

        flat[i] = old_val + eps
        logits_plus = model.forward(x)
        loss_plus, _ = cross_entropy_loss(logits_plus, targets)

        flat[i] = old_val - eps
        logits_minus = model.forward(x)
        loss_minus, _ = cross_entropy_loss(logits_minus, targets)

        num_grad.ravel()[i] = (loss_plus - loss_minus) / (2 * eps)
        flat[i] = old_val

    return analytical_grad, num_grad
```

## 8. Putting It All Together

A complete training run would look like this:

```python
import numpy as np
from tokenizer import BPETokenizer
from data_preprocessing import load_text_data, clean_text, \
    create_training_sequences
from model import TinyLLM
from train import train

# Load and preprocess data
text = load_text_data(['data/corpus.txt'])
text = clean_text(text)

# Tokenize
tokenizer = BPETokenizer(vocab_size=5000)
tokenizer.train(text)
token_ids = tokenizer.encode(text)

# Create training sequences
inputs, targets = create_training_sequences(token_ids, seq_length=128)
print(f"Training sequences: {inputs.shape[0]}")

# Build model
model = TinyLLM(
    vocab_size=5000, d_model=128, n_heads=4,
    n_layers=4, d_ff=512, max_seq_len=128,
)
print(f"Parameters: {model.count_parameters():,}")

# Train!
train(model, inputs, targets, epochs=10, batch_size=32, max_lr=3e-4)
```

## Training Tips

!!! tip "Practical Advice"
    - **Start small**: Test with 1 layer and a tiny dataset first to verify correctness
    - **Monitor loss**: It should decrease steadily; spikes indicate bugs or too-high LR
    - **Gradient norms**: Log these — if they explode or vanish, something is wrong
    - **Save checkpoints**: Periodically save model weights so you can resume training
    - **Batch size**: Larger batches give more stable gradients; smaller batches explore more

## Summary

In this chapter we implemented:

| Component | Purpose |
|-----------|---------|
| `cross_entropy_loss` | Compute loss and gradient |
| Backward passes | Gradient computation for every layer |
| `AdamOptimizer` | Adaptive parameter updates |
| `cosine_lr_schedule` | Learning rate warmup + decay |
| `clip_grad_norm` | Prevent gradient explosion |
| Training loop | Orchestrate the full pipeline |
| Gradient checking | Verify backward correctness |

[Continue to Chapter 5: Evaluation →](05_evaluation.md)
