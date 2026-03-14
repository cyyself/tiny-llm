# Chapter 5: Evaluation

After training our tiny LLM, we need to measure how well it performs. In this chapter we implement two standard evaluation metrics — **perplexity** and **BLEU score** — and show how to generate text from the model.

## 1. Perplexity

**Perplexity (PPL)** is the most common metric for language models. It measures how "surprised" the model is by the test data. Lower is better.

$$\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(x_i \mid x_{<i})\right)$$

Intuitively:

- **PPL = 1** → the model is perfectly confident and always correct
- **PPL = V** (vocabulary size) → the model is random-guessing
- A good small LM on English text might achieve PPL ≈ 20–50

### Implementation

```python
def perplexity(model, token_ids, seq_length=128):
    """
    Compute perplexity of the model on a sequence of token IDs.
    """
    total_log_prob = 0.0
    total_tokens = 0

    for i in range(0, len(token_ids) - seq_length, seq_length):
        input_seq = np.array([token_ids[i : i + seq_length]])
        target_seq = np.array([token_ids[i + 1 : i + seq_length + 1]])

        logits = model.forward(input_seq)            # (1, seq_len, vocab)
        probs = softmax(logits)                       # (1, seq_len, vocab)

        # Gather probabilities of correct tokens
        for t in range(seq_length):
            p = probs[0, t, target_seq[0, t]]
            total_log_prob += np.log(max(p, 1e-9))
            total_tokens += 1

    avg_neg_log_prob = -total_log_prob / total_tokens
    return np.exp(avg_neg_log_prob)
```

### Interpreting Perplexity

| PPL Range | Interpretation |
|-----------|---------------|
| 1–10 | Excellent (likely overfitting on small data) |
| 10–50 | Good for a small model |
| 50–200 | Reasonable for a tiny model on diverse text |
| 200+ | Poor — model hasn't learned much |

## 2. BLEU Score

**BLEU (Bilingual Evaluation Understudy)** measures the quality of generated text by comparing it to reference text. It's widely used for machine translation and text generation.

BLEU computes the precision of n-gram overlaps between the generated text and reference:

$$\text{BLEU} = BP \cdot \exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

where:

- $p_n$ = modified n-gram precision (clipped by reference counts)
- $w_n = 1/N$ (uniform weights, typically $N=4$)
- $BP$ = brevity penalty to penalize short outputs:

$$BP = \begin{cases} 1 & \text{if } c > r \\ e^{1 - r/c} & \text{if } c \le r \end{cases}$$

### Implementation

```python
from collections import Counter

def compute_ngrams(tokens, n):
    """Extract n-grams from a token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def bleu_score(reference_tokens, generated_tokens, max_n=4):
    """
    Compute BLEU score between reference and generated token lists.
    """
    if len(generated_tokens) == 0:
        return 0.0

    # Brevity penalty
    c = len(generated_tokens)
    r = len(reference_tokens)
    bp = np.exp(1 - r / c) if c <= r else 1.0

    # N-gram precisions
    log_precisions = 0.0
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(compute_ngrams(reference_tokens, n))
        gen_ngrams = Counter(compute_ngrams(generated_tokens, n))

        # Clipped counts
        clipped = 0
        total = 0
        for ngram, count in gen_ngrams.items():
            clipped += min(count, ref_ngrams.get(ngram, 0))
            total += count

        if total == 0:
            return 0.0

        precision = clipped / total
        if precision == 0:
            return 0.0

        log_precisions += (1.0 / max_n) * np.log(precision)

    return bp * np.exp(log_precisions)
```

### BLEU Score Ranges

| BLEU | Quality |
|------|---------|
| 0.6+ | Very high overlap (near-exact match) |
| 0.3–0.6 | Good quality |
| 0.1–0.3 | Understandable but imperfect |
| < 0.1 | Poor |

!!! note
    BLEU was designed for machine translation. For Q&A evaluation, checking whether the expected answer appears in the model's output is often more practical. Perplexity remains the best single metric for overall model quality.

## 3. Text Generation & Question Answering

To evaluate qualitatively, we can generate text from the model. We use **autoregressive decoding**: starting from a prompt, we predict one token at a time and append it to the sequence.

For **question answering**, we format the prompt as `Q: <question> A:` and let the model complete the answer. We stop generation at the first newline (which separates Q&A pairs in our training data).

### Answer a Question

```python
def answer_question(model, tokenizer, question, max_tokens=60,
                    temperature=0.1, top_k=3):
    prompt = f"Q: {question} A:"
    full = generate(model, tokenizer, prompt,
                    max_tokens=max_tokens,
                    temperature=temperature, top_k=top_k)

    # Extract answer after "A:"
    if " A:" in full:
        answer = full.split(" A:", 1)[1].strip()
    else:
        answer = full[len(prompt):].strip()

    # Stop at newline (next Q&A pair boundary)
    if "\n" in answer:
        answer = answer.split("\n")[0].strip()

    return answer
```

### Sampling Strategies

#### Greedy Decoding

Always pick the highest-probability token:

```python
next_token = np.argmax(probs)
```

Simple but tends to produce repetitive text.

#### Temperature Sampling

Scale logits before softmax to control randomness:

$$P(x) = \text{softmax}(z / T)$$

- $T = 1.0$: normal sampling
- $T < 1.0$: sharper distribution (more confident)
- $T > 1.0$: flatter distribution (more creative)

#### Top-k Sampling

Only sample from the top $k$ most likely tokens:

```python
top_k_idx = np.argsort(probs)[-k:]
top_k_probs = probs[top_k_idx]
top_k_probs /= top_k_probs.sum()  # renormalize
next_token = np.random.choice(top_k_idx, p=top_k_probs)
```

#### Top-p (Nucleus) Sampling

Sample from the smallest set of tokens whose cumulative probability exceeds $p$:

```python
sorted_idx = np.argsort(probs)[::-1]
cumulative = np.cumsum(probs[sorted_idx])
cutoff = np.searchsorted(cumulative, p) + 1
top_p_idx = sorted_idx[:cutoff]
top_p_probs = probs[top_p_idx]
top_p_probs /= top_p_probs.sum()
next_token = np.random.choice(top_p_idx, p=top_p_probs)
```

### Full Generation Function

```python
def generate(model, tokenizer, prompt, max_tokens=100,
             temperature=1.0, top_k=50):
    token_ids = tokenizer.encode(prompt)
    # Remove EOS token from encode (we'll add it when done)
    if token_ids[-1] == 3:
        token_ids = token_ids[:-1]

    for _ in range(max_tokens):
        # Use last max_seq_len tokens as context
        context = token_ids[-model.max_seq_len:]
        x = np.array([context])

        logits = model.forward(x)
        next_logits = logits[0, -1, :]  # logits for last position

        # Temperature
        next_logits = next_logits / temperature

        # Softmax
        probs = softmax(next_logits)

        # Top-k
        if top_k > 0:
            top_k_idx = np.argsort(probs)[-top_k:]
            mask = np.zeros_like(probs)
            mask[top_k_idx] = probs[top_k_idx]
            probs = mask / mask.sum()

        # Sample
        next_token = np.random.choice(len(probs), p=probs)

        if next_token == 3:  # EOS
            break

        token_ids.append(next_token)

    return tokenizer.decode(token_ids)
```

## 4. Putting It All Together

```python
import numpy as np
from model import TinyLLM
from tokenizer import BPETokenizer
from evaluate import perplexity, bleu_score, generate, answer_question

# Load trained model and tokenizer
model = TinyLLM(
    vocab_size=332, d_model=64, n_heads=4,
    n_layers=2, d_ff=256, max_seq_len=64,
)
tokenizer = BPETokenizer(vocab_size=332)

# --- Perplexity ---
test_ids = tokenizer.encode("Q: What sound does a cat make? A: A cat makes a meow sound.")
ppl = perplexity(model, test_ids, seq_length=64)
print(f"Perplexity: {ppl:.2f}")

# --- Question Answering ---
questions = [
    "What sound does a cat make?",
    "What is 2 plus 3?",
    "What is the opposite of hot?",
    "What day comes after Monday?",
]
for q in questions:
    ans = answer_question(model, tokenizer, q)
    print(f"Q: {q}")
    print(f"A: {ans}")
```

## 5. What to Expect from Our Tiny Model

Our ~121K parameter model trained on a Q&A corpus will:

- **Perplexity**: Around 4 after 120 epochs of training
- **Q&A accuracy**: 5/5 on trained questions; may struggle with unseen questions
- **Generation**: Produce well-formed Q&A answers that follow the training patterns

This is expected! A tiny model can memorize a small Q&A corpus very well but won't generalize to arbitrary questions. Real LLMs use billions of parameters and terabytes of data. The purpose of this tutorial is to understand the **mechanics**, not to build a production model.

## Summary

In this chapter we implemented:

| Tool | Purpose |
|------|---------|
| `perplexity()` | Measures how well the model predicts test data |
| `bleu_score()` | Compares generated vs reference text (n-gram overlap) |
| `generate()` | Autoregressive text generation with temperature and top-k |
| `answer_question()` | Q&A wrapper: formats prompt and extracts answer |

## What's Next?

Congratulations — you've built an LLM from scratch! Here are some directions to explore:

- **Scale up**: Increase model size, data, and training time
- **Add dropout**: Regularize to prevent overfitting
- **Implement KV-caching**: Speed up generation by caching key/value states
- **Try different architectures**: Add rotary positional embeddings (RoPE), grouped query attention, etc.
- **Use a real dataset**: Train on books, Wikipedia, or code
- **Port more to C++**: Move the entire forward pass to C++ for speed

Happy building! 🚀
