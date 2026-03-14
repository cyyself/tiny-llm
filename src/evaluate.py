"""
Evaluation utilities for the tiny LLM.

Implements:
  - Perplexity
  - BLEU score
  - Text generation (greedy, temperature, top-k, top-p)
"""

from collections import Counter

import numpy as np

from model import TinyLLM, softmax


# ======================================================================
# Perplexity
# ======================================================================

def perplexity(model, token_ids, seq_length=128):
    """
    Compute perplexity of *model* on a flat list of token IDs.

    PPL = exp( -1/N * sum log P(x_i | x_{<i}) )
    """
    total_log_prob = 0.0
    total_tokens = 0

    for i in range(0, len(token_ids) - seq_length, seq_length):
        input_seq = np.array([token_ids[i: i + seq_length]])
        target_seq = np.array([token_ids[i + 1: i + seq_length + 1]])

        logits = model.forward(input_seq)
        probs = softmax(logits)

        for t in range(seq_length):
            p = float(probs[0, t, target_seq[0, t]])
            total_log_prob += np.log(max(p, 1e-9))
            total_tokens += 1

    if total_tokens == 0:
        return float('inf')

    return float(np.exp(-total_log_prob / total_tokens))


# ======================================================================
# BLEU
# ======================================================================

def compute_ngrams(tokens, n):
    """Return list of n-gram tuples from *tokens*."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def bleu_score(reference_tokens, generated_tokens, max_n=4):
    """
    Corpus-level BLEU between a single reference and hypothesis.

    Args:
        reference_tokens:  list[int]
        generated_tokens:  list[int]
        max_n:             maximum n-gram order (default 4)

    Returns:
        BLEU score in [0, 1].
    """
    if len(generated_tokens) == 0:
        return 0.0

    c = len(generated_tokens)
    r = len(reference_tokens)
    bp = np.exp(1 - r / c) if c <= r else 1.0

    log_precisions = 0.0
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(compute_ngrams(reference_tokens, n))
        gen_ngrams = Counter(compute_ngrams(generated_tokens, n))

        clipped = 0
        total = 0
        for ngram, count in gen_ngrams.items():
            clipped += min(count, ref_ngrams.get(ngram, 0))
            total += count

        if total == 0 or clipped == 0:
            return 0.0

        precision = clipped / total
        log_precisions += (1.0 / max_n) * np.log(precision)

    return float(bp * np.exp(log_precisions))


# ======================================================================
# Text generation
# ======================================================================

def generate(model, tokenizer, prompt, max_tokens=100,
             temperature=1.0, top_k=50, top_p=1.0):
    """
    Autoregressive text generation.

    Args:
        model:       TinyLLM instance
        tokenizer:   BPETokenizer instance
        prompt:      string prompt
        max_tokens:  maximum tokens to generate
        temperature: sampling temperature (1.0 = normal)
        top_k:       if > 0, keep only top-k tokens
        top_p:       if < 1.0, nucleus sampling threshold

    Returns:
        Generated text as a string.
    """
    token_ids = tokenizer.encode(prompt)
    # Remove trailing EOS so we can continue generating
    if token_ids and token_ids[-1] == 3:
        token_ids = token_ids[:-1]

    for _ in range(max_tokens):
        context = token_ids[-model.max_seq_len:]
        x = np.array([context])

        logits = model.forward(x)
        next_logits = logits[0, -1, :].astype(np.float64)

        # Temperature scaling
        if temperature != 1.0:
            next_logits = next_logits / temperature

        probs = softmax(next_logits)

        # Top-k filtering
        if 0 < top_k < len(probs):
            top_k_idx = np.argsort(probs)[-top_k:]
            mask = np.zeros_like(probs)
            mask[top_k_idx] = probs[top_k_idx]
            probs = mask

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            cumulative = np.cumsum(probs[sorted_idx])
            cutoff = int(np.searchsorted(cumulative, top_p)) + 1
            keep = sorted_idx[:cutoff]
            mask = np.zeros_like(probs)
            mask[keep] = probs[keep]
            probs = mask

        # Re-normalize
        total = probs.sum()
        if total <= 0:
            break
        probs = probs / total

        next_token = int(np.random.choice(len(probs), p=probs))

        if next_token == 3:  # EOS
            break

        token_ids.append(next_token)

    return tokenizer.decode(token_ids)


# ======================================================================
# Question answering
# ======================================================================

def answer_question(model, tokenizer, question, max_tokens=60,
                    temperature=0.1, top_k=3):
    """
    Answer a question using the Q&A-trained model.

    Formats the input as "Q: {question} A:" and generates until a
    newline or EOS token is produced.

    Returns:
        The generated answer string (without the "Q: ... A:" prefix).
    """
    prompt = f"Q: {question} A:"
    full = generate(model, tokenizer, prompt, max_tokens=max_tokens,
                    temperature=temperature, top_k=top_k)

    # Extract only the answer part (after "A:")
    if " A:" in full:
        answer = full.split(" A:", 1)[1].strip()
    else:
        answer = full[len(prompt):].strip()

    # Stop at newline (next Q&A pair)
    if "\n" in answer:
        answer = answer.split("\n")[0].strip()

    return answer


# ======================================================================
# Quick demo — trains a small model then evaluates Q&A
# ======================================================================
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    from tokenizer import BPETokenizer
    from data_preprocessing import load_text_data, clean_text, create_training_sequences
    from train import train

    # ---- Load corpus ----
    corpus_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'corpus.txt')
    raw = load_text_data([corpus_path])
    text = clean_text(raw)

    # ---- Tokenize ----
    tok = BPETokenizer(vocab_size=400)
    tok.train(text)
    ids = tok.encode(text)
    print(f"Corpus: {len(text):,} chars -> {len(ids):,} tokens")

    # ---- Build model ----
    model = TinyLLM(
        vocab_size=len(tok.vocab), d_model=64, n_heads=4,
        n_layers=2, d_ff=256, max_seq_len=64,
        rng=np.random.default_rng(42),
    )
    print(f"Parameters: {model.count_parameters():,}")

    # ---- Train ----
    inputs, targets = create_training_sequences(ids, seq_length=64, stride=32)
    print(f"Training sequences: {inputs.shape[0]}")
    train(model, inputs, targets, epochs=60, batch_size=4,
          max_lr=3e-3, warmup_steps=20, log_every=50,
          dropout_rate=0.05)

    # ---- Perplexity ----
    ppl = perplexity(model, ids[:500], seq_length=64)
    print(f"\nPerplexity (trained): {ppl:.2f}")

    # ---- Question answering (unseen questions) ----
    print("\n--- Question Answering (unseen questions) ---")
    questions = [
        "What sound does a horse make?",
        "What is 5 plus 4?",
        "What is the opposite of strong?",
        "How many legs does a spider have?",
        "What color is the ocean?",
    ]
    for q in questions:
        ans = answer_question(model, tok, q)
        print(f"  Q: {q}")
        print(f"  A: {ans}")
        print()
        print(f"  \"{prompt}\" -> \"{out}\"")

