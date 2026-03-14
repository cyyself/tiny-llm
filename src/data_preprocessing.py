"""
Data preprocessing utilities for the tiny LLM tutorial.

Provides functions for loading text, cleaning, creating training sequences,
and batching.
"""

import re
import unicodedata

import numpy as np


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def load_text_data(file_paths):
    """Load and concatenate text from multiple files."""
    texts = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return '\n'.join(texts)


# ------------------------------------------------------------------
# Cleaning
# ------------------------------------------------------------------

def clean_text(text):
    """Basic text cleaning for LLM training data.

    Steps:
      1. Normalize Unicode (NFKC)
      2. Strip HTML tags
      3. Collapse whitespace
    """
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\S\n]+', ' ', text)   # collapse spaces/tabs, keep newlines
    text = re.sub(r'\n{2,}', '\n', text)     # collapse multiple newlines to one
    text = text.strip()
    return text


# ------------------------------------------------------------------
# Sequence creation
# ------------------------------------------------------------------

def create_training_sequences(token_ids, seq_length=128, stride=None):
    """Create input–target pairs for next-token prediction.

    Returns:
        inputs  — np.ndarray of shape (N, seq_length)
        targets — np.ndarray of shape (N, seq_length)

    For each window of *seq_length* tokens, the target is shifted by one
    position into the future.  *stride* controls the step between windows
    (defaults to *seq_length* for non-overlapping).
    """
    if stride is None:
        stride = seq_length
    sequences_input = []
    sequences_target = []

    for i in range(0, len(token_ids) - seq_length, stride):
        input_seq = token_ids[i: i + seq_length]
        target_seq = token_ids[i + 1: i + seq_length + 1]
        sequences_input.append(input_seq)
        sequences_target.append(target_seq)

    return np.array(sequences_input), np.array(sequences_target)


# ------------------------------------------------------------------
# Batching
# ------------------------------------------------------------------

def create_batches(inputs, targets, batch_size=32):
    """Yield shuffled mini-batches of (input, target) arrays."""
    n = len(inputs)
    indices = np.arange(n)
    np.random.shuffle(indices)

    for start in range(0, n, batch_size):
        batch_idx = indices[start: start + batch_size]
        yield inputs[batch_idx], targets[batch_idx]


# ======================================================================
# Quick demo
# ======================================================================
if __name__ == '__main__':
    sample = "Hello world! This is a <b>test</b> of   the cleaning   pipeline."
    print("Raw:    ", repr(sample))
    print("Cleaned:", repr(clean_text(sample)))

    # Fake token IDs for demonstration
    fake_ids = list(range(300))
    inp, tgt = create_training_sequences(fake_ids, seq_length=128)
    print(f"\nSequences: {inp.shape[0]} (seq_length=128 from 300 tokens)")
    print(f"First input[:5]:  {inp[0][:5]}")
    print(f"First target[:5]: {tgt[0][:5]}")
