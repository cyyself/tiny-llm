# Chapter 2: Data Collection and Preprocessing

Before we can train a language model, we need data — lots of it. In this chapter, we'll cover how to collect text data, clean it, and convert it into a format our model can understand through **tokenization**.

## Data Collection

For our tiny LLM, we'll use a small text corpus. In practice, LLMs are trained on massive datasets (hundreds of gigabytes to terabytes), but for learning purposes, we'll work with a manageable dataset.

### Sources of Text Data

Common sources for training language models include:

- **Books** (Project Gutenberg — public domain literature)
- **Wikipedia** articles
- **Web crawls** (Common Crawl)
- **Code** (GitHub repositories)
- **News articles**, scientific papers, etc.

For this tutorial, we'll create a simple data loader that reads plain text files. You can use any text data you like — a few megabytes of text is enough for our tiny model.

### Sample: Loading Text Data

```python
# src/data_preprocessing.py (excerpt)

def load_text_data(file_paths):
    """Load and concatenate text from multiple files."""
    texts = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return '\n'.join(texts)
```

## Text Cleaning

Raw text data is often messy. We need to clean it before tokenization:

### Common Cleaning Steps

1. **Normalize Unicode**: Convert fancy quotes, dashes, etc. to standard ASCII equivalents
2. **Remove or replace special characters**: HTML tags, URLs, email addresses
3. **Normalize whitespace**: Collapse multiple spaces/newlines
4. **Lowercase** (optional — depends on your design)

```python
import re
import unicodedata

def clean_text(text):
    """Basic text cleaning for LLM training data."""
    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize whitespace (collapse multiple spaces/newlines)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text
```

!!! note "Design Decision"
    We choose **not** to lowercase the text, because case carries meaning (e.g., "Apple" vs "apple"). Modern LLMs preserve case information.

## Tokenization

**Tokenization** is the process of converting raw text into a sequence of integers (token IDs) that the model can process.

### Why Not Just Use Characters or Words?

| Approach | Pros | Cons |
|----------|------|------|
| **Character-level** | Small vocabulary, handles any word | Very long sequences, hard to learn |
| **Word-level** | Semantically meaningful | Huge vocabulary, can't handle unknown words |
| **Subword (BPE)** | Balanced vocabulary, handles rare words | Slightly more complex |

We'll use **Byte-Pair Encoding (BPE)**, the same approach used by GPT-2 and many other LLMs.

### How BPE Works

BPE starts with individual characters and iteratively merges the most frequent pair of adjacent tokens:

1. Start with a vocabulary of all individual characters
2. Count all adjacent pairs in the training corpus
3. Merge the most frequent pair into a new token
4. Repeat steps 2–3 until the desired vocabulary size is reached

**Example:**

```
Corpus: "low lower lowest"

Step 0: Characters → ['l', 'o', 'w', ' ', 'e', 'r', 's', 't']
Step 1: Most frequent pair ('l', 'o') → merge into 'lo'
Step 2: Most frequent pair ('lo', 'w') → merge into 'low'
Step 3: Most frequent pair ('e', 'r') → merge into 'er'
...
```

### Implementing BPE from Scratch

Here's our complete BPE tokenizer implementation:

```python
# src/tokenizer.py

class BPETokenizer:
    """Byte-Pair Encoding tokenizer built from scratch."""
    
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.merges = {}       # (pair) -> merged token
        self.vocab = {}        # token_id -> token_string
        self.inverse_vocab = {} # token_string -> token_id
    
    def _get_pair_counts(self, token_sequences):
        """Count frequency of all adjacent token pairs."""
        counts = {}
        for seq in token_sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def _merge_pair(self, token_sequences, pair, new_token):
        """Merge all occurrences of a pair in the sequences."""
        new_sequences = []
        for seq in token_sequences:
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
                    new_seq.append(new_token)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_sequences.append(new_seq)
        return new_sequences
    
    def train(self, text):
        """Train the BPE tokenizer on the given text."""
        # Start with character-level tokens
        # Pre-tokenize by splitting on whitespace (keep spaces as prefix)
        words = text.split(' ')
        token_sequences = []
        for word in words:
            if word:
                # Add space prefix (like GPT-2)
                token_sequences.append(list(' ' + word))
        
        # Initialize vocabulary with all unique characters
        all_chars = set()
        for seq in token_sequences:
            all_chars.update(seq)
        
        # Reserve special tokens
        self.vocab = {0: '<PAD>', 1: '<UNK>', 2: '<BOS>', 3: '<EOS>'}
        self.inverse_vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        
        next_id = 4
        for char in sorted(all_chars):
            self.vocab[next_id] = char
            self.inverse_vocab[char] = next_id
            next_id += 1
        
        # Iteratively merge most frequent pairs
        num_merges = self.vocab_size - next_id
        for i in range(num_merges):
            pair_counts = self._get_pair_counts(token_sequences)
            if not pair_counts:
                break
            
            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            new_token = best_pair[0] + best_pair[1]
            
            # Record the merge
            self.merges[best_pair] = new_token
            
            # Add to vocabulary
            self.vocab[next_id] = new_token
            self.inverse_vocab[new_token] = next_id
            next_id += 1
            
            # Apply merge to all sequences
            token_sequences = self._merge_pair(
                token_sequences, best_pair, new_token
            )
            
            if (i + 1) % 500 == 0:
                print(f"  Merge {i + 1}/{num_merges}: "
                      f"'{best_pair[0]}' + '{best_pair[1]}' -> '{new_token}'")
        
        print(f"Tokenizer trained: {len(self.vocab)} tokens in vocabulary")
    
    def encode(self, text):
        """Encode text into token IDs."""
        # Split into words and apply BPE merges
        words = text.split(' ')
        all_ids = [2]  # Start with <BOS>
        
        for word in words:
            if not word:
                continue
            tokens = list(' ' + word)
            
            # Apply merges in order
            for pair, merged in self.merges.items():
                i = 0
                new_tokens = []
                while i < len(tokens):
                    if (i < len(tokens) - 1 and 
                            (tokens[i], tokens[i + 1]) == pair):
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            
            # Convert tokens to IDs
            for token in tokens:
                token_id = self.inverse_vocab.get(token, 1)  # 1 = <UNK>
                all_ids.append(token_id)
        
        all_ids.append(3)  # End with <EOS>
        return all_ids
    
    def decode(self, ids):
        """Decode token IDs back to text."""
        tokens = []
        for token_id in ids:
            if token_id in (0, 2, 3):  # Skip PAD, BOS, EOS
                continue
            token = self.vocab.get(token_id, '<UNK>')
            tokens.append(token)
        text = ''.join(tokens)
        # Remove leading space
        if text.startswith(' '):
            text = text[1:]
        return text
```

### Using the Tokenizer

```python
# Example usage
tokenizer = BPETokenizer(vocab_size=5000)

# Train on your corpus
with open('data/corpus.txt', 'r') as f:
    training_text = f.read()
tokenizer.train(training_text)

# Encode
token_ids = tokenizer.encode("The cat sat on the mat")
print(token_ids)  # e.g., [2, 456, 1023, 789, 234, 456, 567, 3]

# Decode
text = tokenizer.decode(token_ids)
print(text)  # "The cat sat on the mat"
```

## Creating Training Sequences

After tokenization, we need to create fixed-length sequences for training. The model is trained to predict the next token, so we create input-target pairs:

```python
import numpy as np

def create_training_sequences(token_ids, seq_length=128):
    """Create input-target pairs for next-token prediction.
    
    For each sequence of length seq_length:
      - input:  tokens[i : i + seq_length]
      - target: tokens[i + 1 : i + seq_length + 1]
    """
    sequences_input = []
    sequences_target = []
    
    for i in range(0, len(token_ids) - seq_length, seq_length):
        input_seq = token_ids[i : i + seq_length]
        target_seq = token_ids[i + 1 : i + seq_length + 1]
        sequences_input.append(input_seq)
        sequences_target.append(target_seq)
    
    return np.array(sequences_input), np.array(sequences_target)
```

For example, given token IDs `[10, 20, 30, 40, 50]` and `seq_length=3`:

```
Input:  [10, 20, 30]    Target: [20, 30, 40]
Input:  [40, 50, ...]   Target: [50, ..., ...]
```

## Batching

For efficient training, we group sequences into **batches**:

```python
def create_batches(inputs, targets, batch_size=32):
    """Yield batches of input-target pairs."""
    n = len(inputs)
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield inputs[batch_idx], targets[batch_idx]
```

## Putting It All Together

Here's the complete data preprocessing pipeline:

```python
# Full pipeline example
from tokenizer import BPETokenizer
from data_preprocessing import load_text_data, clean_text, \
    create_training_sequences, create_batches

# 1. Load data
raw_text = load_text_data(['data/book1.txt', 'data/book2.txt'])

# 2. Clean
clean = clean_text(raw_text)

# 3. Tokenize
tokenizer = BPETokenizer(vocab_size=5000)
tokenizer.train(clean)
token_ids = tokenizer.encode(clean)

print(f"Total tokens: {len(token_ids)}")
print(f"Vocabulary size: {len(tokenizer.vocab)}")

# 4. Create sequences
inputs, targets = create_training_sequences(token_ids, seq_length=128)
print(f"Training sequences: {inputs.shape[0]}")

# 5. Iterate batches
for batch_inputs, batch_targets in create_batches(inputs, targets, batch_size=32):
    print(f"Batch shape: {batch_inputs.shape}")
    break  # Just show the first batch
```

## Summary

In this chapter, we learned:

- How to collect and clean text data for LLM training
- **Byte-Pair Encoding (BPE)** tokenization — the same algorithm used by GPT-2
- How to create fixed-length training sequences with input-target pairs
- How to batch data for efficient training

All the source code is available in the `src/` folder:

- `src/tokenizer.py` — BPE tokenizer implementation
- `src/data_preprocessing.py` — Data loading, cleaning, and sequence creation

[Continue to Chapter 3: Model Architecture →](03_model_architecture.md)
