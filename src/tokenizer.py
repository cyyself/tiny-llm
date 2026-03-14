"""
Byte-Pair Encoding (BPE) Tokenizer — built from scratch.

This module implements a simple BPE tokenizer similar to the one used in GPT-2.
It learns subword tokens by iteratively merging the most frequent adjacent pairs.
"""

import json
import os


class BPETokenizer:
    """Byte-Pair Encoding tokenizer built from scratch."""

    SPECIAL_TOKENS = {0: '<PAD>', 1: '<UNK>', 2: '<BOS>', 3: '<EOS>'}

    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.merges = {}           # (token_a, token_b) -> merged_token
        self.merge_list = []       # ordered list of merges for encoding
        self.vocab = {}            # token_id -> token_string
        self.inverse_vocab = {}    # token_string -> token_id

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @staticmethod
    def _get_pair_counts(token_sequences):
        """Count frequency of all adjacent token pairs."""
        counts = {}
        for seq in token_sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _merge_pair(token_sequences, pair, new_token):
        """Merge all occurrences of *pair* in every sequence."""
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
        """Train the BPE tokenizer on *text*."""
        # Pre-tokenize: split on spaces, keep space as prefix
        words = text.split(' ')
        token_sequences = []
        for word in words:
            if word:
                token_sequences.append(list(' ' + word))

        # Gather all unique characters
        all_chars = set()
        for seq in token_sequences:
            all_chars.update(seq)

        # Initialize vocabulary with special tokens
        self.vocab = dict(self.SPECIAL_TOKENS)
        self.inverse_vocab = {v: k for k, v in self.SPECIAL_TOKENS.items()}

        next_id = len(self.SPECIAL_TOKENS)
        for char in sorted(all_chars):
            self.vocab[next_id] = char
            self.inverse_vocab[char] = next_id
            next_id += 1

        # Iteratively merge the most frequent pair
        num_merges = self.vocab_size - next_id
        self.merges = {}
        self.merge_list = []

        for i in range(num_merges):
            pair_counts = self._get_pair_counts(token_sequences)
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            new_token = best_pair[0] + best_pair[1]

            self.merges[best_pair] = new_token
            self.merge_list.append(best_pair)

            self.vocab[next_id] = new_token
            self.inverse_vocab[new_token] = next_id
            next_id += 1

            token_sequences = self._merge_pair(
                token_sequences, best_pair, new_token
            )

            if (i + 1) % 500 == 0:
                print(
                    f"  Merge {i + 1}/{num_merges}: "
                    f"'{best_pair[0]}' + '{best_pair[1]}' -> '{new_token}'"
                )

        print(f"Tokenizer trained: {len(self.vocab)} tokens in vocabulary")

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, text):
        """Encode *text* into a list of token IDs."""
        words = text.split(' ')
        all_ids = [2]  # <BOS>

        for word in words:
            if not word:
                continue
            tokens = list(' ' + word)

            # Apply merges in training order
            for pair in self.merge_list:
                merged = self.merges[pair]
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if (i < len(tokens) - 1
                            and (tokens[i], tokens[i + 1]) == pair):
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

            for token in tokens:
                all_ids.append(self.inverse_vocab.get(token, 1))  # 1 = <UNK>

        all_ids.append(3)  # <EOS>
        return all_ids

    def decode(self, ids):
        """Decode a list of token IDs back to text."""
        tokens = []
        for token_id in ids:
            if token_id in (0, 2, 3):  # skip PAD / BOS / EOS
                continue
            tokens.append(self.vocab.get(token_id, '<UNK>'))
        text = ''.join(tokens)
        if text.startswith(' '):
            text = text[1:]
        return text

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path):
        """Save tokenizer to a JSON file."""
        data = {
            'vocab_size': self.vocab_size,
            'vocab': {str(k): v for k, v in self.vocab.items()},
            'merge_list': [list(p) for p in self.merge_list],
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        """Load tokenizer from a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab_size = data['vocab_size']
        self.vocab = {int(k): v for k, v in data['vocab'].items()}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merge_list = [tuple(p) for p in data['merge_list']]
        self.merges = {pair: pair[0] + pair[1] for pair in self.merge_list}


# ======================================================================
# Quick demo
# ======================================================================
if __name__ == '__main__':
    sample_text = (
        "The cat sat on the mat. The dog sat on the log. "
        "A cat and a dog are friends. The cat likes the dog."
    )

    tok = BPETokenizer(vocab_size=100)
    tok.train(sample_text)

    encoded = tok.encode("The cat sat on the mat")
    print(f"Encoded: {encoded}")

    decoded = tok.decode(encoded)
    print(f"Decoded: {decoded}")
