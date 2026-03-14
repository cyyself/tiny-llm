#!/usr/bin/env python3
"""
End-to-end pipeline: load data → tokenize → build model → train → evaluate → generate.

Usage:
    python run_pipeline.py
    python run_pipeline.py --corpus data/corpus.txt --epochs 5
"""

import argparse
import os
import sys
import time

import numpy as np

# Add src/ to path so imports work from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tokenizer import BPETokenizer
from data_preprocessing import load_text_data, clean_text, create_training_sequences
from model import TinyLLM
from train import train
from evaluate import perplexity, bleu_score, generate, answer_question
from utils import save_checkpoint, set_seed


def main():
    parser = argparse.ArgumentParser(description='Tiny LLM end-to-end pipeline')
    parser.add_argument('--corpus', default='data/corpus.txt',
                        help='Path to training text file')
    parser.add_argument('--vocab-size', type=int, default=400,
                        help='BPE vocabulary size')
    parser.add_argument('--d-model', type=int, default=64,
                        help='Embedding / hidden dimension')
    parser.add_argument('--n-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=2,
                        help='Number of transformer layers')
    parser.add_argument('--d-ff', type=int, default=256,
                        help='Feed-forward hidden dimension')
    parser.add_argument('--seq-length', type=int, default=64,
                        help='Training sequence length')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Mini-batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-3,
                        help='Peak learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--dropout', type=float, default=0.05,
                        help='Dropout rate during training')
    args = parser.parse_args()

    rng = set_seed(args.seed)
    separator = '=' * 60

    # ==================================================================
    # Step 1: Load & clean data
    # ==================================================================
    print(f'\n{separator}')
    print('STEP 1: Loading and cleaning data')
    print(separator)

    corpus_path = os.path.join(os.path.dirname(__file__), args.corpus)
    raw_text = load_text_data([corpus_path])
    cleaned_text = clean_text(raw_text)
    print(f'  Raw text length:     {len(raw_text):,} chars')
    print(f'  Cleaned text length: {len(cleaned_text):,} chars')
    print(f'  Preview: {cleaned_text[:120]}...')

    # ==================================================================
    # Step 2: Tokenize
    # ==================================================================
    print(f'\n{separator}')
    print('STEP 2: Training BPE tokenizer')
    print(separator)

    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    t0 = time.time()
    tokenizer.train(cleaned_text)
    print(f'  Tokenizer trained in {time.time() - t0:.2f}s')
    print(f'  Vocabulary size: {len(tokenizer.vocab)}')

    token_ids = tokenizer.encode(cleaned_text)
    print(f'  Total tokens in corpus: {len(token_ids):,}')

    # Quick round-trip check
    snippet = 'The sun rose over the mountains'
    encoded = tokenizer.encode(snippet)
    decoded = tokenizer.decode(encoded)
    print(f'  Round-trip check: "{snippet}"')
    print(f'    Encoded ({len(encoded)} tokens): {encoded[:12]}...')
    print(f'    Decoded: "{decoded}"')

    # ==================================================================
    # Step 3: Create training sequences
    # ==================================================================
    print(f'\n{separator}')
    print('STEP 3: Creating training sequences')
    print(separator)

    inputs, targets = create_training_sequences(token_ids, seq_length=args.seq_length,
                                                  stride=args.seq_length // 2)
    print(f'  Sequence length: {args.seq_length}')
    print(f'  Training sequences: {inputs.shape[0]}')
    print(f'  Input shape:  {inputs.shape}')
    print(f'  Target shape: {targets.shape}')

    # Split into train / test (90 / 10)
    n = len(inputs)
    split = int(n * 0.9)
    train_inputs, test_inputs = inputs[:split], inputs[split:]
    train_targets, test_targets = targets[:split], targets[split:]
    print(f'  Train sequences: {len(train_inputs)}')
    print(f'  Test sequences:  {len(test_inputs)}')

    # ==================================================================
    # Step 4: Build model
    # ==================================================================
    print(f'\n{separator}')
    print('STEP 4: Building model')
    print(separator)

    model = TinyLLM(
        vocab_size=len(tokenizer.vocab),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.seq_length,
        rng=np.random.default_rng(args.seed),
    )
    print(f'  Model: TinyLLM')
    print(f'    d_model={args.d_model}, n_heads={args.n_heads}, '
          f'n_layers={args.n_layers}, d_ff={args.d_ff}')
    print(f'    vocab_size={len(tokenizer.vocab)}, '
          f'max_seq_len={args.seq_length}')
    print(f'  Total parameters: {model.count_parameters():,}')

    # Quick forward-pass sanity check
    sample = train_inputs[:2]
    logits = model.forward(sample)
    print(f'  Forward pass check: input {sample.shape} → logits {logits.shape}')

    # ==================================================================
    # Step 5: Evaluate BEFORE training (baseline)
    # ==================================================================
    print(f'\n{separator}')
    print('STEP 5: Baseline evaluation (untrained model)')
    print(separator)

    test_token_ids = list(test_inputs.ravel())
    ppl_before = perplexity(model, test_token_ids, seq_length=args.seq_length)
    print(f'  Perplexity (before training): {ppl_before:.2f}')

    generated_before = answer_question(model, tokenizer, 'What sound does a cat make?')
    print(f'  Q&A (before training): "{generated_before}"')

    # ==================================================================
    # Step 6: Train
    # ==================================================================
    print(f'\n{separator}')
    print('STEP 6: Training')
    print(separator)

    t0 = time.time()
    train(model, train_inputs, train_targets,
          epochs=args.epochs,
          batch_size=args.batch_size,
          max_lr=args.lr,
          warmup_steps=20,
          log_every=5,
          dropout_rate=args.dropout)
    train_time = time.time() - t0
    print(f'  Training completed in {train_time:.1f}s')

    # ==================================================================
    # Step 7: Evaluate AFTER training
    # ==================================================================
    print(f'\n{separator}')
    print('STEP 7: Post-training evaluation')
    print(separator)

    ppl_after = perplexity(model, test_token_ids, seq_length=args.seq_length)
    print(f'  Perplexity (after training): {ppl_after:.2f}')
    print(f'  Improvement: {ppl_before:.2f} → {ppl_after:.2f} '
          f'({(1 - ppl_after / ppl_before) * 100:.1f}% reduction)')

    # Test Q&A on UNSEEN questions (not in training data)
    # - Held-out: entity is known but this specific question was omitted
    # - Fully unseen: entity or concept never appeared in training
    qa_test = [
        ('What does a horse eat?', 'horse eats'),
        ('What does a rabbit eat?', 'rabbit eats'),
        ('What is 4 plus 1?', '4 plus 1 is 5'),
        ('What is the opposite of strong?', 'weak'),
        ('How many legs does a spider have?', 'eight'),
    ]
    correct = 0
    print('\n  Q&A evaluation:')
    for question, expected in qa_test:
        answer = answer_question(model, tokenizer, question)
        match = expected.lower().strip('.') in answer.lower()
        correct += int(match)
        status = 'OK' if match else 'MISS'
        print(f'    [{status}] Q: {question}')
        print(f'         Expected: {expected}')
        print(f'         Got:      {answer}')
    print(f'\n  Q&A accuracy: {correct}/{len(qa_test)}')

    # ==================================================================
    # Step 8: Question answering samples
    # ==================================================================
    print(f'\n{separator}')
    print('STEP 8: Question answering samples')
    print(separator)

    questions = [
        'What does a horse eat?',
        'What does a rabbit eat?',
        'How many legs does a horse have?',
        'What sound does a rabbit make?',
        'What is 4 plus 1?',
        'What is the opposite of strong?',
        'What color is the ocean?',
        'What is 7 plus 3?',
    ]
    for q in questions:
        ans = answer_question(model, tokenizer, q)
        print(f'  Q: {q}')
        print(f'  A: {ans}')
        print()

    # ==================================================================
    # Step 9: Save checkpoint
    # ==================================================================
    print(f'{separator}')
    print('STEP 9: Saving checkpoint')
    print(separator)

    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'tiny_llm.npz')
    save_checkpoint(model, ckpt_path)
    tokenizer.save(os.path.join(ckpt_dir, 'tokenizer.json'))
    print(f'  Tokenizer saved to checkpoints/tokenizer.json')

    # ==================================================================
    # Summary
    # ==================================================================
    print(f'\n{separator}')
    print('PIPELINE COMPLETE — Summary')
    print(separator)
    print(f'  Corpus:       {args.corpus} ({len(cleaned_text):,} chars)')
    print(f'  Vocab size:   {len(tokenizer.vocab)}')
    print(f'  Tokens:       {len(token_ids):,}')
    print(f'  Sequences:    {n} (train {len(train_inputs)}, test {len(test_inputs)})')
    print(f'  Parameters:   {model.count_parameters():,}')
    print(f'  Epochs:       {args.epochs}')
    print(f'  Train time:   {train_time:.1f}s')
    print(f'  Perplexity:   {ppl_before:.2f} → {ppl_after:.2f}')
    print(f'  Checkpoint:   {ckpt_path}')
    print()


if __name__ == '__main__':
    main()
