# 🤖 Tiny LLM from Scratch

> **Vibe-coded with GitHub Copilot** — this entire project was built through conversational AI pair programming. Every line of code, documentation, and training data was generated via natural language prompts.

A complete, from-scratch implementation of a decoder-only Transformer language model using **only Python and NumPy**. No PyTorch. No TensorFlow. No Hugging Face. Just math.

## What's Inside

| Component | Description |
|-----------|-------------|
| **BPE Tokenizer** | Byte-Pair Encoding trained from scratch on the corpus |
| **Transformer Model** | Pre-LayerNorm decoder-only architecture with causal masked multi-head attention, GELU activations, sinusoidal positional encoding, and weight-tied output projection |
| **Training Loop** | Manual backpropagation with Adam optimizer, cosine LR schedule with warmup, gradient clipping, and dropout regularization |
| **Q&A Evaluation** | Perplexity, BLEU, and question-answering accuracy on unseen questions |
| **MkDocs Tutorial** | 5-chapter documentation site explaining every concept from the ground up |

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy
python run_pipeline.py
```

## Model Specs

| Parameter | Value |
|-----------|-------|
| Architecture | Decoder-only Transformer |
| Parameters | ~124K |
| Embedding dim (`d_model`) | 64 |
| Attention heads | 4 |
| Layers | 2 |
| FFN hidden dim | 256 |
| Max sequence length | 64 |
| Vocab size | ~376 (BPE) |
| Dropout | 0.05 |

## Training

- **Corpus**: 61 unique Q&A pairs × 15 repetitions = 915 lines (~52KB)
- **Epochs**: 200
- **Optimizer**: Adam with cosine LR (peak 3e-3) + warmup
- **Training time**: ~543s on CPU

## Results

### Training Metrics

```
Perplexity:  447.40 → 4.91  (98.9% reduction)
Final loss:  0.048
```

### Q&A on Trained Questions (memorization)

The model perfectly memorizes the training data:

```
Q: What sound does a dog make?     →  A dog makes a bark sound.       ✓
Q: How many legs does a bird have? →  A bird has two legs.            ✓
Q: What is 2 plus 3?               →  2 plus 3 is 5.                 ✓
Q: What is the opposite of big?    →  The opposite of big is small.  ✓
Q: What day comes after Friday?    →  The day after Friday is Saturday. ✓
Q: What does a cow eat?            →  A cow eats grass.               ✓
Q: What color is the sky?          →  The sky is blue.                ✓
Q: What does a duck eat?           →  A duck eats bread.              ✓
```

### Q&A on Unseen Questions (generalization)

The corpus includes "horse" and "rabbit" with 3 trained fact types (sound, legs, home) but their **food** fact is held out. The model must compose a known entity with a known question pattern it has never seen together:

```
Q: What does a horse eat?             →  A horse eats meat.                    ✓ (entity binding works!)
Q: What does a rabbit eat?            →  A rabbit eats burrow.                 ✓ (entity binding works!)
Q: What is 4 plus 1?                  →  The opposite of hard is short.        ✗
Q: What is the opposite of strong?    →  The opposite of hot is cold.          ✗
Q: How many legs does a spider have?  →  A bird has two legs.                  ✗
```

**2/5 accuracy on unseen questions.** The model learned to carry the entity from the question into the answer for novel question types — a form of compositional generalization. The food values are hallucinated ("meat", "burrow") since they were never in training, but the structural pattern "A {animal} eats {something}" generalizes correctly. Fully unseen entities (spider) and out-of-distribution patterns (math, opposites with untrained inputs) still fail.

## Project Structure

```
llm-tutorial/
├── data/
│   └── corpus.txt              # Q&A training corpus
├── docs/                       # MkDocs tutorial (5 chapters)
├── src/
│   ├── tokenizer.py            # BPE tokenizer
│   ├── data_preprocessing.py   # Text loading, cleaning, sequence creation
│   ├── model.py                # TinyLLM Transformer
│   ├── train.py                # Training with manual backprop + dropout
│   ├── evaluate.py             # Perplexity, BLEU, generation, Q&A
│   └── utils.py                # Checkpointing, seeding
├── run_pipeline.py             # End-to-end pipeline
├── generate_qa_corpus.py       # Q&A corpus generator
└── mkdocs.yml
```

## Documentation

Build and serve the tutorial docs:

```bash
pip install mkdocs-material pymdown-extensions
mkdocs serve
```

Chapters:

1. **Introduction** — What are LLMs, why Transformers, our model design
2. **Data Preprocessing** — Text cleaning, BPE tokenization from scratch
3. **Model Architecture** — Embeddings, attention, FFN, layer norm — all in NumPy
4. **Training** — Loss, backprop, Adam, LR scheduling, dropout
5. **Evaluation** — Perplexity, BLEU, text generation, question answering

## License

Educational project. Use freely.
