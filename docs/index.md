# Building an LLM from Scratch

Welcome to this hands-on tutorial where you'll learn how to build a **tiny Large Language Model (LLM) from scratch**. This guide is designed for CS undergraduates who want to understand what's really happening inside models like GPT, from the ground up.

## What You'll Learn

By the end of this tutorial, you will:

- Understand the core concepts behind Large Language Models
- Know how to collect, clean, and tokenize text data
- Implement the Transformer architecture using only **Python and NumPy** — no PyTorch or TensorFlow
- Train a tiny language model on a small dataset
- Evaluate your model using standard metrics like perplexity and Q&A accuracy

## Prerequisites

- **Python 3.10+** with NumPy installed
- Basic knowledge of:
    - Linear algebra (matrix multiplication, dot products)
    - Calculus (derivatives, chain rule)
    - Probability and statistics
    - Python programming

## Project Structure

```
llm-tutorial/
├── data/                  # Training data
│   └── corpus.txt         # Q&A training corpus
├── docs/                  # Tutorial documentation (this site)
│   ├── index.md
│   ├── 01_introduction.md
│   ├── 02_data_preprocessing.md
│   ├── 03_model_architecture.md
│   ├── 04_training.md
│   └── 05_evaluation.md
├── src/                   # Source code
│   ├── tokenizer.py       # Byte-Pair Encoding tokenizer
│   ├── data_preprocessing.py
│   ├── model.py           # Transformer model in NumPy
│   ├── train.py           # Training loop with dropout
│   ├── evaluate.py        # Evaluation metrics
│   └── utils.py           # Utility functions
├── run_pipeline.py        # End-to-end pipeline script
├── generate_corpus.py     # Q&A corpus generator
├── mkdocs.yml
└── agent.md
```

## Quick Start

Run the entire pipeline (data → tokenize → train → evaluate → Q&A) in one command:

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy
python run_pipeline.py              # uses data/corpus.txt by default
python run_pipeline.py --epochs 10  # train longer for better results
```

## How to Use This Tutorial

Work through the chapters in order. Each chapter includes:

1. **Conceptual explanations** of the theory
2. **Code walkthroughs** with detailed comments
3. **Runnable source code** in the `src/` folder

Let's get started with [Chapter 1: Introduction to LLMs](01_introduction.md)!
