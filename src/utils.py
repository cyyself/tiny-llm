"""
Utility functions for the tiny LLM tutorial.

Includes:
  - Model checkpoint save/load
  - Reproducibility helpers
"""

import os

import numpy as np


# ======================================================================
# Checkpoint save / load
# ======================================================================

def save_checkpoint(model, path):
    """Save all model parameters to a .npz file."""
    params = model.get_all_parameters()
    np.savez(path, *params)
    print(f"Checkpoint saved to {path} ({len(params)} arrays)")


def load_checkpoint(model, path):
    """Load model parameters from a .npz file (must match architecture)."""
    data = np.load(path)
    params = model.get_all_parameters()
    keys = sorted(data.files, key=lambda k: int(k.replace('arr_', '')))
    assert len(keys) == len(params), (
        f"Checkpoint has {len(keys)} arrays but model has {len(params)}"
    )
    for key, param in zip(keys, params):
        loaded = data[key]
        assert param.shape == loaded.shape, (
            f"Shape mismatch for {key}: {param.shape} vs {loaded.shape}"
        )
        param[:] = loaded
    print(f"Checkpoint loaded from {path}")


# ======================================================================
# Reproducibility
# ======================================================================

def set_seed(seed=42):
    """Set NumPy random seed for reproducibility."""
    np.random.seed(seed)
    return np.random.default_rng(seed)


# ======================================================================
# Demo
# ======================================================================
if __name__ == '__main__':
    rng = set_seed(0)
    print(f"Seed set, rng={rng}")
