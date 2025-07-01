#!/usr/bin/env python3
"""Learning Rate Sweep for D2D DNN

This standalone script trains each predefined hidden-layer configuration
using a sweep of learning rates and plots training-loss curves (no
validation) to help choose the best LR.

Usage (example):
    python learning_rate_sweep.py --out_dir results/lr_sweep

It will create one PNG/EPS per hidden configuration with all loss curves
and annotate the best LR (lowest final loss).
"""

import os
import argparse
import math
from typing import List, Sequence

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# Import helper classes/functions directly from the main training script
from dnn_d2d_pytorch import (
    D2DNet,
    generate_environments,
    select_optimal_loss_function,
    compute_sinr,  # not used but imported to satisfy select_optimal_loss_function
)

# --------------------------- Default Sweeps ---------------------------
HIDDEN_SWEEP: List[Sequence[int]] = [
    [20],
    [100],
    [1000],
    [200, 200],
    [600, 600, 600],
    [1000, 1000, 1000],
    [500, 500, 500, 500],
]

LR_SWEEP: List[float] = [0.1, 0.03, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]

NUM_SAMPLES = 1000  # training samples
EPOCHS = 500
BATCH_SIZE = 32
NUM_PAIRS = 6
NUM_FREQUENCIES = 1

DEFAULT_DEVICE = (
    'cuda' if torch.cuda.is_available() else (
        'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu')
)

def train_one_lr(hidden_sizes: Sequence[int], lr: float, device: str, out_dir: str, epochs: int) -> List[float]:
    """Train a model for NUM_SAMPLES/EPOCHS and return list of training losses per epoch."""
    # Generate / load dataset once per LR for speed (no augment)
    gains = generate_environments(NUM_SAMPLES, NUM_PAIRS, seed=42)
    gains_flat = gains.reshape(NUM_SAMPLES, -1).astype(np.float32)
    # Simple per-sample normalization (mean/std) as in standard training
    mean = gains_flat.mean(axis=1, keepdims=True)
    std = gains_flat.std(axis=1, keepdims=True) + 1e-8
    gains_norm = (gains_flat - mean) / std

    ds = TensorDataset(torch.from_numpy(gains_norm))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = D2DNet(NUM_PAIRS, NUM_FREQUENCIES, hidden_sizes).to(device)
    loss_fn = select_optimal_loss_function(NUM_FREQUENCIES)
    # Noise power same as in main script (simplified constant)
    noise_power = 1e-13

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_curve: List[float] = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            p_lin, fa_probs = model(batch)
            # channel gains need original shape (B, P, P)
            gains_b = torch.from_numpy(gains[:batch.size(0)]).to(device)
            loss = loss_fn(p_lin, fa_probs, gains_b, noise_power)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= NUM_SAMPLES
        loss_curve.append(epoch_loss)
    # Save numpy for reproducibility
    np.save(os.path.join(out_dir, f"loss_lr_{lr}.npy"), np.array(loss_curve))
    return loss_curve


def plot_curves(hidden_str: str, curves: dict, out_dir: str):
    """Plot all LR curves; highlight best (lowest final loss)."""
    plt.figure(figsize=(10, 6))
    best_lr = min(curves, key=lambda k: curves[k][-1])
    for lr_val, lc in curves.items():
        label = f"LR={lr_val}"
        alpha = 1.0 if lr_val == best_lr else 0.4
        lw = 2 if lr_val == best_lr else 1
        plt.plot(lc, label=label, alpha=alpha, linewidth=lw)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"Learning Curves — Hidden {hidden_str} (best LR={best_lr})")
    plt.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=300)
    plt.savefig(os.path.join(out_dir, "learning_curves.eps"), format="eps")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Learning-Rate sweep for D2D DNN training loss")
    parser.add_argument("--out_dir", type=str, default="results/lr_sweep", help="Output directory")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="cpu|cuda|mps")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    device = args.device
    epochs_run = args.epochs

    for hidden in HIDDEN_SWEEP:
        hidden_str = "_".join(map(str, hidden))
        print(f"▶ Hidden {hidden_str}")
        hidden_out = os.path.join(args.out_dir, f"hidden_{hidden_str}")
        curves = {}
        for lr in LR_SWEEP:
            print(f"   • LR {lr}")
            lr_out = os.path.join(hidden_out, f"lr_{lr}")
            os.makedirs(lr_out, exist_ok=True)
            curves[lr] = train_one_lr(hidden, lr, device, lr_out, epochs_run)
        plot_curves(hidden_str, curves, hidden_out)
        print(f"   → Plots saved to {hidden_out}\n")


if __name__ == "__main__":
    main() 