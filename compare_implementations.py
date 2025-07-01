#!/usr/bin/env python3
"""
Comprehensive comparison between our implementation and reference PyTorch approach.
This script implements key improvements from the reference code.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import os

from src.config import SimulationConfig
from src.simulator.scenario import Scenario
from src.algorithms.fs_bruteforce import fs_bruteforce
from src.simulator.metrics import sinr_linear, sum_rate_bps
from src.simulator.environment import db_to_linear

class ReferenceD2DNet(nn.Module):
    """Reference implementation following the PyTorch code structure"""
    def __init__(self, num_pairs: int, num_freq: int, hidden_sizes=[200, 200], seed: int = None):
        super(ReferenceD2DNet, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        
        self.num_pairs = num_pairs
        self.num_freq = num_freq
        
        # Build feature layers
        layers = []
        in_size = num_pairs * num_pairs
        prev = in_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.features = nn.Sequential(*layers)
        
        # Output heads
        self.power_head = nn.Linear(prev, num_pairs)
        if num_freq > 1:
            self.freq_head = nn.Linear(prev, num_pairs * num_freq)
        else:
            self.freq_head = None
    
    def forward(self, x):
        f = self.features(x)
        p_logits = self.power_head(f)
        if self.num_freq > 1:
            f_logits = self.freq_head(f)
            return p_logits, f_logits
        else:
            return p_logits, None

def compute_sinr_reference(p, fa, gains_db, noise_power=1e-13, eps=1e-12):
    """Reference SINR calculation following the PyTorch implementation"""
    # p: (batch, num_pairs) - power in Watts
    # fa: (batch, num_pairs, num_freq) - one-hot or softmax
    # gains_db: (batch, num_pairs, num_pairs) - channel gains in dB
    
    gains_lin = torch.pow(10.0, gains_db / 10.0)
    batch, n_pairs, n_freq = fa.shape
    
    # Signal: each RX gets its own TX's power on its assigned freq
    signal = p.unsqueeze(2) * fa * gains_lin[:, torch.arange(n_pairs), torch.arange(n_pairs)].unsqueeze(2)
    
    # Interference: sum over all other TXs on same freq
    interference = torch.ones_like(signal) * noise_power
    for j in range(n_pairs):
        pj = p[:, j].unsqueeze(1)  # (batch,1)
        fa_j = fa[:, j, :].unsqueeze(1)  # (batch,1,f)
        gains_j = gains_lin[:, :, j]  # (batch, n_pairs)
        interference += pj.unsqueeze(2) * fa_j * gains_j.unsqueeze(2)
    
    # subtract self-interference
    for i in range(n_pairs):
        interference[:, i, :] -= p[:, i].unsqueeze(1) * fa[:, i, :] * gains_lin[:, i, i].unsqueeze(1)
    
    sinr = signal / (interference + eps)
    return sinr

def train_reference_model(cfg: SimulationConfig, scenario, epochs=500, lr=1e-4, device='cpu'):
    """Train using reference approach"""
    n_pairs = cfg.n_pairs
    n_fa = cfg.n_fa
    
    # Reference power range (linear Watts) - match our dBm range
    min_power_dbm = cfg.tx_power_min_dbm  # -50 dBm
    max_power_dbm = cfg.tx_power_max_dbm  # 30 dBm
    min_power = 10 ** ((min_power_dbm - 30) / 10)  # Convert dBm to W
    max_power = 10 ** ((max_power_dbm - 30) / 10)  # Convert dBm to W
    noise_power_dbm = cfg.noise_power_dbm  # Use our noise power
    noise_power = 10 ** ((noise_power_dbm - 30) / 10)  # Convert dBm to W
    
    # Model setup
    model = ReferenceD2DNet(n_pairs, n_fa, hidden_sizes=[200, 200])
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Data preparation - reference normalization approach
    sample = scenario.channel_gains_db()  # Shape: (n_pairs, n_pairs)
    sample_flat = sample.flatten().reshape(1, -1)  # Shape: (1, n_pairs*n_pairs)
    
    # Per-sample normalization (like reference single-sample mode)
    mean = sample_flat.mean(axis=1, keepdims=True)
    std = sample_flat.std(axis=1, keepdims=True) + 1e-8
    sample_norm = (sample_flat - mean) / std
    
    print(f"Reference normalization - Mean: {mean.flatten()[0]:.2f} dB, Std: {std.flatten()[0]:.2f} dB")
    
    # Convert to tensors
    x_norm = torch.tensor(sample_norm, dtype=torch.float32, device=device)
    gains_db_original = torch.tensor(sample.reshape(1, n_pairs, n_pairs), dtype=torch.float32, device=device)
    
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        p_logits, f_logits = model(x_norm)
        
        # Convert to power (linear scale)
        p = torch.sigmoid(p_logits) * (max_power - min_power) + min_power
        
        # Handle frequency allocation
        if n_fa > 1:
            fa_probs = F.softmax(f_logits.view(-1, n_pairs, n_fa), dim=2)
        else:
            fa_probs = torch.ones((1, n_pairs, 1), device=device)
        
        # CRITICAL: Use original unnormalized gains for SINR calculation
        sinr = compute_sinr_reference(p, fa_probs, gains_db_original, noise_power)
        sum_rate = torch.log2(1.0 + sinr).sum(dim=(1,2)) * cfg.bandwidth_hz  # Add bandwidth scaling
        loss = -sum_rate.mean()
        
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}: Sum-rate = {-loss.item():.2e}")
    
    return model, losses, mean, std

def evaluate_reference_model(model, scenario, cfg, mean, std, device='cpu'):
    """Evaluate reference model"""
    n_pairs = cfg.n_pairs
    n_fa = cfg.n_fa
    min_power_dbm = cfg.tx_power_min_dbm
    max_power_dbm = cfg.tx_power_max_dbm
    min_power = 10 ** ((min_power_dbm - 30) / 10)
    max_power = 10 ** ((max_power_dbm - 30) / 10)
    noise_power_dbm = cfg.noise_power_dbm
    noise_power = 10 ** ((noise_power_dbm - 30) / 10)
    
    model.eval()
    
    # Prepare input (same normalization as training)
    sample = scenario.channel_gains_db()
    sample_flat = sample.flatten().reshape(1, -1)
    sample_norm = (sample_flat - mean) / std
    x_norm = torch.tensor(sample_norm, dtype=torch.float32, device=device)
    gains_db_original = torch.tensor(sample.reshape(1, n_pairs, n_pairs), dtype=torch.float32, device=device)
    
    with torch.no_grad():
        p_logits, f_logits = model(x_norm)
        p_continuous = torch.sigmoid(p_logits) * (max_power - min_power) + min_power
        
        # Normal DNN evaluation (continuous power, argmax FA)
        if n_fa > 1:
            fa_probs = F.softmax(f_logits.view(-1, n_pairs, n_fa), dim=2)
            fa_idx = fa_probs.argmax(dim=2)
            fa_onehot = torch.zeros_like(fa_probs)
            fa_onehot.scatter_(2, fa_idx.unsqueeze(2), 1.0)
        else:
            fa_onehot = torch.ones((1, n_pairs, 1), device=device)
        
        # Use original gains for SINR calculation
        sinr = compute_sinr_reference(p_continuous, fa_onehot, gains_db_original, noise_power)
        sum_rate_normal = torch.log2(1.0 + sinr).sum(dim=(1,2)).item() * cfg.bandwidth_hz
        
        # Convert power back to dBm for comparison with our implementation
        power_dbm = 10 * torch.log10(p_continuous * 1000)  # W to dBm
        fa_indices = fa_idx.cpu().numpy().flatten() if n_fa > 1 else np.zeros(n_pairs, dtype=int)
    
    return sum_rate_normal, power_dbm.cpu().numpy().flatten(), fa_indices

def compare_implementations():
    """Compare our implementation with reference approach"""
    
    # Setup
    cfg = SimulationConfig.from_yaml('cfgs/config_fa1.yaml')
    cfg.seed = 42
    device = 'cpu'
    
    print("=== IMPLEMENTATION COMPARISON ===")
    print(f"Configuration: {cfg.n_pairs} pairs, {cfg.n_fa} FA, seed={cfg.seed}")
    
    # Generate test scenario
    scenario = Scenario.random(cfg, restrict_rx_distance=True)
    print(f"TX-RX distances: {np.linalg.norm(np.array(scenario.tx_xy) - np.array(scenario.rx_xy), axis=1)}")
    
    # 1. Reference implementation
    print("\n--- Reference Implementation ---")
    ref_model, ref_losses, ref_mean, ref_std = train_reference_model(
        cfg, scenario, epochs=1000, lr=1e-4, device=device)
    ref_sum_rate, ref_powers, ref_fa = evaluate_reference_model(
        ref_model, scenario, cfg, ref_mean, ref_std, device=device)
    
    # 2. Our implementation
    print("\n--- Our Implementation ---")
    from src.algorithms.ml_dnn import train_model
    our_algo, our_losses, our_meta = train_model(
        cfg, hidden_size=[200, 200], epochs=1000, lr=1e-4, 
        save_path=None, soft_fa=False, device=device, model_seed=42)
    
    our_decision = our_algo.decide(scenario)
    our_powers = our_decision["tx_power_dbm"]
    our_fa = our_decision["fa_indices"]
    
    # Compute our sum-rate
    tx_power_lin = db_to_linear(our_powers) * 1e-3
    noise_power_lin = db_to_linear(cfg.noise_power_dbm) * 1e-3
    g = scenario.get_channel_gain_with_fa_penalty(our_fa)
    sinr = sinr_linear(tx_power_lin, our_fa, g, noise_power_lin)
    our_sum_rate = sum_rate_bps(sinr, cfg.bandwidth_hz)
    
    # 3. Full Search baseline
    print("\n--- Full Search Baseline ---")
    fs_algo = fs_bruteforce(cfg)
    fs_decision = fs_algo.decide(scenario)
    fs_powers = fs_decision["tx_power_dbm"]
    fs_fa = fs_decision["fa_indices"]
    
    tx_power_lin_fs = db_to_linear(fs_powers) * 1e-3
    g_fs = scenario.get_channel_gain_with_fa_penalty(fs_fa)
    sinr_fs = sinr_linear(tx_power_lin_fs, fs_fa, g_fs, noise_power_lin)
    fs_sum_rate = sum_rate_bps(sinr_fs, cfg.bandwidth_hz)
    
    # Results comparison
    print("\n=== RESULTS COMPARISON ===")
    print(f"Reference DNN: {ref_sum_rate:.2e} bit/s")
    print(f"Our DNN:       {our_sum_rate:.2e} bit/s") 
    print(f"Full Search:   {fs_sum_rate:.2e} bit/s")
    print(f"Reference/FS ratio: {ref_sum_rate/fs_sum_rate:.3f}")
    print(f"Our/FS ratio:       {our_sum_rate/fs_sum_rate:.3f}")
    print(f"Reference/Our ratio: {ref_sum_rate/our_sum_rate:.3f}")
    
    print("\n=== POWER ALLOCATION COMPARISON ===")
    print(f"Reference powers (dBm): {ref_powers}")
    print(f"Our powers (dBm):       {our_powers}")
    print(f"FS powers (dBm):        {fs_powers}")
    
    print(f"\nReference FA: {ref_fa}")
    print(f"Our FA:       {our_fa}")
    print(f"FS FA:        {fs_fa}")
    
    # Plot learning curves
    fig_dir = "figs"
    os.makedirs(fig_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(ref_losses, label='Reference Implementation', color='blue')
    plt.plot(our_losses, label='Our Implementation', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Sum Rate Loss')
    plt.title('Learning Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    methods = ['Reference\nDNN', 'Our\nDNN', 'Full\nSearch']
    sum_rates = [ref_sum_rate, our_sum_rate, fs_sum_rate]
    colors = ['blue', 'red', 'green']
    
    bars = plt.bar(methods, sum_rates, color=colors, alpha=0.7)
    plt.ylabel('Sum Rate (bit/s)')
    plt.title('Performance Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add ratio annotations
    for i, (bar, rate) in enumerate(zip(bars, sum_rates)):
        ratio = rate / fs_sum_rate
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sum_rates)*0.01,
                f'{ratio:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'implementation_comparison.png'), dpi=300)
    plt.close()
    
    print(f"\nComparison plots saved to {fig_dir}/implementation_comparison.png")
    
    return {
        'reference': {'sum_rate': ref_sum_rate, 'powers': ref_powers, 'fa': ref_fa},
        'ours': {'sum_rate': our_sum_rate, 'powers': our_powers, 'fa': our_fa},
        'fs': {'sum_rate': fs_sum_rate, 'powers': fs_powers, 'fa': fs_fa}
    }

if __name__ == "__main__":
    results = compare_implementations() 