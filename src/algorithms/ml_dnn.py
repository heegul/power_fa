from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from pathlib import Path
from typing import Union, Optional, List
import argparse
import yaml
from torch.utils.data import DataLoader
from typing import TYPE_CHECKING

from ..config import SimulationConfig
from ..simulator.metrics import sinr_linear as sinr_linear_numpy, sum_rate_bps
from ..algorithms import AllocationAlgorithm, register_algorithm

if TYPE_CHECKING:
    from ..simulator.scenario import Scenario

# Helper: map output to valid power and FA

def dnn_output_to_decision(output, cfg: SimulationConfig, fa_gumbel_softmax=False, gumbel_temp=1.0):
    n_pairs = cfg.n_pairs
    n_fa = cfg.n_fa
    if n_fa == 1:
        # Only power outputs, FA is always 0
        power_norm = torch.sigmoid(output[:n_pairs])  # 0..1
        fa_indices = np.zeros(n_pairs, dtype=int)
    else:
        # First n_pairs: power (normalized 0-1), next n_pairs: FA logits
        power_norm = torch.sigmoid(output[:n_pairs])  # 0..1
        fa_logits = output[n_pairs:].reshape(n_pairs, n_fa)
        if fa_gumbel_softmax:
            fa_probs = torch.nn.functional.gumbel_softmax(fa_logits, tau=gumbel_temp, hard=True)
            fa_indices = torch.argmax(fa_probs, dim=1).detach().cpu().numpy()
        else:
            fa_probs = torch.softmax(fa_logits, dim=1)
            fa_indices = torch.argmax(fa_probs, dim=1).detach().cpu().numpy()
    # Convert normalized power to linear scale first, then to dBm
    power_min_lin = 10 ** ((cfg.tx_power_min_dbm - 30) / 10)  # Convert dBm to W
    power_max_lin = 10 ** ((cfg.tx_power_max_dbm - 30) / 10)  # Convert dBm to W
    power_lin = power_min_lin + power_norm.detach().cpu().numpy() * (power_max_lin - power_min_lin)
    tx_power_dbm = 10 * np.log10(power_lin) + 30  # Convert W to dBm
    return tx_power_dbm, fa_indices

def dnn_output_to_decision_torch(output, cfg: SimulationConfig, device=None, fa_gumbel_softmax=False, gumbel_temp=1.0, discrete_power=False, power_levels=None):
    n_pairs = cfg.n_pairs
    n_fa = cfg.n_fa
    device = device or (output.device if hasattr(output, 'device') else 'cpu')
    if discrete_power:
        if power_levels is None:
            power_levels = [1e-10, 0.25, 0.5, 1.0]
        num_levels = len(power_levels)
    else:
        num_levels = 1  # placeholder
    if n_fa == 1:
        if output.dim() == 1:
            if discrete_power:
                power_logits = output[: n_pairs * num_levels].view(n_pairs, num_levels)
                power_probs = torch.softmax(power_logits, dim=1)
                power_levels_tensor = torch.tensor(power_levels, dtype=power_logits.dtype, device=device)
                power_lin = (power_probs * power_levels_tensor).sum(dim=1)
            else:
                power_norm = torch.sigmoid(output[:n_pairs])
                power_min_lin = 10 ** ((cfg.tx_power_min_dbm - 30) / 10)
                power_max_lin = 10 ** ((cfg.tx_power_max_dbm - 30) / 10)
                power_lin = power_min_lin + power_norm * (power_max_lin - power_min_lin)
            fa_indices = torch.zeros(n_pairs, dtype=torch.long, device=device)
            fa_probs = torch.ones(n_pairs, 1, device=device)
        else:
            batch_size = output.shape[0]
            if discrete_power:
                power_logits = output[:, : n_pairs * num_levels].view(batch_size, n_pairs, num_levels)
                power_probs = torch.softmax(power_logits, dim=2)
                power_levels_tensor = torch.tensor(power_levels, dtype=power_logits.dtype, device=device)
                power_lin = (power_probs * power_levels_tensor.view(1,1,-1)).sum(dim=2)
            else:
                power_norm = torch.sigmoid(output[:, :n_pairs])
                power_min_lin = 10 ** ((cfg.tx_power_min_dbm - 30) / 10)
                power_max_lin = 10 ** ((cfg.tx_power_max_dbm - 30) / 10)
                power_lin = power_min_lin + power_norm * (power_max_lin - power_min_lin)
            fa_indices = torch.zeros(batch_size, n_pairs, dtype=torch.long, device=device)
            fa_probs = torch.ones(batch_size, n_pairs, 1, device=device)
    else:
        if output.dim() == 1:
            if discrete_power:
                power_logits = output[: n_pairs * num_levels].view(n_pairs, num_levels)
                power_probs = torch.softmax(power_logits, dim=1)
                power_levels_tensor = torch.tensor(power_levels, dtype=power_logits.dtype, device=device)
                power_lin = (power_probs * power_levels_tensor).sum(dim=1)
                start_idx = n_pairs * num_levels
            else:
                power_norm = torch.sigmoid(output[:n_pairs])
                power_min_lin = 10 ** ((cfg.tx_power_min_dbm - 30) / 10)
                power_max_lin = 10 ** ((cfg.tx_power_max_dbm - 30) / 10)
                power_lin = power_min_lin + power_norm * (power_max_lin - power_min_lin)
                start_idx = n_pairs
            fa_logits = output[start_idx:].reshape(n_pairs, n_fa)
            if fa_gumbel_softmax:
                fa_probs = torch.nn.functional.gumbel_softmax(fa_logits, tau=gumbel_temp, hard=True)
                fa_indices = torch.argmax(fa_probs, dim=1)
            else:
                fa_probs = torch.softmax(fa_logits, dim=1)
                fa_indices = torch.argmax(fa_probs, dim=1)
        else:
            if discrete_power:
                power_logits = output[:, : n_pairs * num_levels].view(-1, n_pairs, num_levels)
                power_probs = torch.softmax(power_logits, dim=2)
                power_levels_tensor = torch.tensor(power_levels, dtype=power_logits.dtype, device=device)
                power_lin = (power_probs * power_levels_tensor.view(1,1,-1)).sum(dim=2)
                start_idx = n_pairs * num_levels
            else:
                power_norm = torch.sigmoid(output[:, :n_pairs])
                power_min_lin = 10 ** ((cfg.tx_power_min_dbm - 30) / 10)
                power_max_lin = 10 ** ((cfg.tx_power_max_dbm - 30) / 10)
                power_lin = power_min_lin + power_norm * (power_max_lin - power_min_lin)
                start_idx = n_pairs
            fa_logits = output[:, start_idx:].reshape(-1, n_pairs, n_fa)
            if fa_gumbel_softmax:
                fa_probs = torch.nn.functional.gumbel_softmax(fa_logits, tau=gumbel_temp, hard=True)
                fa_indices = torch.argmax(fa_probs, dim=2)
            else:
                fa_probs = torch.softmax(fa_logits, dim=2)
                fa_indices = torch.argmax(fa_probs, dim=2)
    if not discrete_power:
        tx_power_dbm = 10 * torch.log10(power_lin) + 30
    else:
        # Add epsilon for numerical stability with discrete power levels near zero
        tx_power_dbm = 10 * torch.log10(power_lin + 1e-12) + 30
    return tx_power_dbm.to(device), fa_indices, fa_probs, power_lin.to(device)

def dnn_output_to_decision_torch_train(output, cfg: SimulationConfig, device=None, fa_gumbel_softmax=False, gumbel_temp=1.0, discrete_power=False, power_levels=None):
    """
    A simplified version of dnn_output_to_decision_torch for TRAINING ONLY.
    It returns only the differentiable tensors needed for loss calculation
    (power_lin, fa_probs), ensuring the computation graph is not broken.
    """
    n_pairs = cfg.n_pairs
    n_fa = cfg.n_fa
    device = device or (output.device if hasattr(output, 'device') else 'cpu')

    if discrete_power:
        if power_levels is None:
            power_levels = [1e-10, 0.25, 0.5, 1.0]
        num_levels = len(power_levels)
    else:
        num_levels = 1

    # This function now only handles the single-sample case (output.dim() == 1)
    # as that's where we are training.
    if discrete_power:
        power_logits = output[: n_pairs * num_levels].view(n_pairs, num_levels)
        power_probs = torch.softmax(power_logits, dim=1)
        power_levels_tensor = torch.tensor(power_levels, dtype=power_logits.dtype, device=device)
        power_lin = (power_probs * power_levels_tensor).sum(dim=1)
    else:
        power_norm = torch.sigmoid(output[:n_pairs])
        power_min_lin = 10 ** ((cfg.tx_power_min_dbm - 30) / 10)
        power_max_lin = 10 ** ((cfg.tx_power_max_dbm - 30) / 10)
        power_lin = power_min_lin + power_norm * (power_max_lin - power_min_lin)

    if n_fa == 1:
        fa_probs = torch.ones(n_pairs, 1, device=device)
    else:
        start_idx = n_pairs * num_levels if discrete_power else n_pairs
        fa_logits = output[start_idx:].reshape(n_pairs, n_fa)
        if fa_gumbel_softmax:
            fa_probs = torch.nn.functional.gumbel_softmax(fa_logits, tau=gumbel_temp, hard=True)
        else:
            fa_probs = torch.softmax(fa_logits, dim=1)
            
    return power_lin, fa_probs

class D2DNet(nn.Module):
    def __init__(self, n_pairs, n_fa, hidden_sizes, batch_norm=True, *, discrete_power=False, power_levels=None, cfg: SimulationConfig):
        super(D2DNet, self).__init__()
        self.n_pairs = n_pairs
        self.n_fa = n_fa
        self.discrete_power = discrete_power
        self.power_levels = power_levels or [1e-10, 0.25, 0.5, 1.0]
        self.num_power_levels = len(self.power_levels)

        self.min_power_watts = 10**((cfg.tx_power_min_dbm - 30) / 10)
        self.max_power_watts = 10**((cfg.tx_power_max_dbm - 30) / 10)

        layers = []
        input_size = n_pairs * n_pairs
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        self.hidden_layers = nn.ModuleList(layers)
        
        if discrete_power:
            self.power_head = nn.Linear(hidden_sizes[-1], n_pairs * self.num_power_levels)
        else:
            self.power_head = nn.Linear(hidden_sizes[-1], n_pairs)
        
        if n_fa > 1:
            self.fa_head = nn.Linear(hidden_sizes[-1], n_pairs * n_fa)
        else:
            self.fa_head = None

    def forward(self, x):
        h = x
        for layer in self.hidden_layers:
            # Special handling for batch_norm with batch size of 1
            if isinstance(layer, nn.BatchNorm1d) and h.shape[0] == 1 and self.training:
                continue
            h = layer(h)
        
        if self.discrete_power:
            power_logits = self.power_head(h).view(-1, self.n_pairs, self.num_power_levels)
            power_probs = F.softmax(power_logits, dim=2)
            power_levels_tensor = torch.tensor(self.power_levels, device=x.device, dtype=x.dtype)
            power_values = torch.sum(power_probs * power_levels_tensor.view(1, 1, -1), dim=2)
        else:
            power_logits = self.power_head(h)
            power_values = torch.sigmoid(power_logits) * (self.max_power_watts - self.min_power_watts) + self.min_power_watts
        
        if self.fa_head is not None:
            fa_logits = self.fa_head(h)
            fa_probs = F.softmax(fa_logits.view(-1, self.n_pairs, self.n_fa), dim=2)
        else:
            fa_probs = torch.ones((x.size(0), self.n_pairs, 1), device=x.device)
        
        return power_values, fa_probs

@register_algorithm
class ML_DNN(AllocationAlgorithm):
    """Fully connected DNN for power & FA allocation. BatchNorm optional. Supports discrete power grids."""
    def __init__(self, cfg: SimulationConfig, hidden_size=None, batch_norm=True, device=None, fa_gumbel_softmax=False, gumbel_temp=1.0, *, discrete_power=False, power_levels=None):
        super().__init__(cfg)
        self.n_pairs = cfg.n_pairs
        self.n_fa = cfg.n_fa
        # Default: two hidden layers of 200 if not specified
        if hidden_size is None:
            hidden_size = [200, 200]
        self.hidden_sizes = hidden_size
        self.batch_norm = batch_norm  # Always set this attribute
        self.device = device if device is not None else get_device()
        self.fa_gumbel_softmax = fa_gumbel_softmax
        self.gumbel_temp = gumbel_temp
        self.discrete_power = discrete_power
        self.power_levels = power_levels or [1e-10, 0.25, 0.5, 1.0]
        self.num_power_levels = len(self.power_levels)
        self.model = D2DNet(self.n_pairs, self.n_fa, self.hidden_sizes, batch_norm=batch_norm,
                               discrete_power=discrete_power, power_levels=self.power_levels, cfg=self.cfg).to(self.device)
        self.model.eval()
    def decide(self, scenario, /):
        # Input: channel gain matrix (dB), no normalization (BatchNorm in model)
        x_raw = torch.tensor(scenario.channel_gains_db(), dtype=torch.float32, device=self.device).flatten()
        x = x_raw.unsqueeze(0)
        with torch.no_grad():
            power_lin, fa_probs = self.model(x)

        # Convert linear power (Watts) to dBm
        tx_power_dbm = 10 * torch.log10(power_lin + 1e-12) + 30  # Add small epsilon to avoid log(0)

        # Hard assignment for frequency allocation: argmax over probabilities
        fa_indices = torch.argmax(fa_probs, dim=2)

        return {
            "tx_power_dbm": tx_power_dbm.cpu().numpy().flatten(),
            "fa_indices": fa_indices.cpu().numpy().flatten(),
        }
    def load_weights(self, path: Union[str, Path]):
        state, arch = load_checkpoint(path, self.device)
        if arch is not None:
            # Rebuild model with saved architecture
            self.n_pairs = arch['n_pairs']
            self.n_fa = arch['n_fa']
            self.hidden_sizes = arch['hidden_sizes']
            self.batch_norm = arch.get('batch_norm', True)
            self.discrete_power = arch.get('discrete_power', False)
            self.power_levels = arch.get('power_levels', [1e-10, 0.25, 0.5, 1.0])
            self.num_power_levels = len(self.power_levels)
            self.model = D2DNet(self.n_pairs, self.n_fa, self.hidden_sizes, batch_norm=self.batch_norm,
                                   discrete_power=self.discrete_power, power_levels=self.power_levels, cfg=self.cfg).to(self.device)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"[WARNING] load_state_dict loaded with missing keys: {missing} and unexpected keys: {unexpected}. Proceeding with partial load.")
        else:
            # Fallback: try to load as before, tolerating mismatched keys
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"[WARNING] load_state_dict (fallback) loaded with missing keys: {missing} and unexpected keys: {unexpected}.")
    def save_weights(self, path: Union[str, Path]):
        # Save model weights and architecture
        arch = {
            'n_pairs': self.n_pairs,
            'n_fa': self.n_fa,
            'hidden_sizes': self.hidden_sizes,
            'batch_norm': self.batch_norm,
            'discrete_power': self.discrete_power,
            'power_levels': self.power_levels,
        }
        save_checkpoint(self.model, path, arch)

# Custom loss function: negative sum-rate

def negative_sum_rate_loss(tx_power_dbm, fa_indices, scenario, cfg: SimulationConfig):
    n_pairs = cfg.n_pairs
    n_fa = cfg.n_fa
    tx_power_lin = 10 ** (tx_power_dbm / 10) * 1e-3  # dBm to W
    noise_power_lin = 10 ** (cfg.noise_power_dbm / 10) * 1e-3
    if n_fa == 1:
        fa_indices = np.zeros(n_pairs, dtype=int)
    g = scenario.get_channel_gain_with_fa_penalty(fa_indices)
    sinr = sinr_linear_numpy(
        tx_power_lin=tx_power_lin,
        fa_indices=fa_indices,
        channel_gain=g,
        noise_power_lin=noise_power_lin,
    )
    sum_rate = sum_rate_bps(sinr, cfg.bandwidth_hz)
    return -sum_rate  # negative for minimization

# Differentiable loss for training (all torch)
def negative_sum_rate_loss_torch(tx_power_dbm, fa_probs, scenario, cfg: SimulationConfig):
    n_pairs = cfg.n_pairs
    n_fa = cfg.n_fa
    device = tx_power_dbm.device if hasattr(tx_power_dbm, 'device') else 'cpu'
    tx_power_lin = 10 ** (tx_power_dbm / 10) * 1e-3  # dBm to W (torch)
    if n_fa == 1:
        fa_indices = torch.zeros(n_pairs, dtype=torch.long, device=device)
    else:
        fa_indices = torch.argmax(fa_probs, dim=1)
    fa_indices_np = fa_indices.detach().cpu().numpy()
    g_np = scenario.get_channel_gain_with_fa_penalty(fa_indices_np)
    g = torch.tensor(g_np, dtype=torch.float32, device=device)
    noise_power_lin = torch.tensor(10 ** (cfg.noise_power_dbm / 10) * 1e-3, dtype=torch.float32, device=device)
    sinr = []
    for i in range(n_pairs):
        same_fa = (fa_indices == fa_indices[i])
        same_fa_mask = same_fa if same_fa.dtype == torch.bool else same_fa.bool()
        interference = torch.sum(tx_power_lin[same_fa_mask] * g[same_fa_mask, i]) - (tx_power_lin[i] * g[i, i])
        sinr_i = (tx_power_lin[i] * g[i, i]) / (interference + noise_power_lin)
        sinr.append(sinr_i)
    sinr = torch.stack(sinr)
    sum_rate = cfg.bandwidth_hz * torch.sum(torch.log2(1.0 + sinr))
    return -sum_rate  # negative for minimization

def negative_sum_rate_loss_torch_soft(tx_power_dbm, fa_probs, scenario, cfg: SimulationConfig):
    # CORRECTED: Computes E[SINR] properly, not E[desired]/E[interference]
    n_pairs = cfg.n_pairs
    n_fa = cfg.n_fa
    tx_power_lin = 10 ** (tx_power_dbm / 10) * 1e-3  # dBm to W (torch)
    # Get channel gain matrix (fixed for scenario)
    g_np = scenario.channel_gain_lin
    g = torch.tensor(g_np, dtype=torch.float32, device=tx_power_dbm.device if hasattr(tx_power_dbm, 'device') else 'cpu')
    noise_power_lin = torch.tensor(10 ** (cfg.noise_power_dbm / 10) * 1e-3, dtype=torch.float32, device=tx_power_dbm.device if hasattr(tx_power_dbm, 'device') else 'cpu')
    
    sinr = []
    for j in range(n_pairs):  # For each receiver pair j
        # Compute E[SINR_j] = sum_k P(FA_j=k) * SINR_j_k
        expected_sinr_j = 0.0
        
        for k in range(n_fa):  # For each possible FA assignment k for pair j
            # Desired signal when pair j uses FA k
            desired_k = tx_power_lin[j] * g[j, j]
            
            # Interference when pair j uses FA k
            interference_k = 0.0
            for i in range(n_pairs):  # For each interferer pair i
                if i != j:
                    # Expected interference from pair i to pair j when j uses FA k
                    for i_fa in range(n_fa):
                        if i_fa == k:  # Only interferes if same FA
                            interference_k += fa_probs[i, i_fa] * tx_power_lin[i] * g[i, j]
            
            # SINR for this specific FA assignment k
            sinr_j_k = desired_k / (interference_k + noise_power_lin)
            
            # Add weighted contribution to expected SINR
            expected_sinr_j += fa_probs[j, k] * sinr_j_k
        
        sinr.append(expected_sinr_j)
    
    sinr = torch.stack(sinr)
    sum_rate = cfg.bandwidth_hz * torch.sum(torch.log2(1.0 + sinr))
    return -sum_rate  # negative for minimization

def negative_sum_rate_loss_torch_from_matrix(tx_power_dbm, fa_probs, scenario, cfg: SimulationConfig, channel_gains_db):
    """
    Like negative_sum_rate_loss_torch, but takes channel_gains_db (dB, shape [n_pairs, n_pairs] or [B, n_pairs, n_pairs]) directly.
    Applies FA penalty as in get_channel_gain_with_fa_penalty.
    Fully vectorized for batch support.
    """
    n_pairs = cfg.n_pairs
    n_fa = cfg.n_fa
    # Handle batch dimension
    if tx_power_dbm.dim() == 1:
        # Single sample
        tx_power_dbm = tx_power_dbm.unsqueeze(0)
        fa_probs = fa_probs.unsqueeze(0)
        channel_gains_db = channel_gains_db.unsqueeze(0)
    B = tx_power_dbm.shape[0]
    tx_power_lin = 10 ** (tx_power_dbm / 10) * 1e-3  # [B, n_pairs]
    if n_fa == 1:
        # For n_fa=1, all FA indices are 0
        fa_indices = torch.zeros((B, n_pairs), dtype=torch.long, device=tx_power_dbm.device)
    else:
        fa_indices = torch.argmax(fa_probs, dim=2)  # [B, n_pairs]
    fa_indices_np = fa_indices.detach().cpu().numpy()
    channel_gain_db = channel_gains_db.clone().cpu().numpy() if isinstance(channel_gains_db, torch.Tensor) else channel_gains_db.copy()
    # Vectorized FA penalty application
    channel_gain_db -= cfg.fa_penalty_db * fa_indices_np[:, None, :]
    g = torch.tensor(10 ** (channel_gain_db / 10), dtype=torch.float32, device=tx_power_dbm.device)  # [B, n_pairs, n_pairs]
    noise_power_lin = torch.tensor(10 ** (cfg.noise_power_dbm / 10) * 1e-3, dtype=torch.float32, device=tx_power_dbm.device)
    # Vectorized SINR computation
    fa_i = fa_indices.unsqueeze(2)  # [B, n_pairs, 1]
    fa_j = fa_indices.unsqueeze(1)  # [B, 1, n_pairs]
    same_fa = (fa_i == fa_j)        # [B, n_pairs, n_pairs]
    tx_power_lin_exp = tx_power_lin.unsqueeze(1)  # [B, 1, n_pairs]
    g_exp = g                      # [B, n_pairs, n_pairs]
    interf_matrix = same_fa * tx_power_lin_exp * g_exp  # [B, n_pairs, n_pairs]
    total_interf = interf_matrix.sum(dim=2)  # [B, n_pairs]
    self_interf = tx_power_lin * g.diagonal(dim1=1, dim2=2)  # [B, n_pairs]
    interference = total_interf - self_interf  # [B, n_pairs]
    desired = self_interf  # [B, n_pairs]
    sinr = desired / (interference + noise_power_lin)  # [B, n_pairs]
    sum_rate = cfg.bandwidth_hz * torch.sum(torch.log2(1.0 + sinr), dim=1)  # [B]
    return -sum_rate.mean()  # mean loss over batch

def negative_sum_rate_loss_torch_soft_from_matrix(tx_power_dbm, fa_probs, scenario, cfg: SimulationConfig, channel_gains_db):
    """
    CORRECTED: Computes E[SINR] properly, not E[desired]/E[interference].
    Vectorized: supports batched tx_power_dbm, fa_probs, channel_gains_db.
    tx_power_dbm: [B, n_pairs]
    fa_probs: [B, n_pairs, n_fa]
    channel_gains_db: [B, n_pairs, n_pairs]
    Returns mean loss over batch.
    """
    n_pairs = cfg.n_pairs
    n_fa = cfg.n_fa
    # Ensure batch dimension
    if tx_power_dbm.dim() == 1:
        tx_power_dbm = tx_power_dbm.unsqueeze(0)
        fa_probs = fa_probs.unsqueeze(0)
        channel_gains_db = channel_gains_db.unsqueeze(0)
    B = tx_power_dbm.shape[0]
    tx_power_lin = 10 ** (tx_power_dbm / 10) * 1e-3  # [B, n_pairs]
    g = 10 ** (channel_gains_db / 10)  # [B, n_pairs, n_pairs]
    noise_power_lin = 10 ** (cfg.noise_power_dbm / 10) * 1e-3
    noise_power_lin = torch.tensor(noise_power_lin, dtype=torch.float32, device=tx_power_dbm.device)
    
    sinr = []  # Will be [B, n_pairs]
    for j in range(n_pairs):  # For each receiver pair j
        # Compute E[SINR_j] = sum_k P(FA_j=k) * SINR_j_k
        expected_sinr_j = torch.zeros(B, device=tx_power_dbm.device)
        
        for k in range(n_fa):  # For each possible FA assignment k for pair j
            # Desired signal when pair j uses FA k
            desired_k = tx_power_lin[:, j] * g[:, j, j]  # [B]
            
            # Interference when pair j uses FA k
            interference_k = torch.zeros(B, device=tx_power_dbm.device)
            for i in range(n_pairs):  # For each interferer pair i
                if i != j:
                    # Expected interference from pair i to pair j when j uses FA k
                    # This is the sum over all possible FA assignments for pair i
                    for i_fa in range(n_fa):
                        if i_fa == k:  # Only interferes if same FA
                            interference_k += fa_probs[:, i, i_fa] * tx_power_lin[:, i] * g[:, i, j]
            
            # SINR for this specific FA assignment k
            sinr_j_k = desired_k / (interference_k + noise_power_lin)  # [B]
            
            # Add weighted contribution to expected SINR
            expected_sinr_j += fa_probs[:, j, k] * sinr_j_k
        
        sinr.append(expected_sinr_j)
    
    sinr = torch.stack(sinr, dim=1)  # [B, n_pairs]
    sum_rate = cfg.bandwidth_hz * torch.sum(torch.log2(1.0 + sinr), dim=1)  # [B]
    return -sum_rate.mean()  # mean loss over batch

# NEW: bandwidth-free loss for single-frequency memorisation ---------------------------------
def negative_sum_rate_loss_torch_fa1_nobw(tx_power_lin, fa_probs, cfg: SimulationConfig, channel_gains_db):
    """FA=1 loss identical to negative_sum_rate_loss_torch_from_matrix but *without* the
    bandwidth_Hz scale factor.  Intended for tiny-N memorisation where the extra
    10 MHz constant merely rescales gradients.
    Parameters are identical to the original function (scenario unused)."""
    n_pairs = cfg.n_pairs
    # Ensure batch dimension
    if tx_power_lin.dim() == 1:
        tx_power_lin = tx_power_lin.unsqueeze(0)
        fa_probs = fa_probs.unsqueeze(0)
        channel_gains_db = channel_gains_db.unsqueeze(0)
    B = tx_power_lin.shape[0]
    # All FA indices are 0 when n_fa == 1
    fa_indices = torch.zeros((B, n_pairs), dtype=torch.long, device=tx_power_lin.device)
    # Apply FA penalty (none for fa=1)
    g = 10 ** (channel_gains_db / 10)  # [B,P,P]
    noise_power_lin = torch.tensor(10 ** (cfg.noise_power_dbm / 10) * 1e-3, dtype=torch.float32, device=tx_power_lin.device)
    
    # Vectorised SINR
    sinr_numer = tx_power_lin * g.diagonal(dim1=1, dim2=2)               # [B,P]
    tx_exp      = tx_power_lin.unsqueeze(1)                               # [B,1,P]
    interf_mat  = tx_exp * g                                              # [B,P,P]
    interference = interf_mat.sum(dim=2) - sinr_numer                    # [B,P]
    sinr = sinr_numer / (interference + noise_power_lin)                 # [B,P]
    sum_rate = torch.sum(torch.log2(1.0 + sinr), dim=1)                  # [B]
    return -sum_rate.mean()

def negative_sum_rate_loss_torch_fa1_nobw_hard(power_logits, fa_probs, cfg: SimulationConfig, channel_gains_db, power_levels):
    """Hard-decision version of the FA1 no-BW loss, for discrete power training."""
    n_pairs = cfg.n_pairs
    num_levels = len(power_levels)
    B = power_logits.shape[0]
    
    # Reshape logits before argmax
    power_logits_reshaped = power_logits.view(B, n_pairs, num_levels)

    # Hard selection of power levels
    power_indices = torch.argmax(power_logits_reshaped, dim=2) # [B, n_pairs]
    power_levels_tensor = torch.tensor(power_levels, dtype=torch.float32, device=power_logits.device)
    tx_power_lin = power_levels_tensor[power_indices] # [B, n_pairs]

    fa_indices = torch.zeros((B, n_pairs), dtype=torch.long, device=power_logits.device)
    g = 10 ** (channel_gains_db / 10)
    noise_power_lin = torch.tensor(10 ** (cfg.noise_power_dbm / 10) * 1e-3, dtype=torch.float32, device=power_logits.device)
    
    sinr_numer = tx_power_lin * g.diagonal(dim1=1, dim2=2)
    tx_exp = tx_power_lin.unsqueeze(1)
    interf_mat = tx_exp * g
    interference = interf_mat.sum(dim=2) - sinr_numer
    sinr = sinr_numer / (interference + noise_power_lin + 1e-12) # Add epsilon to denominator
    
    sum_rate = torch.sum(torch.log2(1.0 + sinr), dim=1)
    return -sum_rate.mean()

# ---------------------------------------------------------------------------------------------

# Note: Training loop and dataset loading will be added in a separate module or script.

# CLI/Integration note:
# To use custom hidden sizes, instantiate ML_DNN(cfg, hidden_size=[200, 200]) or parse from CLI as needed.

# ------------------------- Training Helper ----------------------------

def normalize_input(x):
    mean = x.mean()
    std = x.std()
    if std == 0:
        std = 1.0
    return (x - mean) / std, mean, std

def normalize_input_with_stats(x, mean, std):
    if std == 0:
        std = 1.0
    return (x - mean) / std

def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def train_model(cfg: SimulationConfig, hidden_size=None, epochs: int = 500, lr: float = 1e-3, verbose: bool = True, save_path: Union[str, Path] = None, patience: int = 50, min_delta: float = 1e-3, soft_fa: bool = False, train_npy: str = None, batch_size: int = 64, n_train_samples: int = None, batch_norm: bool = True, model_seed: int = None, device=None, restrict_rx_distance: bool = False, normalization: str = 'global', fa_gumbel_softmax: bool = False, gumbel_temp: float = 1.0, scale_epochs_by_samples: bool = False, shuffle_data: Optional[bool] = None, *, scenario: Optional[Scenario] = None, discrete_power: bool = False, power_levels: Optional[List[float]] = None, **kwargs):
    """Train ML_DNN on one or many scenarios. Normalization scheme is controlled by the 'normalization' argument."""
    from ..simulator.scenario import Scenario, ChannelGainDataset
    # This is a desperate measure. If some other part of the code is globally
    # disabling gradients (e.g. via a stray torch.no_grad() context that
    # I cannot find), this will explicitly re-enable it for the training scope.
    torch.set_grad_enabled(True)
    
    device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    # Convert placeholder string 'auto' to a real device understood by torch
    if isinstance(device, str) and device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")
    input_mean = None
    input_std = None
    
    if train_npy is not None:
        dataset = ChannelGainDataset(train_npy)
        if n_train_samples is None or n_train_samples <= 0:
            n_train_samples = len(dataset)
        if n_train_samples < len(dataset):
            dataset.data = dataset.data[:n_train_samples]
            print(f"Using first {n_train_samples} samples from {train_npy}")
        else:
            print(f"Using all {len(dataset)} samples from {train_npy}")
        use_batch_norm = batch_norm and batch_size > 1 and n_train_samples > 1
        if n_train_samples is not None and n_train_samples <= 10:
            # BatchNorm statistics are unreliable on tiny datasets; disable it.
            use_batch_norm = False
            if batch_norm:
                print(f"[INFO] Disabling BatchNorm for tiny dataset (N={n_train_samples})")
        all_x = dataset.data.reshape(len(dataset), -1)
        if normalization == 'global':
            input_mean = all_x.mean(axis=0)
            input_std = all_x.std(axis=0)
            input_std[input_std == 0] = 1.0
            print(f"[INFO] Global normalization stats - Mean: {input_mean.mean():.2f} dB, Std: {input_std.mean():.2f} dB")
        elif normalization == 'local':
            print(f"[INFO] Per-sample normalization will be applied (local mode)")
        elif normalization == 'none':
            print(f"[INFO] No normalization will be applied (none mode)")
        else:
            raise ValueError(f"Unknown normalization scheme: {normalization}")
    else:
        # If no npy file, default to generating a single scenario internally,
        # unless one is explicitly passed in.
        use_batch_norm = False
        if scenario is None:
            print("[WARNING] No training data or scenario provided. Generating a new random scenario for training.")
            scenario = Scenario.random(cfg, restrict_rx_distance=restrict_rx_distance)

    algo = ML_DNN(cfg, hidden_size=hidden_size, batch_norm=use_batch_norm, device=device, fa_gumbel_softmax=fa_gumbel_softmax, gumbel_temp=gumbel_temp, discrete_power=discrete_power, power_levels=power_levels)
    model = algo.model
    if model_seed is not None:
        torch.manual_seed(model_seed)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    # Auto-adjust learning-rate for very small memorisation sets
    if (lr == 1e-3) and (n_train_samples is not None) and (n_train_samples <= 10):
        lr = 3e-3  # empirically matches dnn_d2d default for FA=1
        print(f"[INFO] Auto-increase LR to {lr} for tiny dataset (N={n_train_samples})")

    losses = []
    best_loss = float('inf')
    best_epoch = 0
    best_state = None
    
    if train_npy is not None:
        # Do not shuffle when the goal is memorisation (train == val); we rely on
        # the batch index to slice `original_data` so the ordering must stay
        # consistent.
        if shuffle_data is None:
            shuffle_flag = False if (n_train_samples == len(dataset)) else True
        else:
            shuffle_flag = shuffle_data
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            drop_last=False
        )
        print(f"Training on {len(dataset)} samples, batch size {batch_size}, {len(train_loader)} batches per epoch")
        original_data = torch.from_numpy(dataset.data).float()
        # Create one optimiser for the whole training run (keeps Adam momentum/history).
        optimizer = Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            model.train()
            epoch_losses = []
            for batch_idx, batch in enumerate(train_loader):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(dataset))
                x = batch.view(batch.size(0), -1).to(device)
                if normalization == 'global':
                    x_normalized = (x - torch.tensor(input_mean, dtype=x.dtype, device=device)) / torch.tensor(input_std, dtype=x.dtype, device=device)
                elif normalization == 'local':
                    # Per-sample normalization for each sample in batch
                    means = x.mean(dim=1, keepdim=True)
                    stds = x.std(dim=1, keepdim=True) + 1e-8
                    x_normalized = (x - means) / stds
                elif normalization == 'none':
                    x_normalized = x
                else:
                    raise ValueError(f"Unknown normalization scheme: {normalization}")
                original_batch = original_data[batch_start:batch_end].to(device)
                # Forward pass – model returns (power_lin, fa_probs)
                power_lin, fa_probs = model(x_normalized)

                if soft_fa and cfg.n_fa > 1:
                    # Differentiable FA loss (multi-FA case)
                    tx_power_dbm = 10 * torch.log10(power_lin + 1e-12) + 30
                    loss = negative_sum_rate_loss_torch_soft_from_matrix(
                        tx_power_dbm, fa_probs, None, cfg, original_batch)
                else:
                    if cfg.n_fa == 1:
                        # FA=1 bandwidth-free loss (memorisation setting)
                        loss = negative_sum_rate_loss_torch_fa1_nobw(
                            power_lin, fa_probs, cfg, original_batch)
                    else:
                        tx_power_dbm = 10 * torch.log10(power_lin + 1e-12) + 30
                        loss = negative_sum_rate_loss_torch_from_matrix(
                            tx_power_dbm, fa_probs, None, cfg, original_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                best_epoch = ep
                best_state = model.state_dict()
            elif ep - best_epoch >= patience:
                if verbose:
                    print(f"Early stopping at epoch {ep+1} (no improvement for {patience} epochs)")
                break
            if verbose and ((ep + 1) % 10 == 0 or ep == epochs - 1):
                print(f"Epoch {ep+1}/{epochs}: Avg sum-rate = { -avg_loss:.2e} bit/s")
        if best_state is not None:
            model.load_state_dict(best_state)
    else:
        val_scenario = scenario
        x_raw = torch.tensor(scenario.channel_gains_db(), dtype=torch.float32, device=device).flatten()
        sample_flat = x_raw.unsqueeze(0)
        if normalization == 'global':
            # For single sample, global = per-feature stats (degenerates to local)
            input_mean = sample_flat.mean(dim=0)
            input_std = sample_flat.std(dim=0)
            input_std[input_std == 0] = 1.0
            print(f"[INFO] Global normalization (single sample): Mean: {input_mean.mean().item():.2f} dB, Std: {input_std.mean().item():.2f} dB")
            x_normalized = (sample_flat - input_mean) / input_std
        elif normalization == 'local':
            input_mean = sample_flat.mean(dim=1, keepdim=True)
            input_std = sample_flat.std(dim=1, keepdim=True) + 1e-8
            print(f"[INFO] Single-sample normalization - Mean: {input_mean.item():.2f} dB, Std: {input_std.item():.2f} dB")
            x_normalized = (sample_flat - input_mean) / input_std
        elif normalization == 'none':
            print(f"[INFO] No normalization applied (none mode)")
            x_normalized = sample_flat
        else:
            raise ValueError(f"Unknown normalization scheme: {normalization}")
        val_x_raw = torch.tensor(val_scenario.channel_gains_db(), dtype=torch.float32, device=device).flatten()
        if normalization == 'global':
            val_x_normalized = (val_x_raw.unsqueeze(0) - input_mean) / input_std
        elif normalization == 'local':
            val_x_normalized = (val_x_raw.unsqueeze(0) - input_mean) / input_std
        elif normalization == 'none':
            val_x_normalized = val_x_raw.unsqueeze(0)
        else:
            raise ValueError(f"Unknown normalization scheme: {normalization}")
        
        # Ensure channel gains are a tensor that requires gradients for the loss function
        original_gains = torch.tensor(
            scenario.channel_gains_db(), dtype=torch.float32,
            device=device, requires_grad=False).unsqueeze(0)

        print_intervals = set([int(ep * (epochs - 1) / 9) for ep in range(10)])
        val_losses = []
        optimizer = Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            model.train()
            power_lin, fa_probs = model(x_normalized)

            # Create a version of gains that requires grad for this training step
            gains_for_loss = original_gains.clone().detach().requires_grad_(True)

            if soft_fa and cfg.n_fa > 1:
                tx_power_dbm = 10 * torch.log10(power_lin + 1e-12) + 30
                loss = negative_sum_rate_loss_torch_soft_from_matrix(
                    tx_power_dbm, fa_probs, None, cfg, gains_for_loss)
            else:
                if cfg.n_fa == 1:
                    loss = negative_sum_rate_loss_torch_fa1_nobw(
                        power_lin, fa_probs, cfg, gains_for_loss)
                else:
                    tx_power_dbm = 10 * torch.log10(power_lin + 1e-12) + 30
                    loss = negative_sum_rate_loss_torch_from_matrix(
                        tx_power_dbm, fa_probs, None, cfg, gains_for_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            model.eval()
            with torch.no_grad():
                val_power_lin, val_fa_probs = model(val_x_normalized)
                if cfg.n_fa == 1:
                    val_loss = negative_sum_rate_loss_torch_fa1_nobw(
                        val_power_lin, val_fa_probs, cfg, original_gains)
                else:
                    val_tx_power_dbm = 10 * torch.log10(val_power_lin + 1e-12) + 30
                    if soft_fa:
                        val_loss = negative_sum_rate_loss_torch_soft_from_matrix(
                            val_tx_power_dbm, val_fa_probs, None, cfg, original_gains)
                    else:
                        val_loss = negative_sum_rate_loss_torch_from_matrix(
                            val_tx_power_dbm, val_fa_probs, None, cfg, original_gains)
                val_losses.append(val_loss.item())
            if val_loss.item() < best_loss - min_delta:
                best_loss = val_loss.item()
                best_epoch = ep
                best_state = model.state_dict()
            elif ep - best_epoch >= patience:
                if verbose:
                    print(f"Early stopping at epoch {ep+1} (no improvement for {patience} epochs)")
                break
            if verbose and (ep in print_intervals or ep == epochs - 1):
                print(f"Epoch {ep+1}/{epochs}: Train sum-rate = { -loss.item():.2e} bit/s, Val sum-rate = { -val_loss.item():.2e} bit/s")
        if best_state is not None:
            model.load_state_dict(best_state)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        algo.save_weights(save_path)
        if verbose:
            print(f"Saved trained weights to {save_path}")
    meta = None
    if save_path:
        def path_to_str(obj):
            if isinstance(obj, dict):
                return {k: path_to_str(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple, set)):
                return [path_to_str(v) for v in obj]
            elif isinstance(obj, Path):
                return str(obj)
            try:
                yaml.safe_dump(obj)
                return obj
            except Exception:
                return str(obj)
        cfg_dict = cfg.to_dict() if hasattr(cfg, 'to_dict') else dict(cfg.__dict__)
        cfg_dict = path_to_str(cfg_dict)
        meta = {
            "hidden_size": hidden_size,
            "cfg": cfg_dict,
            "normalization": normalization,
            "input_mean": input_mean.tolist() if (normalization == 'global' and input_mean is not None) else None,
            "input_std": input_std.tolist() if (normalization == 'global' and input_std is not None) else None,
        }
        meta = path_to_str(meta)
        meta_path = save_path + ".meta.yaml"
        with open(meta_path, "w") as f:
            yaml.safe_dump(meta, f)
    
    return algo, losses, meta

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Train DNN subcommand
    train_parser = subparsers.add_parser("train_dnn")
    train_parser.add_argument("--soft-fa", action="store_true", help="Use soft FA loss (differentiable)")

    args = parser.parse_args()

    if args.command == "train_dnn":
        # ... load config, etc. ...
        train_model(
            cfg,
            # ... other args ...,
            soft_fa=args.soft_fa
        )

if __name__ == "__main__":
    main()

def save_checkpoint(model, path, arch):
    """Save model state_dict and architecture parameters."""
    torch.save({'state_dict': model.state_dict(), 'arch': arch}, path)


def load_checkpoint(path, device=None):
    """Load checkpoint and return (state_dict, arch)."""
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    arch = checkpoint.get('arch', None)
    return state, arch 

# --- Differentiable SINR and Loss (Reference Implementation) ---

def compute_expected_sinr(p: torch.Tensor,
                          fa: torch.Tensor,
                          gains_db: torch.Tensor,
                          noise_power: float = 1e-13,
                          eps: float = 1e-12) -> torch.Tensor:
    r"""Compute the **expected** SINR for each receiver from the reference script."""

    # Convert gains to linear scale
    gains_lin = torch.pow(10.0, gains_db / 10.0)           # (B, P, P)

    B, num_pairs, num_freq = fa.shape

    g_ii = gains_lin.diagonal(dim1=-2, dim2=-1)
    desired_signal = p * g_ii

    prob_same_freq = torch.einsum('bik,bjk->bij', fa, fa)
    p_expanded = p.unsqueeze(1).expand(-1, num_pairs, -1)
    interference_matrix = prob_same_freq * p_expanded * gains_lin
    interference_matrix = interference_matrix - torch.diag_embed(desired_signal, dim1=-2, dim2=-1)
    expected_interference = interference_matrix.sum(dim=2)
    total_denominator = expected_interference + noise_power + eps
    expected_sinr = desired_signal / total_denominator
    return expected_sinr


def negative_sum_rate_loss_fa1_reference(tx_power_lin, fa_probs, channel_gains_db, noise_power):
    """Expected-SINR loss for single-frequency (FA = 1), from reference."""
    if tx_power_lin.dim() == 1:
        tx_power_lin = tx_power_lin.unsqueeze(0)
    if fa_probs.dim() == 2:
        fa_probs = fa_probs.unsqueeze(0)
    if channel_gains_db.dim() == 2:
        channel_gains_db = channel_gains_db.unsqueeze(0)

    expected_sinr = compute_expected_sinr(tx_power_lin, fa_probs, channel_gains_db, noise_power)
    sum_rate = torch.log2(1.0 + expected_sinr).sum(dim=1).mean()
    return -sum_rate


# --- Model Definition (Reference Implementation) ---

class D2DNet(nn.Module):
    def __init__(self, n_pairs, n_fa, hidden_sizes, batch_norm=True, *, discrete_power=False, power_levels=None, cfg: SimulationConfig):
        super(D2DNet, self).__init__()
        self.n_pairs = n_pairs
        self.n_fa = n_fa
        self.discrete_power = discrete_power
        self.power_levels = power_levels or [1e-10, 0.25, 0.5, 1.0]
        self.num_power_levels = len(self.power_levels)

        self.min_power_watts = 10**((cfg.tx_power_min_dbm - 30) / 10)
        self.max_power_watts = 10**((cfg.tx_power_max_dbm - 30) / 10)

        layers = []
        input_size = n_pairs * n_pairs
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        self.hidden_layers = nn.ModuleList(layers)
        
        if discrete_power:
            self.power_head = nn.Linear(hidden_sizes[-1], n_pairs * self.num_power_levels)
        else:
            self.power_head = nn.Linear(hidden_sizes[-1], n_pairs)
        
        if n_fa > 1:
            self.fa_head = nn.Linear(hidden_sizes[-1], n_pairs * n_fa)
        else:
            self.fa_head = None

    def forward(self, x):
        h = x
        for layer in self.hidden_layers:
            if isinstance(layer, nn.BatchNorm1d) and h.shape[0] == 1 and self.training:
                continue
            h = layer(h)
        
        if self.discrete_power:
            power_logits = self.power_head(h).view(-1, self.n_pairs, self.num_power_levels)
            power_probs = F.softmax(power_logits, dim=2)
            power_levels_tensor = torch.tensor(self.power_levels, device=x.device, dtype=x.dtype)
            power_values = torch.sum(power_probs * power_levels_tensor.view(1, 1, -1), dim=2)
        else:
            power_logits = self.power_head(h)
            power_values = torch.sigmoid(power_logits) * (self.max_power_watts - self.min_power_watts) + self.min_power_watts
        
        if self.fa_head is not None:
            fa_logits = self.fa_head(h)
            fa_probs = F.softmax(fa_logits.view(-1, self.n_pairs, self.n_fa), dim=2)
        else:
            fa_probs = torch.ones((x.size(0), self.n_pairs, 1), device=x.device)
        
        return power_values, fa_probs 