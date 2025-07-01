"""
Training loss functions for D2D power and frequency allocation DNN.

This module contains differentiable loss functions specifically designed for training.
These are separate from validation functions and focus on gradient-based optimization.
"""

import torch
import numpy as np
from typing import Optional, Union

from ..config import SimulationConfig


def negative_sum_rate_loss_hard_fa(
    tx_power_dbm: torch.Tensor,
    fa_probs: torch.Tensor,
    channel_gains_db: torch.Tensor,
    cfg: SimulationConfig
) -> torch.Tensor:
    """
    Training loss function with hard FA assignment (argmax).
    
    This function uses discrete FA assignments obtained via argmax of probabilities.
    It's faster but may have less smooth gradients compared to soft assignment.
    
    Parameters
    ----------
    tx_power_dbm : torch.Tensor
        Transmit power in dBm, shape [B, n_pairs] or [n_pairs]
    fa_probs : torch.Tensor
        FA probability distributions, shape [B, n_pairs, n_fa] or [n_pairs, n_fa]
    channel_gains_db : torch.Tensor
        Channel gain matrix in dB, shape [B, n_pairs, n_pairs] or [n_pairs, n_pairs]
    cfg : SimulationConfig
        Configuration object
        
    Returns
    -------
    torch.Tensor
        Negative sum rate loss (scalar for batch, negative for minimization)
    """
    n_pairs = cfg.n_pairs
    n_fa = cfg.n_fa
    
    # Ensure batch dimension
    if tx_power_dbm.dim() == 1:
        tx_power_dbm = tx_power_dbm.unsqueeze(0)
        fa_probs = fa_probs.unsqueeze(0)
        channel_gains_db = channel_gains_db.unsqueeze(0)
    
    B = tx_power_dbm.shape[0]
    device = tx_power_dbm.device
    
    # Convert power from dBm to linear Watts
    tx_power_lin = 10 ** (tx_power_dbm / 10) * 1e-3  # [B, n_pairs]
    
    # Get FA indices from probabilities (hard assignment)
    if n_fa == 1:
        fa_indices = torch.zeros((B, n_pairs), dtype=torch.long, device=device)
    else:
        fa_indices = torch.argmax(fa_probs, dim=2)  # [B, n_pairs]
    
    # Apply FA penalty to channel gains
    channel_gain_db = channel_gains_db.clone()
    fa_indices_np = fa_indices.detach().cpu().numpy()
    channel_gain_db -= cfg.fa_penalty_db * torch.tensor(
        fa_indices_np[:, None, :], dtype=torch.float32, device=device
    )
    
    # Convert to linear scale
    g = 10 ** (channel_gain_db / 10)  # [B, n_pairs, n_pairs]
    
    # Vectorized SINR computation
    fa_i = fa_indices.unsqueeze(2)  # [B, n_pairs, 1]
    fa_j = fa_indices.unsqueeze(1)  # [B, 1, n_pairs]
    same_fa = (fa_i == fa_j)        # [B, n_pairs, n_pairs]
    
    # Calculate interference matrix
    tx_power_lin_exp = tx_power_lin.unsqueeze(1)  # [B, 1, n_pairs]
    interf_matrix = same_fa * tx_power_lin_exp * g  # [B, n_pairs, n_pairs]
    
    # Total interference and desired signal
    total_interf = interf_matrix.sum(dim=2)  # [B, n_pairs]
    self_interf = tx_power_lin * g.diagonal(dim1=1, dim2=2)  # [B, n_pairs]
    interference = total_interf - self_interf  # [B, n_pairs]
    desired = self_interf  # [B, n_pairs]
    
    # SINR calculation
    noise_power_lin = torch.tensor(
        10 ** (cfg.noise_power_dbm / 10) * 1e-3, 
        dtype=torch.float32, 
        device=device
    )
    sinr = desired / (interference + noise_power_lin)  # [B, n_pairs]
    
    # Sum rate calculation
    sum_rate = cfg.bandwidth_hz * torch.sum(torch.log2(1.0 + sinr), dim=1)  # [B]
    
    return -sum_rate.mean()  # Negative for minimization


def negative_sum_rate_loss_soft_fa(
    tx_power_dbm: torch.Tensor,
    fa_probs: torch.Tensor,
    channel_gains_db: torch.Tensor,
    cfg: SimulationConfig
) -> torch.Tensor:
    """
    Training loss function with soft FA assignment (expected SINR).
    
    This function calculates the expected SINR correctly by computing
    E[SINR] = Σ_k P(FA=k) × SINR_k, not E[desired]/E[interference].
    
    Parameters
    ----------
    tx_power_dbm : torch.Tensor
        Transmit power in dBm, shape [B, n_pairs] or [n_pairs]
    fa_probs : torch.Tensor
        FA probability distributions, shape [B, n_pairs, n_fa] or [n_pairs, n_fa]
    channel_gains_db : torch.Tensor
        Channel gain matrix in dB, shape [B, n_pairs, n_pairs] or [n_pairs, n_pairs]
    cfg : SimulationConfig
        Configuration object
        
    Returns
    -------
    torch.Tensor
        Negative sum rate loss (scalar for batch, negative for minimization)
    """
    n_pairs = cfg.n_pairs
    n_fa = cfg.n_fa
    
    # Ensure batch dimension
    if tx_power_dbm.dim() == 1:
        tx_power_dbm = tx_power_dbm.unsqueeze(0)
        fa_probs = fa_probs.unsqueeze(0)
        channel_gains_db = channel_gains_db.unsqueeze(0)
    
    B = tx_power_dbm.shape[0]
    device = tx_power_dbm.device
    
    # Convert power from dBm to linear Watts
    tx_power_lin = 10 ** (tx_power_dbm / 10) * 1e-3  # [B, n_pairs]
    
    # Channel gains (base, without FA penalty)
    g_base = 10 ** (channel_gains_db / 10)  # [B, n_pairs, n_pairs]
    
    # Noise power
    noise_power_lin = torch.tensor(
        10 ** (cfg.noise_power_dbm / 10) * 1e-3,
        dtype=torch.float32,
        device=device
    )
    
    expected_sinr = []  # Will be [B, n_pairs]
    
    for j in range(n_pairs):
        # Calculate expected SINR for receiver j
        # E[SINR_j] = Σ_k P(j uses FA k) × SINR_j_k
        sinr_j_expected = torch.zeros(B, device=device)
        
        for k_j in range(n_fa):  # FA assignment for receiver j
            # Calculate SINR when receiver j uses FA k_j
            g_penalty_factor_j = 10 ** (-cfg.fa_penalty_db * k_j / 10)
            
            # Desired signal when j uses FA k_j
            desired_j_k = (
                tx_power_lin[:, j] * g_base[:, j, j] * g_penalty_factor_j
            )
            
            # Expected interference to j when j uses FA k_j
            # E[I_j | j uses FA k_j] = Σ_i≠j Σ_k_i P(i uses FA k_i | j uses FA k_j) × interference_i_to_j
            interference_j_k = torch.zeros(B, device=device)
            
            for i in range(n_pairs):
                if i != j:
                    for k_i in range(n_fa):  # FA assignment for interferer i
                        if k_i == k_j:  # Same FA causes interference
                            g_penalty_factor_i = 10 ** (-cfg.fa_penalty_db * k_i / 10)
                            interference_contribution = (
                                tx_power_lin[:, i] * g_base[:, i, j] * g_penalty_factor_i
                            )
                            # Weight by probability that i uses FA k_i
                            interference_j_k += fa_probs[:, i, k_i] * interference_contribution
            
            # SINR when j uses FA k_j
            sinr_j_k = desired_j_k / (interference_j_k + noise_power_lin)
            
            # Weight by probability that j uses FA k_j
            sinr_j_expected += fa_probs[:, j, k_j] * sinr_j_k
        
        expected_sinr.append(sinr_j_expected)
    
    expected_sinr = torch.stack(expected_sinr, dim=1)  # [B, n_pairs]
    sum_rate = cfg.bandwidth_hz * torch.sum(torch.log2(1.0 + expected_sinr), dim=1)  # [B]
    
    return -sum_rate.mean()  # Negative for minimization


def negative_sum_rate_loss_adaptive(
    tx_power_dbm: torch.Tensor,
    fa_probs: torch.Tensor,
    channel_gains_db: torch.Tensor,
    cfg: SimulationConfig,
    epoch: int = 0,
    total_epochs: int = 1000,
    soft_epochs_ratio: float = 0.3
) -> torch.Tensor:
    """
    Adaptive loss function that transitions from soft to hard FA assignment.
    
    Uses soft FA assignment during early training for better gradients, then
    transitions to hard assignment for discrete decisions.
    
    Parameters
    ----------
    tx_power_dbm : torch.Tensor
        Transmit power in dBm
    fa_probs : torch.Tensor
        FA probability distributions
    channel_gains_db : torch.Tensor
        Channel gain matrix in dB
    cfg : SimulationConfig
        Configuration object
    epoch : int
        Current training epoch
    total_epochs : int
        Total number of training epochs
    soft_epochs_ratio : float
        Fraction of epochs to use soft assignment (0.3 = first 30%)
        
    Returns
    -------
    torch.Tensor
        Negative sum rate loss
    """
    soft_epochs = int(total_epochs * soft_epochs_ratio)
    
    if epoch < soft_epochs:
        # Use soft assignment during early training
        return negative_sum_rate_loss_soft_fa(
            tx_power_dbm, fa_probs, channel_gains_db, cfg
        )
    else:
        # Use hard assignment during later training
        return negative_sum_rate_loss_hard_fa(
            tx_power_dbm, fa_probs, channel_gains_db, cfg
        )


def negative_sum_rate_loss_weighted(
    tx_power_dbm: torch.Tensor,
    fa_probs: torch.Tensor,
    channel_gains_db: torch.Tensor,
    cfg: SimulationConfig,
    fairness_weight: float = 0.0,
    power_weight: float = 0.0
) -> torch.Tensor:
    """
    Weighted loss function that can include fairness and power efficiency terms.
    
    Parameters
    ----------
    tx_power_dbm : torch.Tensor
        Transmit power in dBm
    fa_probs : torch.Tensor
        FA probability distributions
    channel_gains_db : torch.Tensor
        Channel gain matrix in dB
    cfg : SimulationConfig
        Configuration object
    fairness_weight : float
        Weight for fairness term (0 = no fairness penalty)
    power_weight : float
        Weight for power efficiency term (0 = no power penalty)
        
    Returns
    -------
    torch.Tensor
        Weighted negative sum rate loss
    """
    # Base sum rate loss
    base_loss = negative_sum_rate_loss_hard_fa(
        tx_power_dbm, fa_probs, channel_gains_db, cfg
    )
    
    total_loss = base_loss
    
    # Add fairness penalty if requested
    if fairness_weight > 0:
        # Calculate individual rates for fairness
        # (This is a simplified version - could be more sophisticated)
        tx_power_lin = 10 ** (tx_power_dbm / 10) * 1e-3
        
        # Ensure batch dimension
        if tx_power_dbm.dim() == 1:
            tx_power_lin = tx_power_lin.unsqueeze(0)
        
        # Simple fairness penalty: variance of power allocations
        power_variance = torch.var(tx_power_lin, dim=1).mean()
        total_loss += fairness_weight * power_variance
    
    # Add power efficiency penalty if requested
    if power_weight > 0:
        # Penalty for high power consumption
        tx_power_lin = 10 ** (tx_power_dbm / 10) * 1e-3
        
        if tx_power_dbm.dim() == 1:
            tx_power_lin = tx_power_lin.unsqueeze(0)
        
        avg_power = torch.mean(tx_power_lin)
        total_loss += power_weight * avg_power
    
    return total_loss


def get_loss_function(
    loss_type: str = "hard",
    **kwargs
) -> callable:
    """
    Factory function to get the appropriate loss function.
    
    Parameters
    ----------
    loss_type : str
        Type of loss function:
        - "hard": Hard FA assignment (default)
        - "soft": Soft FA assignment
        - "adaptive": Adaptive soft-to-hard transition
        - "weighted": Weighted loss with fairness/power terms
    **kwargs
        Additional arguments for specific loss functions
        
    Returns
    -------
    callable
        Loss function that takes (tx_power_dbm, fa_probs, channel_gains_db, cfg)
    """
    if loss_type == "hard":
        return negative_sum_rate_loss_hard_fa
    elif loss_type == "soft":
        return negative_sum_rate_loss_soft_fa
    elif loss_type == "adaptive":
        def adaptive_loss(tx_power_dbm, fa_probs, channel_gains_db, cfg, epoch=0):
            return negative_sum_rate_loss_adaptive(
                tx_power_dbm, fa_probs, channel_gains_db, cfg, epoch, **kwargs
            )
        return adaptive_loss
    elif loss_type == "weighted":
        def weighted_loss(tx_power_dbm, fa_probs, channel_gains_db, cfg):
            return negative_sum_rate_loss_weighted(
                tx_power_dbm, fa_probs, channel_gains_db, cfg, **kwargs
            )
        return weighted_loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Legacy compatibility functions (for backward compatibility with existing code)
def negative_sum_rate_loss_torch_from_matrix(tx_power_dbm, fa_probs, scenario, cfg, channel_gains_db):
    """Legacy compatibility wrapper for hard FA loss."""
    return negative_sum_rate_loss_hard_fa(tx_power_dbm, fa_probs, channel_gains_db, cfg)


def negative_sum_rate_loss_torch_soft_from_matrix(tx_power_dbm, fa_probs, scenario, cfg, channel_gains_db):
    """Legacy compatibility wrapper for soft FA loss."""
    return negative_sum_rate_loss_soft_fa(tx_power_dbm, fa_probs, channel_gains_db, cfg) 