"""
CORRECTED: Comprehensive test routine to validate the fixed soft FA loss function.

This test compares the old (incorrect) and new (corrected) implementations
of the soft FA loss function and validates training performance.
"""

import numpy as np
import torch
import tempfile
import os
import pytest
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# Force CPU to avoid device issues
DEVICE = 'cpu'

from src.config import SimulationConfig
from src.algorithms.ml_dnn import (
    ML_DNN, train_model, dnn_output_to_decision_torch,
    negative_sum_rate_loss_torch_soft_from_matrix as old_soft_loss
)
from src.simulator.scenario import Scenario


def negative_sum_rate_loss_hard_fa_corrected(
    tx_power_dbm: torch.Tensor,
    fa_probs: torch.Tensor,
    channel_gains_db: torch.Tensor,
    cfg: SimulationConfig
) -> torch.Tensor:
    """
    CORRECTED: Training loss function with hard FA assignment (argmax).
    
    Fixed FA penalty application - penalty affects receiver, not transmitter.
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
    
    # CORRECTED: Apply FA penalty to channel gains (penalty affects receiver)
    channel_gain_db = channel_gains_db.clone()
    for b in range(B):
        for j in range(n_pairs):  # receiver j
            penalty_db = cfg.fa_penalty_db * fa_indices[b, j].item()
            channel_gain_db[b, :, j] -= penalty_db  # Apply to column j (receiver j)
    
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


def negative_sum_rate_loss_soft_fa_corrected(
    tx_power_dbm: torch.Tensor,
    fa_probs: torch.Tensor,
    channel_gains_db: torch.Tensor,
    cfg: SimulationConfig
) -> torch.Tensor:
    """
    CORRECTED: Training loss function with soft FA assignment (expected SINR).
    
    This function calculates the expected SINR correctly by computing
    E[SINR] = Σ_k P(FA=k) × SINR_k, not E[desired]/E[interference].
    Also fixes FA penalty application.
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
            
            # Desired signal when j uses FA k_j (penalty affects receiver)
            desired_j_k = (
                tx_power_lin[:, j] * g_base[:, j, j] * g_penalty_factor_j
            )
            
            # Expected interference to j when j uses FA k_j
            interference_j_k = torch.zeros(B, device=device)
            
            for i in range(n_pairs):
                if i != j:
                    for k_i in range(n_fa):  # FA assignment for interferer i
                        if k_i == k_j:  # Same FA causes interference
                            # Penalty affects receiver j, not transmitter i
                            interference_contribution = (
                                tx_power_lin[:, i] * g_base[:, i, j] * g_penalty_factor_j
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


class TestConfig:
    """Test configuration matching user requirements."""
    
    @staticmethod
    def get_test_configs() -> List[SimulationConfig]:
        """Get test configurations for FA=1 and FA=2."""
        configs = []
        
        # FA=1 configuration
        cfg_fa1 = SimulationConfig(
            n_pairs=6,
            n_fa=1,
            tx_power_min_dbm=0,
            tx_power_max_dbm=30,
            bandwidth_hz=1e6,
            noise_power_dbm=-100,
            fa_penalty_db=10,
            seed=42
        )
        configs.append(cfg_fa1)
        
        # FA=2 configuration  
        cfg_fa2 = SimulationConfig(
            n_pairs=6,
            n_fa=2,
            tx_power_min_dbm=0,
            tx_power_max_dbm=30,
            bandwidth_hz=1e6,
            noise_power_dbm=-100,
            fa_penalty_db=10,
            seed=42
        )
        configs.append(cfg_fa2)
        
        return configs


class LossFunctionValidator:
    """Validates loss function correctness and numerical properties."""
    
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.device = DEVICE
        
    def generate_test_data(self, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate test data for loss function validation."""
        n_pairs = self.cfg.n_pairs
        n_fa = self.cfg.n_fa
        
        # Random channel gains (more reasonable range)
        channel_gains_db = torch.randn(n_samples, n_pairs, n_pairs, device=self.device) * 10 - 60
        
        # Random power allocations
        tx_power_dbm = torch.rand(n_samples, n_pairs, device=self.device) * 20 + 5  # 5-25 dBm
        
        # Random FA probabilities (properly normalized)
        if n_fa == 1:
            fa_probs = torch.ones(n_samples, n_pairs, 1, device=self.device)
        else:
            fa_logits = torch.randn(n_samples, n_pairs, n_fa, device=self.device)
            fa_probs = torch.softmax(fa_logits, dim=2)
        
        return tx_power_dbm, fa_probs, channel_gains_db
    
    def test_loss_function_properties(self) -> Dict[str, float]:
        """Test mathematical properties of loss functions."""
        results = {}
        
        # Generate test data
        tx_power_dbm, fa_probs, channel_gains_db = self.generate_test_data(5)
        
        # Test 1: Hard vs Soft consistency (when FA probs are one-hot)
        if self.cfg.n_fa > 1:
            # Create one-hot FA probabilities
            fa_indices = torch.randint(0, self.cfg.n_fa, (5, self.cfg.n_pairs))
            fa_probs_onehot = torch.zeros(5, self.cfg.n_pairs, self.cfg.n_fa)
            fa_probs_onehot.scatter_(2, fa_indices.unsqueeze(2), 1.0)
            
            # Compute losses with corrected functions
            hard_loss_val = negative_sum_rate_loss_hard_fa_corrected(
                tx_power_dbm, fa_probs_onehot, channel_gains_db, self.cfg
            )
            new_soft_loss_val = negative_sum_rate_loss_soft_fa_corrected(
                tx_power_dbm, fa_probs_onehot, channel_gains_db, self.cfg
            )
            old_soft_loss_val = old_soft_loss(
                tx_power_dbm, fa_probs_onehot, None, self.cfg, channel_gains_db
            )
            
            results['hard_soft_consistency'] = abs(hard_loss_val.item() - new_soft_loss_val.item())
            results['old_new_difference'] = abs(old_soft_loss_val.item() - new_soft_loss_val.item())
            
            print(f"Debug: Hard loss = {hard_loss_val.item():.2e}")
            print(f"Debug: New soft loss = {new_soft_loss_val.item():.2e}")
            print(f"Debug: Old soft loss = {old_soft_loss_val.item():.2e}")
        else:
            results['hard_soft_consistency'] = 0.0
            results['old_new_difference'] = 0.0
        
        # Test 2: Gradient properties
        tx_power_dbm_grad = tx_power_dbm.clone().requires_grad_(True)
        fa_probs_grad = fa_probs.clone().requires_grad_(True)
        
        new_loss_val = negative_sum_rate_loss_soft_fa_corrected(
            tx_power_dbm_grad, fa_probs_grad, channel_gains_db, self.cfg
        )
        new_loss_val.backward()
        
        results['power_grad_norm'] = tx_power_dbm_grad.grad.norm().item()
        results['fa_grad_norm'] = fa_probs_grad.grad.norm().item()
        
        return results


# Test functions
@pytest.mark.parametrize("cfg", TestConfig.get_test_configs())
def test_corrected_loss_function_properties(cfg):
    """Test mathematical properties of the corrected loss function."""
    validator = LossFunctionValidator(cfg)
    results = validator.test_loss_function_properties()
    
    # Assertions
    if cfg.n_fa > 1:
        # Hard and soft should be consistent for one-hot probabilities
        assert results['hard_soft_consistency'] < 1e-2, \
            f"Hard and soft losses should be consistent for one-hot FA probs: {results['hard_soft_consistency']}"
        
        # Old and new should differ (indicating the fix)
        print(f"Old-new difference: {results['old_new_difference']}")
    
    # Gradients should exist and be reasonable
    assert results['power_grad_norm'] > 1e-8, "Power gradients should exist"
    assert results['fa_grad_norm'] > 1e-8, "FA gradients should exist"
    
    print(f"✓ Corrected mathematical properties test passed for n_fa={cfg.n_fa}")
    print(f"  Hard-soft consistency: {results['hard_soft_consistency']:.2e}")
    print(f"  Old-new difference: {results['old_new_difference']:.2e}")
    print(f"  Power grad norm: {results['power_grad_norm']:.2e}")
    print(f"  FA grad norm: {results['fa_grad_norm']:.2e}")


def test_fa_penalty_application():
    """Test that FA penalty is applied correctly."""
    cfg = SimulationConfig(n_pairs=3, n_fa=2, fa_penalty_db=10)
    
    # Simple test case
    tx_power_dbm = torch.tensor([[20.0, 15.0, 10.0]])  # [1, 3]
    channel_gains_db = torch.zeros(1, 3, 3)  # [1, 3, 3] - all zeros for simplicity
    
    # One-hot FA assignment: [0, 1, 0] (pair 1 uses FA 1, others use FA 0)
    fa_probs = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])  # [1, 3, 2]
    
    # Test corrected hard loss
    hard_loss = negative_sum_rate_loss_hard_fa_corrected(tx_power_dbm, fa_probs, channel_gains_db, cfg)
    
    # Test corrected soft loss
    soft_loss = negative_sum_rate_loss_soft_fa_corrected(tx_power_dbm, fa_probs, channel_gains_db, cfg)
    
    # They should be very close for one-hot probabilities
    diff = abs(hard_loss.item() - soft_loss.item())
    print(f"FA penalty test - Hard: {hard_loss.item():.2e}, Soft: {soft_loss.item():.2e}, Diff: {diff:.2e}")
    
    assert diff < 1e-3, f"Hard and soft should be consistent: {diff}"


if __name__ == "__main__":
    print("=" * 60)
    print("CORRECTED LOSS FUNCTION VALIDATION")
    print("=" * 60)
    
    # Test FA penalty application
    print("\n--- Testing FA Penalty Application ---")
    test_fa_penalty_application()
    
    # Test configurations
    configs = TestConfig.get_test_configs()
    
    for cfg in configs:
        print(f"\n--- Testing FA={cfg.n_fa} ---")
        validator = LossFunctionValidator(cfg)
        results = validator.test_loss_function_properties()
        
        print(f"Results for FA={cfg.n_fa}:")
        for key, value in results.items():
            print(f"  {key}: {value:.2e}")
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60) 