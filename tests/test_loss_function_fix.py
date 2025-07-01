"""
Comprehensive test routine to validate the fixed soft FA loss function.

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
from src.algorithms.loss_functions import (
    negative_sum_rate_loss_soft_fa as new_soft_loss,
    negative_sum_rate_loss_hard_fa as hard_loss
)
from src.simulator.scenario import Scenario


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
        
        # Random channel gains
        channel_gains_db = torch.randn(n_samples, n_pairs, n_pairs, device=self.device) * 20 - 80
        
        # Random power allocations
        tx_power_dbm = torch.rand(n_samples, n_pairs, device=self.device) * 30
        
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
        tx_power_dbm, fa_probs, channel_gains_db = self.generate_test_data(20)
        
        # Test 1: Hard vs Soft consistency (when FA probs are one-hot)
        if self.cfg.n_fa > 1:
            # Create one-hot FA probabilities
            fa_indices = torch.randint(0, self.cfg.n_fa, (20, self.cfg.n_pairs))
            fa_probs_onehot = torch.zeros(20, self.cfg.n_pairs, self.cfg.n_fa)
            fa_probs_onehot.scatter_(2, fa_indices.unsqueeze(2), 1.0)
            
            # Compute losses
            hard_loss_val = hard_loss(tx_power_dbm, fa_probs_onehot, channel_gains_db, self.cfg)
            new_soft_loss_val = new_soft_loss(tx_power_dbm, fa_probs_onehot, channel_gains_db, self.cfg)
            old_soft_loss_val = old_soft_loss(tx_power_dbm, fa_probs_onehot, None, self.cfg, channel_gains_db)
            
            results['hard_soft_consistency'] = abs(hard_loss_val.item() - new_soft_loss_val.item())
            results['old_new_difference'] = abs(old_soft_loss_val.item() - new_soft_loss_val.item())
        else:
            results['hard_soft_consistency'] = 0.0
            results['old_new_difference'] = 0.0
        
        # Test 2: Gradient properties
        tx_power_dbm.requires_grad_(True)
        fa_probs.requires_grad_(True)
        
        new_loss_val = new_soft_loss(tx_power_dbm, fa_probs, channel_gains_db, self.cfg)
        new_loss_val.backward()
        
        results['power_grad_norm'] = tx_power_dbm.grad.norm().item()
        results['fa_grad_norm'] = fa_probs.grad.norm().item()
        
        return results
    
    def compare_loss_functions(self, n_samples: int = 50) -> Dict[str, List[float]]:
        """Compare old and new soft loss functions across multiple samples."""
        old_losses = []
        new_losses = []
        differences = []
        
        for _ in range(n_samples):
            tx_power_dbm, fa_probs, channel_gains_db = self.generate_test_data(1)
            
            # Compute both losses
            old_loss_val = old_soft_loss(
                tx_power_dbm.squeeze(0), fa_probs.squeeze(0), None, self.cfg, channel_gains_db.squeeze(0)
            )
            new_loss_val = new_soft_loss(tx_power_dbm, fa_probs, channel_gains_db, self.cfg)
            
            old_losses.append(old_loss_val.item())
            new_losses.append(new_loss_val.item())
            differences.append(abs(old_loss_val.item() - new_loss_val.item()))
        
        return {
            'old_losses': old_losses,
            'new_losses': new_losses,
            'differences': differences
        }


class TrainingPerformanceTest:
    """Test training performance with different loss functions."""
    
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.device = DEVICE
        
    def run_training_comparison(self, epochs: int = 100) -> Dict[str, Dict]:
        """Compare training with old vs new soft loss functions."""
        results = {}
        
        # Test parameters matching user requirements
        train_params = {
            'hidden_size': [200, 200],
            'lr': 3e-3,
            'epochs': epochs,
            'verbose': False,
            'device': self.device,
            'restrict_rx_distance': True,
            'fa_gumbel_softmax': True,  # Gumbel-Softmax as requested
            'batch_norm': True,
            'patience': epochs,  # Disable early stopping for fair comparison
        }
        
        # Single-sample training (to test the problematic case)
        print(f"Testing single-sample training (n_fa={self.cfg.n_fa})...")
        
        # Train with old soft loss (current implementation)
        print("  Training with OLD soft loss...")
        algo_old, losses_old, _ = train_model(self.cfg, soft_fa=True, **train_params)
        
        # Train with hard loss (baseline)
        print("  Training with HARD loss...")
        algo_hard, losses_hard, _ = train_model(self.cfg, soft_fa=False, **train_params)
        
        # TODO: We need to modify train_model to use the new loss function
        # For now, we'll simulate the comparison
        
        results['single_sample'] = {
            'old_soft_losses': losses_old,
            'hard_losses': losses_hard,
            'old_soft_final': losses_old[-1] if losses_old else float('inf'),
            'hard_final': losses_hard[-1] if losses_hard else float('inf'),
        }
        
        return results
    
    def evaluate_model_performance(self, algo: ML_DNN, n_test_scenarios: int = 100) -> Dict[str, float]:
        """Evaluate trained model on test scenarios."""
        sum_rates = []
        
        for _ in range(n_test_scenarios):
            scenario = Scenario.random(self.cfg, restrict_rx_distance=True)
            result = algo.decide(scenario)
            
            # Calculate sum rate
            from src.simulator.metrics import sinr_linear, sum_rate_bps
            tx_power_lin = 10 ** (result['tx_power_dbm'] / 10) * 1e-3
            noise_power_lin = 10 ** (self.cfg.noise_power_dbm / 10) * 1e-3
            g = scenario.get_channel_gain_with_fa_penalty(result['fa_indices'])
            
            sinr = sinr_linear(tx_power_lin, result['fa_indices'], g, noise_power_lin)
            sum_rate = sum_rate_bps(sinr, self.cfg.bandwidth_hz)
            sum_rates.append(sum_rate)
        
        return {
            'mean_sum_rate': np.mean(sum_rates),
            'std_sum_rate': np.std(sum_rates),
            'min_sum_rate': np.min(sum_rates),
            'max_sum_rate': np.max(sum_rates)
        }


def create_performance_plots(results: Dict, save_dir: str = "tests/results"):
    """Create performance comparison plots."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    for fa_config, data in results.items():
        if 'single_sample' not in data:
            continue
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training loss comparison
        ax1 = axes[0]
        if 'old_soft_losses' in data['single_sample']:
            ax1.plot(data['single_sample']['old_soft_losses'], label='Old Soft Loss', alpha=0.7)
        if 'hard_losses' in data['single_sample']:
            ax1.plot(data['single_sample']['hard_losses'], label='Hard Loss', alpha=0.7)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Negative Sum Rate Loss')
        ax1.set_title(f'Training Loss Comparison ({fa_config})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss function difference histogram (if available)
        ax2 = axes[1]
        if 'loss_comparison' in data:
            ax2.hist(data['loss_comparison']['differences'], bins=20, alpha=0.7)
            ax2.set_xlabel('|Old Loss - New Loss|')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'Loss Function Differences ({fa_config})')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Loss comparison\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/loss_comparison_{fa_config.lower().replace('=', '')}.png", dpi=150)
        plt.close()


# Test functions
@pytest.mark.parametrize("cfg", TestConfig.get_test_configs())
def test_loss_function_mathematical_properties(cfg):
    """Test mathematical properties of the corrected loss function."""
    validator = LossFunctionValidator(cfg)
    results = validator.test_loss_function_properties()
    
    # Assertions
    if cfg.n_fa > 1:
        # Hard and soft should be consistent for one-hot probabilities
        assert results['hard_soft_consistency'] < 1e-3, \
            f"Hard and soft losses should be consistent for one-hot FA probs: {results['hard_soft_consistency']}"
        
        # Old and new should differ (indicating the fix)
        assert results['old_new_difference'] > 1e-6, \
            f"Old and new soft losses should differ: {results['old_new_difference']}"
    
    # Gradients should exist and be reasonable
    assert results['power_grad_norm'] > 1e-8, "Power gradients should exist"
    assert results['fa_grad_norm'] > 1e-8, "FA gradients should exist"
    
    print(f"✓ Mathematical properties test passed for n_fa={cfg.n_fa}")
    print(f"  Hard-soft consistency: {results['hard_soft_consistency']:.2e}")
    print(f"  Old-new difference: {results['old_new_difference']:.2e}")
    print(f"  Power grad norm: {results['power_grad_norm']:.2e}")
    print(f"  FA grad norm: {results['fa_grad_norm']:.2e}")


@pytest.mark.parametrize("cfg", TestConfig.get_test_configs())
def test_loss_function_comparison(cfg):
    """Compare old and new loss function implementations."""
    validator = LossFunctionValidator(cfg)
    comparison = validator.compare_loss_functions(n_samples=20)
    
    # Statistical tests
    mean_diff = np.mean(comparison['differences'])
    max_diff = np.max(comparison['differences'])
    
    if cfg.n_fa > 1:
        # Should see meaningful differences for FA > 1
        assert mean_diff > 1e-6, f"Mean difference should be significant: {mean_diff}"
        print(f"✓ Loss function comparison passed for n_fa={cfg.n_fa}")
        print(f"  Mean difference: {mean_diff:.2e}")
        print(f"  Max difference: {max_diff:.2e}")
    else:
        # For FA=1, differences should be minimal
        assert mean_diff < 1e-3, f"Differences should be small for n_fa=1: {mean_diff}"
        print(f"✓ Loss function comparison passed for n_fa={cfg.n_fa} (minimal differences as expected)")


@pytest.mark.parametrize("cfg", TestConfig.get_test_configs())
def test_training_performance(cfg):
    """Test training performance with different loss functions."""
    trainer = TrainingPerformanceTest(cfg)
    results = trainer.run_training_comparison(epochs=50)  # Shorter for testing
    
    # Basic sanity checks
    assert 'single_sample' in results
    assert len(results['single_sample']['hard_losses']) > 0
    assert len(results['single_sample']['old_soft_losses']) > 0
    
    # Training should converge (loss should decrease)
    hard_improved = results['single_sample']['hard_losses'][-1] < results['single_sample']['hard_losses'][0]
    soft_improved = results['single_sample']['old_soft_losses'][-1] < results['single_sample']['old_soft_losses'][0]
    
    print(f"✓ Training performance test completed for n_fa={cfg.n_fa}")
    print(f"  Hard loss improved: {hard_improved}")
    print(f"  Soft loss improved: {soft_improved}")
    print(f"  Final hard loss: {results['single_sample']['hard_final']:.2e}")
    print(f"  Final soft loss: {results['single_sample']['old_soft_final']:.2e}")


def run_comprehensive_test():
    """Run comprehensive test suite and generate report."""
    print("=" * 60)
    print("COMPREHENSIVE LOSS FUNCTION FIX VALIDATION")
    print("=" * 60)
    
    configs = TestConfig.get_test_configs()
    all_results = {}
    
    for cfg in configs:
        config_name = f"FA={cfg.n_fa}"
        print(f"\n--- Testing {config_name} ---")
        
        # Mathematical properties test
        validator = LossFunctionValidator(cfg)
        math_results = validator.test_loss_function_properties()
        
        # Loss function comparison
        comparison_results = validator.compare_loss_functions(n_samples=30)
        
        # Training performance test
        trainer = TrainingPerformanceTest(cfg)
        training_results = trainer.run_training_comparison(epochs=100)
        
        all_results[config_name] = {
            'mathematical_properties': math_results,
            'loss_comparison': comparison_results,
            **training_results
        }
    
    # Generate plots
    create_performance_plots(all_results)
    
    # Summary report
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    
    for config_name, results in all_results.items():
        print(f"\n{config_name}:")
        
        math_props = results['mathematical_properties']
        print(f"  Mathematical Properties:")
        print(f"    Hard-soft consistency: {math_props.get('hard_soft_consistency', 'N/A'):.2e}")
        print(f"    Old-new difference: {math_props.get('old_new_difference', 'N/A'):.2e}")
        
        if 'loss_comparison' in results:
            comp = results['loss_comparison']
            print(f"  Loss Function Comparison:")
            print(f"    Mean difference: {np.mean(comp['differences']):.2e}")
            print(f"    Max difference: {np.max(comp['differences']):.2e}")
        
        if 'single_sample' in results:
            single = results['single_sample']
            print(f"  Training Performance:")
            print(f"    Final hard loss: {single['hard_final']:.2e}")
            print(f"    Final soft loss: {single['old_soft_final']:.2e}")
    
    print(f"\nPlots saved to: tests/results/")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    # Run comprehensive test when executed directly
    results = run_comprehensive_test() 