#!/usr/bin/env python3
"""
Demonstration of separated cost functions for FA > 1 case.

This script demonstrates the difference between training loss functions
and validation functions, showing how they are used in practice.
"""

import torch
import numpy as np
import argparse
from pathlib import Path

# Import the separated modules
from src.algorithms.loss_functions import (
    negative_sum_rate_loss_hard_fa,
    negative_sum_rate_loss_soft_fa,
    negative_sum_rate_loss_adaptive,
    get_loss_function
)
from src.algorithms.validation_functions import (
    evaluate_sum_rate,
    validate_model_performance,
    compare_with_baseline,
    validate_multiple_scenarios
)
from src.algorithms.ml_dnn import create_dnn_model, dnn_output_to_decision_torch
from src.config import load_config
from src.simulator.scenario import Scenario


def demonstrate_loss_functions(cfg):
    """Demonstrate different training loss functions."""
    print("=" * 60)
    print("TRAINING LOSS FUNCTIONS DEMONSTRATION")
    print("=" * 60)
    
    # Create a sample scenario
    scenario = Scenario(cfg)
    
    # Create sample inputs
    batch_size = 4
    n_pairs = cfg.n_pairs
    n_fa = cfg.n_fa
    
    # Sample power and FA probabilities
    tx_power_dbm = torch.randn(batch_size, n_pairs) * 5 + 20  # Around 20 dBm
    fa_probs = torch.softmax(torch.randn(batch_size, n_pairs, n_fa), dim=2)
    
    # Channel gains (same for all batch samples for simplicity)
    channel_gains_db = torch.tensor(scenario.channel_gains_db(), dtype=torch.float32)
    channel_gains_db = channel_gains_db.unsqueeze(0).repeat(batch_size, 1, 1)
    
    print(f"Input shapes:")
    print(f"  tx_power_dbm: {tx_power_dbm.shape}")
    print(f"  fa_probs: {fa_probs.shape}")
    print(f"  channel_gains_db: {channel_gains_db.shape}")
    print()
    
    # Test different loss functions
    print("Testing different loss functions:")
    
    # 1. Hard FA assignment
    print("1. Hard FA Assignment Loss:")
    loss_hard = negative_sum_rate_loss_hard_fa(tx_power_dbm, fa_probs, channel_gains_db, cfg)
    print(f"   Loss value: {loss_hard.item():.6f}")
    print(f"   Gradient computation: {'✓' if loss_hard.requires_grad else '✗'}")
    
    # 2. Soft FA assignment
    print("\n2. Soft FA Assignment Loss:")
    loss_soft = negative_sum_rate_loss_soft_fa(tx_power_dbm, fa_probs, channel_gains_db, cfg)
    print(f"   Loss value: {loss_soft.item():.6f}")
    print(f"   Gradient computation: {'✓' if loss_soft.requires_grad else '✗'}")
    
    # 3. Adaptive loss
    print("\n3. Adaptive Loss (early epoch):")
    loss_adaptive_early = negative_sum_rate_loss_adaptive(
        tx_power_dbm, fa_probs, channel_gains_db, cfg, epoch=10, total_epochs=1000
    )
    print(f"   Loss value (epoch 10): {loss_adaptive_early.item():.6f}")
    
    print("\n4. Adaptive Loss (late epoch):")
    loss_adaptive_late = negative_sum_rate_loss_adaptive(
        tx_power_dbm, fa_probs, channel_gains_db, cfg, epoch=800, total_epochs=1000
    )
    print(f"   Loss value (epoch 800): {loss_adaptive_late.item():.6f}")
    
    # 5. Factory function
    print("\n5. Using Factory Function:")
    loss_fn = get_loss_function("hard")
    loss_factory = loss_fn(tx_power_dbm, fa_probs, channel_gains_db, cfg)
    print(f"   Factory loss value: {loss_factory.item():.6f}")
    
    print("\nComputational complexity comparison:")
    import time
    
    # Time hard assignment
    start = time.time()
    for _ in range(100):
        _ = negative_sum_rate_loss_hard_fa(tx_power_dbm, fa_probs, channel_gains_db, cfg)
    hard_time = time.time() - start
    
    # Time soft assignment
    start = time.time()
    for _ in range(100):
        _ = negative_sum_rate_loss_soft_fa(tx_power_dbm, fa_probs, channel_gains_db, cfg)
    soft_time = time.time() - start
    
    print(f"  Hard assignment: {hard_time:.4f}s (100 iterations)")
    print(f"  Soft assignment: {soft_time:.4f}s (100 iterations)")
    print(f"  Soft/Hard ratio: {soft_time/hard_time:.2f}x slower")


def demonstrate_validation_functions(cfg):
    """Demonstrate validation functions."""
    print("\n" + "=" * 60)
    print("VALIDATION FUNCTIONS DEMONSTRATION")
    print("=" * 60)
    
    # Create a sample scenario
    scenario = Scenario(cfg)
    
    # Create sample allocation decisions (discrete)
    n_pairs = cfg.n_pairs
    tx_power_dbm = np.random.uniform(cfg.tx_power_min_dbm, cfg.tx_power_max_dbm, n_pairs)
    fa_indices = np.random.randint(0, cfg.n_fa, n_pairs)
    
    print(f"Sample allocation decisions:")
    print(f"  Power (dBm): {tx_power_dbm}")
    print(f"  FA indices: {fa_indices}")
    print()
    
    # 1. Basic sum rate evaluation
    print("1. Basic Sum Rate Evaluation:")
    results = evaluate_sum_rate(tx_power_dbm, fa_indices, scenario.channel_gains_db(), cfg)
    
    print(f"   Sum Rate: {results['sum_rate']:.2e} bit/s")
    print(f"   SINR (dB): {10 * np.log10(results['sinr'])}")
    print(f"   Individual Rates: {results['individual_rates']}")
    print(f"   FA Distribution: {np.bincount(fa_indices, minlength=cfg.n_fa)}")
    
    # 2. Create a simple model for demonstration
    print("\n2. Model Performance Validation:")
    
    # Create a dummy model
    input_size = n_pairs * n_pairs
    output_size = n_pairs + n_pairs * cfg.n_fa  # power + FA probs
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, output_size)
    )
    
    # Validate model performance
    model_results = validate_model_performance(model, scenario, cfg)
    
    print(f"   Model Sum Rate: {model_results['sum_rate']:.2e} bit/s")
    print(f"   Model Fairness Index: {model_results['fairness_index']:.3f}")
    print(f"   Model Power Efficiency: {model_results['power_efficiency']:.2f} mW")
    print(f"   Model SINR Range: {model_results['min_sinr_db']:.1f} to {model_results['max_sinr_db']:.1f} dB")
    
    # 3. Baseline comparison
    print("\n3. Baseline Comparison:")
    
    # Create a "baseline" result (random allocation)
    baseline_power = np.random.uniform(cfg.tx_power_min_dbm, cfg.tx_power_max_dbm, n_pairs)
    baseline_fa = np.random.randint(0, cfg.n_fa, n_pairs)
    baseline_results = evaluate_sum_rate(baseline_power, baseline_fa, scenario.channel_gains_db(), cfg)
    baseline_results.update({
        'fairness_index': np.random.uniform(0.3, 0.7),  # Dummy fairness
        'power_efficiency': np.mean(10 ** (baseline_power / 10))
    })
    
    comparison = compare_with_baseline(model_results, baseline_results, "Random Allocation")
    
    print(f"   Sum Rate Ratio: {comparison['sum_rate_ratio']:.3f}")
    print(f"   Sum Rate Improvement: {comparison['sum_rate_improvement_percent']:.1f}%")
    print(f"   Model Better Sum Rate: {'✓' if comparison['model_better_sum_rate'] else '✗'}")
    print(f"   Model Better Fairness: {'✓' if comparison['model_better_fairness'] else '✗'}")
    print(f"   Model More Efficient: {'✓' if comparison['model_more_efficient'] else '✗'}")


def demonstrate_training_vs_validation():
    """Demonstrate the key differences between training and validation."""
    print("\n" + "=" * 60)
    print("TRAINING vs VALIDATION COMPARISON")
    print("=" * 60)
    
    print("Key Differences:")
    print()
    
    print("TRAINING LOSS FUNCTIONS:")
    print("  ✓ Fully differentiable for gradient computation")
    print("  ✓ Handle soft FA assignments (probabilities)")
    print("  ✓ Vectorized for batch processing")
    print("  ✓ Return negative sum rate (for minimization)")
    print("  ✓ May use approximations for computational efficiency")
    print("  ✓ Focus on optimization convergence")
    print()
    
    print("VALIDATION FUNCTIONS:")
    print("  ✓ Use discrete FA assignments (interpretable)")
    print("  ✓ Provide detailed performance breakdown")
    print("  ✓ Include additional metrics (fairness, efficiency)")
    print("  ✓ Use exact SINR calculations")
    print("  ✓ Return positive sum rate values")
    print("  ✓ Focus on performance evaluation and analysis")
    print()
    
    print("USAGE PATTERNS:")
    print("  Training: loss.backward() → optimizer.step()")
    print("  Validation: model.eval() → performance analysis")
    print()
    
    print("COMPUTATIONAL FOCUS:")
    print("  Training: Speed and gradient quality")
    print("  Validation: Accuracy and interpretability")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Demonstrate cost function separation")
    parser.add_argument("--config", default="cfgs/config_fa1.yaml", help="Config file path")
    parser.add_argument("--n-fa", type=int, default=3, help="Number of frequency allocations")
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Override n_fa for demonstration
    if args.n_fa > 1:
        cfg.n_fa = args.n_fa
        print(f"Using n_fa = {cfg.n_fa} for demonstration")
    
    print(f"Configuration:")
    print(f"  n_pairs: {cfg.n_pairs}")
    print(f"  n_fa: {cfg.n_fa}")
    print(f"  fa_penalty_db: {cfg.fa_penalty_db}")
    print(f"  bandwidth_hz: {cfg.bandwidth_hz}")
    print(f"  noise_power_dbm: {cfg.noise_power_dbm}")
    
    # Run demonstrations
    demonstrate_loss_functions(cfg)
    demonstrate_validation_functions(cfg)
    demonstrate_training_vs_validation()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nFor more details, see:")
    print("  - COST_FUNCTION_ELABORATION.md")
    print("  - src/algorithms/loss_functions.py")
    print("  - src/algorithms/validation_functions.py")


if __name__ == "__main__":
    main() 