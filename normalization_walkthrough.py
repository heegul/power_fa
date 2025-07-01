#!/usr/bin/env python3
"""
Single Sample Mode Normalization Walkthrough
This script demonstrates exactly how normalization works in our implementation.
"""

import numpy as np
import torch
from src.config import SimulationConfig
from src.simulator.scenario import Scenario

def demonstrate_single_sample_normalization():
    """Step-by-step walkthrough of single sample normalization"""
    
    print("=" * 60)
    print("SINGLE SAMPLE MODE NORMALIZATION WALKTHROUGH")
    print("=" * 60)
    
    # Setup
    cfg = SimulationConfig.from_yaml('cfgs/config_fa1.yaml')
    cfg.seed = 42
    device = 'cpu'
    
    print(f"Configuration: {cfg.n_pairs} pairs, {cfg.n_fa} FA")
    print(f"Power range: {cfg.tx_power_min_dbm} to {cfg.tx_power_max_dbm} dBm")
    print()
    
    # Step 1: Generate scenario
    print("STEP 1: Generate Scenario")
    print("-" * 30)
    scenario = Scenario.random(cfg, restrict_rx_distance=True)
    
    # Get raw channel gains
    channel_gains_raw = scenario.channel_gains_db()  # Shape: (n_pairs, n_pairs)
    print(f"Raw channel gains shape: {channel_gains_raw.shape}")
    print(f"Raw channel gains (dB):")
    print(channel_gains_raw)
    print()
    
    # Step 2: Flatten for DNN input
    print("STEP 2: Flatten Channel Gains for DNN Input")
    print("-" * 45)
    x_raw = torch.tensor(channel_gains_raw, dtype=torch.float32, device=device).flatten()
    print(f"Flattened shape: {x_raw.shape}")
    print(f"Flattened values: {x_raw}")
    print(f"Min value: {x_raw.min():.2f} dB")
    print(f"Max value: {x_raw.max():.2f} dB")
    print()
    
    # Step 3: Add batch dimension
    print("STEP 3: Add Batch Dimension")
    print("-" * 30)
    sample_flat = x_raw.unsqueeze(0)  # Shape: (1, n_pairs*n_pairs)
    print(f"With batch dimension: {sample_flat.shape}")
    print(f"Values: {sample_flat}")
    print()
    
    # Step 4: Compute per-sample normalization statistics
    print("STEP 4: Compute Per-Sample Normalization Statistics")
    print("-" * 55)
    input_mean = sample_flat.mean(dim=1, keepdim=True)  # Per-sample mean
    input_std = sample_flat.std(dim=1, keepdim=True) + 1e-8  # Per-sample std
    
    print(f"Input mean shape: {input_mean.shape}")
    print(f"Input mean value: {input_mean.item():.2f} dB")
    print(f"Input std shape: {input_std.shape}")
    print(f"Input std value: {input_std.item():.2f} dB")
    print()
    
    # Step 5: Apply normalization
    print("STEP 5: Apply Normalization")
    print("-" * 30)
    x_normalized = (sample_flat - input_mean) / input_std
    print(f"Normalized shape: {x_normalized.shape}")
    print(f"Normalized values: {x_normalized}")
    print(f"Normalized mean: {x_normalized.mean():.6f} (should be ~0)")
    print(f"Normalized std: {x_normalized.std():.6f} (should be ~1)")
    print(f"Normalized min: {x_normalized.min():.2f}")
    print(f"Normalized max: {x_normalized.max():.2f}")
    print()
    
    # Step 6: Show what happens during training
    print("STEP 6: Training Process")
    print("-" * 25)
    print("During training:")
    print("  1. DNN receives: x_normalized (mean=0, std=1)")
    print("  2. DNN outputs: power logits and FA logits")
    print("  3. Convert to power/FA decisions")
    print("  4. Compute SINR using ORIGINAL channel_gains_raw (NOT normalized!)")
    print("  5. Compute loss = -sum_rate")
    print("  6. Backpropagate through DNN")
    print()
    
    # Step 7: Show what happens during evaluation
    print("STEP 7: Evaluation Process")
    print("-" * 27)
    print("During evaluation:")
    print("  1. Apply SAME normalization as training:")
    print(f"     - Mean: {input_mean.item():.2f} dB")
    print(f"     - Std: {input_std.item():.2f} dB")
    print("  2. DNN receives normalized input")
    print("  3. DNN outputs power/FA decisions")
    print("  4. Evaluate performance using original channel gains")
    print()
    
    # Step 8: Compare with reference implementation
    print("STEP 8: Comparison with Reference Implementation")
    print("-" * 50)
    
    # Reference normalization (from compare_implementations.py)
    sample_ref = channel_gains_raw.flatten().reshape(1, -1)  # NumPy version
    mean_ref = sample_ref.mean(axis=1, keepdims=True)
    std_ref = sample_ref.std(axis=1, keepdims=True) + 1e-8
    sample_norm_ref = (sample_ref - mean_ref) / std_ref
    
    print("Reference implementation:")
    print(f"  Mean: {mean_ref.flatten()[0]:.2f} dB")
    print(f"  Std: {std_ref.flatten()[0]:.2f} dB")
    print(f"  Normalized range: {sample_norm_ref.min():.2f} to {sample_norm_ref.max():.2f}")
    print()
    
    print("Our implementation:")
    print(f"  Mean: {input_mean.item():.2f} dB")
    print(f"  Std: {input_std.item():.2f} dB")
    print(f"  Normalized range: {x_normalized.min():.2f} to {x_normalized.max():.2f}")
    print()
    
    # Check if they match
    mean_diff = abs(mean_ref.flatten()[0] - input_mean.item())
    std_diff = abs(std_ref.flatten()[0] - input_std.item())
    print(f"Differences:")
    print(f"  Mean difference: {mean_diff:.6f} dB")
    print(f"  Std difference: {std_diff:.6f} dB")
    print(f"  Match: {'✓' if mean_diff < 1e-5 and std_diff < 1e-5 else '✗'}")
    print()
    
    # Step 9: Show the key insight
    print("STEP 9: Key Insight - Why This Matters")
    print("-" * 40)
    print("🔑 CRITICAL POINT:")
    print("   - DNN input: NORMALIZED channel gains (mean=0, std=1)")
    print("   - Loss calculation: ORIGINAL channel gains (actual dB values)")
    print("   - This separation allows:")
    print("     * Stable DNN training (normalized inputs)")
    print("     * Accurate SINR computation (original gains)")
    print("     * Consistent evaluation (same normalization)")
    print()
    
    # Step 10: Show potential issues
    print("STEP 10: Potential Issues & Solutions")
    print("-" * 37)
    print("⚠️  Potential Issues:")
    print("   1. Per-sample normalization may not generalize well")
    print("   2. Different scenarios have different statistics")
    print("   3. Training/evaluation consistency is critical")
    print()
    print("✅ Solutions:")
    print("   1. Use global statistics from training set (for batch mode)")
    print("   2. Ensure same normalization in training and evaluation")
    print("   3. Always use original gains for loss/SINR calculation")
    print()
    
    return {
        'raw_gains': channel_gains_raw,
        'normalized_input': x_normalized,
        'mean': input_mean.item(),
        'std': input_std.item()
    }

def compare_normalization_strategies():
    """Compare different normalization strategies"""
    
    print("=" * 60)
    print("NORMALIZATION STRATEGY COMPARISON")
    print("=" * 60)
    
    cfg = SimulationConfig.from_yaml('cfgs/config_fa1.yaml')
    
    # Generate multiple scenarios
    scenarios = []
    for i in range(5):
        cfg.seed = 42 + i
        scenario = Scenario.random(cfg, restrict_rx_distance=True)
        scenarios.append(scenario.channel_gains_db().flatten())
    
    print(f"Generated {len(scenarios)} scenarios")
    print()
    
    # Strategy 1: Per-sample normalization
    print("STRATEGY 1: Per-Sample Normalization")
    print("-" * 40)
    for i, gains in enumerate(scenarios):
        mean = gains.mean()
        std = gains.std() + 1e-8
        normalized = (gains - mean) / std
        print(f"Scenario {i}: Mean={mean:.2f}, Std={std:.2f}, "
              f"Norm_range=[{normalized.min():.2f}, {normalized.max():.2f}]")
    print()
    
    # Strategy 2: Global normalization
    print("STRATEGY 2: Global Normalization")
    print("-" * 35)
    all_gains = np.concatenate(scenarios)
    global_mean = all_gains.mean()
    global_std = all_gains.std() + 1e-8
    print(f"Global statistics: Mean={global_mean:.2f}, Std={global_std:.2f}")
    
    for i, gains in enumerate(scenarios):
        normalized = (gains - global_mean) / global_std
        print(f"Scenario {i}: Norm_range=[{normalized.min():.2f}, {normalized.max():.2f}]")
    print()
    
    # Strategy 3: Scenario-type-specific (our previous approach)
    print("STRATEGY 3: Scenario-Type-Specific")
    print("-" * 38)
    print("Restricted scenarios: Mean=-87.14, Std=15.98")
    print("Normal scenarios: Mean=-92.86, Std=9.75")
    
    for i, gains in enumerate(scenarios):
        # Assuming all are restricted scenarios
        normalized = (gains - (-87.14)) / 15.98
        print(f"Scenario {i}: Norm_range=[{normalized.min():.2f}, {normalized.max():.2f}]")
    print()
    
    print("📊 ANALYSIS:")
    print("   - Per-sample: Each scenario normalized to mean=0, std=1")
    print("   - Global: Consistent normalization across all scenarios")
    print("   - Type-specific: Different norms for restricted vs normal")
    print()
    print("🎯 RECOMMENDATION:")
    print("   - Single-sample mode: Use per-sample normalization")
    print("   - Batch mode: Use global normalization from training set")
    print("   - Always use original gains for loss calculation")

if __name__ == "__main__":
    # Run the walkthrough
    result = demonstrate_single_sample_normalization()
    
    print("\n" + "=" * 60)
    compare_normalization_strategies()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ Single sample normalization works by:")
    print("   1. Flattening channel gains matrix to vector")
    print("   2. Computing per-sample mean and std")
    print("   3. Normalizing: (x - mean) / std")
    print("   4. Using normalized input for DNN")
    print("   5. Using original gains for loss/SINR calculation")
    print()
    print("🔧 This approach ensures:")
    print("   - Stable DNN training (normalized inputs)")
    print("   - Accurate performance evaluation (original gains)")
    print("   - Consistency between training and evaluation") 