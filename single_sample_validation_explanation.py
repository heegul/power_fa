#!/usr/bin/env python3
"""
Single Sample Mode Validation Explanation
This script demonstrates exactly how validation works in single sample mode.
"""

import numpy as np
import torch
from src.config import SimulationConfig
from src.simulator.scenario import Scenario

def demonstrate_single_sample_validation():
    """Show exactly how validation works in single sample mode"""
    
    print("=" * 70)
    print("SINGLE SAMPLE MODE VALIDATION EXPLANATION")
    print("=" * 70)
    
    # Setup
    cfg = SimulationConfig.from_yaml('cfgs/config_fa1.yaml')
    cfg.seed = 42
    device = 'cpu'
    
    print("🔍 STEP-BY-STEP WALKTHROUGH:")
    print()
    
    # Step 1: Generate training scenario
    print("STEP 1: Generate Training Scenario")
    print("-" * 35)
    scenario = Scenario.random(cfg, restrict_rx_distance=True)
    print(f"Training scenario seed: {cfg.seed}")
    print(f"Channel gains shape: {scenario.channel_gains_db().shape}")
    print(f"Sample channel gains:")
    print(scenario.channel_gains_db())
    print()
    
    # Step 2: Set validation scenario
    print("STEP 2: Set Validation Scenario")
    print("-" * 32)
    print("🚨 CRITICAL LINE FROM CODE:")
    print("   val_scenario = scenario  # Use same scenario for validation")
    print()
    val_scenario = scenario  # This is exactly what the code does!
    
    print(f"Training scenario ID: {id(scenario)}")
    print(f"Validation scenario ID: {id(val_scenario)}")
    print(f"Are they the same object? {scenario is val_scenario}")
    print()
    
    # Step 3: Show channel gains are identical
    print("STEP 3: Verify Channel Gains Are Identical")
    print("-" * 42)
    train_gains = scenario.channel_gains_db()
    val_gains = val_scenario.channel_gains_db()
    
    print("Training channel gains:")
    print(train_gains)
    print()
    print("Validation channel gains:")
    print(val_gains)
    print()
    print(f"Are gains identical? {np.array_equal(train_gains, val_gains)}")
    print(f"Max difference: {np.abs(train_gains - val_gains).max()}")
    print()
    
    # Step 4: Show normalization
    print("STEP 4: Normalization Process")
    print("-" * 30)
    
    # Training normalization
    x_raw = torch.tensor(scenario.channel_gains_db(), dtype=torch.float32, device=device).flatten()
    sample_flat = x_raw.unsqueeze(0)
    input_mean = sample_flat.mean(dim=1, keepdim=True)
    input_std = sample_flat.std(dim=1, keepdim=True) + 1e-8
    x_normalized = (sample_flat - input_mean) / input_std
    
    # Validation normalization (using SAME normalization stats)
    val_x_raw = torch.tensor(val_scenario.channel_gains_db(), dtype=torch.float32, device=device).flatten()
    val_x_normalized = (val_x_raw.unsqueeze(0) - input_mean) / input_std
    
    print(f"Training input mean: {input_mean.item():.6f}")
    print(f"Training input std: {input_std.item():.6f}")
    print()
    print(f"Training normalized input: {x_normalized}")
    print(f"Validation normalized input: {val_x_normalized}")
    print()
    print(f"Are normalized inputs identical? {torch.allclose(x_normalized, val_x_normalized)}")
    print(f"Max difference in normalized inputs: {torch.abs(x_normalized - val_x_normalized).max().item()}")
    print()
    
    # Step 5: Show what happens during training
    print("STEP 5: Training Loop Process")
    print("-" * 28)
    print("In each epoch:")
    print("  1. 🏋️ TRAINING:")
    print("     - Input: x_normalized (from training scenario)")
    print("     - DNN forward pass")
    print("     - Compute loss using original_gains (from training scenario)")
    print("     - Backpropagation and weight update")
    print()
    print("  2. 📊 VALIDATION:")
    print("     - Input: val_x_normalized (from validation scenario)")
    print("     - DNN forward pass (no gradients)")
    print("     - Compute loss using original_gains (from training scenario)")
    print("     - Track validation loss for early stopping")
    print()
    
    # Step 6: Key insight
    print("STEP 6: Key Insight")
    print("-" * 18)
    print("🔑 CRITICAL UNDERSTANDING:")
    print("   - Training and validation use THE EXACT SAME SCENARIO")
    print("   - Same channel gains, same normalization, same loss calculation")
    print("   - Validation loss = Training loss (they should be identical!)")
    print()
    print("❓ WHY DO WE SEE DIFFERENT TRAIN/VAL LOSSES IN OUTPUT?")
    print("   - Training loss: computed during model.train() mode")
    print("   - Validation loss: computed during model.eval() mode")
    print("   - Difference comes from BatchNorm behavior (if enabled)")
    print("   - In eval mode, BatchNorm uses running statistics")
    print("   - In train mode, BatchNorm uses batch statistics")
    print()
    
    # Step 7: Demonstrate the issue
    print("STEP 7: Why This Might Be Problematic")
    print("-" * 37)
    print("⚠️  POTENTIAL ISSUES:")
    print("   1. NO REAL VALIDATION: We're testing on the same data we train on")
    print("   2. OVERFITTING RISK: Model can memorize the single scenario")
    print("   3. NO GENERALIZATION TEST: Can't assess performance on unseen data")
    print("   4. MISLEADING METRICS: Validation loss doesn't indicate true performance")
    print()
    print("✅ WHAT SHOULD WE DO INSTEAD?")
    print("   1. Generate a DIFFERENT scenario for validation")
    print("   2. Use same restrict_rx_distance setting but different seed")
    print("   3. Apply same normalization strategy to validation scenario")
    print("   4. This would give true validation performance")
    print()
    
    return {
        'scenario_id': id(scenario),
        'val_scenario_id': id(val_scenario),
        'same_object': scenario is val_scenario,
        'gains_identical': np.array_equal(train_gains, val_gains),
        'normalized_identical': torch.allclose(x_normalized, val_x_normalized)
    }

def show_improved_validation_approach():
    """Show how validation SHOULD work in single sample mode"""
    
    print("=" * 70)
    print("IMPROVED SINGLE SAMPLE VALIDATION APPROACH")
    print("=" * 70)
    
    cfg = SimulationConfig.from_yaml('cfgs/config_fa1.yaml')
    device = 'cpu'
    
    print("🔧 IMPROVED APPROACH:")
    print()
    
    # Step 1: Generate training scenario
    print("STEP 1: Generate Training Scenario")
    print("-" * 35)
    cfg.seed = 42
    train_scenario = Scenario.random(cfg, restrict_rx_distance=True)
    print(f"Training scenario seed: {cfg.seed}")
    print(f"Training gains diagonal: {np.diag(train_scenario.channel_gains_db())}")
    print()
    
    # Step 2: Generate DIFFERENT validation scenario
    print("STEP 2: Generate DIFFERENT Validation Scenario")
    print("-" * 47)
    cfg.seed = 42 + 1000  # Different seed!
    val_scenario = Scenario.random(cfg, restrict_rx_distance=True)
    print(f"Validation scenario seed: {cfg.seed}")
    print(f"Validation gains diagonal: {np.diag(val_scenario.channel_gains_db())}")
    print()
    
    # Step 3: Show they are different
    print("STEP 3: Verify Scenarios Are Different")
    print("-" * 38)
    train_gains = train_scenario.channel_gains_db()
    val_gains = val_scenario.channel_gains_db()
    
    print(f"Training scenario ID: {id(train_scenario)}")
    print(f"Validation scenario ID: {id(val_scenario)}")
    print(f"Are they the same object? {train_scenario is val_scenario}")
    print(f"Are gains identical? {np.array_equal(train_gains, val_gains)}")
    print(f"Max difference in gains: {np.abs(train_gains - val_gains).max():.2f} dB")
    print()
    
    # Step 4: Show normalization approach
    print("STEP 4: Normalization Strategy")
    print("-" * 30)
    
    # Option A: Per-sample normalization (current approach)
    print("OPTION A: Per-Sample Normalization (Current)")
    train_flat = torch.tensor(train_gains, dtype=torch.float32).flatten().unsqueeze(0)
    train_mean = train_flat.mean(dim=1, keepdim=True)
    train_std = train_flat.std(dim=1, keepdim=True) + 1e-8
    train_norm = (train_flat - train_mean) / train_std
    
    val_flat = torch.tensor(val_gains, dtype=torch.float32).flatten().unsqueeze(0)
    val_mean = val_flat.mean(dim=1, keepdim=True)
    val_std = val_flat.std(dim=1, keepdim=True) + 1e-8
    val_norm = (val_flat - val_mean) / val_std
    
    print(f"  Training: mean={train_mean.item():.2f}, std={train_std.item():.2f}")
    print(f"  Validation: mean={val_mean.item():.2f}, std={val_std.item():.2f}")
    print(f"  Issue: Different normalization for each scenario!")
    print()
    
    # Option B: Use training normalization for validation
    print("OPTION B: Use Training Normalization for Validation (Better)")
    val_norm_consistent = (val_flat - train_mean) / train_std
    print(f"  Training: mean={train_mean.item():.2f}, std={train_std.item():.2f}")
    print(f"  Validation: uses SAME normalization as training")
    print(f"  Validation normalized range: [{val_norm_consistent.min():.2f}, {val_norm_consistent.max():.2f}]")
    print()
    
    print("✅ RECOMMENDED IMPLEMENTATION:")
    print("```python")
    print("# Generate training scenario")
    print("train_scenario = Scenario.random(cfg, restrict_rx_distance=restrict_rx)")
    print()
    print("# Generate DIFFERENT validation scenario")
    print("cfg.seed = cfg.seed + 1000  # Different seed")
    print("val_scenario = Scenario.random(cfg, restrict_rx_distance=restrict_rx)")
    print()
    print("# Compute normalization from TRAINING scenario")
    print("train_flat = torch.tensor(train_scenario.channel_gains_db()).flatten().unsqueeze(0)")
    print("input_mean = train_flat.mean(dim=1, keepdim=True)")
    print("input_std = train_flat.std(dim=1, keepdim=True) + 1e-8")
    print()
    print("# Apply SAME normalization to both scenarios")
    print("train_norm = (train_flat - input_mean) / input_std")
    print("val_flat = torch.tensor(val_scenario.channel_gains_db()).flatten().unsqueeze(0)")
    print("val_norm = (val_flat - input_mean) / input_std  # Same normalization!")
    print("```")

if __name__ == "__main__":
    # Run the demonstration
    result = demonstrate_single_sample_validation()
    
    print("\n" + "=" * 70)
    show_improved_validation_approach()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("🔍 CURRENT IMPLEMENTATION:")
    print("   ✅ Uses exact same scenario for training and validation")
    print("   ✅ Same channel gains, same normalization")
    print("   ⚠️  No real validation - testing on training data")
    print()
    print("📊 WHAT WE OBSERVE:")
    print("   - Train and validation losses are nearly identical")
    print("   - Small differences due to BatchNorm train/eval modes")
    print("   - Early stopping based on 'validation' of same scenario")
    print()
    print("🎯 RECOMMENDATION:")
    print("   - Generate different validation scenario (different seed)")
    print("   - Use training scenario's normalization for validation")
    print("   - This would provide true generalization assessment")
    print()
    print("🤔 CURRENT BEHAVIOR IS ACTUALLY REASONABLE FOR:")
    print("   - Single scenario optimization (like our use case)")
    print("   - When we want to find optimal solution for specific scenario")
    print("   - Early stopping based on convergence rather than generalization") 