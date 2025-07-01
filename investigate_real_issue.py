#!/usr/bin/env python3
"""
Investigate the REAL cause of performance difference in restricted scenarios
Since power range is NOT the issue, what is?
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.config import SimulationConfig
from src.simulator.scenario import Scenario
from src.algorithms.fs_bruteforce import fs_bruteforce
from src.simulator.metrics import sinr_linear, sum_rate_bps
from src.simulator.environment import db_to_linear

def investigate_real_issue():
    """Find the real cause of performance difference"""
    
    print("=" * 80)
    print("INVESTIGATING THE REAL CAUSE OF PERFORMANCE DIFFERENCE")
    print("=" * 80)
    
    print("🔍 KEY FINDING: Power range is NOT the issue!")
    print("   - All power ranges achieve ~1.005 ratio (essentially optimal)")
    print("   - But our main DNN achieves only ~0.65 ratio on restricted scenarios")
    print("   - Something else is causing the performance gap")
    
    # Setup
    cfg = SimulationConfig.from_yaml('cfgs/config_fa1.yaml')
    
    print("\n🧪 HYPOTHESIS TESTING:")
    print("-" * 25)
    
    # Test multiple restricted scenarios to see if it's scenario-specific
    print("\n1. TESTING MULTIPLE RESTRICTED SCENARIOS")
    print("   " + "-" * 35)
    
    ratios = []
    for seed in [42, 100, 200, 300, 400]:
        cfg.seed = seed
        scenario = Scenario.random(cfg, restrict_rx_distance=True)
        
        # Get optimal solution
        fs_solver = fs_bruteforce(cfg)
        optimal_result = fs_solver.decide(scenario)
        noise_lin = db_to_linear(cfg.noise_power_dbm) * 1e-3
        bandwidth = cfg.bandwidth_hz
        tx_power_lin = db_to_linear(optimal_result['tx_power_dbm']) * 1e-3
        g = scenario.get_channel_gain_with_fa_penalty(optimal_result['fa_indices'])
        sinr = sinr_linear(tx_power_lin, optimal_result['fa_indices'], g, noise_lin)
        optimal_sum_rate = sum_rate_bps(sinr, bandwidth)
        
        # Test simple DNN (like our hypothesis test)
        simple_result = test_simple_dnn(scenario, cfg, epochs=500)
        ratio = simple_result['sum_rate'] / optimal_sum_rate
        ratios.append(ratio)
        
        print(f"  Seed {seed}: Simple DNN ratio = {ratio:.3f}")
    
    avg_ratio = np.mean(ratios)
    print(f"  Average ratio: {avg_ratio:.3f}")
    
    if avg_ratio > 0.95:
        print("  ✅ Simple DNN works well on restricted scenarios!")
        print("  ❓ Problem must be in our main DNN implementation")
    else:
        print("  ❌ Simple DNN also struggles with restricted scenarios")
        print("  ❓ Problem is fundamental to restricted scenarios")
    
    # Test 2: Compare with normal scenarios
    print("\n2. TESTING NORMAL SCENARIOS FOR COMPARISON")
    print("   " + "-" * 38)
    
    normal_ratios = []
    for seed in [42, 100, 200, 300, 400]:
        cfg.seed = seed
        scenario = Scenario.random(cfg, restrict_rx_distance=False)
        
        # Get optimal solution
        fs_solver = fs_bruteforce(cfg)
        optimal_result = fs_solver.decide(scenario)
        noise_lin = db_to_linear(cfg.noise_power_dbm) * 1e-3
        bandwidth = cfg.bandwidth_hz
        tx_power_lin = db_to_linear(optimal_result['tx_power_dbm']) * 1e-3
        g = scenario.get_channel_gain_with_fa_penalty(optimal_result['fa_indices'])
        sinr = sinr_linear(tx_power_lin, optimal_result['fa_indices'], g, noise_lin)
        optimal_sum_rate = sum_rate_bps(sinr, bandwidth)
        
        # Test simple DNN
        simple_result = test_simple_dnn(scenario, cfg, epochs=500)
        ratio = simple_result['sum_rate'] / optimal_sum_rate
        normal_ratios.append(ratio)
        
        print(f"  Seed {seed}: Simple DNN ratio = {ratio:.3f}")
    
    avg_normal_ratio = np.mean(normal_ratios)
    print(f"  Average normal ratio: {avg_normal_ratio:.3f}")
    
    # Test 3: Analyze the differences
    print("\n3. ANALYZING KEY DIFFERENCES")
    print("   " + "-" * 26)
    
    print(f"Performance comparison:")
    print(f"  Normal scenarios:     {avg_normal_ratio:.3f} average ratio")
    print(f"  Restricted scenarios: {avg_ratio:.3f} average ratio")
    print(f"  Performance drop:     {(avg_normal_ratio - avg_ratio) / avg_normal_ratio * 100:.1f}%")
    
    # Test 4: Check if it's a training issue
    print("\n4. TRAINING DURATION ANALYSIS")
    print("   " + "-" * 26)
    
    cfg.seed = 42
    restricted_scenario = Scenario.random(cfg, restrict_rx_distance=True)
    
    # Test different training durations
    for epochs in [100, 500, 1000, 2000]:
        result = test_simple_dnn(restricted_scenario, cfg, epochs=epochs)
        
        # Get optimal for comparison
        fs_solver = fs_bruteforce(cfg)
        optimal_result = fs_solver.decide(restricted_scenario)
        noise_lin = db_to_linear(cfg.noise_power_dbm) * 1e-3
        bandwidth = cfg.bandwidth_hz
        tx_power_lin = db_to_linear(optimal_result['tx_power_dbm']) * 1e-3
        g = restricted_scenario.get_channel_gain_with_fa_penalty(optimal_result['fa_indices'])
        sinr = sinr_linear(tx_power_lin, optimal_result['fa_indices'], g, noise_lin)
        optimal_sum_rate = sum_rate_bps(sinr, bandwidth)
        
        ratio = result['sum_rate'] / optimal_sum_rate
        print(f"  {epochs:4d} epochs: ratio = {ratio:.3f}")
    
    # Test 5: Architecture comparison
    print("\n5. ARCHITECTURE COMPARISON")
    print("   " + "-" * 23)
    
    print("Simple DNN (this test):")
    print("  - Architecture: 36 → 128 → 128 → 6 (power) + 36 → 128 → 128 → 18 (FA)")
    print("  - Activation: ReLU + Sigmoid/Softmax")
    print("  - Training: 500-1000 epochs, Adam optimizer")
    print("  - Normalization: Per-sample mean/std")
    
    print("\nOur main DNN:")
    print("  - Architecture: Configurable hidden layers")
    print("  - Activation: ReLU + Sigmoid/Softmax")
    print("  - Training: Up to 5000 epochs with early stopping")
    print("  - Normalization: Per-sample mean/std")
    print("  - Additional features: Batch normalization, validation")
    
    return {
        'restricted_ratios': ratios,
        'normal_ratios': normal_ratios,
        'avg_restricted': avg_ratio,
        'avg_normal': avg_normal_ratio
    }

def test_simple_dnn(scenario, cfg, epochs=500):
    """Test simple DNN on a scenario"""
    
    # Prepare input
    gains_db = scenario.channel_gains_db()
    gains_linear = db_to_linear(gains_db)
    
    # Normalize input (per-sample normalization)
    x_raw = torch.tensor(gains_db.flatten(), dtype=torch.float32).unsqueeze(0)
    x_mean = x_raw.mean()
    x_std = x_raw.std() + 1e-8
    x_normalized = (x_raw - x_mean) / x_std
    
    # Create model
    class SimpleDNN(nn.Module):
        def __init__(self, input_size, n_pairs=6, n_fa=3):
            super().__init__()
            self.n_pairs = n_pairs
            self.n_fa = n_fa
            
            self.power_net = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, n_pairs),
                nn.Sigmoid()
            )
            
            self.fa_net = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, n_pairs * n_fa)
            )
        
        def forward(self, x):
            power_logits = self.power_net(x)
            fa_logits = self.fa_net(x).view(-1, self.n_pairs, self.n_fa)
            fa_probs = torch.softmax(fa_logits, dim=-1)
            return power_logits, fa_probs
    
    model = SimpleDNN(input_size=cfg.n_pairs * cfg.n_pairs, n_pairs=cfg.n_pairs, n_fa=cfg.n_fa)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Power and noise settings
    min_power_dbm = cfg.tx_power_min_dbm
    max_power_dbm = cfg.tx_power_max_dbm
    min_power_w = 10 ** ((min_power_dbm - 30) / 10)
    max_power_w = 10 ** ((max_power_dbm - 30) / 10)
    noise_power_dbm = cfg.noise_power_dbm
    noise_power_w = 10 ** ((noise_power_dbm - 30) / 10)
    
    best_sum_rate = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        power_logits, fa_probs = model(x_normalized)
        
        # Scale powers
        powers_w = min_power_w + power_logits.squeeze() * (max_power_w - min_power_w)
        
        # Calculate SINR using original gains
        gains_tensor = torch.tensor(gains_linear, dtype=torch.float32)
        fa_probs_2d = fa_probs.squeeze()
        if fa_probs_2d.dim() == 1:
            fa_probs_2d = fa_probs_2d.view(cfg.n_pairs, cfg.n_fa)
        
        sinr = compute_sinr_tensor(powers_w, fa_probs_2d, gains_tensor, noise_power_w)
        
        # Loss: negative sum-rate
        sum_rate = torch.sum(torch.log2(1.0 + sinr)) * cfg.bandwidth_hz
        loss = -sum_rate
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track best result
        if sum_rate.item() > best_sum_rate:
            best_sum_rate = sum_rate.item()
    
    return {'sum_rate': best_sum_rate}

def compute_sinr_tensor(powers, fa_probs, gains_linear, noise_power):
    """Compute SINR using PyTorch tensors"""
    n_pairs, n_fa = fa_probs.shape
    sinr = torch.zeros(n_pairs)
    
    for i in range(n_pairs):
        # Signal: own power * own channel gain * frequency allocation
        signal_power = powers[i] * gains_linear[i, i]
        signal = signal_power * fa_probs[i, :]
        
        # Interference: sum of other transmitters on same frequencies
        interference = torch.ones(n_fa) * noise_power
        for j in range(n_pairs):
            if j != i:
                interference += powers[j] * gains_linear[j, i] * fa_probs[j, :]
        
        # SINR for each frequency, weighted by allocation probability
        sinr_per_freq = signal / interference
        sinr[i] = torch.sum(fa_probs[i, :] * sinr_per_freq)
    
    return sinr

if __name__ == "__main__":
    results = investigate_real_issue()
    
    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    
    if results['avg_restricted'] > 0.95:
        print("\n🎯 MAIN FINDING:")
        print("   Simple DNN achieves excellent performance on restricted scenarios!")
        print(f"   Average ratio: {results['avg_restricted']:.3f}")
        print("\n❓ IMPLICATION:")
        print("   The problem is NOT with restricted scenarios themselves")
        print("   The problem is in our MAIN DNN implementation")
        print("\n🔍 LIKELY CAUSES:")
        print("   1. Training hyperparameters (learning rate, epochs)")
        print("   2. Model architecture differences")
        print("   3. Initialization or optimization issues")
        print("   4. Early stopping criteria")
        print("   5. Batch normalization effects")
    else:
        print("\n🎯 MAIN FINDING:")
        print("   Even simple DNN struggles with restricted scenarios")
        print(f"   Average ratio: {results['avg_restricted']:.3f}")
        print("\n❓ IMPLICATION:")
        print("   Restricted scenarios are fundamentally harder")
        print("   But reference implementation still achieves 1.005 ratio")
        print("\n🔍 LIKELY CAUSES:")
        print("   1. Different SINR calculation method")
        print("   2. Different loss function formulation")
        print("   3. Different training strategy")
        print("   4. Numerical precision issues")
    
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"   Normal scenarios:     {results['avg_normal']:.3f} ratio")
    print(f"   Restricted scenarios: {results['avg_restricted']:.3f} ratio")
    print(f"   Performance drop:     {(results['avg_normal'] - results['avg_restricted']) / results['avg_normal'] * 100:.1f}%") 