#!/usr/bin/env python3
"""
Test our hypothesis: Does increasing power dynamic range improve restricted scenario performance?
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

class TestDNN(nn.Module):
    """Simple DNN for testing power range hypothesis"""
    def __init__(self, input_size, hidden_size=128, n_pairs=6, n_fa=3):
        super().__init__()
        self.n_pairs = n_pairs
        self.n_fa = n_fa
        
        self.power_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_pairs),
            nn.Sigmoid()
        )
        
        self.fa_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_pairs * n_fa)
        )
    
    def forward(self, x):
        power_logits = self.power_net(x)
        fa_logits = self.fa_net(x).view(-1, self.n_pairs, self.n_fa)
        fa_probs = torch.softmax(fa_logits, dim=-1)
        return power_logits, fa_probs

def test_power_range_hypothesis():
    """Test if wider power range improves restricted scenario performance"""
    
    print("=" * 80)
    print("TESTING POWER RANGE HYPOTHESIS")
    print("=" * 80)
    
    # Setup
    cfg = SimulationConfig.from_yaml('cfgs/config_fa1.yaml')
    cfg.seed = 42
    device = 'cpu'
    
    # Generate a challenging restricted scenario
    scenario = Scenario.random(cfg, restrict_rx_distance=True)
    gains_db = scenario.channel_gains_db()
    gains_linear = db_to_linear(gains_db)
    
    print(f"Test scenario:")
    print(f"  Channel gains (dB):\n{gains_db}")
    print(f"  Direct link gains: {np.diag(gains_db)}")
    
    # Get optimal solution for comparison
    from src.algorithms.fs_bruteforce import fs_bruteforce
    fs_solver = fs_bruteforce(cfg)
    optimal_result = fs_solver.decide(scenario)
    
    # Calculate sum rate from optimal result
    noise_lin = db_to_linear(cfg.noise_power_dbm) * 1e-3
    bandwidth = cfg.bandwidth_hz
    tx_power_lin = db_to_linear(optimal_result['tx_power_dbm']) * 1e-3
    g = scenario.get_channel_gain_with_fa_penalty(optimal_result['fa_indices'])
    sinr = sinr_linear(tx_power_lin, optimal_result['fa_indices'], g, noise_lin)
    optimal_sum_rate = sum_rate_bps(sinr, bandwidth)
    
    print(f"\nOptimal (Full Search) sum-rate: {optimal_sum_rate:.2e} bit/s")
    
    # Test 1: Our current power range (-50 to 30 dBm)
    print(f"\n" + "="*50)
    print("TEST 1: Current Power Range (-50 to 30 dBm)")
    print("="*50)
    
    min_power_dbm_current = -50
    max_power_dbm_current = 30
    min_power_w_current = 10 ** ((min_power_dbm_current - 30) / 10)
    max_power_w_current = 10 ** ((max_power_dbm_current - 30) / 10)
    
    print(f"Power range: {min_power_dbm_current} to {max_power_dbm_current} dBm")
    print(f"Power range: {min_power_w_current:.2e} to {max_power_w_current:.2e} W")
    print(f"Dynamic range: {max_power_dbm_current - min_power_dbm_current} dB")
    
    # Train DNN with current power range
    result_current = train_test_dnn(scenario, cfg, min_power_w_current, max_power_w_current, 
                                   "Current Range", epochs=1000)
    
    # Test 2: Reference power range (-70 to 30 dBm, matching reference)
    print(f"\n" + "="*50)
    print("TEST 2: Reference Power Range (-70 to 30 dBm)")
    print("="*50)
    
    min_power_dbm_ref = -70
    max_power_dbm_ref = 30
    min_power_w_ref = 10 ** ((min_power_dbm_ref - 30) / 10)
    max_power_w_ref = 10 ** ((max_power_dbm_ref - 30) / 10)
    
    print(f"Power range: {min_power_dbm_ref} to {max_power_dbm_ref} dBm")
    print(f"Power range: {min_power_w_ref:.2e} to {max_power_w_ref:.2e} W")
    print(f"Dynamic range: {max_power_dbm_ref - min_power_dbm_ref} dB")
    
    # Train DNN with reference power range
    result_ref = train_test_dnn(scenario, cfg, min_power_w_ref, max_power_w_ref, 
                               "Reference Range", epochs=1000)
    
    # Test 3: Even wider power range (-90 to 30 dBm)
    print(f"\n" + "="*50)
    print("TEST 3: Ultra-Wide Power Range (-90 to 30 dBm)")
    print("="*50)
    
    min_power_dbm_ultra = -90
    max_power_dbm_ultra = 30
    min_power_w_ultra = 10 ** ((min_power_dbm_ultra - 30) / 10)
    max_power_w_ultra = 10 ** ((max_power_dbm_ultra - 30) / 10)
    
    print(f"Power range: {min_power_dbm_ultra} to {max_power_dbm_ultra} dBm")
    print(f"Power range: {min_power_w_ultra:.2e} to {max_power_w_ultra:.2e} W")
    print(f"Dynamic range: {max_power_dbm_ultra - min_power_dbm_ultra} dB")
    
    # Train DNN with ultra-wide power range
    result_ultra = train_test_dnn(scenario, cfg, min_power_w_ultra, max_power_w_ultra, 
                                 "Ultra-Wide Range", epochs=1000)
    
    # Summary
    print(f"\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"Optimal (Full Search):     {optimal_sum_rate:.2e} bit/s (ratio = 1.000)")
    print(f"Current Range (-50-30):    {result_current['sum_rate']:.2e} bit/s (ratio = {result_current['sum_rate']/optimal_sum_rate:.3f})")
    print(f"Reference Range (-70-30):  {result_ref['sum_rate']:.2e} bit/s (ratio = {result_ref['sum_rate']/optimal_sum_rate:.3f})")
    print(f"Ultra-Wide Range (-90-30): {result_ultra['sum_rate']:.2e} bit/s (ratio = {result_ultra['sum_rate']/optimal_sum_rate:.3f})")
    
    improvement_ref = (result_ref['sum_rate'] - result_current['sum_rate']) / result_current['sum_rate'] * 100
    improvement_ultra = (result_ultra['sum_rate'] - result_current['sum_rate']) / result_current['sum_rate'] * 100
    
    print(f"\nImprovement over current range:")
    print(f"  Reference range: {improvement_ref:+.1f}%")
    print(f"  Ultra-wide range: {improvement_ultra:+.1f}%")
    
    if improvement_ref > 5:
        print(f"\n✅ HYPOTHESIS CONFIRMED!")
        print(f"   Wider power range significantly improves performance")
        print(f"   Reference range gives {improvement_ref:.1f}% improvement")
    else:
        print(f"\n❌ HYPOTHESIS NOT CONFIRMED")
        print(f"   Power range difference doesn't explain performance gap")

def train_test_dnn(scenario, cfg, min_power_w, max_power_w, name, epochs=1000):
    """Train and test DNN with specific power range"""
    
    # Prepare input
    gains_db = scenario.channel_gains_db()
    gains_linear = db_to_linear(gains_db)
    
    # Normalize input (per-sample normalization)
    x_raw = torch.tensor(gains_db.flatten(), dtype=torch.float32).unsqueeze(0)
    x_mean = x_raw.mean()
    x_std = x_raw.std() + 1e-8
    x_normalized = (x_raw - x_mean) / x_std
    
    # Create model
    model = TestDNN(input_size=cfg.n_pairs * cfg.n_pairs, n_pairs=cfg.n_pairs, n_fa=cfg.n_fa)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Noise power
    noise_power_dbm = cfg.noise_power_dbm
    noise_power_w = 10 ** ((noise_power_dbm - 30) / 10)
    
    print(f"Training DNN with {name}...")
    print(f"  Epochs: {epochs}")
    print(f"  Noise power: {noise_power_dbm} dBm ({noise_power_w:.2e} W)")
    
    best_sum_rate = 0
    best_powers = None
    best_fa = None
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        power_logits, fa_probs = model(x_normalized)
        
        # Scale powers to specified range
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
            best_powers = powers_w.detach().numpy()
            best_fa = fa_probs.squeeze().detach().numpy()
        
        if epoch % 200 == 0:
            print(f"    Epoch {epoch}: sum-rate = {sum_rate.item():.2e} bit/s")
    
    print(f"  Final best sum-rate: {best_sum_rate:.2e} bit/s")
    print(f"  Best powers (dBm): {10 * np.log10(best_powers * 1000)}")
    
    return {
        'sum_rate': best_sum_rate,
        'powers_w': best_powers,
        'fa_probs': best_fa
    }

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
    test_power_range_hypothesis() 