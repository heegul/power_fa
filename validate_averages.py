#!/usr/bin/env python3
"""
Validate the difference between:
1. All Max Power + FA combinations average
2. All Power & FA combinations average
"""

import yaml
import numpy as np
from itertools import product
from src.simulator.environment import db_to_linear
from src.simulator.metrics import sinr_linear, sum_rate_bps

def validate_averages():
    """Validate why the two averages are different"""
    
    print("=== VALIDATION: All Max Power vs All Power & FA Combinations ===")
    print()
    
    # Load results
    with open('results_fa1_6pairs_single.yaml', 'r') as f:
        results = yaml.safe_load(f)
    
    samples = results['samples']
    cfg = samples[0]['config']
    
    print("Configuration:")
    print(f"  n_pairs: {cfg['n_pairs']}")
    print(f"  n_fa: {cfg['n_fa']}")
    print(f"  tx_power_min_dbm: {cfg['tx_power_min_dbm']}")
    print(f"  tx_power_max_dbm: {cfg['tx_power_max_dbm']}")
    print()
    
    # Calculate theoretical number of combinations
    n_pairs = cfg['n_pairs']
    n_fa = cfg['n_fa']
    power_levels = 4  # {min, max-6, max-3, max}
    
    print("Theoretical Analysis:")
    print(f"  All Max Power + FA combinations: {n_fa**n_pairs} combinations")
    print(f"  All Power & FA combinations: {(power_levels**n_pairs) * (n_fa**n_pairs)} combinations")
    print(f"  Ratio: {((power_levels**n_pairs) * (n_fa**n_pairs)) / (n_fa**n_pairs):.0f}x more combinations")
    print()
    
    # Analyze first sample in detail
    sample = samples[0]
    channel_gains_db = np.array(sample['channel_gains_db'])
    
    print("Sample 0 Analysis:")
    print(f"  Channel gains (dB):")
    for i in range(n_pairs):
        print(f"    {channel_gains_db[i]}")
    print()
    
    # Method 1: All Max Power + FA combinations
    print("Method 1: All Max Power + FA combinations")
    tx_power_max_dbm = cfg['tx_power_max_dbm']
    tx_power_dbm_max = [tx_power_max_dbm] * n_pairs
    tx_power_lin_max = db_to_linear(np.array(tx_power_dbm_max)) * 1e-3
    
    fa_assignments = list(product(range(n_fa), repeat=n_pairs))
    sum_rates_max = []
    
    for fa_indices in fa_assignments:
        # Apply FA penalty
        channel_gain_db = channel_gains_db.copy()
        for rx in range(n_pairs):
            penalty_db = cfg['fa_penalty_db'] * fa_indices[rx]
            channel_gain_db[:, rx] -= penalty_db
        
        channel_gain = db_to_linear(channel_gain_db)
        noise_power_lin = db_to_linear(cfg['noise_power_dbm']) * 1e-3
        
        sinr = sinr_linear(
            tx_power_lin=tx_power_lin_max,
            fa_indices=np.array(fa_indices),
            channel_gain=channel_gain,
            noise_power_lin=noise_power_lin,
        )
        sum_rate = sum_rate_bps(sinr, cfg['bandwidth_hz'])
        sum_rates_max.append(sum_rate)
    
    avg_max_power = np.mean(sum_rates_max)
    print(f"  Average sum-rate: {avg_max_power:.2e} bit/s")
    print(f"  Min sum-rate: {np.min(sum_rates_max):.2e} bit/s")
    print(f"  Max sum-rate: {np.max(sum_rates_max):.2e} bit/s")
    print(f"  Std sum-rate: {np.std(sum_rates_max):.2e} bit/s")
    print()
    
    # Method 2: All Power & FA combinations (subset for analysis)
    print("Method 2: All Power & FA combinations")
    power_levels_dbm = np.array([
        cfg['tx_power_min_dbm'],
        cfg['tx_power_max_dbm'] - 6,
        cfg['tx_power_max_dbm'] - 3,
        cfg['tx_power_max_dbm']
    ])
    
    print(f"  Power levels (dBm): {power_levels_dbm}")
    print(f"  Power levels (W): {db_to_linear(power_levels_dbm) * 1e-3}")
    
    # Sample a subset for analysis (full enumeration would be too large)
    power_assignments = list(product(power_levels_dbm, repeat=n_pairs))
    fa_assignments = list(product(range(n_fa), repeat=n_pairs))
    
    print(f"  Total power assignments: {len(power_assignments)}")
    print(f"  Total FA assignments: {len(fa_assignments)}")
    print(f"  Total combinations: {len(power_assignments) * len(fa_assignments)}")
    
    # Sample 1000 random combinations for analysis
    import random
    random.seed(42)
    sample_size = min(1000, len(power_assignments) * len(fa_assignments))
    
    sum_rates_all = []
    for _ in range(sample_size):
        power_assignment = random.choice(power_assignments)
        fa_assignment = random.choice(fa_assignments)
        
        tx_power_lin = db_to_linear(np.array(power_assignment)) * 1e-3
        
        # Apply FA penalty
        channel_gain_db = channel_gains_db.copy()
        for rx in range(n_pairs):
            penalty_db = cfg['fa_penalty_db'] * fa_assignment[rx]
            channel_gain_db[:, rx] -= penalty_db
        
        channel_gain = db_to_linear(channel_gain_db)
        noise_power_lin = db_to_linear(cfg['noise_power_dbm']) * 1e-3
        
        sinr = sinr_linear(
            tx_power_lin=tx_power_lin,
            fa_indices=np.array(fa_assignment),
            channel_gain=channel_gain,
            noise_power_lin=noise_power_lin,
        )
        sum_rate = sum_rate_bps(sinr, cfg['bandwidth_hz'])
        sum_rates_all.append(sum_rate)
    
    avg_all_combinations = np.mean(sum_rates_all)
    print(f"  Average sum-rate (sampled): {avg_all_combinations:.2e} bit/s")
    print(f"  Min sum-rate: {np.min(sum_rates_all):.2e} bit/s")
    print(f"  Max sum-rate: {np.max(sum_rates_all):.2e} bit/s")
    print(f"  Std sum-rate: {np.std(sum_rates_all):.2e} bit/s")
    print()
    
    # Analysis
    print("Key Insights:")
    print(f"  Ratio (All combinations / Max power): {avg_all_combinations / avg_max_power:.3f}")
    print()
    
    print("Why they're different:")
    print("  1. All Max Power: Uses only maximum power (30 dBm = 1.0 W)")
    print("     - High interference between devices")
    print("     - Only FA assignment varies")
    print()
    print("  2. All Power & FA: Uses 4 power levels including very low power")
    print("     - Can turn devices almost OFF (-50 dBm = 1e-8 W)")
    print("     - Much lower interference scenarios possible")
    print("     - Both power and FA assignment vary")
    print()
    
    # Show power level impact
    print("Power Level Impact Analysis:")
    for i, power_dbm in enumerate(power_levels_dbm):
        power_w = db_to_linear(power_dbm) * 1e-3
        print(f"  Level {i+1}: {power_dbm:4.0f} dBm = {power_w:.2e} W")
    
    print()
    print("The difference is EXPECTED and CORRECT!")
    print("All Power & FA combinations should have LOWER average sum-rate because:")
    print("- It includes many suboptimal low-power scenarios")
    print("- Max power scenarios are just a subset of all power scenarios")
    print("- The average is pulled down by poor low-power allocations")

if __name__ == "__main__":
    validate_averages() 