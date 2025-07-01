#!/usr/bin/env python3
"""
Compare the "All Power & FA Comb. Avg" calculation between:
1. sample_visualization.py (uses sinr_linear function)
2. matlab_equivalent_fixed.py (uses MATLAB-style SINR calculation)
"""

import yaml
import numpy as np
from itertools import product
from src.simulator.environment import db_to_linear
from src.simulator.metrics import sinr_linear, sum_rate_bps
from joblib import Parallel, delayed

def compute_sum_rate_for_assignment(tx_power_dbm_vec, fa_indices, channel_gains_db, fa_penalty_db, db_to_linear, sinr_linear, sum_rate_bps, bandwidth_hz, noise_power_lin):
    """Helper function for parallel computation (from sample_visualization.py)"""
    tx_power_lin = db_to_linear(np.array(tx_power_dbm_vec)) * 1e-3
    channel_gain_db = channel_gains_db.copy()
    for rx in range(len(fa_indices)):
        penalty_db = fa_penalty_db * fa_indices[rx]
        channel_gain_db[:, rx] -= penalty_db
    channel_gain = db_to_linear(channel_gain_db)
    sinr = sinr_linear(
        tx_power_lin=tx_power_lin,
        fa_indices=np.array(fa_indices),
        channel_gain=channel_gain,
        noise_power_lin=noise_power_lin,
    )
    sum_rate = sum_rate_bps(sinr, bandwidth_hz)
    return sum_rate

def compute_all_power_fa_combinations_sample_viz(samples):
    """
    Compute using sample_visualization.py method (with sinr_linear function)
    """
    n_samples = len(samples)
    sum_rates = np.full(n_samples, np.nan)
    
    for i, sample in enumerate(samples):
        cfg = sample['config']
        n_pairs = int(cfg['n_pairs'])
        n_fa = int(cfg['n_fa'])
        
        if n_pairs > 6:
            continue
            
        tx_power_min_dbm = float(cfg['tx_power_min_dbm'])
        tx_power_max_dbm = float(cfg['tx_power_max_dbm'])
        
        # Power levels: {min, max-6, max-3, max} dBm
        power_levels = np.array([
            tx_power_min_dbm,
            tx_power_max_dbm - 6,
            tx_power_max_dbm - 3,
            tx_power_max_dbm
        ])
        
        channel_gains_db = np.array(sample['channel_gains_db'])
        noise_power_lin = db_to_linear(float(cfg['noise_power_dbm'])) * 1e-3
        fa_penalty_db = float(cfg['fa_penalty_db'])
        bandwidth_hz = float(cfg['bandwidth_hz'])
        
        # Generate all combinations
        power_assignments = list(product(power_levels, repeat=n_pairs))
        fa_assignments = list(product(range(n_fa), repeat=n_pairs))
        
        # Prepare arguments for parallel processing (like sample_visualization.py)
        args_list = [
            (tx_power_dbm_vec, fa_indices, channel_gains_db, fa_penalty_db, db_to_linear, sinr_linear, sum_rate_bps, bandwidth_hz, noise_power_lin)
            for tx_power_dbm_vec in power_assignments
            for fa_indices in fa_assignments
        ]
        
        # Parallel computation (like sample_visualization.py)
        sum_rates_all = Parallel(n_jobs=-1, backend='loky', verbose=0)(
            delayed(compute_sum_rate_for_assignment)(*args)
            for args in args_list
        )
        
        sum_rates[i] = np.mean(sum_rates_all)
    
    return sum_rates

def compute_all_power_fa_combinations_matlab(samples):
    """
    Compute using MATLAB-equivalent method (direct SINR calculation)
    """
    n_samples = len(samples)
    sum_rates = np.full(n_samples, np.nan)
    
    for i, sample in enumerate(samples):
        cfg = sample['config']
        n_pairs = cfg['n_pairs']
        n_fa = cfg['n_fa']
        
        if n_pairs > 6:
            continue
            
        tx_power_min_dbm = cfg['tx_power_min_dbm']
        tx_power_max_dbm = cfg['tx_power_max_dbm']
        
        # Power levels: {min, max-6, max-3, max} dBm
        power_levels = np.array([
            tx_power_min_dbm,
            tx_power_max_dbm - 6,
            tx_power_max_dbm - 3,
            tx_power_max_dbm
        ])
        
        channel_gains_db = np.array(sample['channel_gains_db'])
        noise_power_lin = db_to_linear(cfg['noise_power_dbm']) * 1e-3
        fa_penalty_db = cfg['fa_penalty_db']
        bandwidth_hz = cfg['bandwidth_hz']
        
        # Generate all combinations
        power_assignments = list(product(power_levels, repeat=n_pairs))
        fa_assignments = list(product(range(n_fa), repeat=n_pairs))
        
        rates = []
        for power_assignment in power_assignments:
            for fa_assignment in fa_assignments:
                tx_power_lin_vec = db_to_linear(np.array(power_assignment)) * 1e-3
                
                # Apply FA penalty - MATLAB style
                channel_gain_db = channel_gains_db.copy()
                penalty_db_vec = fa_penalty_db * np.array(fa_assignment)
                # Subtract penalty from each column (RX)
                channel_gain_db = channel_gain_db - penalty_db_vec[np.newaxis, :]
                channel_gain = db_to_linear(channel_gain_db)
                
                # MATLAB-style SINR calculation
                tx_power_mat = np.tile(tx_power_lin_vec, (n_pairs, 1))  # [n_pairs x n_pairs]
                interf_mat = tx_power_mat * channel_gain  # [n_pairs x n_pairs]
                interf_sum = np.sum(interf_mat, axis=0)  # [1 x n_pairs]
                self_interf = tx_power_lin_vec * np.diag(channel_gain)  # [1 x n_pairs]
                interference = interf_sum - self_interf  # [1 x n_pairs]
                sinr = self_interf / (interference + noise_power_lin)  # [1 x n_pairs]
                sum_rate = bandwidth_hz * np.sum(np.log2(1 + sinr))
                rates.append(sum_rate)
        
        sum_rates[i] = np.mean(rates)
    
    return sum_rates

def main():
    """Compare the two methods"""
    yaml_file = 'results_fa1_6pairs_single_rrx.yaml'
    
    print("=== COMPARISON: sample_visualization.py vs matlab_equivalent_fixed.py ===")
    print(f"Loading results from {yaml_file}...")
    
    with open(yaml_file, 'r') as f:
        results = yaml.safe_load(f)
    
    samples = results['samples']
    
    print("Computing All Power & FA Comb. Avg using sample_visualization.py method...")
    sample_viz_results = compute_all_power_fa_combinations_sample_viz(samples)
    
    print("Computing All Power & FA Comb. Avg using MATLAB-equivalent method...")
    matlab_results = compute_all_power_fa_combinations_matlab(samples)
    
    # Compare results
    valid_mask = ~np.isnan(sample_viz_results)
    sample_viz_avg = np.mean(sample_viz_results[valid_mask]) / 1e6
    matlab_avg = np.mean(matlab_results[valid_mask]) / 1e6
    
    print(f"\n=== RESULTS COMPARISON ===")
    print(f"sample_visualization.py method: {sample_viz_avg:.2f} Mbps")
    print(f"MATLAB-equivalent method:       {matlab_avg:.2f} Mbps")
    print(f"Difference:                     {abs(sample_viz_avg - matlab_avg):.2f} Mbps")
    print(f"Relative difference:            {abs(sample_viz_avg - matlab_avg) / matlab_avg * 100:.1f}%")
    
    # Sample-by-sample comparison
    print(f"\n=== SAMPLE-BY-SAMPLE COMPARISON (first 5 samples) ===")
    for i in range(min(5, len(samples))):
        if not np.isnan(sample_viz_results[i]):
            print(f"Sample {i}: sample_viz={sample_viz_results[i]/1e6:.2f} Mbps, matlab={matlab_results[i]/1e6:.2f} Mbps, diff={abs(sample_viz_results[i] - matlab_results[i])/1e6:.2f} Mbps")
    
    print(f"\n=== CONCLUSION ===")
    if abs(sample_viz_avg - matlab_avg) < 1.0:  # Less than 1 Mbps difference
        print("✅ The methods produce very similar results!")
    else:
        print("❌ Significant difference detected - investigation needed!")
        print("This confirms that sample_visualization.py uses sinr_linear() which differs from MATLAB SINR calculation.")

if __name__ == "__main__":
    main() 