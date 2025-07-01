#!/usr/bin/env python3
"""
Extract and print the "All Power & FA Comb. Avg" values from sample_visualization.py method
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

def extract_sample_viz_values(yaml_file):
    """Extract values using exact sample_visualization.py method"""
    
    with open(yaml_file, 'r') as f:
        results = yaml.safe_load(f)
    
    samples = results['samples']
    
    # Extract basic values
    sample_indices = [s['index'] for s in samples]
    dnn_sum_rates = [s['dnn']['sum_rate'] for s in samples]
    fs_sum_rates = [s['fs']['sum_rate'] for s in samples]
    
    all_max_sum_rates = []
    all_power_fa_avg_sum_rates = []
    
    for s in samples:
        cfg = s['config']
        n_pairs = int(cfg['n_pairs'])
        n_fa = int(cfg['n_fa'])
        tx_power_min_dbm = float(cfg['tx_power_min_dbm'])
        tx_power_max_dbm = float(cfg['tx_power_max_dbm'])
        
        # --- All Max Power (Uniform FA Avg) ---
        power_levels = np.array([
            tx_power_min_dbm,
            tx_power_max_dbm - 6,
            tx_power_max_dbm - 3,
            tx_power_max_dbm
        ])
        
        tx_power_dbm = [tx_power_max_dbm] * n_pairs
        channel_gains_db = np.array(s['channel_gains_db'])
        tx_power_lin = db_to_linear(np.array(tx_power_dbm)) * 1e-3
        noise_power_lin = db_to_linear(float(cfg['noise_power_dbm'])) * 1e-3
        fa_penalty_db = float(cfg['fa_penalty_db'])
        
        # Enumerate all possible FA assignments (cartesian product)
        fa_assignments = list(product(range(n_fa), repeat=n_pairs))
        sum_rates = []
        for fa_indices in fa_assignments:
            channel_gain_db = channel_gains_db.copy()
            for rx in range(n_pairs):
                penalty_db = fa_penalty_db * fa_indices[rx]
                channel_gain_db[:, rx] -= penalty_db
            channel_gain = db_to_linear(channel_gain_db)
            sinr = sinr_linear(
                tx_power_lin=tx_power_lin,
                fa_indices=np.array(fa_indices),
                channel_gain=channel_gain,
                noise_power_lin=noise_power_lin,
            )
            sum_rate = sum_rate_bps(sinr, float(cfg['bandwidth_hz']))
            sum_rates.append(sum_rate)
        avg_sum_rate = np.mean(sum_rates)
        all_max_sum_rates.append(avg_sum_rate)
        
        # --- Average over all power and FA combinations (quantized power) ---
        if n_pairs <= 6:
            power_assignments = list(product(power_levels, repeat=n_pairs))
            fa_assignments = list(product(range(n_fa), repeat=n_pairs))
            
            # Prepare arguments for parallel processing
            args_list = [
                (tx_power_dbm_vec, fa_indices, channel_gains_db, fa_penalty_db, db_to_linear, sinr_linear, sum_rate_bps, float(cfg['bandwidth_hz']), noise_power_lin)
                for tx_power_dbm_vec in power_assignments
                for fa_indices in fa_assignments
            ]
            
            # Parallel computation
            sum_rates_all = Parallel(n_jobs=-1, backend='loky', verbose=0)(
                delayed(compute_sum_rate_for_assignment)(*args)
                for args in args_list
            )
            avg_sum_rate_all = np.mean(sum_rates_all)
            all_power_fa_avg_sum_rates.append(avg_sum_rate_all)
        else:
            all_power_fa_avg_sum_rates.append(np.nan)
    
    # Calculate averages
    dnn_avg = np.mean(dnn_sum_rates)
    fs_avg = np.mean(fs_sum_rates)
    all_max_avg = np.mean(all_max_sum_rates)
    if not all(np.isnan(all_power_fa_avg_sum_rates)):
        all_power_fa_avg = np.nanmean(all_power_fa_avg_sum_rates)
    else:
        all_power_fa_avg = np.nan
    
    print(f"=== SAMPLE_VISUALIZATION.PY VALUES ===")
    print(f"DNN Average: {dnn_avg/1e6:.2f} Mbps")
    print(f"FS Average: {fs_avg/1e6:.2f} Mbps")
    print(f"All Max Power Average: {all_max_avg/1e6:.2f} Mbps")
    if not np.isnan(all_power_fa_avg):
        print(f"All Power & FA Comb. Average: {all_power_fa_avg/1e6:.2f} Mbps")
    else:
        print("All Power & FA Comb. Average: N/A (too many pairs)")

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extract_sample_viz_values.py <yaml_file>")
        return
    
    yaml_file = sys.argv[1]
    extract_sample_viz_values(yaml_file)

if __name__ == "__main__":
    main() 