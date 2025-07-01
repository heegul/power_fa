#!/usr/bin/env python3
"""
Python equivalent of the MATLAB scripts:
- compute_all_max_power.m
- plot_power_fa_results.m

This script computes and displays the same results as the MATLAB scripts.
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from src.simulator.environment import db_to_linear
from src.simulator.metrics import sinr_linear, sum_rate_bps

def compute_all_max_power(samples):
    """
    Python equivalent of compute_all_max_power.m
    Computes sum rates for all-max-power scenario with uniform FA averaging
    """
    n_samples = len(samples)
    sum_rates = np.zeros(n_samples)
    
    for i, sample in enumerate(samples):
        cfg = sample['config']
        n_pairs = cfg['n_pairs']
        n_fa = cfg['n_fa']
        tx_power_max_dbm = cfg['tx_power_max_dbm']
        
        # Use max power for all pairs
        tx_power_dbm = [tx_power_max_dbm] * n_pairs
        tx_power_lin = db_to_linear(np.array(tx_power_dbm)) * 1e-3
        
        channel_gains_db = np.array(sample['channel_gains_db'])
        noise_power_lin = db_to_linear(cfg['noise_power_dbm']) * 1e-3
        fa_penalty_db = cfg['fa_penalty_db']
        
        # Calculate sum rate for all possible FA combinations
        fa_combinations = list(product(range(n_fa), repeat=n_pairs))
        rates = []
        
        for fa_indices in fa_combinations:
            # Apply FA penalty
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
            rate = sum_rate_bps(sinr, cfg['bandwidth_hz'])
            rates.append(rate)
        
        # Average over all combinations
        sum_rates[i] = np.mean(rates)
    
    return sum_rates

def compute_all_power_fa_combinations(samples):
    """
    Compute average sum rate over all power and FA combinations
    """
    n_samples = len(samples)
    sum_rates = np.full(n_samples, np.nan)
    
    for i, sample in enumerate(samples):
        cfg = sample['config']
        n_pairs = cfg['n_pairs']
        n_fa = cfg['n_fa']
        
        # Only compute for small systems (computational complexity)
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
        
        # Generate all combinations
        power_assignments = list(product(power_levels, repeat=n_pairs))
        fa_assignments = list(product(range(n_fa), repeat=n_pairs))
        
        rates = []
        for power_assignment in power_assignments:
            for fa_assignment in fa_assignments:
                tx_power_lin = db_to_linear(np.array(power_assignment)) * 1e-3
                
                # Apply FA penalty
                channel_gain_db = channel_gains_db.copy()
                for rx in range(n_pairs):
                    penalty_db = fa_penalty_db * fa_assignment[rx]
                    channel_gain_db[:, rx] -= penalty_db
                
                channel_gain = db_to_linear(channel_gain_db)
                
                sinr = sinr_linear(
                    tx_power_lin=tx_power_lin,
                    fa_indices=np.array(fa_assignment),
                    channel_gain=channel_gain,
                    noise_power_lin=noise_power_lin,
                )
                rate = sum_rate_bps(sinr, cfg['bandwidth_hz'])
                rates.append(rate)
        
        sum_rates[i] = np.mean(rates)
    
    return sum_rates

def plot_power_fa_results(yaml_file, plot_type='sum_rate_vs_sample', save_dir=None):
    """
    Python equivalent of plot_power_fa_results.m
    """
    print(f"Reading results from {yaml_file}...")
    
    with open(yaml_file, 'r') as f:
        results = yaml.safe_load(f)
    
    samples = results['samples']
    n_samples = len(samples)
    
    # Extract data
    sample_indices = np.arange(n_samples)
    dnn_sum_rates = np.array([s['dnn']['sum_rate'] for s in samples])
    fs_sum_rates = np.array([s['fs']['sum_rate'] for s in samples])
    ratios = np.array([s['ratio'] for s in samples])
    
    # Compute additional metrics
    print("Computing all-max-power scenario (uniform FA average)...")
    all_max_sum_rates = compute_all_max_power(samples)
    
    print("Computing all power & FA combinations average...")
    all_power_fa_avg_sum_rates = compute_all_power_fa_combinations(samples)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    if plot_type == 'sum_rate_vs_sample':
        # Plot sum rates
        plt.plot(sample_indices, dnn_sum_rates / 1e6, 'b-o', linewidth=1.5, label='DNN')
        plt.plot(sample_indices, fs_sum_rates / 1e6, 'r-s', linewidth=1.5, label='FS')
        plt.plot(sample_indices, all_max_sum_rates / 1e6, 'g--d', linewidth=1.5, label='All Max Power (Uniform FA Avg)')
        
        # Plot all power & FA combinations if available
        valid_mask = ~np.isnan(all_power_fa_avg_sum_rates)
        if np.any(valid_mask):
            plt.plot(sample_indices[valid_mask], all_power_fa_avg_sum_rates[valid_mask] / 1e6, 
                    'm:', linewidth=2, label='All Power & FA Comb. Avg')
        
        # Add average lines
        dnn_avg = np.mean(dnn_sum_rates) / 1e6
        fs_avg = np.mean(fs_sum_rates) / 1e6
        all_max_avg = np.mean(all_max_sum_rates) / 1e6
        all_power_fa_avg = np.mean(all_power_fa_avg_sum_rates[valid_mask]) / 1e6 if np.any(valid_mask) else np.nan
        
        plt.axhline(dnn_avg, color='b', linestyle=':', linewidth=1.5, alpha=0.7)
        plt.axhline(fs_avg, color='r', linestyle=':', linewidth=1.5, alpha=0.7)
        plt.axhline(all_max_avg, color='g', linestyle=':', linewidth=1.5, alpha=0.7)
        if not np.isnan(all_power_fa_avg):
            plt.axhline(all_power_fa_avg, color='m', linestyle=':', linewidth=1.5, alpha=0.7)
        
        # Annotate averages
        x_right = max(sample_indices) + 1
        plt.text(x_right, dnn_avg, f'{dnn_avg:.2f} Mbps', color='b', fontweight='bold', va='center')
        plt.text(x_right, fs_avg, f'{fs_avg:.2f} Mbps', color='r', fontweight='bold', va='center')
        plt.text(x_right, all_max_avg, f'{all_max_avg:.2f} Mbps', color='g', fontweight='bold', va='center')
        if not np.isnan(all_power_fa_avg):
            plt.text(x_right, all_power_fa_avg, f'{all_power_fa_avg:.2f} Mbps', color='m', fontweight='bold', va='center')
        
        plt.xlabel('Sample Index')
        plt.ylabel('Sum Rate (Mbps)')
        plt.title('Sum Rate vs. Sample')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Print summary
        print(f"\n=== SUMMARY RESULTS ===")
        print(f"DNN Average: {dnn_avg:.2f} Mbps")
        print(f"FS Average: {fs_avg:.2f} Mbps")
        print(f"All Max Power Average: {all_max_avg:.2f} Mbps")
        if not np.isnan(all_power_fa_avg):
            print(f"All Power & FA Comb. Average: {all_power_fa_avg:.2f} Mbps")
        print(f"Average Ratio (DNN/FS): {np.mean(ratios):.3f}")
        
    elif plot_type == 'ratio_vs_sample':
        # Plot ratios
        plt.plot(sample_indices, ratios, 'o-', color=[0.2, 0.6, 0.2], linewidth=1.5, label='ML/FS Sum-Rate Ratio')
        
        # Plot average line
        ratio_avg = np.mean(ratios)
        plt.axhline(ratio_avg, color=[0.2, 0.6, 0.2], linestyle=':', linewidth=1.5, alpha=0.7)
        
        # Annotate average
        x_right = max(sample_indices) + 1
        plt.text(x_right, ratio_avg, f'{ratio_avg:.3f}', color=[0.2, 0.6, 0.2], fontweight='bold', va='center')
        
        plt.xlabel('Sample Index')
        plt.ylabel('ML/FS Sum-Rate Ratio')
        plt.title('ML Model / FS Sum-Rate Ratio per Sample')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        print(f"\n=== RATIO SUMMARY ===")
        print(f"Average Ratio: {ratio_avg:.3f}")
        print(f"Min Ratio: {np.min(ratios):.3f}")
        print(f"Max Ratio: {np.max(ratios):.3f}")
        print(f"Std Ratio: {np.std(ratios):.3f}")
    
    plt.tight_layout()
    
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{plot_type}_matlab_equivalent.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

def main():
    """Main function to demonstrate the MATLAB equivalent functionality"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python matlab_equivalent.py <yaml_file> [plot_type] [save_dir]")
        print("plot_type: 'sum_rate_vs_sample' or 'ratio_vs_sample' (default: sum_rate_vs_sample)")
        return
    
    yaml_file = sys.argv[1]
    plot_type = sys.argv[2] if len(sys.argv) > 2 else 'sum_rate_vs_sample'
    save_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    plot_power_fa_results(yaml_file, plot_type, save_dir)

if __name__ == "__main__":
    main() 