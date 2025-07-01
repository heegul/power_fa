#!/usr/bin/env python3
"""
Detailed analysis of why sinr_linear() systematically overestimates SINR
compared to the correct MATLAB calculation.
"""

import yaml
import numpy as np
from src.simulator.environment import db_to_linear
from src.simulator.metrics import sinr_linear

def analyze_sinr_difference():
    """Analyze the difference between sinr_linear() and MATLAB SINR calculation"""
    
    print("=" * 80)
    print("🔍 DETAILED ANALYSIS: Why sinr_linear() Overestimates SINR")
    print("=" * 80)
    print()
    
    # Load a sample scenario
    with open('results_fa1_6pairs_single_rrx.yaml', 'r') as f:
        results = yaml.safe_load(f)
    
    sample = results['samples'][0]  # Use first sample
    cfg = sample['config']
    
    print("📊 SAMPLE SCENARIO ANALYSIS:")
    print("-" * 40)
    print(f"n_pairs: {cfg['n_pairs']}")
    print(f"n_fa: {cfg['n_fa']}")
    print(f"fa_penalty_db: {cfg['fa_penalty_db']}")
    print()
    
    # Test case: all devices use max power, FA assignment [0, 1, 0, 1, 0, 1]
    n_pairs = cfg['n_pairs']
    tx_power_max_dbm = cfg['tx_power_max_dbm']
    tx_power_lin = db_to_linear(np.array([tx_power_max_dbm] * n_pairs)) * 1e-3
    fa_indices = np.array([0, 1, 0, 1, 0, 1])  # Alternating FA assignment
    channel_gains_db = np.array(sample['channel_gains_db'])
    noise_power_lin = db_to_linear(cfg['noise_power_dbm']) * 1e-3
    fa_penalty_db = cfg['fa_penalty_db']
    
    print("🎯 TEST CASE:")
    print("-" * 40)
    print(f"TX powers (dBm): {[tx_power_max_dbm] * n_pairs}")
    print(f"TX powers (W): {tx_power_lin}")
    print(f"FA assignment: {fa_indices}")
    print(f"Noise power (W): {noise_power_lin:.2e}")
    print()
    
    # Apply FA penalty to channel gains
    channel_gain_db_with_penalty = channel_gains_db.copy()
    for rx in range(n_pairs):
        penalty_db = fa_penalty_db * fa_indices[rx]
        channel_gain_db_with_penalty[:, rx] -= penalty_db
    
    channel_gain_linear = db_to_linear(channel_gain_db_with_penalty)
    
    print("📡 CHANNEL GAINS (after FA penalty):")
    print("-" * 40)
    print("Channel gains (dB):")
    for i in range(n_pairs):
        print(f"  Row {i}: {channel_gain_db_with_penalty[i]}")
    print()
    print("Channel gains (linear):")
    for i in range(n_pairs):
        print(f"  Row {i}: {channel_gain_linear[i]}")
    print()
    
    # Method 1: sinr_linear() function (INCORRECT)
    print("🔴 METHOD 1: sinr_linear() Function (INCORRECT)")
    print("-" * 50)
    sinr_incorrect = sinr_linear(
        tx_power_lin=tx_power_lin,
        fa_indices=fa_indices,
        channel_gain=channel_gain_linear,
        noise_power_lin=noise_power_lin,
    )
    
    print("Step-by-step calculation:")
    for i in range(n_pairs):
        same_fa = fa_indices == fa_indices[i]
        print(f"\nRX {i} (FA {fa_indices[i]}):")
        print(f"  same_fa mask: {same_fa}")
        print(f"  Interfering TXs: {np.where(same_fa)[0]}")
        
        # This is the PROBLEM: sinr_linear() only considers interference from devices on SAME FA
        interference_same_fa = np.sum(tx_power_lin[same_fa] * channel_gain_linear[same_fa, i]) - (
            tx_power_lin[i] * channel_gain_linear[i, i]
        )
        signal = tx_power_lin[i] * channel_gain_linear[i, i]
        sinr_val = signal / (interference_same_fa + noise_power_lin)
        
        print(f"  Signal power: {signal:.2e} W")
        print(f"  Interference (same FA only): {interference_same_fa:.2e} W")
        print(f"  Total denominator: {interference_same_fa + noise_power_lin:.2e} W")
        print(f"  SINR: {sinr_val:.2f} = {10*np.log10(sinr_val):.1f} dB")
    
    print(f"\nsinr_linear() results: {sinr_incorrect}")
    print(f"Sum-rate: {np.sum(np.log2(1 + sinr_incorrect)):.2f}")
    print()
    
    # Method 2: MATLAB-style calculation (CORRECT)
    print("🟢 METHOD 2: MATLAB-style Calculation (CORRECT)")
    print("-" * 50)
    
    print("Step-by-step calculation:")
    sinr_correct = np.zeros(n_pairs)
    for i in range(n_pairs):
        print(f"\nRX {i} (FA {fa_indices[i]}):")
        
        # CORRECT: All transmitters interfere, regardless of FA assignment
        # (FA penalty is already applied to channel gains)
        total_interference = np.sum(tx_power_lin * channel_gain_linear[:, i]) - (
            tx_power_lin[i] * channel_gain_linear[i, i]
        )
        signal = tx_power_lin[i] * channel_gain_linear[i, i]
        sinr_val = signal / (total_interference + noise_power_lin)
        sinr_correct[i] = sinr_val
        
        print(f"  Signal power: {signal:.2e} W")
        print(f"  Interference (ALL TXs): {total_interference:.2e} W")
        print(f"  Total denominator: {total_interference + noise_power_lin:.2e} W")
        print(f"  SINR: {sinr_val:.2f} = {10*np.log10(sinr_val):.1f} dB")
    
    print(f"\nMATLAB-style results: {sinr_correct}")
    print(f"Sum-rate: {np.sum(np.log2(1 + sinr_correct)):.2f}")
    print()
    
    # Analysis of the difference
    print("⚡ ANALYSIS OF THE DIFFERENCE:")
    print("-" * 40)
    sinr_ratio = sinr_incorrect / sinr_correct
    sinr_diff_db = 10 * np.log10(sinr_ratio)
    
    print(f"SINR ratio (incorrect/correct): {sinr_ratio}")
    print(f"SINR difference (dB): {sinr_diff_db}")
    print(f"Average SINR overestimation: {np.mean(sinr_diff_db):.1f} dB")
    print()
    
    sum_rate_incorrect = np.sum(np.log2(1 + sinr_incorrect))
    sum_rate_correct = np.sum(np.log2(1 + sinr_correct))
    sum_rate_ratio = sum_rate_incorrect / sum_rate_correct
    
    print(f"Sum-rate ratio (incorrect/correct): {sum_rate_ratio:.3f}")
    print(f"Sum-rate overestimation: {(sum_rate_ratio - 1) * 100:.1f}%")
    print()
    
    print("🎯 ROOT CAUSE EXPLANATION:")
    print("-" * 40)
    print("1. 🔴 sinr_linear() INCORRECTLY assumes:")
    print("   • Only devices on the SAME frequency interfere with each other")
    print("   • Devices on different frequencies don't interfere")
    print("   • This is WRONG because FA penalty is already applied to channel gains")
    print()
    print("2. 🟢 MATLAB calculation CORRECTLY assumes:")
    print("   • ALL transmitters interfere with each receiver")
    print("   • FA penalty is applied by reducing channel gains (not by filtering interference)")
    print("   • This matches the physical reality of the system")
    print()
    print("3. 💡 WHY sinr_linear() overestimates:")
    print("   • By ignoring interference from devices on different FAs,")
    print("   • it underestimates the total interference power")
    print("   • Lower interference → Higher SINR → Higher sum-rate")
    print("   • This leads to systematic overestimation of performance")
    print()
    
    print("🔧 THE FIX:")
    print("-" * 40)
    print("Replace sinr_linear() calls with direct SINR calculation:")
    print("```python")
    print("# CORRECT SINR calculation")
    print("for i in range(n_pairs):")
    print("    signal = tx_power_lin[i] * channel_gain[i, i]")
    print("    interference = np.sum(tx_power_lin * channel_gain[:, i]) - signal")
    print("    sinr[i] = signal / (interference + noise_power_lin)")
    print("```")
    print()
    print("=" * 80)

def main():
    """Main function"""
    analyze_sinr_difference()

if __name__ == "__main__":
    main() 