#!/usr/bin/env python3
"""
Demonstration of normalization strategies in Power FA training.
Shows the difference between NPY mode (global) and single sample mode normalization.
"""

import numpy as np
import matplotlib.pyplot as plt

def main():
    # Simulate channel gain data (6x6 matrices flattened to 36 features)
    np.random.seed(42)
    n_samples = 100
    n_features = 36  # 6x6 channel matrix

    # Generate synthetic channel gains in dB (typical range: -120 to -60 dB)
    samples = np.random.normal(-90, 15, (n_samples, n_features))

    print('=== NORMALIZATION COMPARISON ===')
    print()

    # NPY Mode: Global normalization across all samples
    global_mean = samples.mean(axis=0)  # Shape: (36,) - per-feature
    global_std = samples.std(axis=0)    # Shape: (36,) - per-feature

    print('NPY MODE (Global Normalization):')
    print(f'  Mean shape: {global_mean.shape}')
    print(f'  Std shape: {global_std.shape}')
    print(f'  Mean range: [{global_mean.min():.2f}, {global_mean.max():.2f}] dB')
    print(f'  Std range: [{global_std.min():.2f}, {global_std.max():.2f}] dB')
    print(f'  Overall mean: {global_mean.mean():.2f} dB')
    print(f'  Overall std: {global_std.mean():.2f} dB')
    print()

    # Single Sample Mode: Per-sample normalization
    sample_idx = 0
    single_sample = samples[sample_idx]  # Shape: (36,)
    sample_mean = single_sample.mean()   # Scalar
    sample_std = single_sample.std()     # Scalar

    print('SINGLE SAMPLE MODE (Per-sample Normalization):')
    print(f'  Mean: {sample_mean:.2f} dB (scalar)')
    print(f'  Std: {sample_std:.2f} dB (scalar)')
    print()

    # Show effect on first sample
    sample_global_norm = (single_sample - global_mean) / global_std
    sample_local_norm = (single_sample - sample_mean) / sample_std

    print('NORMALIZATION EFFECT ON SAMPLE 0:')
    print(f'  Original: mean={single_sample.mean():.2f}, std={single_sample.std():.2f}')
    print(f'  Global norm: mean={sample_global_norm.mean():.2f}, std={sample_global_norm.std():.2f}')
    print(f'  Local norm: mean={sample_local_norm.mean():.2f}, std={sample_local_norm.std():.2f}')
    print()

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparison of Input Normalization Strategies', fontsize=18, fontweight='bold')

    # Plot 1: Raw data distribution
    axes[0,0].hist(samples.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0,0].set_title('Raw Channel Gains (All Samples)', fontsize=13)
    axes[0,0].set_xlabel('Channel Gain (dB)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].grid(True, alpha=0.3)

    # Plot 2: Global normalization effect (NPY mode)
    global_normalized = (samples - global_mean) / global_std
    axes[0,1].hist(global_normalized.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black', label='NPY Mode (Global)')
    axes[0,1].set_title('NPY Mode (Global Normalization)\nAll Samples, Per-Feature Stats', fontsize=13)
    axes[0,1].set_xlabel('Normalized Value')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # Plot 3: Single sample mode normalization (all samples)
    # For each sample, normalize by its own mean/std
    local_normalized = np.zeros_like(samples)
    for i in range(n_samples):
        m = samples[i].mean()
        s = samples[i].std() + 1e-8
        local_normalized[i] = (samples[i] - m) / s
    axes[1,0].hist(local_normalized.flatten(), bins=50, alpha=0.7, color='orange', edgecolor='black', label='Single Sample Mode (Local)')
    axes[1,0].set_title('Single Sample Mode (Per-Sample Normalization)\nAll Samples, Per-Sample Stats', fontsize=13)
    axes[1,0].set_xlabel('Normalized Value')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # Plot 4: Normalization statistics
    x_pos = np.arange(min(10, n_features))
    axes[1,1].bar(x_pos - 0.2, global_mean[:len(x_pos)], 0.4, label='Global Mean', alpha=0.7)
    axes[1,1].bar(x_pos + 0.2, global_std[:len(x_pos)], 0.4, label='Global Std', alpha=0.7)
    axes[1,1].axhline(sample_mean, color='red', linestyle='--', label='Sample Mean')
    axes[1,1].axhline(sample_std, color='orange', linestyle='--', label='Sample Std')
    axes[1,1].set_title('Normalization Statistics per Feature', fontsize=13)
    axes[1,1].set_xlabel('Feature Index')
    axes[1,1].set_ylabel('Value (dB)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('normalization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print('Visualization saved to: normalization_comparison.png')

if __name__ == "__main__":
    main() 