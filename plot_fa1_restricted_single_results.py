#!/usr/bin/env python3
"""
Plot results from FA=1 restricted single sample training.

This script creates comprehensive visualizations for analyzing the performance
of DNN vs Full Search in FA=1 scenarios with restricted RX locations.
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Any

def load_results(file_path: str) -> Dict[str, Any]:
    """Load results from YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def extract_metrics(results: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract key metrics from results."""
    samples = results['samples']
    n_samples = len(samples)
    
    metrics = {
        'ratios': np.zeros(n_samples),
        'dnn_sum_rates': np.zeros(n_samples),
        'fs_sum_rates': np.zeros(n_samples),
        'dnn_powers': [],
        'fs_powers': [],
        'tx_positions': [],
        'rx_positions': [],
        'channel_gains': []
    }
    
    for i, sample in enumerate(samples):
        metrics['ratios'][i] = sample['ratio']
        metrics['dnn_sum_rates'][i] = sample['dnn']['sum_rate']
        metrics['fs_sum_rates'][i] = sample['fs']['sum_rate']
        metrics['dnn_powers'].append(sample['dnn']['tx_power_dbm'])
        metrics['fs_powers'].append(sample['fs']['tx_power_dbm'])
        metrics['tx_positions'].append(sample['tx_xy'])
        metrics['rx_positions'].append(sample['rx_xy'])
        metrics['channel_gains'].append(sample['channel_gains_db'])
    
    return metrics

def plot_ratio_analysis(metrics: Dict[str, np.ndarray], save_dir: Path):
    """Plot ratio analysis (histogram and cumulative distribution)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    ratios = metrics['ratios']
    ax1.hist(ratios, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(ratios), color='red', linestyle='--', 
                label=f'Mean: {np.mean(ratios):.3f}')
    ax1.axvline(1.0, color='green', linestyle='-', alpha=0.7, 
                label='Perfect (ratio=1.0)')
    ax1.set_xlabel('DNN/FS Sum Rate Ratio')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of DNN Performance Ratios')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_ratios = np.sort(ratios)
    cumulative = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
    ax2.plot(sorted_ratios, cumulative, linewidth=2, color='darkblue')
    ax2.axvline(np.mean(ratios), color='red', linestyle='--', 
                label=f'Mean: {np.mean(ratios):.3f}')
    ax2.axvline(1.0, color='green', linestyle='-', alpha=0.7,
                label='Perfect (ratio=1.0)')
    ax2.set_xlabel('DNN/FS Sum Rate Ratio')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Distribution of Performance Ratios')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'ratio_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_sum_rate_comparison(metrics: Dict[str, np.ndarray], save_dir: Path):
    """Plot sum rate comparison between DNN and FS."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    dnn_rates = metrics['dnn_sum_rates'] / 1e6  # Convert to Mbps
    fs_rates = metrics['fs_sum_rates'] / 1e6
    
    # Scatter plot
    ax1.scatter(fs_rates, dnn_rates, alpha=0.7, s=50)
    
    # Perfect line
    min_rate = min(np.min(fs_rates), np.min(dnn_rates))
    max_rate = max(np.max(fs_rates), np.max(dnn_rates))
    ax1.plot([min_rate, max_rate], [min_rate, max_rate], 'r--', 
             label='Perfect (DNN=FS)')
    
    ax1.set_xlabel('Full Search Sum Rate (Mbps)')
    ax1.set_ylabel('DNN Sum Rate (Mbps)')
    ax1.set_title('DNN vs Full Search Sum Rates')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Side-by-side comparison
    sample_indices = np.arange(len(dnn_rates))
    width = 0.35
    
    ax2.bar(sample_indices - width/2, dnn_rates, width, 
            label='DNN', alpha=0.7, color='skyblue')
    ax2.bar(sample_indices + width/2, fs_rates, width, 
            label='Full Search', alpha=0.7, color='orange')
    
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Sum Rate (Mbps)')
    ax2.set_title('Sum Rate Comparison by Sample')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Only show every 5th sample label to avoid crowding
    ax2.set_xticks(sample_indices[::5])
    ax2.set_xticklabels(sample_indices[::5])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'sum_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_power_allocation_analysis(metrics: Dict[str, np.ndarray], save_dir: Path):
    """Analyze power allocation patterns."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    dnn_powers = np.array(metrics['dnn_powers'])
    fs_powers = np.array(metrics['fs_powers'])
    
    # Average power per pair
    avg_dnn_power = np.mean(dnn_powers, axis=0)
    avg_fs_power = np.mean(fs_powers, axis=0)
    pair_indices = np.arange(len(avg_dnn_power))
    
    ax1.bar(pair_indices - 0.2, avg_dnn_power, 0.4, 
            label='DNN', alpha=0.7, color='skyblue')
    ax1.bar(pair_indices + 0.2, avg_fs_power, 0.4, 
            label='Full Search', alpha=0.7, color='orange')
    ax1.set_xlabel('D2D Pair Index')
    ax1.set_ylabel('Average Power (dBm)')
    ax1.set_title('Average Power Allocation per D2D Pair')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Power distribution histograms
    ax2.hist(dnn_powers.flatten(), bins=30, alpha=0.5, 
             label='DNN', color='skyblue', density=True)
    ax2.hist(fs_powers.flatten(), bins=30, alpha=0.5, 
             label='Full Search', color='orange', density=True)
    ax2.set_xlabel('Power (dBm)')
    ax2.set_ylabel('Density')
    ax2.set_title('Power Allocation Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Power variance per sample
    dnn_power_var = np.var(dnn_powers, axis=1)
    fs_power_var = np.var(fs_powers, axis=1)
    
    ax3.scatter(fs_power_var, dnn_power_var, alpha=0.7)
    max_var = max(np.max(fs_power_var), np.max(dnn_power_var))
    ax3.plot([0, max_var], [0, max_var], 'r--', alpha=0.7)
    ax3.set_xlabel('FS Power Variance')
    ax3.set_ylabel('DNN Power Variance')
    ax3.set_title('Power Allocation Variance Comparison')
    ax3.grid(True, alpha=0.3)
    
    # Correlation between power allocation and performance
    ratios = metrics['ratios']
    dnn_power_spread = np.max(dnn_powers, axis=1) - np.min(dnn_powers, axis=1)
    
    ax4.scatter(dnn_power_spread, ratios, alpha=0.7)
    ax4.set_xlabel('DNN Power Spread (max - min)')
    ax4.set_ylabel('Performance Ratio')
    ax4.set_title('Power Spread vs Performance')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'power_allocation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_spatial_analysis(metrics: Dict[str, np.ndarray], save_dir: Path):
    """Analyze spatial distribution of devices and performance."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract spatial data for visualization (first few samples)
    n_viz_samples = min(5, len(metrics['tx_positions']))
    colors = plt.cm.Set1(np.linspace(0, 1, n_viz_samples))
    
    # TX-RX positions for first few samples
    for i in range(n_viz_samples):
        tx_pos = np.array(metrics['tx_positions'][i])
        rx_pos = np.array(metrics['rx_positions'][i])
        
        ax1.scatter(tx_pos[:, 0], tx_pos[:, 1], 
                   color=colors[i], marker='o', s=60, 
                   label=f'Sample {i} TX', alpha=0.7)
        ax1.scatter(rx_pos[:, 0], rx_pos[:, 1], 
                   color=colors[i], marker='s', s=60, 
                   label=f'Sample {i} RX', alpha=0.7)
        
        # Draw lines between TX-RX pairs
        for j in range(len(tx_pos)):
            ax1.plot([tx_pos[j, 0], rx_pos[j, 0]], 
                    [tx_pos[j, 1], rx_pos[j, 1]], 
                    color=colors[i], alpha=0.3, linewidth=1)
    
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Device Positions (First 5 Samples)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Distance distribution
    all_distances = []
    for i in range(len(metrics['tx_positions'])):
        tx_pos = np.array(metrics['tx_positions'][i])
        rx_pos = np.array(metrics['rx_positions'][i])
        distances = np.linalg.norm(tx_pos - rx_pos, axis=1)
        all_distances.extend(distances)
    
    ax2.hist(all_distances, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('TX-RX Distance (m)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of TX-RX Distances')
    ax2.grid(True, alpha=0.3)
    
    # Performance vs distance
    sample_avg_distances = []
    for i in range(len(metrics['tx_positions'])):
        tx_pos = np.array(metrics['tx_positions'][i])
        rx_pos = np.array(metrics['rx_positions'][i])
        distances = np.linalg.norm(tx_pos - rx_pos, axis=1)
        sample_avg_distances.append(np.mean(distances))
    
    ax3.scatter(sample_avg_distances, metrics['ratios'], alpha=0.7)
    ax3.set_xlabel('Average TX-RX Distance (m)')
    ax3.set_ylabel('Performance Ratio')
    ax3.set_title('Performance vs Average Distance')
    ax3.grid(True, alpha=0.3)
    
    # Channel gain distribution
    all_gains = []
    for gains_matrix in metrics['channel_gains']:
        all_gains.extend(np.array(gains_matrix).flatten())
    
    ax4.hist(all_gains, bins=40, alpha=0.7, color='salmon', edgecolor='black')
    ax4.set_xlabel('Channel Gain (dB)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Channel Gains')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'spatial_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_statistics(metrics: Dict[str, np.ndarray], save_dir: Path):
    """Plot comprehensive performance statistics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    ratios = metrics['ratios']
    
    # Box plot of ratios
    ax1.boxplot(ratios, labels=['DNN/FS Ratio'])
    ax1.set_ylabel('Performance Ratio')
    ax1.set_title('Performance Ratio Statistics')
    ax1.grid(True, alpha=0.3)
    
    # Add statistical annotations
    stats_text = f"""
    Mean: {np.mean(ratios):.3f}
    Median: {np.median(ratios):.3f}
    Std: {np.std(ratios):.3f}
    Min: {np.min(ratios):.3f}
    Max: {np.max(ratios):.3f}
    """
    ax1.text(1.05, 0.5, stats_text, transform=ax1.transAxes, 
             verticalalignment='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Performance categories
    perfect_count = np.sum(ratios >= 0.999)
    good_count = np.sum((ratios >= 0.95) & (ratios < 0.999))
    fair_count = np.sum((ratios >= 0.90) & (ratios < 0.95))
    poor_count = np.sum(ratios < 0.90)
    
    categories = ['Perfect\n(≥0.999)', 'Good\n(0.95-0.999)', 
                  'Fair\n(0.90-0.95)', 'Poor\n(<0.90)']
    counts = [perfect_count, good_count, fair_count, poor_count]
    colors = ['green', 'lightgreen', 'yellow', 'red']
    
    bars = ax2.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Performance Categories')
    ax2.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Running average
    window_size = min(10, len(ratios) // 3)
    if window_size > 1:
        running_avg = np.convolve(ratios, np.ones(window_size)/window_size, mode='valid')
        ax3.plot(range(len(ratios)), ratios, 'o-', alpha=0.3, label='Individual')
        ax3.plot(range(window_size-1, len(ratios)), running_avg, 'r-', 
                linewidth=2, label=f'Running Average (window={window_size})')
    else:
        ax3.plot(range(len(ratios)), ratios, 'o-', label='Individual')
    
    ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Performance Ratio')
    ax3.set_title('Performance Over Samples')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Improvement opportunities
    improvement_needed = 1.0 - ratios
    improvement_needed = improvement_needed[improvement_needed > 0]
    
    if len(improvement_needed) > 0:
        ax4.hist(improvement_needed, bins=20, alpha=0.7, 
                color='orange', edgecolor='black')
        ax4.set_xlabel('Improvement Needed (1 - ratio)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Improvement Opportunities')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'All samples achieved\nperfect performance!', 
                ha='center', va='center', transform=ax4.transAxes,
                fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgreen'))
        ax4.set_title('Improvement Analysis')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'performance_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_report(metrics: Dict[str, np.ndarray], file_path: str, save_dir: Path):
    """Create a summary report with key findings."""
    ratios = metrics['ratios']
    
    report = f"""
FA=1 Restricted Single Sample Training Results Summary
======================================================
Data file: {file_path}
Total samples: {len(ratios)}

Performance Statistics:
-----------------------
Average ratio: {np.mean(ratios):.4f} ({np.mean(ratios)*100:.2f}%)
Median ratio:  {np.median(ratios):.4f} ({np.median(ratios)*100:.2f}%)
Std deviation: {np.std(ratios):.4f}
Min ratio:     {np.min(ratios):.4f} ({np.min(ratios)*100:.2f}%)
Max ratio:     {np.max(ratios):.4f} ({np.max(ratios)*100:.2f}%)

Performance Categories:
-----------------------
Perfect (≥99.9%):    {np.sum(ratios >= 0.999):2d} samples ({np.sum(ratios >= 0.999)/len(ratios)*100:.1f}%)
Good (95.0-99.9%):   {np.sum((ratios >= 0.95) & (ratios < 0.999)):2d} samples ({np.sum((ratios >= 0.95) & (ratios < 0.999))/len(ratios)*100:.1f}%)
Fair (90.0-95.0%):   {np.sum((ratios >= 0.90) & (ratios < 0.95)):2d} samples ({np.sum((ratios >= 0.90) & (ratios < 0.95))/len(ratios)*100:.1f}%)
Poor (<90.0%):       {np.sum(ratios < 0.90):2d} samples ({np.sum(ratios < 0.90)/len(ratios)*100:.1f}%)

Sum Rate Statistics:
--------------------
Average DNN sum rate: {np.mean(metrics['dnn_sum_rates'])/1e6:.2f} Mbps
Average FS sum rate:  {np.mean(metrics['fs_sum_rates'])/1e6:.2f} Mbps
DNN sum rate range:   {np.min(metrics['dnn_sum_rates'])/1e6:.2f} - {np.max(metrics['dnn_sum_rates'])/1e6:.2f} Mbps
FS sum rate range:    {np.min(metrics['fs_sum_rates'])/1e6:.2f} - {np.max(metrics['fs_sum_rates'])/1e6:.2f} Mbps

Key Findings:
-------------
• DNN achieves {np.mean(ratios)*100:.1f}% of optimal (Full Search) performance on average
• {np.sum(ratios >= 0.999)} out of {len(ratios)} samples ({np.sum(ratios >= 0.999)/len(ratios)*100:.1f}%) achieve near-perfect performance (≥99.9%)
• Performance variance: {np.std(ratios):.4f} (relatively {'low' if np.std(ratios) < 0.1 else 'moderate' if np.std(ratios) < 0.2 else 'high'})
• Best performance: {np.max(ratios)*100:.2f}% of optimal
• Worst performance: {np.min(ratios)*100:.2f}% of optimal

Configuration:
--------------
FA: 1 (single frequency allocation)
Number of D2D pairs: 6
Restricted RX location: Yes
Training mode: Single sample
"""
    
    with open(save_dir / 'summary_report.txt', 'w') as f:
        f.write(report)
    
    print(report)

def main():
    parser = argparse.ArgumentParser(description='Plot FA=1 restricted single sample results')
    parser.add_argument('--results_file', '-f', type=str, 
                       default='results_fa1_restricted_single.yaml',
                       help='Results file to plot')
    parser.add_argument('--output_dir', '-o', type=str, 
                       default='fa1_restricted_single_plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    save_dir = Path(args.output_dir)
    save_dir.mkdir(exist_ok=True)
    
    print(f"📊 Loading results from {args.results_file}...")
    results = load_results(args.results_file)
    
    print("📈 Extracting metrics...")
    metrics = extract_metrics(results)
    
    print("🎨 Creating visualizations...")
    
    print("  • Plotting ratio analysis...")
    plot_ratio_analysis(metrics, save_dir)
    
    print("  • Plotting sum rate comparison...")
    plot_sum_rate_comparison(metrics, save_dir)
    
    print("  • Plotting power allocation analysis...")
    plot_power_allocation_analysis(metrics, save_dir)
    
    print("  • Plotting spatial analysis...")
    plot_spatial_analysis(metrics, save_dir)
    
    print("  • Plotting performance statistics...")
    plot_performance_statistics(metrics, save_dir)
    
    print("📝 Creating summary report...")
    create_summary_report(metrics, args.results_file, save_dir)
    
    print(f"\n✅ All plots and reports saved to: {save_dir}/")
    print("\nGenerated files:")
    for file in sorted(save_dir.glob('*')):
        print(f"  • {file.name}")

if __name__ == "__main__":
    main() 