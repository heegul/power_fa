#!/usr/bin/env python3
"""
Analyze memorization vs generalization results from network capacity experiments.

This script loads results_mem_* and results_gen_* files, compares performance
across different network architectures and FA configurations, and generates
comprehensive visualizations and statistics.
"""

import os
import glob
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def calculate_parameters(hidden_sizes, input_size=12, output_size=6):
    """Calculate total number of parameters in the network."""
    if isinstance(hidden_sizes, str):
        hidden_sizes = [int(x) for x in hidden_sizes.split('_')]
    
    sizes = [input_size] + hidden_sizes + [output_size]
    total_params = 0
    for i in range(len(sizes) - 1):
        total_params += sizes[i] * sizes[i+1] + sizes[i+1]  # weights + biases
    return total_params

def parse_filename(filepath):
    """Parses a filepath to extract metadata."""
    filename = Path(filepath).name
    parts = filename.split('_')
    
    # Expecting format: results_{type}_{arch}_fa1.yaml
    if len(parts) < 4 or parts[0] != 'results' or parts[-1] != 'fa1.yaml':
        return None
        
    test_type = parts[1]
    architecture = '_'.join(parts[2:-2])
    
    if not architecture:
        return None

    return {
        'type': 'Generalization' if test_type == 'gen' else 'Memorization',
        'architecture': architecture,
    }

def load_data():
    """Loads all mem and gen results into a single DataFrame."""
    all_dfs = []
    
    # Using a more specific glob pattern
    result_files = glob.glob('results_mem_*_fa1.yaml') + glob.glob('results_gen_*_fa1.yaml')

    if not result_files:
        print("No result files found (e.g., 'results_mem_..._fa1.yaml').")
        return pd.DataFrame()

    print(f"Found {len(result_files)} result files.")
    for f in result_files:
        try:
            metadata = parse_filename(f)
            if metadata is None:
                print(f"Warning: Could not parse filename {f}. Skipping.")
                continue

            with open(f, 'r') as file:
                # Handle empty or malformed YAML
                content = file.read()
                if not content.strip():
                    print(f"Warning: File {f} is empty. Skipping.")
                    continue
                data = yaml.safe_load(content)
            
            if not data:
                continue

            df = pd.DataFrame(data)
            df['type'] = metadata['type']
            df['architecture'] = metadata['architecture']
            all_dfs.append(df)
        except Exception as e:
            print(f"Error processing file {f}: {e}")

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)

def create_summary_table(df):
    """Create and save summary statistics table."""
    # Create pivot table for easy comparison
    summary_stats = []
    
    for _, row in df.iterrows():
        summary_stats.append({
            'Architecture': row['architecture'].replace('_', 'x'),
            'Type': row['type'].upper(),
            'Parameters': row['param_count'],
            'Mean_Ratio': row['mean_ratio'],
            'Std_Ratio': row['std_ratio'],
            'Median_Ratio': row['median_ratio'],
            'Min_Ratio': row['min_ratio'],
            'Max_Ratio': row['max_ratio'],
            'N_Samples': row['n_samples']
        })
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Sort by architecture
    summary_df = summary_df.sort_values(['Architecture', 'Type'])
    
    # Save to CSV
    summary_df.to_csv('mem_gen_analysis_summary.csv', index=False)
    print("Summary table saved to: mem_gen_analysis_summary.csv")
    
    # Print formatted table
    print("\n" + "="*100)
    print("MEMORIZATION vs GENERALIZATION SUMMARY")
    print("="*100)
    print(summary_df.to_string(index=False, float_format='%.4f'))
    
    return summary_df

def plot_mem_vs_gen_comparison(df):
    """Create comprehensive comparison plots."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Memorization vs Generalization Analysis', fontsize=16, fontweight='bold')
    
    # Prepare data for plotting
    plot_data = []
    for _, row in df.iterrows():
        for ratio in row['ratios']:
            plot_data.append({
                'Architecture': row['architecture'].replace('_', 'x'),
                'Type': row['type'].upper(),
                'Parameters': row['param_count'],
                'Ratio': ratio
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # 1. Box plot comparison by architecture (Memorization)
    mem_data = plot_df[plot_df['Type'] == 'MEMORIZATION']
    if not mem_data.empty:
        sns.boxplot(data=mem_data, x='Architecture', y='Ratio', ax=axes[0,0])
        axes[0,0].set_title('Memorization by Architecture')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].set_ylim(0, 1)
    
    # 2. Box plot comparison by architecture (Generalization)
    gen_data = plot_df[plot_df['Type'] == 'GENERALIZATION']
    if not gen_data.empty:
        sns.boxplot(data=gen_data, x='Architecture', y='Ratio', ax=axes[0,1])
        axes[0,1].set_title('Generalization by Architecture')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].set_ylim(0, 1)
    
    # 3. Mean performance vs parameters
    mean_data = df.groupby(['architecture', 'type']).agg({
        'mean_ratio': 'mean',
        'param_count': 'first'
    }).reset_index()
    
    for type in ['MEMORIZATION', 'GENERALIZATION']:
        type_data = mean_data[mean_data['type'] == type]
        if not type_data.empty:
            axes[0,2].scatter(type_data['param_count'], type_data['mean_ratio'], 
                            label=type, s=60, alpha=0.7)
    
    axes[0,2].set_xlabel('Number of Parameters')
    axes[0,2].set_ylabel('Mean Ratio')
    axes[0,2].set_title('Mean Performance vs Network Size')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Memorization gap (Mem - Gen) by architecture
    gap_data = []
    for arch in df['architecture'].unique():
        mem_row = df[(df['architecture'] == arch) & (df['type'] == 'MEMORIZATION')]
        gen_row = df[(df['architecture'] == arch) & (df['type'] == 'GENERALIZATION')]
        
        if not mem_row.empty and not gen_row.empty:
            gap = mem_row.iloc[0]['mean_ratio'] - gen_row.iloc[0]['mean_ratio']
            gap_data.append({
                'Architecture': arch.replace('_', 'x'),
                'Gap': gap,
                'Parameters': mem_row.iloc[0]['param_count']
            })
    
    if gap_data:
        gap_df = pd.DataFrame(gap_data)
        sns.barplot(data=gap_df, x='Architecture', y='Gap', ax=axes[1,0])
        axes[1,0].set_title('Memorization Gap (Mem - Gen)')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 5. Performance distribution comparison
    sns.violinplot(data=plot_df, x='Type', y='Ratio', ax=axes[1,1])
    axes[1,1].set_title('Overall Performance Distribution')
    axes[1,1].set_ylim(0, 1)
    
    # 6. Scatter: Memorization vs Generalization
    scatter_data = []
    for arch in df['architecture'].unique():
        mem_row = df[(df['architecture'] == arch) & (df['type'] == 'MEMORIZATION')]
        gen_row = df[(df['architecture'] == arch) & (df['type'] == 'GENERALIZATION')]
        
        if not mem_row.empty and not gen_row.empty:
            scatter_data.append({
                'Memorization': mem_row.iloc[0]['mean_ratio'],
                'Generalization': gen_row.iloc[0]['mean_ratio'],
                'Architecture': arch.replace('_', 'x'),
                'Parameters': mem_row.iloc[0]['param_count']
            })
    
    if scatter_data:
        scatter_df = pd.DataFrame(scatter_data)
        axes[1,2].scatter(scatter_df['Generalization'], scatter_df['Memorization'], 
                        label='Memorization vs Generalization', s=60, alpha=0.7)
        
        # Add diagonal line (perfect correlation)
        lims = [0, 1]
        axes[1,2].plot(lims, lims, 'k--', alpha=0.5, zorder=0)
        axes[1,2].set_xlim(lims)
        axes[1,2].set_ylim(lims)
        axes[1,2].set_xlabel('Generalization Performance')
        axes[1,2].set_ylabel('Memorization Performance')
        axes[1,2].set_title('Memorization vs Generalization')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mem_gen_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    print("Comprehensive analysis plot saved to: mem_gen_analysis_comprehensive.png")
    plt.show()

def analyze_performance_trends(df):
    """Analyze and report key trends."""
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Overall performance comparison
    mem_data = df[df['type'] == 'MEMORIZATION']
    gen_data = df[df['type'] == 'GENERALIZATION']
    
    if not mem_data.empty and not gen_data.empty:
        print(f"\n1. OVERALL PERFORMANCE:")
        print(f"   Memorization - Mean: {mem_data['mean_ratio'].mean():.4f} ± {mem_data['mean_ratio'].std():.4f}")
        print(f"   Generalization - Mean: {gen_data['mean_ratio'].mean():.4f} ± {gen_data['mean_ratio'].std():.4f}")
        print(f"   Average Gap: {mem_data['mean_ratio'].mean() - gen_data['mean_ratio'].mean():.4f}")
    
    # Architecture analysis
    print(f"\n2. ARCHITECTURE ANALYSIS:")
    arch_analysis = []
    for arch in df['architecture'].unique():
        arch_mem = mem_data[mem_data['architecture'] == arch]
        arch_gen = gen_data[gen_data['architecture'] == arch]
        if not arch_mem.empty and not arch_gen.empty:
            gap = arch_mem['mean_ratio'].mean() - arch_gen['mean_ratio'].mean()
            params = arch_mem['param_count'].iloc[0]
            arch_analysis.append((arch, gap, params))
    
    # Sort by gap
    arch_analysis.sort(key=lambda x: x[1], reverse=True)
    
    print("   Architectures ranked by memorization gap (highest first):")
    for arch, gap, params in arch_analysis:
        print(f"     {arch.replace('_', 'x'):15} Gap: {gap:+.4f}  ({params:,} params)")
    
    # Best performers
    print(f"\n3. BEST PERFORMERS:")
    best_mem = mem_data.loc[mem_data['mean_ratio'].idxmax()]
    best_gen = gen_data.loc[gen_data['mean_ratio'].idxmax()]
    
    print(f"   Best Memorization: {best_mem['architecture'].replace('_', 'x')} ({best_mem['mean_ratio']:.4f})")
    print(f"   Best Generalization: {best_gen['architecture'].replace('_', 'x')} ({best_gen['mean_ratio']:.4f})")

def plot_generalization_gap(df):
    """Plots the generalization gap for each architecture."""
    def arch_sort_key(arch):
        parts = [int(p) for p in arch.split('_') if p.isdigit()]
        return (len(parts), sum(parts))

    architectures = sorted(df['architecture'].unique(), key=arch_sort_key)
    
    output_dir = Path('comparison_results')
    output_dir.mkdir(exist_ok=True)

    for arch in architectures:
        plt.figure(figsize=(10, 6))
        
        subset_df = df[df['architecture'] == arch]
        if subset_df.empty or len(subset_df['type'].unique()) < 2:
            print(f"Skipping plot for {arch}: not enough data for comparison.")
            plt.close()
            continue

        sns.lineplot(
            data=subset_df,
            x='N',
            y='dnn_fs_ratio',
            hue='type',
            marker='o',
            errorbar='sd'
        )
        plt.title(f'Generalization vs. Memorization\nArchitecture: {arch.replace("_", "x")}')
        plt.xlabel('Number of Training Samples (N)')
        plt.ylabel('DNN/FS Sum-Rate Ratio')
        plt.xscale('log')
        plt.grid(True, which="both", ls="--")
        plt.axhline(1.0, color='r', linestyle='--', label='FS Performance Baseline')
        plt.legend(title=None)
        
        output_filename = output_dir / f'gap_{arch}.png'
        plt.savefig(output_filename, bbox_inches='tight')
        print(f"Saved plot: {output_filename}")
        plt.close()

def plot_arch_comparison(df):
    """Plots generalization performance across different architectures."""
    gen_df = df[df['type'] == 'Generalization'].copy()
    
    if gen_df.empty:
        print("No generalization data to plot for architecture comparison.")
        return

    def arch_sort_key(arch):
        parts = [int(p) for p in arch.split('_') if p.isdigit()]
        return (len(parts), sum(parts))
        
    arch_order = sorted(gen_df['architecture'].unique(), key=arch_sort_key)

    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=gen_df,
        x='N',
        y='dnn_fs_ratio',
        hue='architecture',
        hue_order=arch_order,
        marker='o',
        errorbar='sd'
    )
    plt.title('Generalization Performance by Model Architecture')
    plt.xlabel('Number of Training Samples (N)')
    plt.ylabel('DNN/FS Sum-Rate Ratio (Generalization)')
    plt.xscale('log')
    plt.grid(True, which="both", ls="--")
    plt.axhline(1.0, color='r', linestyle='--', label='FS Performance Baseline')
    plt.legend(title='Architecture', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    output_dir = Path('comparison_results')
    output_dir.mkdir(exist_ok=True)
    output_filename = output_dir / 'arch_comparison_gen.png'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Saved plot: {output_filename}")
    plt.close()

def main():
    """Main analysis function."""
    print("🔍 Loading memorization and generalization results...")
    
    # Load all results
    df = load_data()
    
    if df.empty:
        print("❌ No results files found!")
        return
    
    print(f"✅ Loaded {len(df)} result files")
    print(f"   - {len(df[df['type'] == 'MEMORIZATION'])} memorization files")
    print(f"   - {len(df[df['type'] == 'GENERALIZATION'])} generalization files")
    print(f"   - Architectures: {sorted(df['architecture'].unique())}")
    
    # Create summary table
    summary_df = create_summary_table(df)
    
    # Create comprehensive plots
    plot_mem_vs_gen_comparison(df)
    
    # Analyze trends
    analyze_performance_trends(df)
    
    # Generate additional plots
    plot_generalization_gap(df)
    plot_arch_comparison(df)
    
    print(f"\n✅ Analysis complete! Check the generated files:")
    print(f"   - mem_gen_analysis_summary.csv")
    print(f"   - mem_gen_analysis_comprehensive.png")
    print(f"   - comparison_results/gap_*.png")
    print(f"   - comparison_results/arch_comparison_gen.png")

if __name__ == '__main__':
    main() 