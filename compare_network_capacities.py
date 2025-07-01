import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import glob
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

# Define a consistent color palette for architectures
COLOR_PALETTE = sns.color_palette("husl", 19)

def calculate_parameters(hidden_sizes, input_size=12, output_size=6):
    """Calculate total number of parameters in the network."""
    sizes = [input_size] + hidden_sizes + [output_size]
    total_params = 0
    for i in range(len(sizes) - 1):
        total_params += sizes[i] * sizes[i+1] + sizes[i+1]
    return total_params

def load_all_results(directory="."):
    """Load all experiment results from YAML files in the given directory."""
    data = []
    pattern = os.path.join(directory, "n_sample_memorization_results_*.yaml")
    files = glob.glob(pattern)
    if not files:
        print(f"Warning: No result files found matching '{pattern}'.")
        return pd.DataFrame()

    for f in files:
        with open(f, 'r') as stream:
            try:
                content = yaml.safe_load(stream)
                if not content:
                    print(f"Warning: File {f} is empty. Skipping.")
                    continue

                hidden_size_str = content.get('hidden_size', [])
                if not hidden_size_str:
                    # Fallback for older format
                    try:
                        hidden_size_str = os.path.basename(f).replace('n_sample_memorization_results_', '').replace('.yaml', '')
                        hidden_size = [int(s) for s in hidden_size_str.split('_')]
                    except (ValueError, IndexError):
                        print(f"Warning: Could not determine hidden_size from filename {f}. Skipping.")
                        continue
                else:
                    hidden_size = list(map(int, hidden_size_str))


                arch_name = '_'.join(map(str, hidden_size))
                total_params = content.get('total_parameters', calculate_parameters(hidden_size))
                
                results = content.get('results', {})
                if not results:
                    print(f"Warning: No 'results' key in {f}. Skipping.")
                    continue
                
                for n_samples_key, sample_data in results.items():
                    n_samples = int(str(n_samples_key).replace('_samples', ''))
                    
                    if isinstance(sample_data, dict) and 'ratios' in sample_data:
                        ratios = sample_data['ratios']
                    elif isinstance(sample_data, list):
                        ratios = sample_data
                    else:
                        continue

                    for ratio in ratios:
                        data.append({
                            'arch_name': arch_name,
                            'hidden_size': tuple(hidden_size),
                            'params': total_params,
                            'n_samples': n_samples,
                            'ratio': ratio
                        })

            except yaml.YAMLError as e:
                print(f"Error parsing YAML file {f}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred processing {f}: {e}")
                
    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)

def plot_overall_performance(df, output_dir, architectures):
    """Generates and saves a boxplot of overall performance distributions."""
    if df.empty:
        print("Skipping overall performance plot: No data.")
        return

    plt.figure(figsize=(15, 10))
    sns.boxplot(x='arch_name', y='ratio', data=df, palette=COLOR_PALETTE, order=architectures)
    plt.title('Overall Performance Distribution by Architecture', fontsize=18, fontweight='bold')
    plt.xlabel('Network Architecture (Hidden Layers)', fontsize=14)
    plt.ylabel('Sum-Rate Ratio (DNN / Full-CSI)', fontsize=14)
    plt.xticks(rotation=90, ha='center', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, "1_overall_performance_distribution.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")

def plot_performance_by_n(df, output_dir, architectures):
    """Generates and saves separate bar plots of mean performance for each N value."""
    if df.empty:
        print("Skipping performance by N plot: No data.")
        return

    n_values = sorted(df['n_samples'].unique())
    
    for n in n_values:
        plt.figure(figsize=(15, 8))
        subset = df[df['n_samples'] == n]
        
        if subset.empty:
            continue

        sns.barplot(x='arch_name', y='ratio', data=subset, palette=COLOR_PALETTE, order=architectures, ci=None)
        
        plt.title(f'Mean Performance for N = {n} Samples', fontsize=16, fontweight='bold')
        plt.xlabel('Network Architecture (Hidden Layers)', fontsize=12)
        plt.ylabel('Mean Sum-Rate Ratio', fontsize=12)
        plt.xticks(rotation=90, ha='right')
        plt.ylim(0.5, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f"2_mean_performance_N_{n}.png")
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved: {filename}")


def plot_performance_stability_bubble(df, output_dir, architectures):
    """
    Generates and saves a bubble chart showing performance vs. parameters,
    with bubble size representing the number of samples.
    """
    if df.empty:
        print("Skipping performance stability plot: No data.")
        return

    agg_df = df.groupby(['arch_name', 'params', 'n_samples']).ratio.mean().reset_index()
    
    arch_color_map = {arch: color for arch, color in zip(architectures, COLOR_PALETTE)}
    
    plt.figure(figsize=(16, 10))
    
    # Normalize bubble sizes for better visualization
    min_size, max_size = 50, 500
    n_min, n_max = agg_df['n_samples'].min(), agg_df['n_samples'].max()
    if n_max == n_min:
        sizes = [min_size] * len(agg_df)
    else:
        sizes = min_size + (agg_df['n_samples'] - n_min) / (n_max - n_min) * (max_size - min_size)

    scatter = plt.scatter(agg_df['params'], agg_df['ratio'], s=sizes, c=agg_df['arch_name'].map(arch_color_map), alpha=0.7, edgecolors='w', linewidth=0.5)

    plt.xscale('log')
    plt.xlabel('Number of Parameters (log scale)', fontsize=14)
    plt.ylabel('Mean Sum-Rate Ratio', fontsize=14)
    plt.title('Performance vs. Network Size (Bubble Size Corresponds to N Samples)', fontsize=18, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.ylim(bottom=max(0, df['ratio'].min() - 0.1))

    # Create a legend for architectures (colors)
    legend_handles_color = [plt.Line2D([0], [0], marker='o', color='w', label=f'{arch} ({params:,})', 
                                  markerfacecolor=color, markersize=10) 
                            for arch, params, color in sorted(list(set(zip(agg_df['arch_name'], agg_df['params'], agg_df['arch_name'].map(arch_color_map)))), key=lambda x: x[1])]
    leg1 = plt.legend(handles=legend_handles_color, title="Architectures", bbox_to_anchor=(1.04, 1), loc='upper left')
    
    # Create a legend for N samples (bubble size)
    n_unique_sorted = sorted(df['n_samples'].unique())
    legend_handles_size = []
    for n in n_unique_sorted:
        size = min_size + (n - n_min) / (n_max - n_min) * (max_size - min_size) if n_max > n_min else min_size
        legend_handles_size.append(plt.scatter([], [], s=size, label=str(n), color='grey', alpha=0.7))
    
    leg2 = plt.legend(handles=legend_handles_size, title="N Samples", bbox_to_anchor=(1.04, 0.4), loc='center left', scatterpoints=1)
    
    plt.gca().add_artist(leg1)
    
    plt.tight_layout(rect=[0, 0, 0.8, 1]) # Adjust rect to make space for legends
    
    filename = os.path.join(output_dir, "3_performance_stability_bubble.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")


def plot_parameter_efficiency(df, output_dir, architectures):
    """Generates and saves a scatter plot of performance vs. parameter count."""
    if df.empty:
        print("Skipping parameter efficiency plot: No data.")
        return

    arch_summary = df.groupby(['arch_name', 'params'])['ratio'].mean().reset_index()
    arch_summary = arch_summary.sort_values('params')

    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=arch_summary, x='params', y='ratio', hue='arch_name', s=150, palette=COLOR_PALETTE,
                    hue_order=architectures)
    
    plt.xscale('log')
    plt.title('Overall Mean Performance vs. Parameter Count', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Parameters (log scale)', fontsize=12)
    plt.ylabel('Overall Mean Sum-Rate Ratio', fontsize=12)
    plt.grid(True, which="both", ls="--")
    plt.legend(title='Architecture', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    filename = os.path.join(output_dir, "4_parameter_efficiency.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")


def main():
    """Main function to load data and generate plots."""
    output_directory = "comparison_results"
    os.makedirs(output_directory, exist_ok=True)
    
    print("Loading experiment results...")
    df = load_all_results()

    if df.empty:
        print("No data found to generate plots. Exiting.")
        return
    
    # Sort architectures by number of parameters
    architectures = sorted(df['arch_name'].unique(), key=lambda s: calculate_parameters([int(x) for x in s.split('_')]))
    
    print(f"\nFound {len(df)} total data points from {len(df['arch_name'].unique())} architectures.")
    print("Architectures found:", architectures)
    
    print("\n--- Generating Plots ---")
    plot_overall_performance(df, output_directory, architectures)
    plot_performance_by_n(df, output_directory, architectures)
    plot_performance_stability_bubble(df, output_directory, architectures)
    plot_parameter_efficiency(df, output_directory, architectures)
    
    print("\n--- Statistical Analysis ---")
    if len(architectures) >= 2:
        print("\nMann-Whitney U Test (comparing smallest and largest architecture):")
        arch1_data = df[df['arch_name'] == architectures[0]]['ratio']
        arch_last_data = df[df['arch_name'] == architectures[-1]]['ratio']
        
        if not arch1_data.empty and not arch_last_data.empty:
            stat, p_value = mannwhitneyu(arch1_data, arch_last_data, alternative='two-sided')
            print(f"Comparing '{architectures[0]}' vs '{architectures[-1]}':")
            print(f"  Statistic = {stat:.2f}, P-value = {p_value:.4f}")
            if p_value < 0.05:
                print("  Result: Statistically significant difference.")
            else:
                print("  Result: No statistically significant difference.")

    print("\n--- Analysis Complete ---")
    print(f"All plots saved in '{output_directory}' directory.")


if __name__ == '__main__':
    main() 