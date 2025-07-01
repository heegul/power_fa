import pandas as pd
import yaml
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

def parse_filename(filepath):
    filename = Path(filepath).name
    match = re.match(r'results_(gen|mem)_([a-zA-Z0-9_]+?)_fa1.yaml', filename)
    if not match:
        return None
    
    test_type, architecture = match.groups()
    if architecture.endswith('_'):
        architecture = architecture[:-1]
    if not architecture: # Handle cases like 'results_mem_fa1.yaml'
        return None

    return {'type': 'Generalization' if test_type == 'gen' else 'Memorization', 'architecture': architecture}

def load_data():
    all_rows = []
    result_files = glob.glob('results_*.yaml')
    print(f'Found {len(result_files)} result files to process.')

    for f in result_files:
        try:
            metadata = parse_filename(f)
            if metadata is None:
                continue

            with open(f, 'r') as file:
                content = file.read()
                if not content.strip():
                    continue
                data = yaml.safe_load(content)

            if not isinstance(data, dict) or 'results' not in data:
                continue

            # Process the nested dictionary structure
            for n_samples, ratios in data['results'].items():
                for ratio in ratios:
                    all_rows.append({
                        'N': n_samples,
                        'dnn_fs_ratio': ratio,
                        'type': metadata['type'],
                        'architecture': metadata['architecture']
                    })
        except Exception as e:
            print(f'Error processing file {f}: {e}')
    
    if not all_rows:
        print('No valid data rows could be created.')
        return pd.DataFrame()

    return pd.DataFrame(all_rows)

def calculate_param_size(architecture_str, n_pairs=6, n_fa=1, discrete_power=False, power_levels=4):
    """
    Calculates the approximate number of parameters in a D2DNet model.
    Assumes n_pairs=6, n_fa=1 based on typical experimental setup.
    """
    try:
        if isinstance(architecture_str, str):
            hidden_sizes = [int(s) for s in architecture_str.split('_') if s.isdigit()]
        elif isinstance(architecture_str, (int, float)):
            hidden_sizes = [int(architecture_str)]
        else:
            return 0
            
        if not hidden_sizes:
            return 0

        input_size = n_pairs * n_pairs
        
        # Output size calculation
        if discrete_power:
            power_output_size = n_pairs * power_levels
        else:
            power_output_size = n_pairs
        
        fa_output_size = 0
        if n_fa > 1:
            fa_output_size = n_pairs * n_fa
        
        # For simplicity in this analysis, we'll focus on the main body of the network
        # and not the heads, as the 'architecture' string primarily defines the hidden layers.
        # We will calculate params for the backbone and the power head (assuming n_fa=1).
        
        layer_sizes = [input_size] + hidden_sizes
        total_params = 0
        
        # Hidden layers
        for i in range(len(layer_sizes) - 1):
            total_params += layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1] # weights + biases
            
        # Output layer (power head)
        output_size = n_pairs
        total_params += layer_sizes[-1] * output_size + output_size

        return total_params
    except Exception:
        return 0

def plot_generalization_gap(df):
    def arch_sort_key(arch):
        parts = [int(p) for p in str(arch).split('_') if p.isdigit()]
        return (len(parts), sum(parts), arch)
    architectures = sorted(df['architecture'].unique(), key=arch_sort_key)
    output_dir = Path('comparison_results')
    output_dir.mkdir(exist_ok=True)

    for arch in architectures:
        plt.figure(figsize=(10, 6))
        subset_df = df[df['architecture'] == arch]
        if subset_df.empty or len(subset_df['type'].unique()) < 2:
            plt.close()
            continue
        
        sns.lineplot(data=subset_df, x='N', y='dnn_fs_ratio', hue='type', marker='o', errorbar='sd')
        plot_title = f"Generalization vs. Memorization\nArchitecture: {str(arch).replace('_', 'x')}"
        plt.title(plot_title)
        plt.xlabel('Number of Training Samples (N)')
        plt.ylabel('DNN/FS Sum-Rate Ratio')
        plt.xscale('log')
        plt.grid(True, which='both', ls='--')
        plt.axhline(1.0, color='r', linestyle='--', label='FS Performance Baseline')
        plt.legend(title=None)
        output_filename = output_dir / f'gap_{arch}.png'
        plt.savefig(output_filename, bbox_inches='tight')
        print(f'Saved plot: {output_filename}')
        plt.close()

def plot_performance_vs_params_by_N(df):
    """
    Creates separate plots for each N value, showing DNN/FS ratio vs. model parameter size.
    """
    output_dir = Path('comparison_results')
    output_dir.mkdir(exist_ok=True)
    
    unique_N = sorted(df['N'].unique())
    
    for n_val in unique_N:
        plt.figure(figsize=(12, 7))
        subset_df = df[df['N'] == n_val].sort_values('param_size')
        
        if subset_df.empty:
            continue

        sns.lineplot(data=subset_df, x='param_size', y='dnn_fs_ratio', hue='type', marker='o', errorbar='sd')
        
        plt.title(f'Performance vs. Model Size (N = {n_val})')
        plt.xlabel('Number of Model Parameters')
        plt.ylabel('DNN/FS Sum-Rate Ratio')
        plt.xscale('log')
        plt.grid(True, which="both", ls="--")
        plt.axhline(1.0, color='r', linestyle='--', label='FS Performance Baseline')
        plt.legend(title='Test Type')
        
        output_filename = output_dir / f'perf_vs_params_N{n_val}.png'
        plt.savefig(output_filename, bbox_inches='tight')
        print(f'Saved plot: {output_filename}')
        plt.close()

def plot_arch_comparison(df):
    gen_df = df[df['type'] == 'Generalization'].copy()
    if gen_df.empty:
        print('No generalization data to plot for architecture comparison.')
        return
        
    def arch_sort_key(arch):
        parts = [int(p) for p in str(arch).split('_') if p.isdigit()]
        return (len(parts), sum(parts), arch)
    arch_order = sorted(gen_df['architecture'].unique(), key=arch_sort_key)

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=gen_df, x='N', y='dnn_fs_ratio', hue='architecture', hue_order=arch_order, marker='o', errorbar='sd')
    plt.title('Generalization Performance by Model Architecture')
    plt.xlabel('Number of Training Samples (N)')
    plt.ylabel('DNN/FS Sum-Rate Ratio (Generalization)')
    plt.xscale('log')
    plt.grid(True, which='both', ls='--')
    plt.axhline(1.0, color='r', linestyle='--', label='FS Performance Baseline')
    plt.legend(title='Architecture', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    output_dir = Path('comparison_results')
    output_dir.mkdir(exist_ok=True)
    output_filename = output_dir / 'arch_comparison_gen.png'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f'Saved plot: {output_filename}')
    plt.close()

if __name__ == '__main__':
    sns.set_theme(style='whitegrid')
    df = load_data()
    if not df.empty:
        df['N'] = pd.to_numeric(df['N'], errors='coerce')
        df['dnn_fs_ratio'] = pd.to_numeric(df['dnn_fs_ratio'], errors='coerce')
        df.dropna(subset=['N', 'dnn_fs_ratio'], inplace=True)
        df['N'] = df['N'].astype(int)

        # Calculate parameter size
        df['param_size'] = df['architecture'].apply(calculate_param_size)
        df = df[df['param_size'] > 0] # Remove entries where param size could not be calculated

        print('\n--- Data Summary ---')
        summary = df.groupby(['architecture', 'param_size', 'type', 'N'])['dnn_fs_ratio'].agg(['mean', 'std', 'count']).round(3)
        print(summary)
        print('-' * 20)
        
        print('\n--- Generating Plots ---')
        plot_generalization_gap(df)
        plot_arch_comparison(df)
        plot_performance_vs_params_by_N(df)
        print('--- Analysis Complete ---')
    else:
        print('Could not load any valid data for analysis.')