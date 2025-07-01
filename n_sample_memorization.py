import numpy as np
import os
import subprocess
import yaml
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys  # add at top alongside other imports
import torch

# Settings
DATA_FILE = 'input_data/samples_rural_6pairs_1fa.npy'
TX_FILE = DATA_FILE.replace('.npy', '_tx_xy.npy')
RX_FILE = DATA_FILE.replace('.npy', '_rx_xy.npy')
CFG_FILE = 'cfgs/config_fa1.yaml'
N_LIST = [1, 2, 5, 10, 100]
EPOCHS = 500
DEVICE = 'mps'
NORMALIZATION = 'global'


def hidden_size_to_string(hidden_size_list):
    """Convert hidden size list to string for filenames"""
    return '_'.join(map(str, hidden_size_list))

def plot_box_vs_N(df, hidden_size, fa):
    # Filter for this network and FA
    df_sub = df[(df['hidden_size'] == str(hidden_size)) & (df['fa'] == fa)]
    N_list = sorted(df_sub['N'].unique())
    mem_data = []
    gen_data = []
    for N in N_list:
        mem_ratios = []
        gen_ratios = []
        for _, row in df_sub[df_sub['N'] == N].iterrows():
            if row['type'] == 'mem':
                mem_ratios.extend(row['ratios'])
            elif row['type'] == 'gen':
                gen_ratios.extend(row['ratios'])
        mem_data.append(mem_ratios)
        gen_data.append(gen_ratios)
    # Plot
    plt.figure(figsize=(10,6))
    positions = np.arange(len(N_list))
    box1 = plt.boxplot(mem_data, positions=positions-0.15, widths=0.25, patch_artist=True, boxprops=dict(facecolor='skyblue'), medianprops=dict(color='navy'))
    box2 = plt.boxplot(gen_data, positions=positions+0.15, widths=0.25, patch_artist=True, boxprops=dict(facecolor='salmon'), medianprops=dict(color='darkred'))
    plt.xticks(positions, N_list)
    plt.xlabel("Number of Training Samples (N)")
    plt.ylabel("DNN/FS Ratio")
    plt.title(f"Memorization vs Generalization\nHidden size: {hidden_size}, FA={fa}")
    plt.legend([box1["boxes"][0], box2["boxes"][0]], ["Memorization", "Generalization"])
    plt.tight_layout()
    plt.savefig(f"box_vs_N_hidden_{'_'.join(map(str,eval(hidden_size)))}_fa{fa}.png")
    plt.close()

def get_fa_from_config(cfg_file):
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg.get('n_fa', 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='N-sample memorization experiment')
    parser.add_argument('--plot_only', action='store_true', help='Only plot results from existing results file, skip all runs')
    parser.add_argument('--target_ratios', type=int, default=None, help='Target number of ratio measurements per N value (overrides in-file setting)')
    parser.add_argument('--hidden_size', nargs='+', type=int, default=[200, 200], help='Hidden layer sizes (e.g., --hidden_size 100 100 or --hidden_size 512 256 128)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs (overrides in-file setting)')
    parser.add_argument('--device', type=str, default='mps', choices=['cpu', 'cuda', 'mps'], help='Training device')
    parser.add_argument('--N', type=int, default=None, help='If set, only run for this N')
    parser.add_argument('--val_npy', type=str, default=None, help='Path to external validation dataset (.npy) for generalization evaluation')
    parser.add_argument('--cfg', type=str, default='cfgs/config_fa1.yaml', help='Path to configuration YAML file for training/validation')
    parser.add_argument('--N_list', nargs='+', type=int, default=None, help='Custom list of N values to iterate over (e.g., --N_list 1 2 5 10 50 100)')
    parser.add_argument('--discrete_power', action='store_true', help='Enable discrete power mode for DNN training and validation.')
    parser.add_argument('--data_file', type=str,
                        default='input_data/samples_rural_6pairs_1fa.npy',
                        help='Path to the .npy dataset of channel gains')
    parser.add_argument('--normalization', type=str, default='global',
                        choices=['global','local','none'],
                        help='Input-normalisation mode passed to train/validate')
    args = parser.parse_args()
    
    # Update settings from arguments, but only override if user provided a value
    TARGET_RATIOS_PER_N = args.target_ratios if args.target_ratios is not None else 200
    EPOCHS = args.epochs if args.epochs is not None else EPOCHS
    DEVICE = args.device
    HIDDEN_SIZE = ' '.join(map(str, args.hidden_size))
    
    # Create a flag for discrete power mode to append to commands
    discrete_power_flag = "--discrete_power" if args.discrete_power else ""
    
    # Optional external validation dataset for generalization
    VAL_NPY_PATH = args.val_npy
    if VAL_NPY_PATH is not None and not os.path.exists(VAL_NPY_PATH):
        print(f"[ERROR] Provided --val_npy file {VAL_NPY_PATH} does not exist.\n" \
              "Generalization evaluation will be skipped.")
        VAL_NPY_PATH = None
    
    # Update N_LIST based on arguments
    if args.N_list is not None:
        N_LIST = args.N_list
    elif args.N is not None:
        N_LIST = [args.N]
    
    # Create filenames with hidden size info
    hidden_str = hidden_size_to_string(args.hidden_size)
    CFG_FILE = args.cfg
    FA = get_fa_from_config(CFG_FILE)
    RESULTS_AGG = f'results_mem_{hidden_str}_fa{FA}.yaml'
    RESULTS_GEN_AGG = f'results_gen_{hidden_str}_fa{FA}.yaml'

    if not args.plot_only:
        # Load full dataset
        samples = np.load(DATA_FILE)
        tx_xy = np.load(TX_FILE)
        rx_xy = np.load(RX_FILE)
        N_TOTAL = samples.shape[0]

        print(f"Starting memorization experiment with hidden_size={args.hidden_size}")
        print(f"Target ratios per N: {TARGET_RATIOS_PER_N}")
        print(f"Epochs: {EPOCHS}, Device: {DEVICE}")
        print(f"Results will be saved to: {RESULTS_AGG}")
        print("="*80)

        agg_results = {}
        gen_results = {}  # For external validation ratios (generalization)

        for N in N_LIST:
            if N > N_TOTAL:
                print(f"Skipping N={N} (not enough samples in dataset: {N_TOTAL})")
                continue
            
            # Calculate number of trials needed to get TARGET_RATIOS_PER_N ratio measurements
            # Each trial with N samples gives us N ratios, so we need TARGET_RATIOS_PER_N / N trials
            n_trials = int(np.ceil(TARGET_RATIOS_PER_N / N))
            expected_ratios = n_trials * N
            
            print(f'For N={N}, running {n_trials} trials to get ~{expected_ratios} ratio measurements (target: {TARGET_RATIOS_PER_N}).')
            agg_results[N] = []
            if VAL_NPY_PATH is not None:
                gen_results[N] = []
            
            for trial in range(n_trials):
                # Randomly select N indices
                idx = np.random.choice(N_TOTAL, N, replace=False)
                # Save temp files with hidden size info
                tmp_prefix = f'tmp_mem_{hidden_str}_N{N}_trial{trial}'
                tmp_data = f'{tmp_prefix}.npy'
                tmp_tx = f'{tmp_prefix}_tx_xy.npy'
                tmp_rx = f'{tmp_prefix}_rx_xy.npy'
                np.save(tmp_data, samples[idx])
                np.save(tmp_tx, tx_xy[idx])
                np.save(tmp_rx, rx_xy[idx])
                
                # Disable DataLoader shuffling (memorisation) and optionally BatchNorm for very small N
                bn_flag = "--no_batch_norm" if N <= 10 else ""
                train_parts = [
                    "python -m src.cli train_dnn",
                    f"--config {CFG_FILE}",
                    f"--epochs {EPOCHS}",
                    "--lr 3e-2",
                    f"--results_path {tmp_prefix}_train.yaml",
                    f"--device {DEVICE}",
                    "--patience 300",
                    "--soft-fa",
                    f"--hidden_size {HIDDEN_SIZE}",
                    f"--train_npy {tmp_data}",
                    f"--n_train_samples {N}",
                    f"--normalization {NORMALIZATION}",
                    f"--save_path {tmp_prefix}_weights.pt",
                    "--no_shuffle",
                    bn_flag,
                    discrete_power_flag
                ]
                train_cmd = " ".join([p for p in train_parts if p]).strip()
                print(f'Running: {train_cmd}')
                try:
                    subprocess.run(train_cmd, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Training failed for N={N}, trial={trial}: {e}")
                    continue
                
                # Validate
                val_parts = [
                    "python -m src.cli validate_dnn",
                    f"--config {CFG_FILE}",
                    f"--weights {tmp_prefix}_weights.pt",
                    f"--n_val_samples {N}",
                    f"--results_path {tmp_prefix}_val.yaml",
                    f"--val_npy {tmp_data}",
                    f"--normalization {NORMALIZATION}",
                    bn_flag,
                    discrete_power_flag
                ]
                val_cmd = " ".join([p for p in val_parts if p]).strip()
                print(f'Running: {val_cmd}')
                try:
                    subprocess.run(val_cmd, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Validation failed for N={N}, trial={trial}: {e}")
                    continue
                
                # Parse validation YAML
                try:
                    with open(f'{tmp_prefix}_val.yaml', 'r') as f:
                        val_results = yaml.safe_load(f)
                    ratios = [s['ratio'] for s in val_results['samples']]
                    agg_results[N].extend(ratios)
                    print(f'N={N}, trial={trial+1}/{n_trials}, collected {len(ratios)} ratios, total so far: {len(agg_results[N])}')
                except Exception as e:
                    print(f"Failed to parse results for N={N}, trial={trial}: {e}")
                    continue
                
                # ---------------------------------------------------
                # Generalization evaluation on external VAL_NPY_PATH
                # ---------------------------------------------------
                if VAL_NPY_PATH is not None:
                    # --- Generalization on N *unseen* samples per trial ---
                    # Load external validation data once
                    if 'VAL_DATA' not in globals():
                        VAL_DATA      = np.load(VAL_NPY_PATH)
                        try:
                            VAL_TX_DATA = np.load(VAL_NPY_PATH.replace('.npy', '_tx_xy.npy'))
                            VAL_RX_DATA = np.load(VAL_NPY_PATH.replace('.npy', '_rx_xy.npy'))
                        except FileNotFoundError:
                            VAL_TX_DATA = VAL_RX_DATA = None
                        VAL_TOTAL = VAL_DATA.shape[0]

                    # Randomly pick N indices (without replacement within this trial)
                    idx_gen = np.random.choice(VAL_TOTAL, N, replace=False)
                    tmp_gen_prefix = f'{tmp_prefix}_gen'
                    tmp_gen_data   = f'{tmp_gen_prefix}.npy'
                    np.save(tmp_gen_data, VAL_DATA[idx_gen])
                    if VAL_TX_DATA is not None:
                        np.save(f'{tmp_gen_prefix}_tx_xy.npy', VAL_TX_DATA[idx_gen])
                        np.save(f'{tmp_gen_prefix}_rx_xy.npy', VAL_RX_DATA[idx_gen])

                    gen_res_path = f'{tmp_prefix}_gen.yaml'
                    gen_parts = [
                        "python -m src.cli validate_dnn",
                        f"--config {CFG_FILE}",
                        f"--weights {tmp_prefix}_weights.pt",
                        f"--n_val_samples {N}",
                        f"--results_path {gen_res_path}",
                        f"--val_npy {tmp_gen_data}",
                        f"--normalization {NORMALIZATION}",
                        bn_flag,
                        discrete_power_flag
                    ]
                    gen_cmd = " ".join([p for p in gen_parts if p]).strip()
                    print(f'Running (generalization): {gen_cmd}')
                    try:
                        subprocess.run(gen_cmd, shell=True, check=True)
                        with open(gen_res_path, 'r') as gf:
                            gen_yaml = yaml.safe_load(gf)
                        gen_ratios = [s['ratio'] for s in gen_yaml['samples']]
                        gen_results[N].extend(gen_ratios)
                    except subprocess.CalledProcessError as e:
                        print(f"Generalization validation failed for N={N}, trial={trial}: {e}")
                    except Exception as e:
                        print(f"Failed to parse generalization results for N={N}, trial={trial}: {e}")
                
                # Clean up temp files (weights are removed at the end to reuse if VAL_NPY not used)
                cleanup_paths = [tmp_data, tmp_tx, tmp_rx, f'{tmp_prefix}_train.yaml', f'{tmp_prefix}_val.yaml']
                if VAL_NPY_PATH is not None:
                    cleanup_paths.append(f'{tmp_prefix}_gen.yaml')
                # Also remove weights after gen eval (or immediately if no gen eval)
                cleanup_paths.extend([f'{tmp_prefix}_weights.pt', f'{tmp_prefix}_weights.pt.meta.yaml'])
                for fpath in cleanup_paths:
                    if os.path.exists(fpath):
                        os.remove(fpath)
                
                # Early stopping if we have enough ratios
                if len(agg_results[N]) >= TARGET_RATIOS_PER_N:
                    print(f'Reached target of {TARGET_RATIOS_PER_N} ratios for N={N}, stopping early.')
                    break
            
            # Trim to exact target if we have more than needed
            if len(agg_results[N]) > TARGET_RATIOS_PER_N:
                agg_results[N] = agg_results[N][:TARGET_RATIOS_PER_N]
                print(f'Trimmed N={N} to exactly {TARGET_RATIOS_PER_N} ratios.')

            # Trim generalization ratios as well to match target
            if VAL_NPY_PATH is not None and N in gen_results and len(gen_results[N]) > TARGET_RATIOS_PER_N:
                gen_results[N] = gen_results[N][:TARGET_RATIOS_PER_N]
                print(f'Trimmed generalization N={N} to exactly {TARGET_RATIOS_PER_N} ratios.')

        # -------------------------------------------------------
        # Save (or merge into) aggregate YAML so that all N values
        # are kept in a single file. If the YAML already exists we
        # load it, update only the entries for the N we just ran,
        # and write it back.  This prevents the file from being
        # overwritten every time `n_sample_memorization.py` is called
        # with a different --N.
        # -------------------------------------------------------

        # Build metadata for *this* run
        new_metadata = {
            'hidden_size': args.hidden_size,
            'target_ratios_per_n': TARGET_RATIOS_PER_N,
            'epochs': EPOCHS,
            'device': DEVICE,
            'normalization': NORMALIZATION,
            'data_file': DATA_FILE,
            'fa': FA,
        }

        # Load existing YAML (if any)
        if os.path.exists(RESULTS_AGG):
            try:
                with open(RESULTS_AGG, 'r') as f:
                    existing = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"[WARNING] Failed to load existing results file {RESULTS_AGG}: {e}. Creating a new one.")
                existing = {}
            existing_results = existing.get('results', {})
            existing_metadata = existing.get('metadata', {})
        else:
            existing_results = {}
            existing_metadata = {}

        # Merge/overwrite the results for each N we processed in this run
        for N, ratios in agg_results.items():
            # Use int keys in YAML for cleaner output
            existing_results[int(N)] = ratios

        # Merge metadata (existing keys are preserved unless we have new values)
        merged_metadata = {**existing_metadata, **new_metadata}

        # Write back to YAML
        with open(RESULTS_AGG, 'w') as f:
            yaml.safe_dump({'metadata': merged_metadata, 'results': existing_results}, f)

        print(f'Aggregated results saved/updated to {RESULTS_AGG}')

        # ------------------ Save generalization YAML ------------------
        if VAL_NPY_PATH is not None and gen_results:
            # Load existing file if present
            if os.path.exists(RESULTS_GEN_AGG):
                try:
                    with open(RESULTS_GEN_AGG, 'r') as fg:
                        existing_gen = yaml.safe_load(fg) or {}
                except Exception as e:
                    print(f"[WARNING] Failed to load existing gen results {RESULTS_GEN_AGG}: {e}. Creating new.")
                    existing_gen = {}
                existing_gen_results = existing_gen.get('results', {})
                existing_gen_meta    = existing_gen.get('metadata', {})
            else:
                existing_gen_results = {}
                existing_gen_meta    = {}

            # Merge
            for N, ratios in gen_results.items():
                existing_gen_results[int(N)] = ratios

            merged_meta_gen = {**existing_gen_meta, **new_metadata}
            with open(RESULTS_GEN_AGG, 'w') as fg:
                yaml.safe_dump({'metadata': merged_meta_gen, 'results': existing_gen_results}, fg)
            print(f'Generalization results saved/updated to {RESULTS_GEN_AGG}')

    # Visualization step (always run if plot_only, or after experiments)
    if not os.path.exists(RESULTS_AGG):
        print(f'{RESULTS_AGG} not found. Run experiments first.')
    else:
        with open(RESULTS_AGG, 'r') as f:
            data = yaml.safe_load(f)
        
        # Handle both old format (direct results) and new format (with metadata)
        if 'results' in data:
            agg_results = data['results']
            metadata = data.get('metadata', {})
            hidden_size_info = metadata.get('hidden_size', args.hidden_size)
        else:
            agg_results = data
            hidden_size_info = args.hidden_size

        # Prepare data for plotting
        N_list = []
        ratios_list = []
        means = []
        stds = []
        medians = []
        q25s = []
        q75s = []
        
        data = []
        labels = []
        for N in sorted(agg_results.keys(), key=lambda x: int(x)):
            ratios = agg_results[N]
            if len(ratios) == 0:
                continue
            data.append(ratios)
            labels.append(f'N={N}')
            N_list.extend([int(N)] * len(ratios))
            ratios_list.extend(ratios)
            means.append(np.mean(ratios))
            stds.append(np.std(ratios))
            medians.append(np.median(ratios))
            q25s.append(np.percentile(ratios, 25))
            q75s.append(np.percentile(ratios, 75))

        # Guard: if no data collected, skip plotting to avoid errors
        if len(data) == 0:
            print("[WARNING] No ratio data collected. Skipping plotting step.")
            sys.exit(0)

        # Create comprehensive visualization with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Add overall title with hidden size info
        hidden_size_str = hidden_size_to_string(hidden_size_info)
        fig.suptitle(f'DNN Memorization Analysis - Hidden Size: {hidden_size_info}', fontsize=16, fontweight='bold')
        
        # 1. Boxplot (top-left)
        ax1 = axes[0, 0]
        sns.boxplot(data=data, ax=ax1)
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels)
        ax1.set_ylabel('DNN/FS Sum-Rate Ratio')
        ax1.set_xlabel('Number of Memorized Samples (N)')
        ax1.set_title('Boxplot: DNN Memorization Performance vs. Number of Samples')
        ax1.grid(True, alpha=0.3)

        # 2. Violin plot (top-right)
        ax2 = axes[0, 1]
        sns.violinplot(data=data, ax=ax2)
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels)
        ax2.set_ylabel('DNN/FS Sum-Rate Ratio')
        ax2.set_xlabel('Number of Memorized Samples (N)')
        ax2.set_title('Violin Plot: Distribution Shape Comparison')
        ax2.grid(True, alpha=0.3)

        # 3. Distribution overlays (bottom-left)
        ax3 = axes[1, 0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
        for i, (ratios, label, color) in enumerate(zip(data, labels, colors)):
            ax3.hist(ratios, bins=20, alpha=0.6, label=label, color=color, density=True)
        ax3.set_xlabel('DNN/FS Sum-Rate Ratio')
        ax3.set_ylabel('Density')
        ax3.set_title('Distribution Overlays')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Summary statistics plot (bottom-right)
        ax4 = axes[1, 1]
        N_values = [int(N) for N in sorted(agg_results.keys(), key=lambda x: int(x)) if len(agg_results[N])>0]
        
        ax4.errorbar(N_values, means, yerr=stds, marker='o', label='Mean ± Std', capsize=5, capthick=2)
        ax4.plot(N_values, medians, marker='s', label='Median', linestyle='--')
        ax4.fill_between(N_values, q25s, q75s, alpha=0.3, label='IQR (25%-75%)')
        
        ax4.set_xlabel('Number of Memorized Samples (N)')
        ax4.set_ylabel('DNN/FS Sum-Rate Ratio')
        ax4.set_title('Statistical Summary Across N')
        ax4.set_xscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        
        # Save with hidden size in filename
        plot_filename = f'n_sample_memorization_{hidden_size_str}_distribution_analysis_equal.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f'Distribution analysis saved to {plot_filename}')

        # Create separate detailed distribution comparison plot
        plt.figure(figsize=(12, 8))
        
        # Create subplots for each N value
        n_cols = 3
        n_rows = int(np.ceil(len(data) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Add overall title
        fig.suptitle(f'Detailed Distribution Analysis - Hidden Size: {hidden_size_info}', fontsize=14, fontweight='bold')
        
        non_empty_keys = [k for k in sorted(agg_results.keys(), key=lambda x: int(x)) if len(agg_results[k])>0]
        for i, N in enumerate(non_empty_keys):
            ratios = np.array(agg_results[N])
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Histogram with KDE overlay
            ax.hist(ratios, bins=15, alpha=0.7, density=True, color='skyblue', edgecolor='black')
            
            # Add KDE if we have enough samples
            if len(ratios) > 5:
                from scipy import stats
                kde = stats.gaussian_kde(ratios)
                x_range = np.linspace(min(ratios), max(ratios), 100)
                ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            
            # Add vertical lines for statistics
            ax.axvline(np.mean(ratios), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ratios):.3f}')
            ax.axvline(np.median(ratios), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(ratios):.3f}')
            
            ax.set_title(f'{labels[i]}\n(n={len(ratios)}, std={np.std(ratios):.3f})')
            ax.set_xlabel('DNN/FS Sum-Rate Ratio')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(data), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        detailed_plot_filename = f'n_sample_memorization_{hidden_size_str}_detailed_distributions_equal.png'
        plt.savefig(detailed_plot_filename, dpi=300, bbox_inches='tight')
        print(f'Detailed distributions saved to {detailed_plot_filename}')

        # Print comprehensive summary stats
        print("\n" + "="*80)
        print(f"DISTRIBUTION ANALYSIS SUMMARY - Hidden Size: {hidden_size_info}")
        print("="*80)
        
        non_empty_keys = [k for k in sorted(agg_results.keys(), key=lambda x: int(x)) if len(agg_results[k])>0]
        for i, N in enumerate(non_empty_keys):
            ratios = np.array(agg_results[N])
            print(f'\nN={N} (n_samples={len(ratios)}):')
            print(f'  Mean:     {means[i]:.4f}')
            print(f'  Median:   {medians[i]:.4f}')
            print(f'  Std:      {stds[i]:.4f}')
            print(f'  Min:      {np.min(ratios):.4f}')
            print(f'  Max:      {np.max(ratios):.4f}')
            print(f'  Q25:      {q25s[i]:.4f}')
            print(f'  Q75:      {q75s[i]:.4f}')
            print(f'  IQR:      {q75s[i] - q25s[i]:.4f}')
            print(f'  Range:    {np.max(ratios) - np.min(ratios):.4f}')
            
            # Count samples in different performance ranges
            excellent = np.sum(ratios >= 0.95)
            good = np.sum((ratios >= 0.90) & (ratios < 0.95))
            fair = np.sum((ratios >= 0.80) & (ratios < 0.90))
            poor = np.sum(ratios < 0.80)
            
            print(f'  Performance distribution:')
            print(f'    Excellent (≥0.95): {excellent:3d} ({100*excellent/len(ratios):5.1f}%)')
            print(f'    Good (0.90-0.95):  {good:3d} ({100*good/len(ratios):5.1f}%)')
            print(f'    Fair (0.80-0.90):  {fair:3d} ({100*fair/len(ratios):5.1f}%)')
            print(f'    Poor (<0.80):      {poor:3d} ({100*poor/len(ratios):5.1f}%)')

        # ------------------------------------------------------------------
        # Statistical comparison section skipped per user request.
        # ------------------------------------------------------------------

        # Only proceed if results exist
        # df_partial = collect_results()
        # if df_partial.empty:
        #     print(f"[WARNING] No results found for hidden size {fa}. Skipping plotting.")
        # else:
        #     plot_summary(df_partial)
        #     plt.savefig(f"capacity_mean_std_after_{fa}.png")
        #     for fa in sorted(df_partial['fa'].unique()):
        #         plot_box_vs_N(df_partial, fa, fa)
        #     print(f"[INFO] Completed hidden size {fa}. Summary updated and plots refreshed.")