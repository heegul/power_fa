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

# Settings for demo
DATA_FILE = 'input_data/samples_rural_6pairs_1fa.npy'
TX_FILE = DATA_FILE.replace('.npy', '_tx_xy.npy')
RX_FILE = DATA_FILE.replace('.npy', '_rx_xy.npy')
CFG_FILE = 'cfgs/config_fa1.yaml'
N_LIST = [1, 2, 5, 10]  # Smaller list for demo
EPOCHS = 500  # Fewer epochs for faster demo
DEVICE = 'mps'  # Use CPU for demo
NORMALIZATION = 'global'
TARGET_RATIOS_PER_N = 20  # Small target for demo

def hidden_size_to_string(hidden_size_list):
    """Convert hidden size list to string for filenames"""
    return '_'.join(map(str, hidden_size_list))

def get_fa_from_config(cfg_file):
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg.get('n_fa', 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='N-sample memorization experiment (demo version)')
    parser.add_argument('--plot_only', action='store_true', help='Only plot results, skip all runs')
    parser.add_argument('--target_ratios', type=int, default=20, help='Target number of ratio measurements per N value')
    parser.add_argument('--hidden_size', nargs='+', type=int, default=[100, 100], help='Hidden layer sizes (e.g., --hidden_size 50 50 or --hidden_size 200 100)')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='Training device')
    args = parser.parse_args()
    
    # Update settings from arguments
    TARGET_RATIOS_PER_N = args.target_ratios
    EPOCHS = args.epochs
    DEVICE = args.device
    HIDDEN_SIZE = ' '.join(map(str, args.hidden_size))
    
    FA = get_fa_from_config(CFG_FILE)
    hidden_str = hidden_size_to_string(args.hidden_size)
    RESULTS_AGG = f'results_mem_{hidden_str}_fa{FA}_demo.yaml'

    if not args.plot_only:
        # Load full dataset
        samples = np.load(DATA_FILE)
        tx_xy = np.load(TX_FILE)
        rx_xy = np.load(RX_FILE)
        N_TOTAL = samples.shape[0]

        print(f"Starting demo memorization experiment with hidden_size={args.hidden_size}")
        print(f"Target ratios per N: {TARGET_RATIOS_PER_N}")
        print(f"Epochs: {EPOCHS}, Device: {DEVICE}")
        print(f"Results will be saved to: {RESULTS_AGG}")
        print("="*60)

        agg_results = {}

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
            
            for trial in range(n_trials):
                print(f'  Trial {trial+1}/{n_trials} for N={N}...')
                
                # Randomly select N indices
                idx = np.random.choice(N_TOTAL, N, replace=False)
                # Save temp files with hidden size info
                tmp_prefix = f'tmp_demo_{hidden_str}_N{N}_trial{trial}'
                tmp_data = f'{tmp_prefix}.npy'
                tmp_tx = f'{tmp_prefix}_tx_xy.npy'
                tmp_rx = f'{tmp_prefix}_rx_xy.npy'
                np.save(tmp_data, samples[idx])
                np.save(tmp_tx, tx_xy[idx])
                np.save(tmp_rx, rx_xy[idx])
                
                # Train
                train_cmd = f"python -m src.cli train_dnn --config {CFG_FILE} --epochs {EPOCHS} --lr 3e-2 --results_path {tmp_prefix}_train.yaml --device {DEVICE} --patience=50 --soft-fa --hidden_size {HIDDEN_SIZE} --train_npy {tmp_data} --n_train_samples {N} --normalization {NORMALIZATION} --save_path {tmp_prefix}_weights.pt"
                try:
                    subprocess.run(train_cmd, shell=True, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"    Training failed for N={N}, trial={trial}: {e}")
                    continue
                
                # Validate
                val_cmd = f"python -m src.cli validate_dnn --config {CFG_FILE} --weights {tmp_prefix}_weights.pt --n_val_samples {N} --results_path {tmp_prefix}_val.yaml --val_npy {tmp_data} --normalization {NORMALIZATION}"
                try:
                    subprocess.run(val_cmd, shell=True, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"    Validation failed for N={N}, trial={trial}: {e}")
                    continue
                
                # Parse validation YAML
                try:
                    with open(f'{tmp_prefix}_val.yaml', 'r') as f:
                        val_results = yaml.safe_load(f)
                    ratios = [s['ratio'] for s in val_results['samples']]
                    agg_results[N].extend(ratios)
                    print(f'    Collected {len(ratios)} ratios, total so far: {len(agg_results[N])}')
                except Exception as e:
                    print(f"    Failed to parse results for N={N}, trial={trial}: {e}")
                    continue
                
                # Clean up temp files
                for fpath in [tmp_data, tmp_tx, tmp_rx, f'{tmp_prefix}_weights.pt', f'{tmp_prefix}_weights.pt.meta.yaml', f'{tmp_prefix}_train.yaml', f'{tmp_prefix}_val.yaml']:
                    if os.path.exists(fpath):
                        os.remove(fpath)
                
                # Early stopping if we have enough ratios
                if len(agg_results[N]) >= TARGET_RATIOS_PER_N:
                    print(f'    Reached target of {TARGET_RATIOS_PER_N} ratios for N={N}, stopping early.')
                    break
            
            # Trim to exact target if we have more than needed
            if len(agg_results[N]) > TARGET_RATIOS_PER_N:
                agg_results[N] = agg_results[N][:TARGET_RATIOS_PER_N]
                print(f'Trimmed N={N} to exactly {TARGET_RATIOS_PER_N} ratios.')
            
            print(f'Final count for N={N}: {len(agg_results[N])} ratios\n')

        # Save aggregate YAML with metadata
        with open(RESULTS_AGG, 'w') as f:
            metadata = {
                'hidden_size': args.hidden_size,
                'target_ratios_per_n': TARGET_RATIOS_PER_N,
                'epochs': EPOCHS,
                'device': DEVICE,
                'normalization': NORMALIZATION,
                'data_file': DATA_FILE
            }
            results_with_metadata = {
                'metadata': metadata,
                'results': agg_results
            }
            yaml.safe_dump(results_with_metadata, f)

        print(f'Aggregated results saved to {RESULTS_AGG}')

    # Visualization step
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

        # Create simple comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Add overall title with hidden size info
        hidden_size_str = hidden_size_to_string(hidden_size_info)
        fig.suptitle(f'Equal Sample Sizes Demo - Hidden Size: {hidden_size_info}', fontsize=14, fontweight='bold')
        
        # 1. Boxplot
        ax1 = axes[0]
        data = []
        labels = []
        for N in sorted(agg_results.keys(), key=lambda x: int(x)):
            data.append(agg_results[N])
            labels.append(f'N={N}')

        sns.boxplot(data=data, ax=ax1)
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels)
        ax1.set_ylabel('DNN/FS Sum-Rate Ratio')
        ax1.set_xlabel('Number of Memorized Samples (N)')
        ax1.set_title('Equal Sample Sizes: Boxplot Comparison')
        ax1.grid(True, alpha=0.3)

        # 2. Distribution overlays
        ax2 = axes[1]
        colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
        for i, (ratios, label, color) in enumerate(zip(data, labels, colors)):
            ax2.hist(ratios, bins=10, alpha=0.6, label=label, color=color, density=True)
        ax2.set_xlabel('DNN/FS Sum-Rate Ratio')
        ax2.set_ylabel('Density')
        ax2.set_title('Equal Sample Sizes: Distribution Overlays')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_filename = f'n_sample_memorization_equal_demo_{hidden_size_str}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f'Demo visualization saved to {plot_filename}')

        # Print summary
        print("\n" + "="*60)
        print(f"EQUAL SAMPLE SIZES DEMO SUMMARY - Hidden Size: {hidden_size_info}")
        print("="*60)
        
        for N in sorted(agg_results.keys(), key=lambda x: int(x)):
            ratios = np.array(agg_results[N])
            print(f'N={N}: n_samples={len(ratios)}, mean={np.mean(ratios):.3f}, std={np.std(ratios):.3f}')
        
        print(f"\nAll distributions now have equal sample sizes for fair comparison!") 