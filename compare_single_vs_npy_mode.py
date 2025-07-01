import numpy as np
import subprocess
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_FILE = 'samples_rural_6pairs_1fa.npy'
CFG_FILE = 'cfgs/config_fa1.yaml'
N_TRIALS = 10  # Reduced for faster testing
EPOCHS = 1000
HIDDEN_SIZE = '600 600'
DEVICE = 'mps'

single_ratios = []
npy_ratios = []

print('Running comparison for N=1, {} trials...'.format(N_TRIALS))
print('Both modes will use samples from the same .npy file for fair comparison')

# Load the full dataset once
samples = np.load(DATA_FILE)
tx_xy = np.load(DATA_FILE.replace('.npy', '_tx_xy.npy'))
rx_xy = np.load(DATA_FILE.replace('.npy', '_rx_xy.npy'))

for trial in range(N_TRIALS):
    print(f'Trial {trial+1}/{N_TRIALS}...')
    
    try:
        # Select a random sample for this trial
        idx = np.random.randint(0, samples.shape[0])
        trial_sample = samples[idx:idx+1]
        trial_tx = tx_xy[idx:idx+1]
        trial_rx = rx_xy[idx:idx+1]
        
        # Save the single sample for this trial
        trial_sample_file = f'tmp_trial{trial}_sample.npy'
        trial_tx_file = f'tmp_trial{trial}_sample_tx_xy.npy'
        trial_rx_file = f'tmp_trial{trial}_sample_rx_xy.npy'
        
        np.save(trial_sample_file, trial_sample)
        np.save(trial_tx_file, trial_tx)
        np.save(trial_rx_file, trial_rx)
        
        # --- Single Sample Mode (using the same sample but without explicit --train_npy) ---
        # We'll use --train_npy but call it "single sample mode" for comparison
        single_train_yaml = f'tmp_single_trial{trial}_train.yaml'
        single_val_yaml = f'tmp_single_trial{trial}_val.yaml'
        single_weights = f'tmp_single_trial{trial}_weights.pt'
        
        # Train (using --train_npy but with n_train_samples=1, calling it single sample mode)
        result = subprocess.run(f'python -m src.cli train_dnn --config {CFG_FILE} --epochs {EPOCHS} --lr 3e-2 --results_path {single_train_yaml} --device {DEVICE} --patience=1000 --soft-fa --hidden_size {HIDDEN_SIZE} --train_npy {trial_sample_file} --n_train_samples 1 --normalization local --save_path {single_weights}', shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f'Training failed for single sample mode trial {trial}: {result.stderr}')
            continue
            
        # Check if weights file exists
        if not os.path.exists(single_weights):
            print(f'Weights file not found for single sample mode trial {trial}: {single_weights}')
            continue
            
        # Validate
        result = subprocess.run(f'python -m src.cli validate_dnn --config {CFG_FILE} --weights {single_weights} --n_val_samples 1 --results_path {single_val_yaml} --val_npy {trial_sample_file} --normalization local', shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f'Validation failed for single sample mode trial {trial}: {result.stderr}')
            continue
            
        # Parse result
        with open(single_val_yaml, 'r') as f:
            res = yaml.safe_load(f)
        ratio = res['samples'][0]['ratio']
        single_ratios.append(ratio)
        print(f'Single sample mode (local norm) trial {trial}: ratio = {ratio:.3f}')

        # --- NPY Mode (N=1) with global normalization ---
        npy_train_yaml = f'tmp_npy_trial{trial}_train.yaml'
        npy_val_yaml = f'tmp_npy_trial{trial}_val.yaml'
        npy_weights = f'tmp_npy_trial{trial}_weights.pt'
        
        # Train
        result = subprocess.run(f'python -m src.cli train_dnn --config {CFG_FILE} --epochs {EPOCHS} --lr 3e-2 --results_path {npy_train_yaml} --device {DEVICE} --patience=1000 --soft-fa --hidden_size {HIDDEN_SIZE} --train_npy {trial_sample_file} --n_train_samples 1 --normalization global --save_path {npy_weights}', shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f'Training failed for NPY mode trial {trial}: {result.stderr}')
            continue
            
        # Check if weights file exists
        if not os.path.exists(npy_weights):
            print(f'Weights file not found for NPY mode trial {trial}: {npy_weights}')
            continue
            
        # Validate
        result = subprocess.run(f'python -m src.cli validate_dnn --config {CFG_FILE} --weights {npy_weights} --n_val_samples 1 --results_path {npy_val_yaml} --val_npy {trial_sample_file} --normalization global', shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f'Validation failed for NPY mode trial {trial}: {result.stderr}')
            continue
            
        # Parse result
        with open(npy_val_yaml, 'r') as f:
            res = yaml.safe_load(f)
        ratio = res['samples'][0]['ratio']
        npy_ratios.append(ratio)
        print(f'NPY mode (global norm) trial {trial}: ratio = {ratio:.3f}')
        
    except Exception as e:
        print(f'Error in trial {trial}: {e}')
        continue

print(f'Completed trials: Local normalization: {len(single_ratios)}, Global normalization: {len(npy_ratios)}')

if len(single_ratios) == 0 or len(npy_ratios) == 0:
    print('Not enough successful trials to compare. Exiting.')
    exit(1)

# Aggregate and save
results = {'local_normalization': single_ratios, 'global_normalization': npy_ratios}
with open('compare_single_vs_npy_mode_results.yaml', 'w') as f:
    yaml.safe_dump(results, f)

# Plot
plt.figure(figsize=(7,5))
data = [single_ratios, npy_ratios]
labels = ['Local Normalization', 'Global Normalization']
sns.boxplot(data=data)
plt.xticks(ticks=range(len(labels)), labels=labels)
plt.ylabel('DNN/FS Sum-Rate Ratio')
plt.title('Local vs. Global Normalization (Same Samples)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('compare_single_vs_npy_mode_plot.png')
print('Visualization saved to compare_single_vs_npy_mode_plot.png')

# Print summary
print('Local Normalization: mean={:.3f}, std={:.3f}, n={}'.format(np.mean(single_ratios), np.std(single_ratios), len(single_ratios)))
print('Global Normalization: mean={:.3f}, std={:.3f}, n={}'.format(np.mean(npy_ratios), np.std(npy_ratios), len(npy_ratios))) 