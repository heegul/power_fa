import os
import glob
import yaml
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = 'results'
PATTERN = os.path.join(RESULTS_DIR, 'npy_experiment_*.yaml')

# Helper to compute parameter count
def param_count(hidden_size, input_size=12, output_size=6):
    sizes = [input_size] + hidden_size + [output_size]
    total = 0
    for i in range(len(sizes)-1):
        total += sizes[i]*sizes[i+1] + sizes[i+1]
    return total

# Collect all results
data = []
for fname in glob.glob(PATTERN):
    with open(fname, 'r') as f:
        d = yaml.safe_load(f)
    if d is None:
        print(f'Warning: {fname} is empty or invalid, skipping.')
        continue
    if 'metadata' not in d or 'results' not in d:
        print(f'Warning: {fname} missing required keys, skipping.')
        continue
    meta = d['metadata']
    res = d['results']
    hidden_size = meta['hidden_size']
    n_train = meta['n_train_samples_used']
    n_val = meta['n_val_samples_used']
    mean = res['mean_ratio']
    std = res['std_ratio']
    median = res['median_ratio']
    param = param_count(hidden_size)
    data.append({
        'file': os.path.basename(fname),
        'hidden_size': str(hidden_size),
        'n_train': n_train,
        'n_val': n_val,
        'mean_ratio': mean,
        'std_ratio': std,
        'median_ratio': median,
        'param_count': param
    })

df = pd.DataFrame(data)
if df.empty:
    print('No results found.')
    exit(0)

# Print summary table
print(df[['file','hidden_size','n_train','n_val','mean_ratio','std_ratio','median_ratio','param_count']].to_string(index=False))

# Save as CSV
csv_path = os.path.join(RESULTS_DIR, 'npy_experiment_summary.csv')
df.to_csv(csv_path, index=False)
print(f'CSV summary saved to {csv_path}')

# Plot mean±std vs. hidden size (as string)
plt.figure(figsize=(10,6))
plt.errorbar(df['hidden_size'], df['mean_ratio'], yerr=df['std_ratio'], fmt='o', capsize=5)
plt.xticks(rotation=45)
plt.xlabel('Hidden Size')
plt.ylabel('Mean DNN/FS Ratio')
plt.title('Mean±Std DNN/FS Ratio vs. Hidden Size')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'npy_experiment_mean_vs_hidden_size.png'))
print('Plot saved: npy_experiment_mean_vs_hidden_size.png')

# Plot mean±std vs. param count
plt.figure(figsize=(10,6))
plt.errorbar(df['param_count'], df['mean_ratio'], yerr=df['std_ratio'], fmt='o', capsize=5)
plt.xlabel('Parameter Count')
plt.ylabel('Mean DNN/FS Ratio')
plt.title('Mean±Std DNN/FS Ratio vs. Parameter Count')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'npy_experiment_mean_vs_param_count.png'))
print('Plot saved: npy_experiment_mean_vs_param_count.png') 