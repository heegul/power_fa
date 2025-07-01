import os
import glob
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def param_count(hidden_size, input_size=36, output_size=12):
    # Adjust input/output size as needed for your problem
    sizes = [input_size] + hidden_size + [output_size]
    total = 0
    for i in range(len(sizes)-1):
        total += sizes[i]*sizes[i+1] + sizes[i+1]
    return total

def parse_hidden_size_from_filename(fname):
    # e.g. results_mem_1024_512_256_512_fa2.yaml
    base = os.path.basename(fname)
    parts = base.split('_')
    hs = []
    for p in parts[2:-1]:
        if p.startswith('fa'):
            break
        try:
            hs.append(int(p))
        except ValueError:
            pass
    return hs

def parse_fa_from_filename(fname):
    base = os.path.basename(fname)
    for part in base.split('_'):
        if part.startswith('fa'):
            return int(part.replace('fa','').replace('.yaml',''))
    return None

def collect_results(pattern):
    data = []
    for fname in glob.glob(pattern):
        with open(fname, 'r') as f:
            d = yaml.safe_load(f)
        if d is None or 'samples' not in d:
            continue
        ratios = [s['ratio'] for s in d['samples']]
        hidden_size = parse_hidden_size_from_filename(fname)
        fa = parse_fa_from_filename(fname)
        param = param_count(hidden_size)
        data.append({
            'file': os.path.basename(fname),
            'hidden_size': str(hidden_size),
            'fa': fa,
            'param_count': param,
            'mean_ratio': float(pd.Series(ratios).mean()),
            'std_ratio': float(pd.Series(ratios).std()),
            'median_ratio': float(pd.Series(ratios).median()),
            'ratios': ratios,
            'type': 'mem' if 'mem' in fname else 'gen'
        })
    return pd.DataFrame(data)

# Collect and merge
df_mem = collect_results('results_mem_*.yaml')
df_gen = collect_results('results_gen_*.yaml')
df = pd.concat([df_mem, df_gen], ignore_index=True)

# Print summary
print(df[['file','hidden_size','fa','param_count','type','mean_ratio','std_ratio','median_ratio']])

# Plot: Mean DNN/FS ratio vs. param count for memorization and generalization
plt.figure(figsize=(10,6))
for fa in sorted(df['fa'].unique()):
    for t, marker, label in zip(['mem','gen'], ['o','s'], ['Memorization','Generalization']):
        sub = df[(df['fa']==fa) & (df['type']==t)]
        plt.errorbar(sub['param_count'], sub['mean_ratio'], yerr=sub['std_ratio'], fmt=marker+'-', capsize=5, label=f'FA={fa} {label}')
plt.xlabel('Parameter Count')
plt.ylabel('Mean DNN/FS Ratio')
plt.title('DNN/FS Ratio vs. Network Size (Memorization vs Generalization)')
plt.legend()
plt.tight_layout()
plt.savefig('dnn_mem_gen_vs_param_count.png')
plt.show()

# Optionally: plot vs. hidden size string
plt.figure(figsize=(12,6))
for fa in sorted(df['fa'].unique()):
    for t, marker, label in zip(['mem','gen'], ['o','s'], ['Memorization','Generalization']):
        sub = df[(df['fa']==fa) & (df['type']==t)]
        plt.errorbar(sub['hidden_size'], sub['mean_ratio'], yerr=sub['std_ratio'], fmt=marker, capsize=5, label=f'FA={fa} {label}')
plt.xlabel('Hidden Size')
plt.ylabel('Mean DNN/FS Ratio')
plt.title('DNN/FS Ratio vs. Hidden Size (Memorization vs Generalization)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('dnn_mem_gen_vs_hidden_size.png')
plt.show()

# --- Side-by-side boxplot for memorization and generalization ---
for fa in sorted(df['fa'].unique()):
    # Prepare data for each network size
    network_labels = []
    mem_data = []
    gen_data = []
    for hs in sorted(df[df['fa']==fa]['hidden_size'].unique(), key=lambda x: eval(x)):
        mem_ratios = []
        gen_ratios = []
        for _, row in df[(df['fa']==fa) & (df['hidden_size']==hs)].iterrows():
            if row['type'] == 'mem':
                mem_ratios.extend(row['ratios'])
            elif row['type'] == 'gen':
                gen_ratios.extend(row['ratios'])
        if mem_ratios and gen_ratios:
            network_labels.append(hs)
            mem_data.append(mem_ratios)
            gen_data.append(gen_ratios)
    if not network_labels:
        continue
    # Plot
    plt.figure(figsize=(max(8, len(network_labels)*1.5), 6))
    positions_mem = np.arange(len(network_labels))*2 - 0.3
    positions_gen = np.arange(len(network_labels))*2 + 0.3
    box1 = plt.boxplot(mem_data, positions=positions_mem, widths=0.5, patch_artist=True, boxprops=dict(facecolor='#4F81BD'), medianprops=dict(color='black'))
    box2 = plt.boxplot(gen_data, positions=positions_gen, widths=0.5, patch_artist=True, boxprops=dict(facecolor='#C0504D'), medianprops=dict(color='black'))
    plt.xticks(np.arange(len(network_labels))*2, network_labels, rotation=45)
    plt.xlabel('Network Hidden Size')
    plt.ylabel('DNN/FS Sum-Rate Ratio')
    plt.title(f'Memorization vs Generalization: DNN/FS Ratio Distribution by Network Size (FA={fa})')
    plt.legend([box1["boxes"][0], box2["boxes"][0]], ['Memorization', 'Generalization'], loc='lower right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'dnn_mem_gen_boxplot_vs_network_fa{fa}.png', dpi=300)
    plt.show()