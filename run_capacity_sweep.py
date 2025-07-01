#!/usr/bin/env python3
"""
run_capacity_sweep.py
A convenience wrapper that:
  • loops over candidate network sizes
  • launches n_sample_memorization.py to train/validate on N = 1,2,5,10,100
  • regenerates the built-in plots (plot_only)
  • aggregates every results_mem_* / results_gen_* YAML into one DataFrame
  • produces summary plots: mean±std vs parameter-count  +  Mem/G
"""

import subprocess, sys, os, yaml, glob, ast, itertools
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# ───────────────────────── USER-EDITABLE SECTION ──────────────────────────
DEVICE          = "mps"          # cuda / cpu / mps
EPOCHS          = 1000
TARGET_RATIOS   = 100
N_LIST = [1, 2, 5, 10, 100]
#N_LIST = [1, 2]
HIDDEN_SIZES = [
    [10],
    [20],
    [100],
    [200, 200],
    [512, 512], 
    [1024, 1024],
    [2048, 2048]
]

NPY_F1   = "1000samples_rural_6pair_1fa_rrx.npy"
NPY_F2   = "1000samples_rural_6pair_2fa_rrx.npy"
VAL_NPY_F1   = "300samples_rural_6pair_1fa_rrx.npy"
VAL_NPY_F2   = "300samples_rural_6pair_2fa_rrx.npy"
CFG_F1   = "cfgs/config_fa1.yaml"
CFG_F2   = "cfgs/debug.yaml"      # your FA-2 config
FA_LIST = [1, 2]  # or dynamically infer from your config/experiment
# ───────────────────────────────────────────────────────────────────────────

def hidden_to_str(hid):      # e.g. [128,128] → "128_128"
    return "_".join(map(str, hid))

def run_mem_script_for_N(hidden, N, cfg_file, val_npy_path):
    base_cmd = (
        f"python n_sample_memorization.py "
        f"--hidden_size {' '.join(map(str, hidden))} "
        f"--epochs {EPOCHS} "
        f"--target_ratios {TARGET_RATIOS} "
        f"--device {DEVICE} "
        f"--cfg {cfg_file} "
        f"--val_npy {val_npy_path} "
    )
    if N is not None:
        base_cmd += f"--N {N} "
    else:
        base_cmd += "--N_list " + " ".join(map(str, N_LIST)) + " "
    print("▶", base_cmd)
    try:
        subprocess.run(base_cmd, shell=True, check=True)
        subprocess.run(base_cmd + " --plot_only", shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Subprocess failed for hidden={hidden}, N={N}: {e}")
        return False
    return True

def collect_results():
    rows = []
    for fname in itertools.chain(glob.glob("results_mem_*.yaml"), glob.glob("results_gen_*.yaml")):
        fname = Path(fname)
        with open(fname, "r") as f:
            d = yaml.safe_load(f) or {}

        # Determine hidden size and FA from filename parts
        parts  = fname.name.split("_")
        hidden = []
        for p in parts[2:-1]:
            if p.startswith("fa"): break
            try:
                hidden.append(int(p))
            except ValueError:
                pass
        fa  = int([p for p in parts if p.startswith("fa")][0][2:].split(".")[0])

        inp, out = 36, 12
        sizes = [inp] + hidden + [out]
        params = sum(sizes[i]*sizes[i+1] + sizes[i+1] for i in range(len(sizes)-1))

        if 'results' in d:  # New aggregated format {results: {N: [ratios]}}
            for N_key, ratios in d['results'].items():
                N_int = int(N_key)
                rows.append(dict(file=str(fname),
                                 hidden=str(hidden),
                                 fa=fa,
                                 param_count=params,
                                 type="mem" if "mem" in str(fname) else "gen",
                                 mean=float(np.mean(ratios)),
                                 std=float(np.std(ratios)),
                                 ratios=ratios,
                                 N=N_int))
        elif 'samples' in d:  # Old flat-list format
            ratios = [s['ratio'] for s in d['samples']]
            # Try to get N from metadata if available
            N_int = d.get('metadata', {}).get('N')
            rows.append(dict(file=str(fname),
                             hidden=str(hidden),
                             fa=fa,
                             param_count=params,
                             type="mem" if "mem" in str(fname) else "gen",
                             mean=float(np.mean(ratios)),
                             std=float(np.std(ratios)),
                             ratios=ratios,
                             N=N_int))
    return pd.DataFrame(rows)

def plot_summary(df):
    # Mean±std vs parameters
    plt.figure(figsize=(10,6))
    for t, m, lbl in zip(["mem","gen"], ["o-","s-"], ["Memorization","Generalization"]):
        grp = df[df.type==t].groupby("param_count").agg(
            mean_val=("mean","mean"),
            err_val=("mean","std")
        ).reset_index()
        plt.errorbar(grp.param_count, grp.mean_val, yerr=grp.err_val,
                     fmt=m, label=lbl, capsize=4)
    plt.xlabel("Parameter count")
    plt.ylabel("Mean DNN/FS ratio")
    plt.title("Capacity sweep")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig("capacity_mean_std.png", dpi=300)

    # Side-by-side boxplots per FA
    import numpy as np
    for fa in sorted(df.fa.unique()):
        nets = sorted(df[df.fa==fa].hidden.unique(),
                      key=lambda x: ast.literal_eval(x))
        mem_data, gen_data = [], []
        for h in nets:
            mem_data.append(list(itertools.chain(*df[(df.hidden==h)&(df.type=="mem")].ratios)))
            gen_data.append(list(itertools.chain(*df[(df.hidden==h)&(df.type=="gen")].ratios)))
        if not mem_data: continue
        positions_mem = np.arange(len(nets))*2 - .3
        positions_gen = np.arange(len(nets))*2 + .3
        plt.figure(figsize=(max(8,len(nets)*1.7),6))
        b1=plt.boxplot(mem_data, positions=positions_mem, widths=.5,
                       patch_artist=True, boxprops=dict(facecolor="#4F81BD"), medianprops=dict(color='k'))
        b2=plt.boxplot(gen_data, positions=positions_gen, widths=.5,
                       patch_artist=True, boxprops=dict(facecolor="#C0504D"), medianprops=dict(color='k'))
        plt.xticks(np.arange(len(nets))*2, nets, rotation=45)
        plt.ylabel("DNN/FS ratio"); plt.xlabel("Hidden size"); 
        plt.title(f"Mem vs Gen distribution (FA={fa})")
        plt.legend([b1["boxes"][0], b2["boxes"][0]], ["Mem", "Gen"])
        plt.grid(axis='y', alpha=.3); plt.tight_layout()
        plt.savefig(f"box_mem_gen_fa{fa}.png", dpi=300)

def plot_box_vs_N(df, hidden_size, fa):
    df_sub = df[(df['hidden'] == str(hidden_size)) & (df['fa'] == fa)]
    N_list = sorted([n for n in df_sub['N'].unique() if n is not None])
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
    if not N_list:
        return
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
    plt.savefig(f"box_vs_N_hidden_{'_'.join(map(str,eval(str(hidden_size))))}_fa{fa}.png")
    plt.close()

def aggregate_and_plot_allN(hidden_size, fa):
    # Find all YAMLs for this hidden size and FA
    files = glob.glob(f"results_mem_*_{hidden_size}_fa{fa}.yaml")
    all_samples = []
    for f in files:
        with open(f) as fp:
            d = yaml.safe_load(fp) or {}
        if 'results' in d:
            for N_key, ratios in d['results'].items():
                for r in ratios:
                    all_samples.append({'N': int(N_key), 'ratio': r})
        elif 'samples' in d:
            N_val = d.get('metadata', {}).get('N')
            for s in d['samples']:
                s['N'] = N_val
                all_samples.append(s)
    if not all_samples:
        print(f"[WARNING] No samples found for hidden size {hidden_size}, fa={fa}.")
        return
    df = pd.DataFrame(all_samples)
    plt.figure(figsize=(8,6))
    df.boxplot(column='ratio', by='N')
    plt.title(f'DNN/FS Ratio vs N for hidden size {hidden_size}, FA={fa}')
    plt.ylabel('DNN/FS Ratio')
    plt.xlabel('Number of Training Samples (N)')
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(f'n_sample_memorization_{hidden_size}_fa{fa}_distribution_analysis_allN.png')
    plt.close()

def aggregate_and_plot_mem_gen(hidden_size, fa):
    import pandas as pd
    import matplotlib.pyplot as plt
    import glob
    import yaml
    # Find all mem and gen YAMLs for this hidden size and FA
    mem_files = glob.glob(f"results_mem_*_{hidden_size}_fa{fa}.yaml")
    gen_files = glob.glob(f"results_gen_*_{hidden_size}_fa{fa}.yaml")
    all_rows = []
    for files, typ in [(mem_files, 'mem'), (gen_files, 'gen')]:
        for file in files:
            with open(file) as fp:
                d = yaml.safe_load(fp) or {}
            if 'results' in d:
                for N_key, ratios in d['results'].items():
                    for r in ratios:
                        all_rows.append({'N': int(N_key), 'ratio': r, 'type': typ})
            elif 'samples' in d:
                N_val = d.get('metadata', {}).get('N')
                for s in d['samples']:
                    all_rows.append({'N': N_val, 'ratio': s['ratio'], 'type': typ})
    if not all_rows:
        print(f"[WARNING] No samples found for hidden size {hidden_size}, fa={fa}.")
        return
    df = pd.DataFrame(all_rows)
    plt.figure(figsize=(10,6))
    # Prepare data for grouped boxplot
    data = []
    labels = []
    N_list = sorted(df['N'].unique())
    for N in N_list:
        data.append(df[(df['N']==N) & (df['type']=='mem')]['ratio'])
        data.append(df[(df['N']==N) & (df['type']=='gen')]['ratio'])
        labels.extend([f'N={N}\nMem', f'N={N}\nGen'])
    plt.boxplot(data, labels=labels, patch_artist=True)
    plt.title(f'DNN/FS Ratio vs N for hidden size {hidden_size}, FA={fa} (Mem & Gen)')
    plt.ylabel('DNN/FS Ratio')
    plt.xlabel('Number of Training Samples (N)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'n_sample_memorization_{hidden_size}_fa{fa}_mem_gen_boxplot.png')
    plt.close()

if __name__ == "__main__":
    for hidden in HIDDEN_SIZES:
        for fa in FA_LIST:
            # Select config & val dataset based on FA value
            if fa == 1:
                cfg_file = CFG_F1
                val_npy  = VAL_NPY_F1
            else:
                cfg_file = CFG_F2
                val_npy  = VAL_NPY_F2

            # Run one job per hidden size+FA; n_sample_memorization will iterate over N_LIST internally
            with ProcessPoolExecutor(max_workers=6) as executor:
                future = executor.submit(run_mem_script_for_N, hidden, None, cfg_file, val_npy)
                try:
                    future.result()
                except Exception as e:
                    print(f"Job failed: {e}")

            # --- Incremental aggregation and plotting after each (hidden, fa) combo ---
            df_partial = collect_results()
            df_partial.to_csv("capacity_sweep_summary_partial.csv", index=False)
            plot_summary(df_partial)
            print(f"[INFO] Completed hidden size {hidden}, FA={fa}. Summary updated and plots refreshed.")
            aggregate_and_plot_allN(hidden, fa)
            aggregate_and_plot_mem_gen(hidden, fa)

    # Final aggregation and plotting
    df = collect_results()
    df.to_csv("capacity_sweep_summary.csv", index=False)
    plot_summary(df)
    print("\n✓ Sweep finished. Figures saved: capacity_mean_std.png + box_mem_gen_fa*.png")