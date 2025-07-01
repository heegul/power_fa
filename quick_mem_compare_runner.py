# quick_mem_compare_runner.py  (paste-and-run)

import os, subprocess, json, pickle, tempfile, shutil, sys, textwrap, math
from pathlib import Path
import numpy as np
import yaml  # NEW
import re, ast

# ---------------------------------------------------------
# User-editable params
# ---------------------------------------------------------
N_LIST          = [1, 2, 5, 10]
HIDDEN_SIZES    = [200, 200]
EPOCHS          = 300
DEVICE          = "mps"      # "cuda" / "cpu" / "mps"
PAIRS           = 6
FREQS           = 1
SEED_BASE       = 12345
FS_LEVELS       = [1e-10, 0.25, 0.5, 1.0]
TARGET_RATIOS   = 100   # how many DNN/FS ratios we want per N
# ---------------------------------------------------------

root = Path(".").resolve()
tmp_dir = Path(tempfile.mkdtemp(prefix="mem_compare_", dir=root))

def run(cmd, cwd=root):
    print("★", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True).stdout

results = []

# ------------------------------------------------------------------
# Helper to extract list of D2D ratios from stdout
# ------------------------------------------------------------------
def extract_d2d_ratios(text: str):
    """Return a list of floats found in the dnn_d2d_pytorch single-sample summary."""
    ratios = []
    # Pattern 1: explicit numpy array
    for line in text.splitlines():
        if "Normal DNN/FS ratio" in line:
            m = re.search(r"array\(\[(.*)\]\)", line)
            if m:
                nums = [float(x) for x in m.group(1).split(',') if x.strip()]
                ratios.extend(nums)
    # Pattern 2: csv already written – skip here (caller will load if empty)
    return ratios

for N in N_LIST:
    print("\n====================  N =", N, "====================")
    ref_ratios = []
    d2d_ratios = []
    trial = 0
    max_trials_needed = math.ceil(TARGET_RATIOS / N)
    while len(ref_ratios) < TARGET_RATIOS:
        trial += 1
        print(f"  → Trial {trial} (collected {len(ref_ratios)}/{TARGET_RATIOS})")

        # -------------------- Generate new environment --------------------
        gen_cmd = [
            sys.executable, "dnn_d2d_pytorch.py",
            "--num_samples", str(2 * N),
            "--val_ratio", "0.5",
            "--use_train_for_val",
            "--num_pairs",   str(PAIRS),
            "--num_frequencies", str(FREQS),
            "--epochs", "0",
            "--figs_dir", tmp_dir.as_posix(),
            "--device", DEVICE
        ]
        run(gen_cmd)
        # Find the specific environment file for this N (not the most recent one)
        expected_pattern = f"environment_samples_db_pairs{PAIRS}_samples{2*N}_rxFalse.pkl"
        env_pickle = tmp_dir / expected_pattern
        if not env_pickle.exists():
            # Fallback to the old logic if the expected file doesn't exist
            env_pickle = sorted(tmp_dir.glob("environment_samples_db_pairs*.pkl"))[-1]

        # -----------------------------------------------------------------
        # Convert pickle to .npy for src.cli train/validate
        # -----------------------------------------------------------------
        with open(env_pickle, "rb") as f:
            gains = pickle.load(f)["channel_gains"]
        npy_path = tmp_dir / f"gains_{N}.npy"
        np.save(npy_path, gains)

        print(f"Generated environments saved to {env_pickle}")

        # -------------------------------------------------
        # Reference pipeline: one model on ALL N samples
        # -------------------------------------------------
        ref_figs = tmp_dir / f"ref_N{N}_t{trial}"
        weights_path = ref_figs / "weights.pt"
        results_yaml = ref_figs / "val.yaml"

        train_cmd = [
            sys.executable, '-m', 'src.cli', 'train_dnn',
            '--config', 'cfgs/config_fa1.yaml',
            '--hidden_size', '100', '100',
            '--epochs', str(EPOCHS),
            '--device', DEVICE,
            '--no_shuffle',
            '--train_npy', npy_path.as_posix(),
            '--n_train_samples', str(N),
            '--save_path', weights_path.as_posix(),
            '--figs_dir', ref_figs.as_posix(),
        ]
        run(train_cmd)

        val_cmd = [
            sys.executable, '-m', 'src.cli', 'validate_dnn',
            '--config', 'cfgs/config_fa1.yaml',
            '--weights', weights_path.as_posix(),
            '--val_npy', npy_path.as_posix(),
            '--n_val_samples', str(N),
            '--results_path', results_yaml.as_posix(),
        ]
        run(val_cmd)

        with open(results_yaml, 'r') as f:
            val_data = yaml.safe_load(f)
        ratios_run = [s['ratio'] for s in val_data['samples']]
        ref_ratios.extend(ratios_run)

        # -------------------------------------------------
        # 3) dnn_d2d_pytorch path (single-sample mode)
        # -------------------------------------------------
        d2d_figs = tmp_dir / f"d2d_N{N}_t{trial}"
        d2d_cmd = [
            sys.executable, "dnn_d2d_pytorch.py",
            "--input_file", env_pickle.as_posix(),
            "--num_samples", str(2 * N),
            "--val_ratio", "0.5",
            "--use_train_for_val",
            "--hidden_sizes", *[str(h) for h in HIDDEN_SIZES],
            "--num_frequencies", "1",
            "--enable_fs",
            "--fs_power_levels", *[str(p) for p in FS_LEVELS],
            "--epochs", str(EPOCHS),
            "--figs_dir", d2d_figs.as_posix(),
            "--device", DEVICE
        ]
        d2d_out = run(d2d_cmd)
        # Extract list of ratios
        d2d_trial_ratios = []
        # Preferred: memorisation CSV (batch mode)
        csv_path = d2d_figs / 'dnn_vs_hard_vs_fs_memorization.csv'
        if csv_path.exists():
            import csv
            with open(csv_path) as cf:
                reader = csv.DictReader(cf)
                for row in reader:
                    if row.get('Normal_FS_Ratio'):
                        d2d_trial_ratios.append(float(row['Normal_FS_Ratio']))
        # Fallback: average ratio txt
        if not d2d_trial_ratios:
            ratio_file = d2d_figs / 'dnn_fs_ratio.txt'
            if ratio_file.exists():
                with open(ratio_file) as rf:
                    try:
                        d2d_trial_ratios.append(float(rf.read().strip()))
                    except ValueError:
                        pass
        d2d_ratios.extend(d2d_trial_ratios)

    # Trim to target
    ref_ratios = ref_ratios[:TARGET_RATIOS]
    d2d_ratios = d2d_ratios[:TARGET_RATIOS]

    ref_mean = sum(ref_ratios)/len(ref_ratios)
    d2d_mean = sum(d2d_ratios)/len(d2d_ratios)

    results.append((N, ref_mean, d2d_mean))

# ---------------------------------------------------------
# 4) Pretty print
# ---------------------------------------------------------
print("\n=========== SUMMARY ===========")
print("N   src.ml_dnn   dnn_d2d")
for N, r1, r2 in results:
    print(f"{N:<3} {r1:10.3f}   {r2:8.3f}")

# Cleanup tmp_dir if you wish:
# shutil.rmtree(tmp_dir)