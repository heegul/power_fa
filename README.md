# Power & Frequency Allocation Simulator (power_fa)

**Goal**: Build a modular, reproducible simulation framework to compare machine-learning / AI algorithms against an exhaustive full-search (FS) baseline for power-level and frequency-assignment decisions in Device-to-Device (D2D) wireless networks.

---

## 1. Objectives

1. Provide a common simulator core (scenario generation, channel modelling, SINR & rate computation).
2. Allow plug-and-play algorithms via a simple API (`AllocationAlgorithm`).
3. Maximise the **system sum-rate** and report the key metric
   \[
     \text{Performance Ratio} = \frac{\text{Sum-Rate}_{\text{Model}}}{\text{Sum-Rate}_{\text{FS}}}
   \]
4. Offer ready-made visualisations: per-device power maps, CDF of sum-rates, and algorithm comparison tables.
5. Exploit all CPU cores on a 64 GB Mac-mini through multi-processing or Ray.
6. Keep experiments **reproducible** with a central `SimulationConfig` (all tunables & random seed).

---

## 2. Project Structure (planned)

```text
power_fa/
├── src/
│   ├── config.py              # Global config dataclass, seed helpers
│   │
│   ├── simulator/
│   │   ├── environment.py     # Path-loss, noise figure, utilities
│   │   ├── scenario.py        # Random topology generator
│   │   ├── engine.py          # Runs one algorithm on one scenario
│   │   ├── metrics.py         # SINR & rate calculations
│   │   └── visualize.py       # Matplotlib/Plotly helpers
│   │
│   ├── algorithms/
│   │   ├── __init__.py        # Registry + abstract base class
│   │   ├── fs_bruteforce.py   # Exhaustive search baseline
│   │   └── ml_deep_rl.py      # Example ML algorithm (placeholder)
│   │
│   ├── parallel.py            # Multiprocessing/Ray utilities
│   └── cli.py                 # Entry-point: python -m power_fa ...
│
├── tests/                     # Pytest unit & integration tests
├── notebooks/                 # Optional research notebooks
├── requirements.txt           # Python dependencies (see below)
├── cursor_rules.md            # Rule-book consumed by Cursor
└── README.md                  # ← **you are here**
```

---

## 3. Activity Roadmap

| Phase | Milestone | Key Tasks |
|-------|-----------|-----------|
| 0     | **Project Bootstrapping** | • Scaffold directory tree<br>• Add `SimulationConfig`, CLI skeleton, algorithm registry<br>• Commit CI & linting config |
| 1     | **Baseline FS & Metrics** | • Implement channel model & SINR<br>• Exhaustive full-search algorithm (small scenarios)<br>• Unit tests for metrics & FS correctness |
| 2     | **Heuristic & ML Hooks** | • Add simple greedy heuristic<br>• Add ML (e.g., Deep-RL) placeholders & training pipeline stubs |
| 3     | **Parallel Engine** | • Wrap simulation loop in `ProcessPoolExecutor` / Ray<br>• Stress-test with 10⁴ seeds |
| 4     | **Visualisation Suite** | • Power-map plots vs. FS<br>• Sum-rate CDF & bar charts |
| 5     | **Experiment Automation** | • Batch-run CLI (`batch` sub-command)<br>• YAML-driven sweeps & result logging |
| 6     | **Optimisation & Docs** | • Numba hot-spots, JIT where needed<br>• Complete README & API docs |

---

## 4. Quick-Start (will evolve)

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. (Optional) Upgrade pip
pip install --upgrade pip

# 3. Install project dependencies
pip install -r requirements.txt

# 4. Run a tiny sanity test (FS on 2 pairs, 1 FA)
python -m src.cli run \
    --config cfgs/debug.yaml \
    --algo fs_bruteforce

# 5. Generate ML input samples (channel gains in dB)
python -m src.cli generate-samples \
    --n_samples 1000 \
    --n_pairs 4 \
    --n_fa 2 \
    --area_size_m 100 \
    --channel_type urban \
    --seed 42 \
    --out_path samples_urban_4pairs_2fa.npy
```

More examples and configuration templates will be added as the implementation progresses.

---

## 5. CLI Workflow: Training, Evaluation, and Visualization

### **A. Training & Evaluation (DNN vs. FS, apples-to-apples)**

Run the following to train a DNN and compare to the full-search (FS) baseline on multiple evaluation samples:

```bash
python -m src.cli train_dnn \
    --config cfgs/debug.yaml \
    --soft-fa \
    --n_eval_samples 5 \
    --results_path results_eval.yaml
```

- For each evaluation sample, this will:
  - Train a DNN on a single scenario (one-sample overfitting)
  - Evaluate both DNN and FS on the same scenario
  - Print and **save** all relevant information (scenario config, locations, channel gains, DNN/FS power, FA, sum-rate, and ratio) to a single YAML file (`results_eval.yaml` by default)
- **FS power levels:** The minimum power is 1e-8 W (no zero power is used in FS search).
+- **Discrete power mode:** Add `--discrete_power` to the CLI command (plus an optional `--power_levels 1e-10 0.25 0.5 1.0` list) to train/evaluate the network on a fixed power grid rather than a continuous range.  The flag works for both `train_dnn` and `validate_dnn` and has been tested in the memorisation experiment.

#### Example YAML structure (results_eval.yaml):
```yaml
samples:
  - index: 0
    config: { ... }
    tx_xy: [[...], ...]
    rx_xy: [[...], ...]
    channel_gains_db: [[...], ...]
    dnn:
      tx_power_dbm: [...]
      fa_indices: [...]
      sum_rate: ...
    fs:
      tx_power_dbm: [...]
      fa_indices: [...]
      sum_rate: ...
    ratio: ...
  - index: 1
    ...
```

### **A2. Mini-batch DNN Training with .npy Datasets (NEW)**

For scalable, reproducible DNN training on large, diverse scenario sets, use the new `.npy`-based mini-batch workflow:

1. **Generate training and validation datasets using a YAML config:**

```bash
python -m src.simulator.scenario --config cfgs/debug.yaml --n_samples 10000 --seed 42 --out_path train_10000.npy
python -m src.simulator.scenario --config cfgs/debug.yaml --n_samples 1000 --seed 123 --out_path val_1000.npy
```
- The `--config` YAML must match your intended training/evaluation setup (n_pairs, n_fa, area_size, etc).
- This ensures apples-to-apples comparison and reproducibility.

2. **Train the DNN using mini-batch learning:**

```bash
python -m src.cli train_dnn \
    --config cfgs/debug.yaml \
    --train_npy train_10000.npy \
    --val_npy val_1000.npy \
    --batch_size 64 \
    --epochs 300 \
    --patience 30 \
    --results_path results_training_validation.yaml
```
- `--train_npy`: Path to training set (.npy, shape [n_samples, n_pairs, n_pairs])
- `--val_npy`: Path to validation set (.npy, same shape)
- `--batch_size`: Mini-batch size for DataLoader
- All other CLI options (learning rate, patience, etc.) are supported
- The loss function is mathematically equivalent to the scenario-based version (see tests for proof)

3. **Results and Reproducibility:**
- All results, configs, and allocations are saved in the YAML as before
- The workflow is fully compatible with previous visualization and evaluation scripts
- You can mix and match one-sample and multi-sample workflows as needed

### **B. Visualization (No Model/FS Rerun Needed)**

To visualize any sample (side-by-side DNN vs. FS allocations, device locations, power, FA, etc.):

```bash
python -m src.sample_visualization \
    --results_yaml results_eval.yaml \
    --sample_idx 0
```

- This will read all information from the YAML file and plot the allocations for the selected sample index.
- **No need to load model weights or rerun FS for visualization.**
- You can quickly browse and analyze any sample by changing `--sample_idx`.

#### **Reproducibility & Fair Comparison**
- All results are saved with full scenario/config info, ensuring apples-to-apples comparison between DNN and FS.
- The workflow is fully reproducible: rerunning with the same config and seeds will yield identical results and visualizations.

---

## 6. Loading Generated Data in PyTorch

You can easily load the generated `.npy` samples for use in PyTorch models:

```python
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# Load the data (shape: [n_samples, n_pairs, n_pairs])
X = np.load('samples_urban_4pairs_2fa.npy')
X_tensor = torch.from_numpy(X).float()

dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Example: iterate over batches
for batch in dataloader:
    x = batch[0]  # shape: [batch_size, n_pairs, n_pairs]
    # ... your model code ...
```

---

## 7. Dependencies (initial draft)

The following libraries are anticipated; see `requirements.txt` for the authoritative list and versions.

* numpy, scipy, numba  – fast numerical kernels
* matplotlib, plotly   – plotting
* pyyaml               – config files
* tqdm                 – progress bars
* ray[default]         – parallel execution (optional)
* torch, stable-baselines3 – for deep-RL algorithms (optional)
* scikit-learn         – for classical ML baselines
* pytest, black, flake8 – testing & linting

---

## 8. Contact & Contributions

This project is currently in active development. Feel free to open issues or submit pull requests once the base framework is stabilised.

## 9. Stand-Alone Utility & Analysis Scripts (NEW)

The repository now contains a collection of single-purpose Python helpers in the *project root* that do **not** live under `src/`.  These scripts make it easy to reproduce the paper figures, sanity-check intermediate results, and run targeted ablation studies without touching the core framework.

| Script | Purpose |
|--------|---------|
| `n_sample_memorization.py` | End-to-end experiment that trains on **N** samples (with `--N_list`) and measures memorisation vs. generalisation. Generates YAML + plots summarising ratio distributions. |
| `n_sample_memorization_equal_demo.py` | Lightweight demo version of the above, fixed `N_list=[1,2,5,10]` for quick runs on a laptop. |
| `analyze_mem_gen.py` / `analyze_mem_gen_results.py` | Post-process the YAML files produced by the memorisation runs and print aggregate statistics (mean/median/std, histogram buckets, etc.). |
| `analyze_npy_results.py` | Similar post-processing helper for the mini-batch (`.npy`) training workflow. |
| `compare_visualization_methods.py` | One-off study comparing `sample_visualization.py` against the MATLAB-equivalent plotting pipeline. |
| `compare_single_vs_npy_mode.py` | Checks that one-sample (scenario-based) and mini-batch (`.npy`) training modes produce identical gradients/losses on overlapping data. |
| `compare_network_capacities.py` | Sweeps different hidden sizes / learning rates to illustrate capacity limits; produces summary bar plots. |
| `compare_implementations.py` | Verifies numerical agreement between the Python and MATLAB baselines after the SINR bug-fix. |
| `sinr_analysis_detailed.py` | Shows a step-by-step SINR matrix build-up, proving no cross-FA interference in the corrected model. |
| `extract_sample_viz_values.py` | Convenience script to dump per-sample TX/RX coordinates and power allocations into CSV for external plotting tools. |
| `validate_averages.py` | Cross-checks that the FS average in YAML files equals the explicit re-compute via `fs_bruteforce`. |
| `final_validation_summary.py` | Collates all validation YAMLs in `results/` and prints a final table ready for publication. |
| `demo_cost_function_separation.py` | Recreates Fig. 3 of the paper, highlighting how the cost function isolates path-loss vs. interference terms. |

All scripts come with `--help` flags—run e.g. `python n_sample_memorization.py --help` to view the available CLI options. 