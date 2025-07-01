import sys
import numpy as np
from src.config import SimulationConfig
from src.simulator.scenario import Scenario
from src.algorithms.ml_dnn import ML_DNN, dnn_output_to_decision
from src.algorithms.fs_bruteforce import fs_bruteforce
from src.sample_visualization import plot_allocations_side_by_side

# --- User parameters ---
config_path = "cfgs/debug.yaml"  # <-- set your config path
model_weights = "trained_weights.pt"  # <-- set your trained DNN weights path

# 1. Load config and scenario
cfg = SimulationConfig.from_yaml(config_path)
scenario = Scenario.random(cfg)

# 2. DNN allocation
dnn_algo = ML_DNN(cfg)
dnn_algo.load_weights(model_weights)
x = np.array(scenario.channel_gains_db(), dtype=np.float32).flatten()
import torch
x_torch = torch.tensor(x).unsqueeze(0)
with torch.no_grad():
    output = dnn_algo.model(x_torch)[0]
tx_power_dbm_dnn, fa_indices_dnn = dnn_output_to_decision(output, cfg)
dnn_decision = {"tx_power_dbm": tx_power_dbm_dnn, "fa_indices": fa_indices_dnn}

# 3. FS allocation
fs_algo = fs_bruteforce(cfg)
fs_decision = fs_algo.decide(scenario)

# 4. Plot side by side
tx_xy, rx_xy = scenario.as_tuple()
for i in range(len(tx_xy)):
    tx, rx = tx_xy[i], rx_xy[i]
    # ...
plot_allocations_side_by_side(scenario, dnn_decision, fs_decision, cfg)