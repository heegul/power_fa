import numpy as np
import torch
import tempfile
import os
import pytest
from pathlib import Path

# Patch: Force all tests to use CPU to avoid PyTorch MPS device errors on Mac (see https://github.com/pytorch/pytorch/issues/77764)
DEVICE = 'cpu'

from src.config import SimulationConfig
from src.algorithms.ml_dnn import ML_DNN, train_model, dnn_output_to_decision

# Helper: create a random scenario with given config
class DummyScenario:
    def __init__(self, cfg):
        self.cfg = cfg
        self._g = np.random.randn(cfg.n_pairs, cfg.n_pairs)
    def channel_gains_db(self):
        return self._g
    def get_channel_gain_with_fa_penalty(self, fa_indices):
        # Just return the channel matrix for testing
        return 10 ** (self._g / 10)

@pytest.mark.parametrize("n_fa", [1, 2])
def test_single_sample_training_and_eval(n_fa):
    cfg = SimulationConfig(n_pairs=3, n_fa=n_fa, tx_power_min_dbm=0, tx_power_max_dbm=30, seed=42)
    # Single-sample training
    algo, losses, _ = train_model(cfg, epochs=20, verbose=False, device=DEVICE)
    # Loss should decrease at some point
    assert min(losses) < losses[0], "Loss did not decrease at any point during training"
    # Evaluate
    scenario = DummyScenario(cfg)
    x = torch.tensor(scenario.channel_gains_db(), dtype=torch.float32, device=DEVICE).flatten().unsqueeze(0)
    algo.model.eval()  # Ensure BatchNorm is in eval mode
    with torch.no_grad():
        output = algo.model(x)[0]
    tx_power_dbm, fa_indices = dnn_output_to_decision(output, cfg)
    assert tx_power_dbm.shape[0] == cfg.n_pairs
    if n_fa == 1:
        assert np.all(fa_indices == 0)
    else:
        assert fa_indices.shape[0] == cfg.n_pairs

@pytest.mark.parametrize("n_fa", [1, 2])
def test_multi_sample_training_and_eval(n_fa):
    cfg = SimulationConfig(n_pairs=3, n_fa=n_fa, tx_power_min_dbm=0, tx_power_max_dbm=30, seed=123)
    # Generate random training data (10 samples)
    n_samples = 10
    npy_path = None
    try:
        data = np.random.randn(n_samples, cfg.n_pairs, cfg.n_pairs).astype(np.float32)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as f:
            np.save(f, data)
            npy_path = f.name
        # Multi-sample training
        algo, losses, _ = train_model(cfg, epochs=20, train_npy=npy_path, batch_size=2, verbose=False, device=DEVICE)
        assert min(losses) < losses[0], "Loss did not decrease at any point during training"
        # Evaluate on a new random sample
        scenario = DummyScenario(cfg)
        x = torch.tensor(scenario.channel_gains_db(), dtype=torch.float32, device=DEVICE).flatten().unsqueeze(0)
        algo.model.eval()  # Ensure BatchNorm is in eval mode
        with torch.no_grad():
            output = algo.model(x)[0]
        tx_power_dbm, fa_indices = dnn_output_to_decision(output, cfg)
        assert tx_power_dbm.shape[0] == cfg.n_pairs
        if n_fa == 1:
            assert np.all(fa_indices == 0)
        else:
            assert fa_indices.shape[0] == cfg.n_pairs
    finally:
        if npy_path and os.path.exists(npy_path):
            os.remove(npy_path) 