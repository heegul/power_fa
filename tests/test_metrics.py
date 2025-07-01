"""Quick sanity tests for metric functions."""
import numpy as np

from src.simulator.metrics import sinr_linear, sum_rate_bps


def test_sum_rate_single_link():
    # One link, no interference, SINR = 10 (linear)
    sinr = np.array([10.0])
    bandwidth = 1e6  # 1 MHz
    rate = sum_rate_bps(sinr, bandwidth)
    expected = bandwidth * np.log2(1 + 10)
    assert np.isclose(rate, expected)


def test_sinr_two_links_no_interference():
    tx_power_lin = np.array([1.0, 1.0])
    fa_indices = np.array([0, 1])  # orthogonal – no interference
    g = np.eye(2)
    noise = 1e-3
    sinr = sinr_linear(tx_power_lin, fa_indices, g, noise)
    # SINR should be power * gain / noise (since no interference)
    expected = tx_power_lin / noise
    assert np.allclose(sinr, expected)


def test_fa_penalty_equivalence():
    """Test that scenario.get_channel_gain_with_fa_penalty and matrix-based penalty logic are equivalent."""
    from src.simulator.scenario import Scenario
    from src.config import SimulationConfig
    cfg = SimulationConfig.from_yaml("cfgs/debug.yaml")
    scenario = Scenario.random(cfg)
    channel_gains_db = scenario.channel_gains_db()
    n_pairs = cfg.n_pairs
    np.random.seed(123)
    fa_indices = np.random.randint(cfg.n_fa, size=n_pairs)
    # Scenario method
    g1 = scenario.get_channel_gain_with_fa_penalty(fa_indices)
    # Matrix patch
    channel_gain_db = channel_gains_db.copy()
    for j in range(n_pairs):
        penalty_db = cfg.fa_penalty_db * fa_indices[j]
        channel_gain_db[:, j] -= penalty_db
    g2 = 10 ** (channel_gain_db / 10)
    assert np.allclose(g1, g2, atol=1e-8), f"Max abs diff: {np.max(np.abs(g1-g2))}" 