"""Exhaustive full-search baseline (placeholder).

The search space grows exponentially with the number of pairs.  In the first
iteration we keep it deliberately naive and support only *tiny* scenarios (e.g.,
`n_pairs <= 6`).  Optimisations such as branch-and-bound will be added in later
phases.
"""
from __future__ import annotations

import itertools
from typing import Dict

import numpy as np

from ..config import SimulationConfig
from ..simulator.environment import db_to_linear, linear_to_db
from ..simulator.metrics import sinr_linear, sum_rate_bps
from ..algorithms import AllocationAlgorithm, register_algorithm


@register_algorithm
class fs_bruteforce(AllocationAlgorithm):  # pylint: disable=invalid-name
    """Naive brute-force enumeration of every power & FA combination."""

    def __init__(self, cfg: SimulationConfig):
        super().__init__(cfg)

        # Use power levels: {min, max-6, max-3, max} dBm, always respecting config
        self._power_levels_dbm = np.array([
            cfg.tx_power_min_dbm,
            cfg.tx_power_max_dbm - 6,
            cfg.tx_power_max_dbm - 3,
            cfg.tx_power_max_dbm
        ])
        # This ensures FS always respects the config's min/max tx power and uses a meaningful set.

    # ------------------------------------------------------------------
    # Main decision interface
    # ------------------------------------------------------------------
    def decide(self, scenario, /) -> Dict[str, np.ndarray]:  # noqa: D401
        cfg = self.cfg
        n_pairs = cfg.n_pairs
        n_fa = cfg.n_fa

        best_sum_rate = -np.inf
        best_power = None
        best_fa = None

        noise_lin = db_to_linear(cfg.noise_power_dbm) * 1e-3
        bandwidth = cfg.bandwidth_hz

        # Enumerate FA assignments (n_fa^n_pairs) first.
        for fa_tuple in itertools.product(range(n_fa), repeat=n_pairs):
            fa_indices = np.array(fa_tuple)
            # Use channel gain with FA penalty
            g = scenario.get_channel_gain_with_fa_penalty(fa_indices)
            # Enumerate power levels for each pair.
            for power_tuple in itertools.product(self._power_levels_dbm, repeat=n_pairs):
                tx_power_dbm = np.array(power_tuple)
                tx_power_lin = db_to_linear(tx_power_dbm) * 1e-3  # W

                sinr = sinr_linear(
                    tx_power_lin=tx_power_lin,
                    fa_indices=fa_indices,
                    channel_gain=g,
                    noise_power_lin=noise_lin,
                )
                sum_rate = sum_rate_bps(sinr, bandwidth)

                if sum_rate > best_sum_rate:
                    best_sum_rate = sum_rate
                    best_power = tx_power_dbm
                    best_fa = fa_indices

        assert best_power is not None and best_fa is not None, "Search failed."

        return {
            "tx_power_dbm": best_power,
            "fa_indices": best_fa,
        } 