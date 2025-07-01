"""Execution engine that evaluates a single algorithm on a single scenario."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..config import SimulationConfig
from ..algorithms import AllocationAlgorithm, get_algorithm
from .metrics import sinr_linear, sum_rate_bps
from .scenario import Scenario
from .environment import db_to_linear

__all__ = ["SimulationResult", "run_once"]


@dataclass
class SimulationResult:
    """Container returned by `run_once`."""

    config: SimulationConfig
    algorithm_name: str
    seed: int
    sum_rate_bps: float

    # For richer post-analysis we also store decisions & scenario (optional)
    tx_power_dbm: np.ndarray
    fa_indices: np.ndarray
    scenario: Scenario


# ------------------------------------------------------------------
# Engine entry-point
# ------------------------------------------------------------------

def run_once(cfg: SimulationConfig, algorithm_name: str) -> SimulationResult:
    """Run one simulation instance with the requested algorithm."""
    cfg.set_global_seeds()

    scenario = Scenario.random(cfg)

    algo_cls = get_algorithm(algorithm_name)
    algo: AllocationAlgorithm = algo_cls(cfg)
    decision = algo.decide(scenario)

    tx_power_dbm = decision["tx_power_dbm"]
    fa_indices = decision["fa_indices"]

    tx_power_lin = db_to_linear(tx_power_dbm) * 1e-3  # dBm → Watt
    noise_power_lin = db_to_linear(cfg.noise_power_dbm) * 1e-3

    sinr = sinr_linear(
        tx_power_lin=tx_power_lin,
        fa_indices=fa_indices,
        channel_gain=scenario.channel_gain_lin,
        noise_power_lin=noise_power_lin,
    )
    sum_rate = sum_rate_bps(sinr, cfg.bandwidth_hz)

    return SimulationResult(
        config=cfg,
        algorithm_name=algorithm_name,
        seed=cfg.seed,
        sum_rate_bps=sum_rate,
        tx_power_dbm=tx_power_dbm,
        fa_indices=fa_indices,
        scenario=scenario,
    ) 