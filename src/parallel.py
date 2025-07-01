"""Parallel execution helpers using ProcessPoolExecutor or Ray (optional)."""
from __future__ import annotations

import multiprocessing as mp
from typing import Iterable, List

from .config import SimulationConfig
from .simulator.engine import SimulationResult, run_once

__all__ = ["run_batch"]


def _worker(args):  # type: ignore
    cfg, algorithm_name = args
    return run_once(cfg, algorithm_name)


def run_batch(configs: Iterable[SimulationConfig], algorithm_name: str, processes: int | None = None) -> List[SimulationResult]:
    """Run many simulations in parallel."""

    with mp.Pool(processes=processes) as pool:
        results = pool.map(_worker, [(cfg, algorithm_name) for cfg in configs])
    return results 