"""Global configuration definitions.

All simulation‐wide tunables and the global random seed live here so that every
component of the framework can access them in a single import.  Config objects
can be created either programmatically or loaded from YAML files to facilitate
batch experiments.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

try:
    import torch
except ImportError:  # pragma: no cover – torch is optional
    torch = None  # type: ignore

__all__ = [
    "SimulationConfig",
]

DEFAULT_YAML_INDENT = 2


@dataclass
class SimulationConfig:
    """Container for all simulation hyperparameters.

    Attributes
    ----------
    area_size_m
        Length of one side of the square deployment area \(metres).
    n_pairs
        Number of transmitter–receiver pairs.
    n_fa
        Number of orthogonal frequency allocations (FAs).
    pathloss_exp
        Path-loss exponent (unit-less). Typical urban micro: 3–4.
    noise_power_dbm
        Receiver noise power in dBm over the bandwidth of a single FA.
    tx_power_min_dbm
        Minimum allowed transmit power in dBm.
    tx_power_max_dbm
        Maximum allowed transmit power in dBm.
    bandwidth_hz
        Bandwidth per FA in Hertz.
    seed
        Global random seed ensuring experiment reproducibility.
    fa_penalty_db
        Additional pathloss penalty per FA index
    """

    area_size_m: float = 100.0
    n_pairs: int = 4
    n_fa: int = 2
    pathloss_exp: float = 3.0
    noise_power_dbm: float = -90.0
    tx_power_min_dbm: float = 0.0
    tx_power_max_dbm: float = 30.0  # 200 mW
    bandwidth_hz: float = 10e6  # 10 MHz per FA
    seed: int = 0
    fa_penalty_db: float = 6.0  # Additional pathloss penalty per FA index

    # Free-form field to store arbitrary user metadata (e.g., experiment name).
    tag: str = field(default="", metadata={"yaml_field": True})

    # Automatically filled, not expected to be loaded from file.
    _yaml_path: Optional[Path] = field(default=None, repr=False, compare=False)

    # ---------------------------------------------------------------------
    # YAML helpers
    # ---------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: os.PathLike | str) -> "SimulationConfig":
        """Load a configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        cfg = cls(**data)
        cfg._yaml_path = Path(path)
        return cfg

    # ------------------------------------------------------------------
    # Random Seed Control
    # ------------------------------------------------------------------
    def set_global_seeds(self) -> None:
        """Seed `random`, `numpy`, and optional frameworks such as PyTorch."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        if torch is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():  # pragma: no cover – CPU default
                torch.cuda.manual_seed_all(self.seed)

    # --------------------------------------------------------------
    # Serialisation utilities
    # --------------------------------------------------------------
    def to_yaml(self, path: os.PathLike | str) -> None:
        """Save the config to YAML."""
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(asdict(self), fh, indent=DEFAULT_YAML_INDENT)

    # Convenience str representation for logging
    def __str__(self) -> str:  # noqa: DunderStr
        return f"SimulationConfig(n_pairs={self.n_pairs}, n_fa={self.n_fa}, seed={self.seed})"

    def __post_init__(self):
        # Ensure bandwidth_hz is always a float (handles YAML string or int)
        if not isinstance(self.bandwidth_hz, float):
            try:
                self.bandwidth_hz = float(self.bandwidth_hz)
            except Exception:
                # Try to evaluate expressions like '10e6'
                self.bandwidth_hz = float(eval(str(self.bandwidth_hz))) 