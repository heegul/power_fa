"""Physical-layer utilities such as path‐loss and noise conversions.

Note: The functions below are deliberately simple placeholders so that the rest
of the system can compile. Optimised or more accurate models (e.g., shadowing,
LOS/NLOS models) can be dropped in later without touching callers.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "db_to_linear",
    "linear_to_db",
    "pathloss_db",
]


def _to_numpy(data):
    """Ensure data is a CPU-bound NumPy array."""
    import torch
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def db_to_linear(db):
    """Convert dB to a linear scale (power ratio). Robust to tensor inputs."""
    db = _to_numpy(db)
    return 10 ** (db / 10.0)


def linear_to_db(lin):
    """Convert linear power ratio to dB. Robust to tensor inputs."""
    lin = _to_numpy(lin)
    # Use a small epsilon to avoid log10 of zero
    return 10.0 * np.log10(lin + 1e-12)


def pathloss_db(distance_m: np.ndarray, pathloss_exp: float) -> np.ndarray:
    """Free-space like path-loss in dB (no antenna gains).

    Parameters
    ----------
    distance_m
        Array of Tx–Rx distances in metres. Should be non-zero.
    pathloss_exp
        Path-loss exponent *n* such that PL ∝ d^{n}.
    """
    # Reference pathloss at 1 metre assumed to be 0 dB for simplicity.
    distance_m = np.maximum(distance_m, 1.0)  # avoid log(0)
    pl_db = 10 * pathloss_exp * np.log10(distance_m)
    return pl_db 