"""Metrics for evaluating allocation decisions."""
from __future__ import annotations
import torch
import numpy as np
from .environment import _to_numpy

__all__ = [
    "sinr_linear",
    "sum_rate_bps",
    "sum_rate_dimensionless",
]


def sinr_linear(
    tx_power_lin,
    channel_gain,
    fa_indices,
    noise_power_lin: float,
) -> np.ndarray:
    """Calculate SINR for each D2D pair in a linear scale.

    Robust to tensor inputs from PyTorch.
    """
    if isinstance(tx_power_lin, torch.Tensor):
        tx_power_lin = tx_power_lin.detach().cpu().numpy()
    if isinstance(channel_gain, torch.Tensor):
        channel_gain = channel_gain.detach().cpu().numpy()
    if isinstance(fa_indices, torch.Tensor):
        fa_indices = fa_indices.detach().cpu().numpy()

    n_pairs = tx_power_lin.shape[0]
    sinr = np.zeros_like(tx_power_lin)

    for i in range(n_pairs):
        fa_i = fa_indices[i]
        signal = tx_power_lin[i] * channel_gain[i, i]
        same_fa = np.where(fa_indices == fa_i)[0]
        interference = np.sum(tx_power_lin[same_fa] * channel_gain[same_fa, i]) - signal
        sinr[i] = signal / (interference + noise_power_lin)
    return sinr


def sum_rate_bps(sinr, bandwidth_hz: float):
    """Calculate sum rate in bits per second. Robust to tensor inputs."""
    if isinstance(sinr, torch.Tensor):
        sinr = sinr.detach().cpu().numpy()
    return float(bandwidth_hz * np.sum(np.log2(1.0 + sinr)))


def sum_rate_dimensionless(sinr):
    """Calculate sum rate without bandwidth scaling. Robust to tensor inputs."""
    if isinstance(sinr, torch.Tensor):
        sinr = sinr.detach().cpu().numpy()
    return float(np.sum(np.log2(1.0 + sinr))) 