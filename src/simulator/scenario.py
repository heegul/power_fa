"""Scenario generation: random topology, channel gains & helpers."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import argparse
import sys
import torch
from torch.utils.data import Dataset

from ..config import SimulationConfig
from .environment import pathloss_db, db_to_linear

__all__ = ["Scenario"]


@dataclass
class Scenario:
    """Concrete random realisation of a D2D network."""

    cfg: SimulationConfig
    tx_xy: np.ndarray  # shape (n_pairs, 2)
    rx_xy: np.ndarray  # shape (n_pairs, 2)

    # Derived fields (filled post-init)
    distance_matrix: np.ndarray = dataclasses.field(init=False, repr=False)
    channel_gain_lin: np.ndarray = dataclasses.field(init=False, repr=False)
    channel_gain_db_base: np.ndarray = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._compute_matrices()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def random(cls, cfg: SimulationConfig, restrict_rx_distance: bool = False) -> "Scenario":
        """Generate a random layout inside a square of side `area_size_m`.
        This implementation matches the topology generation from the reference `dnn_d2d_pytorch.py`.
        """
        rng = np.random.default_rng(cfg.seed)
        tx_xy = rng.uniform(0, cfg.area_size_m, size=(cfg.n_pairs, 2))
        rx_xy = np.zeros_like(tx_xy)

        # Place receivers at a random distance (10m-100m) and angle from their transmitters
        # This matches the logic in `generate_environments` from the reference script
        for i in range(cfg.n_pairs):
            dist = rng.uniform(10, 100)
            ang = rng.uniform(0, 2 * np.pi)
            rx_xy[i, :] = tx_xy[i, :] + dist * np.array([np.cos(ang), np.sin(ang)])
            # Clip coordinates to stay within the area boundaries
            rx_xy[i, :] = np.clip(rx_xy[i, :], 0, cfg.area_size_m)
            
        return cls(cfg=cfg, tx_xy=tx_xy, rx_xy=rx_xy)

    @classmethod
    def from_channel_gains(cls, cfg: SimulationConfig, channel_gains_db: np.ndarray) -> "Scenario":
        n_pairs = cfg.n_pairs
        tx_xy = np.zeros((n_pairs, 2))  # Dummy locations
        rx_xy = np.zeros((n_pairs, 2))
        obj = cls(cfg=cfg, tx_xy=tx_xy, rx_xy=rx_xy)
        obj.channel_gain_db_base = channel_gains_db.copy()
        obj.channel_gain_lin = db_to_linear(channel_gains_db)
        return obj

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _compute_matrices(self) -> None:
        """Compute distance and channel-gain matrices (linear scale).
        This implementation matches the channel gain model from `dnn_d2d_pytorch.py`.
        """
        rng = np.random.default_rng(self.cfg.seed) # Use the same seed for reproducibility
        # Euclidean distances Tx_i → Rx_j for all i, j
        tx = self.tx_xy[:, None, :]  # (n_pairs, 1, 2)
        rx = self.rx_xy[None, :, :]  # (1, n_pairs, 2)
        distance_matrix = np.linalg.norm(tx - rx, axis=-1)  # (n_pairs, n_pairs)
        distance_matrix = np.maximum(distance_matrix, 1.0) # Avoid distance < 1m

        # Path loss model from reference script
        path_loss_db = 10 * self.cfg.pathloss_exp * np.log10(distance_matrix)
        
        # Fading (shadowing) from reference script
        fading_db = rng.normal(0, 8, size=(self.cfg.n_pairs, self.cfg.n_pairs))

        # Final channel gain calculation from reference script
        channel_gain_db = -30.0 - path_loss_db + fading_db
        
        # Store the base channel gain (no FA penalty)
        self.distance_matrix = distance_matrix
        self.channel_gain_db_base = channel_gain_db.copy()
        # The actual channel_gain_lin will be set per FA assignment in the algorithm
        self.channel_gain_lin = db_to_linear(channel_gain_db)  # g_ij (linear power gain)

    def get_channel_gain_with_fa_penalty(self, fa_indices: np.ndarray) -> np.ndarray:
        """Return channel gain matrix (linear) with FA-dependent pathloss penalty.
        For each receiver j, apply fa_penalty_db * fa_indices[j] penalty to all incoming links.
        """
        n_pairs = self.cfg.n_pairs
        channel_gain_db = self.channel_gain_db_base.copy()
        for j in range(n_pairs):
            penalty_db = self.cfg.fa_penalty_db * fa_indices[j]
            channel_gain_db[:, j] -= penalty_db  # Apply penalty to all links to Rx j
        return db_to_linear(channel_gain_db)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def direct_gains(self) -> np.ndarray:
        """Return g_ii – useful for shortcuts."""
        return np.diag(self.channel_gain_lin)

    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (tx_xy, rx_xy) for quick unpacking in visualisations."""
        return self.tx_xy, self.rx_xy

    def channel_gains_db(self) -> np.ndarray:
        """Return the channel gain (pathloss) matrix in dB scale (negative values)."""
        # The base channel gain is now the source of truth
        return self.channel_gain_db_base


CHANNEL_TYPES = {
    "urban": 3.7,
    "suburban": 3.0,
    "rural": 2.7,
}

def generate_samples(
    n_samples: int,
    n_pairs: int,
    n_fa: int,
    area_size_m: float,
    channel_type: str,
    seed: int,
    out_path: str,
    restrict_rx_distance: bool = False,
) -> None:
    """Generate and save input samples (channel gains in dB) for ML models.
    Also saves tx_xy and rx_xy for each sample in parallel .npy files.
    If restrict_rx_distance is True, RX is placed within 1%-10% of area_size_m from its TX.
    """
    pathloss_exp = CHANNEL_TYPES.get(channel_type.lower())
    if pathloss_exp is None:
        raise ValueError(f"Unknown channel type: {channel_type}. Choose from {list(CHANNEL_TYPES)}")

    samples = []
    tx_xy_list = []
    rx_xy_list = []
    for i in range(n_samples):
        cfg = SimulationConfig(
            area_size_m=area_size_m,
            n_pairs=n_pairs,
            n_fa=n_fa,
            pathloss_exp=pathloss_exp,
            seed=seed + i,
        )
        scenario = Scenario.random(cfg, restrict_rx_distance=restrict_rx_distance)
        samples.append(scenario.channel_gains_db())
        tx_xy_list.append(scenario.tx_xy)
        rx_xy_list.append(scenario.rx_xy)
    samples = np.stack(samples)  # shape: (n_samples, n_pairs, n_pairs)
    tx_xy_arr = np.stack(tx_xy_list)  # shape: (n_samples, n_pairs, 2)
    rx_xy_arr = np.stack(rx_xy_list)  # shape: (n_samples, n_pairs, 2)
    np.save(out_path, samples)
    np.save(out_path.replace('.npy', '_tx_xy.npy'), tx_xy_arr)
    np.save(out_path.replace('.npy', '_rx_xy.npy'), rx_xy_arr)
    print(f"Saved {n_samples} samples to {out_path} (shape: {samples.shape})")
    print(f"Saved tx_xy to {out_path.replace('.npy', '_tx_xy.npy')} (shape: {tx_xy_arr.shape})")
    print(f"Saved rx_xy to {out_path.replace('.npy', '_rx_xy.npy')} (shape: {rx_xy_arr.shape})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ML input samples (channel gains in dB) using a YAML config.")
    parser.add_argument("--config", type=str, required=True, help="YAML config file (e.g. cfgs/debug.yaml)")
    parser.add_argument("--n_samples", type=int, required=True, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (base)")
    parser.add_argument("--out_path", type=str, required=True, help="Output .npy file path")
    args = parser.parse_args()
    from ..config import SimulationConfig
    cfg = SimulationConfig.from_yaml(args.config)
    samples = []
    tx_xy_list = []
    rx_xy_list = []
    for i in range(args.n_samples):
        cfg_i = SimulationConfig.from_yaml(args.config)
        cfg_i.seed = args.seed + i
        scenario = Scenario.random(cfg_i)
        samples.append(scenario.channel_gains_db())
        tx_xy_list.append(scenario.tx_xy)
        rx_xy_list.append(scenario.rx_xy)
    samples = np.stack(samples)
    tx_xy_arr = np.stack(tx_xy_list)
    rx_xy_arr = np.stack(rx_xy_list)
    np.save(args.out_path, samples)
    np.save(args.out_path.replace('.npy', '_tx_xy.npy'), tx_xy_arr)
    np.save(args.out_path.replace('.npy', '_rx_xy.npy'), rx_xy_arr)
    print(f"Saved {args.n_samples} samples to {args.out_path} (shape: {samples.shape}) using config {args.config}")
    print(f"Saved tx_xy to {args.out_path.replace('.npy', '_tx_xy.npy')} (shape: {tx_xy_arr.shape})")
    print(f"Saved rx_xy to {args.out_path.replace('.npy', '_rx_xy.npy')} (shape: {rx_xy_arr.shape})")

class ChannelGainDataset(Dataset):
    """PyTorch Dataset for channel gain (pathloss) samples stored in .npy files.

    Args:
        npy_path (str): Path to the .npy file containing samples (n_samples, n_pairs, n_pairs)

    Example:
        >>> dataset = ChannelGainDataset('samples_urban_4pairs_2fa.npy')
        >>> x = dataset[0]  # shape: (n_pairs, n_pairs)
    """
    def __init__(self, npy_path: str):
        self.data = np.load(npy_path)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float() 