"""Visualisation helpers (Matplotlib)."""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from ..config import SimulationConfig
from .scenario import Scenario

__all__ = [
    "plot_power_map",
    "plot_sum_rate_cdf",
]

DEFAULT_CMAP = plt.get_cmap("tab10")


def plot_power_map(
    scenario: Scenario,
    decisions: List[dict],
    cfg: SimulationConfig,
    labels: List[str],
    save_path: Path | None = None,
) -> None:
    """Scatter Tx positions sized by Tx power and coloured by FA assignment."""

    tx_xy, _ = scenario.as_tuple()

    fig, ax = plt.subplots(figsize=(6, 6))
    for idx, (decision, label) in enumerate(zip(decisions, labels)):
        power = decision["tx_power_dbm"]
        fa = decision["fa_indices"]
        sizes = (power - cfg.tx_power_min_dbm + 1) * 20  # arbitrary scale
        colours = [DEFAULT_CMAP(ffa) for ffa in fa]
        ax.scatter(tx_xy[:, 0], tx_xy[:, 1], s=sizes, c=colours, alpha=0.5, label=label)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("TX Power Map per Algorithm")
    ax.set_xlim(0, cfg.area_size_m)
    ax.set_ylim(0, cfg.area_size_m)
    ax.legend()
    ax.grid(True, ls=":", lw=0.5)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(save_path.with_suffix(".svg"), bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def plot_sum_rate_cdf(
    sum_rates: List[np.ndarray], labels: List[str], save_path: Path | None = None
) -> None:
    """Plot empirical CDF of sum-rate for each algorithm."""

    fig, ax = plt.subplots(figsize=(6, 4))
    for sr, label in zip(sum_rates, labels):
        sorted_sr = np.sort(sr)
        cdf = np.arange(1, len(sorted_sr) + 1) / len(sorted_sr)
        ax.plot(sorted_sr, cdf, label=label)

    ax.set_xlabel("Sum‐rate [bit/s]")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Sum‐Rate Distribution Across Seeds")
    ax.grid(True, ls=":", lw=0.5)
    ax.legend()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(save_path.with_suffix(".svg"), bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig) 