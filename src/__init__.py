"""D2D Power & Frequency Allocation Simulator."""
from __future__ import annotations

from importlib import metadata as _metadata

try:
    __version__: str = _metadata.version(__name__)
except _metadata.PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = ["config", "simulator", "algorithms"]

# Import to register algorithms
from . import algorithms
from .algorithms import fs_bruteforce  # Explicitly import to register 