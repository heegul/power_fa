"""
Algorithm module: registry, abstract base class, and public API for D2D allocation algorithms.

This module provides:
- An abstract base class (`AllocationAlgorithm`) for all power/frequency allocation algorithms.
- A registry system for dynamic algorithm discovery and instantiation.
- Public API exposure for all key algorithm modules (DNN, brute-force, loss/validation functions).

Usage Example:
--------------

from src.algorithms import get_algorithm, register_algorithm, AllocationAlgorithm

@register_algorithm
class MyAlgorithm(AllocationAlgorithm):
    def decide(self, scenario):
        # ... implement logic ...
        return {"tx_power_dbm": ..., "fa_indices": ...}

alg_cls = get_algorithm("MyAlgorithm")
alg = alg_cls(cfg)
result = alg.decide(scenario)

# Access DNN and loss/validation utilities
from src.algorithms import ml_dnn, loss_functions, validation_functions

"""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Dict, Type

from ..config import SimulationConfig

# Public API: expose registry, base class, and submodules
__all__ = [
    "AllocationAlgorithm",
    "register_algorithm",
    "get_algorithm",
    "ml_dnn",
    "fs_bruteforce",
    "loss_functions",
    "validation_functions",
]

# Algorithm registry: maps algorithm names to classes
_REGISTRY: Dict[str, Type["AllocationAlgorithm"]] = {}


class AllocationAlgorithm(ABC):
    """
    Abstract interface every allocation algorithm must implement.
    All algorithms should inherit from this class and implement `decide`.
    """
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg

    @abstractmethod
    def decide(self, scenario, /):
        """
        Decide power and frequency allocation for a given scenario.
        Returns a dict with keys `tx_power_dbm` and `fa_indices`.
        """

    @property
    def name(self) -> str:
        """
        Name used in plots/logs (defaults to class name).
        Override if a custom name is desired.
        """
        return self.__class__.__name__


# ------------------------------------------------------------------
# Registry helpers
# ------------------------------------------------------------------

def register_algorithm(cls: Type["AllocationAlgorithm"]) -> Type["AllocationAlgorithm"]:
    """
    Class decorator to auto-register algorithms in the global registry.
    Ensures only subclasses of AllocationAlgorithm are registered.
    Raises if duplicate or invalid registration is attempted.
    """
    if not inspect.isclass(cls):
        raise TypeError("@register_algorithm can only decorate classes")
    if not issubclass(cls, AllocationAlgorithm):
        raise TypeError("Registered class must inherit from AllocationAlgorithm")

    key = cls.__name__
    if key in _REGISTRY:
        raise KeyError(f"Algorithm '{key}' is already registered")
    _REGISTRY[key] = cls
    return cls


def get_algorithm(name: str) -> Type["AllocationAlgorithm"]:
    """
    Retrieve an algorithm class by name from the registry.
    Raises KeyError if not found.
    """
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"Algorithm '{name}' not found in registry. Available: {list(_REGISTRY)}"
        ) from exc


# ------------------------------------------------------------------
# Expose key algorithm modules for convenient import
# ------------------------------------------------------------------

from . import ml_dnn
from . import fs_bruteforce
from . import loss_functions
from . import validation_functions 