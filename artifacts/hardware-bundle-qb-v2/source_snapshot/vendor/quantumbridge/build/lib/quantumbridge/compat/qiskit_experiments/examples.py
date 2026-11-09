# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Qiskit Experiments was copied.
"""Convenience runners for Stage 9I Qiskit Experiments examples."""

from __future__ import annotations

from .experiments_upstream_adapter import run_upstream_experiments_if_available
from .rabi_experiment_native import run_rabi_experiment_native
from .ramsey_experiment_native import run_ramsey_experiment_native
from .t1_experiment_native import run_t1_experiment_native


def run_experiments_demo() -> dict[str, object]:
    return {
        "rabi": run_rabi_experiment_native().to_dict(),
        "t1": run_t1_experiment_native().to_dict(),
        "ramsey": run_ramsey_experiment_native().to_dict(),
        "upstream": run_upstream_experiments_if_available().to_dict(),
    }
