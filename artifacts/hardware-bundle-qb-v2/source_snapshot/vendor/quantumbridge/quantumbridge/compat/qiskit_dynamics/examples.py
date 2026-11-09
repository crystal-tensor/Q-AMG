# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Qiskit Dynamics was copied.
"""Convenience runners for Stage 9I Qiskit Dynamics examples."""

from __future__ import annotations

from .dynamics_upstream_adapter import run_upstream_dynamics_if_available
from .single_qubit_dynamics_native import (
    run_dephasing_metadata_simulation_native,
    run_rabi_drive_dynamics_native,
    run_z_precession_native,
)


def run_dynamics_demo() -> dict[str, object]:
    return {
        "z_precession": run_z_precession_native().to_dict(),
        "rabi_drive": run_rabi_drive_dynamics_native().to_dict(),
        "dephasing": run_dephasing_metadata_simulation_native(gamma=0.05).to_dict(),
        "upstream": run_upstream_dynamics_if_available().to_dict(),
    }
