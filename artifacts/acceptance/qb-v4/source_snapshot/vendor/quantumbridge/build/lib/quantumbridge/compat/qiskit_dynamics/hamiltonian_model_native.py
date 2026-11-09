# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Qiskit Dynamics was copied.
"""Clean-room single-qubit Hamiltonian helpers."""

from __future__ import annotations

import numpy as np


PAULI = {
    "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def build_single_qubit_hamiltonian(axis: str = "z", frequency: float = 1.0) -> dict[str, object]:
    axis = axis.lower()
    if axis not in PAULI:
        raise ValueError("axis must be x, y, or z")
    matrix = 0.5 * float(frequency) * PAULI[axis]
    return {
        "axis": axis,
        "frequency": float(frequency),
        "matrix": matrix,
        "matrix_serialized": [
            [{"real": float(item.real), "imag": float(item.imag)} for item in row]
            for row in matrix
        ],
        "production_ready": False,
        "hardware_access": False,
    }
