# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Algorithms eigensolver passthrough and native metadata adapter."""

from quantumbridge.ecosystem.registry import EcosystemAdapter
from quantumbridge.compat.qiskit_algorithms.algorithms_native import (
    hamiltonian_to_metadata,
    run_vqe_native,
    run_vqe_upstream,
)

ADAPTER = EcosystemAdapter(
    "qiskit-algorithms",
    "qiskit-algorithms",
    ("qiskit_algorithms.eigensolvers",),
    "qiskit-algorithms",
    "qiskit_algorithms_eigensolvers",
)

__all__ = ["ADAPTER", "hamiltonian_to_metadata", "run_vqe_native", "run_vqe_upstream"]
