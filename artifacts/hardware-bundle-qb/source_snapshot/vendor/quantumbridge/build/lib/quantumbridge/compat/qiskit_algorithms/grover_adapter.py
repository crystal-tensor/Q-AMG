# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Algorithms Grover executable and passthrough adapter."""

from quantumbridge.ecosystem.registry import EcosystemAdapter
from quantumbridge.compat.qiskit_algorithms.algorithms_native import (
    build_grover_oracle_marked_bitstrings,
    grover_result_to_dict,
    run_grover_native,
    run_grover_upstream,
    simulate_grover_statevector_native,
)

ADAPTER = EcosystemAdapter(
    "qiskit-algorithms",
    "qiskit-algorithms",
    ("qiskit_algorithms",),
    "qiskit-algorithms",
    "qiskit_algorithms_grover",
)

__all__ = [
    "ADAPTER",
    "build_grover_oracle_marked_bitstrings",
    "grover_result_to_dict",
    "run_grover_native",
    "run_grover_upstream",
    "simulate_grover_statevector_native",
]
