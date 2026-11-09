# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Algorithms optimizer passthrough and native helper adapter."""

from quantumbridge.ecosystem.registry import EcosystemAdapter
from quantumbridge.compat.qiskit_algorithms.algorithms_native import (
    run_qaoa_native_maxcut,
    run_qaoa_upstream,
    solve_maxcut_bruteforce_native,
)

ADAPTER = EcosystemAdapter(
    "qiskit-algorithms",
    "qiskit-algorithms",
    ("qiskit_algorithms.optimizers",),
    "qiskit-algorithms",
    "qiskit_algorithms_optimizers",
)

__all__ = ["ADAPTER", "run_qaoa_native_maxcut", "run_qaoa_upstream", "solve_maxcut_bruteforce_native"]
