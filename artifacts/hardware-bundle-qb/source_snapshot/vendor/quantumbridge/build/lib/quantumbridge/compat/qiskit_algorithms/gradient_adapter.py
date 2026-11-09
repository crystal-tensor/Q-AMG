# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Algorithms gradient passthrough scaffold."""

from quantumbridge.ecosystem.registry import EcosystemAdapter

ADAPTER = EcosystemAdapter(
    "qiskit-algorithms",
    "qiskit-algorithms",
    ("qiskit_algorithms.gradients",),
    "qiskit-algorithms",
    "qiskit_algorithms_gradients",
)
