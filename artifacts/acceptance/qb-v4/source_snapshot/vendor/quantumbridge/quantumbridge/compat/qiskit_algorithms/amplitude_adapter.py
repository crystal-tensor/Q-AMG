# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Algorithms amplitude estimation passthrough scaffold."""

from quantumbridge.ecosystem.registry import EcosystemAdapter

ADAPTER = EcosystemAdapter(
    "qiskit-algorithms",
    "qiskit-algorithms",
    ("qiskit_algorithms.amplitude_estimators",),
    "qiskit-algorithms",
    "qiskit_algorithms_amplitude",
)
