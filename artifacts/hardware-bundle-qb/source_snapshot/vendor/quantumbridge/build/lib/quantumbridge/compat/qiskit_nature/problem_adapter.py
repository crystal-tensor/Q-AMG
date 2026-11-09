# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Nature problem passthrough scaffold."""

from quantumbridge.ecosystem.registry import EcosystemAdapter

ADAPTER = EcosystemAdapter(
    "qiskit-nature",
    "qiskit-nature",
    ("qiskit_nature.second_q.problems",),
    "qiskit-nature",
    "qiskit_nature_problems",
)
