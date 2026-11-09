# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Machine Learning dataset passthrough scaffold."""

from quantumbridge.ecosystem.registry import EcosystemAdapter

ADAPTER = EcosystemAdapter(
    "qiskit-machine-learning",
    "qiskit-machine-learning",
    ("qiskit_machine_learning.datasets",),
    "qiskit-machine-learning",
    "qiskit_machine_learning_datasets",
)
