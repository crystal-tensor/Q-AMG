# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Optimization application passthrough scaffold."""

from quantumbridge.ecosystem.registry import EcosystemAdapter

ADAPTER = EcosystemAdapter(
    "qiskit-optimization",
    "qiskit-optimization",
    ("qiskit_optimization.applications",),
    "qiskit-optimization",
    "qiskit_optimization_applications",
)
