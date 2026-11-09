# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Dynamics solver passthrough scaffold."""

from quantumbridge.ecosystem.registry import EcosystemAdapter

ADAPTER = EcosystemAdapter(
    "qiskit-dynamics",
    "qiskit-dynamics",
    ("qiskit_dynamics",),
    "qiskit-dynamics",
    "qiskit_dynamics_solver",
)
