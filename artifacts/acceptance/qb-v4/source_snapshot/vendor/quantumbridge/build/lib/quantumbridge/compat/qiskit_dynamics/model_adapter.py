# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Dynamics model passthrough scaffold."""

from quantumbridge.ecosystem.registry import EcosystemAdapter

ADAPTER = EcosystemAdapter(
    "qiskit-dynamics",
    "qiskit-dynamics",
    ("qiskit_dynamics.models",),
    "qiskit-dynamics",
    "qiskit_dynamics_models",
)
