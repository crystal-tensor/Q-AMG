# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Metal simulation interface inventory scaffold."""

from quantumbridge.ecosystem.registry import EcosystemAdapter

ADAPTER = EcosystemAdapter(
    "qiskit-metal",
    "qiskit-metal",
    ("qiskit_metal.analyses",),
    "qiskit-metal",
    "qiskit_metal_simulation",
)
