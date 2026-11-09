# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Metal component inventory scaffold."""

from quantumbridge.ecosystem.registry import EcosystemAdapter

ADAPTER = EcosystemAdapter(
    "qiskit-metal",
    "qiskit-metal",
    ("qiskit_metal.qlibrary",),
    "qiskit-metal",
    "qiskit_metal_components",
)
