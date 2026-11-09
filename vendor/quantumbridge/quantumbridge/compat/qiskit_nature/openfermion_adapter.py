# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""OpenFermion optional passthrough scaffold for chemistry workflows."""

from quantumbridge.ecosystem.registry import EcosystemAdapter

ADAPTER = EcosystemAdapter(
    "openfermion",
    "openfermion",
    ("openfermion",),
    "qiskit-nature",
    "openfermion",
)
