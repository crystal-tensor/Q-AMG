# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Nature full public API inventory scaffold."""

from quantumbridge.ecosystem.registry import EcosystemAdapter

ADAPTER = EcosystemAdapter(
    "qiskit-nature",
    "qiskit-nature",
    (
        "qiskit_nature",
        "qiskit_nature.second_q.drivers",
        "qiskit_nature.second_q.operators",
        "qiskit_nature.second_q.mappers",
        "qiskit_nature.second_q.hamiltonians",
        "qiskit_nature.second_q.problems",
        "qiskit_nature.second_q.properties",
        "qiskit_nature.second_q.transformers",
        "qiskit_nature.second_q.circuit.library",
        "qiskit_nature.second_q.algorithms",
        "qiskit_nature.second_q.formats",
        "qiskit_nature.second_q.hamiltonians.lattices",
        "qiskit_nature.utils",
    ),
    "qiskit-nature",
    "qiskit_nature",
)
