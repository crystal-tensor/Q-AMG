# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Experiments randomized benchmarking inventory scaffold."""

from quantumbridge.ecosystem.registry import EcosystemAdapter

ADAPTER = EcosystemAdapter(
    "qiskit-experiments",
    "qiskit-experiments",
    ("qiskit_experiments.library.randomized_benchmarking",),
    "qiskit-experiments",
    "qiskit_experiments_rb",
)
