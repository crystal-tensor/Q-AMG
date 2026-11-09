# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit, PennyLane, or PennyLane-Qiskit was copied.
"""Stage 9H example helpers."""

from __future__ import annotations

from .bridge_native import (
    run_bidirectional_bridge_equivalence,
    run_pennylane_to_qiskit_bridge,
    run_qiskit_to_pennylane_bridge,
)
from .upstream_adapter import run_with_upstream_pennylane_qiskit_if_available


def run_qiskit_to_pennylane_example() -> dict:
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    return {
        "native": run_qiskit_to_pennylane_bridge(circuit, shots=128, seed=7),
        "upstream": run_with_upstream_pennylane_qiskit_if_available(
            "qiskit_to_pennylane_upstream_passthrough"
        ),
    }


def run_pennylane_to_qiskit_example() -> dict:
    operations = [
        {"operation": "Hadamard", "wires": [0]},
        {"operation": "CNOT", "wires": [0, 1]},
        {"operation": "RY", "wires": [1], "parameters": [0.125]},
    ]
    return {
        "native": run_pennylane_to_qiskit_bridge(operations, shots=128, seed=7),
        "upstream": run_with_upstream_pennylane_qiskit_if_available(
            "pennylane_to_qiskit_upstream_passthrough"
        ),
    }


def run_bridge_equivalence_example() -> dict:
    return {
        "native": run_bidirectional_bridge_equivalence(shots=128, seed=7),
        "upstream": run_with_upstream_pennylane_qiskit_if_available(
            "bridge_equivalence_upstream_passthrough"
        ),
    }
