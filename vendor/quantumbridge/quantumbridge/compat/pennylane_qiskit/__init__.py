# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit, PennyLane, or PennyLane-Qiskit was copied.
"""PennyLane-Qiskit clean-room bridge executable slice."""

from .bridge_native import (
    bridge_ir_summary,
    bridge_ir_to_qiskit_circuit,
    pennylane_operations_to_bridge_ir,
    pennylane_operations_to_qiskit_circuit,
    pennylane_qnode_metadata_to_bridge_ir,
    pennylane_tape_to_bridge_ir,
    qiskit_circuit_to_bridge_ir,
    qiskit_circuit_to_pennylane_spec,
    quantumbridge_circuit_to_pennylane_spec,
    run_bidirectional_bridge_equivalence,
    run_pennylane_to_qiskit_bridge,
    run_qiskit_to_pennylane_bridge,
)
from .dependency import (
    dependency_available,
    get_upstream_version,
    validate_pennylane_qiskit_dependencies,
)
from .upstream_adapter import (
    run_with_upstream_pennylane_qiskit_if_available,
    wrap_upstream_pennylane_qiskit_result,
)

__all__ = [
    "bridge_ir_summary",
    "bridge_ir_to_qiskit_circuit",
    "dependency_available",
    "get_upstream_version",
    "pennylane_operations_to_bridge_ir",
    "pennylane_operations_to_qiskit_circuit",
    "pennylane_qnode_metadata_to_bridge_ir",
    "pennylane_tape_to_bridge_ir",
    "qiskit_circuit_to_bridge_ir",
    "qiskit_circuit_to_pennylane_spec",
    "quantumbridge_circuit_to_pennylane_spec",
    "run_bidirectional_bridge_equivalence",
    "run_pennylane_to_qiskit_bridge",
    "run_qiskit_to_pennylane_bridge",
    "run_with_upstream_pennylane_qiskit_if_available",
    "validate_pennylane_qiskit_dependencies",
    "wrap_upstream_pennylane_qiskit_result",
]
