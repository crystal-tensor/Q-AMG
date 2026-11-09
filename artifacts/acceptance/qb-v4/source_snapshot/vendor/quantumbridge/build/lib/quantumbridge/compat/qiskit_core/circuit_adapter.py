# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit circuit public API inventory and passthrough scaffold.

Design source: docs/compat/strategy/qiskit_ecosystem_coverage_strategy.md.
"""

from quantumbridge.ecosystem.registry import EcosystemAdapter

ADAPTER = EcosystemAdapter("qiskit", "qiskit", ("qiskit.circuit",), "qiskit-core", "qiskit_core_circuit")

dependency_available = ADAPTER.dependency_available
get_upstream_version = ADAPTER.get_upstream_version
list_public_api_inventory = ADAPTER.list_public_api_inventory
passthrough_class = ADAPTER.passthrough_class
passthrough_function = ADAPTER.passthrough_function
wrap_result = ADAPTER.wrap_result
to_quantumbridge_schema = ADAPTER.to_quantumbridge_schema
provenance_metadata = ADAPTER.provenance_metadata
warn_unsupported = ADAPTER.warn_unsupported


def quantum_circuit_metadata(qiskit_circuit) -> dict:
    """Return JSON-safe metadata for a Qiskit QuantumCircuit."""

    return {
        "name": getattr(qiskit_circuit, "name", None),
        "num_qubits": getattr(qiskit_circuit, "num_qubits", None),
        "num_clbits": getattr(qiskit_circuit, "num_clbits", None),
        "operation_count": len(getattr(qiskit_circuit, "data", [])),
        "operations": [item.operation.name for item in getattr(qiskit_circuit, "data", [])],
        "provenance": provenance_metadata(),
    }


def qiskit_circuit_to_quantumbridge_ir(qiskit_circuit):
    from quantumbridge.compat.qiskit_adapter import ir_from_qiskit

    return ir_from_qiskit(qiskit_circuit)


def quantumbridge_ir_to_qiskit_circuit(program):
    from quantumbridge.compat.qiskit_adapter import ir_to_qiskit

    return ir_to_qiskit(program)


def basic_gate_inventory() -> list[dict]:
    gates = ("h", "x", "y", "z", "rx", "ry", "rz", "phase", "cx", "cnot", "cz", "swap", "measure")
    return [
        {
            "ecosystem": "qiskit_core",
            "module": "qiskit.circuit",
            "name": gate,
            "object_type": "gate",
            "quantumbridge_level": 2,
            "supported": True,
            "notes": "Basic QuantumBridge IR <-> Qiskit QuantumCircuit bridge gate.",
        }
        for gate in gates
    ]
