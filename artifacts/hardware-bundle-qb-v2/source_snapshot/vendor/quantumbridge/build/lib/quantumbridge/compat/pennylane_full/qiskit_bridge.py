# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Minimal PennyLane-to-QuantumBridge-to-Qiskit bridge scaffold."""

from __future__ import annotations

from quantumbridge.compat.contracts import CapabilityLevel

from .operations_adapter import operation_sequence_to_quantumbridge_ir
from .warnings import adapter_metadata, unsupported


def pennylane_operations_to_quantumbridge_ir(operations) -> dict:
    ir = operation_sequence_to_quantumbridge_ir(operations)
    wires = [wire for instruction in ir["instructions"] for wire in instruction.get("wires", [])]
    numeric_wires = [wire for wire in wires if isinstance(wire, int)]
    size = max(numeric_wires) + 1 if numeric_wires else max(len(set(wires)), 1)
    ir["registers"] = {"quantum": {"size": size}, "classical": {"size": 0}}
    ir["metadata"] = {"source": "pennylane", "warnings": ir.get("warnings", [])}
    return ir


def pennylane_tape_to_quantumbridge_ir(tape) -> dict:
    from .tape_adapter import tape_to_quantumbridge_ir

    ir = tape_to_quantumbridge_ir(tape)
    wires = ir.get("wires", [])
    numeric_wires = [wire for wire in wires if isinstance(wire, int)]
    size = max(numeric_wires) + 1 if numeric_wires else max(len(set(wires)), 1)
    ir["registers"] = {"quantum": {"size": size}, "classical": {"size": 0}}
    ir["metadata"] = {"source": "pennylane_tape", "warnings": ir.get("warnings", [])}
    return ir


def quantumbridge_ir_to_qiskit_circuit(ir):
    try:
        from qiskit import QuantumCircuit
    except Exception:
        return unsupported("Qiskit is not installed; cannot build a Qiskit circuit.", CapabilityLevel.INVENTORY)

    payload = ir.to_dict() if hasattr(ir, "to_dict") else ir
    registers = payload.get("registers", {}) if isinstance(payload, dict) else {}
    operations = payload.get("instructions") or payload.get("operations", [])
    if registers:
        num_qubits = registers.get("quantum", {}).get("size", getattr(ir, "num_qubits", 1))
    else:
        wires = [wire for instruction in operations for wire in instruction.get("wires", [])]
        numeric_wires = [wire for wire in wires if isinstance(wire, int)]
        num_qubits = max(numeric_wires) + 1 if numeric_wires else max(len(set(wires)), 1)
    circuit = QuantumCircuit(num_qubits)
    warnings = []
    for instruction in operations:
        op = instruction.get("op")
        targets = instruction.get("targets", [])
        controls = instruction.get("controls", [])
        params = instruction.get("params", [])
        try:
            if op == "h":
                circuit.h(targets[0])
            elif op == "x":
                circuit.x(targets[0])
            elif op == "y":
                circuit.y(targets[0])
            elif op == "z":
                circuit.z(targets[0])
            elif op == "rx":
                circuit.rx(float(params[0]), targets[0])
            elif op == "ry":
                circuit.ry(float(params[0]), targets[0])
            elif op == "rz":
                circuit.rz(float(params[0]), targets[0])
            elif op == "phase":
                circuit.p(float(params[0]), targets[0])
            elif op == "cx":
                circuit.cx(controls[0], targets[0])
            elif op == "cz":
                circuit.cz(controls[0], targets[0])
            elif op == "swap":
                circuit.swap(targets[0], targets[1])
            else:
                warnings.append(f"Unsupported QuantumBridge operation for Qiskit bridge: {op}")
        except Exception as exc:
            warnings.append(f"Could not convert operation {op}: {type(exc).__name__}: {exc}")
    if warnings:
        circuit.metadata = {"quantumbridge_warnings": warnings}
    return circuit


def pennylane_tape_to_qiskit_circuit(tape):
    return quantumbridge_ir_to_qiskit_circuit(pennylane_tape_to_quantumbridge_ir(tape))


def qiskit_circuit_to_basic_pennylane_spec(circuit) -> dict:
    gate_map = {
        "h": "Hadamard",
        "x": "PauliX",
        "y": "PauliY",
        "z": "PauliZ",
        "rx": "RX",
        "ry": "RY",
        "rz": "RZ",
        "p": "PhaseShift",
        "phase": "PhaseShift",
        "cx": "CNOT",
        "cnot": "CNOT",
        "cz": "CZ",
        "swap": "SWAP",
    }
    operations = []
    warnings = []
    for instruction in getattr(circuit, "data", []):
        operation = instruction.operation
        name = operation.name
        pennylane_name = gate_map.get(name)
        wires = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if pennylane_name is None:
            warnings.append(f"Unsupported Qiskit operation for PennyLane spec: {name}")
            continue
        operations.append(
            {
                "operation": pennylane_name,
                "wires": wires,
                "parameters": [_jsonable_parameter(param) for param in getattr(operation, "params", [])],
                "metadata_only": True,
            }
        )
    return {
        "schema_version": "0.1",
        "ecosystem": "pennylane",
        "source": "qiskit",
        "operations": operations,
        "warnings": warnings,
        "provenance": adapter_metadata(CapabilityLevel.SCHEMA_ADAPTER)["provenance"],
    }


def _jsonable_parameter(parameter):
    try:
        return float(parameter)
    except (TypeError, ValueError):
        return repr(parameter)
