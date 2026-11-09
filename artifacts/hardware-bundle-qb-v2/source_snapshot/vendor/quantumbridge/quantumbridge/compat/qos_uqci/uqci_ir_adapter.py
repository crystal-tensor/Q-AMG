# This file is independently implemented for QuantumBridge SDK.
# No source code from QOS-UQCI was copied.
"""Clean-room QOS-UQCI IR conversion helpers."""

from __future__ import annotations

from typing import Any

from quantumbridge.compat.qiskit_aer.simulator_native import normalize_circuit_to_quantumbridge_ir

from .warnings import qos_uqci_warnings, native_provenance

SUPPORTED_UQCI_GATES = {"x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "phase", "cx", "cz", "swap"}


def quantumbridge_ir_to_uqci_ir(circuit_or_ir: Any) -> dict[str, Any]:
    circuit = normalize_circuit_to_quantumbridge_ir(circuit_or_ir)
    operations = []
    for index, op in enumerate(circuit.operations):
        if op.name not in SUPPORTED_UQCI_GATES:
            raise ValueError(f"Stage 10A QOS-UQCI adapter does not support operation {op.name!r}")
        operations.append(
            {
                "id": f"op_{index}",
                "gate": op.name,
                "targets": list(op.targets),
                "controls": list(op.controls),
                "parameters": [float(param) for param in op.params],
            }
        )
    measurements = [
        {"kind": measurement.kind, "wire": measurement.wire, "bit": measurement.bit}
        for measurement in circuit.measurements
    ]
    return {
        "schema_version": "0.1",
        "ir_type": "qos_uqci_clean_room_ir",
        "source": "quantumbridge_ir",
        "name": circuit.name,
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_bits,
        "operations": operations,
        "measurements": measurements,
        "metadata": {
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "production_runtime_claim": False,
        },
        "warnings": qos_uqci_warnings(),
        "provenance": native_provenance("quantumbridge_ir_to_uqci_ir"),
    }


def uqci_ir_to_quantumbridge_ir(payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    instructions = []
    for item in data.get("operations", []):
        instructions.append(
            {
                "op": item.get("gate"),
                "targets": list(item.get("targets", [])),
                "controls": list(item.get("controls", [])),
                "params": list(item.get("parameters", [])),
                "metadata": {},
            }
        )
    measurements = [
        {"kind": item.get("kind", "measure"), "wires": [item.get("wire", 0)], "bits": [item.get("bit", 0)]}
        for item in data.get("measurements", [])
    ]
    return {
        "ir_version": "qb-ir-v0.1",
        "name": data.get("name"),
        "registers": {
            "quantum": {"size": int(data.get("num_qubits", 0))},
            "classical": {"size": int(data.get("num_clbits", 0))},
        },
        "instructions": instructions,
        "measurements": measurements,
        "metadata": {
            "converted_from": "qos_uqci_clean_room_ir",
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
        },
    }


def validate_uqci_ir(payload: dict[str, Any]) -> bool:
    required = {"schema_version", "ir_type", "num_qubits", "operations", "measurements"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"QOS-UQCI IR missing fields: {', '.join(sorted(missing))}")
    if payload["ir_type"] != "qos_uqci_clean_room_ir":
        raise ValueError("QOS-UQCI IR type must be qos_uqci_clean_room_ir")
    if int(payload["num_qubits"]) <= 0:
        raise ValueError("QOS-UQCI IR requires a positive qubit count")
    for op in payload["operations"]:
        if op.get("gate") not in SUPPORTED_UQCI_GATES:
            raise ValueError(f"Unsupported QOS-UQCI gate {op.get('gate')!r}")
    return True
