# This file is independently implemented for QuantumBridge SDK.
# No source code from Quafu or pyquafu was copied.
"""Clean-room Quafu-compatible payload conversion helpers."""

from __future__ import annotations

from typing import Any

from quantumbridge.compat.qiskit_aer.simulator_native import normalize_circuit_to_quantumbridge_ir

from .warnings import native_provenance, quafu_warnings

SUPPORTED_QUAFU_GATES = {"x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "phase", "cx", "cz", "swap"}


def quantumbridge_ir_to_quafu_payload(circuit_or_ir: Any) -> dict[str, Any]:
    circuit = normalize_circuit_to_quantumbridge_ir(circuit_or_ir)
    gates = []
    for index, op in enumerate(circuit.operations):
        if op.name not in SUPPORTED_QUAFU_GATES:
            raise ValueError(f"Stage 10A Quafu adapter does not support operation {op.name!r}")
        gates.append(
            {
                "index": index,
                "name": op.name,
                "targets": list(op.targets),
                "controls": list(op.controls),
                "params": [float(param) for param in op.params],
            }
        )
    return {
        "schema_version": "0.1",
        "payload_type": "quafu_clean_room_payload",
        "source": "quantumbridge_ir",
        "name": circuit.name,
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_bits,
        "gates": gates,
        "measurements": [
            {"wire": measurement.wire, "bit": measurement.bit, "kind": measurement.kind}
            for measurement in circuit.measurements
        ],
        "metadata": {
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "production_backend_claim": False,
            "official_endorsement_claim": False,
        },
        "warnings": quafu_warnings(),
        "provenance": native_provenance("quantumbridge_ir_to_quafu_payload"),
    }


def quafu_payload_to_quantumbridge_ir(payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    return {
        "ir_version": "qb-ir-v0.1",
        "name": data.get("name"),
        "registers": {
            "quantum": {"size": int(data.get("num_qubits", 0))},
            "classical": {"size": int(data.get("num_clbits", 0))},
        },
        "instructions": [
            {
                "op": gate.get("name"),
                "targets": list(gate.get("targets", [])),
                "controls": list(gate.get("controls", [])),
                "params": list(gate.get("params", [])),
                "metadata": {},
            }
            for gate in data.get("gates", [])
        ],
        "measurements": [
            {"kind": item.get("kind", "measure"), "wires": [item.get("wire", 0)], "bits": [item.get("bit", 0)]}
            for item in data.get("measurements", [])
        ],
        "metadata": {
            "converted_from": "quafu_clean_room_payload",
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
        },
    }


def validate_quafu_payload(payload: dict[str, Any]) -> bool:
    required = {"schema_version", "payload_type", "num_qubits", "gates", "measurements"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Quafu payload missing fields: {', '.join(sorted(missing))}")
    if payload["payload_type"] != "quafu_clean_room_payload":
        raise ValueError("Quafu payload_type must be quafu_clean_room_payload")
    if int(payload["num_qubits"]) <= 0:
        raise ValueError("Quafu payload requires a positive qubit count")
    for gate in payload["gates"]:
        if gate.get("name") not in SUPPORTED_QUAFU_GATES:
            raise ValueError(f"Unsupported Quafu-compatible gate {gate.get('name')!r}")
    return True
