# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit, PennyLane, or QOS-UQCI was copied.
"""Experimental PennyLane to QOS-UQCI job-spec scaffold."""

from __future__ import annotations

from .qiskit_bridge import pennylane_operations_to_quantumbridge_ir
from .warnings import PENNYLANE_ADAPTER_WARNING, get_provenance


def pennylane_tape_to_qos_uqci_job_spec(tape) -> dict:
    payload = pennylane_operations_to_quantumbridge_ir(getattr(tape, "operations", []))
    payload["measurements"] = [type(measurement).__name__ for measurement in getattr(tape, "measurements", [])]
    shots = _shots_to_metadata(getattr(tape, "shots", None))
    return {
        "schema_version": "0.1",
        "target": "qos_uqci",
        "source": "pennylane",
        "wires": payload.get("wires", []),
        "operations": payload.get("operations", []),
        "measurements": payload.get("measurements", []),
        "shots": shots,
        "uqci_payload": quantumbridge_ir_to_uqci_payload(payload),
        "openqasm_compatibility_artifact": _to_openqasm_compatibility_artifact(payload),
        "warnings": [
            PENNYLANE_ADAPTER_WARNING,
            "QOS-UQCI support is experimental and does not access real QOS or real hardware.",
        ],
        "provenance": get_provenance(mode="qos-uqci-job-spec").to_dict(),
        "experimental": True,
    }


def quantumbridge_ir_to_uqci_payload(ir) -> dict:
    payload = ir.to_dict() if hasattr(ir, "to_dict") else dict(ir)
    return {
        "schema_version": "0.1",
        "schema": "quantumbridge.qos_uqci.payload.v0.1",
        "source": "quantumbridge_ir",
        "wires": payload.get("wires", []),
        "operations": payload.get("operations") or payload.get("instructions", []),
        "measurements": payload.get("measurements", []),
        "shots": _shots_to_metadata(payload.get("shots")),
        "experimental": True,
        "executes_hardware": False,
        "cloud_access": False,
        "token_storage": False,
        "source_ir": payload,
        "warnings": [
            PENNYLANE_ADAPTER_WARNING,
            "QOS-UQCI support is an experimental job-spec scaffold and does not access real hardware.",
        ],
        "provenance": get_provenance(mode="qos-uqci-job-spec").to_dict(),
    }


def uqci_payload_to_quantumbridge_ir(payload) -> dict:
    data = payload.to_dict() if hasattr(payload, "to_dict") else dict(payload)
    return {
        "schema_version": "0.1",
        "ir_version": "qb-ir-v0.1",
        "ecosystem": "pennylane",
        "wires": data.get("wires", []),
        "operations": data.get("operations", []),
        "instructions": data.get("operations", []),
        "measurements": data.get("measurements", []),
        "shots": data.get("shots"),
        "unsupported_operations": [],
        "warnings": data.get("warnings", []),
        "provenance": data.get("provenance", get_provenance(mode="uqci-to-qb-ir").to_dict()),
    }


def qos_uqci_job_spec_to_dict(spec) -> dict:
    return spec.to_dict() if hasattr(spec, "to_dict") else dict(spec)


def validate_qos_uqci_job_spec(spec) -> bool:
    payload = qos_uqci_job_spec_to_dict(spec)
    required = {"schema_version", "target", "source", "operations", "uqci_payload", "experimental"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"QOS-UQCI job spec missing fields: {', '.join(missing)}")
    if payload["target"] != "qos_uqci":
        raise ValueError("QOS-UQCI job spec target must be 'qos_uqci'")
    if payload["source"] != "pennylane":
        raise ValueError("QOS-UQCI job spec source must be 'pennylane'")
    if payload["experimental"] is not True:
        raise ValueError("QOS-UQCI job spec must be marked experimental")
    uqci_payload = payload["uqci_payload"]
    if uqci_payload.get("cloud_access") or uqci_payload.get("token_storage") or uqci_payload.get("executes_hardware"):
        raise ValueError("QOS-UQCI job spec must not access cloud, store tokens, or execute hardware")
    return True


def _to_openqasm_compatibility_artifact(ir_payload: dict) -> dict:
    return {
        "format": "openqasm-compatibility-artifact",
        "generated": False,
        "reason": "Stage 8C records OpenQASM compatibility metadata only; UQCI payload is canonical.",
        "operation_count": len(ir_payload.get("operations") or ir_payload.get("instructions", [])),
    }


def _shots_to_metadata(shots):
    if shots is None:
        return None
    if isinstance(shots, (str, int, float, dict)):
        return shots
    total = getattr(shots, "total_shots", None)
    shot_vector = getattr(shots, "shot_vector", None)
    if total is None and shot_vector is None:
        return None if repr(shots) == "Shots(total=None)" else repr(shots)
    return {
        "total_shots": total,
        "shot_vector": repr(shot_vector) if shot_vector else None,
    }
