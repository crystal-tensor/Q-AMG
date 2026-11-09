# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""PennyLane tape metadata and minimal IR bridge."""

from __future__ import annotations

from quantumbridge.compat.contracts import CapabilityLevel

from .measurements_adapter import measurement_to_metadata
from .operations_adapter import operation_sequence_to_quantumbridge_ir, operation_to_metadata
from .warnings import adapter_metadata


def describe_tape(tape) -> dict:
    operations = list(getattr(tape, "operations", []))
    measurements = list(getattr(tape, "measurements", []))
    wires = getattr(tape, "wires", [])
    return {
        "operation_count": len(operations),
        "measurement_count": len(measurements),
        "operations": [getattr(op, "name", type(op).__name__) for op in operations],
        "measurements": [type(measurement).__name__ for measurement in measurements],
        "wires": [_wire_to_python(wire) for wire in wires],
        "shots": _shots_to_metadata(getattr(tape, "shots", None)),
        **adapter_metadata(CapabilityLevel.SCHEMA_ADAPTER),
    }


def _wire_to_python(wire):
    try:
        return int(wire)
    except (TypeError, ValueError):
        return str(wire)


def _shots_to_metadata(shots):
    if shots is None:
        return None
    total = getattr(shots, "total_shots", None)
    shot_vector = getattr(shots, "shot_vector", None)
    if total is None and shot_vector is None:
        return None if repr(shots) == "Shots(total=None)" else repr(shots)
    return {
        "total_shots": total,
        "shot_vector": repr(shot_vector) if shot_vector else None,
    }


def tape_to_operation_metadata(tape) -> list[dict]:
    return [operation_to_metadata(op) for op in getattr(tape, "operations", [])]


def tape_to_measurement_metadata(tape) -> list[dict]:
    return [measurement_to_metadata(measurement) for measurement in getattr(tape, "measurements", [])]


def tape_to_quantumbridge_ir(tape) -> dict:
    operations = list(getattr(tape, "operations", []))
    measurements = tape_to_measurement_metadata(tape)
    ir = operation_sequence_to_quantumbridge_ir(operations)
    all_wires = set(ir.get("wires", []))
    for measurement in measurements:
        all_wires.update(measurement.get("wires", []))
    warning_messages = list(ir.get("warnings", []))
    for measurement in measurements:
        if measurement.get("supported") is False:
            warning_messages.append(measurement["unsupported_reason"])
    return {
        **ir,
        "ecosystem": "pennylane",
        "wires": sorted(all_wires, key=str),
        "measurements": measurements,
        "shots": _shots_to_metadata(getattr(tape, "shots", None)),
        "unsupported_operations": ir.get("unsupported_operations", []),
        "warnings": warning_messages,
        "provenance": adapter_metadata(CapabilityLevel.SCHEMA_ADAPTER)["provenance"],
    }


def quantumbridge_ir_to_tape_spec(ir) -> dict:
    payload = ir.to_dict() if hasattr(ir, "to_dict") else dict(ir)
    operations = payload.get("operations") or payload.get("instructions", [])
    return {
        "schema_version": "0.1",
        "ecosystem": "pennylane",
        "metadata_only": True,
        "wires": payload.get("wires", []),
        "operations": operations,
        "measurements": payload.get("measurements", []),
        "parameters": payload.get("parameters", []),
        "shots": payload.get("shots"),
        "unsupported_operations": payload.get("unsupported_operations", []),
        "warnings": payload.get("warnings", []),
        "provenance": adapter_metadata(CapabilityLevel.SCHEMA_ADAPTER)["provenance"],
    }


def tape_summary(tape) -> dict:
    description = describe_tape(tape)
    return {
        "operation_count": description["operation_count"],
        "measurement_count": description["measurement_count"],
        "wires": description["wires"],
        "shots": description["shots"],
        "warnings": description["warnings"],
        "provenance": description["provenance"],
    }
