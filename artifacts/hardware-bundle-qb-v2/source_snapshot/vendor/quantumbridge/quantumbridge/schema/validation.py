# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_execution_plan_v0.1.md.

from __future__ import annotations

from .errors import SchemaValidationError


REQUIRED_KEYS = {
    "schema_version",
    "counts",
    "probabilities",
    "statevector",
    "density_matrix",
    "expectation_values",
    "metadata",
    "provenance",
}


def _require_mapping(value, key: str) -> None:
    if value is not None and not isinstance(value, dict):
        raise SchemaValidationError(f"QuantumBridge result schema field {key!r} must be an object or null.")


def validate_result_schema(payload: dict) -> bool:
    if not isinstance(payload, dict):
        raise SchemaValidationError("QuantumBridge result schema payload must be a dictionary.")
    missing = sorted(REQUIRED_KEYS - set(payload))
    if missing:
        raise SchemaValidationError(f"QuantumBridge result schema missing required fields: {missing}.")
    if str(payload["schema_version"]) not in {"0.1", "0.2"}:
        raise SchemaValidationError("QuantumBridge result schema version must be 0.1 or 0.2.")
    for key in ("counts", "probabilities", "expectation_values", "metadata", "provenance"):
        _require_mapping(payload.get(key), key)
    if payload.get("errors") is not None and not isinstance(payload.get("errors"), list):
        raise SchemaValidationError("QuantumBridge result schema field 'errors' must be a list.")
    if payload.get("warnings") is not None and not isinstance(payload.get("warnings"), list):
        raise SchemaValidationError("QuantumBridge result schema field 'warnings' must be a list.")
    return True
