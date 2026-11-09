# This file is independently implemented for QuantumBridge SDK.
"""Quafu-compatible offline job spec helpers."""

from __future__ import annotations

from typing import Any

from .payload_adapter import quantumbridge_ir_to_quafu_payload, validate_quafu_payload
from .warnings import native_provenance, quafu_warnings


def build_quafu_job_spec(
    circuit_or_ir: Any,
    shots: int = 1024,
    seed: int | None = None,
    backend_name: str = "quantumbridge_mock_quafu_backend",
) -> dict[str, Any]:
    payload = quantumbridge_ir_to_quafu_payload(circuit_or_ir)
    job_spec = {
        "schema_version": "0.1",
        "job_id": f"qb-quafu-{payload.get('name') or 'job'}",
        "target": "quafu",
        "backend_name": backend_name,
        "execution_mode": "offline_mock",
        "shots": int(shots),
        "seed": seed,
        "payload": payload,
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
        "production_backend": False,
        "official_endorsement_claim": False,
        "warnings": quafu_warnings(),
        "provenance": native_provenance("build_quafu_job_spec"),
    }
    validate_quafu_job_spec(job_spec)
    return job_spec


def validate_quafu_job_spec(job_spec: dict[str, Any]) -> bool:
    required = {"schema_version", "job_id", "target", "execution_mode", "shots", "payload"}
    missing = required - set(job_spec)
    if missing:
        raise ValueError(f"Quafu job spec missing fields: {', '.join(sorted(missing))}")
    if job_spec["target"] != "quafu":
        raise ValueError("Quafu job spec target must be quafu")
    if job_spec["execution_mode"] != "offline_mock":
        raise ValueError("Quafu job spec execution_mode must be offline_mock")
    if int(job_spec["shots"]) <= 0:
        raise ValueError("Quafu job spec shots must be positive")
    for key in ("cloud_access", "token_read", "hardware_access", "production_backend", "official_endorsement_claim"):
        if job_spec.get(key) is True:
            raise ValueError(f"Quafu job spec must not set {key}=True")
    validate_quafu_payload(job_spec["payload"])
    return True
