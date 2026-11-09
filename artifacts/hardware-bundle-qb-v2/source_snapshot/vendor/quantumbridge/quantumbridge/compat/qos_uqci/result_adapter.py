# This file is independently implemented for QuantumBridge SDK.
"""Result wrapping helpers for QOS-UQCI compatibility."""

from __future__ import annotations

from typing import Any

from quantumbridge.schema.backend_results import BackendCompatibilityResult, QOSUQCIBackendResult


def wrap_qos_uqci_result(result: Any) -> BackendCompatibilityResult:
    if isinstance(result, BackendCompatibilityResult):
        return result
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    return QOSUQCIBackendResult(
        workflow=payload.get("workflow", "qos_uqci_result_wrapper"),
        project="qos-uqci",
        mode=payload.get("mode", "native_mock"),
        job_spec=dict(payload.get("job_spec", {})),
        payload=dict(payload.get("payload", {})),
        counts=dict(payload.get("counts", {})),
        probabilities=dict(payload.get("probabilities", {})),
        metadata=dict(payload.get("metadata", {})),
        warnings=list(payload.get("warnings", [])),
        provenance=dict(payload.get("provenance", {})),
    )


def to_quantumbridge_schema(result: Any) -> dict[str, Any]:
    return wrap_qos_uqci_result(result).to_dict()
