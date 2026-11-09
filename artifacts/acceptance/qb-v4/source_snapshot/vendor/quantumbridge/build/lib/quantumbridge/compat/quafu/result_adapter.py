# This file is independently implemented for QuantumBridge SDK.
"""Result wrapping helpers for Quafu compatibility."""

from __future__ import annotations

from typing import Any

from quantumbridge.schema.backend_results import BackendCompatibilityResult, QuafuBackendResult


def wrap_quafu_result(result: Any) -> BackendCompatibilityResult:
    if isinstance(result, BackendCompatibilityResult):
        return result
    payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    return QuafuBackendResult(
        workflow=payload.get("workflow", "quafu_result_wrapper"),
        project="quafu",
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
    return wrap_quafu_result(result).to_dict()
