# This file is independently implemented for QuantumBridge SDK.
# No source code from MQT, IBM, or Qiskit was copied.
"""Result wrapping helpers for Stage 9J MQT compatibility."""

from __future__ import annotations

from typing import Any

from quantumbridge.schema.mqt_results import MQTCompatibilityResult

from .warnings import mqt_warnings, native_provenance


def wrap_mqt_result(raw: Any, metadata: dict[str, Any] | None = None) -> MQTCompatibilityResult:
    if isinstance(raw, MQTCompatibilityResult):
        return raw
    payload = dict(raw) if isinstance(raw, dict) else {"data": raw}
    return MQTCompatibilityResult(
        workflow=str(payload.get("workflow", "mqt_result_wrapper")),
        project=str(payload.get("project", "mqt")),
        mode=str(payload.get("mode", "native_minimal")),
        capability_level=int(payload.get("capability_level", 2)),
        production_ready=False,
        native_implementation=bool(payload.get("native_implementation", False)),
        raw_type=type(raw).__name__,
        metadata={
            **dict(metadata or {}),
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "full_replacement_claim": False,
            "production_parity_claim": False,
        },
        warnings=mqt_warnings(),
        provenance=native_provenance("mqt_result_wrapper"),
    )


to_quantumbridge_schema = wrap_mqt_result

__all__ = ["to_quantumbridge_schema", "wrap_mqt_result"]
