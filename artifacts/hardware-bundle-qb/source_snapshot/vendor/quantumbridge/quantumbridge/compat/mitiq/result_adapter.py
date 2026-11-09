# This file is independently implemented for QuantumBridge SDK.
# No source code from Mitiq was copied.
"""Result wrapper for Mitiq compatibility workflows."""

from __future__ import annotations

from typing import Any

from quantumbridge.schema.error_mitigation_results import ErrorMitigationResult

from .warnings import MITIQ_WARNING, mitiq_warnings, native_provenance


def wrap_mitiq_result(raw: Any, metadata: dict[str, Any] | None = None) -> ErrorMitigationResult:
    metadata = dict(metadata or {})
    return ErrorMitigationResult(
        workflow=str(metadata.get("workflow", "mitiq_result_wrapper")),
        mode=str(metadata.get("mode", "native_minimal")),
        capability_level=int(metadata.get("capability_level", 2)),
        production_ready=False,
        native_implementation=bool(metadata.get("native_implementation", False)),
        data=raw,
        raw_type=type(raw).__name__,
        metadata={
            **metadata,
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "production_error_mitigation": False,
            "mitiq_parity_claim": False,
        },
        warnings=mitiq_warnings(MITIQ_WARNING),
        provenance=native_provenance("mitiq_result_wrapper"),
    )
