# This file is independently implemented for QuantumBridge SDK.
# No source code from Mitiq was copied.
"""Optional upstream Mitiq passthrough result helpers."""

from __future__ import annotations

from typing import Any

from quantumbridge.schema.error_mitigation_results import UpstreamMitiqResult

from .dependency import dependency_available, get_upstream_version
from .warnings import UPSTREAM_MITIQ_WARNING, mitiq_warnings, upstream_provenance


def validate_mitiq_dependencies() -> dict[str, Any]:
    return {
        "upstream_package": "mitiq",
        "available": dependency_available(),
        "upstream_version": get_upstream_version(),
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
        "warnings": mitiq_warnings(UPSTREAM_MITIQ_WARNING),
    }


def run_zne_upstream_if_available(*args: Any, **kwargs: Any) -> UpstreamMitiqResult:
    if not dependency_available():
        return _unsupported(
            "upstream_mitiq_zne",
            "optional upstream dependency unavailable: mitiq",
        )
    return wrap_upstream_mitiq_result(
        {
            "args_supplied": bool(args),
            "kwargs": sorted(str(key) for key in kwargs),
            "requires_user_executor": True,
        },
        workflow="upstream_mitiq_zne",
        raw_type="MitiqDependencyAvailable",
        unsupported_reason=(
            "upstream mitiq execution requires an explicit user-provided executor/backend; "
            "QuantumBridge did not access cloud, hardware, or tokens"
        ),
    )


def run_readout_mitigation_upstream_if_available(*args: Any, **kwargs: Any) -> UpstreamMitiqResult:
    if not dependency_available():
        return _unsupported(
            "upstream_mitiq_readout_mitigation",
            "optional upstream dependency unavailable: mitiq",
        )
    return wrap_upstream_mitiq_result(
        {
            "args_supplied": bool(args),
            "kwargs": sorted(str(key) for key in kwargs),
            "requires_user_calibration_data": True,
        },
        workflow="upstream_mitiq_readout_mitigation",
        raw_type="MitiqDependencyAvailable",
        unsupported_reason=(
            "upstream mitiq readout mitigation requires explicit calibration data; "
            "QuantumBridge did not access cloud, hardware, or tokens"
        ),
    )


def wrap_upstream_mitiq_result(
    raw: Any,
    workflow: str = "upstream_mitiq_passthrough",
    raw_type: str | None = None,
    unsupported_reason: str | None = None,
) -> UpstreamMitiqResult:
    return UpstreamMitiqResult(
        workflow=workflow,
        mode="upstream_passthrough",
        capability_level=1,
        upstream_package="mitiq",
        upstream_version=get_upstream_version(),
        production_ready=False,
        native_implementation=False,
        raw_type=raw_type or type(raw).__name__,
        data=raw,
        metadata={
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "production_error_mitigation": False,
            "mitiq_parity_claim": False,
        },
        warnings=mitiq_warnings(UPSTREAM_MITIQ_WARNING),
        provenance=upstream_provenance(workflow),
        unsupported_reason=unsupported_reason,
    )


def _unsupported(workflow: str, reason: str) -> UpstreamMitiqResult:
    return wrap_upstream_mitiq_result(
        {"available": False},
        workflow=workflow,
        raw_type="UnsupportedUpstreamMitiq",
        unsupported_reason=reason,
    )
