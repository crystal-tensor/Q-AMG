# This file is independently implemented for QuantumBridge SDK.
# No source code from PennyLane-Qiskit was copied.
"""Optional upstream PennyLane-Qiskit passthrough metadata."""

from __future__ import annotations

from typing import Any

from quantumbridge.schema.pennylane_qiskit_bridge_results import UpstreamPennyLaneQiskitResult

from .dependency import (
    dependency_available,
    get_upstream_version,
    validate_pennylane_qiskit_dependencies,
)
from .warnings import UPSTREAM_PLUGIN_WARNING, bridge_warnings, upstream_provenance


def run_with_upstream_pennylane_qiskit_if_available(
    workflow: str = "upstream_pennylane_qiskit_passthrough",
    payload: Any = None,
) -> UpstreamPennyLaneQiskitResult:
    dependency = validate_pennylane_qiskit_dependencies()
    if not dependency["available"]:
        return UpstreamPennyLaneQiskitResult(
            workflow=workflow,
            mode="upstream_passthrough",
            source_ecosystem="pennylane_qiskit",
            target_ecosystem="quantumbridge",
            capability_level=0,
            upstream_package="pennylane-qiskit",
            upstream_version=None,
            production_ready=False,
            native_implementation=False,
            raw_type=None,
            metadata={
                **dependency,
                "cloud_access": False,
                "token_read": False,
                "hardware_access": False,
                "requires_user_supplied_executor": True,
            },
            warnings=bridge_warnings(UPSTREAM_PLUGIN_WARNING),
            provenance=upstream_provenance(workflow),
            unsupported_reason="optional pennylane-qiskit package is not installed",
        )
    return wrap_upstream_pennylane_qiskit_result(
        payload if payload is not None else {"dependency_available": True},
        workflow=workflow,
    )


def wrap_upstream_pennylane_qiskit_result(
    raw: Any,
    workflow: str = "upstream_pennylane_qiskit_passthrough",
) -> UpstreamPennyLaneQiskitResult:
    version = get_upstream_version()
    return UpstreamPennyLaneQiskitResult(
        workflow=workflow,
        mode="upstream_passthrough",
        source_ecosystem="pennylane_qiskit",
        target_ecosystem="quantumbridge",
        capability_level=1,
        upstream_package="pennylane-qiskit",
        upstream_version=version,
        production_ready=False,
        native_implementation=False,
        raw_type=type(raw).__name__,
        data=_json_safe(raw),
        metadata={
            "dependency_available": dependency_available(),
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "upstream_passthrough": True,
            "requires_user_supplied_executor": True,
        },
        warnings=bridge_warnings(UPSTREAM_PLUGIN_WARNING),
        provenance=upstream_provenance(workflow),
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            pass
    return repr(value)
