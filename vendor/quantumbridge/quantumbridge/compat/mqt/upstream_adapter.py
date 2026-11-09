# This file is independently implemented for QuantumBridge SDK.
# No source code from MQT, IBM, or Qiskit was copied.
"""Optional upstream MQT passthrough boundary helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from quantumbridge.schema.mqt_results import UpstreamMQTResult

from .dependency import dependency_available, get_upstream_version, validate_mqt_dependencies
from .warnings import UPSTREAM_MQT_WARNING, mqt_warnings, upstream_provenance


def run_upstream_mqt_core_if_available(payload: Any = None, **metadata: Any) -> UpstreamMQTResult:
    return _upstream_probe("mqt-core", "upstream_mqt_core_passthrough", payload, metadata)


def run_upstream_ddsim_if_available(payload: Any = None, **metadata: Any) -> UpstreamMQTResult:
    return _upstream_probe("mqt-ddsim", "upstream_mqt_ddsim_passthrough", payload, metadata)


def run_upstream_qmap_if_available(payload: Any = None, **metadata: Any) -> UpstreamMQTResult:
    return _upstream_probe("mqt-qmap", "upstream_mqt_qmap_passthrough", payload, metadata)


def wrap_upstream_mqt_result(
    raw: Any,
    workflow: str = "upstream_mqt_passthrough",
    package_name: str = "mqt",
    metadata: dict[str, Any] | None = None,
) -> UpstreamMQTResult:
    return UpstreamMQTResult(
        workflow=workflow,
        project="mqt",
        mode="upstream_passthrough",
        upstream_package=package_name,
        upstream_version=get_upstream_version(package_name),
        capability_level=1,
        production_ready=False,
        native_implementation=False,
        raw_type=type(raw).__name__,
        metadata={
            **dict(metadata or {}),
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "upstream_passthrough": True,
        },
        warnings=mqt_warnings(UPSTREAM_MQT_WARNING),
        provenance=upstream_provenance(workflow, package_name),
    )


def _upstream_probe(
    package_name: str,
    workflow: str,
    payload: Any,
    metadata: dict[str, Any],
) -> UpstreamMQTResult:
    if not dependency_available(package_name):
        return _unsupported(workflow, package_name, f"optional upstream dependency unavailable: {package_name}")
    try:
        module_name = {
            "mqt-core": "mqt.core",
            "mqt-ddsim": "mqt.ddsim",
            "mqt-qmap": "mqt.qmap",
        }[package_name]
        module = import_module(module_name)
        return wrap_upstream_mqt_result(
            module,
            workflow=workflow,
            package_name=package_name,
            metadata={
                **metadata,
                "payload_type": type(payload).__name__ if payload is not None else None,
                "module_name": module_name,
            },
        )
    except Exception as exc:  # pragma: no cover - optional dependency shape
        return _unsupported(workflow, package_name, f"{type(exc).__name__}: {exc}")


def _unsupported(workflow: str, package_name: str, reason: str) -> UpstreamMQTResult:
    return UpstreamMQTResult(
        workflow=workflow,
        project="mqt",
        mode="upstream_passthrough",
        upstream_package=package_name,
        upstream_version=get_upstream_version(package_name),
        capability_level=1,
        production_ready=False,
        native_implementation=False,
        raw_type="UnsupportedUpstreamMQT",
        metadata={
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "upstream_passthrough": True,
        },
        warnings=mqt_warnings(UPSTREAM_MQT_WARNING),
        provenance=upstream_provenance(workflow, package_name),
        unsupported_reason=reason,
    )


__all__ = [
    "dependency_available",
    "get_upstream_version",
    "run_upstream_ddsim_if_available",
    "run_upstream_mqt_core_if_available",
    "run_upstream_qmap_if_available",
    "validate_mqt_dependencies",
    "wrap_upstream_mqt_result",
]
