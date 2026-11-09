# This file is independently implemented for QuantumBridge SDK.
"""Optional upstream QOS-UQCI boundary."""

from __future__ import annotations

from typing import Any

from quantumbridge.schema.backend_results import UpstreamQOSUQCIResult

from .dependency import validate_qos_uqci_dependencies
from .warnings import UPSTREAM_QOS_WARNING, qos_uqci_warnings, upstream_provenance


def run_upstream_qos_uqci_if_available(*args: Any, package_name: str = "qos_uqci", **kwargs: Any) -> UpstreamQOSUQCIResult:
    dependency = validate_qos_uqci_dependencies(package_name)
    if not dependency["available"]:
        return UpstreamQOSUQCIResult(
            workflow="upstream_qos_uqci_passthrough",
            project="qos-uqci",
            mode="upstream_passthrough",
            capability_level=0,
            production_ready=False,
            native_implementation=False,
            upstream_package=package_name,
            upstream_version=dependency["version"],
            unsupported_reason=f"optional {package_name} package is not installed",
            metadata={"dependency": dependency, "args_count": len(args), "kwargs": sorted(kwargs)},
            warnings=qos_uqci_warnings(UPSTREAM_QOS_WARNING),
            provenance=upstream_provenance("upstream_qos_uqci_passthrough", package_name),
        )
    return UpstreamQOSUQCIResult(
        workflow="upstream_qos_uqci_passthrough",
        project="qos-uqci",
        mode="upstream_passthrough",
        capability_level=1,
        production_ready=False,
        native_implementation=False,
        upstream_package=package_name,
        upstream_version=dependency["version"],
        unsupported_reason="Stage 10A records the optional upstream boundary but does not submit cloud jobs",
        metadata={"dependency": dependency, "args_count": len(args), "kwargs": sorted(kwargs)},
        warnings=qos_uqci_warnings(UPSTREAM_QOS_WARNING),
        provenance=upstream_provenance("upstream_qos_uqci_passthrough", package_name),
    )
