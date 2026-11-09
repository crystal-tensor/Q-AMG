# This file is independently implemented for QuantumBridge SDK.
"""Optional pyquafu upstream boundary."""

from __future__ import annotations

from typing import Any

from quantumbridge.schema.backend_results import UpstreamQuafuResult

from .dependency import validate_quafu_dependencies
from .warnings import UPSTREAM_QUAFU_WARNING, quafu_warnings, upstream_provenance


def run_upstream_pyquafu_if_available(*args: Any, package_name: str = "quafu", **kwargs: Any) -> UpstreamQuafuResult:
    dependency = validate_quafu_dependencies(package_name)
    if not dependency["available"]:
        return UpstreamQuafuResult(
            workflow="upstream_pyquafu_passthrough",
            project="quafu",
            mode="upstream_passthrough",
            capability_level=0,
            production_ready=False,
            native_implementation=False,
            upstream_package=package_name,
            upstream_version=dependency["version"],
            unsupported_reason=f"optional {package_name} package is not installed",
            metadata={"dependency": dependency, "args_count": len(args), "kwargs": sorted(kwargs)},
            warnings=quafu_warnings(UPSTREAM_QUAFU_WARNING),
            provenance=upstream_provenance("upstream_pyquafu_passthrough", package_name),
        )
    return UpstreamQuafuResult(
        workflow="upstream_pyquafu_passthrough",
        project="quafu",
        mode="upstream_passthrough",
        capability_level=1,
        production_ready=False,
        native_implementation=False,
        upstream_package=package_name,
        upstream_version=dependency["version"],
        unsupported_reason="Stage 10A records the optional pyquafu boundary but does not submit cloud jobs",
        metadata={"dependency": dependency, "args_count": len(args), "kwargs": sorted(kwargs)},
        warnings=quafu_warnings(UPSTREAM_QUAFU_WARNING),
        provenance=upstream_provenance("upstream_pyquafu_passthrough", package_name),
    )
