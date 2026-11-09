# This file is independently implemented for QuantumBridge SDK.
# No source code from TorchQuantum or PyTorch was copied.
"""Optional upstream TorchQuantum passthrough boundary."""

from __future__ import annotations

from typing import Any

from quantumbridge.schema.torchquantum_results import UpstreamTorchQuantumResult

from .dependency import dependency_available, get_upstream_version, validate_torchquantum_dependencies
from .warnings import UPSTREAM_TORCHQUANTUM_WARNING, torchquantum_warnings, upstream_provenance


def run_upstream_torchquantum_if_available(*args: Any, **kwargs: Any) -> UpstreamTorchQuantumResult:
    available = dependency_available("torchquantum")
    if not available:
        return UpstreamTorchQuantumResult(
            workflow="upstream_torchquantum_passthrough",
            mode="upstream_passthrough",
            upstream_package="torchquantum",
            upstream_version=None,
            capability_level=0,
            production_ready=False,
            native_implementation=False,
            raw_type=None,
            metadata={
                "dependency": validate_torchquantum_dependencies(),
                "args_count": len(args),
                "kwargs": sorted(str(key) for key in kwargs),
                "cloud_access": False,
                "token_read": False,
                "hardware_access": False,
            },
            warnings=torchquantum_warnings(UPSTREAM_TORCHQUANTUM_WARNING),
            provenance=upstream_provenance("upstream_torchquantum_passthrough"),
            unsupported_reason="optional torchquantum package is not installed",
        )
    return UpstreamTorchQuantumResult(
        workflow="upstream_torchquantum_passthrough",
        mode="upstream_passthrough",
        upstream_package="torchquantum",
        upstream_version=get_upstream_version("torchquantum"),
        capability_level=1,
        production_ready=False,
        native_implementation=False,
        raw_type="torchquantum",
        metadata={
            "dependency": validate_torchquantum_dependencies(),
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "boundary_only": True,
        },
        warnings=torchquantum_warnings(UPSTREAM_TORCHQUANTUM_WARNING),
        provenance=upstream_provenance("upstream_torchquantum_passthrough"),
    )


def wrap_upstream_torchquantum_result(raw: Any) -> UpstreamTorchQuantumResult:
    return UpstreamTorchQuantumResult(
        workflow="wrap_upstream_torchquantum_result",
        mode="upstream_passthrough",
        upstream_package="torchquantum",
        upstream_version=get_upstream_version("torchquantum"),
        capability_level=1,
        production_ready=False,
        native_implementation=False,
        raw_type=type(raw).__name__,
        metadata={"repr": repr(raw)[:200], "cloud_access": False, "token_read": False, "hardware_access": False},
        warnings=torchquantum_warnings(UPSTREAM_TORCHQUANTUM_WARNING),
        provenance=upstream_provenance("wrap_upstream_torchquantum_result"),
    )
