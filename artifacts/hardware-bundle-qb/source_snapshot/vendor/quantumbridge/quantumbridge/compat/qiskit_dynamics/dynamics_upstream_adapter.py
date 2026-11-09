# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Qiskit Dynamics was copied.
"""Optional upstream boundary for qiskit-dynamics."""

from __future__ import annotations

from typing import Any

from quantumbridge.schema.dynamics_results import UpstreamDynamicsResult

from .dependency import dependency_available, get_upstream_version, validate_dynamics_dependencies
from .warnings import upstream_provenance, upstream_warnings


def run_upstream_dynamics_if_available(raw: Any = None, workflow: str = "upstream_dynamics_passthrough") -> UpstreamDynamicsResult:
    if not dependency_available():
        return UpstreamDynamicsResult(
            workflow=workflow,
            mode="upstream_passthrough",
            capability_level=1,
            upstream_package="qiskit-dynamics",
            upstream_version=None,
            native_implementation=False,
            raw_type=None,
            data=None,
            metadata=validate_dynamics_dependencies(),
            warnings=upstream_warnings("qiskit-dynamics is not installed; upstream path skipped."),
            provenance=upstream_provenance(workflow, None),
            unsupported_reason="qiskit-dynamics is not installed",
        )
    return wrap_upstream_dynamics_result(raw, workflow=workflow)


def wrap_upstream_dynamics_result(raw: Any, workflow: str = "upstream_dynamics_passthrough") -> UpstreamDynamicsResult:
    version = get_upstream_version()
    return UpstreamDynamicsResult(
        workflow=workflow,
        mode="upstream_passthrough",
        capability_level=2,
        upstream_package="qiskit-dynamics",
        upstream_version=version,
        native_implementation=False,
        raw_type=type(raw).__name__ if raw is not None else None,
        data=repr(raw) if raw is not None else None,
        metadata=validate_dynamics_dependencies(),
        warnings=upstream_warnings(),
        provenance=upstream_provenance(workflow, version),
    )
