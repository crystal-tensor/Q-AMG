# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Qiskit Experiments was copied.
"""Optional upstream boundary for qiskit-experiments."""

from __future__ import annotations

from typing import Any

from quantumbridge.schema.experiments_results import UpstreamExperimentsResult

from .dependency import dependency_available, get_upstream_version, validate_experiments_dependencies
from .warnings import upstream_provenance, upstream_warnings


def run_upstream_experiments_if_available(raw: Any = None, workflow: str = "upstream_experiments_passthrough") -> UpstreamExperimentsResult:
    if not dependency_available():
        return UpstreamExperimentsResult(
            workflow=workflow,
            mode="upstream_passthrough",
            capability_level=1,
            upstream_package="qiskit-experiments",
            upstream_version=None,
            native_implementation=False,
            raw_type=None,
            data=None,
            metadata=validate_experiments_dependencies(),
            warnings=upstream_warnings("qiskit-experiments is not installed; upstream path skipped."),
            provenance=upstream_provenance(workflow, None),
            unsupported_reason="qiskit-experiments is not installed",
        )
    return wrap_upstream_experiments_result(raw, workflow=workflow)


def wrap_upstream_experiments_result(raw: Any, workflow: str = "upstream_experiments_passthrough") -> UpstreamExperimentsResult:
    version = get_upstream_version()
    return UpstreamExperimentsResult(
        workflow=workflow,
        mode="upstream_passthrough",
        capability_level=2,
        upstream_package="qiskit-experiments",
        upstream_version=version,
        native_implementation=False,
        raw_type=type(raw).__name__ if raw is not None else None,
        data=repr(raw) if raw is not None else None,
        metadata=validate_experiments_dependencies(),
        warnings=upstream_warnings(),
        provenance=upstream_provenance(workflow, version),
    )
