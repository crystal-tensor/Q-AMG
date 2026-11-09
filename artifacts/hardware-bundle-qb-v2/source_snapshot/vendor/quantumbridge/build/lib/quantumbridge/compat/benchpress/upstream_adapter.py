# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Benchpress was copied.
"""Optional upstream Benchpress passthrough boundary."""

from __future__ import annotations

from typing import Any

from quantumbridge.schema.benchmark_results import UpstreamBenchpressResult

from .dependency import get_upstream_version, validate_benchpress_dependencies
from .warnings import UPSTREAM_BENCHPRESS_WARNING, default_benchmark_provenance


def run_upstream_benchpress_if_available(*args: Any, package_name: str = "benchpress", **kwargs: Any) -> UpstreamBenchpressResult:
    report = validate_benchpress_dependencies(package_name)
    if not report["available"]:
        return UpstreamBenchpressResult(
            workflow="upstream_benchpress_passthrough",
            mode="upstream_passthrough",
            upstream_package=package_name,
            upstream_version=None,
            capability_level=1,
            native_implementation=False,
            case_id="upstream_benchpress",
            case_name="upstream_benchpress",
            category="upstream",
            status="unsupported",
            passed=False,
            unsupported=True,
            metadata={"dependency_report": report},
            warnings=[UPSTREAM_BENCHPRESS_WARNING],
            provenance=default_benchmark_provenance(),
            unsupported_reason=f"optional {package_name} package is not installed",
        )
    return UpstreamBenchpressResult(
        workflow="upstream_benchpress_passthrough",
        mode="upstream_passthrough",
        upstream_package=package_name,
        upstream_version=get_upstream_version(package_name),
        capability_level=1,
        native_implementation=False,
        case_id="upstream_benchpress",
        case_name="upstream_benchpress",
        category="upstream",
        status="unsupported",
        passed=False,
        unsupported=True,
        metadata={"args_count": len(args), "kwargs": sorted(kwargs)},
        warnings=[UPSTREAM_BENCHPRESS_WARNING],
        provenance=default_benchmark_provenance(),
        unsupported_reason="Stage 10B records the optional Benchpress boundary but does not run official benchmarks",
    )


def wrap_upstream_benchpress_result(raw: Any) -> UpstreamBenchpressResult:
    return UpstreamBenchpressResult(
        workflow="upstream_benchpress_wrap",
        mode="upstream_passthrough",
        upstream_package="benchpress",
        upstream_version=get_upstream_version("benchpress"),
        case_id="upstream_benchpress_wrapped",
        case_name="upstream_benchpress_wrapped",
        category="upstream",
        output_summary={"raw_type": type(raw).__name__, "raw_repr": repr(raw)[:200]},
        warnings=[UPSTREAM_BENCHPRESS_WARNING],
        provenance=default_benchmark_provenance(),
    )
