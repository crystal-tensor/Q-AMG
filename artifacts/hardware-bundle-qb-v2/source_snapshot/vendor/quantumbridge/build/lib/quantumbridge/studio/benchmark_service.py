# This file is independently implemented for QuantumBridge SDK.
"""Benchmark service wrapping the Stage 10B benchmark executable slice."""

from __future__ import annotations

from quantumbridge.compat.benchpress.benchmark_registry import list_default_benchmark_suites
from quantumbridge.compat.benchpress.benchmark_runner import run_benchmark_suite as run_benchpress_suite

from .api_models import StudioBenchmarkReport


def list_benchmark_suites() -> list[str]:
    return sorted(list_default_benchmark_suites())


def run_benchmark_suite_by_id(suite_id: str) -> StudioBenchmarkReport:
    suites = list_default_benchmark_suites()
    if suite_id not in suites:
        raise KeyError(f"unknown benchmark suite_id: {suite_id}")
    result = run_benchpress_suite(suites[suite_id], suite_name=f"studio_{suite_id}")
    return StudioBenchmarkReport(
        suite_id=suite_id,
        result=result.to_dict(),
        report_json=result.to_json(),
        report_markdown=_render_markdown(result),
        warnings=list(result.warnings),
        provenance=dict(result.provenance),
    )


def run_benchmark_suite(suite_id: str) -> StudioBenchmarkReport:
    return run_benchmark_suite_by_id(suite_id)


def run_full_smoke_benchmark() -> StudioBenchmarkReport:
    return run_benchmark_suite_by_id("full_smoke")


def get_benchmark_report(format: str = "json", suite_id: str = "circuit_basic") -> str:
    report = run_benchmark_suite_by_id(suite_id)
    if format == "json":
        return report.report_json
    if format == "markdown":
        return report.report_markdown
    raise ValueError("format must be json or markdown")


def _render_markdown(result) -> str:
    payload = result.to_dict()
    lines = [
        "# QuantumBridge Studio Benchmark Report",
        "",
        f"- workflow: {payload.get('workflow')}",
        f"- status: {payload.get('status')}",
        f"- passed: {payload.get('passed')}",
        f"- elapsed_seconds: {payload.get('elapsed_seconds')}",
        "",
        "## Boundaries",
        "",
        "- not official Benchpress output",
        "- not production performance ranking",
        "- no cloud/token/hardware access",
    ]
    return "\n".join(lines) + "\n"
