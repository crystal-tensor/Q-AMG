# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Benchpress was copied.
"""Example helpers for Stage 10B benchmarking compatibility."""

from __future__ import annotations

from tempfile import TemporaryDirectory
from pathlib import Path

from .benchmark_runner import run_benchmark_suite
from .benchmark_suites import algorithms_suite, backend_suite, circuit_basic_suite, full_smoke_suite
from .reporting import write_benchmark_report_json, write_benchmark_report_markdown
from .upstream_adapter import run_upstream_benchpress_if_available


def run_basic_suite_example() -> dict[str, object]:
    return _run_suite_example("benchpress_basic_suite", circuit_basic_suite())


def run_algorithms_suite_example() -> dict[str, object]:
    return _run_suite_example("benchpress_algorithms_suite", algorithms_suite())


def run_backend_suite_example() -> dict[str, object]:
    return _run_suite_example("benchpress_backend_suite", backend_suite())


def run_full_smoke_suite_example() -> dict[str, object]:
    return _run_suite_example("benchpress_full_smoke_suite", full_smoke_suite())


def _run_suite_example(name: str, cases) -> dict[str, object]:
    result = run_benchmark_suite(cases, suite_name=name)
    upstream = run_upstream_benchpress_if_available()
    with TemporaryDirectory() as tmpdir:
        json_path = write_benchmark_report_json(result, Path(tmpdir) / f"{name}.json")
        md_path = write_benchmark_report_markdown(result, Path(tmpdir) / f"{name}.md")
        json_text = json_path.read_text(encoding="utf-8")
        markdown_text = md_path.read_text(encoding="utf-8")
    return {
        "native": result,
        "upstream": upstream,
        "json_report": json_text,
        "markdown_report": markdown_text,
        "warnings": list(result.warnings),
        "provenance": dict(result.provenance),
        "cloud_access": False,
        "token_access": False,
        "hardware_access": False,
        "official_benchmark_claim": False,
        "production_benchmark_parity_claim": False,
    }
