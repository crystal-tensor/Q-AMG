# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Benchpress was copied.
"""Result wrapping helpers for benchmarking compatibility."""

from __future__ import annotations

from typing import Any

from quantumbridge.schema.benchmark_results import BenchmarkCaseResult

from .warnings import default_benchmark_provenance, default_benchmark_warnings


def wrap_benchmark_result(raw: Any, *, case_id: str = "wrapped", case_name: str = "wrapped") -> BenchmarkCaseResult:
    if hasattr(raw, "to_dict"):
        output = raw.to_dict()
    elif isinstance(raw, dict):
        output = dict(raw)
    else:
        output = {"raw": repr(raw)}
    return BenchmarkCaseResult(
        workflow="benchmark_result_wrap",
        case_id=case_id,
        case_name=case_name,
        category="wrapped",
        output_summary=output,
        warnings=default_benchmark_warnings(),
        provenance=default_benchmark_provenance(),
    )
