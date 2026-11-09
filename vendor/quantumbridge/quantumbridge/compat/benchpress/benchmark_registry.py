# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Benchpress was copied.
"""Registry for default QuantumBridge benchmark cases."""

from __future__ import annotations

from .benchmark_case import BenchmarkCase
from .benchmark_suites import full_smoke_suite, suite_by_name


def list_default_benchmark_cases() -> list[BenchmarkCase]:
    return full_smoke_suite()


def filter_benchmark_cases(category: str | None = None, tags: list[str] | None = None) -> list[BenchmarkCase]:
    cases = list_default_benchmark_cases()
    if category is not None:
        cases = [case for case in cases if case.category == category]
    if tags:
        wanted = set(tags)
        cases = [case for case in cases if wanted.intersection(case.tags)]
    return cases


def list_default_benchmark_suites() -> dict[str, list[BenchmarkCase]]:
    return {
        "circuit_basic": suite_by_name("circuit_basic"),
        "simulator": suite_by_name("simulator"),
        "algorithms": suite_by_name("algorithms"),
        "finance_optimization": suite_by_name("finance_optimization"),
        "chemistry": suite_by_name("chemistry"),
        "qml": suite_by_name("qml"),
        "mitigation": suite_by_name("mitigation"),
        "backend": suite_by_name("backend"),
        "bridge": suite_by_name("bridge"),
        "full_smoke": suite_by_name("full_smoke"),
    }
