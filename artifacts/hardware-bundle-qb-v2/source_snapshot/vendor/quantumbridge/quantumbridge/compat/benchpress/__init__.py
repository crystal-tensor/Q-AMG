# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Benchpress was copied.
"""Stage 10B clean-room benchmarking compatibility."""

from quantumbridge.compat.benchpress.benchmark_case import (
    BenchmarkCase,
    benchmark_case_from_dict,
    benchmark_case_to_dict,
    create_benchmark_case,
    validate_benchmark_case,
)
from quantumbridge.compat.benchpress.benchmark_registry import (
    filter_benchmark_cases,
    list_default_benchmark_cases,
    list_default_benchmark_suites,
)
from quantumbridge.compat.benchpress.benchmark_runner import (
    benchmark_result_to_dict,
    benchmark_suite_result_to_dict,
    compare_benchmark_results,
    run_benchmark_case,
    run_benchmark_suite,
)
from quantumbridge.compat.benchpress.benchmark_suites import (
    algorithms_suite,
    backend_suite,
    bridge_suite,
    chemistry_suite,
    circuit_basic_suite,
    finance_optimization_suite,
    full_smoke_suite,
    mitigation_suite,
    qml_suite,
    simulator_suite,
)
from quantumbridge.compat.benchpress.dependency import (
    dependency_available,
    get_upstream_version,
    validate_benchpress_dependencies,
)
from quantumbridge.compat.benchpress.reporting import (
    write_benchmark_report_json,
    write_benchmark_report_markdown,
)
from quantumbridge.compat.benchpress.upstream_adapter import (
    run_upstream_benchpress_if_available,
    wrap_upstream_benchpress_result,
)

__all__ = [
    "BenchmarkCase",
    "create_benchmark_case",
    "validate_benchmark_case",
    "benchmark_case_to_dict",
    "benchmark_case_from_dict",
    "list_default_benchmark_cases",
    "filter_benchmark_cases",
    "list_default_benchmark_suites",
    "run_benchmark_case",
    "run_benchmark_suite",
    "compare_benchmark_results",
    "benchmark_result_to_dict",
    "benchmark_suite_result_to_dict",
    "write_benchmark_report_json",
    "write_benchmark_report_markdown",
    "dependency_available",
    "get_upstream_version",
    "validate_benchpress_dependencies",
    "run_upstream_benchpress_if_available",
    "wrap_upstream_benchpress_result",
    "circuit_basic_suite",
    "simulator_suite",
    "algorithms_suite",
    "finance_optimization_suite",
    "chemistry_suite",
    "qml_suite",
    "mitigation_suite",
    "backend_suite",
    "bridge_suite",
    "full_smoke_suite",
]
