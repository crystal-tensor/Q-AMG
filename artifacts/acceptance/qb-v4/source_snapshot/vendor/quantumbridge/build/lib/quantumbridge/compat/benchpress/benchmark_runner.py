# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Benchpress was copied.
"""Local deterministic benchmark runner."""

from __future__ import annotations

import inspect
import time
from importlib.util import find_spec
from typing import Any, Iterable

from quantumbridge.schema.benchmark_results import BenchmarkCaseResult, BenchmarkComparisonResult, BenchmarkSuiteResult

from .benchmark_case import BenchmarkCase, benchmark_case_to_dict
from .reporting import write_benchmark_report_json, write_benchmark_report_markdown
from .warnings import default_benchmark_provenance, default_benchmark_warnings


def run_benchmark_case(case: BenchmarkCase, seed: int | None = None) -> BenchmarkCaseResult:
    case.validate()
    start = time.perf_counter()
    chosen_seed = case.seed if seed is None else seed
    missing_dependencies = _missing_dependencies(case)
    if missing_dependencies:
        elapsed = time.perf_counter() - start
        reason = f"optional dependencies unavailable: {', '.join(missing_dependencies)}"
        return BenchmarkCaseResult(
            workflow="benchmark_case",
            case_id=case.case_id,
            case_name=case.name,
            category=case.category,
            tags=list(case.tags),
            status="skipped",
            elapsed_seconds=elapsed,
            passed=False,
            skipped=True,
            metrics={"missing_dependency_count": len(missing_dependencies)},
            output_summary={"missing_dependencies": missing_dependencies},
            expected_summary=dict(case.expected),
            metadata={"case": benchmark_case_to_dict(case)},
            warnings=list(case.warnings),
            provenance={**default_benchmark_provenance(), **dict(case.provenance)},
            unsupported_reason=reason,
        )
    try:
        if isinstance(case.runner, str):
            raise RuntimeError(f"benchmark runner is not callable: {case.runner}")
        if "seed" in inspect.signature(case.runner).parameters:
            raw = case.runner(seed=chosen_seed)
        else:
            raw = case.runner()
        elapsed = time.perf_counter() - start
        payload = _normalize_runner_payload(raw)
        unsupported_reason = payload.get("unsupported_reason")
        skipped_reason = payload.get("skipped_reason")
        passed = bool(payload.get("passed", unsupported_reason is None and skipped_reason is None))
        status = "unsupported" if unsupported_reason else "skipped" if skipped_reason else "passed" if passed else "failed"
        return BenchmarkCaseResult(
            workflow="benchmark_case",
            case_id=case.case_id,
            case_name=case.name,
            category=case.category,
            tags=list(case.tags),
            status=status,
            elapsed_seconds=elapsed,
            passed=passed,
            skipped=skipped_reason is not None,
            unsupported=unsupported_reason is not None,
            metrics=dict(payload.get("metrics", {})),
            output_summary=dict(payload.get("output_summary", {})),
            expected_summary=dict(payload.get("expected_summary", case.expected)),
            raw_type=type(raw).__name__,
            metadata={"case": benchmark_case_to_dict(case), **dict(payload.get("metadata", {}))},
            warnings=list(case.warnings) + [str(value) for value in payload.get("warnings", [])],
            provenance={**default_benchmark_provenance(), **dict(case.provenance), **dict(payload.get("provenance", {}))},
            unsupported_reason=unsupported_reason or skipped_reason,
        )
    except Exception as exc:  # pragma: no cover - exercised by negative tests through status.
        elapsed = time.perf_counter() - start
        return BenchmarkCaseResult(
            workflow="benchmark_case",
            case_id=case.case_id,
            case_name=case.name,
            category=case.category,
            tags=list(case.tags),
            status="failed",
            elapsed_seconds=elapsed,
            passed=False,
            metrics={"exception": type(exc).__name__},
            output_summary={"error": str(exc)},
            expected_summary=dict(case.expected),
            metadata={"case": benchmark_case_to_dict(case)},
            warnings=default_benchmark_warnings(),
            provenance={**default_benchmark_provenance(), **dict(case.provenance)},
            unsupported_reason=None,
        )


def run_benchmark_suite(
    cases: Iterable[BenchmarkCase],
    seed: int | None = None,
    fail_fast: bool = False,
    suite_name: str = "benchmark_suite",
) -> BenchmarkSuiteResult:
    case_results: list[BenchmarkCaseResult] = []
    start = time.perf_counter()
    for case in cases:
        result = run_benchmark_case(case, seed=seed)
        case_results.append(result)
        if fail_fast and result.status in {"failed", "unsupported"}:
            break
    elapsed = time.perf_counter() - start
    summary = _summary(case_results)
    return BenchmarkSuiteResult(
        workflow=suite_name,
        case_id=suite_name,
        case_name=suite_name,
        category="suite",
        tags=["benchpress", "suite"],
        status="passed" if summary["failed"] == 0 and summary["unsupported"] == 0 else "failed",
        elapsed_seconds=elapsed,
        passed=summary["failed"] == 0 and summary["unsupported"] == 0,
        skipped=summary["skipped"] > 0,
        unsupported=summary["unsupported"] > 0,
        metrics=summary,
        output_summary={"cases": [result.to_dict() for result in case_results]},
        expected_summary={"case_count": len(case_results)},
        report={"summary": summary},
        warnings=default_benchmark_warnings(),
        provenance=default_benchmark_provenance(),
    )


def compare_benchmark_results(results: Iterable[BenchmarkCaseResult | BenchmarkSuiteResult]) -> BenchmarkComparisonResult:
    payloads = [result.to_dict() for result in results]
    passed = all(item.get("passed") for item in payloads)
    return BenchmarkComparisonResult(
        workflow="benchmark_comparison",
        mode="comparison",
        case_id="benchmark_comparison",
        case_name="benchmark_comparison",
        category="comparison",
        status="passed" if passed else "failed",
        passed=passed,
        metrics={"result_count": len(payloads), "passed_count": sum(int(item.get("passed", False)) for item in payloads)},
        output_summary={"results": payloads},
        warnings=default_benchmark_warnings(),
        provenance=default_benchmark_provenance(),
    )


def benchmark_result_to_dict(result: BenchmarkCaseResult | BenchmarkSuiteResult) -> dict[str, Any]:
    return result.to_dict()


def benchmark_suite_result_to_dict(result: BenchmarkSuiteResult) -> dict[str, Any]:
    return result.to_dict()


def _normalize_runner_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if hasattr(raw, "to_dict"):
        return {"output_summary": _to_plain(raw), "passed": True}
    return {"output_summary": {"value": _to_plain(raw)}, "passed": True}


def _missing_dependencies(case: BenchmarkCase) -> list[str]:
    required = case.metadata.get("required_dependencies", [])
    missing: list[str] = []
    for module_name in required:
        normalized = str(module_name)
        try:
            available = find_spec(normalized) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        if not available:
            missing.append(normalized)
    return missing


def _to_plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _summary(results: list[BenchmarkCaseResult]) -> dict[str, int]:
    return {
        "total": len(results),
        "passed": sum(int(result.passed) for result in results),
        "failed": sum(int(result.status == "failed") for result in results),
        "skipped": sum(int(result.skipped) for result in results),
        "unsupported": sum(int(result.unsupported) for result in results),
    }


__all__ = [
    "run_benchmark_case",
    "run_benchmark_suite",
    "compare_benchmark_results",
    "benchmark_result_to_dict",
    "benchmark_suite_result_to_dict",
    "write_benchmark_report_json",
    "write_benchmark_report_markdown",
]
