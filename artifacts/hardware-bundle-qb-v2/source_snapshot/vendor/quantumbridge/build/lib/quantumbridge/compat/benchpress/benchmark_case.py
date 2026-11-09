# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Benchpress was copied.
"""Clean-room benchmark case model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .warnings import default_benchmark_provenance, default_benchmark_warnings


BenchmarkRunner = Callable[..., Any]


@dataclass
class BenchmarkCase:
    case_id: str
    name: str
    category: str
    workload_type: str
    runner: BenchmarkRunner | str
    workload: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    timeout_seconds: float = 10.0
    deterministic: bool = True
    seed: int | None = 7
    warnings: list[str] = field(default_factory=default_benchmark_warnings)
    provenance: dict[str, Any] = field(default_factory=default_benchmark_provenance)
    production_ready: bool = False

    def validate(self) -> bool:
        if not self.case_id or not self.name:
            raise ValueError("benchmark case_id and name must be non-empty")
        if not self.category:
            raise ValueError("benchmark category must be non-empty")
        if not self.workload_type:
            raise ValueError("benchmark workload_type must be non-empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.production_ready:
            raise ValueError("Stage 10B benchmark cases must not claim production readiness")
        for key in ("cloud_access", "token_access", "hardware_access", "official_benchmark_claim"):
            if self.metadata.get(key) is True or self.provenance.get(key) is True:
                raise ValueError(f"benchmark cases must not set {key}=True")
        return True


def create_benchmark_case(
    name: str,
    category: str,
    workload: dict[str, Any] | None = None,
    expected: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    *,
    case_id: str | None = None,
    workload_type: str = "callable",
    runner: BenchmarkRunner | str | None = None,
    tags: list[str] | None = None,
    timeout_seconds: float = 10.0,
    deterministic: bool = True,
    seed: int | None = 7,
) -> BenchmarkCase:
    normalized_id = case_id or f"{category}.{name}".lower().replace(" ", "_").replace("/", "_")
    case = BenchmarkCase(
        case_id=normalized_id,
        name=name,
        category=category,
        workload_type=workload_type,
        runner=runner or "unsupported",
        workload=dict(workload or {}),
        expected=dict(expected or {}),
        metadata={
            "cloud_access": False,
            "token_access": False,
            "hardware_access": False,
            "official_benchmark_claim": False,
            "production_benchmark_parity_claim": False,
            **dict(metadata or {}),
        },
        tags=[str(tag) for tag in (tags or [])],
        timeout_seconds=float(timeout_seconds),
        deterministic=bool(deterministic),
        seed=seed,
    )
    case.validate()
    return case


def benchmark_case_to_dict(case: BenchmarkCase) -> dict[str, Any]:
    case.validate()
    runner_name = case.runner if isinstance(case.runner, str) else getattr(case.runner, "__name__", "callable")
    return {
        "case_id": case.case_id,
        "name": case.name,
        "category": case.category,
        "workload_type": case.workload_type,
        "tags": list(case.tags),
        "input_summary": dict(case.workload),
        "expected_summary": dict(case.expected),
        "runner": runner_name,
        "timeout_seconds": case.timeout_seconds,
        "deterministic": case.deterministic,
        "seed": case.seed,
        "warnings": list(case.warnings),
        "provenance": dict(case.provenance),
        "production_ready": case.production_ready,
        "metadata": dict(case.metadata),
    }


def benchmark_case_from_dict(data: dict[str, Any]) -> BenchmarkCase:
    return create_benchmark_case(
        name=str(data["name"]),
        category=str(data["category"]),
        workload=dict(data.get("input_summary", data.get("workload", {}))),
        expected=dict(data.get("expected_summary", data.get("expected", {}))),
        metadata=dict(data.get("metadata", {})),
        case_id=str(data["case_id"]),
        workload_type=str(data.get("workload_type", "callable")),
        runner=str(data.get("runner", "unsupported")),
        tags=[str(tag) for tag in data.get("tags", [])],
        timeout_seconds=float(data.get("timeout_seconds", 10.0)),
        deterministic=bool(data.get("deterministic", True)),
        seed=data.get("seed"),
    )


def validate_benchmark_case(case: BenchmarkCase) -> bool:
    return case.validate()
