# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Benchpress was copied.
"""Serializable schemas for Stage 10B benchmark compatibility workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


@dataclass
class BenchmarkResult:
    """Base envelope for clean-room QuantumBridge benchmark results."""

    workflow: str = "benchmarking"
    mode: str = "native_minimal"
    upstream_package: str | None = None
    upstream_version: str | None = None
    quantumbridge_version: str | None = None
    capability_level: int = 2
    production_ready: bool = False
    native_implementation: bool = True
    case_id: str | None = None
    case_name: str | None = None
    category: str | None = None
    tags: list[str] = field(default_factory=list)
    status: str = "passed"
    elapsed_seconds: float = 0.0
    passed: bool = True
    skipped: bool = False
    unsupported: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    expected_summary: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    raw_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    unsupported_reason: str | None = None
    schema_version: str = "0.1"
    ecosystem: str = "quantumbridge_native_benchmarking"

    ECOSYSTEM: ClassVar[str | None] = None

    def __post_init__(self) -> None:
        if self.ECOSYSTEM is not None:
            self.ecosystem = self.ECOSYSTEM
        self.tags = [str(value) for value in self.tags]
        self.metrics = dict(self.metrics)
        self.output_summary = dict(self.output_summary)
        self.expected_summary = dict(self.expected_summary)
        self.report = dict(self.report)
        self.metadata = dict(self.metadata)
        self.warnings = [str(value) for value in self.warnings]
        self.provenance = dict(self.provenance)
        self.validate()

    def validate(self) -> bool:
        if not self.workflow:
            raise ValueError("workflow must be non-empty")
        if self.mode not in {"native_minimal", "upstream_passthrough", "comparison"}:
            raise ValueError("mode must be native_minimal, upstream_passthrough, or comparison")
        if self.capability_level not in {0, 1, 2, 3, 4}:
            raise ValueError("capability_level must be between 0 and 4")
        if self.production_ready:
            raise ValueError("Stage 10B benchmarks must not claim production readiness")
        if self.status not in {"passed", "failed", "skipped", "unsupported"}:
            raise ValueError("status must be passed, failed, skipped, or unsupported")
        for key in (
            "cloud_access",
            "token_read",
            "token_access",
            "hardware_access",
            "official_benchmark_claim",
            "official_endorsement_claim",
            "production_benchmark_parity_claim",
        ):
            if self.metadata.get(key) is True or self.provenance.get(key) is True:
                raise ValueError(f"Stage 10B benchmark results must not set {key}=True")
        return True

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkResult":
        return cls(**dict(payload))


class BenchmarkCaseResult(BenchmarkResult):
    pass


class BenchmarkSuiteResult(BenchmarkResult):
    pass


class BenchmarkComparisonResult(BenchmarkResult):
    ECOSYSTEM = "quantumbridge_benchmarking_comparison"


class UpstreamBenchpressResult(BenchmarkResult):
    ECOSYSTEM = "benchpress"
