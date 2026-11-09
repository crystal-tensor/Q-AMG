# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Optimization was copied.
"""Optimization result schemas for executable compatibility adapters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


@dataclass
class OptimizationResult:
    """Serializable optimization result envelope."""

    mode: str
    capability_level: int
    objective_sense: str
    objective_value: float | None
    assignment: dict[str, int] = field(default_factory=dict)
    feasible: bool = False
    constraints: list[dict[str, Any]] = field(default_factory=list)
    qubo_metadata: dict[str, Any] = field(default_factory=dict)
    ising_metadata: dict[str, Any] = field(default_factory=dict)
    raw_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    unsupported_reason: str | None = None
    upstream_package: str | None = None
    upstream_version: str | None = None
    quantumbridge_version: str | None = None
    production_ready: bool = False
    native_implementation: bool = False
    num_variables: int = 0
    schema_version: str = "0.1"
    ecosystem: str = "quantumbridge_native_optimization"

    ECOSYSTEM: ClassVar[str | None] = None

    def __post_init__(self) -> None:
        if self.ECOSYSTEM is not None:
            self.ecosystem = self.ECOSYSTEM
        self.validate()

    def validate(self) -> bool:
        if self.capability_level not in {0, 1, 2, 3, 4}:
            raise ValueError("capability_level must be between 0 and 4")
        if self.objective_sense not in {"minimize", "maximize", "unknown"}:
            raise ValueError("objective_sense must be minimize, maximize, or unknown")
        if not isinstance(self.assignment, dict):
            raise TypeError("assignment must be a dictionary")
        if not isinstance(self.feasible, bool):
            raise TypeError("feasible must be a bool")
        if self.feasible and self.objective_value is None:
            raise ValueError("feasible results require objective_value")
        if not isinstance(self.constraints, list):
            raise TypeError("constraints must be a list")
        if not isinstance(self.qubo_metadata, dict) or not isinstance(self.ising_metadata, dict):
            raise TypeError("qubo_metadata and ising_metadata must be dictionaries")
        if not isinstance(self.warnings, list):
            raise TypeError("warnings must be a list")
        if not isinstance(self.provenance, dict) or not isinstance(self.metadata, dict):
            raise TypeError("provenance and metadata must be dictionaries")
        return True

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OptimizationResult":
        data = dict(payload)
        data["assignment"] = {str(key): int(value) for key, value in data.get("assignment", {}).items()}
        data["constraints"] = [dict(row) for row in data.get("constraints", ())]
        data["warnings"] = [str(value) for value in data.get("warnings", ())]
        data["provenance"] = dict(data.get("provenance", {}))
        data["metadata"] = dict(data.get("metadata", {}))
        data["qubo_metadata"] = dict(data.get("qubo_metadata", {}))
        data["ising_metadata"] = dict(data.get("ising_metadata", {}))
        return cls(**data)


class QuadraticProgramResult(OptimizationResult):
    ECOSYSTEM = "quantumbridge_native_optimization"


class BruteForceOptimizationResult(OptimizationResult):
    ECOSYSTEM = "quantumbridge_native_optimization"


class QUBOConversionResult(OptimizationResult):
    ECOSYSTEM = "quantumbridge_native_optimization"


class IsingConversionResult(OptimizationResult):
    ECOSYSTEM = "quantumbridge_native_optimization"


class UpstreamOptimizationResult(OptimizationResult):
    ECOSYSTEM = "qiskit_optimization"


class OptimizationComparisonResult(OptimizationResult):
    ECOSYSTEM = "qiskit_optimization_comparison"
