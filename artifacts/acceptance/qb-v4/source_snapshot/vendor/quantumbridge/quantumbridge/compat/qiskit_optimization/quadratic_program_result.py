# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Optimization was copied.
"""Serializable results for native QuadraticProgram execution."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class NativeOptimizationResult:
    """Result produced by the QuantumBridge native educational optimizer."""

    problem_name: str
    objective_sense: str
    objective_value: float | None
    assignment: dict[str, int]
    feasible: bool
    method: str
    path: str
    samples: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    feasible_assignments: tuple[dict[str, int], ...] = field(default_factory=tuple)
    constraints: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    qubo_metadata: dict[str, Any] = field(default_factory=dict)
    ising_metadata: dict[str, Any] = field(default_factory=dict)
    upstream_package: str | None = None
    upstream_version: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    unsupported_reason: str | None = None
    schema_version: str = "0.1"

    def validate(self) -> bool:
        if self.objective_sense not in {"minimize", "maximize"}:
            raise ValueError("objective_sense must be minimize or maximize")
        if not isinstance(self.assignment, dict):
            raise TypeError("assignment must be a dictionary")
        if not isinstance(self.feasible, bool):
            raise TypeError("feasible must be a bool")
        if self.feasible and self.objective_value is None:
            raise ValueError("objective_value is required for feasible results")
        if not isinstance(self.qubo_metadata, dict) or not isinstance(self.ising_metadata, dict):
            raise TypeError("qubo_metadata and ising_metadata must be dictionaries")
        return True

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["samples"] = list(self.samples)
        payload["feasible_assignments"] = list(self.feasible_assignments)
        payload["constraints"] = list(self.constraints)
        payload["warnings"] = list(self.warnings)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NativeOptimizationResult":
        data = dict(payload)
        data["assignment"] = {str(key): int(value) for key, value in data.get("assignment", {}).items()}
        data["samples"] = tuple(dict(row) for row in data.get("samples", ()))
        data["feasible_assignments"] = tuple(
            {str(key): int(value) for key, value in row.items()}
            for row in data.get("feasible_assignments", ())
        )
        data["constraints"] = tuple(dict(row) for row in data.get("constraints", ()))
        data["warnings"] = tuple(str(value) for value in data.get("warnings", ()))
        data["provenance"] = dict(data.get("provenance", {}))
        data["metadata"] = dict(data.get("metadata", {}))
        data["qubo_metadata"] = dict(data.get("qubo_metadata", {}))
        data["ising_metadata"] = dict(data.get("ising_metadata", {}))
        return cls(**data)
