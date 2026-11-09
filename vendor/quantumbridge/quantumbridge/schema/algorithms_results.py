# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Algorithms was copied.
"""Algorithm result schemas for executable compatibility adapters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


@dataclass
class AlgorithmsResult:
    """Serializable algorithm result envelope for Stage 9C adapters."""

    algorithm: str
    mode: str
    capability_level: int
    problem_type: str
    eigenvalue: float | None = None
    optimal_parameters: list[float] = field(default_factory=list)
    bitstring: str | None = None
    objective_value: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
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
    schema_version: str = "0.1"
    ecosystem: str = "quantumbridge_native_algorithms"

    ECOSYSTEM: ClassVar[str | None] = None

    def __post_init__(self) -> None:
        if self.ECOSYSTEM is not None:
            self.ecosystem = self.ECOSYSTEM
        self.validate()

    def validate(self) -> bool:
        if self.capability_level not in {0, 1, 2, 3, 4}:
            raise ValueError("capability_level must be between 0 and 4")
        if self.mode not in {"native_minimal", "upstream_passthrough", "comparison"}:
            raise ValueError("mode must be native_minimal, upstream_passthrough, or comparison")
        if not self.algorithm:
            raise ValueError("algorithm must be non-empty")
        if not isinstance(self.optimal_parameters, list):
            raise TypeError("optimal_parameters must be a list")
        if not isinstance(self.probabilities, dict):
            raise TypeError("probabilities must be a dictionary")
        if not isinstance(self.input_summary, dict) or not isinstance(self.output_summary, dict):
            raise TypeError("input_summary and output_summary must be dictionaries")
        if not isinstance(self.metadata, dict) or not isinstance(self.provenance, dict):
            raise TypeError("metadata and provenance must be dictionaries")
        if not isinstance(self.warnings, list):
            raise TypeError("warnings must be a list")
        return True

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AlgorithmsResult":
        data = dict(payload)
        data["optimal_parameters"] = [float(value) for value in data.get("optimal_parameters", ())]
        data["probabilities"] = {
            str(key): float(value) for key, value in data.get("probabilities", {}).items()
        }
        data["input_summary"] = dict(data.get("input_summary", {}))
        data["output_summary"] = dict(data.get("output_summary", {}))
        data["metadata"] = dict(data.get("metadata", {}))
        data["warnings"] = [str(value) for value in data.get("warnings", ())]
        data["provenance"] = dict(data.get("provenance", {}))
        return cls(**data)


class NativeAlgorithmResult(AlgorithmsResult):
    ECOSYSTEM = "quantumbridge_native_algorithms"


class VQEResult(AlgorithmsResult):
    ECOSYSTEM = "quantumbridge_native_algorithms"


class QAOAResult(AlgorithmsResult):
    ECOSYSTEM = "quantumbridge_native_algorithms"


class GroverResult(AlgorithmsResult):
    ECOSYSTEM = "quantumbridge_native_algorithms"


class EigensolverResult(AlgorithmsResult):
    ECOSYSTEM = "quantumbridge_native_algorithms"


class UpstreamAlgorithmResult(AlgorithmsResult):
    ECOSYSTEM = "qiskit_algorithms"


class AlgorithmComparisonResult(AlgorithmsResult):
    ECOSYSTEM = "qiskit_algorithms_comparison"
