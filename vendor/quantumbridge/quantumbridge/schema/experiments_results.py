# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Qiskit Experiments was copied.
"""Result schemas for educational offline Qiskit Experiments workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


@dataclass
class ExperimentsResult:
    """Serializable envelope for Stage 9I experiments workflows."""

    workflow: str
    mode: str
    capability_level: int
    upstream_package: str | None = None
    upstream_version: str | None = None
    quantumbridge_version: str | None = None
    production_ready: bool = False
    hardware_calibration: bool = False
    native_implementation: bool = False
    input_summary: dict[str, Any] = field(default_factory=dict)
    data: Any = None
    fit_parameters: dict[str, float] = field(default_factory=dict)
    time_points: list[float] = field(default_factory=list)
    expectation_values: dict[str, list[float]] = field(default_factory=dict)
    final_state: list[complex] | list[float] = field(default_factory=list)
    raw_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    unsupported_reason: str | None = None
    schema_version: str = "0.1"
    ecosystem: str = "quantumbridge_native_experiments"

    ECOSYSTEM: ClassVar[str | None] = None

    def __post_init__(self) -> None:
        if self.ECOSYSTEM is not None:
            self.ecosystem = self.ECOSYSTEM
        self.input_summary = dict(self.input_summary)
        self.fit_parameters = {
            str(key): float(value) for key, value in self.fit_parameters.items()
        }
        self.time_points = [float(value) for value in self.time_points]
        self.expectation_values = {
            str(key): [float(item) for item in values]
            for key, values in self.expectation_values.items()
        }
        self.metadata = dict(self.metadata)
        self.warnings = [str(value) for value in self.warnings]
        self.provenance = dict(self.provenance)
        self.validate()

    def validate(self) -> bool:
        if self.ecosystem not in {"qiskit_experiments", "quantumbridge_native_experiments"}:
            raise ValueError("ecosystem must be qiskit_experiments or quantumbridge_native_experiments")
        if self.schema_version not in {"0.1", "0.2"}:
            raise ValueError("schema_version must be supported")
        if self.mode not in {"native_minimal", "upstream_passthrough", "comparison"}:
            raise ValueError("mode must be native_minimal, upstream_passthrough, or comparison")
        if self.capability_level not in {0, 1, 2, 3, 4}:
            raise ValueError("capability_level must be between 0 and 4")
        if not self.workflow:
            raise ValueError("workflow must be non-empty")
        if self.production_ready:
            raise ValueError("Stage 9I experiments results must not claim production readiness")
        if self.hardware_calibration:
            raise ValueError("Stage 9I experiments results must not claim hardware calibration")
        if not isinstance(self.metadata, dict) or not isinstance(self.provenance, dict):
            raise TypeError("metadata and provenance must be dictionaries")
        if not isinstance(self.warnings, list):
            raise TypeError("warnings must be a list")
        return True

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["final_state"] = [_jsonable_complex(value) for value in self.final_state]
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentsResult":
        data = dict(payload)
        data["input_summary"] = dict(data.get("input_summary", {}))
        data["fit_parameters"] = dict(data.get("fit_parameters", {}))
        data["time_points"] = list(data.get("time_points", ()))
        data["expectation_values"] = dict(data.get("expectation_values", {}))
        data["metadata"] = dict(data.get("metadata", {}))
        data["warnings"] = [str(value) for value in data.get("warnings", ())]
        data["provenance"] = dict(data.get("provenance", {}))
        return cls(**data)


class RabiExperimentResult(ExperimentsResult):
    ECOSYSTEM = "quantumbridge_native_experiments"


class T1ExperimentResult(ExperimentsResult):
    ECOSYSTEM = "quantumbridge_native_experiments"


class RamseyExperimentResult(ExperimentsResult):
    ECOSYSTEM = "quantumbridge_native_experiments"


class CalibrationFitResult(ExperimentsResult):
    ECOSYSTEM = "quantumbridge_native_experiments"


class UpstreamExperimentsResult(ExperimentsResult):
    ECOSYSTEM = "qiskit_experiments"


class ExperimentsComparisonResult(ExperimentsResult):
    ECOSYSTEM = "quantumbridge_native_experiments"


def _jsonable_complex(value: Any) -> Any:
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value
