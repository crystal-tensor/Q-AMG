# This file is independently implemented for QuantumBridge SDK.
# No source code from Mitiq, Qiskit, or Qiskit Aer was copied.
"""Error-mitigation result schemas for Stage 9G executable workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


@dataclass
class ErrorMitigationResult:
    """Serializable envelope for educational error-mitigation workflows."""

    workflow: str
    mode: str
    capability_level: int
    upstream_package: str | None = None
    upstream_version: str | None = None
    quantumbridge_version: str | None = None
    production_ready: bool = False
    native_implementation: bool = False
    circuit_summary: dict[str, Any] = field(default_factory=dict)
    observable: str | None = None
    shots: int | None = None
    seed: int | None = None
    noise_model: dict[str, Any] | None = None
    noise_scales: list[float] = field(default_factory=list)
    noisy_expectation_values: list[float] = field(default_factory=list)
    mitigated_expectation_value: float | None = None
    ideal_expectation_value: float | None = None
    raw_counts: dict[str, int] = field(default_factory=dict)
    noisy_counts: dict[str, int] = field(default_factory=dict)
    mitigated_probabilities: dict[str, float] = field(default_factory=dict)
    calibration_matrix: list[list[float]] = field(default_factory=list)
    raw_type: str | None = None
    data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    unsupported_reason: str | None = None
    schema_version: str = "0.1"
    ecosystem: str = "quantumbridge_native_error_mitigation"

    ECOSYSTEM: ClassVar[str | None] = None

    def __post_init__(self) -> None:
        if self.ECOSYSTEM is not None:
            self.ecosystem = self.ECOSYSTEM
        self.circuit_summary = dict(self.circuit_summary)
        self.noise_scales = [float(value) for value in self.noise_scales]
        self.noisy_expectation_values = [
            float(value) for value in self.noisy_expectation_values
        ]
        if self.mitigated_expectation_value is not None:
            self.mitigated_expectation_value = float(self.mitigated_expectation_value)
        if self.ideal_expectation_value is not None:
            self.ideal_expectation_value = float(self.ideal_expectation_value)
        self.raw_counts = {
            str(key): int(value) for key, value in self.raw_counts.items()
        }
        self.noisy_counts = {
            str(key): int(value) for key, value in self.noisy_counts.items()
        }
        self.mitigated_probabilities = {
            str(key): float(value)
            for key, value in self.mitigated_probabilities.items()
        }
        self.calibration_matrix = [
            [float(value) for value in row] for row in self.calibration_matrix
        ]
        self.metadata = dict(self.metadata)
        self.warnings = [str(value) for value in self.warnings]
        self.provenance = dict(self.provenance)
        self.validate()

    def validate(self) -> bool:
        if self.mode not in {"native_minimal", "upstream_passthrough", "comparison"}:
            raise ValueError("mode must be native_minimal, upstream_passthrough, or comparison")
        if self.capability_level not in {0, 1, 2, 3, 4}:
            raise ValueError("capability_level must be between 0 and 4")
        if not self.workflow:
            raise ValueError("workflow must be non-empty")
        if self.production_ready:
            raise ValueError("Stage 9G results must not claim production readiness")
        if self.shots is not None and self.shots <= 0:
            raise ValueError("shots must be positive when provided")
        if len(self.noise_scales) != len(self.noisy_expectation_values):
            if self.noise_scales or self.noisy_expectation_values:
                raise ValueError("noise_scales and noisy_expectation_values must align")
        if not isinstance(self.metadata, dict) or not isinstance(self.provenance, dict):
            raise TypeError("metadata and provenance must be dictionaries")
        if not isinstance(self.warnings, list):
            raise TypeError("warnings must be a list")
        if self.noise_model is not None and not isinstance(self.noise_model, dict):
            raise TypeError("noise_model must be a dictionary when provided")
        total = sum(self.mitigated_probabilities.values())
        if self.mitigated_probabilities and abs(total - 1.0) > 1e-8:
            raise ValueError("mitigated_probabilities must be normalized")
        return True

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ErrorMitigationResult":
        data = dict(payload)
        data["circuit_summary"] = dict(data.get("circuit_summary", {}))
        data["noise_scales"] = [float(value) for value in data.get("noise_scales", ())]
        data["noisy_expectation_values"] = [
            float(value) for value in data.get("noisy_expectation_values", ())
        ]
        if data.get("mitigated_expectation_value") is not None:
            data["mitigated_expectation_value"] = float(data["mitigated_expectation_value"])
        if data.get("ideal_expectation_value") is not None:
            data["ideal_expectation_value"] = float(data["ideal_expectation_value"])
        data["raw_counts"] = {
            str(key): int(value) for key, value in data.get("raw_counts", {}).items()
        }
        data["noisy_counts"] = {
            str(key): int(value) for key, value in data.get("noisy_counts", {}).items()
        }
        data["mitigated_probabilities"] = {
            str(key): float(value)
            for key, value in data.get("mitigated_probabilities", {}).items()
        }
        data["calibration_matrix"] = [
            [float(value) for value in row]
            for row in data.get("calibration_matrix", ())
        ]
        data["metadata"] = dict(data.get("metadata", {}))
        data["warnings"] = [str(value) for value in data.get("warnings", ())]
        data["provenance"] = dict(data.get("provenance", {}))
        return cls(**data)


class ZNEResult(ErrorMitigationResult):
    ECOSYSTEM = "quantumbridge_native_error_mitigation"


class ReadoutMitigationResult(ErrorMitigationResult):
    ECOSYSTEM = "quantumbridge_native_error_mitigation"


class UpstreamMitiqResult(ErrorMitigationResult):
    ECOSYSTEM = "mitiq"


class ErrorMitigationComparisonResult(ErrorMitigationResult):
    ECOSYSTEM = "error_mitigation_comparison"
