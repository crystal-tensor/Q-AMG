# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Machine Learning was copied.
"""Machine-learning result schemas for executable compatibility slices."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


@dataclass
class MachineLearningResult:
    """Serializable result envelope for Stage 9E QML workflows."""

    workflow: str
    mode: str
    capability_level: int
    upstream_package: str | None = None
    upstream_version: str | None = None
    quantumbridge_version: str | None = None
    production_ready: bool = False
    native_implementation: bool = False
    dataset_summary: dict[str, Any] = field(default_factory=dict)
    feature_map: dict[str, Any] = field(default_factory=dict)
    kernel_matrix: list[list[float]] = field(default_factory=list)
    model_summary: dict[str, Any] = field(default_factory=dict)
    weights: list[float] = field(default_factory=list)
    predictions: list[int] = field(default_factory=list)
    accuracy: float | None = None
    expectation: float | None = None
    raw_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    unsupported_reason: str | None = None
    schema_version: str = "0.1"
    ecosystem: str = "quantumbridge_native_qml"

    ECOSYSTEM: ClassVar[str | None] = None

    def __post_init__(self) -> None:
        if self.ECOSYSTEM is not None:
            self.ecosystem = self.ECOSYSTEM
        self.validate()

    def validate(self) -> bool:
        if self.mode not in {"native_minimal", "upstream_passthrough", "comparison"}:
            raise ValueError("mode must be native_minimal, upstream_passthrough, or comparison")
        if self.capability_level not in {0, 1, 2, 3, 4}:
            raise ValueError("capability_level must be between 0 and 4")
        if not self.workflow:
            raise ValueError("workflow must be non-empty")
        if self.production_ready:
            raise ValueError("Stage 9E QML results must not claim production readiness")
        if not isinstance(self.dataset_summary, dict):
            raise TypeError("dataset_summary must be a dictionary")
        if not isinstance(self.feature_map, dict):
            raise TypeError("feature_map must be a dictionary")
        if not isinstance(self.kernel_matrix, list):
            raise TypeError("kernel_matrix must be a list")
        if not isinstance(self.model_summary, dict):
            raise TypeError("model_summary must be a dictionary")
        if not isinstance(self.weights, list):
            raise TypeError("weights must be a list")
        if not isinstance(self.predictions, list):
            raise TypeError("predictions must be a list")
        if not isinstance(self.metadata, dict) or not isinstance(self.provenance, dict):
            raise TypeError("metadata and provenance must be dictionaries")
        if not isinstance(self.warnings, list):
            raise TypeError("warnings must be a list")
        if self.accuracy is not None and not 0.0 <= float(self.accuracy) <= 1.0:
            raise ValueError("accuracy must be between 0 and 1")
        return True

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MachineLearningResult":
        data = dict(payload)
        data["dataset_summary"] = dict(data.get("dataset_summary", {}))
        data["feature_map"] = dict(data.get("feature_map", {}))
        data["kernel_matrix"] = [
            [float(value) for value in row] for row in data.get("kernel_matrix", ())
        ]
        data["model_summary"] = dict(data.get("model_summary", {}))
        data["weights"] = [float(value) for value in data.get("weights", ())]
        data["predictions"] = [int(value) for value in data.get("predictions", ())]
        data["metadata"] = dict(data.get("metadata", {}))
        data["warnings"] = [str(value) for value in data.get("warnings", ())]
        data["provenance"] = dict(data.get("provenance", {}))
        if data.get("accuracy") is not None:
            data["accuracy"] = float(data["accuracy"])
        if data.get("expectation") is not None:
            data["expectation"] = float(data["expectation"])
        return cls(**data)


class QuantumKernelResult(MachineLearningResult):
    ECOSYSTEM = "quantumbridge_native_qml"


class KernelClassifierResult(MachineLearningResult):
    ECOSYSTEM = "quantumbridge_native_qml"


class QNNForwardResult(MachineLearningResult):
    ECOSYSTEM = "quantumbridge_native_qml"


class QNNClassifierResult(MachineLearningResult):
    ECOSYSTEM = "quantumbridge_native_qml"


class UpstreamMLResult(MachineLearningResult):
    ECOSYSTEM = "qiskit_machine_learning"


class MLComparisonResult(MachineLearningResult):
    ECOSYSTEM = "qiskit_machine_learning_comparison"
