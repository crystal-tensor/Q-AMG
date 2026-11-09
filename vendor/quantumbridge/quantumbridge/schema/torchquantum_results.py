# This file is independently implemented for QuantumBridge SDK.
# No source code from TorchQuantum or PyTorch was copied.
"""Serializable schemas for Stage 9K TorchQuantum-style QML workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


@dataclass
class TorchQuantumCompatibilityResult:
    workflow: str = "torchquantum_compatibility"
    mode: str = "native_minimal"
    upstream_package: str | None = None
    upstream_version: str | None = None
    quantumbridge_version: str | None = None
    capability_level: int = 2
    production_ready: bool = False
    native_implementation: bool = False
    tensor_backend: str | None = None
    dataset_summary: dict[str, Any] = field(default_factory=dict)
    num_qubits: int | None = None
    feature_dim: int | None = None
    batch_size: int | None = None
    weights: list[float] = field(default_factory=list)
    forward_outputs: list[float] = field(default_factory=list)
    probabilities: list[dict[str, float]] = field(default_factory=list)
    predictions: list[int] = field(default_factory=list)
    accuracy: float | None = None
    loss: float | None = None
    training_trace: list[dict[str, Any]] = field(default_factory=list)
    circuit_ir: dict[str, Any] = field(default_factory=dict)
    raw_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    unsupported_reason: str | None = None
    schema_version: str = "0.1"
    ecosystem: str = "quantumbridge_native_torchquantum_compat"

    ECOSYSTEM: ClassVar[str | None] = None

    def __post_init__(self) -> None:
        if self.ECOSYSTEM is not None:
            self.ecosystem = self.ECOSYSTEM
        self.dataset_summary = dict(self.dataset_summary)
        self.weights = [float(value) for value in self.weights]
        self.forward_outputs = [float(value) for value in self.forward_outputs]
        self.probabilities = [
            {str(key): float(value) for key, value in item.items()} for item in self.probabilities
        ]
        self.predictions = [int(value) for value in self.predictions]
        self.training_trace = [dict(item) for item in self.training_trace]
        self.circuit_ir = dict(self.circuit_ir)
        self.metadata = dict(self.metadata)
        self.warnings = [str(value) for value in self.warnings]
        self.provenance = dict(self.provenance)
        self.validate()

    def validate(self) -> bool:
        if not self.workflow:
            raise ValueError("workflow must be non-empty")
        if self.mode not in {"native_minimal", "torch_optional", "upstream_passthrough", "comparison"}:
            raise ValueError("mode must be native_minimal, torch_optional, upstream_passthrough, or comparison")
        if self.production_ready:
            raise ValueError("Stage 9K TorchQuantum-style results must not claim production readiness")
        if self.capability_level not in {0, 1, 2, 3, 4}:
            raise ValueError("capability_level must be between 0 and 4")
        if self.num_qubits is not None and self.num_qubits not in {1, 2, 3}:
            raise ValueError("num_qubits must be 1, 2, or 3 when provided")
        if self.feature_dim is not None and self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive when provided")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive when provided")
        return True

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TorchQuantumCompatibilityResult":
        return cls(**dict(payload))


class TensorAdapterResult(TorchQuantumCompatibilityResult):
    pass


class QuantumLayerResult(TorchQuantumCompatibilityResult):
    pass


class BatchForwardResult(TorchQuantumCompatibilityResult):
    pass


class TorchQuantumTrainingResult(TorchQuantumCompatibilityResult):
    pass


class TorchQuantumClassifierResult(TorchQuantumCompatibilityResult):
    pass


class UpstreamTorchQuantumResult(TorchQuantumCompatibilityResult):
    ECOSYSTEM = "torchquantum"


class TorchQuantumComparisonResult(TorchQuantumCompatibilityResult):
    ECOSYSTEM = "torchquantum_comparison"
