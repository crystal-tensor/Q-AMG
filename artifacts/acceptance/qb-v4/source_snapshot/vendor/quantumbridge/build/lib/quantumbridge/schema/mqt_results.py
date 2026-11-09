# This file is independently implemented for QuantumBridge SDK.
# No source code from MQT, IBM, or Qiskit was copied.
"""Serializable schemas for Stage 9J MQT compatibility workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


def _complex_to_dict(value: complex) -> dict[str, float]:
    item = complex(value)
    return {"real": float(item.real), "imag": float(item.imag)}


def _complex_from_payload(value: Any) -> complex:
    if isinstance(value, dict) and {"real", "imag"} <= set(value):
        return complex(float(value["real"]), float(value["imag"]))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return complex(float(value[0]), float(value[1]))
    return complex(value)


@dataclass
class MQTCompatibilityResult:
    """Envelope for clean-room MQT Core/DDSIM/QMAP compatibility slices."""

    workflow: str = "mqt_compatibility"
    project: str = "mqt"
    mode: str = "native_minimal"
    upstream_package: str | None = None
    upstream_version: str | None = None
    quantumbridge_version: str | None = None
    capability_level: int = 2
    production_ready: bool = False
    native_implementation: bool = False
    circuit_summary: dict[str, Any] = field(default_factory=dict)
    num_qubits: int | None = None
    operations: list[dict[str, Any]] = field(default_factory=list)
    measurements: list[dict[str, Any]] = field(default_factory=list)
    topology: dict[str, Any] | None = None
    initial_layout: dict[int, int] = field(default_factory=dict)
    final_layout: dict[int, int] = field(default_factory=dict)
    swap_count: int | None = None
    depth_estimate: int | None = None
    statevector: list[complex] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    probabilities: dict[str, float] = field(default_factory=dict)
    decision_diagram_metadata: dict[str, Any] = field(default_factory=dict)
    mapped_ir: dict[str, Any] | None = None
    comparison: dict[str, Any] = field(default_factory=dict)
    raw_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    unsupported_reason: str | None = None
    schema_version: str = "0.1"
    ecosystem: str = "quantumbridge_native_mqt_compat"

    ECOSYSTEM: ClassVar[str | None] = None

    def __post_init__(self) -> None:
        if self.ECOSYSTEM is not None:
            self.ecosystem = self.ECOSYSTEM
        self.statevector = [complex(value) for value in self.statevector]
        self.counts = {str(key): int(value) for key, value in self.counts.items()}
        self.probabilities = {str(key): float(value) for key, value in self.probabilities.items()}
        self.initial_layout = {int(key): int(value) for key, value in self.initial_layout.items()}
        self.final_layout = {int(key): int(value) for key, value in self.final_layout.items()}
        self.operations = [dict(item) for item in self.operations]
        self.measurements = [dict(item) for item in self.measurements]
        self.circuit_summary = dict(self.circuit_summary)
        self.decision_diagram_metadata = dict(self.decision_diagram_metadata)
        self.comparison = dict(self.comparison)
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
            raise ValueError("Stage 9J MQT compatibility results must not claim production readiness")
        if self.num_qubits is not None and self.num_qubits <= 0:
            raise ValueError("num_qubits must be positive when provided")
        if self.swap_count is not None and self.swap_count < 0:
            raise ValueError("swap_count cannot be negative")
        if self.depth_estimate is not None and self.depth_estimate < 0:
            raise ValueError("depth_estimate cannot be negative")
        if not isinstance(self.metadata, dict) or not isinstance(self.provenance, dict):
            raise TypeError("metadata and provenance must be dictionaries")
        if not isinstance(self.warnings, list):
            raise TypeError("warnings must be a list")
        return True

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["statevector"] = [_complex_to_dict(value) for value in self.statevector]
        payload["initial_layout"] = {str(key): value for key, value in self.initial_layout.items()}
        payload["final_layout"] = {str(key): value for key, value in self.final_layout.items()}
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MQTCompatibilityResult":
        data = dict(payload)
        data["statevector"] = [_complex_from_payload(value) for value in data.get("statevector", ())]
        data["counts"] = {str(key): int(value) for key, value in data.get("counts", {}).items()}
        data["probabilities"] = {
            str(key): float(value) for key, value in data.get("probabilities", {}).items()
        }
        data["initial_layout"] = {
            int(key): int(value) for key, value in data.get("initial_layout", {}).items()
        }
        data["final_layout"] = {
            int(key): int(value) for key, value in data.get("final_layout", {}).items()
        }
        data["operations"] = [dict(item) for item in data.get("operations", ())]
        data["measurements"] = [dict(item) for item in data.get("measurements", ())]
        data["circuit_summary"] = dict(data.get("circuit_summary", {}))
        data["decision_diagram_metadata"] = dict(data.get("decision_diagram_metadata", {}))
        data["comparison"] = dict(data.get("comparison", {}))
        data["metadata"] = dict(data.get("metadata", {}))
        data["warnings"] = [str(value) for value in data.get("warnings", ())]
        data["provenance"] = dict(data.get("provenance", {}))
        return cls(**data)


class MQTCoreLikeCircuitResult(MQTCompatibilityResult):
    ECOSYSTEM = "quantumbridge_native_mqt_compat"


class DDSIMLikeSimulationResult(MQTCompatibilityResult):
    ECOSYSTEM = "quantumbridge_native_mqt_compat"


class DecisionDiagramMetadataResult(MQTCompatibilityResult):
    ECOSYSTEM = "quantumbridge_native_mqt_compat"


class QMAPLikeMappingResult(MQTCompatibilityResult):
    ECOSYSTEM = "quantumbridge_native_mqt_compat"


class RoutingResult(MQTCompatibilityResult):
    ECOSYSTEM = "quantumbridge_native_mqt_compat"


class UpstreamMQTResult(MQTCompatibilityResult):
    ECOSYSTEM = "mqt"


class MQTComparisonResult(MQTCompatibilityResult):
    ECOSYSTEM = "mqt_comparison"
