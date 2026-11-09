# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit, PennyLane, or PennyLane-Qiskit was copied.
"""Result schemas for Stage 9H PennyLane-Qiskit bridge workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


@dataclass
class PennyLaneQiskitBridgeResult:
    """Serializable envelope for clean-room PennyLane-Qiskit bridge workflows."""

    workflow: str
    mode: str
    source_ecosystem: str
    target_ecosystem: str
    capability_level: int
    upstream_package: str | None = None
    upstream_version: str | None = None
    quantumbridge_version: str | None = None
    production_ready: bool = False
    native_implementation: bool = False
    circuit_summary: dict[str, Any] = field(default_factory=dict)
    operation_count: int | None = None
    num_qubits: int | None = None
    shots: int | None = None
    seed: int | None = None
    statevector_probabilities: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    converted_ir: dict[str, Any] = field(default_factory=dict)
    converted_target: dict[str, Any] = field(default_factory=dict)
    equivalence_status: str | None = None
    tolerance: float | None = None
    raw_type: str | None = None
    data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    unsupported_reason: str | None = None
    schema_version: str = "0.1"
    ecosystem: str = "pennylane_qiskit_bridge"

    ECOSYSTEM: ClassVar[str | None] = None

    def __post_init__(self) -> None:
        if self.ECOSYSTEM is not None:
            self.ecosystem = self.ECOSYSTEM
        self.circuit_summary = dict(self.circuit_summary)
        self.statevector_probabilities = {
            str(key): float(value)
            for key, value in self.statevector_probabilities.items()
        }
        self.counts = {str(key): int(value) for key, value in self.counts.items()}
        self.converted_ir = dict(self.converted_ir)
        self.converted_target = dict(self.converted_target)
        if self.tolerance is not None:
            self.tolerance = float(self.tolerance)
        self.metadata = dict(self.metadata)
        self.warnings = [str(value) for value in self.warnings]
        self.provenance = dict(self.provenance)
        self.validate()

    def validate(self) -> bool:
        if self.ecosystem != "pennylane_qiskit_bridge":
            raise ValueError("ecosystem must be pennylane_qiskit_bridge")
        if self.schema_version not in {"0.1", "0.2"}:
            raise ValueError("schema_version must be supported")
        if self.mode not in {"native_bridge", "upstream_passthrough", "comparison"}:
            raise ValueError("mode must be native_bridge, upstream_passthrough, or comparison")
        if self.capability_level not in {0, 1, 2, 3, 4}:
            raise ValueError("capability_level must be between 0 and 4")
        if not self.workflow:
            raise ValueError("workflow must be non-empty")
        if not self.source_ecosystem or not self.target_ecosystem:
            raise ValueError("source_ecosystem and target_ecosystem must be non-empty")
        if self.production_ready:
            raise ValueError("Stage 9H bridge results must not claim production readiness")
        if self.num_qubits is not None and self.num_qubits <= 0:
            raise ValueError("num_qubits must be positive when provided")
        if self.operation_count is not None and self.operation_count < 0:
            raise ValueError("operation_count cannot be negative")
        if self.shots is not None and self.shots <= 0:
            raise ValueError("shots must be positive when provided")
        if self.tolerance is not None and self.tolerance < 0:
            raise ValueError("tolerance cannot be negative")
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
    def from_dict(cls, payload: dict[str, Any]) -> "PennyLaneQiskitBridgeResult":
        data = dict(payload)
        data["circuit_summary"] = dict(data.get("circuit_summary", {}))
        data["statevector_probabilities"] = {
            str(key): float(value)
            for key, value in data.get("statevector_probabilities", {}).items()
        }
        data["counts"] = {
            str(key): int(value) for key, value in data.get("counts", {}).items()
        }
        data["converted_ir"] = dict(data.get("converted_ir", {}))
        data["converted_target"] = dict(data.get("converted_target", {}))
        if data.get("tolerance") is not None:
            data["tolerance"] = float(data["tolerance"])
        data["metadata"] = dict(data.get("metadata", {}))
        data["warnings"] = [str(value) for value in data.get("warnings", ())]
        data["provenance"] = dict(data.get("provenance", {}))
        return cls(**data)


class QiskitToPennyLaneResult(PennyLaneQiskitBridgeResult):
    ECOSYSTEM = "pennylane_qiskit_bridge"


class PennyLaneToQiskitResult(PennyLaneQiskitBridgeResult):
    ECOSYSTEM = "pennylane_qiskit_bridge"


class BridgeIRResult(PennyLaneQiskitBridgeResult):
    ECOSYSTEM = "pennylane_qiskit_bridge"


class BridgeExecutionResult(PennyLaneQiskitBridgeResult):
    ECOSYSTEM = "pennylane_qiskit_bridge"


class BridgeEquivalenceResult(PennyLaneQiskitBridgeResult):
    ECOSYSTEM = "pennylane_qiskit_bridge"


class UpstreamPennyLaneQiskitResult(PennyLaneQiskitBridgeResult):
    ECOSYSTEM = "pennylane_qiskit_bridge"
