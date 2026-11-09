# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Aer was copied.
"""Qiskit Aer-compatible simulator result schemas."""

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
class AerResult:
    """Serializable envelope for Stage 9F simulator workflows."""

    workflow: str = "aer_result"
    backend: str = "qiskit_aer"
    mode: str = "native_minimal"
    capability_level: int = 2
    upstream_package: str | None = None
    upstream_version: str | None = None
    quantumbridge_version: str | None = None
    production_ready: bool = False
    native_implementation: bool = False
    num_qubits: int | None = None
    shots: int | None = None
    seed: int | None = None
    final_statevector: list[complex] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    probabilities: dict[str, float] = field(default_factory=dict)
    noise_model: dict[str, Any] | None = None
    raw_type: str | None = None
    data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    unsupported_reason: str | None = None
    schema_version: str = "0.1"
    ecosystem: str = "quantumbridge_native_simulator"

    ECOSYSTEM: ClassVar[str | None] = None

    def __post_init__(self) -> None:
        if self.ECOSYSTEM is not None:
            self.ecosystem = self.ECOSYSTEM
        self.final_statevector = [complex(value) for value in self.final_statevector]
        self.counts = {str(key): int(value) for key, value in self.counts.items()}
        self.probabilities = {
            str(key): float(value) for key, value in self.probabilities.items()
        }
        self.metadata = dict(self.metadata)
        self.warnings = [str(value) for value in self.warnings]
        self.provenance = dict(self.provenance)
        self.validate()

    def validate(self) -> bool:
        if self.capability_level not in {0, 1, 2, 3, 4}:
            raise ValueError("capability_level must be between 0 and 4")
        if self.mode not in {"native_minimal", "upstream_passthrough", "comparison", "Adapter"}:
            raise ValueError("mode must be native_minimal, upstream_passthrough, comparison, or Adapter")
        if not self.workflow:
            raise ValueError("workflow must be non-empty")
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if self.production_ready:
            raise ValueError("Stage 9F Aer results must not claim production readiness")
        if self.num_qubits is not None and self.num_qubits <= 0:
            raise ValueError("num_qubits must be positive when provided")
        if self.shots is not None and self.shots <= 0:
            raise ValueError("shots must be positive when provided")
        if not isinstance(self.metadata, dict) or not isinstance(self.provenance, dict):
            raise TypeError("metadata and provenance must be dictionaries")
        if not isinstance(self.warnings, list):
            raise TypeError("warnings must be a list")
        if self.noise_model is not None and not isinstance(self.noise_model, dict):
            raise TypeError("noise_model must be a dictionary when provided")
        return True

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["final_statevector"] = [
            _complex_to_dict(value) for value in self.final_statevector
        ]
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AerResult":
        data = dict(payload)
        data["final_statevector"] = [
            _complex_from_payload(value) for value in data.get("final_statevector", ())
        ]
        data["counts"] = {str(key): int(value) for key, value in data.get("counts", {}).items()}
        data["probabilities"] = {
            str(key): float(value) for key, value in data.get("probabilities", {}).items()
        }
        data["metadata"] = dict(data.get("metadata", {}))
        data["warnings"] = [str(value) for value in data.get("warnings", ())]
        data["provenance"] = dict(data.get("provenance", {}))
        return cls(**data)


class StatevectorSimulationResult(AerResult):
    ECOSYSTEM = "quantumbridge_native_simulator"


@dataclass
class DensityMatrixSimulationResult(AerResult):
    final_density_matrix: list[list[complex]] = field(default_factory=list)

    def __post_init__(self):
        self.final_density_matrix = [[_complex_from_payload(value) for value in row] for row in self.final_density_matrix]
        super().__post_init__()
        if self.final_density_matrix:
            dimension = len(self.final_density_matrix)
            if any(len(row) != dimension for row in self.final_density_matrix):
                raise ValueError("QuantumBridge density result matrix must be square.")
            if self.num_qubits is not None and dimension != 1 << self.num_qubits:
                raise ValueError("QuantumBridge density result width must match num_qubits.")

    def to_dict(self):
        result = super().to_dict()
        result["final_density_matrix"] = [[_complex_to_dict(value) for value in row] for row in self.final_density_matrix]
        return result


class QasmSimulationResult(AerResult):
    ECOSYSTEM = "quantumbridge_native_simulator"


class NoisySimulationResult(AerResult):
    ECOSYSTEM = "quantumbridge_native_simulator"


class UpstreamAerResult(AerResult):
    ECOSYSTEM = "qiskit_aer"


class SimulatorComparisonResult(AerResult):
    ECOSYSTEM = "qiskit_aer_comparison"
