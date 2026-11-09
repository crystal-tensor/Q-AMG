# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit, IBM, Aer, Nature, Finance, ML, Optimization, Experiments, Metal, or Addons was copied.
"""Serializable Qiskit ecosystem result schemas."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


@dataclass
class QiskitResult:
    upstream_package: str
    upstream_version: str | None
    capability_level: int
    mode: str
    raw_type: str
    data: Any
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    unsupported_reason: str | None = None
    advisory: bool = False
    offline_only: bool = False
    quantumbridge_version: str | None = None
    schema_version: str = "0.1"
    ecosystem: str = "qiskit"

    ECOSYSTEM: ClassVar[str | None] = None

    def __post_init__(self) -> None:
        if self.ECOSYSTEM is not None:
            self.ecosystem = self.ECOSYSTEM
        self.validate()

    def validate(self) -> bool:
        if not self.schema_version:
            raise ValueError("schema_version must be non-empty")
        if not self.ecosystem or not self.upstream_package:
            raise ValueError("ecosystem and upstream_package must be non-empty")
        if self.capability_level not in {0, 1, 2, 3, 4}:
            raise ValueError("capability_level must be between 0 and 4")
        if not isinstance(self.metadata, dict) or not isinstance(self.provenance, dict):
            raise TypeError("metadata and provenance must be dictionaries")
        if not isinstance(self.warnings, list):
            raise TypeError("warnings must be a list")
        if not isinstance(self.advisory, bool) or not isinstance(self.offline_only, bool):
            raise TypeError("advisory and offline_only must be booleans")
        return True

    def to_dict(self) -> dict:
        payload = asdict(self)
        try:
            json.dumps(payload)
        except TypeError:
            payload["data"] = repr(self.data)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict):
        return cls(**dict(payload))


class QiskitCoreResult(QiskitResult):
    ECOSYSTEM = "qiskit_core"


class AerResult(QiskitResult):
    ECOSYSTEM = "qiskit_aer"


class NatureResult(QiskitResult):
    ECOSYSTEM = "qiskit_nature"


class AlgorithmsResult(QiskitResult):
    ECOSYSTEM = "qiskit_algorithms"


class FinanceResult(QiskitResult):
    ECOSYSTEM = "qiskit_finance"


class OptimizationResult(QiskitResult):
    ECOSYSTEM = "qiskit_optimization"


class MachineLearningResult(QiskitResult):
    ECOSYSTEM = "qiskit_machine_learning"


class DynamicsResult(QiskitResult):
    ECOSYSTEM = "qiskit_dynamics"


class ExperimentsResult(QiskitResult):
    ECOSYSTEM = "qiskit_experiments"


class MetalAdvisoryResult(QiskitResult):
    ECOSYSTEM = "qiskit_metal"


class RuntimeOfflineResult(QiskitResult):
    ECOSYSTEM = "qiskit_runtime"


class AddonsAdvisoryResult(QiskitResult):
    ECOSYSTEM = "qiskit_addons"


class QiskitConversionResult(QiskitResult):
    ECOSYSTEM = "qiskit_conversion"


_RESULT_CLASS_BY_ECOSYSTEM = {
    "qiskit_core": QiskitCoreResult,
    "qiskit_aer": AerResult,
    "qiskit_nature": NatureResult,
    "qiskit_algorithms": AlgorithmsResult,
    "qiskit_finance": FinanceResult,
    "qiskit_optimization": OptimizationResult,
    "qiskit_machine_learning": MachineLearningResult,
    "qiskit_dynamics": DynamicsResult,
    "qiskit_experiments": ExperimentsResult,
    "qiskit_metal": MetalAdvisoryResult,
    "qiskit_runtime": RuntimeOfflineResult,
    "qiskit_addons": AddonsAdvisoryResult,
    "qiskit_conversion": QiskitConversionResult,
}


def result_class_for_ecosystem(ecosystem: str):
    return _RESULT_CLASS_BY_ECOSYSTEM.get(ecosystem, QiskitResult)
