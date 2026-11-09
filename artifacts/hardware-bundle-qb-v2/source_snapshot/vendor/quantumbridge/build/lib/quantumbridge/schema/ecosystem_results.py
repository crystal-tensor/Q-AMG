# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Serializable result schemas for optional ecosystem adapters.

Design source: docs/compat/full_ecosystem_coverage_policy_v0.1.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


@dataclass
class EcosystemResult:
    ecosystem: str
    upstream_package: str
    upstream_version: str | None
    capability_level: int
    mode: str
    raw_type: str
    data: Any
    metadata: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    schema_version: str = "0.1"

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


class FinanceResult(EcosystemResult):
    ECOSYSTEM = "finance"


class ChemistryResult(EcosystemResult):
    ECOSYSTEM = "chemistry"


class AlgorithmsResult(EcosystemResult):
    ECOSYSTEM = "algorithms"


class OptimizationResult(EcosystemResult):
    ECOSYSTEM = "optimization"


class MLResult(EcosystemResult):
    ECOSYSTEM = "machine-learning"


class DynamicsResult(EcosystemResult):
    ECOSYSTEM = "dynamics"


class ExperimentsResult(EcosystemResult):
    ECOSYSTEM = "experiments"


class MetalDesignResult(EcosystemResult):
    ECOSYSTEM = "metal-design"


class AerResult(EcosystemResult):
    ECOSYSTEM = "aer"


class PennyLaneResult(EcosystemResult):
    ECOSYSTEM = "pennylane"
