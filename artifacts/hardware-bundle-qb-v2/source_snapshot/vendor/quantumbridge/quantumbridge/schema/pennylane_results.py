# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Stage 8B PennyLane result schemas."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


@dataclass
class PennyLaneResult:
    ecosystem: str = "pennylane"
    upstream_package: str = "pennylane"
    upstream_version: str | None = None
    capability_level: int = 2
    mode: str = "schema-adapter"
    raw_type: str = "unknown"
    data: Any = None
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    unsupported_reason: str | None = None
    quantumbridge_version: str | None = None
    schema_version: str = "0.2"

    ECOSYSTEM: ClassVar[str] = "pennylane"

    def __post_init__(self) -> None:
        self.ecosystem = self.ECOSYSTEM
        self.validate()

    def validate(self) -> bool:
        if self.schema_version not in {"0.1", "0.2"}:
            raise ValueError("schema_version must be supported")
        if self.ecosystem != "pennylane":
            raise ValueError("PennyLaneResult ecosystem must be 'pennylane'")
        if not self.upstream_package:
            raise ValueError("upstream_package must be non-empty")
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


class QNodeResult(PennyLaneResult):
    pass


class MeasurementResult(PennyLaneResult):
    pass


class GradientResult(PennyLaneResult):
    pass


class TransformResult(PennyLaneResult):
    pass


class TemplateResult(PennyLaneResult):
    pass


class QChemResult(PennyLaneResult):
    pass


class QNNResult(PennyLaneResult):
    pass


class DeviceResult(PennyLaneResult):
    pass


class ResourceResult(PennyLaneResult):
    pass


class QCutResult(PennyLaneResult):
    pass


class ShadowResult(PennyLaneResult):
    pass


class ConversionResult(PennyLaneResult):
    pass


class OperationBridgeResult(PennyLaneResult):
    pass


class MeasurementBridgeResult(PennyLaneResult):
    pass


class TapeBridgeResult(PennyLaneResult):
    pass


class QiskitBridgeResult(PennyLaneResult):
    pass


class QOSUQCIJobSpecResult(PennyLaneResult):
    pass
