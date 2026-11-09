# This file is independently implemented for QuantumBridge SDK.
# No source code from QOS-UQCI, Quafu, pyquafu, IBM, or Qiskit was copied.
"""Serializable schemas for backend compatibility and cloud workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


@dataclass
class BackendCompatibilityResult:
    """Envelope for clean-room QOS-UQCI and Quafu backend slices."""

    workflow: str = "backend_compatibility"
    project: str = "backend"
    mode: str = "native_mock"
    upstream_package: str | None = None
    upstream_version: str | None = None
    quantumbridge_version: str | None = None
    capability_level: int = 2
    production_ready: bool = False
    native_implementation: bool = False
    job_spec: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    device_spec: dict[str, Any] = field(default_factory=dict)
    calset: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    openqasm_artifact: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    probabilities: dict[str, float] = field(default_factory=dict)
    shots: int | None = None
    seed: int | None = None
    raw_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    unsupported_reason: str | None = None
    schema_version: str = "0.1"
    ecosystem: str = "quantumbridge_backend_compat"

    ECOSYSTEM: ClassVar[str | None] = None

    def __post_init__(self) -> None:
        if self.ECOSYSTEM is not None:
            self.ecosystem = self.ECOSYSTEM
        self.job_spec = dict(self.job_spec)
        self.payload = dict(self.payload)
        self.device_spec = dict(self.device_spec)
        self.calset = dict(self.calset)
        self.manifest = dict(self.manifest)
        self.openqasm_artifact = dict(self.openqasm_artifact)
        self.counts = {str(key): int(value) for key, value in self.counts.items()}
        self.probabilities = {str(key): float(value) for key, value in self.probabilities.items()}
        self.metadata = dict(self.metadata)
        self.warnings = [str(value) for value in self.warnings]
        self.provenance = dict(self.provenance)
        self.validate()

    def validate(self) -> bool:
        if not self.workflow:
            raise ValueError("workflow must be non-empty")
        valid_modes = {"native_mock", "upstream_passthrough", "conversion", "cloud_hardware"}
        if self.mode not in valid_modes:
            raise ValueError("mode must be native_mock, upstream_passthrough, conversion, or cloud_hardware")
        if self.capability_level not in {0, 1, 2, 3, 4}:
            raise ValueError("capability_level must be between 0 and 4")
        if self.production_ready:
            raise ValueError("Stage 10A backend compatibility results must not claim production readiness")
        if self.metadata.get("official_endorsement_claim") is True or self.provenance.get("official_endorsement_claim") is True:
            raise ValueError("backend compatibility results must not claim official endorsement")
        cloud_flags = ("cloud_access", "token_read", "hardware_access")
        if self.mode == "cloud_hardware":
            for key in cloud_flags:
                if self.metadata.get(key) is not True or self.provenance.get(key) is not True:
                    raise ValueError(f"cloud_hardware results must set {key}=True in metadata and provenance")
        else:
            for key in cloud_flags:
                if self.metadata.get(key) is True or self.provenance.get(key) is True:
                    raise ValueError(f"offline backend compatibility must not set {key}=True")
        if self.shots is not None and self.shots <= 0:
            raise ValueError("shots must be positive when provided")
        return True

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BackendCompatibilityResult":
        data = dict(payload)
        data["counts"] = {str(key): int(value) for key, value in data.get("counts", {}).items()}
        data["probabilities"] = {str(key): float(value) for key, value in data.get("probabilities", {}).items()}
        return cls(**data)


class QOSUQCIBackendResult(BackendCompatibilityResult):
    ECOSYSTEM = "quantumbridge_native_qos_uqci_compat"


class QuafuBackendResult(BackendCompatibilityResult):
    ECOSYSTEM = "quantumbridge_native_quafu_compat"


class UpstreamQOSUQCIResult(BackendCompatibilityResult):
    ECOSYSTEM = "qos_uqci"


class UpstreamQuafuResult(BackendCompatibilityResult):
    ECOSYSTEM = "quafu"
