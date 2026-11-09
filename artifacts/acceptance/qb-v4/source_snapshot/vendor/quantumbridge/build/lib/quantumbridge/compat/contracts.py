# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit, PennyLane, Quafu, or QOS-UQCI was copied.
"""Lightweight contracts for optional ecosystem adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Protocol, runtime_checkable


class CapabilityLevel(IntEnum):
    """Common capability levels for optional ecosystem adapters."""

    INVENTORY = 0
    PASSTHROUGH = 1
    SCHEMA_ADAPTER = 2
    NATIVE_SUBSET = 3
    PRODUCTION_EQUIVALENT = 4


@dataclass(frozen=True)
class AdapterWarning:
    """User-facing adapter warning with a stable machine-readable code."""

    code: str
    message: str
    severity: str = "warning"


@dataclass(frozen=True)
class AdapterProvenance:
    """Source metadata attached to adapter outputs and unsupported paths."""

    ecosystem: str
    upstream_package: str
    upstream_version: str | None = None
    quantumbridge_version: str | None = None
    adapter: str | None = None
    capability_level: CapabilityLevel = CapabilityLevel.INVENTORY
    mode: str = "inventory"
    dependency_extra: str | None = None
    official_endorsement: bool = False
    cloud_access: bool = False
    token_storage: bool = False
    source_code_copied: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload["capability_level"] = int(self.capability_level)
        return payload


@dataclass(frozen=True)
class UnsupportedCapability:
    """Structured response for unsupported adapter paths."""

    reason: str
    capability_level: CapabilityLevel = CapabilityLevel.INVENTORY
    warnings: tuple[AdapterWarning, ...] = field(default_factory=tuple)
    provenance: AdapterProvenance | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": False,
            "reason": self.reason,
            "capability_level": int(self.capability_level),
            "warnings": [warning.__dict__ for warning in self.warnings],
            "provenance": None if self.provenance is None else self.provenance.to_dict(),
        }


@runtime_checkable
class EcosystemAdapterContract(Protocol):
    """Protocol for ecosystem adapter convergence in Stage 8."""

    capability_level: CapabilityLevel
    production_ready: bool
    native_implementation: bool
    upstream_required: bool

    def dependency_available(self) -> bool:
        ...

    def get_upstream_version(self) -> str | None:
        ...

    def list_public_api_inventory(self) -> list[Any]:
        ...

    def get_public_object(self, name: str) -> Any:
        ...

    def passthrough_call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        ...

    def wrap_result(self, obj: Any) -> Any:
        ...

    def to_quantumbridge_schema(self, obj: Any) -> Any:
        ...

    def get_warnings(self) -> list[AdapterWarning]:
        ...

    def get_provenance(self) -> AdapterProvenance:
        ...

    def unsupported(self, reason: str) -> UnsupportedCapability:
        ...

    def validate_environment(self) -> bool:
        ...
