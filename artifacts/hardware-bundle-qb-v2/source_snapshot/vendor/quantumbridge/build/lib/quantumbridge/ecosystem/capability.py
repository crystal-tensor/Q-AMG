# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Capability records for optional ecosystem coverage.

Design source: docs/compat/full_ecosystem_coverage_policy_v0.1.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum


class CoverageLevel(IntEnum):
    """QuantumBridge ecosystem coverage levels."""

    INVENTORY = 0
    PASSTHROUGH = 1
    ADAPTER = 2
    NATIVE_SUBSET = 3
    PRODUCTION_EQUIVALENT = 4


@dataclass(frozen=True)
class CapabilityRecord:
    upstream_package: str
    module: str
    public_api: str
    api_type: str
    quantumbridge_level: int
    mode: str
    dependency_extra: str
    test: str
    risk: str
    notes: str

    def to_dict(self) -> dict:
        return asdict(self)
