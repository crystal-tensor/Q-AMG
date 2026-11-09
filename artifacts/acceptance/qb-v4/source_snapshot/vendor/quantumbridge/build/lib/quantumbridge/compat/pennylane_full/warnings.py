# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Warnings and metadata for PennyLane optional passthrough adapters."""

from __future__ import annotations

from quantumbridge.compat.contracts import AdapterProvenance, AdapterWarning, CapabilityLevel, UnsupportedCapability

from .dependency import DEPENDENCY_EXTRA, PACKAGE_NAME, get_upstream_version


PENNYLANE_ADAPTER_WARNING = (
    "This PennyLane adapter is an optional passthrough/schema bridge, "
    "not a full native PennyLane replacement."
)

CAPABILITY_LEVEL = CapabilityLevel.PASSTHROUGH
PRODUCTION_READY = False
NATIVE_IMPLEMENTATION = False
UPSTREAM_REQUIRED = True


def get_warnings() -> list[AdapterWarning]:
    return [
        AdapterWarning(
            code="pennylane_optional_passthrough",
            message=PENNYLANE_ADAPTER_WARNING,
        )
    ]


def get_provenance(mode: str = "pennylane-full-adapter", level: CapabilityLevel = CAPABILITY_LEVEL) -> AdapterProvenance:
    return AdapterProvenance(
        ecosystem="pennylane",
        upstream_package=PACKAGE_NAME,
        upstream_version=get_upstream_version(),
        adapter="quantumbridge.compat.pennylane_full",
        capability_level=level,
        mode=mode,
        dependency_extra=DEPENDENCY_EXTRA,
        official_endorsement=False,
        cloud_access=False,
        token_storage=False,
        source_code_copied=False,
    )


def unsupported(reason: str, level: CapabilityLevel = CapabilityLevel.INVENTORY) -> UnsupportedCapability:
    return UnsupportedCapability(
        reason=reason,
        capability_level=level,
        warnings=tuple(get_warnings()),
        provenance=get_provenance(mode="unsupported", level=level),
    )


def adapter_metadata(level: CapabilityLevel = CAPABILITY_LEVEL) -> dict:
    return {
        "capability_level": int(level),
        "production_ready": PRODUCTION_READY,
        "native_implementation": NATIVE_IMPLEMENTATION,
        "upstream_required": UPSTREAM_REQUIRED,
        "warnings": [warning.__dict__ for warning in get_warnings()],
        "provenance": get_provenance(level=level).to_dict(),
    }
