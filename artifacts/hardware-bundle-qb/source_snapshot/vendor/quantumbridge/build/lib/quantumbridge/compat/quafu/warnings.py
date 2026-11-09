# This file is independently implemented for QuantumBridge SDK.
# No source code from Quafu, pyquafu, IBM, or Qiskit was copied.
"""Warnings and provenance helpers for clean-room Quafu compatibility."""

from __future__ import annotations

from typing import Any

QUAFU_COMPAT_WARNING = (
    "QuantumBridge Quafu compatibility support is a clean-room offline adapter. "
    "It is not a full pyquafu replacement, official Quafu integration, or production backend."
)

MOCK_BACKEND_WARNING = (
    "Mock backend execution uses QuantumBridge local simulation and is not real quantum hardware execution."
)

TOKEN_WARNING = "No token or credential is read by this workflow."

UPSTREAM_QUAFU_WARNING = "Upstream pyquafu passthrough requires optional pyquafu and remains offline by default."

QUAFU_CLOUD_WARNING = (
    "This result was submitted through the optional QuarkStudio client to a credentialed Quafu cloud backend. "
    "Hardware availability, calibration, queueing, and result quality are controlled by the remote platform."
)

QUAFU_CLOUD_SCOPE_WARNING = (
    "QuantumBridge currently guarantees only the documented OpenQASM 2 circuit subset on this cloud path; "
    "this is not a claim that every Qiskit or PennyLane API is hardware-executable."
)


def quafu_warnings(*items: str) -> list[str]:
    values = [QUAFU_COMPAT_WARNING, MOCK_BACKEND_WARNING, TOKEN_WARNING]
    for item in items:
        if item and item not in values:
            values.append(str(item))
    return values


def native_provenance(workflow: str) -> dict[str, Any]:
    return {
        "project": "quafu",
        "workflow": workflow,
        "source": "quantumbridge_clean_room_native",
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
        "copied_upstream_source": False,
        "official_endorsement_claim": False,
        "full_replacement_claim": False,
        "production_parity_claim": False,
        "production_backend_claim": False,
    }


def upstream_provenance(workflow: str, package_name: str) -> dict[str, Any]:
    data = native_provenance(workflow)
    data.update(
        {
            "source": "optional_upstream_passthrough_boundary",
            "upstream_package": package_name,
        }
    )
    return data


def cloud_warnings(*items: str) -> list[str]:
    values = [QUAFU_CLOUD_WARNING, QUAFU_CLOUD_SCOPE_WARNING]
    for item in items:
        if item and item not in values:
            values.append(str(item))
    return values


def cloud_provenance(workflow: str, transport: str = "quarkstudio") -> dict[str, Any]:
    return {
        "project": "quafu",
        "workflow": workflow,
        "source": "quantumbridge_clean_room_cloud_adapter",
        "transport": transport,
        "cloud_access": True,
        "token_read": True,
        "hardware_access": True,
        "copied_upstream_source": False,
        "official_endorsement_claim": False,
        "full_replacement_claim": False,
        "production_parity_claim": False,
        "production_backend_claim": False,
    }
