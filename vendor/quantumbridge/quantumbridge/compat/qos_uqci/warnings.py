# This file is independently implemented for QuantumBridge SDK.
# No source code from QOS-UQCI, IBM, or Qiskit was copied.
"""Warnings and provenance helpers for clean-room QOS-UQCI compatibility."""

from __future__ import annotations

from typing import Any

QOS_UQCI_COMPAT_WARNING = (
    "QuantumBridge QOS-UQCI compatibility support is a clean-room offline adapter. "
    "It is not a production QOS runtime and does not access real cloud services or hardware."
)

MOCK_BACKEND_WARNING = (
    "Mock backend execution uses QuantumBridge local simulation and is not real quantum hardware execution."
)

TOKEN_WARNING = "No token or credential is read by this workflow."

UPSTREAM_QOS_WARNING = "Upstream passthrough requires an optional QOS-UQCI package and remains offline by default."


def qos_uqci_warnings(*items: str) -> list[str]:
    values = [QOS_UQCI_COMPAT_WARNING, MOCK_BACKEND_WARNING, TOKEN_WARNING]
    for item in items:
        if item and item not in values:
            values.append(str(item))
    return values


def native_provenance(workflow: str) -> dict[str, Any]:
    return {
        "project": "qos-uqci",
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
