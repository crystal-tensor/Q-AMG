# This file is independently implemented for QuantumBridge SDK.
# No source code from MQT, IBM, or Qiskit was copied.
"""Warnings and provenance helpers for clean-room MQT compatibility."""

from __future__ import annotations

from typing import Any

MQT_COMPAT_WARNING = (
    "QuantumBridge MQT compatibility support is a clean-room educational adapter. "
    "It is not a full MQT Core, DDSIM, or QMAP replacement and does not provide "
    "production compiler, mapper, or simulator parity."
)

DDSIM_LIKE_WARNING = (
    "Native DDSIM-like support uses QuantumBridge educational simulation plus "
    "decision-diagram-inspired metadata. It is not a full decision-diagram simulator."
)

QMAP_LIKE_WARNING = (
    "Native QMAP-like support is a minimal educational topology mapping and routing "
    "workflow. It is not production quantum compilation or optimal mapping software."
)

UPSTREAM_MQT_WARNING = "Upstream passthrough requires optional MQT ecosystem packages."


def mqt_warnings(*items: str) -> list[str]:
    values = [MQT_COMPAT_WARNING]
    for item in items:
        if item and item not in values:
            values.append(str(item))
    return values


def native_provenance(workflow: str, project: str = "mqt") -> dict[str, Any]:
    return {
        "project": project,
        "workflow": workflow,
        "source": "quantumbridge_clean_room_native",
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
        "copied_upstream_source": False,
        "official_endorsement_claim": False,
        "full_replacement_claim": False,
        "production_parity_claim": False,
    }


def upstream_provenance(workflow: str, package_name: str) -> dict[str, Any]:
    return {
        "project": "mqt",
        "workflow": workflow,
        "source": "optional_upstream_passthrough_boundary",
        "upstream_package": package_name,
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
        "copied_upstream_source": False,
        "official_endorsement_claim": False,
        "full_replacement_claim": False,
        "production_parity_claim": False,
    }
