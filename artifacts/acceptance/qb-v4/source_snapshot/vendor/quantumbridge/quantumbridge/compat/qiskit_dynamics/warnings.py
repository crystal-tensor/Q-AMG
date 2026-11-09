# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Qiskit Dynamics was copied.
"""Warnings and provenance for offline Qiskit Dynamics workflows."""

from __future__ import annotations

DYNAMICS_WARNING = (
    "QuantumBridge Qiskit Dynamics support is an optional passthrough/schema "
    "bridge plus minimal educational single-qubit dynamics workflows. It is not "
    "a full Qiskit Dynamics replacement and is not production dynamics simulation software."
)
UPSTREAM_DYNAMICS_WARNING = (
    "Upstream passthrough requires optional qiskit-dynamics ecosystem packages."
)


def dynamics_warnings(*extra: str) -> list[str]:
    warnings = [DYNAMICS_WARNING]
    warnings.extend(str(item) for item in extra if item)
    return warnings


def upstream_warnings(*extra: str) -> list[str]:
    warnings = [DYNAMICS_WARNING, UPSTREAM_DYNAMICS_WARNING]
    warnings.extend(str(item) for item in extra if item)
    return warnings


def native_provenance(workflow: str) -> dict[str, object]:
    return {
        "adapter": "quantumbridge.compat.qiskit_dynamics",
        "workflow": workflow,
        "stage": "9I",
        "source": "clean_room_native",
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
        "production_dynamics": False,
    }


def upstream_provenance(workflow: str, upstream_version: str | None) -> dict[str, object]:
    return {
        "adapter": "quantumbridge.compat.qiskit_dynamics",
        "workflow": workflow,
        "stage": "9I",
        "source": "optional_upstream_passthrough",
        "upstream_package": "qiskit-dynamics",
        "upstream_version": upstream_version,
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
        "production_dynamics": False,
    }
