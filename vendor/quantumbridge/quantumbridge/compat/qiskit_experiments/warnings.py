# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Qiskit Experiments was copied.
"""Warnings and provenance for offline Qiskit Experiments workflows."""

from __future__ import annotations

EXPERIMENTS_WARNING = (
    "QuantumBridge Qiskit Experiments support is an optional passthrough/schema "
    "bridge plus minimal educational offline experiment workflows. It is not a "
    "full Qiskit Experiments replacement and does not provide hardware calibration."
)
NATIVE_EXPERIMENT_WARNING = (
    "Native experiment workflows use deterministic synthetic data and educational "
    "fitting methods. They are not hardware experiment results."
)
UPSTREAM_EXPERIMENTS_WARNING = (
    "Upstream passthrough requires optional qiskit-experiments ecosystem packages."
)


def experiments_warnings(*extra: str) -> list[str]:
    warnings = [EXPERIMENTS_WARNING, NATIVE_EXPERIMENT_WARNING]
    warnings.extend(str(item) for item in extra if item)
    return warnings


def upstream_warnings(*extra: str) -> list[str]:
    warnings = [EXPERIMENTS_WARNING, UPSTREAM_EXPERIMENTS_WARNING]
    warnings.extend(str(item) for item in extra if item)
    return warnings


def native_provenance(workflow: str) -> dict[str, object]:
    return {
        "adapter": "quantumbridge.compat.qiskit_experiments",
        "workflow": workflow,
        "stage": "9I",
        "source": "clean_room_native",
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
        "hardware_calibration": False,
    }


def upstream_provenance(workflow: str, upstream_version: str | None) -> dict[str, object]:
    return {
        "adapter": "quantumbridge.compat.qiskit_experiments",
        "workflow": workflow,
        "stage": "9I",
        "source": "optional_upstream_passthrough",
        "upstream_package": "qiskit-experiments",
        "upstream_version": upstream_version,
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
        "hardware_calibration": False,
    }
