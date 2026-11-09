# This file is independently implemented for QuantumBridge SDK.
# No source code from TorchQuantum, PyTorch, IBM, or Qiskit was copied.
"""Warnings and provenance helpers for TorchQuantum-style compatibility."""

from __future__ import annotations

from typing import Any

TORCHQUANTUM_COMPAT_WARNING = (
    "QuantumBridge TorchQuantum compatibility support is a clean-room educational "
    "adapter. It is not a full TorchQuantum or PyTorch replacement and does not "
    "provide production QML training parity."
)

NATIVE_LAYER_WARNING = (
    "Native TorchQuantum-like layers are minimal educational workflows for small "
    "toy datasets and small circuits."
)

NATIVE_TRAINING_WARNING = (
    "Native training uses deterministic educational grid search and does not "
    "guarantee training performance or generalization."
)

HIGH_RISK_ML_WARNING = (
    "QuantumBridge QML examples are not intended for medical, financial, "
    "employment, identity, safety, or other high-risk automated decisions."
)

UPSTREAM_TORCHQUANTUM_WARNING = (
    "Upstream passthrough requires optional TorchQuantum ecosystem packages."
)


def torchquantum_warnings(*extra: str) -> list[str]:
    values = [TORCHQUANTUM_COMPAT_WARNING, HIGH_RISK_ML_WARNING]
    for item in extra:
        if item and item not in values:
            values.append(str(item))
    return values


def native_provenance(workflow: str) -> dict[str, Any]:
    return {
        "adapter": "quantumbridge.compat.torchquantum",
        "workflow": workflow,
        "source": "quantumbridge_clean_room_native",
        "stage": "9K",
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
        "copied_upstream_source": False,
        "official_endorsement_claim": False,
        "full_replacement_claim": False,
        "production_parity_claim": False,
        "high_risk_decision_claim": False,
    }


def upstream_provenance(workflow: str) -> dict[str, Any]:
    data = native_provenance(workflow)
    data["source"] = "optional_upstream_passthrough_boundary"
    data["mode"] = "upstream_passthrough"
    data["upstream_package"] = "torchquantum"
    return data
