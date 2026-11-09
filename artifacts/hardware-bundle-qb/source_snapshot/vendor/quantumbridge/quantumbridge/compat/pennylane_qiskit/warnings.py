# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit, PennyLane, or PennyLane-Qiskit was copied.
"""Warnings and provenance helpers for PennyLane-Qiskit bridge workflows."""

BRIDGE_WARNING = (
    "QuantumBridge PennyLane-Qiskit bridge support is a clean-room educational "
    "adapter. It is not a full PennyLane-Qiskit plugin replacement and does "
    "not provide complete Qiskit or PennyLane parity."
)
QISKIT_TO_PENNYLANE_WARNING = (
    "Qiskit-to-PennyLane conversion currently supports a small basic-gate "
    "subset and may not preserve all advanced circuit semantics."
)
PENNYLANE_TO_QISKIT_WARNING = (
    "PennyLane-to-Qiskit conversion currently supports a small operation and "
    "measurement subset and may not preserve all device, transform, or "
    "gradient semantics."
)
UPSTREAM_PLUGIN_WARNING = (
    "Upstream passthrough requires optional PennyLane-Qiskit ecosystem packages."
)


def bridge_warnings(*extra: str) -> list[str]:
    warnings = [BRIDGE_WARNING]
    for item in extra:
        if item and item not in warnings:
            warnings.append(item)
    return warnings


def native_provenance(workflow: str) -> dict[str, object]:
    return {
        "adapter": "quantumbridge.compat.pennylane_qiskit",
        "workflow": workflow,
        "stage": "9H",
        "official_endorsement": False,
        "source_code_copied": False,
        "upstream_source_copied": False,
        "ibm_branding_copied": False,
        "qiskit_source_copied": False,
        "pennylane_source_copied": False,
        "pennylane_qiskit_source_copied": False,
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
    }


def upstream_provenance(workflow: str) -> dict[str, object]:
    data = native_provenance(workflow)
    data["adapter"] = "quantumbridge.compat.pennylane_qiskit.upstream_adapter"
    data["mode"] = "optional_upstream_passthrough"
    return data
