# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Machine Learning was copied.
"""Warnings for Qiskit Machine Learning compatibility workflows."""

QISKIT_ML_WARNING = (
    "QuantumBridge Qiskit Machine Learning support is an optional passthrough/schema "
    "bridge plus minimal educational native QML workflows. It is not a full Qiskit "
    "Machine Learning replacement and is not production machine learning software."
)
NATIVE_KERNEL_WARNING = (
    "Native quantum kernel support is a deterministic educational workflow for small "
    "toy datasets."
)
NATIVE_QNN_WARNING = (
    "Native QNN support is a minimal educational forward/training workflow for small "
    "toy datasets and does not guarantee training performance or generalization."
)
HIGH_RISK_WARNING = (
    "QuantumBridge QML examples are not intended for medical, financial, employment, "
    "identity, safety, or other high-risk automated decisions."
)
UPSTREAM_ML_WARNING = (
    "Upstream passthrough requires optional qiskit-machine-learning ecosystem packages."
)


def ml_warnings(*extra: str) -> list[str]:
    """Return de-duplicated warnings for Stage 9E QML workflows."""

    warnings: list[str] = [QISKIT_ML_WARNING, HIGH_RISK_WARNING]
    for item in extra:
        if item and item not in warnings:
            warnings.append(item)
    return warnings


def native_provenance(workflow: str) -> dict[str, object]:
    return {
        "adapter": "quantumbridge.compat.qiskit_machine_learning",
        "workflow": workflow,
        "stage": "9E",
        "official_endorsement": False,
        "source_code_copied": False,
        "upstream_source_copied": False,
        "ibm_branding_copied": False,
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
    }


def upstream_provenance(workflow: str) -> dict[str, object]:
    data = native_provenance(workflow)
    data["adapter"] = "quantumbridge.compat.qiskit_machine_learning.upstream_adapter"
    data["mode"] = "optional_upstream_passthrough"
    return data
