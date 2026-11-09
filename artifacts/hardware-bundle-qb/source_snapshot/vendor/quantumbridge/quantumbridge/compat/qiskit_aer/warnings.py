# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Aer was copied.
"""Warnings and provenance helpers for Qiskit Aer compatibility workflows."""

QISKIT_AER_WARNING = (
    "QuantumBridge Qiskit Aer support is an optional passthrough/schema bridge "
    "plus minimal educational native simulator workflows. It is not a full "
    "Qiskit Aer replacement and is not production simulator software."
)
NATIVE_SIMULATOR_WARNING = (
    "Native simulator support is a minimal deterministic educational simulator "
    "for small circuits."
)
NATIVE_NOISE_WARNING = (
    "Native noise support is educational metadata / simple sampling noise only "
    "and does not provide Qiskit Aer noise-model parity."
)
UPSTREAM_AER_WARNING = (
    "Upstream passthrough requires optional qiskit-aer ecosystem packages."
)


def aer_warnings(*extra: str) -> list[str]:
    warnings = [QISKIT_AER_WARNING]
    for item in extra:
        if item and item not in warnings:
            warnings.append(item)
    return warnings


def native_provenance(workflow: str) -> dict[str, object]:
    return {
        "adapter": "quantumbridge.compat.qiskit_aer",
        "workflow": workflow,
        "stage": "9F",
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
    data["adapter"] = "quantumbridge.compat.qiskit_aer.aer_upstream_adapter"
    data["mode"] = "optional_upstream_passthrough"
    return data
