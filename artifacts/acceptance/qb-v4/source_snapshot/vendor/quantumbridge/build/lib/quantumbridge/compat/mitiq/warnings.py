# This file is independently implemented for QuantumBridge SDK.
# No source code from Mitiq, Qiskit, or Qiskit Aer was copied.
"""Warnings and provenance helpers for Mitiq compatibility workflows."""

MITIQ_WARNING = (
    "QuantumBridge Mitiq support is an optional passthrough/schema bridge plus "
    "minimal educational native error-mitigation workflows. It is not a full "
    "Mitiq replacement and is not production error-mitigation software."
)
NATIVE_ZNE_WARNING = (
    "Native ZNE support is a deterministic educational workflow for small "
    "simulated circuits and simple extrapolation."
)
NATIVE_READOUT_WARNING = (
    "Native readout mitigation support is an educational calibration-matrix "
    "workflow and does not provide hardware calibration parity."
)
UPSTREAM_MITIQ_WARNING = (
    "Upstream passthrough requires optional mitiq ecosystem packages."
)


def mitiq_warnings(*extra: str) -> list[str]:
    warnings = [MITIQ_WARNING]
    for item in extra:
        if item and item not in warnings:
            warnings.append(item)
    return warnings


def native_provenance(workflow: str) -> dict[str, object]:
    return {
        "adapter": "quantumbridge.compat.mitiq",
        "workflow": workflow,
        "stage": "9G",
        "official_endorsement": False,
        "source_code_copied": False,
        "upstream_source_copied": False,
        "ibm_branding_copied": False,
        "mitiq_source_copied": False,
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
    }


def upstream_provenance(workflow: str) -> dict[str, object]:
    data = native_provenance(workflow)
    data["adapter"] = "quantumbridge.compat.mitiq.mitiq_upstream_adapter"
    data["mode"] = "optional_upstream_passthrough"
    return data
