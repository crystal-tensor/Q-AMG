# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM was copied.
"""Qiskit Addons result schema adapter."""

import warnings

warnings.warn(
    "This adapter is a SCAFFOLD adapter. "
    "Full implementation is not yet available. "
    "This adapter only provides the Result schema wrapper. "
    "Level 0: inventory only; Level 1: passthrough; Level 2: NOT implemented. "
    "Offline only: no real IBM Quantum access, OBP/SQD/MPF/AQC access.",
    UserWarning,
    stacklevel=2
)


def wrap_addon_result(obj, addon_type: str = "generic", metadata=None):
    """Wrap a Qiskit Addons result in QuantumBridge AddonsResult schema."""
    from quantumbridge.schema import AddonsResult
    return AddonsResult(
        ecosystem="qiskit",
        upstream_package="qiskit-addons",
        upstream_version="offline",
        capability_level=1,
        mode="scaffold",
        raw_type=addon_type,
        data=obj,
        metadata=metadata or {},
        provenance={
            "upstream_package": "qiskit-addons",
            "adapter": "result_adapter",
            "capability_level": 1,
            "ibm_quantum_access": False,
            "obp_access": False,
            "sqd_access": False,
            "mpf_access": False,
            "aqc_access": False,
        },
        warnings=[
            "This is a scaffold adapter. Advisory: offline only.",
            "No IBM Quantum access. No OBP/SQD/MPF/AQC access.",
            "All Qiskit Addons capabilities are simulated."
        ]
    )
