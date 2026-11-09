# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Metal design result schema adapter."""

import warnings

warnings.warn(
    "This adapter is a SCAFFOLD adapter. "
    "Full implementation is not yet available. "
    "qiskit-metal installation failed on Python 3.12. "
    "This adapter only provides a placeholder schema wrapper. "
    "Level 0: inventory placeholder; Level 1: NOT verified; Level 2: NOT implemented. "
    "Not chip fabrication.",
    UserWarning,
    stacklevel=2
)


def wrap_metal_result(obj, metadata=None):
    """Wrap a Qiskit Metal design result in QuantumBridge MetalDesignResult schema."""
    from quantumbridge.schema import MetalDesignResult
    return MetalDesignResult(
        ecosystem="qiskit",
        upstream_package="qiskit-metal",
        upstream_version="not_installed",
        capability_level=0,
        mode="scaffold",
        raw_type="metal_design",
        data=obj,
        metadata=metadata or {},
        provenance={
            "upstream_package": "qiskit-metal",
            "adapter": "result_adapter",
            "capability_level": 0,
            "install_status": "FAILED",
        },
        warnings=[
            "This is a scaffold adapter. Advisory: not installed (Python 3.12 incompatibility).",
            "No chip fabrication capability claimed."
        ]
    )
