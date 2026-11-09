# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM was copied.
"""Qiskit IBM Runtime result schema adapter."""

import warnings

warnings.warn(
    "This adapter is a SCAFFOLD adapter. "
    "Full implementation is not yet available. "
    "This adapter only provides the Result schema wrapper. "
    "Level 0: inventory only; Level 1: passthrough; Level 2: NOT implemented. "
    "Offline only: no IBM Cloud token storage, no real IBM Runtime access.",
    UserWarning,
    stacklevel=2
)


def wrap_runtime_result(obj, job_id: str = None, metadata=None):
    """Wrap a Qiskit IBM Runtime result in QuantumBridge RuntimeResult schema."""
    from quantumbridge.schema import RuntimeResult
    return RuntimeResult(
        ecosystem="qiskit",
        upstream_package="qiskit-ibm-runtime",
        upstream_version="offline",
        capability_level=1,
        mode="scaffold",
        raw_type="runtime_job",
        data=obj,
        metadata=metadata or {},
        provenance={
            "upstream_package": "qiskit-ibm-runtime",
            "adapter": "result_adapter",
            "capability_level": 1,
            "cloud_access": False,
            "token_storage": False,
        },
        warnings=[
            "This is a scaffold adapter. Advisory: offline only.",
            "No IBM Cloud access. No token storage.",
            "Real IBM Runtime access requires separate authentication."
        ]
    )
