# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Optimization was copied.
"""Warnings and capability metadata for Qiskit Optimization compatibility."""

OPTIMIZATION_ADAPTER_WARNING = (
    "QuantumBridge Qiskit Optimization support is an optional passthrough/schema "
    "bridge plus minimal educational native examples. It is not a full Qiskit "
    "Optimization replacement and is not production optimization software."
)

NATIVE_QUADRATIC_PROGRAM_WARNING = (
    "Native QuadraticProgram support is a minimal deterministic educational binary "
    "optimization solver for small problems."
)

UPSTREAM_PASSTHROUGH_WARNING = (
    "Upstream passthrough requires optional qiskit-optimization and qiskit-algorithms "
    "packages."
)

capability_level = 3
production_ready = False
native_implementation = True
upstream_required = False


def get_warnings() -> tuple[str, ...]:
    """Return standard warnings for this compatibility layer."""

    return (
        OPTIMIZATION_ADAPTER_WARNING,
        NATIVE_QUADRATIC_PROGRAM_WARNING,
        UPSTREAM_PASSTHROUGH_WARNING,
    )
