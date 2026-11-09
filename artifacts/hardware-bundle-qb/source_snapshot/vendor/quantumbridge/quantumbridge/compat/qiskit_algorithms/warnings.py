# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Algorithms was copied.
"""Warnings for Qiskit Algorithms executable compatibility adapters."""

ALGORITHMS_ADAPTER_WARNING = (
    "QuantumBridge Qiskit Algorithms support is an optional passthrough/schema "
    "bridge plus minimal educational native examples. It is not a full Qiskit "
    "Algorithms replacement and is not production algorithm software."
)
NATIVE_VQE_WARNING = (
    "Native VQE is a minimal deterministic educational implementation for small Hamiltonians."
)
NATIVE_QAOA_WARNING = (
    "Native QAOA is a minimal educational MaxCut-compatible workflow and may use "
    "exact enumeration for verification."
)
NATIVE_GROVER_WARNING = (
    "Native Grover is a minimal educational statevector workflow for small marked-bitstring "
    "search problems."
)
UPSTREAM_ALGORITHMS_WARNING = (
    "Upstream passthrough requires optional qiskit-algorithms ecosystem packages."
)


def algorithm_warnings(*extra: str) -> list[str]:
    """Return de-duplicated adapter warnings."""

    seen: set[str] = set()
    values: list[str] = []
    for warning in (ALGORITHMS_ADAPTER_WARNING, *extra):
        if warning and warning not in seen:
            values.append(warning)
            seen.add(warning)
    return values
