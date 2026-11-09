# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""PennyLane observable metadata and native observable bridge."""

from quantumbridge.compat.pennylane_full.operations_adapter import ADAPTER, get_operation
from quantumbridge.compat.pennylane_full.warnings import adapter_metadata
from quantumbridge.qml.observables import Hadamard, Hamiltonian, Identity, PauliX, PauliY, PauliZ, SparseHamiltonian


OBSERVABLE_NAMES = ("PauliX", "PauliY", "PauliZ", "Identity", "Hadamard", "Hamiltonian", "SparseHamiltonian")
NATIVE_OBSERVABLES = {
    "PauliX": PauliX,
    "PauliY": PauliY,
    "PauliZ": PauliZ,
    "Identity": Identity,
    "Hadamard": Hadamard,
    "Hamiltonian": Hamiltonian,
    "SparseHamiltonian": SparseHamiltonian,
}

dependency_available = ADAPTER.dependency_available
get_upstream_version = ADAPTER.get_upstream_version
list_public_api_inventory = ADAPTER.list_public_api_inventory


def list_observables() -> list[str]:
    return list(OBSERVABLE_NAMES)


def describe_observable(name: str) -> dict:
    supported = name in OBSERVABLE_NAMES
    return {
        "name": name,
        "supported": supported,
        "native": supported,
        "unsupported_reason": None if supported else f"{name!r} is not in the current native PennyLane observable subset.",
        **adapter_metadata(),
    }


def get_observable(name: str):
    if name in NATIVE_OBSERVABLES:
        return NATIVE_OBSERVABLES[name]
    return get_operation(name)
