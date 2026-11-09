# This file is independently implemented for QuantumBridge SDK.
# No source code from Mitiq, Qiskit, or Qiskit Aer was copied.
"""Executable examples for Stage 9G Mitiq compatibility."""

from __future__ import annotations

from quantumbridge.core import Circuit

from .mitiq_upstream_adapter import (
    run_readout_mitigation_upstream_if_available,
    run_zne_upstream_if_available,
)
from .readout_mitigation_native import run_readout_mitigation_native
from .zne_native import run_zne_native


def bell_circuit() -> Circuit:
    circuit = Circuit(2, 2)
    circuit.h(0).cx(0, 1).measure(0, 0).measure(1, 1)
    return circuit


def run_zne_example(shots: int = 256, seed: int = 13) -> dict[str, object]:
    native = run_zne_native(bell_circuit(), observable="ZZ", shots=shots, seed=seed)
    upstream = run_zne_upstream_if_available()
    return {"native": native, "upstream": upstream}


def run_readout_mitigation_example(shots: int = 256, seed: int = 21) -> dict[str, object]:
    native = run_readout_mitigation_native(bell_circuit(), shots=shots, seed=seed)
    upstream = run_readout_mitigation_upstream_if_available()
    return {"native": native, "upstream": upstream}
