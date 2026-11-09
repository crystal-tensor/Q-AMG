# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Aer was copied.
"""Examples for the Stage 9F Qiskit Aer executable simulator slice."""

from __future__ import annotations

from quantumbridge.core import Circuit

from .aer_upstream_adapter import run_qasm_upstream_aer, run_statevector_upstream_aer
from .noise_adapter import create_bitflip_noise_model
from .simulator_native import (
    run_noisy_qasm_simulator_native,
    run_qasm_simulator_native,
    run_statevector_simulator_native,
)


def bell_circuit(measured: bool = True) -> Circuit:
    circuit = Circuit(2, 2 if measured else 0, name="qb_aer_bell_example")
    circuit.h(0).cx(0, 1)
    if measured:
        circuit.measure(0, 0).measure(1, 1)
    return circuit


def run_statevector_example() -> dict[str, object]:
    circuit = bell_circuit(measured=False)
    native = run_statevector_simulator_native(circuit)
    upstream = _optional_upstream_statevector(circuit)
    return {"native": native, "upstream": upstream}


def run_qasm_counts_example(shots: int = 128, seed: int = 7) -> dict[str, object]:
    circuit = bell_circuit(measured=True)
    native = run_qasm_simulator_native(circuit, shots=shots, seed=seed)
    upstream = _optional_upstream_qasm(circuit, shots=shots, seed=seed)
    return {"native": native, "upstream": upstream}


def run_noisy_counts_example(shots: int = 128, seed: int = 7) -> dict[str, object]:
    circuit = bell_circuit(measured=True)
    native = run_noisy_qasm_simulator_native(
        circuit,
        shots=shots,
        seed=seed,
        noise_model=create_bitflip_noise_model(0.1),
    )
    upstream = _optional_upstream_qasm(circuit, shots=shots, seed=seed)
    return {"native": native, "upstream": upstream}


def _optional_upstream_statevector(circuit: Circuit):
    try:
        from quantumbridge.compat.qiskit_adapter import circuit_to_qiskit
    except Exception:
        return run_statevector_upstream_aer(circuit)
    try:
        return run_statevector_upstream_aer(circuit_to_qiskit(circuit))
    except Exception:
        return run_statevector_upstream_aer(circuit)


def _optional_upstream_qasm(circuit: Circuit, shots: int, seed: int):
    try:
        from quantumbridge.compat.qiskit_adapter import circuit_to_qiskit
    except Exception:
        return run_qasm_upstream_aer(circuit, shots=shots, seed_simulator=seed)
    try:
        return run_qasm_upstream_aer(circuit_to_qiskit(circuit), shots=shots, seed_simulator=seed)
    except Exception:
        return run_qasm_upstream_aer(circuit, shots=shots, seed_simulator=seed)
