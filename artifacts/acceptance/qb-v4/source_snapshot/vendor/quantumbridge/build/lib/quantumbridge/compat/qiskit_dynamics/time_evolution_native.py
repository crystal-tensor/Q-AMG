# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Qiskit Dynamics was copied.
"""Clean-room educational single-qubit time evolution."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .hamiltonian_model_native import PAULI


def evolve_single_qubit_state_native(
    initial_state: Iterable[complex],
    hamiltonian: dict[str, object],
    times: Iterable[float],
) -> dict[str, object]:
    state0 = _normalize(np.array(list(initial_state), dtype=complex))
    axis = str(hamiltonian["axis"])
    frequency = float(hamiltonian["frequency"])
    sigma = PAULI[axis]
    time_values = [float(value) for value in times]
    states = []
    expectations = {"x": [], "y": [], "z": []}
    for time in time_values:
        theta = 0.5 * frequency * time
        unitary = math.cos(theta) * np.eye(2, dtype=complex) - 1.0j * math.sin(theta) * sigma
        state = unitary @ state0
        states.append(state)
        for key, pauli in PAULI.items():
            expectations[key].append(float(np.real(np.vdot(state, pauli @ state))))
    return {
        "times": time_values,
        "states": states,
        "expectation_values": expectations,
        "final_state": states[-1],
    }


def serialize_state(state: Iterable[complex]) -> list[dict[str, float]]:
    return [{"real": float(value.real), "imag": float(value.imag)} for value in state]


def serialize_states(states: Iterable[Iterable[complex]]) -> list[list[dict[str, float]]]:
    return [serialize_state(state) for state in states]


def _normalize(state: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(state))
    if norm == 0.0:
        raise ValueError("initial_state must be non-zero")
    if state.shape != (2,):
        raise ValueError("Stage 9I dynamics supports one-qubit states only")
    return state / norm
