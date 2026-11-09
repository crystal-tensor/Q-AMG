# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

import numpy as np

from quantumbridge.core import Circuit


def basis_state(bits: str | list[int] | tuple[int, ...]) -> Circuit:
    """Return a circuit that prepares a computational basis state."""

    labels = tuple(int(bit) for bit in bits)
    if any(bit not in {0, 1} for bit in labels):
        raise ValueError("QuantumBridge basis states require 0/1 bits.")
    circuit = Circuit(len(labels))
    for wire, bit in enumerate(labels):
        if bit:
            circuit.x(wire)
    return circuit


def amplitude_encoding(features, normalize: bool = True, name: str = "state_preparation") -> Circuit:
    """Prepare a state from a complete amplitude vector with a native unitary."""

    state = np.asarray(features, dtype=complex).reshape(-1)
    if state.size <= 0 or state.size & (state.size - 1):
        raise ValueError("QuantumBridge amplitude encoding requires a non-empty power-of-two vector.")
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("QuantumBridge amplitude encoding cannot prepare the zero vector.")
    if normalize:
        state = state / norm
    elif not np.isclose(norm, 1.0, atol=1e-10):
        raise ValueError("QuantumBridge amplitude encoding requires normalized data unless normalize=True.")
    num_qubits = int(np.log2(state.size))
    circuit = Circuit(num_qubits)
    circuit.unitary(_unitary_with_first_column(state), tuple(range(num_qubits)), name=name)
    circuit.metadata["state_preparation"] = {
        "method": "amplitude_encoding",
        "num_amplitudes": int(state.size),
        "normalized": True,
    }
    return circuit


def initialize(features, normalize: bool = True) -> Circuit:
    return amplitude_encoding(features, normalize=normalize, name="initialize")


def _unitary_with_first_column(state: np.ndarray) -> np.ndarray:
    columns = [state.astype(complex)]
    dim = state.size
    for index in range(dim):
        vector = np.zeros(dim, dtype=complex)
        vector[index] = 1.0
        for column in columns:
            vector = vector - column * np.vdot(column, vector)
        norm = np.linalg.norm(vector)
        if norm > 1e-12:
            columns.append(vector / norm)
        if len(columns) == dim:
            break
    return np.column_stack(columns)
