# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from math import ceil, log2

import numpy as np

from quantumbridge.core import Circuit
from quantumbridge.utils.math import bit_at, set_bit


def weighted_adder(num_state_qubits: int, weights: list[int] | tuple[int, ...], num_sum_qubits: int | None = None) -> Circuit:
    """Return a reversible modular weighted-adder circuit."""

    if len(weights) != num_state_qubits:
        raise ValueError("QuantumBridge weighted_adder requires one weight per state qubit.")
    if any(int(weight) < 0 for weight in weights):
        raise ValueError("QuantumBridge weighted_adder supports non-negative integer weights.")
    max_sum = sum(int(weight) for weight in weights)
    if num_sum_qubits is None:
        num_sum_qubits = max(1, ceil(log2(max_sum + 1))) if max_sum else 1
    total = num_state_qubits + num_sum_qubits
    matrix = np.zeros((1 << total, 1 << total), dtype=complex)
    modulus = 1 << num_sum_qubits
    for basis in range(1 << total):
        weighted = 0
        for wire, weight in enumerate(weights):
            weighted += int(weight) * bit_at(basis, wire, total)
        current = _read_register(basis, range(num_state_qubits, total), total)
        updated = (current + weighted) % modulus
        dest = _write_register(basis, range(num_state_qubits, total), updated, total)
        matrix[dest, basis] = 1.0
    circuit = Circuit(total)
    circuit.unitary(matrix, tuple(range(total)), name="weighted_adder")
    circuit.metadata["arithmetic"] = {
        "name": "weighted_adder",
        "num_state_qubits": num_state_qubits,
        "num_sum_qubits": num_sum_qubits,
        "weights": [int(weight) for weight in weights],
    }
    return circuit


def integer_comparator(num_state_qubits: int, value: int, geq: bool = True) -> Circuit:
    """Return a reversible comparator that toggles a flag qubit."""

    total = num_state_qubits + 1
    matrix = np.zeros((1 << total, 1 << total), dtype=complex)
    for basis in range(1 << total):
        state_value = _read_register(basis, range(num_state_qubits), total)
        marked = state_value >= int(value) if geq else state_value < int(value)
        dest = basis
        if marked:
            flag = bit_at(basis, num_state_qubits, total)
            dest = set_bit(basis, num_state_qubits, 1 - flag, total)
        matrix[dest, basis] = 1.0
    circuit = Circuit(total)
    circuit.unitary(matrix, tuple(range(total)), name="integer_comparator")
    circuit.metadata["arithmetic"] = {
        "name": "integer_comparator",
        "num_state_qubits": num_state_qubits,
        "value": int(value),
        "geq": bool(geq),
    }
    return circuit


def _read_register(basis: int, wires, num_qubits: int) -> int:
    value = 0
    for wire in wires:
        value = (value << 1) | bit_at(basis, wire, num_qubits)
    return value


def _write_register(basis: int, wires, value: int, num_qubits: int) -> int:
    out = basis
    wires_tuple = tuple(wires)
    for offset, wire in enumerate(reversed(wires_tuple)):
        out = set_bit(out, wire, (value >> offset) & 1, num_qubits)
    return out
