# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from math import pi
from typing import Iterable, Sequence

from quantumbridge.core import Circuit


def two_local(
    num_qubits: int,
    parameters: Sequence[Sequence[float]],
    rotation_blocks: Sequence[str] = ("ry", "rz"),
    entanglement: str = "linear",
) -> Circuit:
    circuit = Circuit(num_qubits)
    for layer in parameters:
        expected = num_qubits * len(rotation_blocks)
        if len(layer) != expected:
            raise ValueError("QuantumBridge two_local parameter layer has the wrong width.")
        offset = 0
        for gate in rotation_blocks:
            for wire in range(num_qubits):
                getattr(circuit, gate)(layer[offset], wire)
                offset += 1
        _entangle(circuit, entanglement)
    circuit.metadata["template"] = {"name": "two_local", "entanglement": entanglement, "rotation_blocks": list(rotation_blocks)}
    return circuit


def real_amplitudes(num_qubits: int, parameters: Sequence[Sequence[float]], entanglement: str = "linear") -> Circuit:
    return two_local(num_qubits, parameters, rotation_blocks=("ry",), entanglement=entanglement)


def efficient_su2(num_qubits: int, parameters: Sequence[Sequence[float]], entanglement: str = "linear") -> Circuit:
    return two_local(num_qubits, parameters, rotation_blocks=("ry", "rz"), entanglement=entanglement)


def zz_feature_map(features: Sequence[float], reps: int = 1, entanglement: str = "linear") -> Circuit:
    num_qubits = len(features)
    circuit = Circuit(num_qubits)
    for _ in range(reps):
        for wire, value in enumerate(features):
            circuit.h(wire).phase(2.0 * float(value), wire)
        for left, right in _entanglement_pairs(num_qubits, entanglement):
            circuit.cx(left, right)
            circuit.rz(2.0 * float(features[left]) * float(features[right]), right)
            circuit.cx(left, right)
    circuit.metadata["template"] = {"name": "zz_feature_map", "reps": reps, "entanglement": entanglement}
    return circuit


def qft(num_qubits: int, inverse: bool = False, do_swaps: bool = True) -> Circuit:
    circuit = Circuit(num_qubits)
    for target in range(num_qubits):
        circuit.h(target)
        for control in range(target + 1, num_qubits):
            circuit.cp(pi / (2 ** (control - target)), control, target)
    if do_swaps:
        for left in range(num_qubits // 2):
            circuit.swap(left, num_qubits - 1 - left)
    circuit.metadata["template"] = {"name": "qft", "inverse": inverse, "do_swaps": do_swaps}
    return circuit.inverse() if inverse else circuit


def _entangle(circuit: Circuit, entanglement: str) -> None:
    for left, right in _entanglement_pairs(circuit.num_qubits, entanglement):
        circuit.cx(left, right)


def _entanglement_pairs(num_qubits: int, entanglement: str) -> Iterable[tuple[int, int]]:
    if entanglement == "linear":
        return tuple((wire, wire + 1) for wire in range(num_qubits - 1))
    if entanglement == "circular":
        return tuple((wire, (wire + 1) % num_qubits) for wire in range(num_qubits))
    if entanglement == "full":
        return tuple((left, right) for left in range(num_qubits) for right in range(left + 1, num_qubits))
    raise ValueError("QuantumBridge entanglement must be linear, circular, or full.")
