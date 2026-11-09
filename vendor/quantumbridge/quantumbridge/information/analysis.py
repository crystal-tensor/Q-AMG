# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

import numpy as np

from quantumbridge.information.density_matrix import DensityMatrix
from quantumbridge.information.statevector import Statevector
from quantumbridge.information.state_dimensions import require_qubits


def _require_qubit_partition(state):
    if isinstance(state,(Statevector,DensityMatrix)):
        require_qubits(state.dims(),"mixed_radix_state_analysis")


def _as_state_data(state) -> np.ndarray:
    return np.asarray(state.data if hasattr(state, "data") else state, dtype=complex)


def _as_density_data(state) -> np.ndarray:
    data = _as_state_data(state)
    if data.ndim == 1:
        return np.outer(data, data.conj())
    if data.ndim != 2 or data.shape[0] != data.shape[1]:
        raise ValueError("QuantumBridge state analysis requires a statevector or square density matrix.")
    return data


def _num_qubits(dimension: int) -> int:
    if dimension <= 0 or dimension & (dimension - 1):
        raise ValueError("QuantumBridge state analysis currently requires qubit dimensions.")
    return int(np.log2(dimension))


def _validate_base(base: float) -> float:
    value = float(base)
    if value <= 0 or np.isclose(value, 1):
        raise ValueError("QuantumBridge entropy logarithm base must be positive and not equal to one.")
    return value


def _validate_qargs(qargs, num_qubits: int) -> tuple[int, ...]:
    normalized = tuple(int(qarg) for qarg in qargs)
    if not normalized:
        raise ValueError("QuantumBridge state analysis qargs cannot be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("QuantumBridge state analysis qargs must be distinct.")
    if any(qarg < 0 or qarg >= num_qubits for qarg in normalized):
        raise ValueError("QuantumBridge state analysis qargs are outside the state.")
    return normalized


def _reduced_density_matrix(rho: np.ndarray, keep: tuple[int, ...]) -> np.ndarray:
    num_qubits = _num_qubits(rho.shape[0])
    keep_set = set(keep)
    trace_out = [wire for wire in range(num_qubits) if wire not in keep_set]
    tensor = rho.reshape([2] * (2 * num_qubits))
    current_qubits = num_qubits
    for wire in reversed(trace_out):
        tensor = np.trace(tensor, axis1=wire, axis2=wire + current_qubits)
        current_qubits -= 1
    remaining = [wire for wire in range(num_qubits) if wire in keep_set]
    if tuple(remaining) != keep:
        position = {wire: index for index, wire in enumerate(remaining)}
        order = [position[wire] for wire in keep]
        tensor = tensor.transpose(order + [index + len(keep) for index in order])
    dimension = 1 << len(keep)
    return tensor.reshape(dimension, dimension)


def shannon_entropy(pvec, base: float = 2) -> float:
    probabilities = np.asarray(pvec, dtype=float)
    if probabilities.ndim != 1:
        raise ValueError("QuantumBridge shannon_entropy requires a one-dimensional probability vector.")
    if np.any(probabilities < -1e-12):
        raise ValueError("QuantumBridge shannon_entropy probabilities cannot be negative.")
    nonzero = probabilities[probabilities > 0]
    if nonzero.size == 0:
        return 0.0
    logarithm_base = _validate_base(base)
    return float(-np.sum(nonzero * np.log(nonzero) / np.log(logarithm_base)))


def entropy(state, base: float = 2) -> float:
    data = _as_state_data(state)
    if data.ndim == 1:
        return 0.0
    rho = _as_density_data(data)
    eigenvalues = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    return shannon_entropy(np.clip(eigenvalues.real, 0.0, None), base=base)


def mutual_information(state, base: float = 2) -> float:
    _require_qubit_partition(state)
    rho = _as_density_data(state)
    if rho.shape != (4, 4):
        raise ValueError("QuantumBridge mutual_information requires a two-qubit bipartite state.")
    reduced_a = _reduced_density_matrix(rho, (0,))
    reduced_b = _reduced_density_matrix(rho, (1,))
    return float(entropy(reduced_a, base) + entropy(reduced_b, base) - entropy(rho, base))


def concurrence(state) -> float:
    _require_qubit_partition(state)
    data = _as_state_data(state)
    if data.ndim == 1:
        if data.size != 4:
            raise ValueError("QuantumBridge concurrence currently requires a two-qubit state.")
        return float(2 * abs(data[0] * data[3] - data[1] * data[2]))

    rho = _as_density_data(data)
    if rho.shape != (4, 4):
        raise ValueError("QuantumBridge mixed-state concurrence requires a two-qubit density matrix.")
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    spin_flip = np.kron(y, y)
    product = rho @ spin_flip @ rho.conj() @ spin_flip
    eigenvalues = np.linalg.eigvals(product)
    roots = np.sort(np.sqrt(np.clip(eigenvalues.real, 0.0, None)))[::-1]
    return float(max(0.0, roots[0] - np.sum(roots[1:])))


def entanglement_of_formation(state) -> float:
    value = float(np.clip(concurrence(state), 0.0, 1.0))
    probability = (1 + np.sqrt(max(0.0, 1 - value * value))) / 2
    return shannon_entropy([probability, 1 - probability], base=2)


def negativity(state, qargs) -> float:
    _require_qubit_partition(state)
    rho = _as_density_data(state)
    num_qubits = _num_qubits(rho.shape[0])
    selected = _validate_qargs(qargs, num_qubits)
    tensor = rho.reshape([2] * (2 * num_qubits))
    axes = list(range(2 * num_qubits))
    for qarg in selected:
        axes[qarg], axes[qarg + num_qubits] = axes[qarg + num_qubits], axes[qarg]
    partial_transpose = tensor.transpose(axes).reshape(rho.shape)
    trace_norm = np.sum(np.linalg.svd(partial_transpose, compute_uv=False))
    return float(max(0.0, (trace_norm - 1) / 2))


def schmidt_decomposition(state, qargs):
    _require_qubit_partition(state)
    vector = _as_state_data(state)
    if vector.ndim != 1:
        raise ValueError("QuantumBridge schmidt_decomposition requires a pure statevector.")
    num_qubits = _num_qubits(vector.size)
    selected = _validate_qargs(qargs, num_qubits)
    complement = tuple(wire for wire in range(num_qubits) if wire not in selected)
    if not complement:
        raise ValueError("QuantumBridge Schmidt partition must leave a non-empty complement.")
    tensor = vector.reshape([2] * num_qubits).transpose(selected + complement)
    matrix = tensor.reshape(1 << len(selected), 1 << len(complement))
    left_vectors, values, right_adjoint = np.linalg.svd(matrix, full_matrices=False)
    decomposition = []
    for index in np.flatnonzero(values > 1e-12):
        decomposition.append(
            (
                float(values[index]),
                Statevector(left_vectors[:, index]),
                Statevector(right_adjoint[index, :]),
            )
        )
    return decomposition


def commutator(a, b):
    return a @ b - b @ a


def anti_commutator(a, b):
    return a @ b + b @ a


def double_commutator(a, b, c, *, commutator: bool = True):
    abc = a @ b @ c
    cba = c @ b @ a
    bac = b @ a @ c
    cab = c @ a @ b
    acb = a @ c @ b
    bca = b @ c @ a
    if commutator:
        return (2 * abc + 2 * cba - bac - cab - acb - bca) / 2
    return (2 * abc - 2 * cba - bac + cab - acb + bca) / 2


__all__ = [
    "anti_commutator",
    "commutator",
    "concurrence",
    "double_commutator",
    "entanglement_of_formation",
    "entropy",
    "mutual_information",
    "negativity",
    "schmidt_decomposition",
    "shannon_entropy",
]
