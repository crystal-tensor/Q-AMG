# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

import numpy as np

from quantumbridge.information.density_matrix import DensityMatrix
from quantumbridge.information.channels import Stinespring
from quantumbridge.information.operator import Operator
from quantumbridge.information.statevector import Statevector
from quantumbridge.operators import Pauli, PauliList


def _dimension(dims: int | tuple[int, ...] | list[int]) -> int:
    if isinstance(dims, int):
        dim = dims
    else:
        dim = int(np.prod(tuple(dims)))
    if dim <= 0:
        raise ValueError("QuantumBridge random dimensions must be positive.")
    return dim


def random_statevector(dims: int | tuple[int, ...] | list[int], seed: int | None = None) -> Statevector:
    dim = _dimension(dims)
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return Statevector(vector, normalize=True)


def random_density_matrix(dims: int | tuple[int, ...] | list[int], seed: int | None = None, rank: int | None = None) -> DensityMatrix:
    dim = _dimension(dims)
    rank = dim if rank is None else int(rank)
    if not 1 <= rank <= dim:
        raise ValueError("QuantumBridge random_density_matrix rank must be between 1 and dimension.")
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dim, rank)) + 1j * rng.normal(size=(dim, rank))
    rho = matrix @ matrix.conj().T
    rho = rho / np.trace(rho)
    return DensityMatrix(rho)


def random_unitary(dims: int | tuple[int, ...] | list[int], seed: int | None = None) -> Operator:
    dim = _dimension(dims)
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(matrix)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) == 0, 1, phases / np.abs(phases))
    return Operator(q * phases)


def random_hermitian(
    dims: int | tuple[int, ...] | list[int],
    traceless: bool = False,
    seed: int | np.random.Generator | None = None,
) -> Operator:
    dim = _dimension(dims)
    rng = np.random.default_rng(seed)
    matrix = np.zeros((dim, dim), dtype=complex)
    if not traceless:
        np.fill_diagonal(matrix, rng.normal(size=dim))
    pair_count = dim * (dim - 1) // 2
    real = rng.normal(size=pair_count)
    imaginary = rng.normal(size=pair_count)
    rows, columns = np.tril_indices(dim, k=-1)
    matrix[rows, columns] = (real + 1j * imaginary) / 2
    matrix[columns, rows] = matrix[rows, columns].conj()
    return Operator(matrix)


def pauli_basis(num_qubits: int, weight: bool = False) -> PauliList:
    if not isinstance(num_qubits, int) or num_qubits < 1:
        raise ValueError("QuantumBridge pauli_basis requires a positive number of qubits.")
    labels = [""]
    for _ in range(num_qubits):
        labels = [prefix + suffix for prefix in labels for suffix in "IXYZ"]
    if weight:
        labels.sort(key=lambda label: sum(char != "I" for char in label))
    return PauliList(labels)


def random_pauli(
    num_qubits: int,
    group_phase: bool = False,
    seed: int | np.random.Generator | None = None,
) -> Pauli:
    if not isinstance(num_qubits, int) or num_qubits < 0:
        raise ValueError("QuantumBridge random_pauli requires a non-negative integer qubit count.")
    rng = np.random.default_rng(seed)
    z = rng.integers(2, size=num_qubits, dtype=bool)
    x = rng.integers(2, size=num_qubits, dtype=bool)
    phase = int(rng.integers(4)) if group_phase else 0
    return Pauli((z, x, phase))


def random_pauli_list(
    num_qubits: int,
    size: int = 1,
    seed: int | np.random.Generator | None = None,
    phase: bool = True,
) -> PauliList:
    if not isinstance(num_qubits, int) or num_qubits < 0:
        raise ValueError("QuantumBridge random_pauli_list requires a non-negative integer qubit count.")
    if not isinstance(size, int) or size < 0:
        raise ValueError("QuantumBridge random_pauli_list size must be a non-negative integer.")
    rng = np.random.default_rng(seed)
    bits = rng.integers(2, size=(2, size, num_qubits))
    phases = rng.integers(4, size=size) if phase else np.zeros(size, dtype=int)
    return PauliList.from_symplectic(bits[0], bits[1], phases)


def random_clifford(
    num_qubits: int,
    seed: int | np.random.Generator | None = None,
):
    from quantumbridge.information.clifford import Clifford, _gf2_basis, _symplectic_product

    if not isinstance(num_qubits, int) or num_qubits < 0:
        raise ValueError("QuantumBridge random_clifford requires a non-negative integer qubit count.")
    rng = np.random.default_rng(seed)
    basis = np.eye(2 * num_qubits, dtype=bool)
    destabilizers = []
    stabilizers = []
    for _ in range(num_qubits):
        while True:
            coefficients = rng.integers(2, size=len(basis), dtype=bool)
            if np.any(coefficients):
                first = np.logical_xor.reduce(basis[coefficients], axis=0)
                break
        while True:
            coefficients = rng.integers(2, size=len(basis), dtype=bool)
            if not np.any(coefficients):
                continue
            second = np.logical_xor.reduce(basis[coefficients], axis=0)
            if _symplectic_product(first, second):
                break
        destabilizers.append(first)
        stabilizers.append(second)
        # Project the remaining space onto the symplectic complement of this pair.
        projected = basis.copy()
        for index, row in enumerate(basis):
            if _symplectic_product(row, second):
                projected[index] ^= first
            if _symplectic_product(row, first):
                projected[index] ^= second
        basis = _gf2_basis(projected)
    if num_qubits == 0:
        return Clifford.identity(0)
    matrix = np.array(destabilizers + stabilizers, dtype=bool)
    phases = rng.integers(2, size=2 * num_qubits, dtype=bool)
    return Clifford(np.column_stack([matrix, phases]))


def random_quantum_channel(
    input_dims: int | tuple[int, ...] | list[int] | None = None,
    output_dims: int | tuple[int, ...] | list[int] | None = None,
    rank: int | None = None,
    seed: int | np.random.Generator | None = None,
) -> Stinespring:
    input_dim = _dimension(2 if input_dims is None else input_dims)
    output_dim = _dimension(input_dim if output_dims is None else output_dims)
    rank = input_dim * output_dim if rank is None else int(rank)
    if rank < 1 or rank > input_dim * output_dim:
        raise ValueError("QuantumBridge channel rank must be between 1 and input_dim * output_dim.")
    row_count = output_dim * rank
    if row_count < input_dim:
        raise ValueError("QuantumBridge channel dimensions and rank cannot form an isometry.")

    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(row_count, input_dim)) + 1j * rng.normal(
        size=(row_count, input_dim)
    )
    isometry, triangular = np.linalg.qr(matrix, mode="reduced")
    phases = np.diag(triangular)
    phases = np.where(np.abs(phases) == 0, 1, phases / np.abs(phases))
    isometry = isometry * phases
    return Stinespring._from_matrix(isometry, input_dim, output_dim)
