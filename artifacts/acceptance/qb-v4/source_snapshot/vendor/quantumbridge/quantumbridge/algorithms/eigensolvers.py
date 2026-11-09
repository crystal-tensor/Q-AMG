# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_upstream_integration_strategy_v0.1.md.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from quantumbridge.chemistry.solvers import ExactDiagonalizationSolver
from quantumbridge.utils.math import Hamiltonian, PauliString


@dataclass(frozen=True)
class EigensolverResult:
    """Native eigensolver result with Qiskit-like attributes."""

    eigenvalues: np.ndarray
    eigenstates: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def eigenvalue(self):
        return self.eigenvalues[0]

    @property
    def eigenstate(self):
        return self.eigenstates[:, 0].copy()

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "NumPyEigensolver",
            "eigenvalues": [_complex_to_dict(value) for value in self.eigenvalues],
            "eigenstates": [[_complex_to_dict(value) for value in column] for column in self.eigenstates.T],
            "metadata": dict(self.metadata),
        }


class NumPyEigensolver:
    """Exact dense-matrix eigensolver for small native operators."""

    mode = "Native Core"

    def __init__(self, k: Optional[int] = None):
        if k is not None and int(k) <= 0:
            raise ValueError("QuantumBridge NumPyEigensolver k must be positive.")
        self.k = None if k is None else int(k)

    def compute_eigenvalues(self, operator) -> EigensolverResult:
        matrix = _operator_to_matrix(operator)
        values, vectors = np.linalg.eigh(matrix) if _is_hermitian(matrix) else np.linalg.eig(matrix)
        order = np.argsort(np.real(values))
        if self.k is not None:
            order = order[: self.k]
        values = values[order]
        vectors = vectors[:, order]
        return EigensolverResult(
            eigenvalues=np.asarray(values, dtype=complex),
            eigenstates=np.asarray(vectors, dtype=complex),
            metadata={
                "mode": self.mode,
                "k": self.k,
                "dimension": matrix.shape[0],
                "hermitian": _is_hermitian(matrix),
            },
        )


def eigensolver_result(operator, k: Optional[int] = None) -> EigensolverResult:
    return NumPyEigensolver(k=k).compute_eigenvalues(operator)


def _operator_to_matrix(operator) -> np.ndarray:
    if isinstance(operator, Hamiltonian):
        return _hamiltonian_matrix(operator)
    if isinstance(operator, PauliString):
        return operator.matrix()
    if hasattr(operator, "matrix") and callable(operator.matrix):
        return np.asarray(operator.matrix(), dtype=complex)
    return np.asarray(operator, dtype=complex)


def _hamiltonian_matrix(hamiltonian: Hamiltonian) -> np.ndarray:
    num_qubits = 1 + max((wire for _, pauli in hamiltonian.terms for wire, _ in pauli.terms), default=0)
    dim = 1 << num_qubits
    matrix = np.zeros((dim, dim), dtype=complex)
    for coeff, pauli in hamiltonian.terms:
        matrix += coeff * pauli.matrix(num_qubits)
    return matrix


def _is_hermitian(matrix: np.ndarray) -> bool:
    return bool(np.allclose(matrix, matrix.conj().T, atol=1e-10))


def _complex_to_dict(value: complex) -> dict[str, float]:
    return {"real": float(np.real(value)), "imag": float(np.imag(value))}


__all__ = ["ExactDiagonalizationSolver", "EigensolverResult", "NumPyEigensolver", "eigensolver_result"]
