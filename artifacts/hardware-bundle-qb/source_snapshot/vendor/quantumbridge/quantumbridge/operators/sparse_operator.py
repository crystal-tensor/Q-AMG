# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number, Real
from typing import Iterable

from quantumbridge.utils.math import Hamiltonian, PauliString


def _multiply_labels(left: str, right: str) -> tuple[complex, str]:
    from quantumbridge.operators.pauli import Pauli

    if len(left) != len(right):
        raise ValueError("QuantumBridge sparse Pauli compose requires equal label widths.")
    phase = 1 + 0j
    label = []
    for a, b in zip(left, right):
        local_phase, local_pauli = Pauli(a).multiply(Pauli(b))
        phase *= local_phase
        label.append(local_pauli.label)
    return phase, "".join(label)


def _embed_label(label: str, qargs: tuple[int, ...], num_qubits: int) -> str:
    if len(label) != len(qargs):
        raise ValueError("QuantumBridge sparse Pauli qargs width must match embedded label.")
    out = ["I"] * num_qubits
    for char, qarg in zip(label, qargs):
        out[qarg] = char
    return "".join(out)


@dataclass(frozen=True)
class SparsePauliOperator:
    terms: tuple[tuple[str, complex], ...]

    def __init__(self, terms: Iterable[tuple[str, Number]]):
        normalized = []
        width = None
        for label, coeff in terms:
            label = label.upper()
            if any(ch not in "IXYZ" for ch in label):
                raise ValueError("QuantumBridge sparse Pauli labels may contain only I, X, Y, Z.")
            if width is None:
                width = len(label)
            elif len(label) != width:
                raise ValueError("QuantumBridge sparse Pauli terms must have equal label length.")
            normalized.append((label, complex(coeff)))
        object.__setattr__(self, "terms", tuple(normalized))

    @classmethod
    def from_list(cls, terms: Iterable[tuple[str, Number]]) -> "SparsePauliOperator":
        return cls(terms)

    @classmethod
    def from_operator(cls, operator) -> "SparsePauliOperator":
        import numpy as np

        matrix = operator.data if hasattr(operator, "data") else np.asarray(operator, dtype=complex)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("QuantumBridge SparsePauliOperator.from_operator requires a square matrix.")
        dim = matrix.shape[0]
        if dim == 0 or dim & (dim - 1):
            raise ValueError("QuantumBridge operator dimension must be a power of two.")
        width = int(np.log2(dim))
        basis = ["I", "X", "Y", "Z"]
        terms = []
        for indices in np.ndindex(*(4,) * width):
            label = "".join(basis[index] for index in indices)
            pauli = SparsePauliOperator([(label, 1)]).matrix()
            coeff = np.trace(pauli.conj().T @ matrix) / dim
            if abs(coeff) > 1e-12:
                terms.append((label, coeff))
        return cls(terms)

    def to_list(self) -> list[tuple[str, complex]]:
        return list(self.terms)

    @property
    def size(self) -> int:
        return len(self.terms)

    @property
    def num_qubits(self) -> int:
        return 0 if not self.terms else len(self.terms[0][0])

    @property
    def paulis(self):
        from quantumbridge.operators.pauli_list import PauliList

        if not self.terms:
            return None
        return PauliList(label for label, _ in self.terms)

    @property
    def coeffs(self):
        import numpy as np

        return np.array([coeff for _, coeff in self.terms], dtype=complex)

    def copy(self) -> "SparsePauliOperator":
        return SparsePauliOperator(self.terms)

    def to_hamiltonian(self) -> Hamiltonian:
        converted = []
        for label, coeff in self.terms:
            if abs(coeff.imag) > 1e-12:
                raise ValueError("QuantumBridge Hamiltonian conversion requires real Pauli coefficients.")
            pauli_terms = [(wire, char) for wire, char in enumerate(label) if char != "I"]
            converted.append((float(coeff.real), PauliString(pauli_terms)))
        return Hamiltonian(converted)

    def __add__(self, other: "SparsePauliOperator") -> "SparsePauliOperator":
        if self.terms and other.terms and self.num_qubits != other.num_qubits:
            raise ValueError("QuantumBridge sparse Pauli addition requires equal label widths.")
        return SparsePauliOperator(self.terms + other.terms)

    def __sub__(self, other: "SparsePauliOperator") -> "SparsePauliOperator":
        return self + (-1 * other)

    def __neg__(self) -> "SparsePauliOperator":
        return -1 * self

    def __mul__(self, scalar: Number) -> "SparsePauliOperator":
        return SparsePauliOperator((label, coeff * scalar) for label, coeff in self.terms)

    __rmul__ = __mul__

    def __truediv__(self, scalar: Number) -> "SparsePauliOperator":
        return SparsePauliOperator((label, coeff / scalar) for label, coeff in self.terms)

    def simplify(self, atol: float = 1e-12) -> "SparsePauliOperator":
        combined: dict[str, complex] = {}
        for label, coeff in self.terms:
            combined[label] = combined.get(label, 0j) + coeff
        return SparsePauliOperator((label, coeff) for label, coeff in combined.items() if abs(coeff) > atol)

    def chop(self, atol: float = 1e-12) -> "SparsePauliOperator":
        return SparsePauliOperator((label, coeff) for label, coeff in self.terms if abs(coeff) > atol)

    def adjoint(self) -> "SparsePauliOperator":
        return SparsePauliOperator((label, coeff.conjugate()) for label, coeff in self.terms)

    def tensor(self, other: "SparsePauliOperator") -> "SparsePauliOperator":
        return SparsePauliOperator((a + b, ca * cb) for a, ca in self.terms for b, cb in other.terms)

    def expand(self, other: "SparsePauliOperator") -> "SparsePauliOperator":
        return other.tensor(self)

    def compose(self, other: "SparsePauliOperator", qargs=None, front: bool = False) -> "SparsePauliOperator":
        other = other if isinstance(other, SparsePauliOperator) else SparsePauliOperator(other)
        if not self.terms:
            return SparsePauliOperator([])
        if not other.terms:
            return SparsePauliOperator([])
        other_terms = other.terms
        if qargs is not None:
            qargs_tuple = tuple(int(qarg) for qarg in qargs)
            if len(set(qargs_tuple)) != len(qargs_tuple):
                raise ValueError("QuantumBridge sparse Pauli qargs must be distinct.")
            if any(qarg < 0 or qarg >= self.num_qubits for qarg in qargs_tuple):
                raise ValueError("QuantumBridge sparse Pauli qargs are outside this operator.")
            other_terms = tuple((_embed_label(label, qargs_tuple, self.num_qubits), coeff) for label, coeff in other_terms)
        elif self.num_qubits != other.num_qubits:
            raise ValueError("QuantumBridge sparse Pauli compose requires equal label widths unless qargs are provided.")
        terms = []
        for left_label, left_coeff in self.terms:
            for right_label, right_coeff in other_terms:
                first_label, first_coeff = (right_label, right_coeff) if front else (left_label, left_coeff)
                second_label, second_coeff = (left_label, left_coeff) if front else (right_label, right_coeff)
                phase, label = _multiply_labels(first_label, second_label)
                terms.append((label, first_coeff * second_coeff * phase))
        return SparsePauliOperator(terms).simplify()

    def dot(self, other: "SparsePauliOperator", qargs=None) -> "SparsePauliOperator":
        return self.compose(other, qargs=qargs)

    def power(self, n: int) -> "SparsePauliOperator":
        if not isinstance(n, int) or n < 0:
            raise ValueError("QuantumBridge sparse Pauli power requires a non-negative integer exponent.")
        if n == 0:
            if not self.terms:
                return SparsePauliOperator([])
            return SparsePauliOperator([("I" * self.num_qubits, 1)])
        out = self
        for _ in range(n - 1):
            out = out.compose(self)
        return out

    def commutator(self, other: "SparsePauliOperator") -> "SparsePauliOperator":
        return (self.compose(other) - other.compose(self)).simplify()

    def matrix(self):
        from quantumbridge.execution.resource_context import current_resources
        if current_resources() is not None:
            from quantumbridge.utils.math import _sum_pauli_dense
            return _sum_pauli_dense(self.num_qubits, self.terms)
        import numpy as np

        if not self.terms:
            return np.array([[0]], dtype=complex)
        width = len(self.terms[0][0])
        out = np.zeros((2 ** width, 2 ** width), dtype=complex)
        for label, coeff in self.terms:
            if len(label) != width:
                raise ValueError("QuantumBridge sparse Pauli terms must have equal label length.")
            pauli = PauliString([(wire, char) for wire, char in enumerate(label) if char != "I"])
            out += coeff * pauli.matrix(num_qubits=width)
        return out

    def to_matrix(self):
        return self.matrix()

    def to_operator(self):
        from quantumbridge.information.operator import Operator

        return Operator(self.matrix())

    def expectation_value(self, state) -> complex:
        matrix = self.matrix()
        data = state.data if hasattr(state, "data") else state
        import numpy as np

        arr = np.asarray(data, dtype=complex)
        if arr.ndim == 1:
            return np.vdot(arr, matrix @ arr)
        if arr.ndim == 2:
            return np.trace(arr @ matrix)
        raise ValueError("QuantumBridge sparse Pauli expectation requires statevector or density matrix data.")

    def equiv(self, other, atol: float = 1e-12) -> bool:
        import numpy as np

        other_op = other if isinstance(other, SparsePauliOperator) else SparsePauliOperator(other)
        return bool(np.allclose(self.simplify(atol).matrix(), other_op.simplify(atol).matrix(), atol=atol))

    def __matmul__(self, other: "SparsePauliOperator") -> "SparsePauliOperator":
        return self.compose(other)

    def __eq__(self, other) -> bool:
        try:
            return self.equiv(other)
        except Exception:
            return False


SparsePauliOp = SparsePauliOperator
