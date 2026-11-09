# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from numbers import Integral

import numpy as np

from quantumbridge.operators.pauli import Pauli


class PauliList:
    """A phase-preserving list of Pauli operators in little-endian symplectic order."""

    def __init__(self, data):
        if isinstance(data, PauliList):
            self._paulis = [pauli.copy() for pauli in data._paulis]
            self._num_qubits = data.num_qubits
            return
        if isinstance(data, (str, Pauli)):
            data = [data]
        paulis = [Pauli(value) for value in data]
        widths = {pauli.num_qubits for pauli in paulis}
        if len(widths) > 1:
            raise ValueError("QuantumBridge PauliList labels must have equal length.")
        self._paulis = paulis
        self._num_qubits = next(iter(widths), 0)

    @classmethod
    def _from_paulis(cls, paulis, num_qubits: int) -> "PauliList":
        result = cls(paulis)
        if result.size and result.num_qubits != num_qubits:
            raise ValueError("QuantumBridge PauliList internal width mismatch.")
        result._num_qubits = num_qubits
        return result

    @classmethod
    def from_symplectic(cls, z, x, phase=0) -> "PauliList":
        z_array = np.asarray(z, dtype=bool)
        x_array = np.asarray(x, dtype=bool)
        if z_array.ndim != 2 or z_array.shape != x_array.shape:
            raise ValueError("QuantumBridge PauliList symplectic arrays must have equal two-dimensional shapes.")
        phases = np.asarray(0 if phase is None else phase)
        if not np.issubdtype(phases.dtype, np.integer):
            raise ValueError("QuantumBridge PauliList phases must be integers.")
        try:
            phases = np.broadcast_to(phases, (len(z_array),))
        except ValueError as exc:
            raise ValueError("QuantumBridge PauliList phase count must match its rows.") from exc
        return cls._from_paulis(
            [Pauli((z_row, x_row, int(exponent))) for z_row, x_row, exponent in zip(z_array, x_array, phases)],
            z_array.shape[1],
        )

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.to_labels())

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def size(self) -> int:
        return len(self._paulis)

    @property
    def shape(self) -> tuple[int, int]:
        return self.size, self.num_qubits

    @property
    def dim(self) -> tuple[int, int]:
        dimension = 1 << self.num_qubits
        return dimension, dimension

    @property
    def phase(self) -> np.ndarray:
        return np.array([pauli.phase for pauli in self._paulis], dtype=int)

    @phase.setter
    def phase(self, values) -> None:
        array = np.asarray(values)
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError("QuantumBridge PauliList phases must be integers.")
        array = np.broadcast_to(array, (self.size,))
        for pauli, phase in zip(self._paulis, array):
            pauli.phase = int(phase)

    @property
    def z(self) -> np.ndarray:
        return np.array([pauli.z for pauli in self._paulis], dtype=bool).reshape(self.shape)

    @z.setter
    def z(self, values) -> None:
        updated = PauliList.from_symplectic(values, self.x, self.phase)
        if updated.shape != self.shape:
            raise ValueError("QuantumBridge PauliList z data must preserve its shape.")
        self._paulis = updated._paulis

    @property
    def x(self) -> np.ndarray:
        return np.array([pauli.x for pauli in self._paulis], dtype=bool).reshape(self.shape)

    @x.setter
    def x(self, values) -> None:
        updated = PauliList.from_symplectic(self.z, values, self.phase)
        if updated.shape != self.shape:
            raise ValueError("QuantumBridge PauliList x data must preserve its shape.")
        self._paulis = updated._paulis

    def input_dims(self, qargs=None) -> tuple[int, ...]:
        if qargs is None:
            return (2,) * self.num_qubits
        return Pauli("I" * self.num_qubits).input_dims(qargs)

    output_dims = input_dims

    def copy(self) -> "PauliList":
        return PauliList(self)

    def to_symplectic(self) -> tuple[np.ndarray, np.ndarray]:
        return self.z, self.x

    def to_labels(self, array: bool = False):
        labels = [pauli.to_label() for pauli in self._paulis]
        return np.array(labels, dtype=str) if array else labels

    def to_matrix(self, sparse: bool = False, array: bool = False):
        matrices = [pauli.to_matrix(sparse=sparse) for pauli in self._paulis]
        if array:
            if sparse:
                raise ValueError("QuantumBridge PauliList cannot combine sparse matrices into a dense array.")
            dimension = 1 << self.num_qubits
            return np.array(matrices, dtype=complex).reshape(self.size, dimension, dimension)
        return matrices

    def label_iter(self):
        return iter(self.to_labels())

    def matrix_iter(self, sparse: bool = False):
        return (pauli.to_matrix(sparse=sparse) for pauli in self._paulis)

    def _broadcast(self, other):
        operand = other if isinstance(other, PauliList) else PauliList(other)
        if self.size == operand.size:
            pairs = list(zip(self._paulis, operand._paulis))
        elif self.size == 1:
            pairs = [(self._paulis[0], pauli) for pauli in operand._paulis]
        elif operand.size == 1:
            pairs = [(pauli, operand._paulis[0]) for pauli in self._paulis]
        else:
            raise ValueError("QuantumBridge PauliList operands require equal row counts or a one-row operand.")
        return operand, pairs

    def compose(self, other, qargs=None, front: bool = False, inplace: bool = False) -> "PauliList":
        operand, pairs = self._broadcast(other)
        if qargs is None and self.num_qubits != operand.num_qubits:
            raise ValueError("QuantumBridge PauliList composition requires equal widths unless qargs are provided.")
        result = PauliList._from_paulis(
            [left.compose(right, qargs=qargs, front=front) for left, right in pairs],
            self.num_qubits,
        )
        if inplace:
            self._paulis = result._paulis
            return self
        return result

    def dot(self, other, qargs=None, inplace: bool = False) -> "PauliList":
        return self.compose(other, qargs=qargs, front=True, inplace=inplace)

    def tensor(self, other) -> "PauliList":
        operand = other if isinstance(other, PauliList) else PauliList(other)
        return PauliList._from_paulis(
            [left.tensor(right) for left in self._paulis for right in operand._paulis],
            self.num_qubits + operand.num_qubits,
        )

    def expand(self, other) -> "PauliList":
        operand = PauliList(other)
        if operand.size not in (1, self.size):
            raise ValueError("QuantumBridge PauliList expand requires a one-row or equal-length operand.")
        return operand.tensor(self)

    def commutes(self, other, qargs=None) -> np.ndarray:
        operand, pairs = self._broadcast(other)
        if qargs is None and self.num_qubits != operand.num_qubits:
            raise ValueError("QuantumBridge PauliList commutation requires equal widths.")
        return np.array([left.commutes(right, qargs=qargs) for left, right in pairs], dtype=bool)

    def anticommutes(self, other, qargs=None) -> np.ndarray:
        return np.logical_not(self.commutes(other, qargs=qargs))

    def evolve(self, other, qargs=None, frame: str = "h") -> "PauliList":
        from quantumbridge.information.clifford import Clifford

        if frame not in {"h", "s"}:
            raise ValueError("QuantumBridge Pauli evolution frame must be h or s.")
        clifford = other if isinstance(other, Clifford) else Clifford(other)
        if qargs is not None:
            clifford = clifford._embedded(qargs, self.num_qubits)
        if frame == "h":
            clifford = clifford.adjoint()
        return PauliList._from_paulis(
            [clifford.conjugate_pauli(pauli) for pauli in self._paulis],
            self.num_qubits,
        )

    def commutation_matrix(self, other) -> np.ndarray:
        """Return all pairwise commutation relations without row broadcasting."""
        operand = other if isinstance(other, PauliList) else PauliList(other)
        if self.num_qubits != operand.num_qubits:
            raise ValueError("QuantumBridge PauliList commutation requires equal widths.")
        return np.array(
            [[left.commutes(right) for right in operand._paulis] for left in self._paulis],
            dtype=bool,
        ).reshape(self.size, operand.size)

    def commutes_with_all(self, other) -> np.ndarray:
        return np.flatnonzero(np.all(self.commutation_matrix(other), axis=1))

    def anticommutes_with_all(self, other) -> np.ndarray:
        return np.flatnonzero(np.all(~self.commutation_matrix(other), axis=1))

    def _map(self, name: str) -> "PauliList":
        return PauliList._from_paulis(
            [getattr(pauli, name)() for pauli in self._paulis],
            self.num_qubits,
        )

    def adjoint(self) -> "PauliList":
        return self._map("adjoint")

    inverse = adjoint

    def conjugate(self) -> "PauliList":
        return self._map("conjugate")

    def transpose(self) -> "PauliList":
        return self._map("transpose")

    def power(self, n: int) -> "PauliList":
        return PauliList._from_paulis([pauli.power(n) for pauli in self._paulis], self.num_qubits)

    def equiv(self, other) -> np.ndarray:
        operand, pairs = self._broadcast(other)
        if self.num_qubits != operand.num_qubits:
            raise ValueError("QuantumBridge PauliList equivalence requires equal widths.")
        return np.array([left.equiv(right) for left, right in pairs], dtype=bool)

    def weight(self) -> np.ndarray:
        return np.count_nonzero(self.z | self.x, axis=1)

    def argsort(self, weight: bool = False, phase: bool = False) -> np.ndarray:
        weights = self.weight()
        return np.array(
            sorted(
                range(self.size),
                key=lambda index: (
                    self._paulis[index].phase if phase else 0,
                    int(weights[index]) if weight else 0,
                    self._paulis[index]._body,
                    self._paulis[index].phase,
                ),
            ),
            dtype=int,
        )

    def sort(self, weight: bool = False, phase: bool = False) -> "PauliList":
        return self[self.argsort(weight=weight, phase=phase)]

    def unique(self, return_index: bool = False, return_counts: bool = False):
        positions = {}
        indices = []
        counts = []
        for index, label in enumerate(self.to_labels()):
            if label not in positions:
                positions[label] = len(indices)
                indices.append(index)
                counts.append(0)
            counts[positions[label]] += 1
        result = self[indices]
        extra = []
        if return_index:
            extra.append(np.array(indices, dtype=int))
        if return_counts:
            extra.append(np.array(counts, dtype=int))
        return (result, *extra) if extra else result

    def group_commuting(self, qubit_wise: bool = False) -> list["PauliList"]:
        if qubit_wise:
            conflicts = np.zeros((self.size, self.size), dtype=bool)
            for row, left in enumerate(self._paulis):
                for column, right in enumerate(self._paulis):
                    conflicts[row, column] = any(
                        a != b and a != "I" and b != "I"
                        for a, b in zip(left._body, right._body)
                    )
        else:
            conflicts = ~self.commutation_matrix(self)
        groups = []
        for index in sorted(range(self.size), key=lambda value: -np.count_nonzero(conflicts[value])):
            for group in groups:
                if not np.any(conflicts[index, group]):
                    group.append(index)
                    break
            else:
                groups.append([index])
        return [self[sorted(group)] for group in groups]

    def group_qubit_wise_commuting(self) -> list["PauliList"]:
        return self.group_commuting(qubit_wise=True)

    def repeat(self, repeats: int) -> "PauliList":
        if not isinstance(repeats, Integral) or repeats < 0:
            raise ValueError("QuantumBridge PauliList repeat count must be a non-negative integer.")
        return PauliList._from_paulis(
            [pauli for pauli in self._paulis for _ in range(repeats)],
            self.num_qubits,
        )

    def delete(self, ind, qubit: bool = False) -> "PauliList":
        if qubit:
            selected = np.atleast_1d(ind)
            paulis = [pauli.delete(ind) for pauli in self._paulis]
            return PauliList._from_paulis(paulis, self.num_qubits - len(selected))
        retained = np.delete(np.arange(self.size), ind)
        return self[retained]

    def insert(self, ind: int, value, qubit: bool = False) -> "PauliList":
        operand = value if isinstance(value, PauliList) else PauliList(value)
        if qubit:
            _, pairs = self._broadcast(operand)
            return PauliList._from_paulis(
                [left.insert(ind, right) for left, right in pairs],
                self.num_qubits + operand.num_qubits,
            )
        if operand.num_qubits != self.num_qubits:
            raise ValueError("QuantumBridge PauliList insertion requires equal widths.")
        if not isinstance(ind, Integral) or not 0 <= ind <= self.size:
            raise ValueError("QuantumBridge PauliList insertion index is outside the list.")
        return PauliList._from_paulis(
            self._paulis[:ind] + operand._paulis + self._paulis[ind:],
            self.num_qubits,
        )

    def __getitem__(self, key):
        if isinstance(key, Integral):
            return self._paulis[int(key)].copy()
        selected = np.atleast_1d(np.arange(self.size)[key]).tolist()
        return PauliList._from_paulis([self._paulis[index] for index in selected], self.num_qubits)

    def __setitem__(self, key, value) -> None:
        indices = np.atleast_1d(np.arange(self.size)[key]).tolist()
        replacement = PauliList(value)
        if replacement.num_qubits != self.num_qubits or replacement.size not in (1, len(indices)):
            raise ValueError("QuantumBridge PauliList assignment has incompatible shape.")
        for offset, index in enumerate(indices):
            self._paulis[index] = replacement._paulis[0 if replacement.size == 1 else offset].copy()

    def __iter__(self):
        return (pauli.copy() for pauli in self._paulis)

    def __len__(self) -> int:
        return self.size

    def __matmul__(self, other) -> "PauliList":
        return self.dot(other)

    def __xor__(self, other) -> "PauliList":
        return self.tensor(other)

    def __add__(self, other) -> "PauliList":
        return self.insert(self.size, other)

    def __mul__(self, scalar) -> "PauliList":
        return PauliList._from_paulis([scalar * pauli for pauli in self._paulis], self.num_qubits)

    __rmul__ = __mul__

    def __neg__(self) -> "PauliList":
        return -1 * self

    def __eq__(self, other) -> bool:
        try:
            operand = PauliList(other)
        except (TypeError, ValueError):
            return False
        return self.shape == operand.shape and self.to_labels() == operand.to_labels()

    def __repr__(self) -> str:
        return f"PauliList({self.to_labels()!r})"
