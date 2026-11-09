# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_upstream_integration_strategy_v0.1.md.

from __future__ import annotations

import numpy as np
from numbers import Integral

from quantumbridge.utils.math import Hamiltonian, PauliString


class QubitHamiltonian:
    def __init__(self, terms, num_qubits=None):
        from quantumbridge.operators import SparsePauliOperator
        if isinstance(terms, SparsePauliOperator):
            if num_qubits is None:
                num_qubits = terms.num_qubits
            terms = [(value, PauliString([(q, char) for q, char in enumerate(label) if char != "I"])) for label, value in terms.terms]
        elif isinstance(terms, Hamiltonian):
            terms = terms.terms
        self._terms = tuple((complex(value), pauli) for value, pauli in terms)
        if any(not np.isfinite(value) or not isinstance(pauli, PauliString) for value, pauli in self._terms):
            raise ValueError("Qubit operator requires finite coefficients and PauliString terms.")
        if any(any(q < 0 for q, _ in pauli.terms) or len({q for q, _ in pauli.terms}) != len(pauli.terms) for _, pauli in self._terms):
            raise ValueError("Qubit operator Pauli wires must be distinct and nonnegative.")
        required = 1 + max((q for _, pauli in self._terms for q, _ in pauli.terms), default=-1)
        if num_qubits is None:
            num_qubits = max(1, required)
        if isinstance(num_qubits, bool) or not isinstance(num_qubits, Integral) or not required <= num_qubits <= 128:
            raise ValueError("Qubit operator width is invalid.")
        self.num_qubits = int(num_qubits)

    @property
    def hamiltonian(self):
        return self.to_quantumbridge()

    @property
    def terms(self):
        return tuple((value.real if abs(value.imag) <= 1e-12 else value, pauli) for value, pauli in self._terms)

    def to_quantumbridge(self) -> Hamiltonian:
        if any(abs(value.imag) > 1e-12 for value, _ in self._terms):
            raise ValueError("Non-Hermitian qubit operator is not a Hamiltonian observable.")
        return Hamiltonian([(value.real, pauli) for value, pauli in self._terms])

    def matrix(self, num_qubits: int | None = None) -> np.ndarray:
        if num_qubits is None:
            num_qubits = self.num_qubits
        if isinstance(num_qubits, bool) or not isinstance(num_qubits, Integral) or num_qubits < self.num_qubits:
            raise ValueError("Requested matrix width cannot omit qubit operator wires.")
        if num_qubits > 10:
            raise MemoryError("Dense qubit operator exceeds the ten-wire budget.")
        from quantumbridge.execution.resource_context import current_resources
        if current_resources() is not None:
            from quantumbridge.utils.math import _sum_pauli_dense
            return _sum_pauli_dense(num_qubits, (("".join(dict(pauli.terms).get(wire, "I") for wire in range(num_qubits)), coefficient)
                for coefficient, pauli in self.terms))
        out = np.zeros((1 << num_qubits, 1 << num_qubits), dtype=complex)
        for coeff, pauli in self.terms:
            out += coeff * pauli.matrix(num_qubits)
        return out

    @classmethod
    def from_pauli_terms(cls, terms):
        return cls([(coeff, PauliString(paulis)) for coeff, paulis in terms])
