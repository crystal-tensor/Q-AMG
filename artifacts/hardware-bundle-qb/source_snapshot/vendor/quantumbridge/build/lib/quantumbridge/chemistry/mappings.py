# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_upstream_integration_strategy_v0.1.md.

from __future__ import annotations

from quantumbridge.utils.math import Hamiltonian, PauliString

from .fermion import FermionicOp, _tokens
from .qubit_hamiltonian import QubitHamiltonian


class JordanWignerMapper:
    mode = "Native Core"

    def map(self, operator: FermionicOp, *, max_terms=100000) -> QubitHamiltonian:
        from numbers import Integral
        from quantumbridge.operators import SparsePauliOperator
        if not isinstance(operator, FermionicOp):
            raise TypeError("Jordan-Wigner mapping requires native FermionicOp.")
        if isinstance(max_terms, bool) or not isinstance(max_terms, Integral) or max_terms < 1:
            raise ValueError("Mapping term budget must be positive.")
        width, terms = operator.register_length, []
        for word, coefficient in operator.terms:
            product = SparsePauliOperator([("I" * width, coefficient)])
            for sign, mode in _tokens(word):
                x = "Z" * mode + "X" + "I" * (width - mode - 1)
                y = "Z" * mode + "Y" + "I" * (width - mode - 1)
                if 2 * len(product.terms) > max_terms:
                    raise MemoryError("Jordan-Wigner product exceeds Pauli term budget.")
                product = product.dot(SparsePauliOperator([(x, .5), (y, -.5j if sign == "+" else .5j)]))
            terms.extend(product.terms)
            if len(terms) > max_terms:
                raise MemoryError("Jordan-Wigner sum exceeds Pauli term budget.")
        return QubitHamiltonian(SparsePauliOperator(terms).simplify(), num_qubits=width)


class ParityMapper:
    status = "planned"

    def map(self, operator):
        raise NotImplementedError("QuantumBridge ParityMapper is planned for a later P2 issue.")


class BravyiKitaevMapper:
    status = "planned"

    def map(self, operator):
        raise NotImplementedError("QuantumBridge BravyiKitaevMapper is planned for a later P2 issue.")
