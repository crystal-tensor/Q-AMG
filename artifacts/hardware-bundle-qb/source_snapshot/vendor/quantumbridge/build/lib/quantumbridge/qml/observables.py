# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from math import sqrt
from numbers import Real
from typing import Iterable

from quantumbridge.operators import SparsePauliOperator
from quantumbridge.utils.math import Hamiltonian as NativeHamiltonian
from quantumbridge.utils.math import PauliString


def Identity(wire: int) -> PauliString:
    return PauliString([(wire, "I")])


def PauliX(wire: int) -> PauliString:
    return PauliString.x(wire)


def PauliY(wire: int) -> PauliString:
    return PauliString.y(wire)


def PauliZ(wire: int) -> PauliString:
    return PauliString.z(wire)


def Hadamard(wire: int) -> NativeHamiltonian:
    scale = 1.0 / sqrt(2.0)
    return NativeHamiltonian([(scale, PauliX(wire)), (scale, PauliZ(wire))])


def Hamiltonian(coeffs: Iterable[Real], observables: Iterable[PauliString | NativeHamiltonian]) -> NativeHamiltonian:
    terms = []
    for coeff, observable in zip(coeffs, observables):
        if isinstance(observable, NativeHamiltonian):
            terms.extend((float(coeff) * inner_coeff, pauli) for inner_coeff, pauli in observable.terms)
        elif isinstance(observable, PauliString):
            terms.append((float(coeff), observable))
        elif hasattr(observable, "to_hamiltonian"):
            native = observable.to_hamiltonian()
            terms.extend((float(coeff) * inner_coeff, pauli) for inner_coeff, pauli in native.terms)
        else:
            raise ValueError("QuantumBridge qml.Hamiltonian expects Pauli-like observables.")
    return NativeHamiltonian(terms)


def SparseHamiltonian(terms: Iterable[tuple[str, complex]]) -> SparsePauliOperator:
    return SparsePauliOperator.from_list(terms)
