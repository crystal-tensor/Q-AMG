# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_upstream_integration_strategy_v0.1.md.

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Number
import re
import numpy as np


def _tokens(label):
    if not isinstance(label, str):
        raise TypeError("Fermionic labels must be strings.")
    if label.strip() in {"", "I"}:
        return ()
    result = []
    for token in label.split():
        match = re.fullmatch(r"([+-])_([0-9]+)", token)
        if match is None:
            raise ValueError("Invalid fermionic ladder token: " + token)
        result.append((match[1], int(match[2])))
    if len(result) > 64:
        raise MemoryError("Fermionic word exceeds 64 ladder factors.")
    return tuple(result)


def _label(tokens):
    return " ".join(f"{sign}_{index}" for sign, index in tokens)


@dataclass(frozen=True)
class FermionicOp:
    terms: tuple[tuple[str, complex], ...]
    register_length: int | None = None

    def __init__(self, terms, register_length=None):
        items = terms.items() if isinstance(terms, dict) else terms
        normalized, required = {}, 0
        for count, (label, coefficient) in enumerate(items, 1):
            if count > 100000:
                raise MemoryError("Too many fermionic input terms.")
            tokens = _tokens(label)
            required = max(required, 1 + max((i for _, i in tokens), default=-1))
            coefficient = complex(coefficient)
            if not np.isfinite(coefficient):
                raise ValueError("Fermionic coefficients must be finite.")
            name = _label(tokens)
            normalized[name] = normalized.get(name, 0j) + coefficient
        if any(not np.isfinite(value) for value in normalized.values()):
            raise ValueError("Accumulated fermionic coefficients must remain finite.")
        if register_length is None:
            register_length = required
        if isinstance(register_length, bool) or not isinstance(register_length, Integral) or not required <= register_length <= 128:
            raise ValueError("Fermionic register must contain every index and be at most 128 modes.")
        object.__setattr__(self, "terms", tuple((key, value) for key, value in normalized.items() if value != 0))
        object.__setattr__(self, "register_length", int(register_length))

    def normal_ordered(self, *, max_terms=100000):
        """Creators first, descending mode index in each group, with CAR contractions."""
        if isinstance(max_terms, bool) or not isinstance(max_terms, Integral) or max_terms < 1:
            raise ValueError("Normal-order term budget must be positive.")
        pending = [(_tokens(label), coefficient) for label, coefficient in reversed(self.terms)]
        result, steps = {}, 0
        while pending:
            tokens, coefficient = pending.pop()
            steps += 1
            if steps > max_terms * 64 or len(pending) + len(result) > max_terms:
                raise MemoryError("Fermionic normal ordering exceeds expansion budget.")
            for position in range(len(tokens) - 1):
                left, right = tokens[position:position + 2]
                if left == right:
                    break  # Equal adjacent creators/annihilators square to zero.
                key = lambda token: (token[0] == "-", -token[1])
                if key(left) > key(right):
                    prefix, suffix = tokens[:position], tokens[position + 2:]
                    if left[0] == "-" and right[0] == "+" and left[1] == right[1]:
                        pending.append((prefix + suffix, coefficient))
                    pending.append((prefix + (right, left) + suffix, -coefficient))
                    break
            else:
                label = _label(tokens)
                result[label] = result.get(label, 0j) + coefficient
                if len(result) > max_terms:
                    raise MemoryError("Fermionic normal ordering exceeds expansion budget.")
        return FermionicOp(result, self.register_length)

    def adjoint(self):
        return FermionicOp([(_label(tuple(("-" if sign == "+" else "+", index) for sign, index in reversed(_tokens(label)))), coefficient.conjugate()) for label, coefficient in self.terms], self.register_length)

    def __add__(self, other):
        if not isinstance(other, FermionicOp):
            return NotImplemented
        return FermionicOp(self.terms + other.terms, max(self.register_length, other.register_length))

    def __mul__(self, scalar):
        if not isinstance(scalar, Number):
            return NotImplemented
        return FermionicOp([(label, scalar * value) for label, value in self.terms], self.register_length)

    __rmul__ = __mul__

    def __sub__(self, other):
        return self + (-1 * other)

    def __matmul__(self, other):
        if not isinstance(other, FermionicOp):
            return NotImplemented
        if len(self.terms) * len(other.terms) > 100000:
            raise MemoryError("Fermionic product exceeds term budget.")
        return FermionicOp([(a + " " + b, x * y) for a, x in self.terms for b, y in other.terms], max(self.register_length, other.register_length))

    def to_matrix(self, *, max_qubits=10):
        """Independent occupation-basis action, mode zero first; no Pauli mapping."""
        width = self.register_length
        if width > max_qubits or width > 10:
            raise MemoryError("Dense fermionic matrix exceeds the ten-mode budget.")
        dimension = 1 << width
        if dimension * max(1, len(self.terms)) * 64 > 100000000:
            raise MemoryError("Dense fermionic action exceeds work budget.")
        matrix = np.zeros((dimension, dimension), complex)
        for label, coefficient in self.terms:
            for column in range(dimension):
                bits = [(column >> (width - 1 - q)) & 1 for q in range(width)]
                sign = 1
                for action, mode in reversed(_tokens(label)):
                    if bits[mode] == (action == "+"):
                        break
                    sign *= (-1) ** sum(bits[:mode])
                    bits[mode] ^= 1
                else:
                    row = 0
                    for bit in bits:
                        row = 2 * row + bit
                    matrix[row, column] += sign * coefficient
        return matrix

    def to_dict(self) -> dict[str, complex]:
        return dict(self.terms)
