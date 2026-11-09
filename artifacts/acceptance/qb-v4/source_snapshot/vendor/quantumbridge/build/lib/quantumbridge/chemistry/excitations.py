# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
from numbers import Integral
import numpy as np
from .fermion import FermionicOp
from .mappings import JordanWignerMapper


def hf_state(electrons, orbitals, basis="occupation_number"):
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (electrons, orbitals)) or not 0 <= electrons <= orbitals <= 128 or orbitals < 1:
        raise ValueError("HF occupations require integer 0<=electrons<=orbitals<=128.")
    if basis != "occupation_number":
        raise NotImplementedError("Native HF state currently uses occupation-number encoding.")
    return np.array([1] * electrons + [0] * (orbitals - electrons), dtype=int)


def excitations(electrons, orbitals, delta_sz=0):
    hf_state(electrons, orbitals)
    if delta_sz not in {-2, -1, 0, 1, 2} or isinstance(delta_sz, bool) or not isinstance(delta_sz, Integral):
        raise ValueError("Excitation spin change must be an integer in [-2,2].")
    spin = lambda mode: .5 if mode % 2 == 0 else -.5
    singles = [[i, a] for i in range(electrons) for a in range(electrons, orbitals) if spin(a) - spin(i) == delta_sz]
    doubles = [[i, j, a, b] for i in range(electrons) for j in range(i + 1, electrons)
               for a in range(electrons, orbitals) for b in range(a + 1, orbitals)
               if spin(a) + spin(b) - spin(i) - spin(j) == delta_sz]
    return singles, doubles


def excitations_to_wires(singles, doubles, wires=None):
    singles, doubles = list(singles), list(doubles)
    if any(len(ex) != 2 or any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in ex) or ex[0] >= ex[1] for ex in singles):
        raise ValueError("Invalid single-excitation indices.")
    if any(len(ex) != 4 or any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in ex) or len(set(ex)) != 4 or not ex[0] < ex[1] < ex[2] < ex[3] for ex in doubles):
        raise ValueError("Invalid double-excitation indices.")
    width = 1 + max((mode for ex in singles + doubles for mode in ex), default=-1)
    wires = tuple(range(width)) if wires is None else tuple(wires)
    if len(wires) < width or len(set(wires)) != len(wires):
        raise ValueError("Excitation wires must cover indices without duplicates.")
    return ([[wires[q] for q in range(i, a + 1)] for i, a in singles],
            [[[wires[q] for q in range(i, j + 1)], [wires[q] for q in range(a, b + 1)]] for i, j, a, b in doubles])


def apply_fermionic_excitation(target, theta, excitation):
    """Exact commuting-Pauli circuit for exp(theta/2 * (A - A-dagger))."""
    circuit = getattr(target, "circuit", target)
    width, indices = circuit.num_qubits, tuple(excitation)
    if len(indices) not in {2, 4} or any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < width for i in indices) or len(set(indices)) != len(indices):
        raise ValueError("A supported excitation needs two or four distinct in-register modes.")
    if len(indices) == 2:
        i, a = indices
        label = f"+_{a} -_{i}"
    else:
        i, j, a, b = indices
        label = f"+_{a} +_{b} -_{j} -_{i}"
    operator = FermionicOp({label: 1}, width)
    generator = JordanWignerMapper().map(1j * (operator - operator.adjoint())).to_quantumbridge()
    # Single/double excitation generators have commuting Pauli terms: exact, not Trotter.
    labels = [dict(pauli.terms) for _, pauli in generator.terms]
    if any(sum(x.get(q, "I") != "I" and y.get(q, "I") != "I" and x[q] != y[q] for q in set(x) | set(y)) % 2 for x in labels for y in labels):
        raise NotImplementedError("Excitation requires noncommuting synthesis not implemented here.")
    for coefficient, pauli in generator.terms:
        wires = [q for q, char in pauli.terms if char != "I"]
        for q, char in pauli.terms:
            if char == "X":
                target.h(q)
            elif char == "Y":
                target.phase(-np.pi / 2, q)
                target.h(q)
        for left, right in zip(wires, wires[1:]):
            target.cx(left, right)
        target.rz(theta * coefficient, wires[-1])
        for left, right in reversed(list(zip(wires, wires[1:]))):
            target.cx(left, right)
        for q, char in reversed(pauli.terms):
            if char == "X":
                target.h(q)
            elif char == "Y":
                target.h(q)
                target.phase(np.pi / 2, q)
    return target
