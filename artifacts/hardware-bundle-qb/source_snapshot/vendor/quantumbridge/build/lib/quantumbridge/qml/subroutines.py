# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Native IQP, QFT and bounded first/second-order Pauli product formulas."""
from itertools import combinations
from numbers import Integral
import numpy as np
from quantumbridge.circuit_library.templates import qft as qft_circuit
from quantumbridge.utils.math import Hamiltonian, PauliString
from .template_core import (resolve_limits, selected_wires, angle_array, angle_value,
                            operation_budget, native_circuit, bounded_pairs)


def iqp_embedding(circuit, features, wires=None, n_repeats=1, pattern=None, *, limits=None):
    limits = resolve_limits(limits)
    wires = selected_wires(circuit, wires, limits)
    if angle_array(features, 1, limits) != (len(wires),):
        raise ValueError("IQPEmbedding requires one feature per selected wire.")
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, Integral) or n_repeats < 1:
        raise ValueError("IQP repeats must be a positive integer.")
    limits.check("max_layers", n_repeats)
    pairs = bounded_pairs(combinations(wires,2) if pattern is None else pattern,limits)
    if any(len(pair) != 2 or pair[0] == pair[1] or any(q not in wires for q in pair) for pair in pairs):
        raise ValueError("IQP patterns require pairs of distinct selected wires.")
    operation_budget(circuit, limits, n_repeats*(2*len(wires)+3*len(pairs)))
    indices = {wire:position for position,wire in enumerate(wires)}
    for _ in range(n_repeats):
        for wire,value in zip(wires,features):
            circuit.h(wire)
            circuit.rz(value,wire)
        for a,b in pairs:
            circuit.cx(a,b)
            circuit.rz(features[indices[a]]*features[indices[b]],b)
            circuit.cx(a,b)
    return circuit


def qft(circuit, wires=None, inverse=False, do_swaps=True, *, limits=None):
    limits = resolve_limits(limits)
    wires = selected_wires(circuit,wires,limits)
    limits.check("max_qft_wires",len(wires))
    if not isinstance(inverse,bool) or not isinstance(do_swaps,bool):
        raise TypeError("QFT inverse and do_swaps must be boolean.")
    count = len(wires)*(len(wires)+1)//2 + (len(wires)//2 if do_swaps else 0)
    operation_budget(circuit,limits,count)
    native = native_circuit(circuit)
    parameter = getattr(circuit,"parameter_transform",lambda value:value)
    for op in qft_circuit(len(wires),inverse=inverse,do_swaps=do_swaps).operations:
        native.append(op.name,tuple(wires[q] for q in op.targets),tuple(wires[q] for q in op.controls),
                      tuple(parameter(value) for value in op.params),op.metadata)
    return circuit


def _pauli_exponential(target, angle, pauli, fallback_wire):
    # exp(+i angle P); RZ has the opposite-sign half-angle convention.
    active = [(q,char) for q,char in pauli.terms if char != "I"]
    if not active:
        target.rz(-2*angle,fallback_wire)
        target.phase(2*angle,fallback_wire)
        return
    for q,char in active:
        if char == "X": target.h(q)
        elif char == "Y": target.phase(-np.pi/2,q);target.h(q)
    for (a,_),(b,_) in zip(active,active[1:]): target.cx(a,b)
    target.rz(-2*angle,active[-1][0])
    for (a,_),(b,_) in reversed(list(zip(active,active[1:]))): target.cx(a,b)
    for q,char in reversed(active):
        if char == "X": target.h(q)
        elif char == "Y": target.h(q);target.phase(np.pi/2,q)


def trotter_product(circuit, hamiltonian, time, wires=None, n=1, order=1, *, limits=None):
    """Approximate exp(+i time H), preserving the listed Pauli operand order."""
    limits = resolve_limits(limits)
    wires = selected_wires(circuit,wires,limits)
    if hasattr(hamiltonian,"to_quantumbridge"):
        hamiltonian = hamiltonian.to_quantumbridge()
    elif hasattr(hamiltonian,"to_hamiltonian"):
        hamiltonian = hamiltonian.to_hamiltonian()
    if not isinstance(hamiltonian,Hamiltonian) or not hamiltonian.terms:
        raise TypeError("TrotterProduct needs a nonempty native Pauli Hamiltonian.")
    if isinstance(n,bool) or not isinstance(n,Integral) or n < 1:
        raise ValueError("Trotter steps must be a positive integer.")
    if isinstance(order,bool) or not isinstance(order,Integral) or order not in (1,2):
        raise NotImplementedError("Native Trotter supports product-formula orders one and two.")
    limits.check("max_trotter_steps",n)
    angle_value(time)
    terms, per_step = [], 0
    for coefficient,pauli in hamiltonian.terms:
        angle_value(coefficient)
        if not isinstance(pauli,PauliString) or len({q for q,_ in pauli.terms}) != len(pauli.terms) or any(not 0 <= q < len(wires) for q,_ in pauli.terms):
            raise ValueError("Trotter Pauli terms must address distinct template-local wire indices.")
        mapped = PauliString([(wires[q],char) for q,char in pauli.terms])
        active = [char for _,char in mapped.terms if char != "I"]
        per_step += 2 if not active else 2*(len(active)-1)+1+2*active.count("X")+4*active.count("Y")
        terms.append((coefficient,mapped))
    operation_budget(circuit,limits,n*per_step*(2 if order==2 else 1))
    sequence = list(reversed(terms)) if order == 1 else terms+list(reversed(terms))
    for _ in range(n):
        for coefficient,pauli in sequence:
            _pauli_exponential(circuit,time*coefficient/(n*(2 if order==2 else 1)),pauli,wires[0])
    return circuit
