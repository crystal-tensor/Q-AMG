# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from copy import deepcopy
from numbers import Integral

import numpy as np

from quantumbridge.information.clifford import (
    Clifford, _gf2_basis, _gf2_solve, _symplectic_product, _stack_binary,
    _swap_halves, _label_array, _workspace_notice,
)
from quantumbridge.operators import Pauli, SparsePauliOperator
from quantumbridge.operators.pauli import _PHASE_FACTORS, _validate_qargs
from quantumbridge.execution import mps_arrays as arrays
from quantumbridge.execution.resource_context import current_resources, meter_native_samples, record_completed_native_shot


def _pauli_vector(pauli: Pauli):
    result=arrays.owned("stabilizer_pauli_vector",(2*pauli.num_qubits,),bool)
    for index,symbol in enumerate(reversed(pauli._body)):
        result[index]=symbol in "XY"
        result[pauli.num_qubits+index]=symbol in "ZY"
    return result


def _generator_rows(paulis,width):
    result=arrays.owned("stabilizer_generator_rows",(len(paulis),2*width),bool)
    for row,pauli in enumerate(paulis):
        for index,symbol in enumerate(reversed(pauli._body)):
            result[row,index]=symbol in "XY"
            result[row,width+index]=symbol in "ZY"
    return result


def _nullspace(matrix):
    reduced = _gf2_basis(matrix)
    width = reduced.shape[1]
    pivots=[next(index for index,value in enumerate(row) if value) for row in reduced]
    free=[column for column in range(width) if column not in pivots]
    vectors=arrays.filled("stabilizer_nullspace",(len(free),width),bool,False)
    for vector,column in zip(vectors,free):
        vector[column]=True
        for row, pivot in enumerate(pivots):
            vector[pivot] = reduced[row, column]
    return vectors


class StabilizerState:
    """Pure stabilizer state with polynomial-memory Clifford-tableau execution."""

    def __init__(self, data, validate: bool = True):
        _workspace_notice()
        if isinstance(data, StabilizerState):
            self._clifford = data.clifford.copy()
            self._rng = deepcopy(data._rng)
        else:
            self._clifford = Clifford(data, validate=validate)
            self._rng = np.random.default_rng()

    @classmethod
    def from_stabilizer_list(
        cls,
        stabilizers,
        allow_redundant: bool = False,
        allow_underconstrained: bool = False,
    ) -> "StabilizerState":
        if current_resources() is not None and type(stabilizers) not in (list,tuple):
            from quantumbridge.core.classical import UnsupportedFeature
            raise UnsupportedFeature("stabilizer_generator_sequence","Managed generator conversion requires an explicit builtin sequence.")
        supplied = [Pauli(label) for label in stabilizers]
        if not supplied:
            raise ValueError("QuantumBridge needs a non-empty stabilizer list to infer state width.")
        width = supplied[0].num_qubits
        if width == 0 or any(pauli.num_qubits != width or pauli.phase % 2 for pauli in supplied):
            raise ValueError("QuantumBridge stabilizers must be matching-width Hermitian Paulis.")
        if any(not left.commutes(right) for index, left in enumerate(supplied) for right in supplied[index + 1:]):
            raise ValueError("QuantumBridge stabilizer generators must commute.")

        basis = {}
        independent = []
        for pauli in supplied:
            reduced = pauli.copy()
            for pivot in sorted(basis):
                if _pauli_vector(reduced)[pivot]:
                    reduced = reduced.dot(basis[pivot])
            vector=_pauli_vector(reduced)
            pivot=next((index for index,value in enumerate(vector) if value),None)
            if pivot is None:
                if reduced.phase:
                    raise ValueError("QuantumBridge stabilizer constraints contain negative identity.")
                if not allow_redundant:
                    raise ValueError("QuantumBridge stabilizer generators are redundant.")
                continue
            basis[pivot] = reduced
            independent.append(pauli)

        if len(independent) < width and not allow_underconstrained:
            raise ValueError("QuantumBridge stabilizer list is underconstrained.")
        while len(independent) < width:
            vectors=_generator_rows(independent,width)
            constraints=_swap_halves(vectors)
            for candidate in _nullspace(constraints):
                if len(_gf2_basis(_stack_binary(vectors,candidate[None,:]))) > len(independent):
                    independent.append(Pauli((candidate[width:], candidate[:width])))
                    break
            else:
                raise ValueError("QuantumBridge cannot complete the stabilizer constraints.")

        stabilizer_vectors=_generator_rows(independent,width)
        constraints=_swap_halves(stabilizer_vectors)
        destabilizers = []
        for index in range(width):
            additional=_generator_rows(destabilizers,width)
            equations=_stack_binary(constraints,_swap_halves(additional))
            target=arrays.filled("stabilizer_dual_target",(width+index,),bool,False)
            target[index] = True
            vector = _gf2_solve(equations, target)
            destabilizers.append(Pauli((vector[width:], vector[:width])))
        clifford = Clifford.identity(width)
        for row, pauli in enumerate(destabilizers + independent):
            clifford._set_row(row, pauli)
        return cls(clifford)

    @property
    def clifford(self):
        return self._clifford

    @property
    def num_qubits(self):
        return self.clifford.num_qubits

    @property
    def dim(self):
        return 1 << self.num_qubits

    def _qubits(self, qargs):
        selected = tuple(range(self.num_qubits) if qargs is None else qargs)
        return _validate_qargs(selected, len(selected), self.num_qubits)

    def dims(self, qargs=None):
        return (2,) * len(self._qubits(qargs))

    def is_valid(self, atol=None, rtol=None) -> bool:
        return self.clifford.is_unitary()

    def copy(self):
        return StabilizerState(self)

    def seed(self, value=None) -> None:
        self._rng = np.random.default_rng(value)

    def trace(self):
        return 1.0

    def purity(self):
        return 1.0

    def expectation_value(self, oper, qargs=None):
        if isinstance(oper, SparsePauliOperator):
            return sum(coefficient * self.expectation_value(Pauli(label), qargs=qargs) for label, coefficient in oper.terms)
        pauli = oper if isinstance(oper, Pauli) else Pauli(oper)
        if qargs is not None:
            pauli = pauli._embedded(qargs, self.num_qubits)
        if pauli.num_qubits != self.num_qubits:
            raise ValueError("QuantumBridge stabilizer observable width must match the state.")
        vector = _pauli_vector(pauli)
        if any(_symplectic_product(vector, row[:-1]) for row in self.clifford.stab):
            return 0j
        represented = Pauli("I" * self.num_qubits)
        for index, row in enumerate(self.clifford.destab):
            if _symplectic_product(vector, row[:-1]):
                represented = represented.dot(self.clifford._row_pauli(self.num_qubits + index))
        return complex(_PHASE_FACTORS[(pauli.phase - represented.phase) % 4])

    def evolve(self, other, qargs=None):
        result = self.copy()
        result._clifford = self.clifford.compose(other, qargs=qargs)
        return result

    def conjugate(self):
        result = self.copy()
        result._clifford = self.clifford.conjugate()
        return result

    def tensor(self, other):
        operand = other if isinstance(other, StabilizerState) else StabilizerState(other)
        return StabilizerState(self.clifford.tensor(operand.clifford))

    def expand(self, other):
        operand = other if isinstance(other, StabilizerState) else StabilizerState(other)
        return operand.tensor(self)

    def equiv(self, other) -> bool:
        operand = other if isinstance(other, StabilizerState) else StabilizerState(other)
        if self.num_qubits != operand.num_qubits:
            return False
        return all(self.expectation_value(operand.clifford._row_pauli(self.num_qubits + row)) == 1 for row in range(self.num_qubits))

    def _measure_z(self, qubit: int, forced=None, rng=None):
        destabilizer_row=next((row for row in range(self.num_qubits) if self.clifford.stab_x[row,qubit]),None)
        label="I"*(self.num_qubits-1-qubit)+"Z"+"I"*qubit
        if destabilizer_row is None:
            observable=Pauli(label)
            outcome = 0 if self.expectation_value(observable).real > 0 else 1
            return outcome, float(forced is None or forced == outcome)

        outcome = int((self._rng if rng is None else rng).integers(2)) if forced is None else int(forced)
        pivot = self.num_qubits + destabilizer_row
        generator = self.clifford._row_pauli(pivot)
        for row in range(2*self.num_qubits):
            if self.clifford.x[row,qubit] and row not in (pivot,destabilizer_row):
                self.clifford._set_row(row,self.clifford._row_pauli(row).dot(generator))
        self.clifford._set_row(destabilizer_row, generator)
        generator=Pauli(label)
        generator.phase=2*outcome
        self.clifford._set_row(pivot,generator)
        return outcome, 0.5

    @meter_native_samples(result_count=lambda result:1)
    def _measure_once(self,qargs=None,shots=1,reset=False):
        selected = self._qubits(qargs)
        result = self.copy()
        outcomes=[]
        for qubit in selected:
            outcome=result._measure_z(qubit,rng=self._rng)[0]
            outcomes.append(outcome)
            if reset and outcome:result.clifford._append_gate("x",(qubit,))
        record_completed_native_shot()
        result._rng = deepcopy(self._rng)
        if reset:return result
        return "".join(str(value) for value in reversed(outcomes)), result

    def measure(self,qargs=None):
        return self._measure_once(qargs)

    def reset(self, qargs=None):
        return self._measure_once(qargs,reset=True)

    @meter_native_samples
    def sample_memory(self, shots: int, qargs=None):
        if isinstance(shots,bool) or not isinstance(shots, Integral) or shots < 0:
            raise ValueError("QuantumBridge stabilizer shots must be a non-negative integer.")
        selected = self._qubits(qargs)
        scope=current_resources()
        if scope is not None:
            scope.ledger.preflight(memory_bytes=4*max(1,len(selected))*shots+(self.clifford.tableau.nbytes if shots else 0))
        memory = []
        for _ in range(shots):
            collapsed = self.copy()
            outcomes = [collapsed._measure_z(qubit, rng=self._rng)[0] for qubit in selected]
            record_completed_native_shot()
            memory.append("".join(str(value) for value in reversed(outcomes)))
        return _label_array(memory,len(selected),"stabilizer_sample_labels")

    def sample_counts(self, shots: int, qargs=None):
        counts = {}
        for bitstring in self.sample_memory(shots, qargs=qargs):
            bitstring = str(bitstring)
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    def probabilities_dict(self, qargs=None, decimals=None):
        selected = self._qubits(qargs)
        branches = [(self.copy(), (), 1.0)]
        for qubit in selected:
            next_branches = []
            for state, bits, probability in branches:
                for outcome in (0, 1):
                    collapsed = state.copy()
                    _, weight = collapsed._measure_z(qubit, forced=outcome)
                    if weight:
                        next_branches.append((collapsed, bits + (outcome,), probability * weight))
            branches = next_branches
        probabilities = {
            "".join(str(value) for value in reversed(bits)): probability
            for _, bits, probability in branches
        }
        if decimals is not None:
            probabilities = {key: round(value, decimals) for key, value in probabilities.items()}
        return dict(sorted((key, value) for key, value in probabilities.items() if value))

    def probabilities(self, qargs=None, decimals=None):
        selected = self._qubits(qargs)
        probabilities=arrays.filled("stabilizer_probability_output",(1<<len(selected),),float,0)
        for key, value in self.probabilities_dict(selected, decimals=decimals).items():
            probabilities[int(key or "0", 2)] = value
        return probabilities

    def to_statevector(self):
        from quantumbridge.information.statevector import Statevector
        from quantumbridge.execution.resource_context import current_resources
        scope = current_resources()
        if scope is not None:
            from quantumbridge.execution.array_reservations import array_layout
            scope.ledger.preflight(memory_bytes=array_layout((1 << self.num_qubits,), complex)[2])

        if self.num_qubits == 0:
            return Statevector([1])
        return Statevector.from_circuit(self.clifford.to_circuit())

    def to_operator(self):
        """Return the state's Clifford preparation unitary, not its projector."""
        return self.clifford.to_operator()

    def to_density_matrix(self):
        from quantumbridge.information.density_matrix import DensityMatrix

        return DensityMatrix.from_statevector(self.to_statevector())

    def __repr__(self):
        return f"StabilizerState({self.clifford.to_labels(mode='S')!r})"
