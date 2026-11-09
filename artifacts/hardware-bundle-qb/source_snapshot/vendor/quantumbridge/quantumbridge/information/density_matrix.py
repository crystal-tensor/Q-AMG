# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

import numpy as np

from typing import Union

from .statevector import Statevector, _sample_measurements, _tensor_arrays, _mapped_evolution_data, _expectation_value
from .statevector import _state_arithmetic, _state_trace, _state_dictionary, _state_valid, _state_data, _state_arrays
from .statevector import _state_operator_data
from quantumbridge.execution.resource_context import current_resources, meter_native_samples
from .state_dimensions import (
    state_dimensions, qubit_count, basis_digits, basis_index, basis_label,
    require_qubits, subsystem_positions,
)


def _validate_qargs(qargs, num_qubits: int) -> tuple[int, ...]:
    return subsystem_positions(qargs,num_qubits)


class DensityMatrix:
    def __init__(self, data, *, dims=None):
        from quantumbridge.core import Circuit, Instruction, Gate, ControlledGate
        from quantumbridge.core.annotated_operation import AnnotatedOperation
        if type(data) is AnnotatedOperation: data = Circuit.from_instruction(data)
        from quantumbridge.execution.resource_context import current_resources
        scope=current_resources()
        from quantumbridge.core.gates import is_native_instruction
        from quantumbridge.core.circuit import is_native_circuit
        if (is_native_circuit(data) or is_native_instruction(data) if scope is not None else isinstance(data,(Circuit,Instruction))):
            data = self.from_instruction(data).data
        if scope is None:
            self.data = np.asarray(data, dtype=complex)
            shape=self.data.shape
        else:
            from quantumbridge.execution.array_reservations import numeric_array, numeric_shape
            shape=numeric_shape(data)
        if len(shape)!=2 or shape[0]!=shape[1]:
            raise ValueError("QuantumBridge DensityMatrix data must be a square matrix.")
        self._dims=state_dimensions(dims,size=shape[0])
        if scope is not None:
            self.data=numeric_array(data,category="density_input_conversion")

    @property
    def dim(self) -> tuple[int, int]:
        return self.data.shape

    def dims(self, qargs=None) -> tuple[int, ...]:
        return tuple(self._dims[position] for position in _validate_qargs(qargs,len(self._dims)))

    @classmethod
    def from_statevector(cls, statevector: Union[Statevector, np.ndarray]) -> "DensityMatrix":
        scope=current_resources()
        native=type(statevector) is Statevector if scope is not None else isinstance(statevector,Statevector)
        dimensions=statevector.dims() if native else None
        if scope is not None:
            from quantumbridge.execution import mps_arrays as arrays
            from quantumbridge.execution.array_reservations import numeric_shape, numeric_array, array_layout
            raw=statevector.data if type(statevector) is Statevector else statevector
            shape=numeric_shape(raw)
            if len(shape)!=1:raise ValueError("DensityMatrix.from_statevector requires a vector.")
            dimension=shape[0]
            conversion=0 if issubclass(type(raw),np.ndarray) and np.ndarray.view(raw,np.ndarray).dtype==np.dtype(complex) else 16*dimension
            scope.ledger.preflight(memory_bytes=array_layout((dimension,dimension),complex)[2]+16*dimension+conversion)
            vector=numeric_array(raw,category="density_vector_input")
            return cls(arrays.einsum("i,j->ij",(dimension,dimension),vector,arrays.conjugated(vector)),dims=dimensions)
        vector = statevector.data if isinstance(statevector, Statevector) else np.asarray(statevector, dtype=complex)
        return cls(np.outer(vector, np.conjugate(vector)),dims=dimensions)

    @classmethod
    def from_int(cls, index: int, dims) -> "DensityMatrix":
        return cls.from_statevector(Statevector.from_int(index, dims))

    @classmethod
    def from_label(cls, label: str) -> "DensityMatrix":
        return cls.from_statevector(Statevector.from_label(label))

    @classmethod
    def from_circuit(cls, circuit) -> "DensityMatrix":
        from quantumbridge.devices import DensityMatrixDevice

        return cls(DensityMatrixDevice().density_matrix(circuit))

    @classmethod
    def from_instruction(cls, instruction) -> "DensityMatrix":
        from quantumbridge.core import Circuit, Instruction, Gate, ControlledGate
        from quantumbridge.core.annotated_operation import AnnotatedOperation
        if type(instruction) is AnnotatedOperation: instruction = Circuit.from_instruction(instruction)
        scope=current_resources()
        if scope is not None:
            from quantumbridge.core.gates import is_native_instruction
            from quantumbridge.core.circuit import is_native_circuit
            if not is_native_circuit(instruction) and not is_native_instruction(instruction):
                from quantumbridge.core.classical import UnsupportedFeature
                raise UnsupportedFeature("managed_state_instruction","Managed state construction requires a native Circuit or Instruction.")
            from quantumbridge.execution.array_reservations import array_layout
            scope.ledger.preflight(memory_bytes=array_layout((1<<instruction.num_qubits,)*2,complex)[2])
        if isinstance(instruction,Instruction):
            instruction = Circuit.from_instruction(instruction)
        if not isinstance(instruction,Circuit):
            raise TypeError("DensityMatrix.from_instruction requires a native Circuit or Instruction.")
        return cls.from_circuit(instruction)

    def trace(self) -> complex:
        return _state_trace(self)

    @property
    def num_qubits(self) -> int | None:
        return qubit_count(self._dims)

    def is_valid(self, atol: float | None = None, rtol: float | None = None) -> bool:
        return _state_valid(self,atol,rtol)

    def purity(self) -> complex:
        return _state_trace(self,square=True)

    def to_dict(self, decimals: int | None = None) -> dict[str, complex]:
        return _state_dictionary(self,decimals)

    def __add__(self, other):
        data,dimensions=_state_arithmetic(self,other,"add")
        return DensityMatrix(data,dims=dimensions)

    def __sub__(self, other):
        data,dimensions=_state_arithmetic(self,other,"subtract")
        return DensityMatrix(data,dims=dimensions)

    def __neg__(self):
        return self.__mul__(-1)

    def __mul__(self, scalar):
        result=_state_arithmetic(self,scalar,"multiply")
        if result is NotImplemented:return NotImplemented
        data,dimensions=result
        return DensityMatrix(data,dims=dimensions)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        result=_state_arithmetic(self,scalar,"divide")
        if result is NotImplemented:return NotImplemented
        data,dimensions=result
        return DensityMatrix(data,dims=dimensions)

    def partial_trace(self, keep: tuple[int, ...] | list[int]) -> "DensityMatrix":
        from quantumbridge.information.partial_trace import partial_trace

        selected=_validate_qargs(keep,len(self._dims))
        dimensions=tuple(dim for position,dim in enumerate(self._dims) if position in selected)
        return DensityMatrix(partial_trace(self.data,selected,dims=self._dims),dims=dimensions)

    def partial_transpose(self, qargs) -> "DensityMatrix":
        """Transpose selected row/column subsystem indices without conjugation."""
        from quantumbridge.execution.array_ownership import retained_array, storage_ledger
        from quantumbridge.execution.array_reservations import array_layout, numeric_shape
        data,dimensions,_=_state_data(self)
        scope=current_resources()
        if qargs is None:raise TypeError("Partial transpose requires explicit subsystem targets.")
        if scope is not None:
            if type(qargs) not in (tuple,list):
                from quantumbridge.core.classical import UnsupportedFeature
                raise UnsupportedFeature("managed_partial_transpose_targets","Managed partial transpose requires explicit builtin subsystem targets.")
            numeric_shape(qargs,real_only=True)
        selected=_validate_qargs(qargs,len(dimensions))
        ledger=storage_ledger(data) if scope is None else scope.ledger
        data,=_state_arrays((data,),output_bytes=array_layout(data.shape,complex)[2],ledger=ledger)
        result=retained_array(ledger,"density_partial_transpose",data.shape,complex)
        for row in range(len(data)):
            digits=basis_digits(row,dimensions)
            for column in range(len(data)):
                r,c=list(digits),list(basis_digits(column,dimensions))
                for axis in selected:r[axis],c[axis]=c[axis],r[axis]
                result[row,column]=data[basis_index(r,dimensions),basis_index(c,dimensions)]
        return DensityMatrix(result,dims=dimensions)

    def probabilities(self, wires=None) -> dict[str, float]:
        selected = _validate_qargs(wires,len(self._dims))
        probs: dict[str, float] = {}
        for index in range(len(self.data)):
            key = basis_label(index,self._dims,selected)
            probs[key] = probs.get(key, 0.0) + complex(self.data[index,index]).real
        return probs

    def probabilities_dict(self, qargs=None) -> dict[str, float]:
        return self.probabilities(qargs)

    def expectation_value(self, observable, qargs=None) -> complex:
        if current_resources() is not None and type(self) is not DensityMatrix:
            from quantumbridge.core.classical import UnsupportedFeature
            raise UnsupportedFeature("managed_expectation_state","Managed expectation requires a native state with explicit numeric storage.")
        return _expectation_value(self.data,self._dims,observable,qargs,density=True)

    def evolve(self, operator, qargs=None) -> "DensityMatrix":
        from quantumbridge.core import Circuit, Instruction
        from quantumbridge.information.channels import _QuantumChannel
        from quantumbridge.information.operator import Operator

        if isinstance(operator,Instruction):
            require_qubits(self._dims,"mixed_radix_circuit_evolution")
            if qargs is not None:
                qargs = _validate_qargs(qargs,self.num_qubits)
                if len(qargs)!=operator.num_qubits:
                    raise ValueError("Instruction target count must match instruction width.")
            operator = Circuit.from_instruction(operator)
        if isinstance(operator, _QuantumChannel):
            require_qubits(self._dims,"mixed_radix_channel_evolution")
            return operator.evolve(self, qargs=qargs)
        if hasattr(operator, "kraus") and not hasattr(operator, "operations"):
            require_qubits(self._dims,"mixed_radix_channel_evolution")
            from quantumbridge.information.channels import Kraus

            return Kraus(operator.kraus()).evolve(self, qargs=qargs)
        if hasattr(operator, "operations"):
            require_qubits(self._dims,"mixed_radix_circuit_evolution")
            from quantumbridge.devices import DensityMatrixDevice
            from quantumbridge.core import Circuit

            circuit = operator
            if qargs is not None:
                selected = _validate_qargs(qargs, self.num_qubits)
                circuit = Circuit(self.num_qubits, operator.num_bits).compose(operator, qubits=selected)
            return DensityMatrix(DensityMatrixDevice(initial_state=self.data).density_matrix(circuit),dims=self._dims)
        op = operator
        from quantumbridge.execution.resource_context import current_resources
        scope=current_resources()
        native=type(op) is Operator if scope is not None else isinstance(op,Operator)
        if qargs is not None or native and op.input_dims()!=op.output_dims():
            from quantumbridge.utils.math import apply_density_operator
            matrix,selected,outgoing,result_dims=_mapped_evolution_data(op,qargs,self._dims,density=True)
            return DensityMatrix(apply_density_operator(self.data,matrix,selected,self.num_qubits,dims=self._dims,output_dims=outgoing),dims=result_dims)
        if scope is not None:
            from quantumbridge.information.channels import _operator_data
            matrix=_operator_data(op)
        else:matrix = np.asarray(op.to_matrix(), dtype=complex) if hasattr(op, "to_matrix") else np.asarray(op, dtype=complex)
        if matrix.shape != self.data.shape:
            raise ValueError("QuantumBridge evolution operator dimension must match DensityMatrix dimension.")
        if scope is not None:
            from quantumbridge.execution import mps_arrays as arrays
            return DensityMatrix(arrays.contracted(arrays.contracted(matrix,self.data),arrays.conjugated(matrix).T),dims=self._dims)
        return DensityMatrix(matrix @ self.data @ matrix.conj().T,dims=self._dims)

    def reset(self, qargs=None) -> "DensityMatrix":
        from quantumbridge.utils.math import apply_density_channel

        selected = _validate_qargs(qargs,len(self._dims))
        if not selected:return self.copy()
        if self.num_qubits is None:
            from quantumbridge.execution import mps_arrays as arrays
            result=arrays.filled("density_mixed_radix_reset",self.data.shape,complex,0)
            for row in range(len(self.data)):
                r=list(basis_digits(row,self._dims))
                target_r=[0 if position in selected else digit for position,digit in enumerate(r)]
                for column in range(len(self.data)):
                    c=list(basis_digits(column,self._dims))
                    if any(r[position]!=c[position] for position in selected):continue
                    target_c=[0 if position in selected else digit for position,digit in enumerate(c)]
                    result[basis_index(target_r,self._dims),basis_index(target_c,self._dims)]+=self.data[row,column]
            return DensityMatrix(result,dims=self._dims)
        from quantumbridge.execution.resource_context import current_resources
        if current_resources() is not None:
            from quantumbridge.utils.math import validate_kraus_operators
            from quantumbridge.execution.array_reservations import numeric_array
            reset=validate_kraus_operators(([[1,0],[0,0]],[[0,1],[0,0]]),2)
            data=numeric_array(self.data,copy=True,category="density_reset_copy")
        else:
            reset = (np.array([[1, 0], [0, 0]], complex), np.array([[0, 1], [0, 0]], complex))
            data = self.data.copy()
        for wire in selected:
            data = apply_density_channel(data, reset, (wire,), self.num_qubits)
        return DensityMatrix(data,dims=self._dims)

    def measure(self, qargs=None, seed=None):
        selected = _validate_qargs(qargs,len(self._dims))
        if current_resources() is not None:
            current_resources().ledger.preflight(memory_bytes=self.data.nbytes)
            outcome=self.sample_memory(1,qargs=selected,seed=seed)[0]
            indices=tuple(index for index in range(len(self.data))
                if basis_label(index,self._dims,selected)==outcome)
            probability=sum(complex(self.data[index,index]).real for index in indices)
            if probability<=0:raise ValueError("Measurement produced a zero-probability density collapse.")
            from quantumbridge.execution import mps_arrays as arrays
            result=arrays.filled("density_projection",self.data.shape,complex,0)
            for row in indices:
                for column in indices:result[row,column]=self.data[row,column]/probability
            return outcome,DensityMatrix(result,dims=self._dims)
        outcome = self.sample_memory(1, qargs=selected, seed=seed)[0]
        mask = np.array([
            basis_label(index,self._dims,selected)==outcome
            for index in range(len(self.data))
        ])
        projected = self.data * mask[:, None] * mask[None, :]
        probability = np.trace(projected).real
        return outcome, DensityMatrix(projected / probability,dims=self._dims)

    def tensor(self, other) -> "DensityMatrix":
        other_state = other if isinstance(other, DensityMatrix) else DensityMatrix(other)
        return DensityMatrix(_tensor_arrays(self.data,other_state.data),dims=self._dims+other_state._dims)

    def expand(self, other) -> "DensityMatrix":
        other_state = other if isinstance(other, DensityMatrix) else DensityMatrix(other)
        return DensityMatrix(_tensor_arrays(other_state.data,self.data),dims=other_state._dims+self._dims)

    def reverse_qargs(self) -> "DensityMatrix":
        from quantumbridge.execution.array_ownership import copied_array
        result=copied_array(self.data,category="density_reverse")
        order=tuple(basis_index(basis_digits(index,self._dims)[::-1],self._dims[::-1]) for index in range(len(self.data)))
        for row,destination in enumerate(order):
            for column,target in enumerate(order):result[destination,target]=self.data[row,column]
        return DensityMatrix(result,dims=self._dims[::-1])

    @meter_native_samples(result_count=lambda counts:sum(counts.values()))
    def sample_counts(self, shots: int, qargs=None, seed: int | None = None) -> dict[str, int]:
        from . import statevector as sampling
        return sampling._counts_from_indices(*_sample_measurements(self,_validate_qargs(qargs,len(self._dims)),shots,seed))

    @meter_native_samples
    def sample_memory(self, shots: int, qargs=None, seed: int | None = None) -> list[str]:
        from . import statevector as sampling
        return sampling._memory_from_indices(*_sample_measurements(self,_validate_qargs(qargs,len(self._dims)),shots,seed))

    def copy(self) -> "DensityMatrix":
        from quantumbridge.execution.array_ownership import copied_array
        return DensityMatrix(copied_array(self.data,category="density_copy"),dims=self._dims)

    def conjugate(self) -> "DensityMatrix":
        copied=self.copy()
        np.conjugate(copied.data,out=copied.data)
        return copied

    def transpose(self) -> "DensityMatrix":
        return DensityMatrix(self.data.T,dims=self._dims)

    def adjoint(self) -> "DensityMatrix":
        return DensityMatrix(self.conjugate().data.T,dims=self._dims)

    def to_operator(self):
        from quantumbridge.information.operator import Operator

        data,dimensions,_=_state_operator_data(self,DensityMatrix)
        operator=Operator(data,input_dims=dimensions,output_dims=dimensions)
        operator._public_matrix_basis="native"
        return operator

    def to_statevector(self, atol: float = 1e-10) -> Statevector:
        from quantumbridge.execution.resource_context import current_resources
        scope=current_resources()
        if scope is None:values, vectors = np.linalg.eigh(self.data)
        else:
            from quantumbridge.execution import mps_arrays as arrays
            values,vectors=arrays.decomposed(self.data,"eigh")
        index = int(np.argmax(values))
        if not np.isclose(values[index], 1.0, atol=atol):
            raise ValueError("QuantumBridge DensityMatrix is not a pure state.")
        if (np.count_nonzero(values > atol) if scope is None else sum(value>atol for value in values)) != 1:
            raise ValueError("QuantumBridge DensityMatrix is not a pure state.")
        return Statevector(vectors[:, index],dims=self._dims)
