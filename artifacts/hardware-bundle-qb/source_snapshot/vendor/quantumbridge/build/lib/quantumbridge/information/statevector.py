# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from math import isfinite, prod
from numbers import Integral, Number

import numpy as np

from quantumbridge.devices import StatevectorDevice
from quantumbridge.utils.math import apply_unitary, expectation_hamiltonian
from quantumbridge.execution.resource_context import (
    current_resources, meter_native_samples, record_completed_native_shots,
)
from .state_dimensions import (
    state_dimensions, qubit_count, basis_digits, basis_index, basis_label,
    require_qubits, subsystem_positions,
)


def _validate_qargs(qargs, num_qubits: int) -> tuple[int, ...]:
    return subsystem_positions(qargs,num_qubits)


def _sampling_preflight(outcomes, shots):
    if isinstance(shots,bool) or not isinstance(shots,Integral) or shots<0:
        raise ValueError("Native state sampling requires nonnegative integer shots.")
    scope=current_resources()
    if scope is not None:
        from quantumbridge.execution.array_reservations import array_layout
        scope.ledger.preflight(memory_bytes=(2*array_layout((outcomes,),float)[2]
            +array_layout((shots,),float)[2]+array_layout((shots,),np.intp)[2]) if shots else 0)
        scope.ledger.unavailable("state_sampling_python_rng_workspace",
            "Probability/CDF/uniform/outcome ndarray buffers are explicit; labels, Python containers and bit-generator internals are not measured RSS.")


def _sample_measurements(state, selected, shots, seed):
    _sampling_preflight(prod(state.dims(selected)),shots)
    distribution=state.probabilities(selected)
    labels=sorted(distribution)
    values=tuple(distribution[label] for label in labels)
    if not values or any(not isfinite(value) or value<0 for value in values) or not 0<sum(values)<float("inf"):
        raise ValueError("State sampling probabilities must be finite, nonnegative and nonzero.")
    if not shots:return labels,()
    scope=current_resources()
    if scope is None:
        weights=np.array(values,dtype=float)
        weights/=weights.sum()
        draws=np.random.default_rng(seed).choice(len(labels),size=shots,p=weights)
    else:
        from quantumbridge.execution.array_reservations import ArrayReservation, owned_arrays
        from quantumbridge.execution.array_ownership import retain_reservation, native_storage_bytes
        reservation=ArrayReservation(scope.ledger,"state_sampling_outcomes",(shots,),np.intp)
        try:
            weights,cdf,uniforms=owned_arrays([((len(labels),),float),((len(labels),),float),((shots,),float)],category="state_sampling_workspace")
            for index,value in enumerate(values):weights[index]=value
            np.divide(weights,weights.sum(),out=weights)
            np.cumsum(weights,out=cdf)
            np.divide(cdf,cdf[-1],out=cdf)
            np.random.default_rng(seed).random(shots,out=uniforms)
            draws=reservation.create(lambda:np.searchsorted(cdf,uniforms,side="right"))
            if native_storage_bytes(draws)!=draws.nbytes or np.may_share_memory(draws,uniforms):
                raise RuntimeError("State sampling produced unexpected outcome storage.")
            retain_reservation(reservation,draws)
        except BaseException:
            reservation.release()
            raise
    record_completed_native_shots(shots)
    return labels,draws


def _memory_from_indices(labels,indices):
    return [labels[int(index)] for index in indices]


def _counts_from_indices(labels,indices):
    counts={label:0 for label in labels}
    for index in indices:counts[labels[int(index)]]+=1
    return {label:count for label,count in counts.items() if count}


def _tensor_arrays(first,second):
    if current_resources() is None:return np.kron(first,second)
    from quantumbridge.execution import mps_arrays as arrays
    from quantumbridge.execution.array_reservations import kron_into
    if first.ndim==1:
        result=arrays.owned("state_tensor",(first.size*second.size,),complex)
        kron_into(first[:,None],second[:,None],result.reshape(-1,1),scope=current_resources())
    else:
        result=arrays.owned("density_tensor",(first.shape[0]*second.shape[0],first.shape[1]*second.shape[1]),complex)
        kron_into(first,second,result,scope=current_resources())
    return result


def _mapped_evolution_data(operator, qargs, dimensions, *, density=False):
    from quantumbridge.information.operator import Operator
    from quantumbridge.information.channels import _operator_data
    from quantumbridge.utils.math import _local_operator_shapes
    from quantumbridge.execution.array_reservations import numeric_shape, array_layout
    from quantumbridge.core.classical import UnsupportedFeature
    scope=current_resources()
    selected=subsystem_positions(qargs,len(dimensions))
    incoming=tuple(dimensions[position] for position in selected)
    native=type(operator) is Operator if scope is not None else isinstance(operator,Operator)
    raw=operator.data if native else operator
    if scope is not None:
        if not native and not (issubclass(type(raw),np.ndarray) or type(raw) in (tuple,list)):
            raise UnsupportedFeature("managed_channel_operator","Mapped evolution requires a native operator or explicit numeric buffer.")
        shape=numeric_shape(raw)
    else:
        if not native and hasattr(operator,"to_matrix"):raw=operator.to_matrix()
        raw=np.asarray(raw,dtype=complex)
        shape=raw.shape
    if len(shape)!=2 or shape[1]!=prod(incoming):
        raise ValueError("Local operator input dimension does not match the selected subsystems.")
    if native:
        if qargs is not None and operator.input_dims()!=incoming:
            raise ValueError("Local operator input subsystem dimensions do not match qargs order.")
        outgoing=operator.output_dims()
    else:
        if shape[0]!=shape[1]:
            raise ValueError("Rectangular evolution requires explicit native Operator output_dims.")
        outgoing=incoming
    if qargs is not None and len(outgoing)!=len(selected):
        raise UnsupportedFeature("mixed_radix_output_subsystem_count","Mapped evolution cannot insert or remove subsystem positions.")
    if qargs is not None and shape[0]!=shape[1] and not density:
        raise UnsupportedFeature("mixed_radix_rectangular_statevector","Public mapped Statevector rectangular evolution fails in both verified versions; use a full typed map or DensityMatrix.")
    result_dims,shapes=_local_operator_shapes(dimensions,selected,outgoing,density=density)
    if scope is not None:
        conversion=array_layout(shape,complex)[2] if native or not issubclass(type(raw),np.ndarray) or np.ndarray.view(raw,np.ndarray).dtype!=np.dtype(complex) else 0
        scope.ledger.preflight(memory_bytes=conversion+sum(array_layout(shape,complex)[2] for shape in shapes))
        matrix=_operator_data(operator)
    else:matrix=raw
    return matrix,selected,outgoing,result_dims


def _expectation_value(data, dimensions, observable, qargs, *, density=False):
    from quantumbridge.information.operator import Operator
    from quantumbridge.operators import Pauli, SparsePauliOperator
    from quantumbridge.execution.array_reservations import numeric_shape, numeric_array, array_layout
    from quantumbridge.utils.math import _expectation_local_operator
    scope=current_resources()
    selected=subsystem_positions(qargs,len(dimensions))
    incoming=tuple(dimensions[position] for position in selected)
    size,local_size=prod(dimensions),prod(incoming)
    native=type(observable) is Operator if scope is not None else isinstance(observable,Operator)
    pauli=type(observable) in (Pauli,SparsePauliOperator) if scope is not None else isinstance(observable,(Pauli,SparsePauliOperator))
    factory=None
    if native:
        if observable._input_dims!=observable._output_dims:
            raise ValueError("An observable must preserve its declared input/output subsystem dimensions.")
        if qargs is not None and observable._input_dims!=incoming:
            raise ValueError("Observable dimensions do not match the ordered selected subsystems.")
        raw=observable.data
    elif pauli:
        require_qubits(dimensions,"mixed_radix_pauli_expectation")
        if (2,)*observable.num_qubits!=incoming:
            raise ValueError("Pauli observable width does not match the selected qubits.")
        factory=observable.to_matrix
        raw=None
    else:
        raw=observable
        if scope is None:
            if hasattr(observable,"to_matrix"):raw=observable.to_matrix()
            elif not isinstance(observable,np.ndarray) and hasattr(observable,"data"):raw=observable.data
    if scope is None:
        data=np.asarray(data,dtype=complex)
        if factory is None:raw=np.asarray(raw,dtype=complex)
    shape=(local_size,local_size) if factory is not None else numeric_shape(raw) if scope is not None else raw.shape
    data_shape=numeric_shape(data) if scope is not None else data.shape
    if shape!=(local_size,local_size):raise ValueError("Observable must be a dimension-preserving square matrix for these subsystems.")
    if data_shape!=((size,size) if density else (size,)):
        raise ValueError("State dimensions do not match its numeric buffer.")
    if scope is not None:
        needed=0 if density else array_layout((size,),complex)[2]
        if factory is not None:needed+=array_layout(shape,complex)[2]
        for value,layout in ((data,data_shape),) if factory is not None else ((data,data_shape),(raw,shape)):
            if not issubclass(type(value),np.ndarray) or np.ndarray.view(value,np.ndarray).dtype!=np.dtype(complex):
                needed+=array_layout(layout,complex)[2]
        scope.ledger.preflight(memory_bytes=needed)
        scope.ledger.unavailable("expectation_scalar_iterator_workspace",
            "Known numeric conversions and the pure-state action buffer are admitted; density uses direct indices. Python/scalar iterator storage is not measured RSS.")
        data=numeric_array(data,category="expectation_state_input")
        if factory is None:raw=numeric_array(raw,category="expectation_observable_input")
    matrix=factory() if factory is not None else raw
    return _expectation_local_operator(data,matrix,selected,dimensions,density=density)


def _state_data(state):
    from .density_matrix import DensityMatrix
    from quantumbridge.execution.array_reservations import numeric_shape
    from quantumbridge.core.classical import UnsupportedFeature
    scope=current_resources()
    if scope is not None and type(state) not in (Statevector,DensityMatrix):
        raise UnsupportedFeature("managed_state_math","Managed state mathematics requires a native state object.")
    dimensions=state_dimensions(state._dims)
    density=isinstance(state,DensityMatrix)
    size=prod(dimensions)
    raw=state.data
    if scope is not None:
        shape=numeric_shape(raw)
        if not issubclass(type(raw),np.ndarray):
            raise UnsupportedFeature("managed_state_buffer","Native state queries require explicit numeric ndarray storage.")
        data=np.ndarray.view(raw,np.ndarray)
    else:
        data=np.asarray(raw,dtype=complex)
        shape=data.shape
    if shape!=((size,size) if density else (size,)):
        raise ValueError("State dimensions do not match its numeric buffer.")
    return data,dimensions,density


def _state_operator_data(state, expected_type):
    """Qualify a native coordinate owner before publishing matrix provenance."""
    from quantumbridge.core.classical import UnsupportedFeature
    from quantumbridge.core.composition import _admit_graph
    from quantumbridge.core.gate_limits import current_gate_limits
    if type(state) is not expected_type:
        raise UnsupportedFeature("state_operator_origin", "Operator construction requires the native state type.")
    fields = object.__getattribute__(state, "__dict__")
    data, dims = fields.get("data"), fields.get("_dims")
    if type(dims) is not tuple or any(type(dim) is not int or dim < 1 for dim in dims):
        raise ValueError("State operator construction requires native dimension descriptors.")
    if type(data) is not np.ndarray or data.dtype.kind not in "biufc":
        raise UnsupportedFeature("state_operator_storage")
    size = prod(dims)
    shape = (size,) if expected_type is Statevector else (size, size)
    if data.shape != shape:
        raise ValueError("State dimensions do not match its numeric buffer.")
    current_gate_limits().check("max_matrix_dimension", size)
    current_gate_limits().check("max_decomposition_work", size * size)
    _admit_graph(state)
    return _state_data(state)


def _state_operand(value, size, *, density):
    from .density_matrix import DensityMatrix
    from quantumbridge.execution.array_reservations import numeric_shape
    scope=current_resources()
    dimensions=None
    native=type(value) in (Statevector,DensityMatrix) if scope is not None else isinstance(value,(Statevector,DensityMatrix))
    if native:
        raw,dimensions,is_density=_state_data(value)
        if is_density and not density:raise ValueError("Statevector arithmetic requires a vector operand.")
        shape=raw.shape
    else:
        raw=value if scope is not None else np.asarray(value,dtype=complex)
        shape=numeric_shape(raw) if scope is not None else raw.shape
    # A 1x1 density operand is a matrix, not another pure-state outer product.
    vector=shape in ((size,),(size,1)) and not (density and shape==(size,size))
    if not (vector or density and shape==(size,size)):
        raise ValueError("State arithmetic operands must have matching total dimensions.")
    return raw,shape,vector,dimensions


def _state_arrays(values, *, output_bytes=0, ledger=None):
    from quantumbridge.execution.array_reservations import numeric_array, numeric_shape, array_layout
    scope=current_resources()
    needed=output_bytes
    if scope is not None:
        for value in values:
            shape=numeric_shape(value)
            if not issubclass(type(value),np.ndarray) or np.ndarray.view(value,np.ndarray).dtype!=np.dtype(complex):
                needed+=array_layout(shape,complex)[2]
    if ledger is not None:
        ledger.preflight(memory_bytes=needed)
        ledger.unavailable("state_math_iterator_workspace",
            "Explicit conversions and outputs are admitted; scalar/ufunc iterator and Python storage are not measured RSS.")
    return tuple(numeric_array(value,category="state_math_input") if scope is not None
        else np.asarray(value,dtype=complex) for value in values)


def _state_scalar(value, *, reciprocal=False):
    from quantumbridge.execution.array_reservations import numeric_shape
    if current_resources() is not None:
        if numeric_shape(value)!=():raise TypeError("State scalar arithmetic requires a number.")
    if issubclass(type(value),np.ndarray):
        if not reciprocal or numeric_shape(value)!=():raise TypeError("State multiplication requires a numeric scalar, not an ndarray.")
        value=np.ndarray.view(value,np.ndarray)[()]
    if not isinstance(value,Number):return NotImplemented
    value=1/value if reciprocal else value
    if current_resources() is None and not isinstance(value,(bool,int,float,complex,np.number)):
        value=complex(value)
    return value


def _state_arithmetic(state, other, operation):
    from quantumbridge.execution.array_reservations import array_layout
    from quantumbridge.execution.array_ownership import retained_array, storage_ledger
    data,dimensions,density=_state_data(state)
    scope=current_resources()
    ledger=storage_ledger(data) if scope is None else scope.ledger
    vector=False
    if operation in ("add","subtract"):
        raw,shape,vector,_=_state_operand(other,prod(dimensions),density=density)
        inputs=(data,raw)
    else:
        other=_state_scalar(other,reciprocal=operation=="divide")
        if other is NotImplemented:return NotImplemented
        inputs=(data,)
    arrays=_state_arrays(inputs,output_bytes=array_layout(data.shape,complex)[2],ledger=ledger)
    data=arrays[0]
    output=retained_array(ledger,"state_arithmetic_output",data.shape,complex)
    if len(arrays)==1:
        np.multiply(data,other,out=output)
    else:
        right=arrays[1]
        if vector and right.ndim==2:right=right[:,0]
        if density and vector:
            sign=1 if operation=="add" else -1
            for row in range(len(data)):
                for column in range(len(data)):
                    output[row,column]=data[row,column]+sign*right[row]*right[column].conjugate()
        else:
            (np.add if operation=="add" else np.subtract)(data,right,out=output)
    return output,dimensions


def _state_norm(data):
    length=np.float64(0)
    for entry in data:
        if np.isnan(entry):return np.float64(np.nan)
        length=np.hypot(length,np.abs(entry))
    return length


def _state_trace(state, *, square=False):
    data,_,density=_state_data(state)
    if density:
        value=(sum(data[row,column]*data[column,row] for row in range(len(data)) for column in range(len(data)))
            if square else sum(data[index,index] for index in range(len(data))))
        return np.complex128(value)
    length=_state_norm(data)
    trace=length*length
    return np.float64(trace*trace if square else trace)


def _state_inner(state, other):
    data,dimensions,_=_state_data(state)
    raw,shape,_,other_dims=_state_operand(other,len(data),density=False)
    if (state_dimensions(len(data)) if other_dims is None else other_dims)!=dimensions:
        raise ValueError("Statevector inner product requires matching subsystem dimensions.")
    scope=current_resources()
    data,right=_state_arrays((data,raw),ledger=None if scope is None else scope.ledger)
    if len(shape)==2:right=right[:,0]
    return np.complex128(sum(left.conjugate()*value for left,value in zip(data,right)))


def _state_dictionary(state, decimals):
    data,dimensions,density=_state_data(state)
    if decimals is not None:
        if current_resources() is not None:
            from quantumbridge.execution.array_reservations import numeric_shape
            numeric_shape(decimals,real_only=True)
        if not isinstance(decimals,Integral):raise TypeError("State dictionary decimals must be an integer or None.")
        decimals=int(decimals)
    separator="," if len(dimensions)>1 and any(dim>10 for dim in dimensions) else ""
    def label(index):
        return separator.join(str(digit) for digit in basis_digits(index,dimensions)) if dimensions else "0"
    scope=current_resources()
    if scope is not None:
        scope.ledger.unavailable("state_dictionary_python_storage",
            "Dictionary entries and scalar rounding are Python/scalar storage, not retained ndarray bytes or measured RSS.")
    result={}
    for index,entry in enumerate(data.flat):
        entry=np.complex128(entry)
        if decimals is not None:entry=np.round(entry,decimals=decimals)
        if entry==0:continue
        if density:
            row,column=divmod(index,len(data))
            key=label(column)+"|"+label(row)
        else:key=label(index)
        result[key]=entry
    return result


def _state_valid(state, atol, rtol):
    from quantumbridge.execution.array_ownership import retained_array
    data,_,density=_state_data(state)
    atol,rtol=1e-8 if atol is None else atol,1e-5 if rtol is None else rtol
    if current_resources() is not None:
        from quantumbridge.execution.array_reservations import numeric_shape
        numeric_shape(atol,real_only=True)
        numeric_shape(rtol,real_only=True)
    atol,rtol=float(atol),float(rtol)
    if not isfinite(atol) or not isfinite(rtol) or atol<0 or rtol<0:
        raise ValueError("State validity tolerances must be finite and nonnegative.")
    if any(not np.isfinite(entry) for entry in data.flat):return False
    if not density:return bool(np.isclose(_state_norm(data),1.,atol=atol,rtol=rtol))
    for row in range(len(data)):
        for column in range(len(data)):
            reference=data[column,row].conjugate()
            if abs(data[row,column]-reference)>atol+rtol*abs(reference):return False
    if not np.isclose(sum(data[index,index] for index in range(len(data))),1.,atol=atol,rtol=rtol):return False
    scope=current_resources()
    ledger=None if scope is None else scope.ledger
    needs_copy=data.dtype!=np.dtype(complex) or not (data.flags.c_contiguous or data.flags.f_contiguous)
    if ledger is not None:
        ledger.preflight(memory_bytes=len(data)*8+(data.size*16 if needs_copy else 0))
        ledger.unavailable("density_validity_lapack_workspace",
            "Known input packing and eigenvalue output are admitted; unexposed NumPy/LAPACK workspace is not measured RSS.")
    if needs_copy:
        packed=retained_array(ledger,"density_validity_input",data.shape,complex,
            lambda:np.empty(data.shape,complex,order="F"))
        np.copyto(packed,data,casting="unsafe")
        data=packed
    values=retained_array(ledger,"density_validation_eigenvalues",(len(data),),float,
        lambda:np.linalg.eigvalsh(data))
    return bool(np.min(values)>=-atol)


class Statevector:
    def __init__(self, data, normalize: bool = False, *, dims=None):
        from quantumbridge.core import Circuit, Instruction, Gate, ControlledGate
        from quantumbridge.core.annotated_operation import AnnotatedOperation
        if type(data) is AnnotatedOperation: data = Circuit.from_instruction(data)
        from quantumbridge.execution.resource_context import current_resources
        scope = current_resources()
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
        if len(shape)!=1:
            raise ValueError("QuantumBridge Statevector data must be one-dimensional.")
        self._dims=state_dimensions(dims,size=shape[0])
        if scope is not None:
            self.data = numeric_array(data, category="statevector_input_conversion")
        if normalize:
            if scope is None:norm = np.linalg.norm(self.data)
            else:
                from quantumbridge.execution import mps_arrays as arrays
                norm = np.sqrt(arrays.norm_squared(self.data))
            if norm == 0:
                raise ValueError("QuantumBridge cannot normalize a zero Statevector.")
            if scope is None:self.data = self.data / norm
            else:
                output = arrays.owned("statevector_normalized", self.data.shape, complex)
                np.divide(self.data, norm, out=output)
                self.data = output
        self.num_qubits = qubit_count(self._dims)

    @property
    def dim(self) -> int:
        return len(self.data)

    def dims(self, qargs=None) -> tuple[int, ...]:
        return tuple(self._dims[position] for position in _validate_qargs(qargs,len(self._dims)))

    @classmethod
    def from_circuit(cls, circuit) -> "Statevector":
        return cls(StatevectorDevice().statevector(circuit))

    @classmethod
    def from_instruction(cls, instruction) -> "Statevector":
        from quantumbridge.core import Circuit, Instruction, Gate, ControlledGate
        from quantumbridge.core.annotated_operation import AnnotatedOperation
        if type(instruction) is AnnotatedOperation: instruction = Circuit.from_instruction(instruction)
        from quantumbridge.core.gates import is_native_instruction
        from quantumbridge.core.circuit import is_native_circuit
        if current_resources() is not None and not is_native_circuit(instruction) and not is_native_instruction(instruction):
            from quantumbridge.core.classical import UnsupportedFeature
            raise UnsupportedFeature("managed_state_instruction","Managed state construction requires a native Circuit or Instruction.")
        if isinstance(instruction,Instruction):
            instruction = Circuit.from_instruction(instruction)
        if not isinstance(instruction,Circuit):
            raise TypeError("Statevector.from_instruction requires a native Circuit or Instruction.")
        return cls.from_circuit(instruction)

    @classmethod
    def from_int(cls, index: int, dims) -> "Statevector":
        if isinstance(index,bool) or not isinstance(index,Integral):
            raise ValueError("Statevector.from_int index must be an integer.")
        dimensions=state_dimensions(dims)
        dim=prod(dimensions)
        if index < 0 or index >= dim:
            raise ValueError("QuantumBridge Statevector.from_int index is outside dims.")
        from quantumbridge.execution import mps_arrays as arrays
        data=arrays.filled("statevector_basis",(dim,),complex,0)
        data[int(index)] = 1.0
        return cls(data,dims=dimensions)

    @classmethod
    def from_label(cls, label: str) -> "Statevector":
        if type(label) is not str or any(bit not in "01+-rl" for bit in label):
            raise ValueError("Statevector labels must contain Pauli eigenstates 0, 1, +, -, r or l.")
        from quantumbridge.execution import mps_arrays as arrays
        coefficients={"0":(1,0),"1":(0,1),"+":(1,1),"-":(1,-1),"r":(1,1j),"l":(1,-1j)}
        data=arrays.owned("statevector_label",(1<<len(label),),complex)
        normalization=2**(-.5*sum(symbol not in "01" for symbol in label))
        for index in range(len(data)):
            value=complex(normalization)
            for wire,symbol in enumerate(label):value*=coefficients[symbol][(index>>(len(label)-1-wire))&1]
            data[index]=value
        return cls(data)

    def is_valid(self, atol: float | None = None, rtol: float | None = None) -> bool:
        return _state_valid(self,atol,rtol)

    def trace(self) -> np.float64:
        return _state_trace(self)

    def purity(self) -> np.float64:
        return _state_trace(self,square=True)

    def probabilities(self, wires=None) -> dict[str, float]:
        selected=_validate_qargs(wires,len(self._dims))
        probabilities={}
        for index,amplitude in enumerate(self.data):
            label=basis_label(index,self._dims,selected)
            probabilities[label]=probabilities.get(label,0.)+float(abs(amplitude)**2)
        return probabilities

    def probabilities_dict(self, qargs=None) -> dict[str, float]:
        return self.probabilities(qargs)

    def expectation_value(self, observable, qargs=None) -> complex:
        from quantumbridge.operators import SparsePauliOperator
        from quantumbridge.core.classical import UnsupportedFeature
        scope=current_resources()
        if scope is not None and type(self) is not Statevector:
            raise UnsupportedFeature("managed_expectation_state","Managed expectation requires a native state with explicit numeric storage.")
        if type(observable) is SparsePauliOperator and qargs is None:
            require_qubits(self._dims,"mixed_radix_hamiltonian")
            if observable.terms and observable.num_qubits!=self.num_qubits:
                raise ValueError("Sparse Pauli observable width must match the state.")
            data=self.data
            if scope is not None:
                from quantumbridge.execution.array_reservations import numeric_shape, numeric_array
                if numeric_shape(data)!=(prod(self._dims),):raise ValueError("State dimensions do not match its numeric buffer.")
                data=numeric_array(data,category="sparse_expectation_state_input")
            if all(coefficient.imag==0 for _,coefficient in observable.terms):
                return expectation_hamiltonian(data,self.num_qubits,observable.to_hamiltonian())
            from quantumbridge.utils.math import expectation_pauli_string, PauliString
            return np.complex128(sum(coefficient*expectation_pauli_string(data,self.num_qubits,
                PauliString((wire,symbol) for wire,symbol in enumerate(label) if symbol!="I"))
                for label,coefficient in observable.terms))
        if scope is None and qargs is None and hasattr(observable,"to_hamiltonian"):
            require_qubits(self._dims,"mixed_radix_hamiltonian")
            return expectation_hamiltonian(self.data, self.num_qubits, observable.to_hamiltonian())
        return _expectation_value(self.data,self._dims,observable,qargs)

    def evolve(self, operator, qargs=None) -> "Statevector":
        from quantumbridge.core import Circuit, Instruction
        from quantumbridge.information.operator import Operator
        from quantumbridge.execution.resource_context import current_resources
        scope=current_resources()

        if isinstance(operator,Instruction):
            require_qubits(self._dims,"mixed_radix_circuit_evolution")
            if qargs is not None:
                qargs = _validate_qargs(qargs,self.num_qubits)
                if len(qargs)!=operator.num_qubits:
                    raise ValueError("Instruction target count must match instruction width.")
            operator = Circuit.from_instruction(operator)
        if hasattr(operator,"operations"):require_qubits(self._dims,"mixed_radix_circuit_evolution")
        op = Operator.from_circuit(operator) if hasattr(operator, "operations") else operator
        native=type(op) is Operator if scope is not None else isinstance(op,Operator)
        if qargs is not None or native and op.input_dims()!=op.output_dims():
            matrix,selected,outgoing,result_dims=_mapped_evolution_data(op,qargs,self._dims)
            return Statevector(apply_unitary(self.data,matrix,selected,self.num_qubits,dims=self._dims,output_dims=outgoing),dims=result_dims)
        if scope is not None:
            from quantumbridge.information.channels import _operator_data
            matrix=_operator_data(op)
        else:matrix = np.asarray(op.to_matrix(), dtype=complex) if hasattr(op, "to_matrix") else np.asarray(op, dtype=complex)
        if matrix.shape != (len(self.data), len(self.data)):
            raise ValueError("QuantumBridge evolution operator dimension must match Statevector dimension.")
        if scope is not None:
            from quantumbridge.execution import mps_arrays as arrays
            return Statevector(arrays.contracted(matrix,self.data),dims=self._dims)
        return Statevector(matrix @ self.data,dims=self._dims)

    def tensor(self, other) -> "Statevector":
        other_state = other if isinstance(other, Statevector) else Statevector(other)
        return Statevector(_tensor_arrays(self.data, other_state.data),dims=self._dims+other_state._dims)

    def expand(self, other) -> "Statevector":
        other_state = other if isinstance(other, Statevector) else Statevector(other)
        return Statevector(_tensor_arrays(other_state.data, self.data),dims=other_state._dims+self._dims)

    def reverse_qargs(self) -> "Statevector":
        from quantumbridge.execution.array_ownership import copied_array
        data=copied_array(self.data,category="statevector_reverse")
        for source, amplitude in enumerate(self.data):
            target=basis_index(basis_digits(source,self._dims)[::-1],self._dims[::-1])
            data[target] = amplitude
        return Statevector(data,dims=self._dims[::-1])

    def inner(self, other) -> complex:
        return _state_inner(self,other)

    def equiv(self, other, atol: float = 1e-10) -> bool:
        other_state = other if isinstance(other, Statevector) else Statevector(other)
        if len(self.data) != len(other_state.data):
            return False
        if np.allclose(self.data, other_state.data, atol=atol):
            return True
        nonzero = np.argwhere(np.abs(other_state.data) > atol)
        if nonzero.size == 0:
            return bool(np.allclose(self.data, other_state.data, atol=atol))
        index = int(nonzero[0][0])
        phase = self.data[index] / other_state.data[index]
        if abs(abs(phase) - 1.0) > atol:
            return False
        return bool(np.allclose(self.data, phase * other_state.data, atol=atol))

    @meter_native_samples(result_count=lambda counts:sum(counts.values()))
    def sample_counts(self, shots: int, qargs=None, seed: int | None = None) -> dict[str, int]:
        # Keep the old positional integer seed while accepting subsystem lists.
        if isinstance(qargs,Integral) and seed is None:seed,qargs=int(qargs),None
        selected=_validate_qargs(qargs,len(self._dims))
        return _counts_from_indices(*_sample_measurements(self,selected,shots,seed))

    @meter_native_samples
    def sample_memory(self, shots: int, qargs=None, seed: int | None = None) -> list[str]:
        qargs_tuple = _validate_qargs(qargs,len(self._dims))
        return _memory_from_indices(*_sample_measurements(self,qargs_tuple,shots,seed))

    def measure(self, qargs=None, seed: int | None = None) -> tuple[str, "Statevector"]:
        qargs_tuple = _validate_qargs(qargs,len(self._dims))
        if current_resources() is not None:
            current_resources().ledger.preflight(memory_bytes=self.data.nbytes)
        memory = self.sample_memory(1, qargs=qargs_tuple, seed=seed)
        outcome = memory[0] if memory else ""
        from quantumbridge.execution.array_ownership import copied_array
        collapsed=copied_array(self.data,category="statevector_projection")
        for index in range(len(collapsed)):
            if basis_label(index,self._dims,qargs_tuple) != outcome:
                collapsed[index] = 0.0
        if current_resources() is None:norm=np.linalg.norm(collapsed)
        else:
            from quantumbridge.execution import mps_arrays as arrays
            norm=np.sqrt(arrays.norm_squared(collapsed))
        if norm == 0:
            raise ValueError("QuantumBridge measurement produced a zero-probability collapse.")
        np.divide(collapsed,norm,out=collapsed)
        return outcome, Statevector(collapsed,dims=self._dims)

    def reset(self, qargs=None, seed: int | None = None) -> "Statevector":
        selected=_validate_qargs(qargs,len(self._dims))
        if qargs is None:
            return Statevector.from_int(0,self._dims)
        if not selected:return self.copy()
        _,collapsed=self.measure(selected,seed=seed)
        # Zeroing measured digits maps uniquely to earlier basis indices.
        for index in range(len(collapsed.data)):
            digits=list(basis_digits(index,self._dims))
            for position in selected:digits[position]=0
            target=basis_index(digits,self._dims)
            if target!=index and collapsed.data[index]!=0:
                collapsed.data[target]=collapsed.data[index]
                collapsed.data[index]=0
        return collapsed

    def to_operator(self):
        from quantumbridge.information.operator import Operator
        from quantumbridge.execution.array_ownership import retained_array, storage_ledger
        from quantumbridge.execution.array_reservations import array_layout
        data,dimensions,_=_state_operator_data(self,Statevector)
        scope=current_resources()
        ledger=storage_ledger(data) if scope is None else scope.ledger
        shape=(len(data),len(data))
        data,=_state_arrays((data,),output_bytes=array_layout(shape,complex)[2],ledger=ledger)
        result=retained_array(ledger,"statevector_operator_projector",shape,complex)
        for column in range(len(data)):
            np.multiply(data,complex(data[column]).conjugate(),out=result[:,column])
        operator=Operator(result,input_dims=dimensions,output_dims=dimensions)
        operator._public_matrix_basis="native"
        return operator

    def to_density_matrix(self):
        from quantumbridge.information.density_matrix import DensityMatrix

        return DensityMatrix.from_statevector(self)

    def partial_trace(self, keep: tuple[int, ...] | list[int]):
        from quantumbridge.information.density_matrix import DensityMatrix

        return DensityMatrix.from_statevector(self).partial_trace(keep)

    def conjugate(self) -> "Statevector":
        copied=self.copy()
        np.conjugate(copied.data,out=copied.data)
        return copied

    def transpose(self) -> "Statevector":
        return self.copy()

    def adjoint(self) -> np.ndarray:
        return self.conjugate().data.reshape(1,-1)

    def copy(self) -> "Statevector":
        from quantumbridge.execution.array_ownership import copied_array
        return Statevector(copied_array(self.data,category="statevector_copy"),dims=self._dims)

    def to_dict(self, decimals: int | None = None) -> dict[str, complex]:
        return _state_dictionary(self,decimals)

    def __len__(self) -> int:
        return len(self.data)

    def __add__(self, other):
        data,dimensions=_state_arithmetic(self,other,"add")
        return Statevector(data,dims=dimensions)

    def __sub__(self, other):
        data,dimensions=_state_arithmetic(self,other,"subtract")
        return Statevector(data,dims=dimensions)

    def __neg__(self):
        return self.__mul__(-1)

    def __mul__(self, scalar):
        result=_state_arithmetic(self,scalar,"multiply")
        if result is NotImplemented:return NotImplemented
        data,dimensions=result
        return Statevector(data,dims=dimensions)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        result=_state_arithmetic(self,scalar,"divide")
        if result is NotImplemented:return NotImplemented
        data,dimensions=result
        return Statevector(data,dims=dimensions)

    def __eq__(self, other) -> bool:
        try:
            return self.equiv(other)
        except Exception:
            return False
