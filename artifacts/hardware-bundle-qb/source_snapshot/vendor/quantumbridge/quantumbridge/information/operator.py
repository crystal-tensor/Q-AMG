# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from cmath import phase as scalar_phase, rect
from math import isfinite, prod
from numbers import Number

import numpy as np

from quantumbridge.core.operations import Instruction
from quantumbridge.utils.math import apply_unitary, gate_matrix
from .state_dimensions import state_dimensions, qubit_count, subsystem_positions, basis_digits, basis_index


def _validate_qargs(qargs, num_qubits: int, width: int) -> tuple[int, ...]:
    normalized = tuple(int(qarg) for qarg in qargs)
    if len(normalized) != width:
        raise ValueError("QuantumBridge Operator qargs width must match the composed operator.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("QuantumBridge Operator qargs must be distinct.")
    if any(qarg < 0 or qarg >= num_qubits for qarg in normalized):
        raise ValueError("QuantumBridge Operator qargs are outside this operator.")
    return normalized


def _index_to_bits(index: int, num_qubits: int) -> list[int]:
    return [(index >> (num_qubits - 1 - wire)) & 1 for wire in range(num_qubits)]


def _bits_to_index(bits: list[int] | tuple[int, ...]) -> int:
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


def _expand_subsystem_operator(data: np.ndarray, qargs: tuple[int, ...], num_qubits: int) -> np.ndarray:
    size = 1 << num_qubits
    sub_dim = 1 << len(qargs)
    from quantumbridge.execution.resource_context import current_resources
    if current_resources() is None:out = np.zeros((size, size), dtype=complex)
    else:
        from quantumbridge.execution.array_reservations import numeric_array, numeric_shape
        from quantumbridge.execution import mps_arrays as arrays
        if numeric_shape(data)!=(sub_dim,sub_dim):raise ValueError("Subsystem operator shape does not match qargs.")
        data=numeric_array(data,category="subsystem_operator_input")
        out=arrays.filled("subsystem_operator_expansion",(size,size),complex,0)
    for col in range(size):
        basis_bits = _index_to_bits(col, num_qubits)
        sub_col = _bits_to_index([basis_bits[qarg] for qarg in qargs])
        for sub_row in range(sub_dim):
            row_bits = basis_bits.copy()
            replacement_bits = _index_to_bits(sub_row, len(qargs))
            for offset, qarg in enumerate(qargs):
                row_bits[qarg] = replacement_bits[offset]
            row = _bits_to_index(row_bits)
            out[row, col] += data[sub_row, sub_col]
    return out


def _reverse_qubit_order(data: np.ndarray, num_qubits: int) -> np.ndarray:
    return _permute_qubit_axes(data, num_qubits, tuple(reversed(range(num_qubits))))


def _permute_qubit_axes(data, num_qubits, axes):
    from quantumbridge.execution.array_ownership import copied_array
    size = 1 << num_qubits
    output=copied_array(data,category="operator_reverse")
    order=tuple(_bits_to_index([_index_to_bits(index,num_qubits)[axis] for axis in axes]) for index in range(size))
    for row,destination in enumerate(order):
        for column,target in enumerate(order):output[destination,target]=data[row,column]
    return output


def _arithmetic_convention(owner):
    from quantumbridge.core.classical import UnsupportedFeature
    if type(owner) is not Operator:
        raise UnsupportedFeature("operator_arithmetic_owner", "Arithmetic requires the native Operator type, not foreign subclasses.")
    fields = object.__getattribute__(owner, "__dict__")
    mode = fields.get("_arithmetic_convention", "native")
    if type(mode) is not str or mode not in ("native", "qiskit"):
        raise UnsupportedFeature("operator_arithmetic_convention")
    return mode


def _explicit_arithmetic(owner):
    _arithmetic_convention(owner)
    return "_arithmetic_convention" in object.__getattribute__(owner, "__dict__")


def _representation_admission(owner):
    explicit = _explicit_arithmetic(owner)
    fields = object.__getattribute__(owner, "__dict__")
    basis = fields.get("_public_matrix_basis")
    return explicit or "_literal_target_width" in fields or type(basis) is str and basis == "hybrid"


def _operator_matrix_origin(source, *, from_matrix=False):
    """Classify a validated construction path, never the matrix's values."""
    from quantumbridge.core.gates import Gate
    from quantumbridge.core.standard_gates import UnitaryGate
    from quantumbridge.core.operations import _matrix_owner_family
    if type(source) in (UnitaryGate, Gate):
        basis = object.__getattribute__(source, "__dict__").get("_public_matrix_basis")
        if basis is not None and (type(basis) is not str or basis not in ("little", "native", "literal_wrapper")):
            from quantumbridge.core.classical import UnsupportedFeature
            raise UnsupportedFeature("public_operator_matrix_origin")
    literal = (type(source) is UnitaryGate and (
        basis is None or from_matrix and basis == "little")
        or type(source) is Gate and basis == "literal_wrapper"
        and not _matrix_owner_family(source))
    if literal:return None
    if isinstance(source, Instruction):
        if _matrix_owner_family(source, public_unmarked=True) and not _matrix_owner_family(source):
            return "unresolved"
    return "native"


def _admit_public_arithmetic_owner(owner):
    """Qualify fields before numeric conversion, origin equality or callbacks."""
    from quantumbridge.core.classical import UnsupportedFeature
    from quantumbridge.core.gate_limits import current_gate_limits
    from quantumbridge.core.composition import _admit_graph
    mode = _arithmetic_convention(owner)
    fields = object.__getattribute__(owner, "__dict__")
    basis = fields.get("_public_matrix_basis")
    if basis is not None and (type(basis) is not str or basis not in ("native", "unresolved", "hybrid")):
        raise UnsupportedFeature("public_operator_matrix_origin")
    if basis == "unresolved" or mode == "qiskit" and basis is not None:
        raise UnsupportedFeature("public_operator_matrix_origin")
    data, incoming, outgoing = (fields.get(key) for key in ("data", "_input_dims", "_output_dims"))
    if type(data) is not np.ndarray or data.dtype.kind not in "biufc":
        raise UnsupportedFeature("operator_arithmetic_storage")
    for dims in (incoming, outgoing):
        if type(dims) is not tuple or any(type(dim) is not int or dim < 1 for dim in dims):
            raise ValueError("Operator arithmetic requires native dimension descriptors.")
    if data.shape != (prod(outgoing), prod(incoming)):
        raise ValueError("Operator dimensions do not match its numeric buffer.")
    current_gate_limits().check("max_matrix_dimension", max(data.shape))
    current_gate_limits().check("max_decomposition_work", prod(data.shape))
    expected_qubits = qubit_count(incoming) if incoming == outgoing else None
    width = fields.get("num_qubits")
    if expected_qubits is None:
        if width is not None:raise ValueError("Operator qubit width differs from its dimensions.")
    elif type(width) is not int or width != expected_qubits:
        raise ValueError("Operator qubit width differs from its dimensions.")
    if basis == "hybrid":
        targets = fields.get("_literal_target_width")
        if data.dtype != np.dtype(complex) or type(targets) is not int or expected_qubits is None or not 0 < targets < expected_qubits:
            raise UnsupportedFeature("public_operator_matrix_origin", "Hybrid target/control widths are inconsistent.")
    elif "_literal_target_width" in fields:
        raise UnsupportedFeature("public_operator_matrix_origin", "Hybrid descriptor does not match its basis.")
    _admit_graph(owner)
    return basis


def _explicit_tensor_operand(owner, other):
    from quantumbridge.core.classical import UnsupportedFeature
    from quantumbridge.core.gate_limits import current_gate_limits
    from quantumbridge.execution.array_reservations import numeric_shape, array_layout
    from quantumbridge.execution.resource_context import current_resources
    _admit_public_arithmetic_owner(owner)
    mode = _arithmetic_convention(owner)
    typed = isinstance(other, Operator)
    if typed:
        if _arithmetic_convention(other) != mode:
            raise UnsupportedFeature("operator_arithmetic_convention")
        _admit_public_arithmetic_owner(other)
        raw = other.data
    else:
        raw = other
    shape = numeric_shape(raw)
    if len(shape) != 2 or not all(shape):
        raise ValueError("Operator tensor requires a nonempty matrix.")
    output_shape = (owner.data.shape[0]*shape[0], owner.data.shape[1]*shape[1])
    limits = current_gate_limits()
    limits.check("max_matrix_dimension", max(output_shape))
    limits.check("max_decomposition_work", prod(output_shape))
    scope = current_resources()
    if scope is not None:
        # Input conversion and the tensor destination are one known allocation
        # group, even when a raw float/list operand needs a complex buffer.
        view = np.ndarray.view(raw, np.ndarray) if issubclass(type(raw), np.ndarray) else None
        snapshot = typed and mode == "native" and not _explicit_arithmetic(other)
        conversion = array_layout(shape, complex)[2] if snapshot else 0 if typed or view is not None and view.dtype == np.dtype(complex) else array_layout(shape, complex)[2]
        scope.ledger.preflight(memory_bytes=array_layout(output_shape, complex)[2] + conversion)
    if not typed:
        other = Operator(raw)
        other._arithmetic_convention = mode
        if mode == "native":other._public_matrix_basis = "native"
    elif mode == "native" and not _explicit_arithmetic(other):
        other = other.as_native_operator()
    return other


class Operator:
    """Dense linear operator with a Qiskit quantum_info-like core API."""

    __array_priority__ = 1000

    def __setattr__(self, name, value):
        if name in ("_arithmetic_convention", "_literal_target_width") and name in self.__dict__:
            raise AttributeError("Operator representation selection is immutable; select a new view.")
        object.__setattr__(self, name, value)

    @property
    def arithmetic_convention(self):
        return _arithmetic_convention(self)

    def as_qiskit_operator(self):
        """Return an owned, explicit public-arithmetic Operator of this type.

        Legacy constructors and arithmetic retain native order. Raw matrices
        are literals; only a proved circuit-native origin needs permutation.
        """
        basis = _admit_public_arithmetic_owner(self)
        from quantumbridge.execution.array_reservations import numeric_array
        if basis == "hybrid":
            controls = self.num_qubits - self._literal_target_width
            axes = (*range(controls, self.num_qubits), *reversed(range(controls)))
            data = _permute_qubit_axes(self.data, self.num_qubits, axes)
            incoming, outgoing = self._input_dims, self._output_dims
        elif basis == "native":
            converted = self.reverse_qargs()
            data, incoming, outgoing = converted.data, converted._input_dims, converted._output_dims
        else:
            data = numeric_array(self.data, copy=True, category="operator_public_copy")
            incoming, outgoing = self._input_dims, self._output_dims
        result = Operator(data, incoming, outgoing)
        result._arithmetic_convention = "qiskit"
        return result

    def as_native_operator(self):
        """Own an explicit high-first native physical representation.

        Legacy native data is unchanged. Public data uses the existing reverse
        subsystem map once, including its input/output dimension descriptors.
        """
        basis = _admit_public_arithmetic_owner(self)
        if basis == "hybrid":
            controls = self.num_qubits - self._literal_target_width
            axes = (*range(controls), *reversed(range(controls, self.num_qubits)))
            data = _permute_qubit_axes(self.data, self.num_qubits, axes)
            incoming, outgoing = self._input_dims, self._output_dims
        elif _arithmetic_convention(self) == "qiskit":
            converted = self.reverse_qargs()
            data, incoming, outgoing = converted.data, converted._input_dims, converted._output_dims
        else:
            from quantumbridge.execution.array_reservations import numeric_array
            data = numeric_array(self.data, copy=True, category="operator_native_copy")
            incoming, outgoing = self._input_dims, self._output_dims
        result = Operator(data, incoming, outgoing)
        result._arithmetic_convention = "native"
        result._public_matrix_basis = "native"
        return result

    def __init__(self, data, input_dims=None, output_dims=None):
        from quantumbridge.core import Circuit
        from quantumbridge.execution.resource_context import current_resources
        scope=current_resources()
        from quantumbridge.core.gates import Gate, ControlledGate
        from quantumbridge.core.annotated_operation import AnnotatedOperation
        from quantumbridge.information.clifford import Clifford, _clifford_matrix_origin
        if issubclass(type(data), Clifford):
            origin = _clifford_matrix_origin(data, dense=True)
            dimensions = (2,) * data.num_qubits
            if any(value is not None and state_dimensions(value, size=1 << data.num_qubits) != dimensions
                for value in (input_dims, output_dims)):
                from quantumbridge.core.classical import UnsupportedFeature
                raise UnsupportedFeature("clifford_operator_partition", "Clifford construction requires its qubit partition.")
            converted = data.to_operator()
            if origin is not None:self._public_matrix_basis = origin
            data = converted.data
        if type(data) is AnnotatedOperation:
            from quantumbridge.core.annotated_operation import _annotation_literal_target_width
            if _annotation_literal_target_width(data) is not None:
                dimensions = (2,) * data.num_qubits
                if any(value is not None and state_dimensions(value, size=1 << data.num_qubits) != dimensions
                    for value in (input_dims, output_dims)):
                    from quantumbridge.core.classical import UnsupportedFeature
                    raise UnsupportedFeature("public_operator_matrix_origin", "Hybrid construction requires its qubit partition.")
                converted = data.to_matrix()
                self._public_matrix_basis = converted._public_matrix_basis
                self._literal_target_width = converted._literal_target_width
                data = converted.data
            else:data = Circuit.from_instruction(data)
        from quantumbridge.core.gates import is_native_instruction
        from quantumbridge.core.circuit import is_native_circuit
        if (is_native_circuit(data) or is_native_instruction(data) if scope is not None else isinstance(data,(Circuit,Instruction))):
            converted = self.from_circuit(data)
            data = converted.data
            if hasattr(converted,"_public_matrix_basis"):
                self._public_matrix_basis = converted._public_matrix_basis
        if scope is None:
            self.data = np.asarray(data, dtype=complex)
            shape=self.data.shape
        else:
            from quantumbridge.execution.array_reservations import numeric_array, numeric_shape
            shape=numeric_shape(data)
        if len(shape)!=2 or not all(shape):
            raise ValueError("QuantumBridge Operator data must be a nonempty matrix.")
        self._input_dims=state_dimensions(shape[1] if input_dims is None else input_dims,size=shape[1])
        self._output_dims=state_dimensions(shape[0] if output_dims is None else output_dims,size=shape[0])
        self.num_qubits=qubit_count(self._input_dims) if self._input_dims==self._output_dims else None
        if scope is not None:self.data=numeric_array(data,category="operator_input_conversion")

    @property
    def dim(self) -> tuple[int, int]:
        return (self.data.shape[1], self.data.shape[0])

    def input_dims(self, qargs=None) -> tuple[int, ...]:
        dims = self._input_dims[::-1] if _arithmetic_convention(self) == "qiskit" else self._input_dims
        return tuple(dims[position] for position in subsystem_positions(qargs,len(dims)))

    def output_dims(self, qargs=None) -> tuple[int, ...]:
        dims = self._output_dims[::-1] if _arithmetic_convention(self) == "qiskit" else self._output_dims
        return tuple(dims[position] for position in subsystem_positions(qargs,len(dims)))

    @classmethod
    def from_label(cls, label: str) -> "Operator":
        if type(label) is not str or any(char not in "IXYZ" for char in label.upper()):
            raise ValueError("QuantumBridge Operator labels may contain only I, X, Y, Z.")
        from quantumbridge.execution import mps_arrays as arrays
        label=label.upper()
        out=arrays.filled("operator_pauli_label",(1<<len(label),)*2,complex,0)
        for column in range(len(out)):
            row,amplitude=column,1.+0j
            for wire,symbol in enumerate(label):
                mask=1<<(len(label)-1-wire)
                bit=bool(column&mask)
                if symbol in "XY":row^=mask
                if symbol=="Y":amplitude*=(-1j if bit else 1j)
                elif symbol=="Z" and bit:amplitude=-amplitude
            out[row,column]=amplitude
        return cls(out)

    @classmethod
    def from_circuit(cls, circuit) -> "Operator":
        from quantumbridge.core import Circuit
        from quantumbridge.core.circuit import is_native_circuit
        from quantumbridge.core.gates import is_native_instruction
        from quantumbridge.core.composition import _admit_graph
        if not is_native_circuit(circuit) and not is_native_instruction(circuit):
            raise TypeError("Operator.from_circuit requires a native circuit or instruction owner.")
        _admit_graph(circuit)
        basis=_operator_matrix_origin(circuit)
        if isinstance(circuit,Instruction):
            circuit = Circuit.from_instruction(circuit)
        circuit = circuit.lowered()
        if circuit.has_dynamic_operations or circuit.measurements:
            raise ValueError("QuantumBridge Operator cannot represent measurement, reset or classical control flow as a unitary.")
        size = 1 << circuit.num_qubits
        from quantumbridge.execution.resource_context import current_resources
        scope=current_resources()
        if scope is None:matrix = np.zeros((size, size), dtype=complex)
        else:
            from quantumbridge.execution import mps_arrays as arrays
            matrix=arrays.filled("circuit_operator_output",(size,size),complex,0)
        for col in range(size):
            state = np.zeros(size, dtype=complex) if scope is None else arrays.filled("circuit_operator_column",(size,),complex,0)
            state[col] = 1.0
            for op in circuit.operations:
                if op.metadata.get("directive") and op.name in {"barrier", "delay"}:
                    continue
                gate = gate_matrix(op.name, op.params, op.metadata)
                state = apply_unitary(state, gate, op.controls + op.targets, circuit.num_qubits)
            np.multiply(state,np.exp(1j*float(circuit.global_phase)),out=matrix[:,col])
        result=cls(matrix)
        if basis is not None:result._public_matrix_basis=basis
        return result

    def is_unitary(self, atol: float = 1e-10) -> bool:
        if self.data.shape[0]!=self.data.shape[1]:return False
        from quantumbridge.execution.resource_context import current_resources
        if current_resources() is not None:
            from quantumbridge.utils.math import validate_unitary_matrix
            return validate_unitary_matrix(self.data,atol=atol,rtol=1e-5)
        eye = np.eye(self.data.shape[0], dtype=complex)
        return bool(np.allclose(self.data.conj().T @ self.data, eye, atol=atol))

    def adjoint(self) -> "Operator":
        if _representation_admission(self):_admit_public_arithmetic_owner(self)
        return self._new(self.conjugate().data.T,input_dims=self._output_dims,output_dims=self._input_dims)

    def transpose(self) -> "Operator":
        if _representation_admission(self):_admit_public_arithmetic_owner(self)
        return self._new(self.data.T,input_dims=self._output_dims,output_dims=self._input_dims)

    def conjugate(self) -> "Operator":
        copied=self.copy()
        np.conjugate(copied.data,out=copied.data)
        return copied

    def copy(self) -> "Operator":
        if _representation_admission(self):_admit_public_arithmetic_owner(self)
        from quantumbridge.execution.array_ownership import copied_array
        return self._new(copied_array(self.data,category="operator_copy"))

    def _new(self, data, *, input_dims=None, output_dims=None, _matrix_basis="inherit"):
        if _representation_admission(self):_admit_public_arithmetic_owner(self)
        result=Operator(data,input_dims=self._input_dims if input_dims is None else input_dims,
                        output_dims=self._output_dims if output_dims is None else output_dims)
        basis=getattr(self,"_public_matrix_basis",None) if _matrix_basis=="inherit" else _matrix_basis
        if basis is not None:result._public_matrix_basis=basis
        if basis == "hybrid":result._literal_target_width = self._literal_target_width
        if _explicit_arithmetic(self):
            result._arithmetic_convention = _arithmetic_convention(self)
        return result

    def compose(self, other: "Operator", qargs=None, front: bool = False) -> "Operator":
        from quantumbridge.execution.resource_context import current_resources
        from quantumbridge.execution.array_reservations import numeric_shape, numeric_array, array_layout
        from quantumbridge.core.classical import UnsupportedFeature
        from quantumbridge.utils.math import _local_operator_shapes, _compose_local_operator
        scope=current_resources()
        public = _arithmetic_convention(self) == "qiskit"
        if _representation_admission(self):_admit_public_arithmetic_owner(self)
        if isinstance(other, Operator) and _representation_admission(other):_admit_public_arithmetic_owner(other)
        explicit = _explicit_arithmetic(self)
        if explicit:
            _admit_public_arithmetic_owner(self)
            from quantumbridge.core.composition import _admit_graph
            _admit_graph((qargs, front))
            if type(front) is not bool or qargs is not None and type(qargs) not in (tuple, list, range):
                raise ValueError("Explicit composition requires boolean front and builtin qargs.")
            if public:front = not front
            if qargs is not None:
                integer_types = (int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
                if any(not any(type(pos) is kind for kind in integer_types) for pos in qargs):
                    raise ValueError("Explicit qargs require native integer subsystem indices.")
                count = len(self._output_dims if front else self._input_dims)
                positions = subsystem_positions(qargs, count)
                qargs = tuple(count - 1 - pos for pos in reversed(positions)) if public else positions
        if isinstance(other, Operator):
            if _arithmetic_convention(other) != _arithmetic_convention(self):
                raise UnsupportedFeature("operator_arithmetic_convention", "Select the same explicit arithmetic convention on both operands.")
            if explicit:
                _admit_public_arithmetic_owner(other)
        elif explicit:
            numeric_shape(other)
        if scope is not None and type(self) is not Operator:
            raise UnsupportedFeature("managed_operator_compose","Managed composition requires native operators or explicit numeric buffers.")
        side=self._output_dims if front else self._input_dims
        selected=None if qargs is None else subsystem_positions(qargs,len(side))
        native=type(other) is Operator if scope is not None else isinstance(other,Operator)
        raw=other.data if native else other
        if scope is None and not explicit:raw=np.asarray(raw,dtype=complex)
        shape=numeric_shape(raw) if scope is not None or explicit else raw.shape
        if len(shape)!=2 or not all(shape):raise ValueError("Operator composition requires a nonempty matrix.")
        if native:other_input,other_output=other._input_dims,other._output_dims
        elif shape[0]==shape[1]:
            other_input=other_output=side if selected is None else tuple(side[position] for position in selected)
        else:
            if selected is not None:raise ValueError("Rectangular local composition requires explicit Operator dimensions.")
            other_input,other_output=state_dimensions(shape[1]),state_dimensions(shape[0])
        if shape!=(prod(other_output),prod(other_input)):
            raise ValueError("Operator composition dimensions do not match its numeric buffer.")
        contracted=other_input if front else other_output
        changed=other_output if front else other_input
        if selected is None:
            if contracted!=side:raise ValueError("QuantumBridge Operator composition subsystem dimensions do not match.")
            result_side=changed
        else:
            if len(other_input)!=len(selected) or len(other_output)!=len(selected):
                raise UnsupportedFeature("mixed_radix_output_subsystem_count","Mapped composition cannot insert or remove subsystem positions.")
            if contracted!=tuple(side[position] for position in selected):
                raise ValueError("Operator composition dimensions do not match qargs order.")
            result_side,_=_local_operator_shapes(side,selected,changed)
        incoming,outgoing=(self._input_dims,result_side) if front else (result_side,self._output_dims)
        host_shape=numeric_shape(self.data) if scope is not None else self.data.shape
        if host_shape!=(prod(self._output_dims),prod(self._input_dims)):
            raise ValueError("Operator dimensions do not match its numeric buffer.")
        if explicit:
            from quantumbridge.core.gate_limits import current_gate_limits
            limits = current_gate_limits()
            limits.check("max_matrix_dimension", max(prod(incoming), prod(outgoing)))
            limits.check("max_decomposition_work", prod(incoming) * prod(outgoing) * (shape[0] if not front else shape[1]))
        if scope is not None:
            needed=array_layout((prod(outgoing),prod(incoming)),complex)[2]
            if selected is not None:needed+=array_layout((host_shape[1] if front else host_shape[0],),complex)[2]
            for value,layout in ((self.data,host_shape),(raw,shape)):
                array=np.ndarray.view(value,np.ndarray) if issubclass(type(value),np.ndarray) else None
                if array is None or array.dtype!=np.dtype(complex) or selected is None and not (array.flags.c_contiguous or array.flags.f_contiguous):
                    needed+=array_layout(layout,complex)[2]
            scope.ledger.preflight(memory_bytes=needed)
            host_data=numeric_array(self.data,category="operator_compose_input")
            other_data=numeric_array(raw,category="operator_compose_input")
        else:
            host_data = self.data
            other_data = numeric_array(raw, category="operator_compose_input") if explicit else raw
        if selected is not None:
            data=_compose_local_operator(host_data,other_data,selected,side,changed,front=front)
        elif scope is not None:
            from quantumbridge.execution import mps_arrays as arrays
            data=arrays.contracted(other_data,host_data) if front else arrays.contracted(host_data,other_data)
        else:data=other_data @ host_data if front else host_data @ other_data
        # Explicit native calls fix both matrix coordinates and physical qargs.
        # Legacy ambiguous compounds retain their existing refusal boundary.
        basis=None if public else "native" if explicit else "unresolved" if getattr(self,"_public_matrix_basis",None) is not None or native and getattr(other,"_public_matrix_basis",None) is not None else None
        return self._new(data,input_dims=incoming,output_dims=outgoing,_matrix_basis=basis)

    def dot(self, other: "Operator", qargs=None) -> "Operator":
        return self.compose(other, qargs=qargs, front=_arithmetic_convention(self) == "qiskit")

    def tensor(self, other: "Operator") -> "Operator":
        public = _arithmetic_convention(self) == "qiskit"
        if _representation_admission(self):_admit_public_arithmetic_owner(self)
        if isinstance(other, Operator) and _representation_admission(other):_admit_public_arithmetic_owner(other)
        explicit = _explicit_arithmetic(self)
        if explicit:
            other = _explicit_tensor_operand(self, other)
        elif isinstance(other, Operator):
            if _arithmetic_convention(other) != _arithmetic_convention(self):
                from quantumbridge.core.classical import UnsupportedFeature
                raise UnsupportedFeature("operator_arithmetic_convention", "Select the same explicit arithmetic convention on both operands.")
        other = other if isinstance(other, Operator) else Operator(other)
        incoming,outgoing=self._input_dims+other._input_dims,self._output_dims+other._output_dims
        basis=None if public else "native" if explicit else "unresolved" if getattr(self,"_public_matrix_basis",None) is not None or getattr(other,"_public_matrix_basis",None) is not None else None
        if explicit:
            from quantumbridge.core.gate_limits import current_gate_limits
            current_gate_limits().check("max_matrix_dimension", max(prod(incoming), prod(outgoing)))
            current_gate_limits().check("max_decomposition_work", prod(incoming)*prod(outgoing))
        from quantumbridge.execution.resource_context import current_resources
        if current_resources() is not None:
            from quantumbridge.execution import mps_arrays as arrays
            from quantumbridge.execution.array_reservations import kron_into
            out=arrays.owned("operator_tensor",(self.data.shape[0]*other.data.shape[0],self.data.shape[1]*other.data.shape[1]),complex)
            kron_into(self.data,other.data,out,scope=current_resources())
            return self._new(out,input_dims=incoming,output_dims=outgoing,_matrix_basis=basis)
        return self._new(np.kron(self.data, other.data),input_dims=incoming,output_dims=outgoing,_matrix_basis=basis)

    def expand(self, other: "Operator") -> "Operator":
        if _explicit_arithmetic(self):
            other = _explicit_tensor_operand(self, other)
        elif isinstance(other, Operator):
            if _arithmetic_convention(other) != _arithmetic_convention(self):
                from quantumbridge.core.classical import UnsupportedFeature
                raise UnsupportedFeature("operator_arithmetic_convention")
        other = other if isinstance(other, Operator) else Operator(other)
        return other.tensor(self)

    def power(self, n: float, branch_cut_rotation=3.141592653589793e-12, assume_unitary=False) -> "Operator":
        if _representation_admission(self):_admit_public_arithmetic_owner(self)
        from quantumbridge.execution.resource_context import current_resources
        from quantumbridge.execution.array_reservations import numeric_shape
        from quantumbridge.core.classical import UnsupportedFeature
        if current_resources() is not None:
            if type(self) is not Operator:raise UnsupportedFeature("managed_operator_power")
            from quantumbridge.core.composition import _admit_graph
            _admit_graph((self._input_dims,self._output_dims))
        shape=numeric_shape(self.data)
        if shape!=(prod(self._output_dims),prod(self._input_dims)):
            raise ValueError("Operator dimensions do not match its current numeric buffer.")
        if self._input_dims!=self._output_dims:
            raise ValueError("Operator powers require matching input and output subsystems.")
        from .matrix_functions import matrix_power
        return self._new(matrix_power(self.data,n,branch_cut_rotation,assume_unitary))

    def tensorpower(self, n: int) -> "Operator":
        if not isinstance(n, int) or n < 0:
            raise ValueError("QuantumBridge Operator tensorpower requires a non-negative integer exponent.")
        out = Operator.from_label("")
        if _arithmetic_convention(self) == "qiskit":
            out = out.as_qiskit_operator()
        elif _explicit_arithmetic(self):
            out = out.as_native_operator()
        for _ in range(n):
            out = out.tensor(self)
        return out

    def reverse_qargs(self) -> "Operator":
        if _representation_admission(self):_admit_public_arithmetic_owner(self)
        if self.num_qubits is not None:
            basis = "unresolved" if self.__dict__.get("_public_matrix_basis") == "hybrid" else "inherit"
            return self._new(_reverse_qubit_order(self.data,self.num_qubits), _matrix_basis=basis)
        from quantumbridge.execution.array_ownership import copied_array
        data=copied_array(self.data,category="operator_reverse")
        rows=tuple(basis_index(basis_digits(index,self._output_dims)[::-1],self._output_dims[::-1]) for index in range(self.data.shape[0]))
        columns=tuple(basis_index(basis_digits(index,self._input_dims)[::-1],self._input_dims[::-1]) for index in range(self.data.shape[1]))
        for row,target_row in enumerate(rows):
            for column,target_column in enumerate(columns):data[target_row,target_column]=self.data[row,column]
        return self._new(data,input_dims=self._input_dims[::-1],output_dims=self._output_dims[::-1])

    def is_identity(self, atol: float = 1e-10) -> bool:
        if self.data.shape[0]!=self.data.shape[1]:return False
        return bool(np.allclose(self.data, np.eye(self.data.shape[0], dtype=complex), atol=atol))

    def reshape(self, input_dims=None, output_dims=None, num_qubits: int | None = None) -> "Operator":
        if _representation_admission(self):_admit_public_arithmetic_owner(self)
        incoming=self._input_dims if input_dims is None else state_dimensions(input_dims,size=self.data.shape[1])
        outgoing=self._output_dims if output_dims is None else state_dimensions(output_dims,size=self.data.shape[0])
        count=qubit_count(incoming) if incoming==outgoing else None
        if num_qubits is not None and int(num_qubits)!=count:
            raise ValueError("QuantumBridge Operator reshape cannot change dense matrix size.")
        from quantumbridge.execution.array_ownership import copied_array
        basis = "unresolved" if self.__dict__.get("_public_matrix_basis") == "hybrid" and (
            incoming != self._input_dims or outgoing != self._output_dims) else "inherit"
        return self._new(copied_array(self.data,category="operator_copy"),input_dims=incoming,output_dims=outgoing,_matrix_basis=basis)

    def equiv(self, other, atol: float = 1e-10) -> bool:
        from quantumbridge.execution.resource_context import current_resources
        first=self.data
        other_data = other.data if isinstance(other, Operator) else other
        if current_resources() is not None:
            from quantumbridge.execution.array_reservations import numeric_array
            first=numeric_array(first,category="operator_equiv_input")
            other_data=numeric_array(other_data,category="operator_equiv_input")
        else:other_data=np.asarray(other_data,dtype=complex)
        if first.shape != other_data.shape:
            return False
        if not isfinite(atol) or atol<0:raise ValueError("Operator equivalence tolerance must be finite and nonnegative.")
        pivot,largest=0,0.
        for index,(a,b) in enumerate(zip(first.flat,other_data.flat)):
            a,b=complex(a),complex(b)
            if not all(isfinite(value) for value in (a.real,a.imag,b.real,b.imag)):return False
            size=max(abs(b.real),abs(b.imag))
            if size>largest:pivot,largest=index,size
        def matches(phase):
            for a,b in zip(first.flat,other_data.flat):
                a,b=complex(a),complex(b)
                scale=max(1.,abs(a.real),abs(a.imag),abs(b.real),abs(b.imag))
                reference=(b/scale)*phase
                if abs(a/scale-reference)>atol/scale+1e-5*abs(reference):return False
            return True
        if matches(1.):return True
        if not largest:return False
        # A largest-component pivot avoids amplifying noise in a tiny entry.
        phase=rect(1.,scalar_phase(complex(first.flat[pivot]))-scalar_phase(complex(other_data.flat[pivot])))
        return matches(phase)

    def to_instruction(self, name: str = "unitary", label: str | None = None) -> Instruction:
        basis = _admit_public_arithmetic_owner(self)
        if self.num_qubits is None:
            from quantumbridge.core.classical import UnsupportedFeature
            raise UnsupportedFeature("mixed_radix_operator_instruction","Qubit instructions cannot reinterpret qudits or rectangular maps.")
        # Instructions execute on the existing native high-first kernel. An
        # explicit public/hybrid owner needs one owned normalization here.
        if _arithmetic_convention(self) == "qiskit" or basis == "hybrid":
            from quantumbridge.core.standard_gates import UnitaryGate
            gate = UnitaryGate(self.as_native_operator().data, label=label)
            object.__setattr__(gate, "_public_matrix_basis", "native")
            if label is not None or name != "unitary":gate.name = label or name
            return gate
        matrix = self.to_matrix()
        return Instruction(
            name=label or name,
            num_qubits=self.num_qubits,
            metadata={"matrix": matrix, "source": "quantumbridge.information.Operator"},
            label=label,
        )

    def to_matrix(self) -> np.ndarray:
        if _representation_admission(self):_admit_public_arithmetic_owner(self)
        from quantumbridge.execution.array_ownership import copied_array
        return copied_array(self.data,category="operator_matrix_export")

    def __matmul__(self, other: "Operator") -> "Operator":
        return self.dot(other)

    def __rmatmul__(self, other) -> "Operator":
        return self.compose(other,front=_arithmetic_convention(self) != "qiskit")

    def _arithmetic(self, other, operation, *, scalar=False, reverse=False):
        from quantumbridge.execution.resource_context import current_resources
        from quantumbridge.core.classical import UnsupportedFeature
        scope=current_resources()
        mode = _arithmetic_convention(self)
        typed = isinstance(other, Operator)
        if typed and _arithmetic_convention(other) != mode:
            raise UnsupportedFeature("operator_arithmetic_convention", "Select the same arithmetic convention before combining matrices.")
        if _representation_admission(self):_admit_public_arithmetic_owner(self)
        if typed and _representation_admission(other):_admit_public_arithmetic_owner(other)
        hybrid = self.__dict__.get("_public_matrix_basis") == "hybrid"
        other_hybrid = typed and other.__dict__.get("_public_matrix_basis") == "hybrid"
        if not scalar and (hybrid or other_hybrid) and (not typed or not hybrid or not other_hybrid
            or self._literal_target_width != other._literal_target_width):
            raise UnsupportedFeature("operator_arithmetic_convention", "Select matching coordinates before combining hybrid matrices.")
        explicit = _explicit_arithmetic(self) or typed and _explicit_arithmetic(other)
        if explicit:
            _admit_public_arithmetic_owner(self)
            if typed:_admit_public_arithmetic_owner(other)
        first=self.data
        second=other.data if typed else other
        if explicit:
            from quantumbridge.execution.array_reservations import numeric_shape, array_layout
            first_shape, second_shape = numeric_shape(first), numeric_shape(second)
            if scalar:
                if second_shape:raise ValueError("Operator scaling requires a numeric scalar.")
            elif first_shape != second_shape:
                raise ValueError("QuantumBridge Operator arithmetic requires equal dimensions.")
            if typed and (self._input_dims != other._input_dims or self._output_dims != other._output_dims):
                raise ValueError("Operator arithmetic requires matching subsystem dimensions.")
            if scope is not None:
                needed = array_layout(first_shape, complex)[2]
                for value, shape in ((first, first_shape),) if scalar else ((first, first_shape), (second, second_shape)):
                    view = np.ndarray.view(value, np.ndarray) if issubclass(type(value), np.ndarray) else None
                    if view is None or view.dtype != np.dtype(complex):needed += array_layout(shape, complex)[2]
                scope.ledger.preflight(memory_bytes=needed)
        if scope is not None or explicit:
            from quantumbridge.execution.array_reservations import numeric_array, numeric_shape
            first=numeric_array(first,category="operator_arithmetic_input")
            if scalar:
                if numeric_shape(second):raise ValueError("Operator scaling requires a numeric scalar.")
            else:second=numeric_array(second,category="operator_arithmetic_input")
        elif not scalar:second=np.asarray(second,dtype=complex)
        if not scalar and first.shape!=second.shape:
            raise ValueError("QuantumBridge Operator arithmetic requires equal dimensions.")
        if not scalar and typed and (self._input_dims!=other._input_dims or self._output_dims!=other._output_dims):
            raise ValueError("Operator arithmetic requires matching subsystem dimensions.")
        left,right=(second,first) if reverse else (first,second)
        if scope is None:return self._new(operation(left,right))
        from quantumbridge.execution import mps_arrays as arrays
        output=arrays.owned("operator_arithmetic_output",first.shape,complex)
        scope.ledger.unavailable("operator_ufunc_iterator_workspace",
            "The independent matrix output is reserved; NumPy scalar/iterator workspace is not measured RSS.")
        operation(left,right,out=output)
        return self._new(output)

    def __mul__(self, scalar) -> "Operator":
        if not isinstance(scalar, Number):
            return NotImplemented
        return self._arithmetic(scalar,np.multiply,scalar=True)

    def __rmul__(self, scalar) -> "Operator":
        return self.__mul__(scalar)

    def __truediv__(self, scalar) -> "Operator":
        if not isinstance(scalar, Number):
            return NotImplemented
        return self._arithmetic(scalar,np.divide,scalar=True)

    def __add__(self, other) -> "Operator":
        return self._arithmetic(other,np.add)

    def __radd__(self, other) -> "Operator":
        return self._arithmetic(other,np.add,reverse=True)

    def __sub__(self, other) -> "Operator":
        return self._arithmetic(other,np.subtract)

    def __rsub__(self, other) -> "Operator":
        return self._arithmetic(other,np.subtract,reverse=True)

    def __xor__(self, n: int) -> "Operator":
        return self.tensorpower(n)

    def __copy__(self) -> "Operator":
        return self.copy()

    def __deepcopy__(self, memo) -> "Operator":
        from quantumbridge.execution.array_ownership import native_object_deepcopy
        return native_object_deepcopy(self,memo)

    def __eq__(self, other) -> bool:
        try:
            return self.equiv(other)
        except Exception:
            return False
