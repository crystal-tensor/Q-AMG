# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from math import isqrt

import numpy as np

from quantumbridge.information.density_matrix import DensityMatrix
from quantumbridge.information.operator import Operator
from quantumbridge.execution.resource_context import current_resources


def _matrix_array(value, *, copy=False):
    if current_resources() is not None:
        from quantumbridge.execution.array_reservations import numeric_array
        return numeric_array(value, copy=copy, category="channel_matrix_conversion")
    return np.array(value, copy=True) if copy else np.asarray(value, dtype=complex)


def _matrix_group(values, *, copy=False):
    scope=current_resources()
    if scope is None:return tuple(_matrix_array(value,copy=copy) for value in values)
    from quantumbridge.execution.array_reservations import numeric_operator_sequence, numeric_shape, array_layout
    values=numeric_operator_sequence(values)
    shapes=[numeric_shape(value) for value in values]
    needed=sum(array_layout(shape,complex)[2] for value,shape in zip(values,shapes)
        if copy or not issubclass(type(value),np.ndarray) or np.ndarray.view(value,np.ndarray).dtype!=np.dtype(complex))
    scope.ledger.preflight(memory_bytes=needed)
    return tuple(_matrix_array(value,copy=copy) for value in values)


def _channel_input(value):
    native=type(value) is Operator if current_resources() is not None else isinstance(value,Operator)
    if native and value.num_qubits is None:
        from quantumbridge.core.classical import UnsupportedFeature
        raise UnsupportedFeature("mixed_radix_operator_channel","Channel conversion does not yet preserve mixed-radix Operator input/output partitions.")
    if current_resources() is None:return value
    from quantumbridge.core import Circuit
    from quantumbridge.core.circuit import is_native_circuit
    from quantumbridge.core.classical import UnsupportedFeature
    kind=type(value)
    if issubclass(kind,np.ndarray):return np.ndarray.view(value,np.ndarray)
    if is_native_circuit(value) or any(kind is native for native in (Kraus,Choi,SuperOp,PTM,Chi,Stinespring,Operator,list,tuple)):return value
    raise UnsupportedFeature("managed_channel_input","Managed channels cannot invoke foreign constructor or matrix protocols.")


def _channel_matrix_origin(value):
    """Admit an existing coordinate owner before reading its numeric fields."""
    from quantumbridge.core.classical import UnsupportedFeature
    from quantumbridge.core.composition import _admit_graph
    from quantumbridge.core.circuit import is_native_circuit
    from quantumbridge.core.gate_limits import current_gate_limits
    from quantumbridge.information.operator import _admit_public_arithmetic_owner
    kind = type(value)
    if kind is Operator:
        origin = _admit_public_arithmetic_owner(value)
        if origin == "hybrid":
            from math import isfinite
            # Canonical ndarray scalars need no temporary Boolean matrix.
            if any(not isfinite(item.real) or not isfinite(item.imag) for item in value.data.flat):
                raise ValueError("Hybrid channel input must contain finite coefficients.")
            return "native"
        return origin
    if is_native_circuit(value):
        _admit_graph(value)
        return "native"
    native_types = (Kraus, SuperOp, Choi, PTM, Chi, Stinespring)
    if not any(kind is native for native in native_types):
        if issubclass(kind, (*native_types, Operator)):
            raise UnsupportedFeature("channel_matrix_origin_owner")
        return None
    fields = object.__getattribute__(value, "__dict__")
    origin = fields.get("_public_matrix_basis")
    if origin is not None and (type(origin) is not str or origin not in ("native", "unresolved")):
        raise UnsupportedFeature("channel_matrix_origin")
    incoming, outgoing = fields.get("_input_dim"), fields.get("_output_dim")
    if any(type(dim) is not int or dim < 1 for dim in (incoming, outgoing)):
        raise ValueError("Channel dimensions require positive native integers.")
    if kind is Kraus:
        left, right, standard = (fields.get(key) for key in ("_left", "_right", "_is_standard"))
        if type(left) is not tuple or type(right) is not tuple or not left or len(left) != len(right) or type(standard) is not bool:
            raise ValueError("Invalid native Kraus storage descriptor.")
        matrices, shape = left + right, (outgoing, incoming)
        if standard and right is not left:raise ValueError("Standard Kraus factors must share their canonical tuple.")
        data = fields.get("data")
        if standard:
            if data is not left:raise ValueError("Kraus public data differs from its factors.")
        elif type(data) is not tuple or len(data) != 2 or data[0] is not left or data[1] is not right:
            raise ValueError("Generalized Kraus public data differs from its factors.")
    elif kind is Stinespring:
        environment = fields.get("_environment_dim")
        if type(environment) is not int or environment < 1:raise ValueError("Invalid Stinespring environment dimension.")
        matrices = (fields.get("_left_data"), fields.get("_right_data"))
        shape = (outgoing * environment, incoming)
        standard, data = fields.get("_is_standard"), fields.get("data")
        if type(standard) is not bool:raise ValueError("Invalid Stinespring factor descriptor.")
        if standard:
            if matrices[0] is not matrices[1] or data is not matrices[0]:
                raise ValueError("Standard Stinespring data differs from its factors.")
        elif type(data) is not tuple or len(data) != 2 or data[0] is not matrices[0] or data[1] is not matrices[1]:
            raise ValueError("Generalized Stinespring data differs from its factors.")
    else:
        matrices = (fields.get("data"),)
        shape = (outgoing * incoming,) * 2 if kind is Choi else (outgoing ** 2, incoming ** 2)
    limits = current_gate_limits()
    limits.check("max_matrix_dimension", max(shape))
    limits.check("max_decomposition_work", len(matrices) * shape[0] * shape[1])
    for matrix in matrices:
        if type(matrix) is not np.ndarray or matrix.dtype.kind not in "biufc":
            raise UnsupportedFeature("channel_matrix_origin_storage")
        if matrix.shape != shape:raise ValueError("Channel dimensions do not match its numeric buffer.")
    _admit_graph(value)
    return origin


def _set_channel_origin(result, origin):
    if origin is not None and (type(origin) is not str or origin not in ("native", "unresolved")):
        from quantumbridge.core.classical import UnsupportedFeature
        raise UnsupportedFeature("channel_matrix_origin")
    if origin is not None:
        fields = object.__getattribute__(result, "__dict__")
        if "_public_matrix_basis" not in fields:result._public_matrix_basis = origin
        elif fields["_public_matrix_basis"] != origin:raise ValueError("Channel origin cannot be overwritten.")
    return result


def _combined_channel_origin(first, second):
    return first if first == second else "unresolved"


def _matrix_rank(value):
    if current_resources() is not None:
        from quantumbridge.execution.array_reservations import numeric_shape
        return len(numeric_shape(value))
    return np.asarray(value).ndim


def _same_matrix(first, second):
    if current_resources() is None:return np.array_equal(first, second)
    return first.shape==second.shape and all(a==b for a,b in zip(first.flat, second.flat))


def _real_if_close(value):
    if current_resources() is None:return np.real_if_close(value)
    threshold=100*np.finfo(value.dtype).eps
    return value.real if all(abs(complex(scalar).imag)<threshold for scalar in value.flat) else value


def _is_power_of_two(value: int) -> bool:
    return value > 0 and not value & (value - 1)


def _num_qubits(input_dim: int, output_dim: int) -> int | None:
    if input_dim != output_dim or not _is_power_of_two(input_dim):
        return None
    return int(np.log2(input_dim))


def _dims_for_dimension(dimension: int) -> tuple[int, ...]:
    if _is_power_of_two(dimension):
        return (2,) * int(np.log2(dimension))
    return (dimension,)


def _operator_data(data, *, _normalize_hybrid=True) -> np.ndarray:
    if issubclass(type(data), Operator):
        _channel_matrix_origin(data)
        if object.__getattribute__(data, "__dict__").get("_public_matrix_basis") == "hybrid":
            if not _normalize_hybrid:
                from quantumbridge.core.classical import UnsupportedFeature
                raise UnsupportedFeature("channel_matrix_origin", "Nested channel factors require a common declared representation.")
            # This is already one independent, owned native snapshot. Do not
            # copy it again before handing it to the existing channel kernel.
            return data.as_native_operator().data
        return _matrix_array(data.data, copy=True)
    if current_resources() is not None:
        from quantumbridge.core.circuit import is_native_circuit
        from quantumbridge.core.classical import UnsupportedFeature
        if is_native_circuit(data):return Operator.from_circuit(data).data
        if issubclass(type(data), np.ndarray) or type(data) is tuple or type(data) is list:return _matrix_array(data)
        raise UnsupportedFeature("managed_channel_operator", "Managed channel input requires a native operator/circuit or numeric buffer; foreign matrix factories are not dispatched.")
    if hasattr(data, "operations"):
        return Operator.from_circuit(data).data
    if hasattr(data, "to_matrix"):
        return np.asarray(data.to_matrix(), dtype=complex)
    if hasattr(data, "data") and not isinstance(data, np.ndarray):
        candidate = np.asarray(data.data, dtype=complex)
        if candidate.ndim == 2:
            return candidate
    return np.asarray(data, dtype=complex)


def _infer_superop_dimensions(shape: tuple[int, int]) -> tuple[int, int]:
    output_dim = isqrt(shape[0])
    input_dim = isqrt(shape[1])
    if output_dim * output_dim != shape[0] or input_dim * input_dim != shape[1]:
        raise ValueError("QuantumBridge SuperOp dimensions must be perfect squares.")
    return input_dim, output_dim


def _superop_to_choi(data: np.ndarray, input_dim: int, output_dim: int) -> np.ndarray:
    if current_resources() is not None:
        from quantumbridge.execution import mps_arrays as arrays
        tensor=arrays.reshaped(_matrix_array(data),(output_dim,output_dim,input_dim,input_dim))
        result=arrays.owned("superop_choi_conversion",(input_dim*output_dim,)*2,complex)
        np.copyto(result.reshape(input_dim,output_dim,input_dim,output_dim),tensor.transpose(3,1,2,0),casting="no")
        return result
    return (
        data.reshape(output_dim, output_dim, input_dim, input_dim)
        .transpose(3, 1, 2, 0)
        .reshape(input_dim * output_dim, input_dim * output_dim)
    )


def _choi_to_superop(data: np.ndarray, input_dim: int, output_dim: int) -> np.ndarray:
    if current_resources() is not None:
        from quantumbridge.execution import mps_arrays as arrays
        tensor=arrays.reshaped(_matrix_array(data),(input_dim,output_dim,input_dim,output_dim))
        result=arrays.owned("choi_superop_conversion",(output_dim**2,input_dim**2),complex)
        np.copyto(result.reshape(output_dim,output_dim,input_dim,input_dim),tensor.transpose(3,1,2,0),casting="no")
        return result
    return (
        data.reshape(input_dim, output_dim, input_dim, output_dim)
        .transpose(3, 1, 2, 0)
        .reshape(output_dim * output_dim, input_dim * input_dim)
    )


def _kraus_from_choi(
    data: np.ndarray,
    input_dim: int,
    output_dim: int,
    atol: float = 1e-12,
) -> tuple[np.ndarray, ...]:
    if current_resources() is not None:
        from quantumbridge.execution import mps_arrays as arrays
        hermitian=arrays.conjugated(data.T)
        np.add(hermitian,data,out=hermitian);np.multiply(hermitian,.5,out=hermitian)
        values,vectors=arrays.decomposed(hermitian,"eigh")
        if np.min(values)<-atol:raise ValueError("QuantumBridge cannot convert a non-CP channel to standard Kraus form.")
        return _eigen_kraus(values,vectors,input_dim,output_dim,atol)
    hermitian = (data + data.conj().T) / 2
    values, vectors = np.linalg.eigh(hermitian)
    if np.min(values) < -atol:
        raise ValueError("QuantumBridge cannot convert a non-CP channel to standard Kraus form.")
    operators = []
    for index in np.flatnonzero(values > atol):
        vector = np.sqrt(values[index]) * vectors[:, index]
        operators.append(vector.reshape((output_dim, input_dim), order="F"))
    if not operators:
        operators.append(np.zeros((output_dim, input_dim), dtype=complex))
    return tuple(operators)


def _kraus_pair_from_choi(
    data: np.ndarray,
    input_dim: int,
    output_dim: int,
    atol: float = 1e-12,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    if current_resources() is not None:
        from quantumbridge.execution import mps_arrays as arrays
        from quantumbridge.execution.array_reservations import owned_arrays
        from quantumbridge.utils.math import validate_hermitian_matrix
        if validate_hermitian_matrix(data,len(data),atol=atol,rtol=1e-5):
            hermitian=arrays.conjugated(data.T)
            np.add(hermitian,data,out=hermitian);np.multiply(hermitian,.5,out=hermitian)
            values,vectors=arrays.decomposed(hermitian,"eigh")
            if np.min(values)>=-atol:
                operators=_eigen_kraus(values,vectors,input_dim,output_dim,atol)
                return operators,operators
        left_vectors,values,right_adjoint=arrays.decomposed(data,"svd")
        selected=[index for index,value in enumerate(values) if value>atol]
        if not selected:
            zero=arrays.filled("channel_zero_kraus",(output_dim,input_dim),complex,0)
            return (zero,),(zero,)
        outputs=owned_arrays([((output_dim,input_dim),complex)]*(2*len(selected)),category="generalized_kraus_operator")
        left,right=outputs[:len(selected)],outputs[len(selected):]
        for a,b,index in zip(left,right,selected):
            scale=np.sqrt(values[index])
            np.multiply(left_vectors[:,index].reshape(output_dim,input_dim,order="F"),scale,out=a)
            np.conjugate(right_adjoint[index,:].reshape(output_dim,input_dim,order="F"),out=b)
            np.multiply(b,scale,out=b)
        return left,right
    hermitian = (data + data.conj().T) / 2
    if np.allclose(data, data.conj().T, atol=atol):
        values = np.linalg.eigvalsh(hermitian)
        if np.min(values) >= -atol:
            operators = _kraus_from_choi(data, input_dim, output_dim, atol=atol)
            return operators, operators

    left_vectors, singular_values, right_adjoint = np.linalg.svd(data)
    right_vectors = right_adjoint.conj().T
    left = []
    right = []
    for index in np.flatnonzero(singular_values > atol):
        scale = np.sqrt(singular_values[index])
        left.append((scale * left_vectors[:, index]).reshape((output_dim, input_dim), order="F"))
        right.append((scale * right_vectors[:, index]).reshape((output_dim, input_dim), order="F"))
    if not left:
        zero = np.zeros((output_dim, input_dim), dtype=complex)
        return (zero,), (zero,)
    return tuple(left), tuple(right)


def _eigen_kraus(values,vectors,input_dim,output_dim,atol):
    from quantumbridge.execution.array_reservations import owned_arrays
    selected=[index for index,value in enumerate(values) if value>atol]
    outputs=owned_arrays([((output_dim,input_dim),complex)]*max(1,len(selected)),category="eigen_kraus_operator")
    if not selected:outputs[0].fill(0)
    for output,index in zip(outputs,selected):
        np.multiply(vectors[:,index].reshape(output_dim,input_dim,order="F"),np.sqrt(values[index]),out=output)
    return outputs


def _pauli_basis(num_qubits: int) -> tuple[np.ndarray, ...]:
    one_qubit = (
        np.eye(2, dtype=complex),
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    )
    basis = (np.array([[1]], dtype=complex),)
    for _ in range(num_qubits):
        basis = tuple(np.kron(prefix, pauli) for prefix in basis for pauli in one_qubit)
    return basis


def _pauli_change_matrix(num_qubits: int) -> np.ndarray:
    if current_resources() is not None:
        from itertools import product
        from quantumbridge.execution import mps_arrays as arrays
        from quantumbridge.utils.math import _pauli_dense
        dimension=1<<num_qubits
        change=arrays.owned("channel_pauli_basis",(dimension**2,dimension**2),complex)
        for column,label in enumerate(product("IXYZ",repeat=num_qubits)):
            matrix=_pauli_dense("".join(label))
            np.copyto(change[:,column].reshape(dimension,dimension),matrix.T,casting="no")
        return change
    return np.column_stack([matrix.reshape(-1, order="F") for matrix in _pauli_basis(num_qubits)])


def _validate_qargs(qargs, num_qubits: int, width: int) -> tuple[int, ...]:
    normalized = tuple(int(qarg) for qarg in qargs)
    if len(normalized) != width:
        raise ValueError("QuantumBridge channel qargs width must match the channel width.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("QuantumBridge channel qargs must be distinct.")
    if any(qarg < 0 or qarg >= num_qubits for qarg in normalized):
        raise ValueError("QuantumBridge channel qargs are outside the target state.")
    return normalized


def _embed_operator(matrix: np.ndarray, qargs: tuple[int, ...], num_qubits: int) -> np.ndarray:
    if current_resources() is not None:
        from quantumbridge.information.operator import _expand_subsystem_operator
        return _expand_subsystem_operator(matrix,qargs,num_qubits)
    identity = Operator(np.eye(1 << num_qubits, dtype=complex)).as_native_operator()
    return identity.compose(Operator(matrix).as_native_operator(), qargs=qargs).data


def _validation_choi(channel):
    """Existing representation conversions with explicitly owned outputs."""
    from quantumbridge.core.classical import UnsupportedFeature
    from quantumbridge.execution.resource_context import current_resources
    from quantumbridge.execution.array_ownership import retained_array
    from quantumbridge.execution.array_reservations import numeric_array
    from quantumbridge.execution import mps_arrays as arrays
    scope = current_resources()
    kind = type(channel)
    if not any(kind is native for native in (Kraus, SuperOp, Choi, PTM, Chi, Stinespring)):
        raise UnsupportedFeature("managed_channel_validation", "Managed channel validation requires a native representation, not a foreign conversion factory.")
    input_dim, output_dim = channel._input_dim, channel._output_dim
    size = input_dim*output_dim
    def output():
        return retained_array(scope.ledger, "channel_validation_choi", (size, size), complex)
    if kind is Choi:
        return numeric_array(channel.data, category="channel_validation_input")
    if kind in (PTM, Chi):
        change = _pauli_change_matrix(channel.num_qubits or 0)
        data = numeric_array(channel.data, category="channel_validation_input")
        converted = arrays.contracted(arrays.contracted(change, data), arrays.conjugated(change).T)
        np.divide(converted, input_dim, out=converted)
        if kind is Chi:return converted
        data = converted
    elif kind is SuperOp:
        data = numeric_array(channel.data, category="channel_validation_input")
    elif kind is Kraus:
        left = tuple(numeric_array(value, category="channel_validation_input") for value in channel._left)
        right = left if channel._is_standard else tuple(numeric_array(value, category="channel_validation_input") for value in channel._right)
        result = output()
        for row in range(size):
            for column in range(size):
                result[row, column] = sum(complex(a[row % output_dim, row // output_dim])*complex(b[column % output_dim, column // output_dim]).conjugate()
                    for a, b in zip(left, right))
        return result
    else:
        left = arrays.reshaped(numeric_array(channel._left_data, category="channel_validation_input"),
            (output_dim, channel._environment_dim, input_dim))
        right = left if channel._is_standard else arrays.reshaped(numeric_array(channel._right_data, category="channel_validation_input"), left.shape)
        result = output()
        for row in range(size):
            for column in range(size):
                result[row, column] = sum(complex(left[row % output_dim, index, row // output_dim])*complex(right[column % output_dim, index, column // output_dim]).conjugate()
                    for index in range(channel._environment_dim))
        return result
    result = output()
    tensor = arrays.reshaped(data, (output_dim, output_dim, input_dim, input_dim))
    np.copyto(result.reshape(input_dim, output_dim, input_dim, output_dim), tensor.transpose(3, 1, 2, 0), casting="no")
    return result


def _validation_cp(choi, *, atol, rtol):
    from quantumbridge.execution.resource_context import current_resources
    from quantumbridge.execution.array_ownership import retained_array
    from quantumbridge.utils.math import validate_hermitian_matrix
    scope = current_resources()
    if not validate_hermitian_matrix(choi, len(choi), atol=atol, rtol=rtol):return False
    hermitian = retained_array(scope.ledger, "channel_validation_hermitian", choi.shape, complex)
    np.conjugate(choi.T, out=hermitian)
    np.add(hermitian, choi, out=hermitian)
    np.multiply(hermitian, .5, out=hermitian)
    scope.ledger.unavailable("channel_validation_lapack_workspace",
        "Choi/explicit Hermitian/eigenvalue buffers are reserved; unexposed eigvalsh/LAPACK input/workspace and Python scalar memory are not measured RSS.")
    values = retained_array(scope.ledger, "channel_validation_eigenvalues", (len(choi),), float,
        lambda: np.linalg.eigvalsh(hermitian))
    return bool(np.min(values) >= -atol)


def _validation_tp(choi, input_dim, output_dim, *, atol, rtol):
    # This is the existing output partial trace, evaluated scalar by scalar.
    for row in range(input_dim):
        for column in range(input_dim):
            value = sum(complex(choi[row*output_dim+index, column*output_dim+index]) for index in range(output_dim))
            expected = int(row == column)
            if not abs(value-expected) <= atol+rtol*expected:return False
    return True


class _QuantumChannel:
    """Shared native channel operations across matrix representations."""

    _input_dim: int
    _output_dim: int

    def __setattr__(self, name, value):
        if name == "_public_matrix_basis":
            if name in self.__dict__:raise AttributeError("Channel matrix origin is immutable.")
            if type(value) is not str or value not in ("native", "unresolved"):
                raise ValueError("Unknown channel matrix origin.")
        object.__setattr__(self, name, value)

    @property
    def dim(self) -> tuple[int, int]:
        return (self._input_dim, self._output_dim)

    @property
    def num_qubits(self) -> int | None:
        return _num_qubits(self._input_dim, self._output_dim)

    def input_dims(self, qargs=None) -> tuple[int, ...]:
        dims = _dims_for_dimension(self._input_dim)
        if qargs is None:
            return dims
        if self.num_qubits is None:
            raise ValueError("QuantumBridge subsystem dimensions require a qubit channel.")
        selected = tuple(int(qarg) for qarg in qargs)
        _validate_qargs(selected, self.num_qubits, len(selected))
        return tuple(dims[qarg] for qarg in selected)

    def output_dims(self, qargs=None) -> tuple[int, ...]:
        dims = _dims_for_dimension(self._output_dim)
        if qargs is None:
            return dims
        if self.num_qubits is None:
            raise ValueError("QuantumBridge subsystem dimensions require a qubit channel.")
        selected = tuple(int(qarg) for qarg in qargs)
        _validate_qargs(selected, self.num_qubits, len(selected))
        return tuple(dims[qarg] for qarg in selected)

    def copy(self):
        return type(self)(self)

    def to_superop(self) -> "SuperOp":
        return SuperOp(self)

    def to_choi(self) -> "Choi":
        return Choi(self)

    def to_kraus(self) -> "Kraus":
        return Kraus(self)

    def to_ptm(self) -> "PTM":
        return PTM(self)

    def to_chi(self) -> "Chi":
        return Chi(self)

    def to_stinespring(self) -> "Stinespring":
        return Stinespring(self)

    def compose(self, other, qargs=None, front: bool = False):
        origin = _combined_channel_origin(_channel_matrix_origin(self), _channel_matrix_origin(other))
        other_channel = SuperOp(other)
        if qargs is not None:
            if self.num_qubits is None or other_channel.num_qubits is None:
                raise ValueError("QuantumBridge subsystem channel composition requires qubit channels.")
            normalized = _validate_qargs(qargs, self.num_qubits, other_channel.num_qubits)
            other_kraus = other_channel.to_kraus()
            left = tuple(
                _embed_operator(op, normalized, self.num_qubits)
                for op in other_kraus._left
            )
            right = tuple(
                _embed_operator(op, normalized, self.num_qubits)
                for op in other_kraus._right
            )
            other_channel = _set_channel_origin(Kraus(left if other_kraus._is_standard else (left, right)),
                _channel_matrix_origin(other_kraus)).to_superop()

        this = self.to_superop()
        if current_resources() is not None:
            from quantumbridge.execution import mps_arrays as arrays
        if front:
            if other_channel._output_dim != this._input_dim:
                raise ValueError("QuantumBridge front composition has incompatible channel dimensions.")
            result = this.data @ other_channel.data if current_resources() is None else arrays.contracted(this.data,other_channel.data)
            input_dim = other_channel._input_dim
            output_dim = this._output_dim
        else:
            if this._output_dim != other_channel._input_dim:
                raise ValueError("QuantumBridge channel composition has incompatible dimensions.")
            result = other_channel.data @ this.data if current_resources() is None else arrays.contracted(other_channel.data,this.data)
            input_dim = this._input_dim
            output_dim = other_channel._output_dim
        return type(self)(SuperOp._from_matrix(result, input_dim, output_dim, _origin=origin))

    def dot(self, other, qargs=None):
        return self.compose(other, qargs=qargs, front=True)

    def tensor(self, other):
        origin = _combined_channel_origin(_channel_matrix_origin(self), _channel_matrix_origin(other))
        other_kraus = Kraus(other)
        this_kraus = self.to_kraus()
        if current_resources() is not None:
            from quantumbridge.execution.array_reservations import owned_arrays, kron_into
            from itertools import product
            shape=(this_kraus._output_dim*other_kraus._output_dim,this_kraus._input_dim*other_kraus._input_dim)
            count=len(this_kraus._left)*len(other_kraus._left)
            standard=this_kraus._is_standard and other_kraus._is_standard
            outputs=owned_arrays([(shape,complex)]*(count if standard else 2*count),category="channel_tensor_operator")
            groups=((outputs[:count],this_kraus._left,other_kraus._left),) if standard else (
                (outputs[:count],this_kraus._left,other_kraus._left),(outputs[count:],this_kraus._right,other_kraus._right))
            for targets,first,second in groups:
                for output,(a,b) in zip(targets,product(first,second)):
                    kron_into(a,b,output,scope=current_resources())
            return type(self)(_set_channel_origin(Kraus(outputs if standard else (outputs[:count],outputs[count:])), origin))
        left = tuple(
            np.kron(left, right)
            for left in this_kraus._left
            for right in other_kraus._left
        )
        right = tuple(
            np.kron(left, right)
            for left in this_kraus._right
            for right in other_kraus._right
        )
        data = left if this_kraus._is_standard and other_kraus._is_standard else (left, right)
        return type(self)(_set_channel_origin(Kraus(data), origin))

    def expand(self, other):
        return type(self)(other).tensor(self)

    def power(self, n: int):
        origin = _channel_matrix_origin(self)
        if not isinstance(n, int):
            raise ValueError("QuantumBridge channel power requires an integer exponent.")
        if self._input_dim != self._output_dim:
            raise ValueError("QuantumBridge channel power requires equal input and output dimensions.")
        from quantumbridge.execution import mps_arrays as arrays
        powered=arrays.matrix_power(self.to_superop().data,n)
        return type(self)(SuperOp._from_matrix(powered, self._input_dim, self._output_dim, _origin=origin))

    def tensorpower(self, n: int):
        if not isinstance(n, int) or n < 0:
            raise ValueError("QuantumBridge channel tensorpower requires a non-negative integer exponent.")
        if n == 0:
            if current_resources() is None:return type(self)(Operator(np.ones((1,1),dtype=complex)))
            from quantumbridge.utils.math import _gate_identity
            return type(self)(Operator(_gate_identity(1)))
        result = self.copy()
        for _ in range(1, n):
            result = result.tensor(self)
        return result

    def adjoint(self):
        origin = _channel_matrix_origin(self)
        data=self.to_superop().data
        if current_resources() is None:matrix=data.conj().T
        else:
            from quantumbridge.execution import mps_arrays as arrays
            matrix=arrays.conjugated(data).T
        return type(self)(SuperOp._from_matrix(matrix, self._output_dim, self._input_dim, _origin=origin))

    def conjugate(self):
        origin = _channel_matrix_origin(self)
        data=self.to_superop().data
        if current_resources() is None:matrix=np.conjugate(data)
        else:
            from quantumbridge.execution import mps_arrays as arrays
            matrix=arrays.conjugated(data)
        return type(self)(SuperOp._from_matrix(matrix, self._input_dim, self._output_dim, _origin=origin))

    def transpose(self):
        return self.conjugate().adjoint()

    def is_cp(self, atol: float = 1e-8, rtol: float = 1e-5) -> bool:
        from quantumbridge.execution.resource_context import current_resources
        if current_resources() is not None:
            return _validation_cp(_validation_choi(self), atol=atol, rtol=rtol)
        choi = self.to_choi().data
        if not np.allclose(choi, choi.conj().T, atol=atol, rtol=rtol):
            return False
        hermitian = (choi + choi.conj().T) / 2
        return bool(np.min(np.linalg.eigvalsh(hermitian)) >= -atol)

    def is_tp(self, atol: float = 1e-8, rtol: float = 1e-5) -> bool:
        from quantumbridge.execution.resource_context import current_resources
        if current_resources() is not None:
            return _validation_tp(_validation_choi(self), self._input_dim, self._output_dim, atol=atol, rtol=rtol)
        choi = self.to_choi().data.reshape(
            self._input_dim,
            self._output_dim,
            self._input_dim,
            self._output_dim,
        )
        partial = np.trace(choi, axis1=1, axis2=3)
        return bool(np.allclose(partial, np.eye(self._input_dim), atol=atol, rtol=rtol))

    def is_cptp(self, atol: float = 1e-8, rtol: float = 1e-5) -> bool:
        from quantumbridge.execution.resource_context import current_resources
        if current_resources() is not None:
            choi = _validation_choi(self)
            return _validation_cp(choi, atol=atol, rtol=rtol) and _validation_tp(choi, self._input_dim, self._output_dim, atol=atol, rtol=rtol)
        return self.is_cp(atol=atol, rtol=rtol) and self.is_tp(atol=atol, rtol=rtol)

    def evolve(self, state, qargs=None) -> DensityMatrix:
        if isinstance(state,DensityMatrix):
            from quantumbridge.information.state_dimensions import require_qubits
            require_qubits(state.dims(),"mixed_radix_channel_evolution")
        scope=current_resources()
        rho = state.data if isinstance(state, DensityMatrix) else _matrix_array(state)
        if rho.ndim == 1:
            if scope is None:rho = np.outer(rho, rho.conj())
            else:
                from quantumbridge.execution import mps_arrays as arrays
                rho=arrays.contracted(rho[:,None],arrays.conjugated(rho)[None,:])
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise ValueError("QuantumBridge channel evolution requires a statevector or density matrix.")

        kraus = self.to_kraus()
        left = kraus._left
        right = kraus._right
        if qargs is not None:
            state_dim = rho.shape[0]
            if not _is_power_of_two(state_dim) or self.num_qubits is None:
                raise ValueError("QuantumBridge subsystem channel evolution requires qubit dimensions.")
            state_qubits = int(np.log2(state_dim))
            normalized = _validate_qargs(qargs, state_qubits, self.num_qubits)
            left = tuple(_embed_operator(op, normalized, state_qubits) for op in left)
            right = tuple(_embed_operator(op, normalized, state_qubits) for op in right)
        elif rho.shape[0] != self._input_dim:
            raise ValueError("QuantumBridge state and channel input dimensions do not match.")

        if scope is not None:
            from quantumbridge.execution import mps_arrays as arrays
            output=arrays.filled("channel_evolution_output",(left[0].shape[0],)*2,complex,0)
            for a,b in zip(left,right):
                product=arrays.contracted(a,rho)
                contribution=arrays.contracted(product,arrays.conjugated(b).T)
                np.add(output,contribution,out=output)
        else:output = sum(a @ rho @ b.conj().T for a, b in zip(left, right))
        return DensityMatrix(output)

    def to_operator(self) -> Operator:
        origin = _channel_matrix_origin(self)
        kraus = self.to_kraus()
        if not kraus._is_standard or len(kraus._left) != 1:
            raise ValueError("QuantumBridge channel is not representable by one operator.")
        operator = Operator(kraus._left[0])
        if not operator.is_unitary():
            raise ValueError("QuantumBridge channel operator is not unitary.")
        if origin is not None:operator._public_matrix_basis = origin
        return operator

    def to_instruction(self):
        from quantumbridge.core.operations import Instruction
        from quantumbridge.utils.math import validate_kraus_operators

        if self.num_qubits is None or self.num_qubits < 1:
            raise ValueError("QuantumBridge channel instructions require equal positive qubit input/output widths.")
        kraus = self.to_kraus()
        if not kraus._is_standard:
            kraus = Kraus(self.to_choi())
        operators = validate_kraus_operators(kraus._left, self._input_dim)
        return Instruction("kraus", self.num_qubits, metadata={"kraus": operators})

    def __xor__(self, n: int):
        return self.tensorpower(n)

    def __matmul__(self, other):
        return self.dot(other)

    def __add__(self, other):
        origin = _combined_channel_origin(_channel_matrix_origin(self), _channel_matrix_origin(other))
        other_channel = SuperOp(other)
        this = self.to_superop()
        if this.dim != other_channel.dim:
            raise ValueError("QuantumBridge channel addition requires equal dimensions.")
        if current_resources() is None:matrix=this.data+other_channel.data
        else:
            from quantumbridge.execution import mps_arrays as arrays
            matrix=arrays.owned("channel_add_output",this.data.shape,complex)
            np.add(this.data,other_channel.data,out=matrix)
        return type(self)(SuperOp._from_matrix(
            matrix,
            self._input_dim,
            self._output_dim,
            _origin=origin,
        ))

    def __sub__(self, other):
        origin = _combined_channel_origin(_channel_matrix_origin(self), _channel_matrix_origin(other))
        other_channel = SuperOp(other)
        this = self.to_superop()
        if this.dim != other_channel.dim:
            raise ValueError("QuantumBridge channel subtraction requires equal dimensions.")
        if current_resources() is None:matrix=this.data-other_channel.data
        else:
            from quantumbridge.execution import mps_arrays as arrays
            matrix=arrays.owned("channel_subtract_output",this.data.shape,complex)
            np.subtract(this.data,other_channel.data,out=matrix)
        return type(self)(SuperOp._from_matrix(
            matrix,
            self._input_dim,
            self._output_dim,
            _origin=origin,
        ))

    def __mul__(self, scalar):
        origin = _channel_matrix_origin(self)
        if not np.isscalar(scalar):
            return NotImplemented
        data=self.to_superop().data
        if current_resources() is None:matrix=data*scalar
        else:
            from quantumbridge.execution import mps_arrays as arrays
            from quantumbridge.execution.array_reservations import numeric_shape
            if numeric_shape(scalar):raise ValueError("Channel scale must be scalar.")
            matrix=arrays.owned("channel_scale_output",data.shape,complex)
            np.multiply(data,scalar,out=matrix)
        return type(self)(SuperOp._from_matrix(
            matrix,
            self._input_dim,
            self._output_dim,
            _origin=origin,
        ))

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        if not np.isscalar(scalar):
            return NotImplemented
        return self.__mul__(1 / scalar)

    def reshape(self, input_dims=None, output_dims=None, num_qubits: int | None = None):
        if input_dims is not None:
            requested = (input_dims,) if isinstance(input_dims, int) else tuple(input_dims)
            if int(np.prod(requested)) != self._input_dim:
                raise ValueError("QuantumBridge channel reshape cannot change the input dimension.")
        if output_dims is not None:
            requested = (output_dims,) if isinstance(output_dims, int) else tuple(output_dims)
            if int(np.prod(requested)) != self._output_dim:
                raise ValueError("QuantumBridge channel reshape cannot change the output dimension.")
        if num_qubits is not None:
            expected = 1 << int(num_qubits)
            if self._input_dim != expected or self._output_dim != expected:
                raise ValueError("QuantumBridge channel reshape num_qubits does not match its dimensions.")
        return self.copy()

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        from quantumbridge.execution.array_ownership import native_object_deepcopy
        return native_object_deepcopy(self,memo)


class Kraus(_QuantumChannel):
    """Standard or generalized Kraus representation of a quantum map."""

    def __init__(self, data):
        origin = _channel_matrix_origin(data)
        data=_channel_input(data)
        if isinstance(data, Kraus):
            if current_resources() is not None and not data._is_standard:
                converted=_matrix_group(data._left+data._right,copy=True)
                left,right=converted[:len(data._left)],converted[len(data._left):]
            else:
                left = _matrix_group(data._left,copy=True)
                right = left if data._is_standard else _matrix_group(data._right,copy=True)
        elif isinstance(data, _QuantumChannel):
            choi = data.to_superop().to_choi()
            left, right = _kraus_pair_from_choi(
                choi.data,
                choi._input_dim,
                choi._output_dim,
            )
        elif isinstance(data, Operator) or hasattr(data, "operations") or hasattr(data, "to_matrix"):
            left = (_operator_data(data),)
            right = left
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            left = (_matrix_array(data),)
            right = left
        elif (
            isinstance(data, tuple)
            and len(data) == 2
            and _matrix_rank(data[0]) == 3
            and _matrix_rank(data[1]) == 3
        ):
            if current_resources() is not None:
                from quantumbridge.execution.array_reservations import numeric_operator_sequence
                first,second=numeric_operator_sequence(data[0]),numeric_operator_sequence(data[1])
                converted=_matrix_group(first+second)
                left,right=converted[:len(first)],converted[len(first):]
            else:
                left = _matrix_group(data[0])
                right = _matrix_group(data[1])
        else:
            left = _matrix_group(data)
            right = left

        if not left or not right:
            raise ValueError("QuantumBridge Kraus channel requires at least one operator.")
        if len(left) != len(right):
            raise ValueError("QuantumBridge generalized Kraus lists must have equal length.")
        shape = left[0].shape
        if len(shape) != 2:
            raise ValueError("QuantumBridge Kraus operators must be matrices.")
        if any(operator.shape != shape for operator in left + right):
            raise ValueError("QuantumBridge Kraus operators must have equal shape.")
        self._left = left
        self._right = right
        self._is_standard = right is left or all(
            _same_matrix(a, b) for a, b in zip(left, right)
        )
        if self._is_standard:
            self._right = self._left
            self.data = self._left
        else:
            self.data = (self._left, self._right)
        self._output_dim, self._input_dim = shape
        _set_channel_origin(self, origin)

    @classmethod
    def from_channel(cls, channel) -> "Kraus":
        return cls(channel)

    def to_kraus(self) -> "Kraus":
        return self.copy()

    def to_superop(self) -> "SuperOp":
        origin = _channel_matrix_origin(self)
        if current_resources() is not None:
            return SuperOp._from_matrix(_choi_to_superop(_validation_choi(self),self._input_dim,self._output_dim),self._input_dim,self._output_dim,_origin=origin)
        matrix = sum(
            np.kron(right.conj(), left)
            for left, right in zip(self._left, self._right)
        )
        return SuperOp._from_matrix(matrix, self._input_dim, self._output_dim, _origin=origin)

    def to_choi(self) -> "Choi":
        origin = _channel_matrix_origin(self)
        if current_resources() is not None:
            return Choi._from_matrix(_validation_choi(self),self._input_dim,self._output_dim,_origin=origin)
        matrix = sum(
            np.outer(
                left.reshape(-1, order="F"),
                right.reshape(-1, order="F").conj(),
            )
            for left, right in zip(self._left, self._right)
        )
        return Choi._from_matrix(matrix, self._input_dim, self._output_dim, _origin=origin)


class SuperOp(_QuantumChannel):
    """Column-vectorized matrix representation of a quantum channel."""

    def __init__(self, data):
        origin = _channel_matrix_origin(data)
        data=_channel_input(data)
        if isinstance(data, SuperOp):
            self.data = _matrix_array(data.data, copy=True)
            self._input_dim = data._input_dim
            self._output_dim = data._output_dim
            _set_channel_origin(self, origin)
            return
        if isinstance(data, _QuantumChannel):
            converted = data.to_superop()
            self.data = _matrix_array(converted.data, copy=True)
            self._input_dim = converted._input_dim
            self._output_dim = converted._output_dim
            _set_channel_origin(self, origin)
            return
        if isinstance(data, Operator) or hasattr(data, "operations") or hasattr(data, "to_matrix"):
            converted = self.from_operator(data)
            self.data = converted.data
            self._input_dim = converted._input_dim
            self._output_dim = converted._output_dim
            _set_channel_origin(self, origin)
            return
        matrix = _matrix_array(data)
        if matrix.ndim != 2:
            raise ValueError("QuantumBridge SuperOp data must be a matrix.")
        self._input_dim, self._output_dim = _infer_superop_dimensions(matrix.shape)
        self.data = matrix
        _set_channel_origin(self, origin)

    @classmethod
    def _from_matrix(cls, matrix: np.ndarray, input_dim: int, output_dim: int, *, _origin=None) -> "SuperOp":
        instance = cls.__new__(cls)
        instance.data = _matrix_array(matrix)
        instance._input_dim = int(input_dim)
        instance._output_dim = int(output_dim)
        return _set_channel_origin(instance, _origin)

    @classmethod
    def from_operator(cls, operator) -> "SuperOp":
        origin = _channel_matrix_origin(operator)
        matrix = _operator_data(operator)
        if matrix.ndim != 2:
            raise ValueError("QuantumBridge SuperOp operator input must be a matrix.")
        output_dim, input_dim = matrix.shape
        if current_resources() is not None:
            from quantumbridge.execution import mps_arrays as arrays
            result=arrays.owned("operator_superop_conversion",(output_dim**2,input_dim**2),complex)
            for row in range(output_dim**2):
                for column in range(input_dim**2):
                    result[row,column]=complex(matrix[row//output_dim,column//input_dim]).conjugate()*matrix[row%output_dim,column%input_dim]
            return cls._from_matrix(result,input_dim,output_dim,_origin=origin)
        return cls._from_matrix(np.kron(matrix.conj(), matrix), input_dim, output_dim,_origin=origin)

    def to_superop(self) -> "SuperOp":
        return self.copy()

    def to_choi(self) -> "Choi":
        origin = _channel_matrix_origin(self)
        matrix = _superop_to_choi(self.data, self._input_dim, self._output_dim)
        return Choi._from_matrix(matrix, self._input_dim, self._output_dim, _origin=origin)


class Choi(_QuantumChannel):
    """Choi-matrix representation of a quantum channel."""

    def __init__(self, data):
        origin = _channel_matrix_origin(data)
        data=_channel_input(data)
        if isinstance(data, Choi):
            self.data = _matrix_array(data.data, copy=True)
            self._input_dim = data._input_dim
            self._output_dim = data._output_dim
            _set_channel_origin(self, origin)
            return
        if isinstance(data, _QuantumChannel):
            converted = data.to_superop().to_choi()
            self.data = _matrix_array(converted.data, copy=True)
            self._input_dim = converted._input_dim
            self._output_dim = converted._output_dim
            _set_channel_origin(self, origin)
            return
        if isinstance(data, Operator) or hasattr(data, "operations") or hasattr(data, "to_matrix"):
            converted = Kraus(data).to_choi()
            self.data = converted.data
            self._input_dim = converted._input_dim
            self._output_dim = converted._output_dim
            _set_channel_origin(self, origin)
            return
        matrix = _matrix_array(data)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("QuantumBridge Choi data must be a square matrix.")
        channel_dim = matrix.shape[0]
        inferred = isqrt(channel_dim)
        if inferred * inferred == channel_dim:
            self._input_dim = inferred
            self._output_dim = inferred
        else:
            self._input_dim = channel_dim
            self._output_dim = 1
        self.data = matrix
        _set_channel_origin(self, origin)

    @classmethod
    def _from_matrix(cls, matrix: np.ndarray, input_dim: int, output_dim: int, *, _origin=None) -> "Choi":
        instance = cls.__new__(cls)
        instance.data = _matrix_array(matrix)
        instance._input_dim = int(input_dim)
        instance._output_dim = int(output_dim)
        return _set_channel_origin(instance, _origin)

    @classmethod
    def from_channel(cls, channel) -> "Choi":
        return cls(channel)

    def to_choi(self) -> "Choi":
        return self.copy()

    def to_superop(self) -> SuperOp:
        origin = _channel_matrix_origin(self)
        matrix = _choi_to_superop(self.data, self._input_dim, self._output_dim)
        return SuperOp._from_matrix(matrix, self._input_dim, self._output_dim, _origin=origin)


class PTM(_QuantumChannel):
    """Pauli-transfer-matrix representation for equal-dimension qubit channels."""

    def __init__(self, data):
        origin = _channel_matrix_origin(data)
        data=_channel_input(data)
        if isinstance(data, PTM):
            self.data = _matrix_array(data.data, copy=True)
            self._input_dim = data._input_dim
            self._output_dim = data._output_dim
            _set_channel_origin(self, origin)
            return
        if (
            isinstance(data, _QuantumChannel)
            or isinstance(data, Operator)
            or hasattr(data, "operations")
            or hasattr(data, "to_matrix")
        ):
            channel = data if isinstance(data, _QuantumChannel) else Kraus(data)
            converted = self.from_superop(channel.to_superop())
            self.data = converted.data
            self._input_dim = converted._input_dim
            self._output_dim = converted._output_dim
            _set_channel_origin(self, origin)
            return
        matrix = _matrix_array(data)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("QuantumBridge PTM data must be square.")
        dimension = isqrt(matrix.shape[0])
        if dimension * dimension != matrix.shape[0] or not _is_power_of_two(dimension):
            raise ValueError("QuantumBridge PTM data must represent an n-qubit channel.")
        self.data = _real_if_close(matrix)
        self._input_dim = dimension
        self._output_dim = dimension
        _set_channel_origin(self, origin)

    @classmethod
    def from_superop(cls, channel: SuperOp) -> "PTM":
        origin = _channel_matrix_origin(channel)
        if channel._input_dim != channel._output_dim or not _is_power_of_two(channel._input_dim):
            raise ValueError("QuantumBridge PTM requires an equal-dimension qubit channel.")
        num_qubits = int(np.log2(channel._input_dim))
        change = _pauli_change_matrix(num_qubits)
        if current_resources() is None:matrix=change.conj().T@channel.data@change/channel._input_dim
        else:
            from quantumbridge.execution import mps_arrays as arrays
            matrix=arrays.contracted(arrays.contracted(arrays.conjugated(change).T,channel.data),change)
            np.divide(matrix,channel._input_dim,out=matrix)
        return _set_channel_origin(cls(_real_if_close(matrix)), origin)

    def to_superop(self) -> SuperOp:
        origin = _channel_matrix_origin(self)
        if current_resources() is not None:
            return SuperOp._from_matrix(_choi_to_superop(_validation_choi(self),self._input_dim,self._output_dim),self._input_dim,self._output_dim,_origin=origin)
        change = _pauli_change_matrix(self.num_qubits or 0)
        matrix = change @ self.data @ change.conj().T / self._input_dim
        return SuperOp._from_matrix(matrix, self._input_dim, self._output_dim, _origin=origin)


class Chi(_QuantumChannel):
    """Pauli-basis process-matrix representation for n-qubit channels."""

    def __init__(self, data):
        origin = _channel_matrix_origin(data)
        data=_channel_input(data)
        if isinstance(data, Chi):
            self.data = _matrix_array(data.data, copy=True)
            self._input_dim = data._input_dim
            self._output_dim = data._output_dim
            _set_channel_origin(self, origin)
            return
        if (
            isinstance(data, _QuantumChannel)
            or isinstance(data, Operator)
            or hasattr(data, "operations")
            or hasattr(data, "to_matrix")
        ):
            channel = data if isinstance(data, _QuantumChannel) else Kraus(data)
            converted = self.from_choi(channel.to_choi())
            self.data = converted.data
            self._input_dim = converted._input_dim
            self._output_dim = converted._output_dim
            _set_channel_origin(self, origin)
            return
        matrix = _matrix_array(data)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("QuantumBridge Chi data must be square.")
        dimension = isqrt(matrix.shape[0])
        if dimension * dimension != matrix.shape[0] or not _is_power_of_two(dimension):
            raise ValueError("QuantumBridge Chi data must represent an n-qubit channel.")
        self.data = matrix
        self._input_dim = dimension
        self._output_dim = dimension
        _set_channel_origin(self, origin)

    @classmethod
    def from_choi(cls, channel: Choi) -> "Chi":
        origin = _channel_matrix_origin(channel)
        if channel._input_dim != channel._output_dim or not _is_power_of_two(channel._input_dim):
            raise ValueError("QuantumBridge Chi requires an equal-dimension qubit channel.")
        num_qubits = int(np.log2(channel._input_dim))
        change = _pauli_change_matrix(num_qubits)
        if current_resources() is None:matrix=change.conj().T@channel.data@change/channel._input_dim
        else:
            from quantumbridge.execution import mps_arrays as arrays
            matrix=arrays.contracted(arrays.contracted(arrays.conjugated(change).T,channel.data),change)
            np.divide(matrix,channel._input_dim,out=matrix)
        return _set_channel_origin(cls(matrix), origin)

    def to_choi(self) -> Choi:
        origin = _channel_matrix_origin(self)
        if current_resources() is not None:
            return Choi._from_matrix(_validation_choi(self),self._input_dim,self._output_dim,_origin=origin)
        change = _pauli_change_matrix(self.num_qubits or 0)
        matrix = change @ self.data @ change.conj().T / self._input_dim
        return Choi._from_matrix(matrix, self._input_dim, self._output_dim, _origin=origin)

    def to_superop(self) -> SuperOp:
        return self.to_choi().to_superop()


class Stinespring(_QuantumChannel):
    """Standard or generalized Stinespring representation of a quantum map."""

    def __init__(self, data):
        origin = _channel_matrix_origin(data)
        data=_channel_input(data)
        if isinstance(data, Stinespring):
            self._left_data = _matrix_array(data._left_data, copy=True)
            self._right_data = (
                self._left_data
                if data._is_standard
                else _matrix_array(data._right_data, copy=True)
            )
            self._input_dim = data._input_dim
            self._output_dim = data._output_dim
            self._environment_dim = data._environment_dim
        elif isinstance(data, _QuantumChannel):
            kraus = data.to_kraus()
            environment_dim = len(kraus.data)
            if not kraus._is_standard:
                environment_dim = len(kraus._left)
            if current_resources() is None:
                self._left_data = np.stack(kraus._left, axis=1).reshape(kraus._output_dim*environment_dim,kraus._input_dim)
                self._right_data=self._left_data if kraus._is_standard else np.stack(kraus._right,axis=1).reshape(kraus._output_dim*environment_dim,kraus._input_dim)
            else:
                from quantumbridge.execution.array_reservations import owned_arrays
                layout=((kraus._output_dim*environment_dim,kraus._input_dim),complex)
                outputs=owned_arrays([layout]*(1 if kraus._is_standard else 2),category="stinespring_stack")
                for target,source in zip(outputs,(kraus._left,kraus._right)):
                    tensor=target.reshape(kraus._output_dim,environment_dim,kraus._input_dim)
                    for index,matrix in enumerate(source):np.copyto(tensor[:,index,:],matrix,casting="no")
                self._left_data=outputs[0]
                self._right_data=outputs[0] if kraus._is_standard else outputs[1]
            self._input_dim = kraus._input_dim
            self._output_dim = kraus._output_dim
            self._environment_dim = environment_dim
        else:
            generalized = isinstance(data, tuple) and len(data) == 2
            left_data = _operator_data(data[0] if generalized else data, _normalize_hybrid=not generalized)
            right_data = _operator_data(data[1], _normalize_hybrid=False) if generalized else left_data
            if left_data.ndim != 2 or right_data.ndim != 2:
                raise ValueError("QuantumBridge Stinespring data must be a matrix.")
            if left_data.shape != right_data.shape:
                raise ValueError("QuantumBridge generalized Stinespring matrices must have equal shape.")
            self._left_data = left_data
            self._right_data = right_data
            self._input_dim = left_data.shape[1]
            self._output_dim = left_data.shape[1]
            if left_data.shape[0] % self._output_dim:
                raise ValueError("QuantumBridge cannot infer Stinespring output dimensions.")
            self._environment_dim = left_data.shape[0] // self._output_dim

        self._set_public_data()
        _set_channel_origin(self, origin)

    @classmethod
    def _from_matrix(
        cls,
        matrix: np.ndarray,
        input_dim: int,
        output_dim: int,
    ) -> "Stinespring":
        data = _matrix_array(matrix)
        if data.ndim != 2 or data.shape[1] != input_dim:
            raise ValueError("QuantumBridge Stinespring matrix does not match input dimensions.")
        if data.shape[0] % output_dim:
            raise ValueError("QuantumBridge Stinespring matrix does not match output dimensions.")
        instance = cls.__new__(cls)
        instance._left_data = data
        instance._right_data = data
        instance._input_dim = int(input_dim)
        instance._output_dim = int(output_dim)
        instance._environment_dim = data.shape[0] // output_dim
        instance._set_public_data()
        return instance

    def _set_public_data(self) -> None:
        self._is_standard = self._right_data is self._left_data or _same_matrix(
            self._left_data,
            self._right_data,
        )
        if self._is_standard:
            self._right_data = self._left_data
            self.data = self._left_data
        else:
            self.data = (self._left_data, self._right_data)

    def to_kraus(self) -> Kraus:
        origin = _channel_matrix_origin(self)
        left_tensor = self._left_data.reshape(
            self._output_dim,
            self._environment_dim,
            self._input_dim,
        )
        left = tuple(left_tensor[:, index, :] for index in range(self._environment_dim))
        if self._is_standard:
            return _set_channel_origin(Kraus(left), origin)
        right_tensor = self._right_data.reshape(
            self._output_dim,
            self._environment_dim,
            self._input_dim,
        )
        right = tuple(right_tensor[:, index, :] for index in range(self._environment_dim))
        return _set_channel_origin(Kraus((left, right)), origin)

    def to_superop(self) -> SuperOp:
        origin = _channel_matrix_origin(self)
        if current_resources() is not None:
            return SuperOp._from_matrix(_choi_to_superop(_validation_choi(self),self._input_dim,self._output_dim),self._input_dim,self._output_dim,_origin=origin)
        return self.to_kraus().to_superop()

    def to_choi(self) -> Choi:
        origin = _channel_matrix_origin(self)
        if current_resources() is not None:
            return Choi._from_matrix(_validation_choi(self),self._input_dim,self._output_dim,_origin=origin)
        return self.to_kraus().to_choi()


__all__ = ["Chi", "Choi", "Kraus", "PTM", "Stinespring", "SuperOp"]
