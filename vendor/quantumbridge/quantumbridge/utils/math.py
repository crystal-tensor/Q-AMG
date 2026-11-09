# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.
# Design source: docs/mvp/simulator_design_v0.1.md, docs/mvp/algorithm_design_v0.1.md,
# and docs/mvp/gradient_design_v0.1.md.

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, sqrt, prod
from numbers import Integral, Real
from typing import Iterable, Optional, Union

import numpy as np


I2 = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def validate_kraus_operators(operators, dimension=None):
    from quantumbridge.execution.resource_context import current_resources
    scope = current_resources()
    if scope is not None:
        from quantumbridge.core.classical import UnsupportedFeature
        from quantumbridge.execution.array_reservations import ArrayReservation, numeric_shape, numeric_into, numeric_operator_sequence
        from quantumbridge.execution.array_ownership import retain_reservation
        operators = numeric_operator_sequence(operators, copy_dtype=complex)
        shapes = [numeric_shape(operator) for operator in operators]
        if not shapes or len(shapes[0]) != 2:
            raise ValueError("QuantumBridge Kraus channels require nonempty square matrices.")
        dimension = shapes[0][0] if dimension is None else dimension
        if not isinstance(dimension, Integral) or isinstance(dimension, bool) or dimension < 1 or dimension & (dimension-1):
            raise ValueError("QuantumBridge channel dimension must be a power of two.")
        if any(shape != (dimension, dimension) for shape in shapes):
            raise ValueError("QuantumBridge Kraus operators must be finite matching square matrices.")
        reservations, matrices, workspace_reservation, workspace = [], [], None, None
        try:
            # Admit the whole defensive copy, not just the first operator.
            for shape in shapes:
                reservations.append(ArrayReservation(scope.ledger, "kraus_operator_copy", shape, complex))
            workspace_reservation = ArrayReservation(scope.ledger, "isometry_validation_workspace", (33*dimension,), np.uint8)
            for operator, reservation in zip(operators, reservations):
                matrix = reservation.create()
                retain_reservation(reservation, matrix)
                matrices.append(matrix)
                numeric_into(matrix, operator)
            workspace = workspace_reservation.create()
            retain_reservation(workspace_reservation, workspace)
            if not _validate_isometry_sum(matrices, dimension, atol=1e-10, rtol=0, arena=workspace):
                raise ValueError("QuantumBridge circuit channels must be finite and trace preserving.")
            return tuple(matrices)
        finally:
            for reservation in reservations[len(matrices):]:reservation.release()
            if workspace_reservation is not None and workspace is None:workspace_reservation.release()
    matrices = tuple(np.array(operator, dtype=complex, copy=True) for operator in operators)
    if not matrices:
        raise ValueError("QuantumBridge Kraus channels require at least one operator.")
    if matrices[0].ndim != 2:
        raise ValueError("QuantumBridge Kraus operators must be square matrices.")
    dimension = matrices[0].shape[0] if dimension is None else dimension
    if dimension < 1 or dimension & (dimension - 1):
        raise ValueError("QuantumBridge channel dimension must be a power of two.")
    if any(matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)) for matrix in matrices):
        raise ValueError("QuantumBridge Kraus operators must be finite matching square matrices.")
    total = sum(matrix.conj().T @ matrix for matrix in matrices)
    if not np.allclose(total, np.eye(dimension), atol=1e-10, rtol=0):
        raise ValueError("QuantumBridge circuit channels must be trace preserving.")
    return matrices


def _validate_isometry_sum(matrices, dimension, *, atol, rtol, arena=None):
    """Check sum(K-dagger K)=I with column vectors, not dense Gram matrices."""
    from quantumbridge.execution.resource_context import current_resources
    from quantumbridge.execution.array_ownership import retained_array
    scope = current_resources()
    rows = matrices[0].shape[0]
    if arena is None:arena = retained_array(scope.ledger, "isometry_validation_workspace", (33*rows,), np.uint8)
    first = arena[:16*rows].view(np.complex128)
    second = arena[16*rows:32*rows].view(np.complex128)
    valid = arena[32*rows:].view(np.bool_)
    scope.ledger.unavailable("validation_scalar_blas_workspace",
        "Two explicit contiguous columns and a finite mask are owned; internal vdot/iterator and Python scalar allocator memory is not observed RSS.")
    for matrix in matrices:
        for column in range(dimension):
            np.isfinite(matrix[:, column], out=valid)
            if not np.all(valid):return False
    for left in range(dimension):
        for right in range(left, dimension):
            value = 0j
            for matrix in matrices:
                np.copyto(first, matrix[:, left], casting="unsafe")
                np.copyto(second, matrix[:, right], casting="unsafe")
                value += complex(np.vdot(first, second))
            expected = int(left == right)
            if not abs(value-expected) <= atol+rtol*expected:return False
    return True


def validate_unitary_matrix(matrix, dimension=None, *, atol=1e-10, rtol=0):
    """Preserve unitarity tolerances with bounded, pre-reserved validation."""
    from quantumbridge.execution.resource_context import current_resources
    scope = current_resources()
    if scope is not None:
        from quantumbridge.execution.array_reservations import numeric_array
        matrix = numeric_array(matrix, category="unitary_validation_input")
    if matrix.ndim != 2:return False
    dimension = matrix.shape[0] if dimension is None else dimension
    if matrix.shape != (dimension, dimension):return False
    if scope is not None:return _validate_isometry_sum((matrix,), dimension, atol=atol, rtol=rtol)
    return bool(np.all(np.isfinite(matrix)) and np.allclose(matrix.conj().T @ matrix, np.eye(dimension), atol=atol, rtol=rtol))


def _local_operator_shapes(dimensions, targets, output_dims, *, density=False):
    if len(output_dims)==len(targets):
        outgoing=list(dimensions)
        for position,dimension in zip(targets,output_dims):outgoing[position]=dimension
        outgoing=tuple(outgoing)
    elif targets==tuple(range(len(dimensions))):outgoing=tuple(output_dims)
    else:raise ValueError("Local operator output dimensions must match the selected subsystem count.")
    before,after=prod(dimensions),prod(outgoing)
    shapes=((after,before),(after,after),(max(before,after),)) if density else ((after,),)
    return outgoing,shapes


def _local_operator_indices(dimensions, targets, output_dims):
    from quantumbridge.information.state_dimensions import basis_digits, basis_index
    outgoing,_=_local_operator_shapes(dimensions,targets,output_dims)
    incoming=tuple(dimensions[position] for position in targets)
    strides=tuple(prod(outgoing[position+1:]) for position in range(len(outgoing)))
    replace_all=len(output_dims)!=len(targets)
    for source in range(prod(dimensions)):
        digits=basis_digits(source,dimensions)
        column=basis_index(tuple(digits[position] for position in targets),incoming)
        base=0 if replace_all else sum(digit*strides[position] for position,digit in enumerate(digits) if position not in targets)
        for row in range(prod(output_dims)):
            destination=row if replace_all else base+sum(digit*strides[position] for position,digit in zip(targets,basis_digits(row,output_dims)))
            yield source,destination,row,column


def _expectation_local_operator(data, matrix, targets, dimensions, *, density=False):
    """The existing ordered local map, contracted with the original state."""
    local=tuple(dimensions[position] for position in targets)
    if density:
        value=sum(complex(data[source,destination])*complex(matrix[row,column])
            for source,destination,row,column in _local_operator_indices(dimensions,targets,local))
    else:
        applied=apply_unitary(data,matrix,targets,None,dims=dimensions)
        value=sum(complex(original).conjugate()*complex(result) for original,result in zip(data,applied))
    return np.complex128(value)


def _compose_local_operator(data, matrix, targets, dimensions, changed, *, front):
    from quantumbridge.execution.array_reservations import owned_arrays
    from quantumbridge.execution.resource_context import current_resources
    outgoing,_=_local_operator_shapes(dimensions,targets,changed)
    shape=(prod(outgoing),data.shape[1]) if front else (data.shape[0],prod(outgoing))
    width=data.shape[1] if front else data.shape[0]
    scope=current_resources()
    if scope is not None:
        scope.ledger.unavailable("operator_compose_iterator_workspace",
            "The final dense Operator and one row/column scratch are owned; Python indices/scalars and hidden ufunc iterator workspace are not measured RSS.")
    result,scratch=owned_arrays([(shape,complex),((width,),complex)],category="operator_mapped_composition")
    result.fill(0)
    old_local=tuple(dimensions[position] for position in targets)
    indices=_local_operator_indices(dimensions,targets,changed) if front else _local_operator_indices(outgoing,targets,old_local)
    # Right multiplication maps each new input column into an old input
    # column; it is linear B, not the conjugated right action in A rho A-dagger.
    for source,destination,row,column in indices:
        old=data[source,:] if front else data[:,destination]
        new=result[destination,:] if front else result[:,source]
        np.multiply(old,matrix[row,column],out=scratch)
        np.add(new,scratch,out=new)
    return result


def _apply_mixed_operator(data, matrix, targets, dimensions, output_dims, *, density=False):
    from quantumbridge.execution import mps_arrays as arrays
    from quantumbridge.execution.array_reservations import numeric_shape, numeric_array, owned_arrays, array_layout
    from quantumbridge.execution.resource_context import current_resources
    from quantumbridge.information.state_dimensions import state_dimensions, subsystem_positions
    scope=current_resources()
    dimensions=state_dimensions(dimensions)
    targets=subsystem_positions(targets,len(dimensions))
    incoming=tuple(dimensions[position] for position in targets)
    output_dims=incoming if output_dims is None else state_dimensions(output_dims)
    outgoing,shapes=_local_operator_shapes(dimensions,targets,output_dims,density=density)
    data_shape=numeric_shape(data) if scope is not None else np.shape(data)
    matrix_shape=numeric_shape(matrix) if scope is not None else np.shape(matrix)
    size=prod(dimensions)
    if data_shape!=((size,size) if density else (size,)) or matrix_shape!=(prod(output_dims),prod(incoming)):
        raise ValueError("Mixed-radix evolution dimensions do not match the numeric buffers.")
    if scope is not None:
        conversions=sum(array_layout(shape,complex)[2] for value,shape in ((data,data_shape),(matrix,matrix_shape))
                        if not issubclass(type(value),np.ndarray) or np.ndarray.view(value,np.ndarray).dtype!=np.dtype(complex))
        scope.ledger.preflight(memory_bytes=conversions+sum(array_layout(shape,complex)[2] for shape in shapes))
        scope.ledger.unavailable("mixed_radix_scalar_iterator_workspace",
            "Known input conversions, outputs and row scratch are admitted; Python index/scalar containers and hidden ufunc iterator workspace are not measured RSS.")
        data=numeric_array(data,category="mixed_evolution_input")
        matrix=numeric_array(matrix,category="mixed_evolution_operator")
    else:
        data=np.asarray(data,dtype=complex)
        matrix=np.asarray(matrix,dtype=complex)
    if not density:
        output=arrays.filled("mixed_state_evolution_output",shapes[0],complex,0)
        for source,destination,row,column in _local_operator_indices(dimensions,targets,output_dims):
            output[destination]+=matrix[row,column]*data[source]
        return output
    # Admit both matrix products and their one reusable row/column scratch together.
    first,output,scratch=owned_arrays([(shape,complex) for shape in shapes],category="mixed_density_evolution")
    first.fill(0)
    output.fill(0)
    old_size,new_size=prod(dimensions),prod(outgoing)
    for source,destination,row,column in _local_operator_indices(dimensions,targets,output_dims):
        np.multiply(data[source,:],matrix[row,column],out=scratch[:old_size])
        np.add(first[destination,:],scratch[:old_size],out=first[destination,:])
    for source,destination,row,column in _local_operator_indices(dimensions,targets,output_dims):
        np.multiply(first[:,source],complex(matrix[row,column]).conjugate(),out=scratch[:new_size])
        np.add(output[:,destination],scratch[:new_size],out=output[:,destination])
    return output


def apply_density_operator(density, operator, wires, num_qubits, *, dims=None, output_dims=None):
    """Apply a local A rho A-dagger without constructing a full-system A."""
    if dims is not None:
        return _apply_mixed_operator(density,operator,wires,dims,output_dims,density=True)
    selected = tuple(wires)
    from quantumbridge.execution.resource_context import current_resources
    if current_resources() is not None:
        from quantumbridge.execution.array_reservations import numeric_array, numeric_shape
        from quantumbridge.execution import mps_arrays as arrays
        dimension = 1 << num_qubits
        if numeric_shape(density)!=(dimension, dimension) or numeric_shape(operator)!=(1<<len(selected),)*2:
            raise ValueError("Density evolution dimensions do not match the local operator.")
        tensor = arrays.reshaped(numeric_array(density, category="density_evolution_input"), (2,)*(2*num_qubits))
        operator = numeric_array(operator, category="density_operator_input")
        def apply_axes(value, matrix, axes):
            order = tuple(axes)+tuple(axis for axis in range(2*num_qubits) if axis not in axes)
            flat = arrays.reshaped(value.transpose(order), (1<<len(axes), 1<<(2*num_qubits-len(axes))))
            output = arrays.contracted(matrix, flat)
            return arrays.reshaped(output, (2,)*(2*num_qubits)).transpose(tuple(order.index(axis) for axis in range(2*num_qubits)))
        tensor = apply_axes(tensor, operator, selected)
        tensor = apply_axes(tensor, arrays.conjugated(operator), tuple(num_qubits+wire for wire in selected))
        return arrays.reshaped(tensor, (dimension, dimension))
    tensor = np.asarray(density, dtype=complex).reshape((2,) * (2 * num_qubits))

    def apply_axes(value, matrix, axes):
        order = tuple(axes) + tuple(axis for axis in range(2 * num_qubits) if axis not in axes)
        flat = value.transpose(order).reshape(1 << len(axes), -1)
        return (matrix @ flat).reshape((2,) * (2 * num_qubits)).transpose(np.argsort(order))

    tensor = apply_axes(tensor, operator, selected)
    tensor = apply_axes(tensor, np.conjugate(operator), tuple(num_qubits + wire for wire in selected))
    return tensor.reshape(1 << num_qubits, 1 << num_qubits)


def apply_density_channel(density, operators, wires, num_qubits):
    from quantumbridge.execution.resource_context import current_resources
    if current_resources() is not None:
        from quantumbridge.execution.array_reservations import numeric_array, numeric_operator_sequence
        from quantumbridge.execution import mps_arrays as arrays
        density = numeric_array(density, category="density_channel_input")
        operators = numeric_operator_sequence(operators)
        if not operators:raise ValueError("Density channels require at least one operator.")
        output = arrays.filled("density_channel_output", density.shape, complex, 0)
        for operator in operators:
            child = apply_density_operator(density, operator, wires, num_qubits)
            np.add(output, child, out=output)
        return output
    return sum(apply_density_operator(density, operator, wires, num_qubits) for operator in operators)


def gate_matrix(name: str, params: tuple[Real, ...] = (), metadata: Optional[dict] = None) -> np.ndarray:
    from quantumbridge.execution.resource_context import current_resources
    scope = current_resources()
    if scope is not None:
        from quantumbridge.execution.array_reservations import numeric_shape
        from quantumbridge.core.classical import UnsupportedFeature
        if type(name) is not str or metadata is not None and type(metadata) is not dict:
            raise UnsupportedFeature("managed_gate_description", "Managed gates require native names and builtin metadata, not foreign dispatch protocols.")
        if len(numeric_shape(params, real_only=True)) != 1:
            raise ValueError("Gate parameters must be a one-dimensional numeric sequence.")
        if issubclass(type(params), np.ndarray):params = np.ndarray.view(params, np.ndarray)
        if name in {"ch", "csx", "csxdg", "crx", "cry", "crz", "cphase", "cp"}:
            scope.ledger.preflight(memory_bytes=16*(4+16))
        elif name == "dcx":scope.ledger.preflight(memory_bytes=3*16*16)
    if metadata and "controlled_base" in metadata:
        from quantumbridge.core.gate_limits import current_gate_limits
        limits = current_gate_limits()
        cursor, seen, total_controls = metadata, set(), 0
        while "controlled_base" in cursor:
            if id(cursor) in seen: raise ValueError("Cyclic controlled primitive description.")
            seen.add(id(cursor))
            limits.check("max_definition_depth", len(seen))
            description = cursor["controlled_base"]
            if type(description) is not dict or type(description.get("metadata")) is not dict or type(description.get("name")) is not str:
                raise ValueError("Malformed controlled primitive description.")
            count = description.get("controls")
            if type(count) is not int or count < 1: raise ValueError("Invalid controlled primitive width.")
            total_controls += count
            limits.check("max_control_qubits", total_controls)
            if total_controls >= limits.max_matrix_dimension.bit_length():
                raise MemoryError("Controlled matrix exceeds max_matrix_dimension before allocation.")
            cursor = description["metadata"]
        description = metadata["controlled_base"]
        base = gate_matrix(description["name"], params, description["metadata"])
        if description.get("inverse"): base = base.conj().T
        dimension = (1 << description["controls"])*len(base)
        limits.check("max_matrix_dimension", dimension)
        out = _gate_identity(dimension)
        out[-len(base):, -len(base):] = base
        return out
    if name == "native_matrix_power":
        from quantumbridge.diff.matrix_power_jet import matrix_power_value
        return matrix_power_value(params, metadata)
    if name == "global_phase": return _gate_array(I2, factor=np.exp(1j * float(params[0])))
    if name in {"id", "i"}:
        return I2 if scope is None else _gate_array(I2)
    if name == "x":
        return PAULI_X if scope is None else _gate_array(PAULI_X)
    if name == "y":
        return PAULI_Y if scope is None else _gate_array(PAULI_Y)
    if name == "z":
        return PAULI_Z if scope is None else _gate_array(PAULI_Z)
    if name == "h":
        return _gate_array([[1, 1], [1, -1]], divisor=sqrt(2))
    if name == "sx":
        return _gate_array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], factor=.5)
    if name == "sxdg":
        return _gate_array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], factor=.5)
    if name == "s":
        return _gate_array([[1, 0], [0, 1j]])
    if name == "sdg":
        return _gate_array([[1, 0], [0, -1j]])
    if name == "t":
        return _gate_array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
    if name == "tdg":
        return _gate_array([[1, 0], [0, np.exp(-1j * np.pi / 4)]])
    if name == "rx":
        theta = float(params[0])
        return _gate_array([[cos(theta / 2), -1j * sin(theta / 2)], [-1j * sin(theta / 2), cos(theta / 2)]])
    if name == "ry":
        theta = float(params[0])
        return _gate_array([[cos(theta / 2), -sin(theta / 2)], [sin(theta / 2), cos(theta / 2)]])
    if name == "rz":
        theta = float(params[0])
        return _gate_array([[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]])
    if name in {"phase", "p", "u1"}:
        theta = float(params[0])
        return _gate_array([[1, 0], [0, np.exp(1j * theta)]])
    if name == "u2":
        return _u_matrix(np.pi / 2, float(params[0]), float(params[1]))
    if name in {"u", "u3"}:
        return _u_matrix(float(params[0]), float(params[1]), float(params[2]))
    if name == "cx":
        return _gate_array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        )
    if name == "cy":
        return _controlled_one_qubit_matrix(PAULI_Y)
    if name == "cz":
        out = _gate_identity(4)
        out[3, 3] = -1
        return out
    if name == "ch":
        return _controlled_one_qubit_matrix(gate_matrix("h"))
    if name == "csx":
        return _controlled_one_qubit_matrix(gate_matrix("sx"))
    if name == "csxdg":
        return _controlled_one_qubit_matrix(gate_matrix("sxdg"))
    if name in {"crx", "cry", "crz", "cphase", "cp"}:
        target_name = {"crx": "rx", "cry": "ry", "crz": "rz", "cphase": "phase", "cp": "phase"}[name]
        target = gate_matrix(target_name, params)
        return _controlled_one_qubit_matrix(target)
    if name in {"rxx", "ryy", "rzz"}:
        theta = float(params[0])
        out = _gate_identity(4, factor=cos(theta / 2))
        flip, parity, phase = _pauli_permutation(name[-1].upper()*2, -1j*sin(theta/2))
        for column in range(4):out[column ^ flip, column] += -phase if (column & parity).bit_count() & 1 else phase
        return out
    if name == "swap":
        return _gate_array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    if name == "dcx":
        from quantumbridge.execution import mps_arrays as arrays
        return arrays.contracted(_reverse_control_cx_matrix(), gate_matrix("cx"))
    if name == "ecr":
        return (
            _gate_array(
                [[0, 0, 1, 1j], [0, 0, 1j, 1], [1, -1j, 0, 0], [-1j, 1, 0, 0]],
                divisor=sqrt(2),
            )
        )
    if name == "iswap":
        return _gate_array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
    if name == "ccx":
        matrix = _gate_identity(8)
        matrix[6, 6] = 0
        matrix[7, 7] = 0
        matrix[6, 7] = 1
        matrix[7, 6] = 1
        return matrix
    if name == "cswap":
        matrix = _gate_identity(8)
        matrix[5, 5] = 0
        matrix[6, 6] = 0
        matrix[5, 6] = 1
        matrix[6, 5] = 1
        return matrix
    if metadata and "matrix" in metadata:
        if scope is not None:
            from quantumbridge.execution.array_reservations import numeric_array
            return numeric_array(metadata["matrix"], category="native_gate_input_conversion")
        return np.asarray(metadata["matrix"], dtype=complex)
    raise ValueError(f"QuantumBridge operation {name!r} is not supported by the MVP simulator.")


def _gate_array(value, *, factor=None, divisor=None):
    from quantumbridge.execution.resource_context import current_resources
    if current_resources() is None:result = np.array(value, dtype=complex)
    else:
        from quantumbridge.execution.array_reservations import numeric_array
        result = numeric_array(value, category="native_gate_matrix", copy=True)
    if factor is not None:np.multiply(result, factor, out=result)
    if divisor is not None:np.divide(result, divisor, out=result)
    return result


def _gate_identity(dimension, *, factor=1):
    from quantumbridge.execution.resource_context import current_resources
    from quantumbridge.execution.array_ownership import retained_array
    scope = current_resources()
    result = retained_array(None if scope is None else scope.ledger, "native_gate_matrix", (dimension, dimension), complex)
    result.fill(0)
    result.flat[::dimension+1] = factor
    return result


def _u_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    from quantumbridge.execution.resource_context import current_resources
    if current_resources() is not None:
        from quantumbridge.execution.array_reservations import numeric_shape
        if any(numeric_shape(value, real_only=True) for value in (theta, phi, lam)):
            raise ValueError("U gate angles must be real scalars.")
    return _gate_array(
        [
            [cos(theta / 2), -np.exp(1j * lam) * sin(theta / 2)],
            [np.exp(1j * phi) * sin(theta / 2), np.exp(1j * (phi + lam)) * cos(theta / 2)],
        ],
    )


def _controlled_one_qubit_matrix(target: np.ndarray) -> np.ndarray:
    from quantumbridge.execution.resource_context import current_resources
    scope = current_resources()
    if scope is not None:
        from quantumbridge.execution.array_reservations import numeric_shape, numeric_into
        if numeric_shape(target) != (2, 2):raise ValueError("Controlled gate target must have shape (2,2).")
    out = _gate_identity(4)
    if scope is None:out[2:4, 2:4] = np.asarray(target, dtype=complex)
    else:numeric_into(out[2:4, 2:4], target)
    return out


def _reverse_control_cx_matrix() -> np.ndarray:
    return _gate_array(
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
    )


def bit_at(index: int, wire: int, num_qubits: int) -> int:
    return (index >> (num_qubits - 1 - wire)) & 1


def set_bit(index: int, wire: int, value: int, num_qubits: int) -> int:
    mask = 1 << (num_qubits - 1 - wire)
    return (index | mask) if value else (index & ~mask)


def apply_unitary(state: np.ndarray, matrix: np.ndarray, targets: tuple[int, ...], num_qubits: int | None, *, dims=None, output_dims=None) -> np.ndarray:
    if dims is not None:return _apply_mixed_operator(state,matrix,targets,dims,output_dims)
    targets = tuple(targets)
    size = 1 << num_qubits
    dim = 1 << len(targets)
    from quantumbridge.execution.resource_context import current_resources
    scope = current_resources()
    if scope is not None:
        from quantumbridge.execution.array_reservations import numeric_array, numeric_shape
        if numeric_shape(state)!=(size,) or numeric_shape(matrix)!=(dim, dim):
            raise ValueError("QuantumBridge unitary dimensions do not match the state and wires.")
        state = numeric_array(state, category="dense_evolution_input")
        matrix = numeric_array(matrix, category="dense_evolution_operator")
    if matrix.shape != (dim, dim):
        raise ValueError("QuantumBridge unitary shape does not match target wires.")
    if scope is None:out = np.zeros_like(state, dtype=complex)
    else:
        from quantumbridge.execution import mps_arrays as arrays
        out = arrays.filled("dense_evolution_output", state.shape, complex, 0)
    for basis in range(size):
        col = 0
        base = basis
        for wire in targets:
            col = (col << 1) | bit_at(basis, wire, num_qubits)
            base = set_bit(base, wire, 0, num_qubits)
        for row in range(dim):
            dest = base
            for offset, wire in enumerate(reversed(targets)):
                dest = set_bit(dest, wire, (row >> offset) & 1, num_qubits)
            out[dest] += matrix[row, col] * state[basis]
    return out


def probabilities_from_state(state: np.ndarray, num_qubits: int, wires: Optional[Iterable[int]] = None) -> dict[str, float]:
    selected = tuple(range(num_qubits) if wires is None else wires)
    probs: dict[str, float] = {}
    for index, amp in enumerate(state):
        key = "".join(str(bit_at(index, wire, num_qubits)) for wire in selected)
        probs[key] = probs.get(key, 0.0) + float(abs(amp) ** 2)
    return probs


@dataclass(frozen=True)
class PauliString:
    """Tensor product of Pauli operators over selected wires."""

    terms: tuple[tuple[int, str], ...]

    def __init__(self, terms: Iterable[tuple[int, str]]):
        normalized = tuple((int(wire), label.upper()) for wire, label in terms)
        for _, label in normalized:
            if label not in {"I", "X", "Y", "Z"}:
                raise ValueError("QuantumBridge PauliString labels must be I, X, Y, or Z.")
        object.__setattr__(self, "terms", normalized)

    @staticmethod
    def x(wire: int) -> "PauliString":
        return PauliString([(wire, "X")])

    @staticmethod
    def y(wire: int) -> "PauliString":
        return PauliString([(wire, "Y")])

    @staticmethod
    def z(wire: int) -> "PauliString":
        return PauliString([(wire, "Z")])

    def tensor(self, other: "PauliString") -> "PauliString":
        occupied = {wire for wire, _ in self.terms}
        for wire, _ in other.terms:
            if wire in occupied:
                raise ValueError("QuantumBridge PauliString tensor inputs cannot overlap wires.")
        return PauliString(self.terms + other.terms)

    def matrix(self, num_qubits: Optional[int] = None) -> np.ndarray:
        if num_qubits is None:
            num_qubits = 1 + max((wire for wire, _ in self.terms), default=0)
        term_map = dict(self.terms)
        return _pauli_dense("".join(term_map.get(wire, "I") for wire in range(_dense_width(num_qubits))))


@dataclass(frozen=True)
class Hamiltonian:
    """Weighted sum of PauliString observables."""

    terms: tuple[tuple[object, PauliString], ...]

    def __init__(self, terms: Iterable[tuple[Real, PauliString]]):
        from quantumbridge.diff.scalars import retain_coefficient
        normalized = []
        for coeff, pauli in terms:
            if not isinstance(pauli, PauliString):
                raise ValueError("QuantumBridge Hamiltonian terms require PauliString observables.")
            normalized.append((retain_coefficient(coeff), pauli))
        object.__setattr__(self, "terms", tuple(normalized))

    def __add__(self, other: "Hamiltonian") -> "Hamiltonian":
        return Hamiltonian(self.terms + other.terms)

    def __mul__(self, scalar: Real) -> "Hamiltonian":
        from quantumbridge.diff.scalars import retain_coefficient
        scalar = retain_coefficient(scalar)
        return Hamiltonian([(scalar * coeff, pauli) for coeff, pauli in self.terms])

    __rmul__ = __mul__


def PauliX(wire: int) -> PauliString:
    return PauliString.x(wire)


def PauliY(wire: int) -> PauliString:
    return PauliString.y(wire)


def PauliZ(wire: int) -> PauliString:
    return PauliString.z(wire)


def pauli_matrix(label: str) -> np.ndarray:
    return {"I": I2, "X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z}[label]


def expectation_pauli_string(state: np.ndarray, num_qubits: int, pauli: PauliString) -> float:
    op_state = state
    term_map = dict(pauli.terms)
    for wire in range(num_qubits):
        matrix = pauli_matrix(term_map.get(wire, "I"))
        op_state = apply_unitary(op_state, matrix, (wire,), num_qubits)
    value = np.vdot(state, op_state)
    return float(np.real_if_close(value))


def expectation_hamiltonian(state: np.ndarray, num_qubits: int, hamiltonian: Union[Hamiltonian, PauliString]) -> float:
    if isinstance(hamiltonian, PauliString):
        return expectation_pauli_string(state, num_qubits, hamiltonian)
    return float(sum(coeff * expectation_pauli_string(state, num_qubits, pauli) for coeff, pauli in hamiltonian.terms))


def _dense_width(width):
    if isinstance(width, bool) or not isinstance(width, Integral) or width < 0:
        raise ValueError("Dense observable width must be a nonnegative integer.")
    return int(width)


def _sum_pauli_dense(width, terms):
    """Fill the existing dense Pauli sum from its signed basis permutation."""
    from quantumbridge.execution.resource_context import current_resources
    from quantumbridge.execution.array_ownership import retained_array
    scope = current_resources()
    size = 1 << _dense_width(width)
    out = retained_array(None if scope is None else scope.ledger, "dense_pauli_output", (size, size), complex)
    out.fill(0)
    for label, coefficient in terms:
        if len(label) != width or set(label)-set("IXYZ"):
            raise ValueError("Dense Pauli terms require matching I/X/Y/Z labels.")
        flip, parity, phase = _pauli_permutation(label, coefficient)
        for column in range(size):
            out[column ^ flip, column] += -phase if (column & parity).bit_count() & 1 else phase
    return out


def _pauli_permutation(label, phase=1):
    """Signed computational-basis permutation shared by dense/CSR/moments."""
    flip = parity = 0
    for index, char in enumerate(label):
        mask = 1 << (len(label)-1-index)
        if char in "XY":flip |= mask
        if char in "YZ":parity |= mask
        if char == "Y":phase *= 1j
    return flip, parity, phase


def _pauli_dense(label, phase=1):
    return _sum_pauli_dense(len(label), ((label, phase),))


def validate_observable_matrix(matrix, width, *, atol=1e-10, rtol=0):
    return validate_hermitian_matrix(matrix, 1 << width, atol=atol, rtol=rtol)


def validate_hermitian_matrix(matrix, dimension, *, atol=1e-10, rtol=0):
    """Finite Hermitian validation with one explicitly owned scratch arena."""
    from quantumbridge.execution.resource_context import current_resources
    from quantumbridge.execution.array_ownership import retained_array
    if matrix.shape != (dimension,)*2:return False
    scope = current_resources()
    if scope is None:
        return bool(np.all(np.isfinite(matrix)) and np.allclose(matrix, matrix.conj().T, atol=atol, rtol=rtol))
    count, shape = matrix.size, matrix.shape
    arena = retained_array(scope.ledger, "observable_validation_workspace", (33*count,), np.uint8)
    delta = arena[:16*count].view(np.complex128).reshape(shape)
    distance = arena[16*count:24*count].view(np.float64).reshape(shape)
    threshold = arena[24*count:32*count].view(np.float64).reshape(shape)
    valid = arena[32*count:].view(np.bool_).reshape(shape)
    scope.ledger.unavailable("observable_validation_iterator_workspace",
        "The explicit validation arena is owned; NumPy iterator/scalar allocator workspace is not observed process RSS.")
    np.isfinite(matrix, out=valid)
    if not np.all(valid):return False
    np.conjugate(matrix.T, out=delta)
    np.subtract(matrix, delta, out=delta)
    np.absolute(delta, out=distance)
    np.absolute(matrix.T, out=threshold)
    np.multiply(threshold, rtol, out=threshold)
    np.add(threshold, atol, out=threshold)
    np.less_equal(distance, threshold, out=valid)
    return bool(np.all(valid))


def observable_matrix(observable, num_qubits: int) -> np.ndarray:
    from quantumbridge.execution.resource_context import current_resources
    scope = current_resources()
    if scope is not None:
        from quantumbridge.execution.array_reservations import numeric_array, array_layout
        from quantumbridge.core.classical import UnsupportedFeature
        from quantumbridge.operators.pauli import Pauli, _PHASE_FACTORS
        from quantumbridge.operators.sparse_operator import SparsePauliOperator
        from quantumbridge.information.operator import Operator
        from quantumbridge.chemistry.qubit_hamiltonian import QubitHamiltonian
        width = _dense_width(num_qubits)
        if type(observable) is PauliString:return observable.matrix(width)
        if type(observable) is Hamiltonian:
            return _sum_pauli_dense(width, (("".join(dict(pauli.terms).get(wire, "I") for wire in range(width)), coefficient)
                for coefficient, pauli in observable.terms))
        if type(observable) is Pauli:return _pauli_dense(observable._body, _PHASE_FACTORS[observable.phase])
        if type(observable) is SparsePauliOperator:
            return _sum_pauli_dense(width, ((label.ljust(width, "I"), coefficient) for label, coefficient in observable.terms))
        if type(observable) is Operator:return numeric_array(observable.data, category="observable_dtype_conversion")
        if type(observable) is QubitHamiltonian:return QubitHamiltonian.matrix(observable, width)
        if issubclass(type(observable), np.ndarray) or type(observable) is list or type(observable) is tuple:
            return numeric_array(observable, category="observable_dtype_conversion")
        scope.ledger.preflight(memory_bytes=array_layout((1 << width,)*2, complex)[2])
        raise UnsupportedFeature("managed_observable_conversion", "Managed observables require native Pauli/Operator objects or explicit numeric buffers; opaque conversion factories are not bounded.")
    if isinstance(observable, PauliString):
        return observable.matrix(num_qubits)
    if isinstance(observable, Hamiltonian):
        out = np.zeros((1 << num_qubits, 1 << num_qubits), dtype=complex)
        for coeff, pauli in observable.terms:
            out += coeff * pauli.matrix(num_qubits)
        return out
    if hasattr(observable, "to_hamiltonian"):
        return observable_matrix(observable.to_hamiltonian(), num_qubits)
    if hasattr(observable, "to_matrix"):
        try:
            matrix = observable.to_matrix(num_qubits=num_qubits)
        except TypeError:
            matrix = observable.to_matrix()
        return np.asarray(matrix, dtype=complex)
    if hasattr(observable, "matrix"):
        try:
            matrix = observable.matrix(num_qubits=num_qubits)
        except TypeError:
            matrix = observable.matrix()
        return np.asarray(matrix, dtype=complex)
    return np.asarray(observable, dtype=complex)


def expectation_observable(state: np.ndarray, num_qubits: int, observable) -> complex:
    if isinstance(observable, (Hamiltonian, PauliString)) or hasattr(observable, "to_hamiltonian"):
        hamiltonian = observable.to_hamiltonian() if hasattr(observable, "to_hamiltonian") else observable
        return complex(expectation_hamiltonian(state, num_qubits, hamiltonian))
    matrix = observable_matrix(observable, num_qubits)
    if matrix.shape != (1 << num_qubits, 1 << num_qubits):
        raise ValueError("QuantumBridge observable matrix shape does not match state dimension.")
    return np.vdot(state, matrix @ state)


def variance_observable(state: np.ndarray, num_qubits: int, observable) -> float:
    matrix = observable_matrix(observable, num_qubits)
    if matrix.shape != (1 << num_qubits, 1 << num_qubits):
        raise ValueError("QuantumBridge observable matrix shape does not match state dimension.")
    mean = np.vdot(state, matrix @ state)
    second_moment = np.vdot(state, matrix @ (matrix @ state))
    return float(np.real_if_close(second_moment - mean * mean))
