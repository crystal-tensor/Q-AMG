# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Known ndarray storage around the existing native MPS numerical kernels."""

from math import prod
import numpy as np

from quantumbridge.core.classical import UnsupportedFeature
from .array_ownership import (
    native_storage_bytes, retain_reservation, retained_array,
    remember_storage_ledger as _remember, storage_ledger as owner_account,
    _storage_owners as _owners,
)
from .array_reservations import ArrayReservation, array_layout, matmul_into
from .resource_context import current_resources


_NO_COPY_RESHAPE = np.lib.NumpyVersion(np.__version__) >= "2.1.0"


def adopt_for_mutation(tensors):
    scope = current_resources()
    if scope is None: return list(tensors)
    adopted = []
    for tensor in tensors:
        owner = owner_account(tensor)
        accounted = owner is not None and any(account is scope.ledger for account in owner._accounts())
        adopted.append(tensor if accounted else copied(tensor, category="mps_adopted_tensor"))
    return adopted


def owned(category, shape, dtype, factory=None):
    scope = current_resources()
    ledger = None if scope is None else scope.ledger
    result = retained_array(ledger, category, shape, dtype, factory)
    _remember(result, ledger)
    return result


def filled(category, shape, dtype, value):
    result = owned(category, shape, dtype)
    result.fill(value)
    return result


def conjugated(value):
    result = owned("mps_conjugate_buffer", value.shape, value.dtype)
    np.conjugate(value, out=result)
    return result


def einsum(expression, shape, *values):
    result = owned("mps_einsum_output", shape, np.result_type(*(value.dtype for value in values)))
    scope = current_resources()
    if scope is not None:
        scope.ledger.unavailable("mps_einsum_allocator_workspace",
            "Explicit output/conjugate buffers are owned; unexposed NumPy iterator workspace is not measured process memory.")
    np.einsum(expression, *values, out=result, optimize=False)
    return result


def copied(value, *, dtype=None, category="mps_tensor_copy"):
    if not issubclass(type(value), np.ndarray):
        if current_resources() is not None:
            raise UnsupportedFeature("mps_array_conversion", "Managed MPS input conversion requires an explicit numeric ndarray buffer.")
        value = np.asarray(value, dtype=dtype)
    value = np.ndarray.view(value, np.ndarray)
    if current_resources() is not None and value.dtype.kind not in "biufc":
        raise UnsupportedFeature("mps_array_conversion", "Managed MPS conversion cannot execute an object-array scalar conversion protocol.")
    dtype = value.dtype if dtype is None else np.dtype(dtype)
    scope = current_resources()
    ledger = owner_account(value) if scope is None else scope.ledger
    result = retained_array(ledger, category, value.shape, dtype)
    _remember(result, ledger)
    np.copyto(result, value, casting="unsafe")
    return result


def copied_tensors(tensors):
    """Admit a complete MPS copy before allocating its first tensor."""
    scope = current_resources()
    inputs, reservations, outputs = [], [], []
    try:
        for tensor in tensors:
            if not issubclass(type(tensor), np.ndarray):
                raise UnsupportedFeature("mps_array_conversion", "Native tensor copies require numeric ndarray buffers.")
            tensor = np.ndarray.view(tensor, np.ndarray)
            if tensor.dtype.kind not in "biufc":
                raise UnsupportedFeature("mps_array_conversion", "Native tensor copies require numeric ndarray buffers.")
            ledger = owner_account(tensor) if scope is None else scope.ledger
            inputs.append((tensor, ledger))
            reservations.append(ArrayReservation(ledger, "mps_tensor_copy", tensor.shape, tensor.dtype))
        for (tensor, ledger), reservation in zip(inputs, reservations):
            result = reservation.create()
            retain_reservation(reservation, result)
            outputs.append(result)
            _remember(result, ledger)
            np.copyto(result, tensor, casting="no")
        return outputs
    finally:
        # Created buffers retain their leases even if an exception owns them.
        for reservation in reservations[len(outputs):]: reservation.release()


def numeric(value):
    """Borrow an existing complex128 buffer or pre-reserve its dtype copy."""
    if issubclass(type(value), np.ndarray):
        value = np.ndarray.view(value, np.ndarray)
        if value.dtype == np.dtype(complex): return value
    return copied(value, dtype=complex, category="mps_input_conversion")


def reshaped(value, shape):
    shape = array_layout(shape, value.dtype)[0]
    if prod(shape) != value.size:
        raise ValueError("MPS reshape dimensions do not match the input buffer.")
    result = np.ndarray.view(value, np.ndarray)
    try:
        if _NO_COPY_RESHAPE:
            return np.reshape(result, shape, copy=False)
        result.shape = shape
        return result
    except (AttributeError, ValueError):
        result = owned("mps_reshape_copy", shape, value.dtype)
        np.copyto(result.reshape(value.shape), value, casting="no")
        return result


def finite(value):
    check = owned("mps_validation_buffer", value.shape, bool)
    np.isfinite(value, out=check)
    return bool(np.all(check))


def contracted(left, right):
    """The existing last/first-axis contraction with explicit matmul output."""
    if left.ndim < 1 or right.ndim < 1 or left.shape[-1] != right.shape[0]:
        raise ValueError("MPS contracted dimensions do not match.")
    shape, dtype = array_layout(left.shape[:-1]+right.shape[1:], np.result_type(left.dtype, right.dtype))[:2]
    result = owned("mps_contraction_output", shape, dtype)
    first = reshaped(left, (prod(left.shape[:-1]), left.shape[-1]))
    second = reshaped(right, (right.shape[0], prod(right.shape[1:])))
    scope = current_resources()
    if scope is not None:
        scope.ledger.unavailable("mps_blas_allocator_workspace",
            "Contraction outputs and explicit operand copies are owned; hidden BLAS/NumPy allocator workspace is not observed RSS.")
    matmul_into(first, second, result.reshape(first.shape[0], second.shape[1]), scope=scope)
    return result


def matrix_power(matrix, exponent):
    """Keep integer-power products under the existing array ownership model."""
    scope = current_resources()
    if scope is None:return np.linalg.matrix_power(matrix, exponent)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Native matrix powers require a square matrix.")
    if exponent == 0:
        result = filled("matrix_power_identity", matrix.shape, matrix.dtype, 0)
        result.flat[::matrix.shape[0]+1] = 1
        return result
    if exponent < 0:
        matrix = owned("matrix_power_inverse", matrix.shape, matrix.dtype, lambda: np.linalg.inv(matrix))
        scope.ledger.unavailable("matrix_inverse_lapack_workspace",
            "The inverse output is owned; internal LAPACK workspace is not measured process memory.")
        exponent = -exponent
    result = None
    while exponent:
        if exponent & 1:result = matrix if result is None else contracted(result, matrix)
        exponent >>= 1
        if exponent:matrix = contracted(matrix, matrix)
    return result


def decomposed(matrix, kind):
    """Reserve all reduced QR/SVD outputs before invoking NumPy/LAPACK."""
    if matrix.ndim != 2 or matrix.dtype not in (np.dtype("float32"), np.dtype("float64"),
            np.dtype("complex64"), np.dtype("complex128")):
        raise ValueError("Native MPS decompositions require a real/complex floating matrix.")
    rows, columns = matrix.shape
    rank = min(rows, columns)
    real = np.dtype("float32" if matrix.dtype.itemsize == (8 if matrix.dtype.kind == "c" else 4) else "float64")
    if kind == "qr":
        layouts = (((rows, rank), matrix.dtype), ((rank, columns), matrix.dtype))
    elif kind == "svd":
        layouts = (((rows, rank), matrix.dtype), ((rank,), real), ((rank, columns), matrix.dtype))
    elif kind == "eigh":
        if rows != columns:raise ValueError("Hermitian eigendecomposition requires a square matrix.")
        layouts = (((rows,), real), ((rows, rows), matrix.dtype))
    else:
        raise ValueError("Unknown native MPS decomposition.")
    scope = current_resources()
    ledger = None if scope is None else scope.ledger
    if ledger is not None:
        ledger.unavailable("mps_lapack_allocator_workspace",
            "Native decomposition output storage is reserved; unexposed NumPy/LAPACK input/workspace allocation is not a measured process-memory bound.")
    reservations, transferred = [], False
    try:
        for index, (shape, dtype) in enumerate(layouts):
            reservations.append(ArrayReservation(ledger, "mps_"+kind+"_output_"+str(index), shape, dtype))
        if kind == "qr":result = np.linalg.qr(matrix, mode="reduced")
        elif kind == "svd":result = np.linalg.svd(matrix, full_matrices=False)
        else:result = np.linalg.eigh(matrix)
        for reservation, value in zip(reservations, result):
            reservation.attach(value)
            if native_storage_bytes(value) != value.nbytes:
                raise UnsupportedFeature("mps_decomposition_storage", "The decomposition returned an unbounded external storage layout.")
        for index, value in enumerate(result):
            if np.shares_memory(value, matrix) or any(np.shares_memory(value, other) for other in result[:index]):
                raise UnsupportedFeature("mps_decomposition_storage", "The decomposition returned aliased external storage.")
        for reservation, value in zip(reservations, result):
            retain_reservation(reservation, value)
            _remember(value, ledger)
        transferred = True
        return result
    finally:
        if not transferred:
            for reservation in reservations: reservation.release()


def divided(value, scalar):
    result = owned("mps_normalized_block", value.shape, np.result_type(value.dtype, scalar))
    np.divide(value, scalar, out=result)
    return result


def norm_squared(value):
    vector = reshaped(value, (value.size,))
    if not vector.flags.c_contiguous:
        vector = copied(vector, category="mps_norm_input_copy")
    scope = current_resources()
    if scope is not None:
        scope.ledger.unavailable("mps_norm_allocator_workspace", "The explicit norm input is bounded; internal linalg/BLAS workspace is not observed RSS.")
    return float(np.linalg.norm(vector)**2)


def component_scale(value):
    """Largest real/imaginary component without squaring tiny amplitudes."""
    scratch = owned("mps_component_scale", value.shape, float)
    np.absolute(value.real, out=scratch)
    scale = float(np.max(scratch))
    if value.dtype.kind == "c":
        np.absolute(value.imag, out=scratch)
        scale = max(scale, float(np.max(scratch)))
    return scale


def rescaled(value, scalar):
    """Scale a tangent buffer, refusing silent loss of nonzero components."""
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        result = divided(value, scalar)
    for original, changed in zip(value.flat, result.flat):
        if ((original.real != 0 and changed.real == 0)
                or (original.imag != 0 and changed.imag == 0)):
            raise FloatingPointError("MPS tangent buffer rescaling underflows complex128.")
    return result
