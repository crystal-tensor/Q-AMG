# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Preallocation contracts for native array outputs and explicit workspace."""

from math import prod
from numbers import Integral
from contextlib import ExitStack

import numpy as np

from .derivative_status import ResourceLimitError


_NUMERIC_SCALAR_TYPES = (bool, int, float, complex, np.bool_, np.int8, np.int16,
    np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.longlong,
    np.ulonglong, np.float16, np.float32, np.float64, np.longdouble,
    np.complex64, np.complex128, np.clongdouble)
_DLPACK_COPY_KEYWORD = np.lib.NumpyVersion(np.__version__) >= "2.1.0"


def borrow_callback_array(value):
    """Borrow CPU callback buffers without a hidden dtype/device copy."""
    if isinstance(value, np.ndarray):
        return value
    from quantumbridge.core.classical import UnsupportedFeature
    if not hasattr(value, "__dlpack_device__") or value.__dlpack_device__()[0] != 1:
        raise UnsupportedFeature("framework_callback_buffer", "Native callback inputs require a borrowable CPU buffer.")
    convert = getattr(np, "from_dlpack", None)
    if convert is None:
        raise UnsupportedFeature("framework_callback_buffer", "This NumPy version cannot borrow DLPack callback storage.")
    try:
        # Older NumPy borrows via the original DLPack protocol, before copy
        # negotiation was added. Do not retry a producer's own TypeError.
        return convert(value, copy=False) if _DLPACK_COPY_KEYWORD else convert(value)
    except (BufferError, TypeError, RuntimeError) as error:
        raise UnsupportedFeature("framework_callback_buffer", "This callback buffer cannot be borrowed without copying.") from error


def array_layout(shape, dtype):
    """Validate shape arithmetic with Python integers before NumPy allocates."""
    shape = tuple(shape)
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in shape):
        raise ValueError("Native array dimensions must be nonnegative integers.")
    shape = tuple(int(n) for n in shape)
    dtype = np.dtype(dtype)
    if dtype.kind not in "biufc":
        raise TypeError("Native resource arrays require numeric non-object dtypes.")
    maximum = int(np.iinfo(np.intp).max)
    count = prod(shape)
    nbytes = count*dtype.itemsize
    if any(n > maximum for n in shape) or count > maximum or nbytes > maximum:
        raise ResourceLimitError("array_bytes", max(nbytes, count, *shape), maximum, stage="array_preflight")
    return shape, dtype, nbytes


def numeric_shape(value, *, real_only=False, max_depth=32, _depth=0, _ancestors=None):
    """Inspect supported numeric containers without invoking array protocols."""
    from quantumbridge.core.classical import UnsupportedFeature
    if _depth > max_depth:
        raise UnsupportedFeature("numeric_container_depth", "Numeric conversion nesting exceeds its supported bound.")
    if issubclass(type(value), np.ndarray):
        value = np.ndarray.view(value, np.ndarray)
        if value.dtype.kind not in ("biuf" if real_only else "biufc"):
            raise UnsupportedFeature("numeric_array_dtype", "Numeric conversion cannot execute object/string scalar protocols or discard imaginary data.")
        return value.shape
    if any(type(value) is kind for kind in _NUMERIC_SCALAR_TYPES):
        if real_only and (type(value) is complex or isinstance(value, np.complexfloating)):
            raise ValueError("Real IR components cannot contain complex scalars.")
        if isinstance(value, np.generic) and value.dtype.kind not in "biufc":
            raise UnsupportedFeature("numeric_scalar_dtype")
        return ()
    if type(value) is not list and type(value) is not tuple:
        raise UnsupportedFeature("numeric_array_conversion", "Managed conversion requires explicit numeric buffers or builtin containers.")
    ancestors = set() if _ancestors is None else _ancestors
    if id(value) in ancestors:raise ValueError("Numeric containers cannot be cyclic.")
    ancestors.add(id(value))
    try:
        shape = None
        for child in value:
            current = numeric_shape(child, real_only=real_only, max_depth=max_depth,
                _depth=_depth+1, _ancestors=ancestors)
            if shape is None:shape = current
            elif current != shape:raise ValueError("Numeric containers must have rectangular matching shapes.")
        return (len(value),)+(() if shape is None else shape)
    finally:
        ancestors.remove(id(value))


def _numeric_leaves(value):
    if issubclass(type(value), np.ndarray):
        yield from np.ndarray.view(value, np.ndarray).flat
    elif type(value) is list or type(value) is tuple:
        for child in value:yield from _numeric_leaves(child)
    else:
        yield value


def numeric_into(output, value):
    """Fill an admitted buffer without a full temporary array for Python lists."""
    if issubclass(type(value), np.ndarray):
        np.copyto(output, np.ndarray.view(value, np.ndarray), casting="unsafe")
    else:
        flat = output.flat
        for index, scalar in enumerate(_numeric_leaves(value)):flat[index] = scalar
    return output


def numeric_array(value, dtype=complex, *, category="native_numeric_conversion", copy=False):
    from .resource_context import current_resources
    from .array_ownership import retained_array
    scope, dtype = current_resources(), np.dtype(dtype)
    shape = numeric_shape(value, real_only=dtype.kind != "c")
    if issubclass(type(value), np.ndarray):
        value = np.ndarray.view(value, np.ndarray)
        if not copy and value.dtype == dtype:return value
    result = retained_array(None if scope is None else scope.ledger, category, shape, dtype)
    if scope is not None:
        scope.ledger.unavailable("numeric_conversion_iterator_workspace",
            "Owned conversion buffers are reserved; internal scalar/iterator allocator workspace is not observed Python heap or RSS.")
    return numeric_into(result, value)


def numeric_operator_sequence(values, *, max_count=None, copy_dtype=None):
    """Borrow explicit matrix groups without executing foreign iterators."""
    from quantumbridge.core.classical import UnsupportedFeature
    from .resource_context import current_resources
    if issubclass(type(values), np.ndarray):
        shape = numeric_shape(values)
        values = np.ndarray.view(values, np.ndarray)
        if len(shape) != 3:raise ValueError("Stacked native operators require rank three.")
        if max_count is not None and shape[0] > max_count:raise ValueError("Native operator group exceeds its construction limit.")
        scope = current_resources()
        if scope is not None and copy_dtype is not None:
            scope.ledger.preflight(memory_bytes=array_layout(shape, copy_dtype)[2])
        return tuple(values[index] for index in range(shape[0]))
    if type(values) is not list and type(values) is not tuple:
        raise UnsupportedFeature("managed_operator_sequence", "Managed operators require a builtin sequence or explicit numeric stack; foreign iteration is not bounded.")
    if max_count is not None and len(values) > max_count:raise ValueError("Native operator group exceeds its construction limit.")
    return tuple(values)


def numeric_array_copies(values, *, category="native_array_group_copy"):
    """Admit all independent numeric destinations before the first copy."""
    values = numeric_operator_sequence(values, copy_dtype=complex)
    shapes = [numeric_shape(value) for value in values]
    outputs = owned_arrays([(shape, complex) for shape in shapes], category=category)
    for value, output in zip(values, outputs):numeric_into(output, value)
    return outputs


def owned_arrays(layouts, *, category="native_array_group"):
    """Acquire a whole known group, then give every storage root its lease."""
    from .resource_context import current_resources
    from .array_ownership import retain_reservation
    scope = current_resources()
    reservations, outputs = [], []
    try:
        for shape, dtype in layouts:
            reservations.append(ArrayReservation(None if scope is None else scope.ledger, category, shape, dtype))
        for reservation in reservations:
            output = reservation.create()
            retain_reservation(reservation, output)
            outputs.append(output)
        return tuple(outputs)
    finally:
        for reservation in reservations[len(outputs):]:reservation.release()


class ArrayReservation:
    """One numerical buffer reservation, acquired before its factory is called.

    It deliberately does not hold an array reference: callers determine whether
    its lifetime is a temporary scope or a separately retained owner. Views of
    an already owned buffer must reuse its owner, not acquire a new allocation.
    """

    def __init__(self, ledger, category, shape, dtype):
        self.shape, self.dtype, self.nbytes = array_layout(shape, dtype)
        self.ticket = None if ledger is None else ledger.reserve(category, memory_bytes=self.nbytes)
        self.released = False

    def __enter__(self):
        if self.released: raise ValueError("Cannot reopen a released array reservation.")
        return self

    def __exit__(self, *exception):
        self.release()

    def release(self):
        if not self.released:
            if self.ticket is not None: self.ticket.release()
            self.released = True

    def attach(self, array):
        if self.released: raise ValueError("Cannot attach to a released array reservation.")
        if not isinstance(array, np.ndarray) or array.shape != self.shape or array.dtype != self.dtype or array.nbytes != self.nbytes:
            raise RuntimeError("Native array allocation differs from its reserved shape or dtype.")
        if self.ticket is not None: self.ticket.observe(array.nbytes)
        return array

    def create(self, factory=None):
        try:
            return self.attach(np.empty(self.shape, self.dtype) if factory is None else factory())
        except BaseException:
            self.release()
            raise


def scope_array(scope, category, shape, dtype, factory=None):
    """Retain an owned output until this request scope ends."""
    allocation = ArrayReservation(None if scope is None else scope.ledger, category, shape, dtype)
    if scope is not None: scope.stack.enter_context(allocation)
    return allocation.create(factory)


def matmul_layout(left, right):
    if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
        raise TypeError("Native mapped products require ndarray inputs; conversion has its own ownership.")
    if left.ndim < 1 or right.ndim < 1:
        raise ValueError("Native matmul does not accept scalar operands.")
    inner = right.shape[-2] if right.ndim > 1 else right.shape[0]
    if left.shape[-1] != inner:
        raise ValueError("Native matmul inner dimensions do not match.")
    first_batch = left.shape[:-2] if left.ndim > 1 else ()
    second_batch = right.shape[:-2] if right.ndim > 1 else ()
    batch = np.broadcast_shapes(first_batch, second_batch) if first_batch or second_batch else ()
    shape = batch+(() if left.ndim == 1 else (left.shape[-2],))+(() if right.ndim == 1 else (right.shape[-1],))
    return array_layout(shape, np.result_type(left.dtype, right.dtype))[:2]


def matmul_into(left, right, output, *, scope=None):
    shape, dtype = matmul_layout(left, right)
    if output.shape != shape or not np.can_cast(dtype, output.dtype, casting="safe"):
        raise ValueError("Native matmul output shape or dtype is incompatible.")
    if not output.flags.writeable or not (output.flags.c_contiguous or output.flags.f_contiguous):
        raise ValueError("Native matmul requires a writable contiguous output buffer.")
    if np.may_share_memory(output, left) or np.may_share_memory(output, right):
        raise ValueError("Native mapped products require non-overlapping output storage.")
    ledger = None if scope is None else scope.ledger
    with ExitStack() as temporaries:
        def operand(array):
            if array.dtype == output.dtype and (array.flags.c_contiguous or array.flags.f_contiguous):
                return array
            reservation = temporaries.enter_context(ArrayReservation(ledger,
                "matmul_operand_workspace", array.shape, output.dtype))
            return reservation.create(lambda: np.array(array, dtype=output.dtype, order="C", copy=True))
        first = second = None
        try:
            first, second = operand(left), operand(right)
            np.matmul(first, second, out=output)
        finally:
            first = second = None
    return output


def kron_into(left, right, output, *, scope=None):
    """Fill a pre-admitted two-matrix Kronecker destination without stacking."""
    if left.ndim!=2 or right.ndim!=2 or output.shape!=(left.shape[0]*right.shape[0],left.shape[1]*right.shape[1]):
        raise ValueError("Native Kronecker matrix dimensions do not match.")
    if not output.flags.c_contiguous or np.may_share_memory(output,left) or np.may_share_memory(output,right):
        raise ValueError("Native Kronecker output requires independent contiguous storage.")
    if scope is not None:
        scope.ledger.unavailable("kron_iterator_workspace","The Kronecker destination is owned; internal NumPy iterator/scalar workspace is not measured RSS.")
    np.einsum("ij,kl->ikjl",left,right,out=output.reshape(left.shape[0],right.shape[0],left.shape[1],right.shape[1]),optimize=False)
    return output


def mapped_matmul(scope, category, left, right):
    shape, dtype = matmul_layout(left, right)
    result = scope_array(scope, category, shape, dtype)
    if scope is not None:
        scope.ledger.unavailable("numpy_blas_allocator_peak",
            "Owned outputs and explicit work arrays are reserved; unexposed NumPy/BLAS allocator workspace is not an observed process-memory bound.")
    return matmul_into(left, right, result, scope=scope)


def mapped_second(scope, internal, classical, own, source_second, *, directional):
    """The existing second source-map chain rule with explicit bounded arrays.

    Full: J.T @ H[r] @ J, one recorder row at a time. Directional: the
    already-directional native map @ J. No full input Hessian is formed for HVP.
    """
    if not all(isinstance(a, np.ndarray) for a in (internal, classical, own, source_second)):
        raise TypeError("Native second maps require ndarray inputs.")
    if own.ndim != 2 or classical.ndim != 2 or own.shape[1] != classical.shape[0]:
        raise ValueError("Native second map dimensions do not match.")
    rows, packed = own.shape
    inputs = classical.shape[1]
    columns = 1 if directional else inputs
    if internal.shape != (rows, packed, 1 if directional else packed) or source_second.shape != (packed, inputs, columns):
        raise ValueError("Native second map tensor shapes do not match the declared direction.")
    dtype = np.result_type(internal.dtype, classical.dtype, own.dtype, source_second.dtype)
    result = scope_array(scope, "derivative_mapped_second", (rows, inputs, columns), dtype)
    if scope is not None:
        scope.ledger.unavailable("numpy_blas_allocator_peak",
            "Second mapping uses row/column-bounded owned workspace; internal BLAS/LAPACK allocation is not observed RSS.")
    if rows == 0 or inputs == 0:
        return result
    ledger = None if scope is None else scope.ledger
    if directional:
        matmul_into(internal[:, :, 0], classical, result[:, :, 0], scope=scope)
    else:
        with ArrayReservation(ledger, "second_contraction_workspace", (inputs, packed), dtype) as reservation:
            working = reservation.create()
            try:
                for row in range(rows):
                    matmul_into(classical.T, internal[row], working, scope=scope)
                    matmul_into(working, classical, result[row], scope=scope)
            finally:
                del working
    with ArrayReservation(ledger, "classical_second_workspace", (rows, columns), dtype) as reservation:
        working = reservation.create()
        try:
            for index in range(inputs):
                matmul_into(own, source_second[:, index, :], working, scope=scope)
                np.add(result[:, index, :], working, out=result[:, index, :])
        finally:
            del working
    return result
