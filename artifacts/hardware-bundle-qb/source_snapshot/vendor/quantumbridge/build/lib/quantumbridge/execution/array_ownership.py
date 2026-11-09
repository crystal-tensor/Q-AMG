# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Bind an existing reservation to native storage, not callback duration."""

from threading import RLock
import weakref
from copy import deepcopy
from types import FunctionType, BuiltinFunctionType, MethodType, ModuleType
import sys

import numpy as np

from .array_reservations import ArrayReservation


_storage_owners, _storage_owner_lock = {}, RLock()


def _storage_root(array):
    while True:
        base = np.ndarray.base.__get__(array)
        if not issubclass(type(base), np.ndarray):return array
        array = base


def remember_storage_ledger(array, ledger):
    """Weak provenance only; this neither owns storage nor reserves a budget."""
    if ledger is None:return
    root = _storage_root(array)
    key = id(root)
    def forgotten(reference):
        with _storage_owner_lock:
            record = _storage_owners.get(key)
            if record is not None and record[0] is reference:del _storage_owners[key]
    with _storage_owner_lock:
        _storage_owners[key] = (weakref.ref(root, forgotten), weakref.ref(ledger))


def storage_ledger(array):
    root = _storage_root(array)
    with _storage_owner_lock:
        record = _storage_owners.get(id(root))
        return None if record is None or record[0]() is not root else record[1]()


def _roots(value):
    roots, pending, seen = {}, [value], set()
    while pending:
        item = pending.pop()
        if id(item) in seen: continue
        seen.add(id(item))
        if isinstance(item, np.ndarray):
            while isinstance(item.base, np.ndarray):
                item = item.base
            roots[id(item)] = item
        elif isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (tuple, list)):
            pending.extend(item)
    return tuple(roots.values())


def native_storage_bytes(value):
    """Observe the same distinct ndarray roots that a result lease retains."""
    return sum(root.nbytes for root in _roots(value))


def retain_reservation(reservation, value):
    """Transfer a pre-acquired, possibly conservative lease to distinct roots.

    Views keep their storage root alive and do not acquire another lease. This
    tracks native ndarray storage only; a framework must copy a returned buffer
    or explicitly own its lifetime before dropping the ndarray reference.
    """
    roots = _roots(value)
    if not roots:
        reservation.release()
        return
    remaining, lock = len(roots), RLock()
    ticket = reservation.ticket if isinstance(reservation, ArrayReservation) else reservation
    ledger = None if ticket is None else ticket.ledger
    def released():
        nonlocal remaining
        with lock:
            remaining -= 1
            if remaining == 0: reservation.release()
    for root in roots:
        remember_storage_ledger(root, ledger)
        weakref.finalize(root, released)


def retained_array(ledger, category, shape, dtype, factory=None):
    """Acquire before allocation and release when the new buffer's views die."""
    reservation = ArrayReservation(ledger, category, shape, dtype)
    result = reservation.create(factory)
    if not result.flags.owndata:
        reservation.release()
        raise RuntimeError("Retained array factories must create new owned storage, not borrow a view.")
    retain_reservation(reservation, result)
    return result


def copied_array(value, *, category="native_defensive_copy"):
    """A real copy follows the active request, otherwise its actual source owner."""
    from .array_reservations import numeric_shape
    from .resource_context import current_resources
    numeric_shape(value)
    if not issubclass(type(value), np.ndarray):
        raise TypeError("Native defensive copies require an explicit numeric ndarray.")
    value = np.ndarray.view(value, np.ndarray)
    scope = current_resources()
    ledger = storage_ledger(value) if scope is None else scope.ledger
    result = retained_array(ledger, category, value.shape, value.dtype)
    if ledger is not None:
        ledger.unavailable("native_copy_iterator_workspace",
            "The copy destination has a storage lease; NumPy iterator and Python allocator workspace is not measured RSS.")
    np.copyto(result, value, casting="no")
    return result


def select_columns(array, indices):
    """Basic slices borrow; controlled advanced columns own one prior lease."""
    from numbers import Integral
    from .resource_context import current_resources
    from quantumbridge.core.classical import UnsupportedFeature
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError("Native column selection requires a matrix.")
    array = np.ndarray.view(array, np.ndarray)
    if type(indices) is slice:
        return array[:, indices]
    if type(indices) not in (tuple, list):
        raise UnsupportedFeature("native_column_index", "Native mapped columns require a slice or an explicit integer index sequence.")
    selected = []
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise ValueError("Native selected columns require integer indices, not masks.")
        index = int(index)
        if not -array.shape[1] <= index < array.shape[1]:
            raise IndexError("Native selected column is outside the matrix.")
        selected.append(index)
    scope = current_resources()
    output = retained_array(None if scope is None else scope.ledger,
        "native_column_selection", (array.shape[0], len(selected)), array.dtype)
    for column, index in enumerate(selected):
        np.copyto(output[:, column], array[:, index], casting="no")
    return output


def _exact_type(kind, choices):
    return any(kind is candidate for candidate in choices)


def _defined_type(kind, prefix):
    # Read type/module layout descriptors directly: opaque metaclasses may
    # override attribute access and even equality on class objects.
    module_name = type.__dict__["__module__"].__get__(kind)
    if type(module_name) is not str: return False
    package_prefix = prefix if prefix.endswith(".") else prefix + "."
    if module_name != prefix and not module_name.startswith(package_prefix): return False
    module = sys.modules.get(module_name)
    if type(module) is not ModuleType: return False
    fields = ModuleType.__dict__["__dict__"].__get__(module)
    name = type.__dict__["__name__"].__get__(kind)
    return fields.get(name) is kind


def _immutable_callback_data(value, seen=None):
    """Borrow code references only when their explicit captures have no storage.

    This does not snapshot a function's module globals or certify allocations
    made when a transform later executes. It only bounds construction copying.
    """
    seen = set() if seen is None else seen
    if id(value) in seen: return True
    seen.add(id(value))
    kind = type(value)
    if _exact_type(kind, (type(None), bool, int, float, complex, str, bytes, type)): return True
    if _exact_type(kind, (tuple, frozenset)):
        return all(_immutable_callback_data(item, seen) for item in value)
    if kind is FunctionType:
        try:
            captured = [cell.cell_contents for cell in value.__closure__ or ()]
        except ValueError:
            return False
        captured.extend(value.__defaults__ or ())
        captured.extend((value.__kwdefaults__ or {}).values())
        captured.extend(value.__dict__.values())
        captured.extend(value.__annotations__.values())
        return all(_immutable_callback_data(item, seen) for item in captured)
    if kind is BuiltinFunctionType:
        return type(value.__self__) is ModuleType or _immutable_callback_data(value.__self__, seen)
    if _defined_type(kind, "numpy"):
        mro = type.__dict__["__mro__"].__get__(kind)
        return any(base is np.generic for base in mro) and value.dtype.kind in "biufc"
    return False


def _strict_configuration_copy(value, memo, ledger, category, on_array, *, preflight_group=False):
    from quantumbridge.core.classical import UnsupportedFeature
    from quantumbridge.core.operations import Operation
    from quantumbridge.core.unitary_inverse import inverse_matrix_owner,seal_inverse_copy
    pending, seen, arrays, native = [value], set(memo), [], []
    inverse_operations=[]
    atomic = (type(None), bool, int, float, complex, str, bytes, type, range)
    while pending:
        item = pending.pop()
        if id(item) in seen: continue
        seen.add(id(item))
        kind = type(item)
        if _exact_type(kind, atomic): continue
        if kind is np.ndarray:
            arrays.append(item)
        elif kind is dict:
            pending.extend(item.keys())
            pending.extend(item.values())
        elif _exact_type(kind, (tuple, list, set, frozenset)):
            pending.extend(item)
        elif kind is slice:
            pending.extend((item.start, item.stop, item.step))
        elif _exact_type(kind, (FunctionType, BuiltinFunctionType)):
            if not _immutable_callback_data(item):
                raise UnsupportedFeature("persistent_callable_capture", "Persistent callbacks require immutable explicit captures; opaque captured storage is not copied or executed.")
            if ledger is not None:
                ledger.unavailable("callable_module_state",
                    "Executable code is borrowed; module globals and allocations made by later callback execution are not a copied immutable data snapshot or a process-memory bound.")
        elif kind is MethodType:
            pending.extend((item.__func__, item.__self__))
        elif _defined_type(kind, "numpy") and any(base is np.generic for base in type.__dict__["__mro__"].__get__(kind)):
            if item.dtype.kind not in "biufc":
                raise UnsupportedFeature("persistent_scalar_storage")
        elif _defined_type(kind, "numpy") and any(base is np.dtype for base in type.__dict__["__mro__"].__get__(kind)):
            if item.metadata is not None:
                raise UnsupportedFeature("persistent_dtype_metadata")
        elif _defined_type(kind, "quantumbridge."):
            # Native objects are reconstructed from their audited instance
            # dictionaries, never through reduce/deepcopy/constructors.
            try:
                fields = object.__getattribute__(item, "__dict__")
            except AttributeError:
                raise UnsupportedFeature("persistent_native_slots", "This native slotted configuration has no controlled copy contract.") from None
            if type(fields) is not dict:
                raise UnsupportedFeature("persistent_native_fields")
            if kind is Operation and ("_native_inverse_owner" in fields or "_native_inverse_qualification" in fields):
                inverse_matrix_owner(item)
                inverse_operations.append(item)
            native.append((item, fields))
            pending.append(fields)
        else:
            # No isinstance/hasattr/vars call on an opaque instance: even
            # looking up __class__ or __dict__ may execute foreign callbacks.
            raise UnsupportedFeature("persistent_foreign_copy", "Persistent storage cannot bound this foreign copy protocol.")
    if preflight_group:
        from .array_reservations import array_layout
        layouts=[array_layout(array.shape,array.dtype) for array in arrays]
        if ledger is not None:ledger.preflight(memory_bytes=sum(layout[2] for layout in layouts))
    for item, fields in native:
        memo[id(item)] = object.__new__(type(item))
    for array in arrays:
        copied = (copied_array(array,category=category) if ledger is None else
            retained_array(ledger, category, array.shape, array.dtype,
                lambda array=array: array.copy(order="K")))
        memo[id(array)] = copied
        if on_array is not None: on_array(copied)
    # Hashable native symbols can already be referenced by copied sets/dicts.
    # Prime their audited fields; replace every field with owned data below
    # before any reconstructed object escapes this function.
    for item, fields in native:
        object.__getattribute__(memo[id(item)], "__dict__").update(fields)
    for item, fields in native:
        object.__getattribute__(memo[id(item)], "__dict__").update(deepcopy(fields, memo))
    for item in inverse_operations:
        seal_inverse_copy(memo[id(item)])
    return deepcopy(value, memo)


def native_object_deepcopy(value, memo):
    """Native object copy hooks share the caller's memo, never foreign hooks."""
    from .resource_context import current_resources
    scope=current_resources()
    return _strict_configuration_copy(value,memo,None if scope is None else scope.ledger,
        "native_object_deepcopy",None,preflight_group=True)


def native_configuration_copy(value, *, memo=None, category="native_configuration_copy",
                              ledger=None, strict=False, on_array=None):
    """Pre-copy known native ndarray fields with the same alias memo as deepcopy.

    Shared object references get one destination. Distinct ndarray views get
    independent copies, matching deepcopy rather than assuming shared output
    storage from shared input addresses. Graph scalars already in memo remain
    borrowed. Foreign custom copy protocols are explicitly outside this audit.
    """
    from .resource_context import current_resources
    scope = current_resources()
    ledger = ledger if ledger is not None else None if scope is None else scope.ledger
    memo = {} if memo is None else dict(memo)
    if strict:
        if ledger is None:
            raise ValueError("Strict native configuration copying requires an explicit storage account.")
        return _strict_configuration_copy(value, memo, ledger, category, on_array)
    if ledger is None: return deepcopy(value, memo)
    pending, seen, arrays = [value], set(memo), []
    while pending:
        item = pending.pop()
        if id(item) in seen: continue
        seen.add(id(item))
        if isinstance(item, np.ndarray):
            arrays.append(item)
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (tuple, list, set, frozenset)):
            pending.extend(item)
        elif type(item).__module__.startswith("quantumbridge.") and hasattr(item, "__dict__"):
            pending.extend(vars(item).values())
        elif (hasattr(type(item), "__deepcopy__") or hasattr(item, "__dict__")) and not isinstance(item,
                (np.generic, FunctionType, BuiltinFunctionType, MethodType, type)):
            ledger.unavailable("foreign_configuration_copy_protocol",
                "Known native ndarray fields are controlled; a foreign object's custom copy allocator is not inspected or a process-memory bound.")
    for array in arrays:
        if type(array) is not np.ndarray:
            ledger.unavailable("ndarray_subclass_copy_workspace",
                "Subclass copy protocols retain their own behavior; their additional internal workspace is not an observed native-buffer bound.")
        copied = retained_array(ledger, category, array.shape, array.dtype,
            lambda array=array: array.copy(order="K") if type(array) is np.ndarray else deepcopy(array, memo))
        memo[id(array)] = copied
        if on_array is not None: on_array(copied)
    return deepcopy(value, memo)
