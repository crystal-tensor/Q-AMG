# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Shaped native Pauli sums at the public little-endian observable boundary."""

from collections.abc import Mapping
from math import prod
from numbers import Integral, Number

import numpy as np

from quantumbridge.execution.array_ownership import native_storage_bytes, retain_reservation, storage_ledger
from quantumbridge.execution.array_reservations import array_layout
from quantumbridge.execution.resource_context import ResourceScope, current_resources
from quantumbridge.execution.resource_ledger import UnifiedResourceLedger
from quantumbridge.operators import Pauli, SparseObservable, SparsePauliOperator
from quantumbridge.utils.math import Hamiltonian, PauliString

from .bindings_array import _shape_tuple


def _observable_shape(value, depth=0, ancestors=None):
    if depth > 32:
        raise ValueError("Observable nesting exceeds the construction limit.")
    if isinstance(value, np.ndarray):
        return value.shape
    if not isinstance(value, (list, tuple)):
        return ()
    ancestors = set() if ancestors is None else ancestors
    if id(value) in ancestors:
        raise ValueError("Observable arrays cannot be cyclic.")
    ancestors.add(id(value))
    try:
        shapes = [_observable_shape(child, depth + 1, ancestors) for child in value]
        if shapes and any(shape != shapes[0] for shape in shapes):
            raise ValueError("Observable arrays must be rectangular.")
        return (len(value),) + (shapes[0] if shapes else ())
    finally:
        ancestors.remove(id(value))


def _observable_leaves(value):
    if isinstance(value, np.ndarray):
        yield from value.flat
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _observable_leaves(child)
    else:
        yield value


def _pauli_string_label(value, width):
    minimum = max((wire for wire, _ in value.terms), default=-1) + 1
    width = minimum if width is None else width
    if minimum > width or any(wire < 0 for wire, _ in value.terms):
        raise ValueError("Observable wires exceed the circuit width.")
    label = ["I"] * width
    for wire, char in value.terms:
        label[width - 1 - wire] = char
    return "".join(label)


def _terms(value, width=None):
    if isinstance(value, SparseObservable):
        yield from value._label_dict().items()
    elif isinstance(value, str):
        if set(value) - set("IXYZ+-rl01"):
            raise ValueError("Observable labels must use IXYZ+-rl01.")
        yield value, 1
    elif isinstance(value, Pauli):
        pauli = Pauli(value)
        yield pauli._body, (1, -1j, -1, 1j)[pauli.phase]
    elif isinstance(value, SparsePauliOperator):
        for label, coefficient in value.terms:
            yield label[::-1], coefficient
    elif isinstance(value, PauliString):
        yield _pauli_string_label(value, width), 1
    elif isinstance(value, Hamiltonian):
        if width is None:
            width = max((wire for _, term in value.terms for wire, _ in term.terms), default=-1) + 1
        for coefficient, term in value.terms:
            yield _pauli_string_label(term, width), coefficient
    elif isinstance(value, Mapping):
        for label, coefficient in value.items():
            if not isinstance(label, (str, Pauli)) or not isinstance(coefficient, Number):
                raise TypeError("Pauli-sum coefficients must be numeric and keys must be Pauli labels.")
            if isinstance(label, str):
                if set(label) - set("IXYZ+-rl01"):
                    raise ValueError("Observable labels must use IXYZ+-rl01.")
                yield label, coefficient
            else:
                yield label._body, coefficient * (1, -1j, -1, 1j)[label.phase]
    else:
        raise TypeError("Native observables require Pauli labels, mappings or native Pauli operators.")


def _object_array(shape, ledger):
    # Object slots hold real Native SparseObservable instances, not opaque
    # foreign objects. Admit pointer storage with checked numeric-size math.
    size = array_layout(shape, np.uintp)[2]
    if np.dtype(object).itemsize != np.dtype(np.uintp).itemsize:
        raise RuntimeError("Object-slot accounting requires native pointer width.")
    ticket = ledger.reserve("primitive_observable_object_slots", memory_bytes=size)
    try:
        value = np.empty(shape, dtype=object)
        ticket.observe(value.nbytes)
        retain_reservation(ticket, value)
        return value
    except BaseException:
        ticket.release()
        raise


def _coerce_sparse(value, width=None):
    if isinstance(value, SparseObservable):
        result = value.simplify()
    else:
        result = SparseObservable.from_list(list(_terms(value, width))).simplify()
    if result.num_terms == 0:
        raise ValueError("Empty observable was detected.")
    if any(coefficient.imag != 0 for coefficient in result._coeffs):
        raise ValueError("Non-Hermitian observable has imaginary coefficients.")
    return result


class ObservablesArray:
    """An owned or explicitly borrowed ND array of Native sparse observables."""

    def __init__(self, observables, num_qubits=None, copy=True, validate=True):
        if num_qubits is not None and (isinstance(num_qubits, bool) or not isinstance(num_qubits, Integral) or num_qubits < 0):
            raise ValueError("Observable width must be a nonnegative integer.")
        if isinstance(observables, ObservablesArray):
            num_qubits = observables.num_qubits if num_qubits is None else num_qubits
            observables = observables._array
        shape = _shape_tuple(_observable_shape(observables))
        scope = current_resources()
        self._ledger = UnifiedResourceLedger() if scope is None else scope.ledger
        if prod(shape) > self._ledger.limits.max_total_tapes:
            raise MemoryError("Observable array exceeds the construction point limit.")
        self._ledger.preflight(memory_bytes=array_layout(shape, np.uintp)[2])
        if not copy and isinstance(observables, np.ndarray) and observables.dtype == object:
            self._array = observables.view()
            if storage_ledger(observables) is None:
                ticket = self._ledger.reserve("primitive_observable_borrowed_slots", memory_bytes=native_storage_bytes(observables), _resident=True)
                retain_reservation(ticket, observables)
        else:
            self._array = _object_array(shape, self._ledger)
            for loc, value in zip(np.ndindex(shape), _observable_leaves(observables)):
                self._array[loc] = value
        inferred = None
        if validate:
            with ResourceScope(self._ledger):
                for loc in np.ndindex(shape):
                    value = _coerce_sparse(self._array[loc], num_qubits)
                    if inferred is not None and value.num_qubits != inferred:
                        raise ValueError("All observables must have the same number of qubits.")
                    inferred = value.num_qubits
                    self._array[loc] = value
        elif self._array.size and isinstance(self._array.flat[0], SparseObservable):
            inferred = self._array.flat[0].num_qubits
        self._num_qubits = (inferred or 0) if num_qubits is None else int(num_qubits)
        if inferred is not None and inferred != self._num_qubits:
            raise ValueError("Observable width does not match num_qubits.")
        self._ledger.unavailable("primitive_observable_python_objects", "Object slots and each sparse numeric buffer have leases; Python object headers, dictionary exports and allocator RSS are not measured.")

    @property
    def shape(self):
        return self._array.shape

    @property
    def ndim(self):
        return self._array.ndim

    @property
    def size(self):
        return self._array.size

    @property
    def num_qubits(self):
        return self._num_qubits

    def _active_ledger(self):
        scope = current_resources()
        return self._ledger if scope is None else scope.ledger

    def _at(self, loc):
        values = self._array[loc]._label_dict()
        return {label: float(coefficient.real) if coefficient.imag == 0 else complex(coefficient) for label, coefficient in values.items()}

    def validate(self):
        with ResourceScope(self._active_ledger()):
            for loc in np.ndindex(self.shape):
                value = self._array[loc]
                if not isinstance(value, SparseObservable):
                    value = self._array[loc] = _coerce_sparse(value)
                value.validate()
                if value.num_qubits != self.num_qubits or not value.num_terms:
                    raise ValueError("Invalid observable width or empty sum.")
                if any(coefficient.imag != 0 for coefficient in value._coeffs):
                    raise ValueError("Non-Hermitian observable has imaginary coefficients.")

    def validate_for_execution(self):
        self.validate()
        if any(not np.isfinite(coefficient) for value in self._array.flat for coefficient in value._coeffs):
            raise ValueError("Observable coefficients must be finite before execution.")

    @classmethod
    def coerce(cls, observables):
        return observables if isinstance(observables, cls) else cls(observables)

    @classmethod
    def _for_circuit(cls, observables, width):
        if isinstance(observables, cls):
            return observables
        shape = _shape_tuple(_observable_shape(observables))
        scope = current_resources()
        ledger = UnifiedResourceLedger() if scope is None else scope.ledger
        if prod(shape) > ledger.limits.max_total_tapes:
            raise MemoryError("Observable array exceeds the construction point limit.")
        native_wires = any(isinstance(value, (PauliString, Hamiltonian)) for value in _observable_leaves(observables))
        return cls(observables, num_qubits=width if native_wires else None)

    @classmethod
    def coerce_observable(cls, observable):
        return _coerce_sparse(observable)

    def _view(self, array):
        result = object.__new__(type(self))
        result._ledger, result._num_qubits = self._ledger, self.num_qubits
        result._array, result._owner = array, self
        return result

    def slice(self, args):
        items = args if isinstance(args, tuple) else (args,)
        if any(item is not None and item is not Ellipsis and not isinstance(item, (Integral, slice)) for item in items):
            raise TypeError("Observable views require basic integer/slice indices.")
        result = self._array[args]
        return result if isinstance(result, SparseObservable) else self._view(result)

    def __getitem__(self, index):
        result = self.slice(index)
        return self._at(index) if isinstance(result, SparseObservable) else result

    def __iter__(self):
        if not self.ndim:
            raise TypeError("Cannot iterate a zero-dimensional observable array.")
        return (self[index] for index in range(self.shape[0]))

    def reshape(self, *shape):
        shape = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else tuple(shape)
        if any(isinstance(n, bool) or not isinstance(n, Integral) or n < -1 for n in shape) or shape.count(-1) > 1:
            raise ValueError("Invalid observable reshape dimensions.")
        if -1 in shape:
            known = prod(n for n in shape if n != -1)
            if known == 0 or self.size % known:
                raise ValueError("Cannot infer the observable reshape dimension.")
            shape = tuple(self.size//known if n == -1 else n for n in shape)
        shape = _shape_tuple(shape)
        if prod(shape) != self.size:
            raise ValueError("Observable reshape must preserve the number of entries.")
        if self._array.flags.c_contiguous:
            return self._view(self._array.reshape(shape))
        array = _object_array(shape, self._active_ledger())
        for index, value in enumerate(self._array.flat):
            array.flat[index] = value
        return self._view(array)

    def ravel(self):
        return self.reshape(self.size)

    def tolist(self):
        def build(loc):
            if len(loc) == self.ndim:
                return self._at(loc)
            return [build(loc + (index,)) for index in range(self.shape[len(loc)])]
        return build(())

    def sparse_observables_array(self, copy=False):
        if not copy:
            return self._array
        ledger = self._active_ledger()
        with ResourceScope(ledger):
            result = _object_array(self.shape, ledger)
            for loc in np.ndindex(self.shape):
                result[loc] = self._array[loc].copy()
        return result

    def copy(self):
        with ResourceScope(self._active_ledger()):
            return type(self)(self.sparse_observables_array(copy=True), num_qubits=self.num_qubits, copy=False, validate=False)

    def apply_layout(self, layout, num_qubits=None):
        ledger = self._active_ledger()
        with ResourceScope(ledger):
            width = SparseObservable.zero(self.num_qubits).apply_layout(layout, num_qubits).num_qubits
            output = _object_array(self.shape, ledger)
            for loc in np.ndindex(self.shape):
                output[loc] = self._array[loc].apply_layout(layout, num_qubits)
            return type(self)(output, num_qubits=width, copy=False, validate=False)

    def equivalent(self, other, tol=1e-8):
        if not isinstance(other, ObservablesArray) or (self.shape, self.num_qubits) != (other.shape, other.num_qubits):
            return False
        return all((self._array[loc] - other._array[loc]).simplify(tol).num_terms == 0 for loc in np.ndindex(self.shape))
