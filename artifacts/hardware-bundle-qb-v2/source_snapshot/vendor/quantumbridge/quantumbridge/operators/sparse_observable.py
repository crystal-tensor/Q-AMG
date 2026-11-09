# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Native qubit-sparse Pauli/projector sums with independently owned CSR data."""

from enum import IntEnum
from functools import wraps
from itertools import chain, product
from numbers import Integral, Number

import numpy as np

from quantumbridge.execution.array_ownership import retained_array
from quantumbridge.execution.array_reservations import array_layout, owned_arrays
from quantumbridge.execution.resource_context import ResourceScope, current_resources
from quantumbridge.execution.resource_ledger import UnifiedResourceLedger

from .pauli import Pauli, _multiply_bodies
from .pauli_list import PauliList
from .sparse_operator import SparsePauliOperator


_CODES = {"Z": 1, "X": 2, "Y": 3, "1": 5, "-": 6,
          "l": 7, "0": 9, "+": 10, "r": 11}
_LABELS = {code: label for label, code in _CODES.items()}
_FIELDS = {"coeffs": np.complex128, "bit_terms": np.uint8,
           "indices": np.uint32, "boundaries": np.uint64}
BitTerm = IntEnum("BitTerm", {"X": 2, "PLUS": 10, "+": 10,
                              "MINUS": 6, "-": 6, "Y": 3,
                              "RIGHT": 11, "r": 11, "LEFT": 7,
                              "l": 7, "Z": 1, "ZERO": 9, "0": 9,
                              "ONE": 5, "1": 5}, module=__name__)
BitTerm.label = property(lambda self: _LABELS[int(self)])


def _ledger():
    scope = current_resources()
    return UnifiedResourceLedger() if scope is None else scope.ledger


def _width(value):
    if not isinstance(value, Integral):
        raise TypeError("Observable num_qubits must be an integer.")
    if not 0 <= value <= np.iinfo(np.uint32).max:
        raise OverflowError("Observable num_qubits is outside uint32 range.")
    return int(value)


def _integer(value, maximum):
    if not isinstance(value, Integral):
        raise TypeError("Sparse indices and bit terms must be integers.")
    if not 0 <= value <= maximum:
        raise OverflowError("Sparse integer value is outside its storage range.")
    return int(value)


def _label_pairs(label):
    if not isinstance(label, str) or set(label) - set("IXYZ+-rl01"):
        raise ValueError("Observable labels must only contain IXYZ+-rl01.")
    return tuple((index, _CODES[char]) for index, char in enumerate(reversed(label)) if char != "I")


def _factor_product(first, second):
    """Multiply two sparse factors without expanding spectator projectors."""
    if first is None or second is None:
        return ((1, second if first is None else first),)
    a, b = first & 3, second & 3
    a_projector, b_projector = first > 3, second > 3
    sa, sb = (1 if first & 8 else -1), (1 if second & 8 else -1)
    if a == b:
        if a_projector and b_projector:
            return ((1, first),) if sa == sb else ()
        if a_projector:
            return ((sa, first),)
        if b_projector:
            return ((sb, second),)
        return ((1, None),)
    exponent, label = _multiply_bodies(_LABELS[a], _LABELS[b])
    phase, axis = (1, -1j, -1, 1j)[exponent], _CODES[label]
    if not a_projector and not b_projector:
        return ((phase, axis),)
    if not a_projector:
        return ((0.5, first), (0.5*sb*phase, axis))
    if not b_projector:
        return ((0.5, second), (0.5*sa*phase, axis))
    return ((0.5, second), (0.25*sa, a), (0.25*sa*sb*phase, axis))


def _sequence(value):
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.dtype.kind not in "biufc":
            raise TypeError("Sparse raw arrays must be one-dimensional numeric buffers.")
        return value
    if not isinstance(value, (list, tuple)):
        raise TypeError("Sparse raw fields require numeric arrays or builtin sequences.")
    return value


def _validate_parts(width, coeffs, bits, indices, boundaries):
    if len(bits) != len(indices) or len(boundaries) != len(coeffs) + 1:
        raise ValueError("Sparse coefficients, boundaries and factor arrays have inconsistent lengths.")
    if len(boundaries) == 0 or int(boundaries[0]) != 0 or int(boundaries[-1]) != len(bits):
        raise ValueError("Sparse boundaries must start at zero and end at the factor count.")
    for term in range(len(coeffs)):
        start, stop = int(boundaries[term]), int(boundaries[term + 1])
        if not 0 <= start <= stop <= len(bits):
            raise ValueError("Sparse boundaries must be monotone and in range.")
        previous = -1
        for index in range(start, stop):
            wire, code = int(indices[index]), int(bits[index])
            if not previous < wire < width:
                raise ValueError("Sparse indices must be term-wise increasing and within num_qubits.")
            if code not in _LABELS:
                raise ValueError("Invalid sparse single-qubit bit term.")
            previous = wire


class _ArrayView:
    """A mutable field handle whose sliced/array exports are owned copies."""

    def __init__(self, owner, name):
        self._owner, self._name = owner, name

    @property
    def _array(self):
        return getattr(self._owner, "_" + self._name)

    def __len__(self):
        return len(self._array)

    def __iter__(self):
        return (value.item() for value in self._array)

    def __getitem__(self, index):
        if not isinstance(index, (Integral, slice)):
            raise TypeError("Sparse field indexing requires an integer or slice.")
        value = self._array[index]
        if np.isscalar(value):
            return value.item()
        if not isinstance(index, slice):
            raise TypeError("Sparse field indexing requires an integer or slice.")
        output = retained_array(_active_or_owner(self._owner), "sparse_observable_export", value.shape, value.dtype)
        np.copyto(output, value)
        return output

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("Sparse raw field exports require an owned copy.")
        dtype = self._array.dtype if dtype is None else np.dtype(dtype)
        output = retained_array(_active_or_owner(self._owner), "sparse_observable_array_export", self._array.shape, dtype)
        np.copyto(output, self._array, casting="unsafe")
        return output

    def __setitem__(self, index, value):
        if not isinstance(index, (Integral, slice)):
            raise TypeError("Sparse field assignment requires an integer or slice.")
        candidate = self[:]
        candidate[index] = value
        fields = {name: getattr(self._owner, "_" + name) for name in _FIELDS}
        fields[self._name] = candidate
        _validate_parts(self._owner.num_qubits, **_part_arguments(fields))
        setattr(self._owner, "_" + self._name, candidate)


def _part_arguments(fields):
    return dict(coeffs=fields["coeffs"], bits=fields["bit_terms"],
                indices=fields["indices"], boundaries=fields["boundaries"])


def _active_or_owner(owner):
    scope = current_resources()
    return owner._ledger if scope is None else scope.ledger


def _owned_operation(function):
    @wraps(function)
    def scoped(self, *args, **kwargs):
        with ResourceScope(_active_or_owner(self)):
            return function(self, *args, **kwargs)
    return scoped


class Term:
    def __init__(self, num_qubits, coeff, bit_terms, indices):
        self._num_qubits, self._coeff = _width(num_qubits), complex(coeff)
        bits, wires = _sequence(bit_terms), _sequence(indices)
        for code in bits:
            if not isinstance(code, Integral):
                raise TypeError("Sparse bit terms must be integer alphabet values.")
            if int(code) not in _LABELS:
                raise ValueError("Invalid sparse single-qubit bit term.")
        for wire in wires:
            _integer(wire, np.iinfo(np.uint32).max)
        _validate_parts(self.num_qubits, [self.coeff], bits, wires, [0, len(bits)])
        ledger = self._ledger = _ledger()
        with ResourceScope(ledger):
            self._bits, self._indices = owned_arrays(
                [((len(bits),), np.uint8), ((len(wires),), np.uint32)], category="sparse_observable_term"
            )
        self._bits[:] = bits
        self._indices[:] = wires
        self._bits.flags.writeable = self._indices.flags.writeable = False

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def coeff(self):
        return self._coeff

    @property
    def bit_terms(self):
        return self._bits.view()

    @property
    def indices(self):
        return self._indices.view()

    def bit_labels(self):
        return "".join(_LABELS[int(code)] for code in self._bits)

    @_owned_operation
    def copy(self):
        return type(self)(self.num_qubits, self.coeff, self._bits, self._indices)

    @_owned_operation
    def to_observable(self):
        return SparseObservable.from_terms([self])

    @_owned_operation
    def pauli_base(self):
        return self.to_observable().pauli_bases()[0]


class SparseObservable:
    """Complex sums of sparse Pauli factors and Pauli-eigenstate projectors.

    Raw factor indices are little-endian physical qubit indices. Term and
    raw-field exports own their storage; no upstream SDK object executes here.
    """

    BitTerm = BitTerm
    Term = Term

    def __init__(self, data, /, num_qubits=None):
        if isinstance(data, SparseObservable):
            converted = data.copy()
        elif isinstance(data, str):
            converted = type(self).from_label(data)
        elif isinstance(data, Pauli):
            converted = type(self).from_pauli(data)
        elif isinstance(data, SparsePauliOperator):
            converted = type(self).from_sparse_pauli_op(data)
        else:
            iterator = iter(data)
            first = next(iterator, None)
            items = () if first is None else chain((first,), iterator)
            if isinstance(first, Term):
                converted = type(self).from_terms(items, num_qubits=num_qubits)
            elif first is not None and len(first) == 3:
                converted = type(self).from_sparse_list(items, num_qubits)
            else:
                converted = type(self).from_list(items, num_qubits=num_qubits)
        if num_qubits is not None and converted.num_qubits != _width(num_qubits):
            raise ValueError("Observable num_qubits does not match its data.")
        self.__dict__.update(converted.__dict__)

    @classmethod
    def _from_rows(cls, width, rows):
        width, ledger, records, factors = _width(width), _ledger(), [], 0
        for coefficient, pairs in rows:
            if not isinstance(coefficient, Number):
                raise TypeError("Sparse coefficients must be numeric.")
            pairs = tuple(pairs)
            factors += len(pairs)
            count = len(records) + 1
            ledger.preflight(memory_bytes=16*count + 5*factors + 8*(count+1))
            if count + factors > ledger.limits.max_derivative_work:
                raise MemoryError("Sparse construction exceeds its bounded work limit.")
            records.append((complex(coefficient), pairs))
        coeffs, bits, wires, boundaries = [], [], [], [0]
        for coefficient, pairs in records:
            coeffs.append(coefficient)
            for wire, code in pairs:
                wires.append(wire)
                bits.append(code)
            boundaries.append(len(bits))
        with ResourceScope(ledger):
            return cls.from_raw_parts(width, coeffs, bits, wires, boundaries)

    @classmethod
    def from_raw_parts(cls, num_qubits, coeffs, bit_terms, indices, boundaries, check=True):
        width = _width(num_qubits)
        values = {"coeffs": _sequence(coeffs), "bit_terms": _sequence(bit_terms),
                  "indices": _sequence(indices), "boundaries": _sequence(boundaries)}
        for name in ("bit_terms", "indices", "boundaries"):
            maximum = np.iinfo(_FIELDS[name]).max
            for value in values[name]:
                _integer(value, maximum)
        if check:
            _validate_parts(width, **_part_arguments(values))
        ledger = _ledger()
        with ResourceScope(ledger):
            buffers = owned_arrays([((len(value),), _FIELDS[name]) for name, value in values.items()], category="sparse_observable_storage")
        result = object.__new__(cls)
        result._ledger, result._num_qubits = ledger, width
        for (name, value), output in zip(values.items(), buffers):
            output[:] = value
            setattr(result, "_" + name, output)
        ledger.unavailable("sparse_observable_python_workspace", "The four numeric buffers are reserved; Python row metadata, iterators and allocator RSS are not measured.")
        return result

    @classmethod
    def from_label(cls, label, /):
        return cls._from_rows(len(label), [(1, _label_pairs(label))])

    @classmethod
    def from_list(cls, items, /, *, num_qubits=None):
        rows, width = [], None if num_qubits is None else _width(num_qubits)
        ledger, factors = _ledger(), 0
        for label, coefficient in items:
            pairs = _label_pairs(label)
            if width is None:
                width = len(label)
            if width != len(label):
                raise ValueError("All sparse labels must match num_qubits.")
            factors += len(pairs)
            ledger.preflight(memory_bytes=24*(len(rows)+1) + 5*factors + 8)
            rows.append((coefficient, pairs))
        if width is None:
            raise ValueError("Cannot infer num_qubits from an empty list.")
        return cls._from_rows(width, rows)

    @classmethod
    def from_sparse_list(cls, items, /, num_qubits):
        width, rows, ledger, factors = _width(num_qubits), [], _ledger(), 0
        for label, indices, coefficient in items:
            if not isinstance(label, str) or set(label) - set("IXYZ+-rl01"):
                raise ValueError("Invalid sparse bit labels.")
            wires = tuple(_integer(value, np.iinfo(np.uint32).max) for value in indices)
            if len(wires) != len(label) or len(set(wires)) != len(wires) or any(wire >= width for wire in wires):
                raise ValueError("Sparse labels require distinct matching qubit indices.")
            pairs = tuple(sorted((wire, _CODES[char]) for wire, char in zip(wires, label) if char != "I"))
            factors += len(pairs)
            ledger.preflight(memory_bytes=24*(len(rows)+1) + 5*factors + 8)
            rows.append((coefficient, pairs))
        return cls._from_rows(width, rows)

    @classmethod
    def from_pauli(cls, pauli, /):
        if not isinstance(pauli, Pauli):
            raise TypeError("from_pauli requires a Native Pauli.")
        return cls._from_rows(pauli.num_qubits, [((1, -1j, -1, 1j)[pauli.phase], _label_pairs(pauli._body))])

    @classmethod
    def from_sparse_pauli_op(cls, op, /):
        if not isinstance(op, SparsePauliOperator):
            raise TypeError("Expected the existing Native SparsePauliOperator.")
        # Existing native dense labels are q0-first; this class explicitly
        # stores rightmost-q0 labels instead of renaming that legacy operator.
        return cls.from_list([(label[::-1], coefficient) for label, coefficient in op.terms], num_qubits=op.num_qubits)

    @classmethod
    def from_terms(cls, terms, /, num_qubits=None):
        iterator = iter(terms)
        first = next(iterator, None)
        width = None if num_qubits is None else _width(num_qubits)
        if first is not None and not isinstance(first, Term):
            raise TypeError("from_terms requires Native SparseObservable.Term values.")
        if width is None and first is not None:
            width = first.num_qubits
        if width is None:
            raise ValueError("Cannot infer num_qubits from empty terms.")

        def rows():
            for term in (() if first is None else chain((first,), iterator)):
                if not isinstance(term, Term):
                    raise TypeError("from_terms requires Native SparseObservable.Term values.")
                if width != term.num_qubits:
                    raise ValueError("Term widths differ.")
                yield term.coeff, zip(term._indices, term._bits)

        return cls._from_rows(width, rows())

    @classmethod
    def zero(cls, num_qubits):
        return cls._from_rows(num_qubits, ())

    @classmethod
    def identity(cls, num_qubits):
        return cls._from_rows(num_qubits, [(1, ())])

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def num_terms(self):
        return len(self._coeffs)

    @property
    def coeffs(self):
        return _ArrayView(self, "coeffs")

    @property
    def bit_terms(self):
        return _ArrayView(self, "bit_terms")

    @property
    def indices(self):
        return _ArrayView(self, "indices")

    @property
    def boundaries(self):
        return _ArrayView(self, "boundaries")

    def validate(self):
        _validate_parts(self.num_qubits, self._coeffs, self._bit_terms, self._indices, self._boundaries)

    def _rows(self):
        self.validate()
        for index, coefficient in enumerate(self._coeffs):
            start, stop = int(self._boundaries[index]), int(self._boundaries[index+1])
            yield coefficient, tuple((int(self._indices[i]), int(self._bit_terms[i])) for i in range(start, stop))

    def to_sparse_list(self):
        return [("".join(_LABELS[code] for _, code in pairs), [wire for wire, _ in pairs], complex(coefficient)) for coefficient, pairs in self._rows()]

    def _label_dict(self):
        self.validate()
        _active_or_owner(self).preflight(memory_bytes=self.num_terms*(4*self.num_qubits+32))
        result = {}
        for coefficient, pairs in self._rows():
            label = ["I"]*self.num_qubits
            for wire, code in pairs:
                label[self.num_qubits-1-wire] = _LABELS[code]
            label = "".join(label)
            result[label] = result.get(label, 0j) + coefficient
        return result

    def __len__(self):
        return self.num_terms

    def __iter__(self):
        return (self[index] for index in range(len(self)))

    @_owned_operation
    def __getitem__(self, index):
        if isinstance(index, Integral):
            index = int(index)
            if index < 0:
                index += self.num_terms
            if not 0 <= index < self.num_terms:
                raise IndexError("Sparse term index out of range.")
            start, stop = int(self._boundaries[index]), int(self._boundaries[index+1])
            return self.Term(self.num_qubits, self._coeffs[index], self._bit_terms[start:stop], self._indices[start:stop])
        if not isinstance(index, slice):
            raise TypeError("Sparse term selection requires an integer or slice.")
        selected = range(self.num_terms)[index]
        return type(self)._from_rows(self.num_qubits, ((self._coeffs[i], tuple(zip(self._indices[int(self._boundaries[i]):int(self._boundaries[i+1])], self._bit_terms[int(self._boundaries[i]):int(self._boundaries[i+1])]))) for i in selected))

    @_owned_operation
    def copy(self):
        return type(self).from_raw_parts(self.num_qubits, self._coeffs, self._bit_terms, self._indices, self._boundaries)

    @_owned_operation
    def clear(self):
        self.__dict__.update(type(self).zero(self.num_qubits).__dict__)

    @_owned_operation
    def simplify(self, tol=1e-8):
        combined = {}
        for coefficient, pairs in self._rows():
            combined[pairs] = combined.get(pairs, 0j) + coefficient
        return type(self)._from_rows(self.num_qubits, ((coefficient, pairs) for pairs, coefficient in sorted(combined.items()) if not abs(coefficient) <= tol))

    def as_paulis(self):
        rows, count, factors = list(self._rows()), 0, 0
        ledger = _active_or_owner(self)
        for _, pairs in rows:
            expanded = 1 << sum(code > 3 for _, code in pairs)
            count += expanded
            factors += expanded*len(pairs)
        ledger.preflight(memory_bytes=24*count + 5*factors + 8, derivative_work=count)
        output = []
        for coefficient, pairs in rows:
            partial = [(coefficient, ())]
            for wire, code in pairs:
                if code <= 3:
                    partial = [(value, terms + ((wire, code),)) for value, terms in partial]
                else:
                    sign = 1 if code & 8 else -1
                    partial = [(value*factor, terms + suffix) for value, terms in partial
                               for factor, suffix in ((0.5, ()), (sign*0.5, ((wire, code & 3),)))]
            output.extend(partial)
        with ResourceScope(ledger):
            return type(self)._from_rows(self.num_qubits, output)

    @_owned_operation
    def apply_layout(self, layout, num_qubits=None):
        from quantumbridge.compiler.layout import Layout

        target = self.num_qubits if num_qubits is None else _width(num_qubits)
        if target < self.num_qubits:
            raise ValueError("Cannot shrink the qubit count in an observable.")
        if layout is None:
            mapping = range(self.num_qubits)
        elif isinstance(layout, Layout):
            if set(layout.mapping) != set(range(self.num_qubits)):
                raise ValueError("Layout must assign every logical qubit.")
            mapping = [layout.mapping[index] for index in range(self.num_qubits)]
        elif isinstance(layout, (list, tuple)):
            mapping = layout
        else:
            raise TypeError("Observable layout requires a Native Layout or explicit index list.")
        if len(mapping) != self.num_qubits:
            raise ValueError("Layout width must match the input observable.")
        if layout is not None:
            mapping = tuple(_integer(value, np.iinfo(np.uint32).max) for value in mapping)
            if len(set(mapping)) != len(mapping) or any(value >= target for value in mapping):
                raise ValueError("Layout indices must be distinct and within the output width.")
        return type(self)._from_rows(target, ((coefficient, tuple(sorted((mapping[wire], code) for wire, code in pairs))) for coefficient, pairs in self._rows()))

    @_owned_operation
    def __add__(self, other):
        if not isinstance(other, SparseObservable):
            other = type(self)(other)
        if self.num_qubits != other.num_qubits:
            raise ValueError("Sparse observable widths must match.")
        return type(self)._from_rows(self.num_qubits, chain(self._rows(), other._rows()))

    def __sub__(self, other):
        return self + (-type(self)(other))

    def __neg__(self):
        return self * -1

    @_owned_operation
    def __mul__(self, scalar):
        if not isinstance(scalar, Number):
            return NotImplemented
        return type(self)._from_rows(self.num_qubits, ((coefficient*scalar, pairs) for coefficient, pairs in self._rows()))

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        return self * (1/scalar)

    def _replace(self, result):
        self.__dict__.update(result.__dict__)
        return self

    def __iadd__(self, other):
        return self._replace(self + other)

    def __isub__(self, other):
        return self._replace(self - other)

    def __imul__(self, scalar):
        return self._replace(self * scalar)

    def __itruediv__(self, scalar):
        return self._replace(self / scalar)

    @_owned_operation
    def adjoint(self):
        return type(self)._from_rows(self.num_qubits, ((coefficient.conjugate(), pairs) for coefficient, pairs in self._rows()))

    @_owned_operation
    def transpose(self):
        def rows():
            for coefficient, pairs in self._rows():
                sign = -1 if sum(code == 3 for _, code in pairs) % 2 else 1
                yield coefficient*sign, tuple((wire, 7 if code == 11 else 11 if code == 7 else code) for wire, code in pairs)
        return type(self)._from_rows(self.num_qubits, rows())

    def conjugate(self):
        return self.transpose().adjoint()

    @_owned_operation
    def tensor(self, other, /):
        other = other if isinstance(other, SparseObservable) else type(self)(other)
        return type(self)._from_rows(self.num_qubits + other.num_qubits,
            ((a*b, right + tuple((wire+other.num_qubits, code) for wire, code in left))
             for a, left in self._rows() for b, right in other._rows()))

    __xor__ = tensor

    def expand(self, other, /):
        return type(self)(other).tensor(self)

    @_owned_operation
    def compose(self, other, /, qargs=None, *, front=False):
        other = other if isinstance(other, SparseObservable) else type(self)(other)
        if qargs is None:
            if self.num_qubits != other.num_qubits:
                raise ValueError("Composed observable widths must match.")
            mapping = range(self.num_qubits)
        else:
            mapping = tuple(_integer(value, np.iinfo(np.uint32).max) for value in qargs)
            if len(mapping) != other.num_qubits or len(set(mapping)) != len(mapping) or any(value >= self.num_qubits for value in mapping):
                raise ValueError("Invalid sparse compose qargs.")
        ledger = _active_or_owner(self)
        ledger.preflight(derivative_work=self.num_terms*other.num_terms)

        def choices(first, second):
            first, mapped = dict(first), {mapping[wire]: code for wire, code in second}
            wires = sorted(first.keys() | mapped.keys())
            factors = [_factor_product(first.get(wire), mapped.get(wire)) if front else
                       _factor_product(mapped.get(wire), first.get(wire)) for wire in wires]
            return wires, factors

        count, factors_count = 0, 0
        for _, first in self._rows():
            for _, second in other._rows():
                wires, factors = choices(first, second)
                branches = 1
                for factor in factors:
                    branches *= len(factor)
                count += branches
                factors_count += branches*len(wires)
                ledger.preflight(memory_bytes=24*count + 5*factors_count + 8,
                                 derivative_work=count + factors_count)

        def rows():
            for a, first in self._rows():
                for b, second in other._rows():
                    wires, factors = choices(first, second)
                    for branch in product(*factors):
                        coefficient, pairs = a*b, []
                        for wire, (phase, code) in zip(wires, branch):
                            coefficient *= phase
                            if code is not None:
                                pairs.append((wire, code))
                        yield coefficient, tuple(pairs)
        return type(self)._from_rows(self.num_qubits, rows())

    __and__ = compose

    def commutes(self, other, tol=1e-12):
        return (self.compose(other) - self.compose(other, front=True)).simplify(tol).num_terms == 0

    def evolve(self, other, /, qargs=None):
        if not isinstance(other, Pauli):
            raise TypeError("SparseObservable.evolve requires a Native Pauli.")
        pauli = type(self).from_pauli(other)
        return self.compose(pauli, qargs=qargs, front=True).compose(pauli.adjoint(), qargs=qargs)

    @_owned_operation
    def pauli_bases(self):
        _active_or_owner(self).preflight(memory_bytes=self.num_terms*4*self.num_qubits)
        labels = []
        for _, pairs in self._rows():
            label = ["I"]*self.num_qubits
            for wire, code in pairs:
                label[self.num_qubits-1-wire] = _LABELS[code & 3]
            labels.append("".join(label))
        return PauliList._from_paulis(labels, self.num_qubits)

    def to_matrix(self):
        from quantumbridge.utils.math import _sum_pauli_dense

        if self.num_qubits >= 31:
            raise MemoryError("Dense observable dimensions exceed checked array bounds.")
        ledger = _active_or_owner(self)
        ledger.preflight(memory_bytes=array_layout((1 << self.num_qubits,)*2, complex)[2])
        paulis = self.as_paulis()
        with ResourceScope(ledger):
            return _sum_pauli_dense(self.num_qubits, ((label, coefficient) for label, coefficient in paulis._label_dict().items()))

    matrix = to_matrix

    def __eq__(self, other):
        return isinstance(other, SparseObservable) and self.num_qubits == other.num_qubits and all(
            np.array_equal(getattr(self, "_" + name), getattr(other, "_" + name)) for name in _FIELDS)

    def __repr__(self):
        return f"SparseObservable(num_qubits={self.num_qubits}, num_terms={self.num_terms})"
