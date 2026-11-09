# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Owned packed-register operations; all statistics derive from existing memory."""

from collections import Counter
from collections.abc import Mapping
from math import prod
from numbers import Integral

import numpy as np

from quantumbridge.execution.array_ownership import retained_array, storage_ledger
from quantumbridge.execution.resource_context import current_resources
from quantumbridge.execution.resource_ledger import UnifiedResourceLedger

from .bindings_array import _shape_tuple


def allocate(shape, dtype, source=None):
    scope = current_resources()
    ledger = (
        scope.ledger
        if scope is not None
        else storage_ledger(source)
        if source is not None
        else None
    )
    return retained_array(
        ledger or UnifiedResourceLedger(), "primitive_bit_array_helper", shape, dtype
    )


def _indices(indices, size):
    if isinstance(indices, Integral) and not isinstance(indices, bool):
        indices = (int(indices),)
    else:
        indices = tuple(indices)
    if any(
        isinstance(index, bool)
        or not isinstance(index, Integral)
        or not 0 <= index < size
        for index in indices
    ):
        raise IndexError(
            "BitArray indices must be nonnegative integers within the selected axis."
        )
    return tuple(int(index) for index in indices)


def _sample_value(value):
    if isinstance(value, str):
        radix = 16 if value.startswith("0x") else 2
        return int(value, radix), len(value.removeprefix("0b")) if radix == 2 else 0
    if isinstance(value, Integral):
        return int(value), 0
    raise TypeError("BitArray samples must be binary strings or integers.")


class BitArray:
    """Packed register memory: batch axes, shot axis, big-endian byte axis."""

    def __init__(self, array, num_bits):
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != np.uint8
            or array.ndim < 2
        ):
            raise TypeError(
                "Packed bit memory requires a uint8 ndarray with shot and byte axes."
            )
        if (
            isinstance(num_bits, bool)
            or not isinstance(num_bits, Integral)
            or num_bits < 0
        ):
            raise ValueError("Register width must be a nonnegative integer.")
        if array.shape[-1] != (int(num_bits) + 7) // 8:
            raise ValueError("Packed byte count does not match the register width.")
        self._array, self._num_bits = array, int(num_bits)

    @property
    def array(self):
        return self._array

    @property
    def num_bits(self):
        return self._num_bits

    @property
    def num_shots(self):
        return self.array.shape[-2]

    @property
    def shape(self):
        return self.array.shape[:-2]

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return prod(self.shape)

    def _rows(self, loc=None):
        values = self.array if loc is None else self.array[loc]
        for index in np.ndindex(values.shape[:-1]):
            yield values[index]

    def _integer(self, row):
        return int.from_bytes(row.tobytes(), "big") & ((1 << self.num_bits) - 1)

    def _write(self, row, value):
        for index in range(len(row)):
            row[-1 - index] = (value >> (8 * index)) & 255

    def get_bitstrings(self, loc=None):
        return [
            format(self._integer(row), "0" + str(self.num_bits) + "b")
            if self.num_bits
            else ""
            for row in self._rows(loc)
        ]

    def get_counts(self, loc=None):
        return dict(Counter(self.get_bitstrings(loc)))

    def get_int_counts(self, loc=None):
        return dict(Counter(self._integer(row) for row in self._rows(loc)))

    def __getitem__(self, index):
        index = index if isinstance(index, tuple) else (index,)
        if any(
            item is not None
            and item is not Ellipsis
            and not isinstance(item, (Integral, slice))
            or isinstance(item, (bool, np.bool_))
            for item in index
        ):
            raise TypeError(
                "Native packed-memory indexing requires basic integer/slice indices."
            )
        if sum(item is not None and item is not Ellipsis for item in index) > self.ndim:
            raise IndexError("Only BitArray batch axes can be indexed.")
        return type(self)(self.array[index + (slice(None), slice(None))], self.num_bits)

    @classmethod
    def from_samples(cls, samples, num_bits=None):
        limit = UnifiedResourceLedger().limits.max_total_shots
        values, width = [], 0
        for index, sample in enumerate(samples):
            if index >= limit:
                raise MemoryError(
                    "BitArray sample construction exceeds the sample limit."
                )
            value, leading = _sample_value(sample)
            if value < 0:
                raise ValueError("BitArray samples cannot be negative.")
            width = max(width, leading, value.bit_length())
            values.append(value)
        if not values:
            raise ValueError("BitArray samples cannot be empty.")
        width = max(1, width) if num_bits is None else num_bits
        if isinstance(width, bool) or not isinstance(width, Integral) or width < 0:
            raise ValueError("Invalid BitArray width.")
        if any(value.bit_length() > width for value in values):
            raise ValueError("A sample does not fit the requested bit width.")
        result = cls(allocate((len(values), (width + 7) // 8), np.uint8), width)
        for row, value in zip(result.array, values):
            result._write(row, value)
        return result

    @classmethod
    def from_counts(cls, counts, num_bits=None):
        scalar = isinstance(counts, Mapping)
        batches = (counts,) if scalar else tuple(counts)
        if not batches:
            raise ValueError("Counts cannot be empty.")
        records, width, shots = [], 0, None
        for batch in batches:
            current, total = [], 0
            for key, count in batch.items():
                value, leading = _sample_value(key)
                if (
                    value < 0
                    or isinstance(count, bool)
                    or not isinstance(count, Integral)
                    or count < 0
                ):
                    raise ValueError(
                        "Counts require nonnegative outcomes and integer multiplicities."
                    )
                total += int(count)
                width = max(width, leading, value.bit_length())
                current.append((value, int(count)))
            if total == 0 or shots is not None and total != shots:
                raise ValueError("Count batches require equal positive shot totals.")
            shots = total
            records.append(current)
        width = max(1, width) if num_bits is None else num_bits
        if not isinstance(width, Integral) or isinstance(width, bool) or width < 0:
            raise ValueError("Invalid BitArray width.")
        if any(value.bit_length() > width for record in records for value, _ in record):
            raise ValueError("A count outcome does not fit the bit width.")
        shape = (() if scalar else (len(batches),)) + (shots, (width + 7) // 8)
        result = cls(allocate(shape, np.uint8), width)
        arrays = (result.array,) if scalar else result.array
        for output, record in zip(arrays, records):
            position = 0
            for value, count in record:
                for index in range(position, position + count):
                    result._write(output[index], value)
                position += count
        return result

    @classmethod
    def from_bool_array(cls, array, order="big"):
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != np.bool_
            or array.ndim < 2
        ):
            raise TypeError(
                "Boolean sample data requires a bool ndarray with shot and bit axes."
            )
        if order not in {"big", "little"}:
            raise ValueError("Bit order must be big or little.")
        width = array.shape[-1]
        result = cls(
            allocate(array.shape[:-1] + ((width + 7) // 8,), np.uint8, array), width
        )
        for index in np.ndindex(array.shape[:-1]):
            value = 0
            for bit in range(width):
                value |= int(array[index + (bit,)]) << (
                    width - 1 - bit if order == "big" else bit
                )
            result._write(result.array[index], value)
        return result

    def to_bool_array(self, order="big"):
        if order not in {"big", "little"}:
            raise ValueError("Bit order must be big or little.")
        result = allocate(self.array.shape[:-1] + (self.num_bits,), bool, self.array)
        for index in np.ndindex(self.array.shape[:-1]):
            value = self._integer(self.array[index])
            for bit in range(self.num_bits):
                result[index + (bit,)] = bool(
                    (value >> (self.num_bits - 1 - bit if order == "big" else bit)) & 1
                )
        return result

    def bitcount(self):
        result = allocate(self.array.shape[:-1], np.uint64, self.array)
        for index in np.ndindex(result.shape):
            result[index] = bin(self._integer(self.array[index])).count("1")
        return result

    def slice_bits(self, indices):
        indices = _indices(indices, self.num_bits)
        result = type(self)(
            allocate(
                self.array.shape[:-1] + ((len(indices) + 7) // 8,), np.uint8, self.array
            ),
            len(indices),
        )
        for index in np.ndindex(self.array.shape[:-1]):
            value = self._integer(self.array[index])
            selected = sum(
                ((value >> bit) & 1) << local for local, bit in enumerate(indices)
            )
            result._write(result.array[index], selected)
        return result

    def slice_shots(self, indices):
        indices = _indices(indices, self.num_shots)
        result = allocate(
            self.shape + (len(indices), self.array.shape[-1]), np.uint8, self.array
        )
        np.take(self.array, indices, axis=-2, out=result)
        return type(self)(result, self.num_bits)

    def reshape(self, *shape):
        shape = (
            shape[0]
            if len(shape) == 1 and isinstance(shape[0], (tuple, list))
            else shape
        )
        shape = _shape_tuple(shape)
        if prod(shape) == self.size:
            target = shape + (self.num_shots, self.array.shape[-1])
        elif shape and prod(shape) == self.size * self.num_shots:
            target = shape + (self.array.shape[-1],)
        else:
            raise ValueError("Cannot change the size of the bit array.")
        if self.array.flags.c_contiguous or self.array.size == 0:
            return type(self)(self.array.reshape(target), self.num_bits)
        result = allocate(target, np.uint8, self.array)
        for index, value in enumerate(self.array.flat):
            result.flat[index] = value
        return type(self)(result, self.num_bits)

    def transpose(self, *axes):
        axes = (
            axes[0] if len(axes) == 1 and isinstance(axes[0], (tuple, list)) else axes
        )
        axes = tuple(reversed(range(self.ndim))) if not axes else tuple(axes)
        if sorted(axes) != list(range(self.ndim)):
            raise ValueError("Transpose requires each batch axis once.")
        return type(self)(
            self.array.transpose(axes + (self.ndim, self.ndim + 1)), self.num_bits
        )

    @staticmethod
    def concatenate(bit_arrays, axis=0):
        values = tuple(bit_arrays)
        if not values:
            raise ValueError("At least one bit array is required.")
        first = values[0]
        axis = int(axis)
        if axis < 0:
            axis += first.ndim
        if not 0 <= axis < first.ndim:
            raise ValueError("Concatenation axis must be a batch axis.")
        shape = list(first.array.shape)
        shape[axis] = 0
        for value in values:
            if (
                value.num_bits != first.num_bits
                or value.array.ndim != first.array.ndim
                or any(
                    a != b
                    for index, (a, b) in enumerate(
                        zip(value.array.shape, first.array.shape)
                    )
                    if index != axis
                )
            ):
                raise ValueError(
                    "Bit arrays have incompatible concatenation dimensions."
                )
            shape[axis] += value.array.shape[axis]
        output = allocate(tuple(shape), np.uint8, first.array)
        np.concatenate([value.array for value in values], axis=axis, out=output)
        return BitArray(output, first.num_bits)

    @staticmethod
    def concatenate_shots(bit_arrays):
        values = tuple(bit_arrays)
        if not values:
            raise ValueError("At least one bit array is required.")
        first = values[0]
        if any(
            value.num_bits != first.num_bits or value.shape != first.shape
            for value in values
        ):
            raise ValueError(
                "Shot concatenation requires matching batch shapes and register widths."
            )
        output = allocate(
            first.shape
            + (sum(value.num_shots for value in values), first.array.shape[-1]),
            np.uint8,
            first.array,
        )
        np.concatenate([value.array for value in values], axis=-2, out=output)
        return BitArray(output, first.num_bits)

    @staticmethod
    def concatenate_bits(bit_arrays):
        values = tuple(bit_arrays)
        if not values:
            raise ValueError("At least one bit array is required.")
        first = values[0]
        if any(
            value.shape != first.shape or value.num_shots != first.num_shots
            for value in values
        ):
            raise ValueError("Bit concatenation requires matching batch and shot axes.")
        width = sum(value.num_bits for value in values)
        result = BitArray(
            allocate(
                first.array.shape[:-1] + ((width + 7) // 8,), np.uint8, first.array
            ),
            width,
        )
        for index in np.ndindex(first.array.shape[:-1]):
            selected, offset = 0, 0
            for value in values:
                selected |= value._integer(value.array[index]) << offset
                offset += value.num_bits
            result._write(result.array[index], selected)
        return result

    def postselect(self, indices, selection):
        indices = _indices(indices, self.num_bits)
        selection = (
            (selection,)
            if isinstance(selection, (Integral, bool))
            else tuple(selection)
        )
        if len(selection) != len(indices) or any(
            value not in (0, 1, False, True) for value in selection
        ):
            raise ValueError(
                "Postselection requires one binary outcome for each selected bit."
            )

        def matches(row):
            value = self._integer(row)
            return all(
                ((value >> bit) & 1) == int(outcome)
                for bit, outcome in zip(indices, selection)
            )

        count = sum(matches(row) for row in self._rows())
        output = allocate((count, self.array.shape[-1]), np.uint8, self.array)
        index = 0
        for row in self._rows():
            if matches(row):
                output[index] = row
                index += 1
        return type(self)(output, self.num_bits)

    def expectation_values(self, observables):
        def unpack(value):
            if isinstance(value, (str, Mapping)):
                return (), [value]
            if isinstance(value, np.ndarray):
                return value.shape, list(value.flat)
            if isinstance(value, (list, tuple)):
                children = [unpack(child) for child in value]
                if not children or any(
                    child[0] != children[0][0] for child in children
                ):
                    raise ValueError(
                        "Observable arrays must be nonempty and rectangular."
                    )
                return (len(children),) + children[0][0], [
                    term for _, terms in children for term in terms
                ]
            raise ValueError(
                "Sample expectations require I/Z labels or mappings of such labels."
            )

        shape, descriptions = unpack(observables)
        terms = []
        for description in descriptions:
            description = (
                {description: 1.0} if isinstance(description, str) else description
            )
            current = []
            if not isinstance(description, Mapping):
                raise ValueError("Invalid diagonal observable.")
            for label, coefficient in description.items():
                if (
                    not isinstance(label, str)
                    or len(label) != self.num_bits
                    or set(label) - set("IZ")
                ):
                    raise ValueError(
                        "Sample expectation observables must contain exactly num_bits I/Z factors."
                    )
                coefficient = complex(coefficient)
                if coefficient.imag or not np.isfinite(coefficient):
                    raise ValueError("Observable coefficients must be finite and real.")
                mask = sum(
                    1 << bit
                    for bit, factor in enumerate(reversed(label))
                    if factor == "Z"
                )
                current.append((mask, coefficient.real))
            terms.append(current)
        output_shape = np.broadcast_shapes(self.shape, shape)
        result = allocate(output_shape, float, self.array)

        def location(index, dimensions):
            return (
                tuple(
                    0 if width == 1 else item
                    for item, width in zip(index[-len(dimensions) :], dimensions)
                )
                if dimensions
                else ()
            )

        for index in np.ndindex(output_shape):
            batch, observed = location(index, self.shape), location(index, shape)
            observable = terms[
                int(np.ravel_multi_index(observed, shape)) if shape else 0
            ]
            total = 0.0
            for row in self.array[batch]:
                value = self._integer(row)
                total += sum(
                    coefficient * (-1 if bin(value & mask).count("1") % 2 else 1)
                    for mask, coefficient in observable
                )
            result[index] = total / self.num_shots if self.num_shots else np.nan
        return result
