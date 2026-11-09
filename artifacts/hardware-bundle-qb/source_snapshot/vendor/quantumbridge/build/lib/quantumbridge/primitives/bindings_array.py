# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Shaped native primitive bindings, using the existing parameter identities."""

from collections.abc import Mapping
from math import prod
from numbers import Integral

import numpy as np

from quantumbridge.core import Circuit, Parameter
from quantumbridge.execution.array_ownership import retained_array, retain_reservation
from quantumbridge.execution.array_reservations import (
    array_layout, numeric_into, numeric_shape, owned_arrays,
)
from quantumbridge.execution.resource_context import current_resources, ResourceScope
from quantumbridge.execution.resource_ledger import UnifiedResourceLedger


def _shape_tuple(shape):
    if isinstance(shape, Integral) and not isinstance(shape, bool):
        shape = (shape,)
    if not isinstance(shape, (tuple, list)):
        raise ValueError("A primitive array shape must contain nonnegative integer dimensions.")
    return array_layout(shape, np.uint8)[0]


def _parameter_group(key):
    group = key if isinstance(key, tuple) else (key,)
    if any(not isinstance(item, (str, Parameter)) for item in group):
        raise TypeError("Binding keys must be native Parameters or parameter names.")
    names = tuple(item if isinstance(item, str) else item.name for item in group)
    if any(not name for name in names):
        raise ValueError("Binding parameter names cannot be empty.")
    return group, names


class BindingsArray:
    """An N-dimensional collection of native parameter binding sets.

    Numeric storage uses a final parameter axis. Parameter objects keep their
    native UUID identity; names are exposed in ``data`` for public inspection.
    Owned conversions and returned arrays use the active resource ledger.
    """

    def __init__(self, data=None, shape=None):
        resources = current_resources()
        self._ledger = UnifiedResourceLedger() if resources is None else resources.ledger
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise TypeError("BindingsArray requires a mapping from parameters to numeric arrays.")
        declared = None if shape is None else _shape_tuple(shape)
        records, owners, inferred = {}, {}, None
        for key, values in data.items():
            group, names = _parameter_group(key)
            raw_shape = numeric_shape(values, real_only=True)
            if declared is not None and len(names) == 1 and raw_shape == declared:
                target = declared+(1,)
            elif raw_shape and raw_shape[-1] == len(names):
                target = raw_shape
            elif len(names) == 1:
                target = raw_shape+(1,)
            else:
                raise ValueError("The last binding axis must match its parameter group.")
            batch = target[:-1]
            if declared is not None and batch != declared:
                raise ValueError("Binding values do not match the declared batch shape.")
            if inferred is not None and batch != inferred:
                raise ValueError("All parameter groups must have the same batch shape.")
            inferred = batch
            array_layout(target, np.float64)
            records[names] = (values, raw_shape, target)
            owners[names] = group
        self._shape = declared if declared is not None else inferred if inferred is not None else ()
        array_layout(self._shape+(sum(map(len, records)),), np.float64)
        self._owners = owners
        self._data = {}
        self._ledger.preflight(memory_bytes=sum(array_layout(target, np.float64)[2]
                                               for _, _, target in records.values()))
        # Acquire every conversion lease before allocating the first group.
        conversions = [(target, np.float64) for value, _, target in records.values()
                       if not isinstance(value, np.ndarray) or value.dtype != np.float64]
        with ResourceScope(self._ledger):
            outputs = iter(owned_arrays(conversions, category="primitive_binding_values"))
        for names, (values, raw_shape, target) in records.items():
            if isinstance(values, np.ndarray) and values.dtype == np.float64:
                array = np.ndarray.view(values, np.ndarray).reshape(target)
            else:
                array = next(outputs)
                numeric_into(array.reshape(raw_shape), values)
            self._data[names] = array
        self.validate()

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        return prod(self._shape)

    @property
    def num_parameters(self):
        return sum(map(len, self._data))

    def validate(self):
        for names, array in self._data.items():
            _, normalized = _parameter_group(names)
            if normalized != names:
                raise ValueError("Invalid parameter names in binding data.")
            if not isinstance(array, np.ndarray) or array.dtype.kind not in "biuf":
                raise TypeError("Binding values must be real numeric arrays.")
            if array.shape != self._shape+(len(names),):
                raise ValueError("Binding data changed its declared parameter or batch axes.")
        array_layout(self._shape+(self.num_parameters,), np.float64)

    @classmethod
    def coerce(cls, bindings_array):
        return bindings_array if isinstance(bindings_array, cls) else cls(bindings_array)

    def as_array(self, parameters=None):
        self.validate()
        if parameters is None:
            columns = [(values, column) for values in self._data.values()
                       for column in range(values.shape[-1])]
        else:
            _, order = _parameter_group(tuple(parameters))
            by_name = {name: (values, column) for names, values in self._data.items()
                       for column, name in enumerate(names)}
            if len(by_name) != self.num_parameters or len(set(order)) != len(order):
                raise ValueError("Explicit parameter reordering is ambiguous for duplicate names.")
            if len(order) != self.num_parameters or set(order) != set(by_name):
                raise ValueError("Requested parameter order must contain every parameter exactly once.")
            columns = [by_name[name] for name in order]
        resources = current_resources()
        result = retained_array(self._ledger if resources is None else resources.ledger,
                                "primitive_binding_array", self.shape+(len(columns),), np.float64)
        for index, (values, column) in enumerate(columns):
            np.copyto(result[..., index], values[..., column])
        return result

    def _validate_circuit(self, circuit):
        if not isinstance(circuit, Circuit):
            raise TypeError("Native primitive bindings require a native Circuit.")
        self.validate()
        all_names = [name for group in self.data for name in group]
        if len(set(all_names)) != len(all_names):
            raise ValueError("Execution cannot resolve duplicate or ambiguous binding descriptions.")
        parameters = {parameter.name: parameter for parameter in circuit.parameters}
        names = {name for group in self.data for name in group}
        if names != set(parameters):
            raise ValueError("Binding parameters do not match the circuit's free parameters.")
        for group in self.data:
            for name, owner in zip(group, self._owners.get(group, group)):
                if isinstance(owner, Parameter) and owner != parameters[name]:
                    raise ValueError("A binding Parameter has a different UUID from its circuit owner.")
        # Representation may contain NaN/Inf; quantum rotation binding may not.
        if any(not np.isfinite(value) for values in self.data.values() for value in values.flat):
            raise ValueError("Primitive parameter values must be finite before execution.")

    def bind(self, circuit, loc):
        self._validate_circuit(circuit)
        if not isinstance(loc, tuple) or len(loc) != self.ndim:
            raise ValueError("Binding coordinates must match the number of batch dimensions.")
        if any(isinstance(index, bool) or not isinstance(index, Integral) for index in loc):
            raise TypeError("Binding coordinates must be integer indices.")
        if any(not -width <= index < width for index, width in zip(loc, self.shape)):
            raise IndexError("Binding coordinates are outside the batch shape.")
        mapping = {owner: values[loc+(column,)] for names, values in self.data.items()
                   for column, owner in enumerate(self._owners.get(names, names))}
        return circuit.bind(mapping)

    def bind_all(self, circuit):
        self._validate_circuit(circuit)
        resources = current_resources()
        ledger = self._ledger if resources is None else resources.ledger
        # This bounds eager circuit materialization, not a claim about heap/RSS.
        if self.size > ledger.limits.max_total_tapes:
            from quantumbridge.execution.derivative_status import ResourceLimitError
            raise ResourceLimitError("binding_circuits", self.size, ledger.limits.max_total_tapes,
                                     stage="primitive_binding_materialization")
        nbytes = array_layout(self.shape, np.intp)[2]
        ticket = ledger.reserve("primitive_bound_circuit_array", memory_bytes=nbytes)
        try:
            result = np.empty(self.shape, dtype=object)
            retain_reservation(ticket, result)
        except BaseException:
            ticket.release()
            raise
        ledger.unavailable("primitive_bound_circuit_python_heap",
                           "Object-array pointer storage is reserved; circuit objects and Python allocator RSS are not measured.")
        with ResourceScope(ledger):
            for loc in np.ndindex(self.shape):
                result[loc] = self.bind(circuit, loc)
        return result

    @classmethod
    def _from_views(cls, source, data, shape):
        result = cls.__new__(cls)
        result._shape, result._data, result._owners = shape, data, dict(source._owners)
        result._ledger = source._ledger
        return result

    def __getitem__(self, index):
        self.validate()
        index = index if isinstance(index, tuple) else (index,)
        # Only batch axes may be indexed; the parameter axis is never exposed.
        if sum(item is not None and item is not Ellipsis for item in index) > self.ndim:
            raise IndexError("Too many binding batch indices.")
        if any(item is not None and item is not Ellipsis and not isinstance(item, (Integral, slice))
               or isinstance(item, (bool, np.bool_)) for item in index):
            raise TypeError("Binding batch selection requires basic integer/slice indices.")
        selected = {names: values[index+(slice(None),)] for names, values in self.data.items()}
        if selected:
            shape = next(iter(selected.values())).shape[:-1]
        else:
            # Zero-stride read-only shape probe; no batch-sized allocation.
            shape = np.broadcast_to(np.uint8(0), self.shape)[index].shape
        return self._from_views(self, selected, shape)

    def reshape(self, *shape):
        self.validate()
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        if any(isinstance(n, bool) or not isinstance(n, Integral) or n < -1 for n in shape):
            raise ValueError("Invalid binding reshape dimensions.")
        if shape.count(-1) > 1:
            raise ValueError("Only one inferred reshape dimension is allowed.")
        if -1 in shape:
            known = prod(n for n in shape if n != -1)
            if known == 0 or self.size % known:
                raise ValueError("Cannot infer the binding reshape dimension.")
            shape = tuple(self.size//known if n == -1 else n for n in shape)
        shape = _shape_tuple(shape)
        if prod(shape) != self.size:
            raise ValueError("Reshaping cannot change the number of binding sets.")
        # Basic selections can be noncontiguous. Admit a copy before reshape
        # could otherwise allocate one invisibly inside NumPy.
        resources = current_resources()
        ledger = self._ledger if resources is None else resources.ledger
        copies = [(shape+(len(names),), values.dtype) for names, values in self.data.items()
                  if not values.flags.c_contiguous and values.size != 0]
        with ResourceScope(ledger):
            outputs = iter(owned_arrays(copies, category="primitive_binding_reshape"))
        data = {}
        for names, values in self.data.items():
            target = shape+(len(names),)
            if values.flags.c_contiguous or values.size == 0:
                data[names] = values.reshape(target)
            else:
                output = next(outputs)
                for index, value in enumerate(values.flat):
                    output.flat[index] = value
                data[names] = output
        return self._from_views(self, data, shape)

    def ravel(self):
        return self.reshape(self.size)
