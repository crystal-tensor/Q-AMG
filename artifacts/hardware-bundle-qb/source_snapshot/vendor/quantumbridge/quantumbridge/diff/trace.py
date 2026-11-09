# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Independent first-order classical tracing for native quantum derivatives.

import operator
import numpy as np

from quantumbridge.core.classical import UnsupportedFeature

_NO_COPY_RESHAPE = np.lib.NumpyVersion(np.__version__) >= "2.1.0"


class Dual:
    __array_priority__ = 1000

    def __init__(self, value, derivative, *, _reserved=False):
        if not _reserved:
            self._reserve(np.size(derivative))
        self.value = float(value)
        self.derivative = np.asarray(derivative, dtype=float)

    @staticmethod
    def _reserve(size):
        from quantumbridge.execution.resource_context import current_resources
        scope = current_resources()
        if scope is not None:
            scope.retain("first_parameter_trace_nodes", 8*size)

    def _new(self, value, compute):
        self._reserve(self.derivative.size)
        return Dual(value, compute(), _reserved=True)

    def _other(self, value):
        return value if isinstance(value, Dual) else self._new(value, lambda: np.zeros_like(self.derivative))

    def __add__(self, other):
        other = self._other(other)
        return self._new(self.value + other.value, lambda: self.derivative + other.derivative)

    __radd__ = __add__

    def __neg__(self): return self._new(-self.value, lambda: -self.derivative)
    def __sub__(self, other): return self + -self._other(other)
    def __rsub__(self, other): return self._other(other) + -self

    def __mul__(self, other):
        other = self._other(other)
        return self._new(self.value * other.value, lambda: self.derivative * other.value + other.derivative * self.value)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._other(other)
        return self._new(self.value / other.value, lambda: (self.derivative * other.value - other.derivative * self.value) / other.value ** 2)

    def __rtruediv__(self, other): return self._other(other) / self

    def __pow__(self, other):
        other = self._other(other)
        value = self.value ** other.value
        if np.any(other.derivative):
            if self.value <= 0: raise UnsupportedFeature("classical_power", "Variable powers require a positive base.")
            return self._new(value, lambda: value * (other.derivative * np.log(self.value) + other.value * self.derivative / self.value))
        elif other.value == 0:
            return self._new(value, lambda: np.zeros_like(self.derivative))
        else:
            return self._new(value, lambda: other.value * self.value ** (other.value - 1) * self.derivative)

    def __rpow__(self, other): return self._other(other) ** self
    def sin(self): return self._new(np.sin(self.value), lambda: np.cos(self.value) * self.derivative)
    def cos(self): return self._new(np.cos(self.value), lambda: -np.sin(self.value) * self.derivative)
    def tan(self): return self._new(np.tan(self.value), lambda: self.derivative / np.cos(self.value) ** 2)
    def exp(self): return self._new(np.exp(self.value), lambda: np.exp(self.value) * self.derivative)
    def log(self): return self._new(np.log(self.value), lambda: self.derivative / self.value)
    def sqrt(self): return self ** .5

    def arcsin(self):
        if not -1 < self.value < 1:
            raise ValueError("Inverse sine derivatives require an input strictly inside (-1, 1).")
        return self._new(np.arcsin(self.value), lambda: self.derivative / np.sqrt(1-self.value**2))

    def arccos(self):
        if not -1 < self.value < 1:
            raise ValueError("Inverse cosine derivatives require an input strictly inside (-1, 1).")
        return self._new(np.arccos(self.value), lambda: -self.derivative / np.sqrt(1-self.value**2))

    def arctan(self):
        return self._new(np.arctan(self.value), lambda: self.derivative / (1+self.value**2))

    def sign(self):
        if self.value == 0:
            raise UnsupportedFeature("nonsmooth_preprocessing", "Sign is not differentiable at zero.")
        return self._new(np.sign(self.value), lambda: np.zeros_like(self.derivative))

    def conjugate(self): return self

    def __abs__(self):
        if self.value == 0: raise UnsupportedFeature("nonsmooth_preprocessing", "Absolute value is not differentiable at zero.")
        return self._new(abs(self.value), lambda: np.sign(self.value) * self.derivative)

    def __float__(self):
        raise UnsupportedFeature("classical_cast", "Converting a trainable input to float loses its classical derivative.")

    def __bool__(self):
        raise UnsupportedFeature("trainable_classical_branch", "Differentiation through trainable Python branch conditions is unsupported.")

    def _compare(self, other): return bool(self)
    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = _compare

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs:
            raise UnsupportedFeature("classical_ufunc", "Unsupported classical differentiation operation.")
        unary = {np.sin: "sin", np.cos: "cos", np.tan: "tan", np.exp: "exp", np.log: "log", np.sqrt: "sqrt", np.absolute: "__abs__", np.negative: "__neg__",
                 np.arcsin: "arcsin", np.arccos: "arccos", np.arctan: "arctan", np.sign: "sign", np.conjugate: "conjugate"}
        if ufunc in unary: return getattr(self, unary[ufunc])()
        binary = {np.add: operator.add, np.subtract: operator.sub, np.multiply: operator.mul,
                  np.divide: operator.truediv, np.true_divide: operator.truediv, np.power: operator.pow}
        if ufunc in binary:
            left, right = (self._other(value) for value in inputs)
            return binary[ufunc](left, right)
        raise UnsupportedFeature("classical_ufunc", f"No native derivative rule for {ufunc.__name__}.")


def trace_arguments(args, kwargs, argnums=0, argnames=(), *, order=1, direction=None, limits=None):
    if order not in {1, 2}:
        raise UnsupportedFeature("higher_order_derivatives", "Native classical tracing implements orders one and two.")
    positions = (argnums,) if isinstance(argnums, int) else tuple(argnums)
    names = (argnames,) if isinstance(argnames, str) else tuple(argnames)
    single = isinstance(argnums, int) and not argnames or not positions and len(names) == 1
    selected = [("arg", index, args[index]) for index in positions] + [("kw", name, kwargs[name]) for name in names]
    if not selected: raise ValueError("Select at least one trainable argument.")
    if len(set((kind, key) for kind, key, _ in selected)) != len(selected): raise ValueError("Duplicate trainable argument selection.")
    arrays = [np.asarray(value) for _, _, value in selected]
    if any(array.dtype.kind not in "f" for array in arrays):
        raise TypeError("Native QNode differentiation requires real floating-point trainable inputs.")
    if any(not np.all(np.isfinite(array)) for array in arrays): raise ValueError("Trainable inputs must be finite.")
    total, offset = sum(array.size for array in arrays), 0
    if total > 1024: raise MemoryError("Native classical trace exceeds the 1024-input derivative budget.")
    context = None
    if order == 2:
        from .second_trace import SecondContext
        context = SecondContext(total, direction, limits)
    traced_args, traced_kwargs, layout = list(args), dict(kwargs), []
    for (kind, key, _), array in zip(selected, arrays):
        traced = np.empty(array.shape, dtype=object)
        for index in np.ndindex(array.shape):
            position = offset + np.ravel_multi_index(index, array.shape)
            if context is None:
                Dual._reserve(total)
                derivative = np.zeros(total)
                derivative[position] = 1
                traced[index] = Dual(array[index], derivative, _reserved=True)
            else:
                traced[index] = context.variable(array[index], position)
        value = traced.item() if array.ndim == 0 else traced
        if kind == "arg": traced_args[key] = value
        else: traced_kwargs[key] = value
        layout.append((array.shape, slice(offset, offset + array.size)))
        offset += array.size
    return tuple(traced_args), traced_kwargs, layout, single, total


def flatten_tree(value):
    from quantumbridge.execution.resource_context import current_resources
    from quantumbridge.execution.array_ownership import retained_array, _defined_type
    scope = current_resources()
    leaves, offset = [], 0
    def visit(item):
        nonlocal offset
        kind = type(item)
        mapping, sequence = issubclass(kind, dict), issubclass(kind, (tuple, list))
        if scope is not None and (mapping or sequence) and not any(kind is base for base in (dict, tuple, list)):
            raise UnsupportedFeature("differentiable_output_conversion", "Managed output containers require a controlled builtin tree layout.")
        if mapping: return ("dict", tuple((key, visit(child)) for key, child in item.items()))
        if sequence: return ("tuple" if issubclass(kind, tuple) else "list", tuple(visit(child) for child in item))
        if issubclass(kind, np.ndarray):
            # A base view bypasses subclass conversion/copy hooks. Source
            # strides remain intact; only the contiguous destination reshapes.
            item = np.ndarray.view(item, np.ndarray)
            shape, size, kind = item.shape, item.size, item.dtype.kind
        elif issubclass(kind, np.generic):
            if scope is not None and not _defined_type(kind, "numpy"):
                raise UnsupportedFeature("differentiable_output_conversion", "Managed output scalars require a known numeric storage layout.")
            shape, size, kind = (), 1, item.dtype.kind
        elif kind is float or kind is int:
            shape, size, kind = (), 1, "f"
        else:
            if scope is not None:
                raise UnsupportedFeature("differentiable_output_conversion", "Managed native output trees cannot execute an opaque array conversion protocol.")
            item = np.asarray(item)
            shape, size, kind = item.shape, item.size, item.dtype.kind
        if kind not in "fiu": raise UnsupportedFeature("differentiable_output", "Derivatives currently require real numeric outputs.")
        start = offset
        offset += size
        leaves.append((item, shape, start, size))
        return ("leaf", shape, start, size)
    try:
        spec = visit(value)
        flat = retained_array(None if scope is None else scope.ledger, "native_tree_flatten", (offset,), float)
        for item, shape, start, size in leaves:
            np.copyto(flat[start:start+size].reshape(shape), item, casting="unsafe")
        return flat, spec
    finally:
        # The recursive visitor must not keep borrowed source storage in an
        # unreachable closure cycle until a future framework GC happens.
        visit = None
        leaves.clear()


def unflatten_tree(flat, spec, trailing_shape=(), *, reshape=None):
    kind = spec[0]
    if kind == "dict": return {key: unflatten_tree(flat, child, trailing_shape, reshape=reshape) for key, child in spec[1]}
    if kind in {"tuple", "list"}:
        values = [unflatten_tree(flat, child, trailing_shape, reshape=reshape) for child in spec[1]]
        return tuple(values) if kind == "tuple" else values
    _, shape, start, size = spec
    part = flat[start:start + size]
    target = shape + trailing_shape
    if reshape is not None:
        value = reshape(part, target)
    elif isinstance(part, np.ndarray):
        value = np.ndarray.view(part, np.ndarray)
        try:
            if _NO_COPY_RESHAPE:
                value = np.reshape(value, target, copy=False)
            else:
                # NumPy 1.x has no copy=False; shape assignment is view-only.
                value.shape = target
        except (AttributeError, ValueError):
            from math import prod
            if prod(target) != part.size:
                raise ValueError("Tree reconstruction shape does not match its numerical payload.") from None
            from quantumbridge.execution.resource_context import current_resources
            from quantumbridge.execution.array_ownership import retained_array
            scope = current_resources()
            value = retained_array(None if scope is None else scope.ledger,
                "native_tree_reconstruction", target, part.dtype)
            np.copyto(value.reshape(part.shape), part, casting="no")
    else:
        value = part.reshape(target)
    return value.item() if not shape and not trailing_shape and isinstance(value, np.ndarray) else value
