# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Second classical derivatives or Hessian-vector products, never finite differences."""

from numbers import Integral

import numpy as np

from quantumbridge.core.classical import UnsupportedFeature
from .higher_order_limits import resolve_higher_order_limits
from .trace import Dual


class SecondContext:
    """Shared directional seeds and conservative cumulative buffer/node budgets."""

    def __init__(self, size, direction=None, limits=None, *, node_budget="max_parameter_graph_nodes"):
        if isinstance(size, bool) or not isinstance(size, Integral) or size < 0:
            raise ValueError("Classical trace size must be a nonnegative integer.")
        self.limits = resolve_higher_order_limits(limits)
        self.limits.check("max_derivative_order", 2)
        self.limits.check("max_input_parameters", size)
        self.size, self.node_budget, self.nodes = size, node_budget, 0
        self.directional = direction is not None
        from quantumbridge.execution.resource_context import current_resources
        self.resources = current_resources()
        if self.resources is not None:
            self.resources.retain("parameter_trace_directions", 8*size*(size if direction is None else 1))
        if direction is None:
            self.limits.check("max_hessian_elements", size*size)
            self.limits.check("max_memory_bytes", 8*size*size)
            self.directions = np.eye(size)
        else:
            raw = np.asarray(direction)
            if raw.shape != (size,) or raw.dtype.kind not in "fiu" or not np.all(np.isfinite(raw)):
                raise ValueError("Classical HVP direction must be a finite real vector matching the inputs.")
            self.limits.check("max_memory_bytes", 8*size)
            self.directions = np.array(raw, dtype=float, copy=True).reshape(size, 1)
        self.directions.flags.writeable = False
        self.second_shape = (size, self.directions.shape[1])
        self.buffer_bytes_per_node = 8*(size + size*self.second_shape[1])

    def reserve(self):
        self.limits.check(self.node_budget, self.nodes+1)
        self.limits.check("max_memory_bytes", self.directions.nbytes + (self.nodes+1)*self.buffer_bytes_per_node)
        if self.resources is not None:
            self.resources.retain("parameter_trace_nodes", self.buffer_bytes_per_node)
        self.nodes += 1

    def constant(self, value):
        self.reserve()
        return SecondDual(value, np.zeros(self.size), np.zeros(self.second_shape), self,
                          depends_on_inputs=False, _reserved=True)

    def variable(self, value, index):
        if not 0 <= index < self.size:
            raise ValueError("Classical seed index is out of range.")
        self.reserve()
        derivative = np.zeros(self.size)
        derivative[index] = 1.
        return SecondDual(value, derivative, np.zeros(self.second_shape), self,
                          depends_on_inputs=True, _reserved=True)


class SecondDual(Dual):
    """Primal, gradient and gradient-direction derivative in one native graph."""

    __array_priority__ = 1100

    def __init__(self, value, derivative, second, context, *, depends_on_inputs=None, _reserved=False):
        if not _reserved:
            context.reserve()
        if np.shape(derivative) != (context.size,) or np.shape(second) != context.second_shape:
            raise ValueError("Second-order classical seed shapes must match the context.")
        self.value, self.context = float(value), context
        self.derivative = np.asarray(derivative, dtype=float)
        self.second = np.asarray(second, dtype=float)
        if depends_on_inputs is None:
            depends_on_inputs = bool(np.any(self.derivative) or np.any(self.second))
        if not isinstance(depends_on_inputs, bool):
            raise TypeError("Classical input dependency must be a boolean.")
        self.depends_on_inputs = depends_on_inputs
        if not np.isfinite(self.value) or not np.all(np.isfinite(self.derivative)) or not np.all(np.isfinite(self.second)):
            raise ValueError("Second-order classical values and derivatives must be finite.")

    def _other(self, value):
        if isinstance(value, SecondDual):
            if value.context is not self.context:
                raise ValueError("Cannot mix independent second-order classical contexts.")
            return value
        if isinstance(value, Dual):
            raise UnsupportedFeature("mixed_trace_order", "A first-order trace cannot supply missing second-order information.")
        return self.context.constant(value)

    def _result(self, calculate, *, depends_on_inputs=None):
        self.context.reserve()
        value, derivative, second = calculate()
        return SecondDual(value, derivative, second, self.context,
                          depends_on_inputs=self.depends_on_inputs if depends_on_inputs is None else depends_on_inputs,
                          _reserved=True)

    def _unary(self, value, first, second):
        return self._result(lambda: (value, first*self.derivative,
                                    first*self.second + second*np.outer(self.derivative, self.derivative @ self.context.directions)))

    def __add__(self, other):
        other = self._other(other)
        return self._result(lambda: (self.value+other.value, self.derivative+other.derivative, self.second+other.second),
                            depends_on_inputs=self.depends_on_inputs or other.depends_on_inputs)

    __radd__ = __add__

    def __neg__(self):
        return self._result(lambda: (-self.value, -self.derivative, -self.second))

    def __sub__(self, other): return self + -self._other(other)
    def __rsub__(self, other): return self._other(other) + -self

    def __mul__(self, other):
        other = self._other(other)
        directions = self.context.directions
        return self._result(lambda: (self.value*other.value,
                                    self.derivative*other.value+other.derivative*self.value,
                                    self.second*other.value+other.second*self.value
                                    + np.outer(self.derivative, other.derivative @ directions)
                                    + np.outer(other.derivative, self.derivative @ directions)),
                            depends_on_inputs=self.depends_on_inputs or other.depends_on_inputs)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._other(other)
        return self * other._unary(1/other.value, -1/other.value**2, 2/other.value**3)

    def __rtruediv__(self, other): return self._other(other) / self

    def __pow__(self, other):
        other = self._other(other)
        if other.depends_on_inputs:
            if self.value <= 0:
                raise UnsupportedFeature("classical_power", "Variable powers require a positive base.")
            return (other*self.log()).exp()
        power = other.value
        if power == 0: return self.context.constant(1.)
        if power == 1: return self._unary(self.value, 1., 0.)
        return self._unary(self.value**power, power*self.value**(power-1), power*(power-1)*self.value**(power-2))

    def __rpow__(self, other): return self._other(other) ** self
    def sin(self): return self._unary(np.sin(self.value), np.cos(self.value), -np.sin(self.value))
    def cos(self): return self._unary(np.cos(self.value), -np.sin(self.value), -np.cos(self.value))
    def tan(self):
        first = 1/np.cos(self.value)**2
        return self._unary(np.tan(self.value), first, 2*np.tan(self.value)*first)
    def exp(self):
        value = np.exp(self.value)
        return self._unary(value, value, value)
    def log(self):
        if self.value <= 0:
            raise ValueError("Real classical logarithms require a positive base.")
        return self._unary(np.log(self.value), 1/self.value, -1/self.value**2)
    def sqrt(self): return self ** .5

    def arcsin(self):
        if not -1 < self.value < 1:
            raise ValueError("Inverse sine derivatives require an input strictly inside (-1, 1).")
        base = 1-self.value**2
        return self._unary(np.arcsin(self.value), 1/np.sqrt(base), self.value/base**1.5)

    def arccos(self):
        if not -1 < self.value < 1:
            raise ValueError("Inverse cosine derivatives require an input strictly inside (-1, 1).")
        base = 1-self.value**2
        return self._unary(np.arccos(self.value), -1/np.sqrt(base), -self.value/base**1.5)

    def arctan(self):
        base = 1+self.value**2
        return self._unary(np.arctan(self.value), 1/base, -2*self.value/base**2)

    def sign(self):
        if self.value == 0:
            raise UnsupportedFeature("nonsmooth_preprocessing", "Sign is not differentiable at zero.")
        return self._unary(np.sign(self.value), 0., 0.)

    def __abs__(self):
        if self.value == 0:
            raise UnsupportedFeature("nonsmooth_preprocessing", "Absolute value is not twice differentiable at zero.")
        return self._unary(abs(self.value), np.sign(self.value), 0.)
