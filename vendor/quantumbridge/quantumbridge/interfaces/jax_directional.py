# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Optional JAX linear second-rule bridge with explicit transpose and batching.

Only classical transformation plumbing lives here. Every quantum HVP uses
the shared native host plan; no dense Hessian or quantum SDK is captured.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir

from quantumbridge.core.classical import UnsupportedFeature


@dataclass(eq=False)
class DirectionalRule:
    kernel: object
    dtype: object

    def shape(self, transpose):
        return (self.kernel.size,) if transpose else (self.kernel.output_size, self.kernel.size)

    def host(self, point, vector, *, transpose=False, checked=False):
        try:
            if transpose:
                value, _ = self.kernel.run_second_transpose(np.asarray(point), np.asarray(vector))
            else:
                _, _, product, _ = self.kernel.run_second(np.asarray(point), direction=np.asarray(vector))
                value = product[:, :, 0]
            return np.asarray(value, dtype=self.dtype)
        except Exception:
            if not checked:
                raise
            return np.full(self.shape(transpose), np.nan, dtype=self.dtype)


_directional = Primitive("quantumbridge_directional_second")


def _impl(point, vector, *, rule, transpose):
    return jnp.asarray(rule.host(point, vector, transpose=transpose))


def _abstract(point, vector, *, rule, transpose):
    expected = (rule.kernel.output_size, rule.kernel.size) if transpose else (rule.kernel.size,)
    if point.shape != (rule.kernel.size,) or vector.shape != expected or point.dtype != vector.dtype:
        raise ValueError("JAX directional second-rule shapes/dtypes must match packed parameters and outputs.")
    if np.dtype(point.dtype) not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("JAX directional second rules require real float32/float64 arrays.")
    return jax.core.ShapedArray(rule.shape(transpose), point.dtype)


def _lower(ctx, point, vector, *, rule, transpose):
    spec = jax.ShapeDtypeStruct(rule.shape(transpose), rule.dtype)
    def callback(x, v):
        return rule.host(x, v, transpose=transpose, checked=True)
    function = lambda x, v: jax.pure_callback(callback, spec, x, v, vmap_method="sequential")
    return mlir.lower_fun(function, multiple_results=False)(ctx, point, vector)


def _transpose(incoming, point, vector, *, rule, transpose):
    if ad.is_undefined_primal(point):
        raise UnsupportedFeature("higher_order_derivatives", "JAX native third derivatives are not implemented.")
    if isinstance(incoming, ad.Zero):
        return None, ad.Zero(vector.aval)
    return None, _directional.bind(point, incoming, rule=rule, transpose=not transpose)


def _batch(args, axes, *, rule, transpose):
    sizes = [value.shape[axis] for value, axis in zip(args, axes) if axis is not None]
    if not sizes:
        return _directional.bind(*args, rule=rule, transpose=transpose), None
    size = sizes[0]
    if any(other != size for other in sizes):
        raise ValueError("JAX directional mapped axes must have equal lengths.")
    rule.kernel.limits.check("max_vmap_batch", size)
    rule.kernel.limits.check("max_callback_executions", size)
    values = tuple(jnp.broadcast_to(value, (size,)+value.shape) if axis is None else jnp.moveaxis(value, axis, 0)
                   for value, axis in zip(args, axes))
    result = jax.lax.map(lambda pair: _directional.bind(*pair, rule=rule, transpose=transpose), values)
    return result, 0


def _jvp(primals, tangents, *, rule, transpose):
    point, vector = primals
    point_tangent, vector_tangent = tangents
    if not isinstance(point_tangent, ad.Zero):
        raise UnsupportedFeature("higher_order_derivatives", "JAX native third derivatives are not implemented.")
    value = _directional.bind(point, vector, rule=rule, transpose=transpose)
    tangent = ad.Zero.from_primal_value(value) if isinstance(vector_tangent, ad.Zero) else _directional.bind(
        point, vector_tangent, rule=rule, transpose=transpose)
    return value, tangent


_directional.def_impl(_impl)
_directional.def_abstract_eval(_abstract)
mlir.register_lowering(_directional, _lower)
ad.primitive_transposes[_directional] = _transpose
ad.primitive_jvps[_directional] = _jvp
batching.primitive_batchers[_directional] = _batch


def directional_second(point, vector, rule):
    return _directional.bind(point, vector, rule=rule, transpose=False)
