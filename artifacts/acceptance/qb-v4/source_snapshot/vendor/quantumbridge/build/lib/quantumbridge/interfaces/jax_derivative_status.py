# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Explicit CPU derivative queries with runtime outcomes and native source maps.

These return computed derivative data, not a third-order differentiable graph.
The existing QNode JAX graph remains the automatic differentiation interface.
"""

from copy import copy

import numpy as np

from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.diff.trace import flatten_tree, unflatten_tree
from .framework_derivatives import execute_mapped_derivative
from .jax_autodiff import JAXBridgeLimits
from .native_host import NativeHostKernel, describe_native_source


def execute_jax_derivative_status(node, args, kwargs, *, kind, tangent=None):
    import jax
    import jax.numpy as jnp
    from jax.custom_derivatives import SymbolicZero
    if jax.default_backend() != "cpu":
        raise UnsupportedFeature("jax_device", "Derivative status queries currently require CPU.")
    if len(args) != 1 or kwargs:
        raise UnsupportedFeature("jax_derivative_status_arguments", "Explicit derivative status currently requires one floating array argument and no keyword inputs.")
    x = jnp.asarray(args[0])
    if x.dtype not in (jnp.float32, jnp.float64):
        raise TypeError("Derivative status input must use float32 or float64.")
    if node.max_diff != 2 and kind in {"hvp", "hessian"}:
        raise UnsupportedFeature("higher_order_derivatives", "Second derivatives require max_diff=2.")
    limits = node.jax_limits or JAXBridgeLimits()
    limits.check("max_input_elements", x.size)
    if kind == "hvp":
        tangent = jnp.asarray(tangent)
        if tangent.shape != x.shape or tangent.dtype.kind not in "fiu":
            raise ValueError("JAX HVP tangent must match the input array shape.")
        tangent = tangent.astype(x.dtype)
    owner = copy(node)
    requested_method = node.diff_method
    owner.diff_method = "adjoint" if node.diff_method == "second_adjoint" else node.diff_method
    source, recorder, program = owner._construct((x,), {}, prepare_only=True)
    kernel = NativeHostKernel(owner, source, recorder, program, limits, validate_derivatives=False)
    kernel.angles = []
    size, dtype = x.size, np.dtype(x.dtype)
    input_shape = x.shape
    order = 2 if kind in {"hvp", "hessian"} else 1
    tail = input_shape+input_shape if kind == "hessian" else input_shape
    flat_shape = (kernel.output_size, size, size) if kind == "hessian" else (kernel.output_size, size)
    limits.check("max_jacobian_elements", int(np.prod(flat_shape)))

    def pack(raw):
        source, recorder, program = owner._construct((raw.reshape(input_shape),), {}, prepare_only=True)
        description = describe_native_source(owner, source, recorder, limits)
        if len(description.angles) != kernel.size or description.spec != kernel.spec:
            raise UnsupportedFeature("jax_derivative_structure", "Derivative query source structure must be static.")
        return jnp.stack([jnp.asarray(a, dtype=x.dtype) for a in description.angles]) if description.angles else jnp.empty(0, x.dtype)

    def callback(raw, packed, classical, second_map, direction):
        result = execute_mapped_derivative(kernel, raw, packed, classical, second_map, direction,
            kind=kind, input_shape=input_shape, method=requested_method, limits=getattr(owner, "unified_limits", None))
        derivative = np.full(flat_shape, np.nan, dtype=dtype) if result.code else flatten_tree(result.derivative)[0].reshape(flat_shape).astype(dtype)
        return derivative, np.asarray(result.code, dtype=np.int32)

    specs = (jax.ShapeDtypeStruct(flat_shape, x.dtype), jax.ShapeDtypeStruct((), jnp.int32))
    def evaluate(raw, direction):
        packed = pack(raw)
        classical = jax.jacfwd(pack)(raw)
        if order == 1:
            second_map = jnp.empty((kernel.size, size, 0), dtype=x.dtype)
        elif kind == "hvp":
            product = jax.jvp(jax.jacfwd(pack), (raw,), (direction,))[1]
            second_map = product[:, :, None]
        else:
            second_map = jax.jacfwd(jax.jacfwd(pack))(raw)
        return jax.pure_callback(callback, specs, raw, packed, classical, second_map, direction,
            vmap_method="sequential")

    @jax.custom_vjp
    def checked(raw, direction):
        return evaluate(raw, direction)

    def forward(raw, direction):
        raw, direction = raw.value, direction.value
        return evaluate(raw, direction), (raw, direction)

    def backward(residuals, cotangents):
        derivative_cotangent, _ = cotangents
        if not isinstance(derivative_cotangent, SymbolicZero):
            raise UnsupportedFeature("derivative_status_higher_order", "Differentiate the QNode graph; explicit derivative-query payloads do not supply additional derivative rules.")
        return tuple(jnp.zeros_like(value) for value in residuals)

    checked.defvjp(forward, backward, symbolic_zeros=True)
    raw = x.reshape(-1)
    direction = jnp.zeros_like(raw) if tangent is None else tangent.reshape(-1)
    derivative, status = checked(raw, direction)
    return unflatten_tree(derivative, kernel.spec, tail), status
