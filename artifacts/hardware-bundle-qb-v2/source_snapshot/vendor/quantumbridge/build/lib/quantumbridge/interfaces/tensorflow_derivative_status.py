# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""CPU TensorFlow derivative queries returning the actual native outcome."""

from copy import copy

import numpy as np

from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.diff.trace import flatten_tree, unflatten_tree
from .framework_derivatives import execute_mapped_derivative
from .native_host import NativeHostKernel, describe_native_source
from .tensorflow_autodiff import TensorFlowBridgeLimits
from .tensorflow_callbacks import TensorFlowCallback, status_numpy_function


def execute_tensorflow_derivative_status(node, args, kwargs, *, kind, tangent=None, cotangent=None):
    import tensorflow as tf
    if len(args) != 1 or kwargs:
        raise UnsupportedFeature("tensorflow_derivative_status_arguments", "Explicit derivative status currently requires one floating array argument and no keyword inputs.")
    x = tf.convert_to_tensor(args[0])
    if x.dtype not in (tf.float32, tf.float64):
        raise TypeError("Derivative queries require float32/float64 inputs.")
    if not x.shape.is_fully_defined():
        raise UnsupportedFeature("tensorflow_dynamic_shape")
    if tf.distribute.has_strategy() or any(label in x.device.upper() for label in ("GPU", "TPU")):
        raise UnsupportedFeature("tensorflow_device", "Derivative queries require a local CPU host.")
    if node.max_diff != 2 and kind in {"hessian", "hvp"}:
        raise UnsupportedFeature("higher_order_derivatives", "Second derivatives require max_diff=2.")
    input_shape, dtype, size = tuple(x.shape), x.dtype, x.shape.num_elements()
    limits = node.tensorflow_limits or TensorFlowBridgeLimits()
    limits.check("max_input_elements", size)
    if kind == "hvp":
        tangent = tf.convert_to_tensor(tangent)
        if tangent.shape != x.shape or not (tangent.dtype.is_floating or tangent.dtype.is_integer):
            raise ValueError("TensorFlow HVP tangent must match the input shape.")
        tangent = tf.cast(tangent, dtype)
    owner, requested_method = copy(node), node.diff_method
    owner.diff_method = "adjoint" if requested_method == "second_adjoint" else requested_method
    source, recorder, program = owner._construct((x,), {}, prepare_only=True)
    kernel = NativeHostKernel(owner, source, recorder, program, limits,
        interface="tensorflow", validate_derivatives=False)
    kernel.angles = []
    order = 2 if kind in {"hvp", "hessian"} else 0 if kind == "value" else 1
    tail = () if kind in {"value", "vjp"} else input_shape+input_shape if kind == "hessian" else input_shape
    flat_shape = (size,) if kind == "vjp" else (kernel.output_size,) if kind == "value" else (kernel.output_size, size, size) if kind == "hessian" else (kernel.output_size, size)
    limits.check("max_jacobian_elements", int(np.prod(flat_shape)))

    def prepare_pack(raw):
        source, recorder, program = owner._construct((tf.reshape(raw, input_shape),), {}, prepare_only=True)
        description = describe_native_source(owner, source, recorder, limits, interface="tensorflow")
        if len(description.angles) != kernel.size or description.spec != kernel.spec:
            raise UnsupportedFeature("tensorflow_derivative_structure", "Derivative query source structure must be static.")
        return tf.stack([tf.cast(a, dtype) for a in description.angles]) if description.angles else tf.zeros((0,), dtype)

    pack = tf.function(prepare_pack, autograph=False)
    concrete = pack.get_concrete_function(tf.TensorSpec((size,), dtype))
    if concrete.variables:
        raise UnsupportedFeature("tensorflow_closure_variables", "Pass every trainable variable as an explicit derivative-query input.")
    raw = tf.reshape(x, (size,))
    direction = tf.zeros_like(raw) if tangent is None else tf.reshape(tangent, (size,))
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as outer:
        outer.watch(raw)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as inner:
            inner.watch(raw)
            packed = pack(raw)
        classical = inner.jacobian(packed, raw, experimental_use_pfor=False,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
        del inner
        projected = tf.linalg.matvec(classical, direction)
    if order != 2:
        second_map = tf.zeros((kernel.size, size, 0), dtype)
    elif kind == "hvp":
        second_map = outer.jacobian(projected, raw, experimental_use_pfor=False,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)[:, :, None]
    else:
        second_map = outer.jacobian(classical, raw, experimental_use_pfor=False,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
    del outer
    if kind == "vjp":
        def ordered_cotangents(tree, spec):
            tag = spec[0]
            if tag == "dict":
                if not isinstance(tree, dict) or set(tree) != {key for key, _ in spec[1]}:
                    raise ValueError("VJP cotangent dict keys must match the output tree.")
                return [leaf for key, child in spec[1] for leaf in ordered_cotangents(tree[key], child)]
            if tag in {"tuple", "list"}:
                if not isinstance(tree, tuple if tag == "tuple" else list) or len(tree) != len(spec[1]):
                    raise ValueError("VJP cotangent containers must match the output tree.")
                return [leaf for item, child in zip(tree, spec[1]) for leaf in ordered_cotangents(item, child)]
            leaf = tf.convert_to_tensor(tree, dtype_hint=dtype)
            if leaf.shape != spec[1] or not (leaf.dtype.is_floating or leaf.dtype.is_integer):
                raise ValueError("VJP cotangent leaf shapes/dtypes must match real outputs.")
            return [leaf]
        cot_leaves = ordered_cotangents(cotangent, kernel.spec)
        cotangent_flat = tf.concat([tf.reshape(tf.cast(c, dtype), (-1,)) for c in cot_leaves], axis=0)
        if cotangent_flat.shape != (kernel.output_size,):
            raise ValueError("TensorFlow VJP cotangent size must match the output.")
    else:
        cotangent_flat = tf.zeros((0,), dtype)

    def callback(raw, packed, classical, second_map, direction, cotangent_flat):
        result = execute_mapped_derivative(kernel, raw, packed, classical, second_map, direction,
            kind=kind, input_shape=input_shape, method=requested_method,
            limits=getattr(owner, "unified_limits", None),
            cotangent=unflatten_tree(cotangent_flat, kernel.spec) if kind == "vjp" else None)
        value = result.value if kind == "value" else result.derivative
        payload = np.full(flat_shape, np.nan, dtype.as_numpy_dtype) if result.code else flatten_tree(value)[0].reshape(flat_shape).astype(dtype.as_numpy_dtype)
        return payload, np.int32(result.code)

    callback_owner = TensorFlowCallback(callback)

    with tf.device("/CPU:0"):
        payload, status = status_numpy_function(callback_owner,
            (raw, packed, classical, second_map, direction, cotangent_flat), (dtype, tf.int32))
        payload.set_shape(flat_shape)
        status.set_shape(())
    value = tf.reshape(payload, input_shape) if kind == "vjp" else unflatten_tree(payload, kernel.spec, tail, reshape=tf.reshape)
    return value, status
