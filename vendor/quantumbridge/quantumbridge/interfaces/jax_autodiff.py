# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Optional CPU JAX graphs around existing native plans and quantum Jacobians."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.diff.trace import unflatten_tree
from .native_host import MissingOptionalDependency, NativeHostKernel


@dataclass(frozen=True)
class JAXBridgeLimits:
    max_input_elements: int = 1024
    max_output_elements: int = 100000
    max_jacobian_elements: int = 2000000
    max_parameter_elements: int = 1024
    max_vmap_batch: int = 128
    max_callback_executions: int = 128
    max_total_shots: int = 1000000
    max_transform_tapes: int = 128

    def __post_init__(self):
        if any(isinstance(x, bool) or not isinstance(x, Integral) or x < 1 for x in self.__dict__.values()):
            raise ValueError("JAX bridge limits require positive integers.")

    def check(self, name, value):
        if value > getattr(self, name):
            raise MemoryError("Native JAX bridge exceeded " + name)


def execute_jax(node, args, kwargs, *, return_status=False):
    if getattr(node, "unified_limits", None) is not None:
        raise UnsupportedFeature("jax_unified_graph_resources", "Ordinary JAX graph callbacks do not yet have a shared request account; use explicit native derivative-status queries for per-request budgets.")
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise MissingOptionalDependency("Native JAX QNode requires optional jax/jaxlib.") from exc
    if node.max_diff not in {1, 2}:
        raise UnsupportedFeature("higher_order_derivatives", "Native JAX supports derivative orders at most two.")
    if set(node.gradient_kwargs) - {"step"}:
        raise UnsupportedFeature("jax_gradient_options", "Set the native gradient method through diff_method; JAX gradient_kwargs supports only a static step.")
    step = node.gradient_kwargs.get("step", 1e-6)
    if isinstance(step, bool) or not isinstance(step, Real) or not np.isfinite(step) or step <= 0:
        raise ValueError("Native JAX derivative step must be a finite positive real scalar.")
    if jax.default_backend() != "cpu":
        raise UnsupportedFeature("jax_device", "The native JAX bridge currently supports CPU host execution only.")
    from quantumbridge.devices import StatevectorDevice, DensityMatrixDevice, MatrixProductStateDevice
    if not isinstance(node.device, (StatevectorDevice, DensityMatrixDevice, MatrixProductStateDevice)):
        raise UnsupportedFeature("jax_device", "JAX execution requires a supported native vector/density/MPS device.")
    limits = JAXBridgeLimits() if node.jax_limits is None else node.jax_limits
    if not isinstance(limits, JAXBridgeLimits):
        raise TypeError("jax_limits requires JAXBridgeLimits.")
    leaves = jax.tree_util.tree_leaves((args, kwargs))
    numeric = [value for value in leaves if hasattr(value, "dtype") or isinstance(value, (float, int, bool))]
    if any(jnp.asarray(value).dtype.kind == "c" for value in numeric):
        raise UnsupportedFeature("jax_complex", "Complex inputs/outputs do not have a verified native JAX derivative contract.")
    floating = [jnp.asarray(value) for value in numeric if jnp.asarray(value).dtype.kind == "f"]
    limits.check("max_input_elements", sum(jnp.asarray(value).size for value in numeric))
    dtype = jnp.result_type(*floating) if floating else jnp.dtype("float32")
    if dtype not in (jnp.float32, jnp.float64):
        raise UnsupportedFeature("jax_dtype", "Native JAX supports float32/float64, respecting process x64 configuration.")
    source, recorder, program = node._construct(args, kwargs, prepare_only=True)
    kernel = NativeHostKernel(node, source, recorder, program, limits)
    packed = jnp.stack([jnp.asarray(value, dtype=dtype) for value in kernel.angles]) if kernel.angles else jnp.empty(0, dtype=dtype)
    for value in floating:
        packed = packed + 0 * jnp.sum(value)
    # Callbacks capture only native source structure, never live JAX tracers.
    kernel.angles = []
    out_spec = jax.ShapeDtypeStruct((kernel.output_size,), dtype)
    jac_spec = jax.ShapeDtypeStruct((kernel.output_size, kernel.size), dtype)
    status_spec = jax.ShapeDtypeStruct((), dtype)
    def host(values, derivative):
        array = np.asarray(values, dtype=float)
        value, matrix, metadata = kernel.run(array, derivative)
        if derivative:
            return value.astype(dtype), matrix.astype(dtype), metadata
        return value.astype(dtype), metadata

    def checked(values, derivative):
        # The explicit status is returned as data, not a callback side effect.
        # It is nondifferentiable and always has zero tangent.
        try:
            result = host(values, derivative)
            outputs = result[:2] if derivative else result[:1]
            return (*outputs, np.asarray(0, dtype=dtype)), result[-1]
        except UnsupportedFeature:
            status = 2
        except MemoryError:
            status = 3
        except (ValueError, FloatingPointError):
            status = 1
        except Exception:
            status = 4
        flat = np.full(out_spec.shape, np.nan, dtype=dtype)
        outputs = (flat, np.full(jac_spec.shape, np.nan, dtype=dtype)) if derivative else (flat,)
        return (*outputs, np.asarray(status, dtype=dtype)), {"runtime_status": status, "interface": "jax"}

    def callback_function(derivative):
        from jax.custom_batching import custom_vmap
        specs = (out_spec, jac_spec, status_spec) if derivative else (out_spec, status_spec)
        @custom_vmap
        def callback(values):
            return jax.pure_callback(lambda x: checked(x, derivative)[0], specs, values, vmap_method="sequential")
        @callback.def_vmap
        def batched(axis_size, in_batched, values):
            limits.check("max_vmap_batch", axis_size)
            limits.check("max_callback_executions", axis_size)
            if not in_batched[0]:
                outputs = callback(values)
                return outputs, jax.tree_util.tree_map(lambda _: False, outputs)
            outputs = jax.lax.map(callback, values)
            return outputs, jax.tree_util.tree_map(lambda _: True, outputs)
        return callback

    callbacks = {False: callback_function(False), True: callback_function(True)}

    def evaluate(values, derivative):
        limits.check("max_callback_executions", 1)
        if isinstance(values, jax.core.Tracer):
            node.last_metadata = {"interface": "jax", "execution": "CPU pure host callback",
                                  "trace_configuration_only": True,
                                  "jit_callback_execution_count": "not guaranteed; no side-effect receipt",
                                  "callback_budget_semantics": "logical mapped expansion per axis, not a strict JIT scheduling count",
                                  "runtime_failure": "NaN output plus explicit jax_with_status code 1/2/3/4"}
            return callbacks[derivative](values)
        if return_status:
            outputs, metadata = checked(values, derivative)
        else:
            result = host(values, derivative)
            outputs, metadata = (*result[:-1], np.asarray(0, dtype=dtype)), result[-1]
        node.last_metadata = metadata
        return tuple(jnp.asarray(x) for x in outputs)

    @jax.custom_jvp
    def second_point(values):
        return values

    @second_point.defjvp
    def reject_third_order(primals, tangents):
        raise UnsupportedFeature("higher_order_derivatives", "Native JAX third derivatives are not implemented.")

    @jax.custom_jvp
    def derivatives(values):
        return evaluate(values, True)

    @derivatives.defjvp
    def second_order_jvp(primals, tangents):
        if node.max_diff != 2:
            raise UnsupportedFeature("higher_order_derivatives", "Native JAX second derivatives require max_diff=2.")
        from .jax_directional import DirectionalRule, directional_second
        point = second_point(primals[0])
        value, matrix, status = evaluate(point, True)
        rule = DirectionalRule(kernel, np.dtype(dtype))
        product = directional_second(point, tangents[0], rule)
        node.last_metadata = {**node.last_metadata, "hvp_streamed": True, "native_linear_second_rule": True,
                              "second_rule_execution": "CPU directional host primitive; native transpose, no packed Hessian"}
        return (value, matrix, status), (matrix @ tangents[0], product, jnp.zeros_like(status))

    @jax.custom_jvp
    def native(values):
        return evaluate(values, False)

    @native.defjvp
    def native_jvp(primals, tangents):
        value, matrix, status = derivatives(primals[0])
        return (value, status), (matrix @ tangents[0], jnp.zeros_like(status))

    flat, status = native(packed)
    result = unflatten_tree(flat, kernel.spec)
    return (result, status) if return_status else result
