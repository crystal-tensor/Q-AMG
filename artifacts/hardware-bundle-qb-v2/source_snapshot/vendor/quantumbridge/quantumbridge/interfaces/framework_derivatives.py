# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Native execution of classical source maps supplied by optional frameworks."""

import numpy as np
from math import prod

from quantumbridge.execution import DerivativeExecutionRequest
from quantumbridge.execution.device_derivatives import execute_device_derivatives
from quantumbridge.execution.derivative_planning import finish_derivative
from quantumbridge.execution.resource_context import current_resources
from quantumbridge.diff.trace import flatten_tree
from quantumbridge.execution.array_reservations import array_layout, borrow_callback_array, mapped_matmul, mapped_second as compose_second_maps, scope_array


def execute_mapped_derivative(kernel, raw, packed, classical, second_map, direction, *,
                              kind, input_shape, method, limits, cotangent=None):
    """The framework supplies only classical maps; quantum work stays native."""
    order = 2 if kind in {"hvp", "hessian"} else 0 if kind == "value" else 1
    input_shape = array_layout(input_shape, float)[0]
    size = prod(input_shape)
    request = DerivativeExecutionRequest(kernel.source, (), derivative_order=order,
        derivative_kind=kind, method=method, tangent=None if kind != "hvp" else direction,
        cotangent=cotangent, limits=limits)
    def prepare(device, request):
        nonlocal raw, packed, classical, second_map, direction
        raw, packed, classical, second_map, direction = (
            borrow_callback_array(array) for array in (raw, packed, classical, second_map, direction))
        if method == "second_adjoint" and order != 2:
            raise ValueError("second_adjoint requires derivative order two.")
        for array in (raw, packed, classical, second_map, direction):
            if not isinstance(array, np.ndarray) or array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
                raise ValueError("Derivative source maps must be finite and real.")
        if np.shape(raw) != (size,) or np.shape(packed) != (kernel.size,) or np.shape(classical) != (kernel.size, size):
            raise ValueError("Framework source map shape changed after tracing.")
        if np.shape(direction) != (size,):
            raise ValueError("Framework direction must match the input vector.")
        if order == 2 and np.shape(second_map) != (kernel.size, size, 1 if kind == "hvp" else size):
            raise ValueError("Framework second source map has an invalid shape.")
        if kind == "vjp":
            cot, spec = flatten_tree(cotangent)
            if spec != kernel.spec or not np.all(np.isfinite(cot)):
                raise ValueError("Derivative cotangent must match the native output.")
        scope = current_resources()
        if hasattr(kernel, "admit"): kernel.admit(scope)
        packed_direction = mapped_matmul(scope, "framework_packed_direction", classical, direction) if kind == "hvp" else None
        plan = kernel.plan(packed, derivative=max(1, order), direction=packed_direction)
        count = len(plan.recorder.parameters)
        own = scope_array(scope, "framework_native_jacobian", (count, kernel.size), float,
            lambda: plan.recorder.classical_jacobian(kernel.size))
        mapped = mapped_matmul(scope, "framework_source_maps", own, classical)
        scope.ledger.unavailable("framework_allocator_memory", "Classical framework graphs and allocator-owned source maps are outside the native owned-array account.")
        mapped_second = None
        if order == 2:
            columns = 1 if kind == "hvp" else kernel.size
            internal = scope_array(scope, "framework_native_second_map", (count, kernel.size, columns), float,
                lambda: plan.recorder.classical_second(kernel.size, columns))
            mapped_second = compose_second_maps(scope, internal, classical, own, second_map,
                directional=kind == "hvp")
        point = scope_array(scope, "framework_native_point", (count,), float, plan.recorder.native_values)
        prepared = finish_derivative(request, plan.node, plan, point, mapped,
            mapped_second, direction if kind == "hvp" else None, size,
            [(input_shape, slice(0, size))], True, kernel.spec, kernel.output_size)
        prepared.buffer_bytes -= mapped.nbytes+(0 if mapped_second is None else mapped_second.nbytes)
        return prepared
    result = execute_device_derivatives(kernel.node.device, request, return_status=True, _prepare=prepare)
    if hasattr(kernel, "storage_owner"):
        result.metadata["persistent_owned_bytes"] = kernel.storage_owner.nbytes
        result.metadata["resident_charge_semantics"] = "request admission for existing immutable storage, not another physical allocation"
    return result
