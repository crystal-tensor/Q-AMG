# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Preserve native QNode input selection while sharing Device execution policy."""

from copy import copy

import numpy as np

from quantumbridge.core.classical import UnsupportedFeature
from .derivative_request import DerivativeExecutionRequest
from .device_derivatives import execute_device_derivatives
from .resource_context import current_resources


def manages_native_node(node):
    return node.unified_limits is not None or current_resources() is not None


def execute_native_qnode(node, args, kwargs, *, order=0, argnums=0, argnames=(), method=None, tangent=None):
    from quantumbridge.diff.higher_order_limits import resolve_higher_order_limits
    from quantumbridge.diff.trace import flatten_tree, trace_arguments
    from quantumbridge.interfaces.native_host import _shape_tree
    from quantumbridge.qml.shot_vector import ShotSpec
    from .derivative_planning import finish_derivative

    owner = copy(node)
    owner.diff_method = method or node.diff_method
    if owner.diff_method == "second_adjoint": owner.diff_method = "adjoint"
    kind = "value" if not order else "jacobian" if order == 1 else "hessian" if tangent is None else "hvp"
    request = DerivativeExecutionRequest(node, args, derivative_order=order, derivative_kind=kind,
        method=method or node.diff_method, tangent=tangent, limits=node.unified_limits)
    prepared = None
    def prepare(device, request):
        nonlocal prepared
        limits = resolve_higher_order_limits(owner.higher_order_limits)
        if owner.max_diff not in {1, 2} or order == 2 and owner.max_diff != 2:
            raise UnsupportedFeature("higher_order_derivatives", "Second derivatives require max_diff=2; orders above two are not implemented.")
        positions, names = argnums, argnames
        if isinstance(positions, str): names, positions = (positions,), ()
        if order and positions == 0 and not args and not names:
            import inspect
            name = next(iter(tuple(inspect.signature(owner.fn).parameters)[1:]), None)
            if name not in kwargs: raise ValueError("No first trainable QNode argument was supplied.")
            names, positions = (name,), ()
        direction = None
        if tangent is not None:
            selected_positions = (positions,) if isinstance(positions, int) else tuple(positions)
            selected_names = (names,) if isinstance(names, str) else tuple(names)
            selected = [args[index] for index in selected_positions]+[kwargs[name] for name in selected_names]
            one = isinstance(positions, int) and not names or not selected_positions and len(selected_names) == 1
            tangents = [tangent] if one else list(tangent)
            if len(tangents) != len(selected) or any(np.shape(a) != np.shape(b) for a, b in zip(tangents, selected)):
                raise ValueError("HVP tangent shapes must match selected trainable arguments.")
            if any(np.asarray(value).dtype.kind not in "fiu" or not np.all(np.isfinite(value)) for value in tangents):
                raise ValueError("HVP tangent must be finite and real.")
            direction = np.concatenate([np.asarray(value, dtype=float).reshape(-1) for value in tangents])
        if order:
            traced, traced_kwargs, layout, single, size = trace_arguments(args, kwargs, positions, names,
                order=order, direction=direction, limits=limits)
        else:
            traced, traced_kwargs, layout, single, size = args, kwargs, [], True, 0
        plan = owner._construct(traced, traced_kwargs)
        count = len(plan.recorder.parameters)
        columns = size if direction is None else 1
        scope = current_resources()
        map_bytes = 8*count*size*(1+(columns if order == 2 else 0))
        scope.retain("classical_source_maps", map_bytes)
        classical = plan.recorder.classical_jacobian(size) if order else np.zeros((count, 0))
        second = plan.recorder.classical_second(size, columns) if order == 2 else None
        tree = plan.source_tree if hasattr(plan, "source_tree") else plan.tree
        shape = _shape_tree(tree, owner.num_qubits, ShotSpec.parse(owner.shots), limits, "native_qnode")
        flat, spec = flatten_tree(shape)
        prepared = finish_derivative(request, owner, plan, plan.recorder.native_values(), classical,
            second, direction, size, layout, single, spec, len(flat))
        prepared.buffer_bytes -= map_bytes
        return prepared
    try:
        result = execute_device_derivatives(node.device, request, _prepare=prepare)
    except Exception as error:
        if hasattr(error, "receipt"):
            node.last_metadata = {"unified_resource_receipt": error.receipt}
        raise
    node.last_metadata = {**result.metadata, "unified_resource_receipt": result.receipt}
    return result, prepared
