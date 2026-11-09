# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""One request account across ordinary eager framework quantum callbacks."""

import numpy as np

from quantumbridge.diff.higher_order_limits import resolve_higher_order_limits
from quantumbridge.diff.trace import flatten_tree
from quantumbridge.execution import DerivativeExecutionRequest, UnifiedResourceLedger
from quantumbridge.execution.derivative_planning import finish_derivative
from quantumbridge.execution.device_derivatives import execute_device_derivatives
from quantumbridge.execution.resource_context import ResourceScope, current_resources
from quantumbridge.execution.array_reservations import scope_array
from quantumbridge.qml.shot_vector import ShotSpec


def managed_framework_plan(plan):
    limits = getattr(plan.node, "unified_limits", None)
    parent = current_resources()
    if limits is None and parent is None:
        return plan
    return _ManagedFrameworkPlan(plan, limits, None if parent is None else parent.ledger)


class _ManagedFrameworkPlan:
    """Delegate numerical work; retain only this graph invocation's account.

    This is eager Torch/Autograd callback ownership, not a mutable ledger
    captured by pure JAX callbacks or a claim about framework allocator memory.
    """

    def __init__(self, plan, limits, parent):
        self.plan = plan
        self.ledger = UnifiedResourceLedger(limits, parent=parent)
        from .native_host import _shape_tree
        tree = plan.source_tree if hasattr(plan, "source_tree") else plan.tree
        shape = _shape_tree(tree, plan.node.num_qubits, ShotSpec.parse(plan.node.shots),
            resolve_higher_order_limits(plan.node.higher_order_limits), "ordinary_framework")
        flat, self.output_spec = flatten_tree(shape)
        self.output_size = len(flat)
        self.ledger.unavailable("framework_allocator_memory", "Framework graphs and retained framework tensors are outside the native numerical-array account.")

    def __getattr__(self, name):
        return getattr(self.plan, name)

    def _publish(self):
        receipt = self.ledger.receipt()
        self.plan.metadata["unified_resource_receipt"] = receipt
        self.plan.node.last_metadata = {**self.plan.metadata, "unified_resource_receipt": receipt}

    def _run(self, values, kind, method=None, *, direction=None, classical=None, classical_second=None):
        order = 0 if kind == "value" else 2 if kind in {"hessian", "hvp"} else 1
        count = len(self.plan.recorder.parameters)
        size = count if classical is None else classical.shape[1]
        columns = 1 if direction is not None else size
        request = DerivativeExecutionRequest(self.plan.node, (), derivative_kind=kind,
            derivative_order=order, method=method or self.plan.node.diff_method, tangent=direction)
        try:
            with ResourceScope(self.ledger) as scope:
                if classical is None:
                    classical = scope_array(scope, "framework_identity_map", (count, size), float,
                        lambda: np.eye(count))
                if order == 2 and classical_second is None:
                    classical_second = scope_array(scope, "framework_zero_second_map", (count, size, columns), float,
                        lambda: np.zeros((count, size, columns)))
                def prepare(device, request):
                    prepared = finish_derivative(request, self.plan.node, self.plan, np.asarray(values),
                        classical, classical_second, direction, size,
                        [((size,), slice(0, size))], True, self.output_spec, self.output_size)
                    # Source maps are already owned by the surrounding callback
                    # scope; the retained result lease must not count them again.
                    prepared.buffer_bytes -= classical.nbytes
                    if order == 2: prepared.buffer_bytes -= classical_second.nbytes
                    return prepared
                result = execute_device_derivatives(self.plan.node.device, request,
                    ledger=self.ledger, _prepare=prepare)
            self._publish()
            return result
        except Exception as error:
            self._publish()
            if hasattr(error, "receipt"):
                error.receipt = self.ledger.receipt()
            raise

    def evaluate(self, values, *args, **kwargs):
        if args or kwargs:
            raise ValueError("Managed outer framework forward requires its unshifted plan evaluation.")
        return self._run(values, "value").value

    def jacobian(self, values, method=None, **kwargs):
        if set(kwargs)-{"step"}:
            raise ValueError("Unsupported managed native Jacobian options.")
        result = self._run(values, "jacobian", method)
        return result.value, result.jacobian, self.output_spec

    def second_derivatives(self, values, method=None, *, direction=None, classical=None, classical_second=None, limits=None):
        result = self._run(values, "hessian" if direction is None else "hvp", method,
            direction=direction, classical=classical, classical_second=classical_second)
        return result.value, result.jacobian, result.second, self.output_spec
