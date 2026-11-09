# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Framework graphs use the same native quantum Jacobian, not an SDK oracle.

import numpy as np
from contextlib import contextmanager, nullcontext
from functools import lru_cache

from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.diff.trace import flatten_tree, unflatten_tree


def _leaves(value):
    if isinstance(value, dict): return sum((_leaves(item) for item in value.values()), [])
    if isinstance(value, (tuple, list)): return sum((_leaves(item) for item in value), [])
    return [value]


def infer_interface(args, kwargs):
    modules = [type(value).__module__ for value in _leaves((args, kwargs))]
    if any(module.startswith("torch") for module in modules): return "torch"
    if any(module.startswith("autograd") for module in modules): return "autograd"
    if any(module.startswith(("jax", "jaxlib")) for module in modules): return "jax"
    if any(module.startswith("tensorflow") for module in modules): return "tf"
    return None


def execute_interface(plan, interface):
    if interface == "torch": return _torch_execute(plan)
    if interface == "autograd": return _autograd_execute(plan)
    raise UnsupportedFeature("qnode_interface", f"Native {interface} autodiff is not implemented/verified; do not treat array conversion as autodiff.")


class _FrameworkCallbacks:
    """One graph's native snapshot and existing-plan callback operations."""

    def __init__(self, plan, indices, interface):
        from quantumbridge.diff.scalars import scalar_primal
        self.plan, self.indices, self.interface = plan, tuple(indices), interface
        self.ledger = getattr(plan, "ledger", None)
        self.fixed = self.array("framework_fixed_snapshot", (len(plan.recorder.angles),),
            lambda: np.fromiter((scalar_primal(angle) for angle in plan.recorder.angles),
                dtype=float, count=len(plan.recorder.angles)))

    def array(self, category, shape, factory=None):
        from quantumbridge.execution.array_ownership import retained_array
        return retained_array(self.ledger, category, shape, float, factory)

    def publish(self, second=False):
        if self.ledger is not None: self.plan._publish()
        self.plan.node.last_metadata = {**self.plan.metadata, "interface": self.interface,
            "native_vjp": True, "native_second_order": second,
            "classical_chain_rule": "Torch graph" if self.interface == "torch" else "Autograd graph"}

    @contextmanager
    def scope(self, second=False):
        from quantumbridge.execution.resource_context import ResourceScope
        try:
            with ResourceScope(self.ledger) if self.ledger is not None else nullcontext():
                yield
        finally:
            self.publish(second)

    def real(self, value):
        if not isinstance(value, np.ndarray):
            if not isinstance(value, (np.generic, float, int)) or isinstance(value, (bool, complex)):
                raise TypeError("Framework callbacks require numeric scalars or ndarray buffers.")
            value = self.array("framework_scalar_conversion", (), lambda: np.array(value, dtype=float))
        if value.dtype.kind not in "fiu" or not np.all(np.isfinite(value)):
            raise ValueError("Native framework callbacks require finite real values.")
        if value.dtype == np.dtype(float): return value
        return self.array("framework_dtype_conversion", value.shape,
            lambda: np.array(value, dtype=float, copy=True))

    def unpack(self, values):
        values = self.real(values)
        if values.shape != (len(self.indices),):
            raise ValueError("Framework packed trainable occurrences have the wrong shape.")
        point = self.array("framework_unpacked_point", self.fixed.shape, self.fixed.copy)
        for index, value in zip(self.indices, values): point[index] = value
        return point

    def forward(self, values):
        with self.scope():
            result = self.plan.evaluate(self.unpack(values))
            flat, self.plan.last_spec = flatten_tree(result)
            return flat

    def selected_jacobian(self, values):
        _, matrix, _ = self.plan.jacobian(self.unpack(values), **self.plan.node.gradient_kwargs)
        selected = self.array("framework_selected_jacobian", (matrix.shape[0], len(self.indices)))
        for column, index in enumerate(self.indices): selected[:, column] = matrix[:, index]
        return selected

    def vjp(self, values, cotangent):
        from quantumbridge.execution.array_reservations import matmul_into
        from quantumbridge.execution.resource_context import current_resources
        with self.scope():
            matrix = self.selected_jacobian(values)
            cotangent = self.real(cotangent)
            # The returned array can be retained by an Autograd graph; its lease
            # follows storage lifetime instead of this callback's scope.
            result = self.array("framework_vjp_result", (len(self.indices),))
            matmul_into(cotangent, matrix, result, scope=current_resources())
            return result

    def jvp(self, values, direction):
        from quantumbridge.execution.array_reservations import matmul_into
        from quantumbridge.execution.resource_context import current_resources
        with self.scope():
            matrix, direction = self.selected_jacobian(values), self.real(direction)
            if direction.shape != (len(self.indices),):
                raise ValueError("Framework JVP direction has the wrong shape.")
            result = self.array("framework_jvp_result", (matrix.shape[0],))
            matmul_into(matrix, direction, result, scope=current_resources())
            return result

    def _second_direction(self, values, direction):
        from quantumbridge.diff.higher_order_limits import resolve_higher_order_limits
        limits = resolve_higher_order_limits(self.plan.node.higher_order_limits)
        count = len(self.indices)
        limits.check("max_input_parameters", count)
        limits.check("max_memory_bytes", 16*len(self.fixed)*count)
        direction = self.real(direction)
        if direction.shape != (count,): raise ValueError("Framework HVP direction has the wrong shape.")
        classical = self.array("framework_selected_source_map", (len(self.fixed), count),
            lambda: np.zeros((len(self.fixed), count)))
        for column, index in enumerate(self.indices): classical[index, column] = 1.
        _, matrix, product, _ = self.plan.second_derivatives(self.unpack(values),
            direction=direction, classical=classical, limits=limits)
        return direction, matrix, product

    def second_vjp(self, values, cotangent, direction):
        from quantumbridge.execution.array_reservations import matmul_into
        from quantumbridge.execution.resource_context import current_resources
        with self.scope(True):
            count = len(self.indices)
            cotangent = self.real(cotangent)
            direction, matrix, product = self._second_direction(values, direction)
            result = self.array("framework_second_vjp_result", (count+matrix.shape[0],))
            matmul_into(cotangent, product[:, :, 0], result[:count], scope=current_resources())
            matmul_into(matrix, direction, result[count:], scope=current_resources())
            return result

    def second_jvp(self, values, left, right):
        from quantumbridge.execution.array_reservations import matmul_into
        from quantumbridge.execution.resource_context import current_resources
        with self.scope(True):
            left = self.real(left)
            if left.shape != (len(self.indices),):
                raise ValueError("Framework mixed JVP direction has the wrong shape.")
            _, matrix, product = self._second_direction(values, right)
            result = self.array("framework_second_jvp_result", (matrix.shape[0],))
            matmul_into(product[:, :, 0], left, result, scope=current_resources())
            return result


def _torch_execute(plan):
    try: import torch
    except ImportError as exc: raise ImportError("QuantumBridge Torch QNode requires optional torch.") from exc
    angles = plan.recorder.angles
    inputs = [value for value in _leaves(plan.input_values) if isinstance(value, torch.Tensor)]
    tensors = [value for value in angles if isinstance(value, torch.Tensor)]
    if any(value.device.type != "cpu" for value in tensors + inputs): raise UnsupportedFeature("torch_device", "Native QNode Torch execution currently supports CPU tensors only.")
    if any(value.ndim != 0 or not value.is_floating_point() for value in tensors): raise TypeError("Quantum gate angles must be real floating-point scalar tensors.")
    dtype = tensors[0].dtype if tensors else torch.float64
    if any(value.dtype != dtype for value in tensors): raise TypeError("QuantumBridge gate tensors must use one floating dtype.")
    def active(value):
        return isinstance(value, torch.Tensor) and (value.requires_grad
            or torch.autograd.forward_ad.unpack_dual(value).tangent is not None)
    indices = [index for index, value in enumerate(angles) if active(value)]
    from .ordinary_resources import managed_framework_plan
    plan = managed_framework_plan(plan)
    callbacks = _FrameworkCallbacks(plan, indices, "torch")
    packed = torch.stack([angles[index] for index in indices]) if indices else torch.empty(0, dtype=dtype)
    for value in inputs:
        if active(value): packed = packed + 0 * value.sum()

    flat = _torch_primitives().apply(packed, callbacks)
    return unflatten_tree(flat, plan.last_spec)


@lru_cache(maxsize=1)
def _torch_primitives():
    import torch

    class NativeSecondVJP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, values, cotangent, incoming, owner):
            result = owner.second_vjp(values.detach().numpy(), cotangent.detach().numpy(), incoming.detach().numpy())
            count = len(owner.indices)
            return torch.tensor(result[:count], dtype=values.dtype), torch.tensor(result[count:], dtype=cotangent.dtype)

        @staticmethod
        def backward(ctx, *unused):
            raise UnsupportedFeature("higher_order_derivatives", "Torch native third derivatives are not implemented.")

        @staticmethod
        def jvp(ctx, *unused):
            raise UnsupportedFeature("higher_order_derivatives", "Torch native third derivatives are not implemented.")

    class NativeSecondJVP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, values, left, right, owner):
            result = owner.second_jvp(values.detach().numpy(), left.detach().numpy(), right.detach().numpy())
            return torch.tensor(result, dtype=values.dtype)

        @staticmethod
        def backward(ctx, *unused):
            raise UnsupportedFeature("higher_order_derivatives", "Torch native third derivatives are not implemented.")

        @staticmethod
        def jvp(ctx, *unused):
            raise UnsupportedFeature("higher_order_derivatives", "Torch native third derivatives are not implemented.")

    class NativeQuantumVJP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, values, cotangent, owner):
            ctx.save_for_backward(values, cotangent)
            ctx.save_for_forward(values, cotangent)
            ctx.owner = owner
            result = owner.vjp(values.detach().numpy(), cotangent.detach().numpy())
            return torch.tensor(result, dtype=values.dtype)

        @staticmethod
        def backward(ctx, incoming):
            first, second = NativeSecondVJP.apply(*ctx.saved_tensors, incoming, ctx.owner)
            return first, second, None

        @staticmethod
        def jvp(ctx, values_tangent, cotangent_tangent, unused):
            if ctx.owner.plan.node.max_diff != 2:
                raise UnsupportedFeature("higher_order_derivatives", "Torch mixed second derivatives require max_diff=2.")
            values, cotangent = ctx.saved_tensors
            result = NativeSecondVJP.apply(values, cotangent, values_tangent, ctx.owner)[0]
            if cotangent_tangent is not None:
                result = result+NativeQuantumVJP.apply(values, cotangent_tangent, ctx.owner)
            return result

    class NativeQuantumJVP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, values, direction, owner):
            ctx.save_for_backward(values, direction)
            ctx.save_for_forward(values, direction)
            ctx.owner = owner
            return torch.tensor(owner.jvp(values.detach().numpy(), direction.detach().numpy()), dtype=values.dtype)

        @staticmethod
        def backward(ctx, cotangent):
            if ctx.owner.plan.node.max_diff != 2:
                raise UnsupportedFeature("higher_order_derivatives", "Torch mixed second derivatives require max_diff=2.")
            values, direction = ctx.saved_tensors
            first = NativeSecondVJP.apply(values, cotangent, direction, ctx.owner)[0]
            second = NativeQuantumVJP.apply(values, cotangent, ctx.owner)
            return first, second, None

        @staticmethod
        def jvp(ctx, values_tangent, direction_tangent, unused):
            if ctx.owner.plan.node.max_diff != 2:
                raise UnsupportedFeature("higher_order_derivatives", "Torch mixed second derivatives require max_diff=2.")
            values, direction = ctx.saved_tensors
            result = NativeSecondJVP.apply(values, values_tangent, direction, ctx.owner)
            if direction_tangent is not None:
                result = result+NativeQuantumJVP.apply(values, direction_tangent, ctx.owner)
            return result

    class NativeQuantumFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, values, owner):
            ctx.save_for_backward(values)
            ctx.save_for_forward(values)
            ctx.owner = owner
            return torch.tensor(owner.forward(values.detach().numpy()), dtype=values.dtype)

        @staticmethod
        def backward(ctx, cotangent):
            owner = ctx.owner
            if owner.plan.node.max_diff not in {1, 2}: raise UnsupportedFeature("higher_order_derivatives")
            if torch.is_grad_enabled():
                if owner.plan.node.max_diff != 2:
                    raise UnsupportedFeature("higher_order_derivatives", "Torch higher-order second derivatives require max_diff=2.")
                return NativeQuantumVJP.apply(ctx.saved_tensors[0], cotangent, owner), None
            result = owner.vjp(ctx.saved_tensors[0].detach().numpy(), cotangent.detach().numpy())
            return torch.tensor(result, dtype=cotangent.dtype), None

        @staticmethod
        def jvp(ctx, values_tangent, unused):
            if ctx.owner.plan.node.max_diff not in {1, 2}:
                raise UnsupportedFeature("higher_order_derivatives")
            return NativeQuantumJVP.apply(ctx.saved_tensors[0], values_tangent, ctx.owner)

    return NativeQuantumFunction


@lru_cache(maxsize=1)
def _autograd_primitives():
    """Register generic rules once; invocation owners live only in real graphs."""
    try:
        from autograd.extend import primitive, defvjp, defvjp_argnums, defjvp
    except ImportError as exc: raise ImportError("QuantumBridge Autograd QNode requires optional autograd.") from exc

    @primitive
    def native_second_vjp(values, cotangent, direction, owner):
        return owner.second_vjp(values, cotangent, direction)

    def reject_third(*unused):
        raise UnsupportedFeature("higher_order_derivatives", "Autograd native third derivatives are not implemented.")
    defvjp(native_second_vjp, reject_third, reject_third, reject_third, None)
    defjvp(native_second_vjp, reject_third, reject_third, reject_third, None)

    @primitive
    def native_second_jvp(values, left, right, owner):
        return owner.second_jvp(values, left, right)

    defvjp(native_second_jvp, reject_third, reject_third, reject_third, None)
    defjvp(native_second_jvp, reject_third, reject_third, reject_third, None)

    def require_second(owner):
        if owner.plan.node.max_diff != 2:
            raise UnsupportedFeature("higher_order_derivatives", "Autograd second derivatives require max_diff=2.")

    @primitive
    def native_vjp(values, cotangent, owner):
        return owner.vjp(values, cotangent)

    def second_vjp_rule(argnums, answer, arguments, keywords):
        values, cotangent, owner = arguments
        if owner.plan.node.max_diff != 2:
            raise UnsupportedFeature("higher_order_derivatives", "Autograd second derivatives require max_diff=2.")
        def backward(direction):
            result = native_second_vjp(values, cotangent, direction, owner)
            count = len(owner.indices)
            full = (result[:count], result[count:])
            return tuple(full[index] for index in argnums)
        return backward
    defvjp_argnums(native_vjp, second_vjp_rule)

    def vjp_values_jvp(direction, answer, values, cotangent, owner):
        require_second(owner)
        return native_second_vjp(values, cotangent, direction, owner)[:len(owner.indices)]

    defjvp(native_vjp, vjp_values_jvp,
        lambda direction, answer, values, cotangent, owner: native_vjp(values, direction, owner), None)

    @primitive
    def native_jvp(values, direction, owner):
        return owner.jvp(values, direction)

    def jvp_values_vjp(answer, values, direction, owner):
        require_second(owner)
        return lambda cotangent: native_second_vjp(values, cotangent, direction, owner)[:len(owner.indices)]

    defvjp(native_jvp, jvp_values_vjp,
        lambda answer, values, direction, owner: lambda cotangent: native_vjp(values, cotangent, owner), None)

    def jvp_values_jvp(tangent, answer, values, direction, owner):
        require_second(owner)
        return native_second_jvp(values, direction, tangent, owner)

    defjvp(native_jvp, jvp_values_jvp,
        lambda tangent, answer, values, direction, owner: native_jvp(values, tangent, owner), None)

    @primitive
    def native_quantum(values, owner):
        return owner.forward(values)

    def vjp_rule(answer, values, owner):
        if owner.plan.node.max_diff not in {1, 2}: raise UnsupportedFeature("higher_order_derivatives")
        if owner.plan.node.max_diff == 1 and type(values).__module__.startswith("autograd"):
            raise UnsupportedFeature("higher_order_derivatives", "Autograd higher-order native QNode derivatives are not implemented.")
        return lambda cotangent: native_vjp(values, cotangent, owner)

    defvjp(native_quantum, vjp_rule, None)
    defjvp(native_quantum,
        lambda tangent, answer, values, owner: native_jvp(values, tangent, owner), None)
    return native_quantum


def _autograd_execute(plan):
    try: import autograd.numpy as anp
    except ImportError as exc: raise ImportError("QuantumBridge Autograd QNode requires optional autograd.") from exc
    native_quantum = _autograd_primitives()
    indices = [index for index, angle in enumerate(plan.recorder.angles) if type(angle).__module__.startswith("autograd")]
    from .ordinary_resources import managed_framework_plan
    plan = managed_framework_plan(plan)
    callbacks = _FrameworkCallbacks(plan, indices, "autograd")
    packed = anp.stack([plan.recorder.angles[index] for index in indices]) if indices else anp.empty(0)
    for value in _leaves(plan.input_values):
        if type(value).__module__.startswith("autograd"): packed = packed + 0 * anp.sum(value)
    flat = native_quantum(packed, callbacks)
    return unflatten_tree(flat, plan.last_spec)
