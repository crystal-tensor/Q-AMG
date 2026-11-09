# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Immutable requests for native numerical plans, not upstream SDK dispatch."""

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType

from .resource_limits import UnifiedExecutionLimits, resolve_unified_limits


@dataclass(frozen=True)
class DerivativeExecutionRequest:
    circuit_or_tape: object
    parameters: object = ()
    derivative_order: int = 1
    method: str = "best"
    derivative_kind: str = "jacobian"
    cotangent: object = None
    tangent: object = None
    shots: object = None
    seed: object = None
    limits: UnifiedExecutionLimits | None = None
    observable: object = None

    def validate_order(self):
        orders = {"value": 0, "gradient": 1, "jacobian": 1, "vjp": 1, "jvp": 1, "hessian": 2, "hvp": 2}
        if self.derivative_kind not in orders:
            raise ValueError("Unknown native derivative kind.")
        if isinstance(self.derivative_order, bool) or not isinstance(self.derivative_order, Integral) or self.derivative_order != orders[self.derivative_kind]:
            raise ValueError("Derivative order must match derivative kind.")
        return int(self.derivative_order)

    def describe(self, *, inputs, outputs, directions, cotangents, parameters, plan_identity):
        """Immutable numerical contract; no live source or framework payload."""
        order = self.validate_order()
        counts = dict(input_count=inputs, output_count=outputs, direction_count=directions, cotangent_count=cotangents)
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in counts.values()):
            raise ValueError("Derivative contract counts require nonnegative integers.")
        limits = None if self.limits is None else tuple(sorted((name, int(value)) for name, value in resolve_unified_limits(self.limits).__dict__.items()))
        return MappingProxyType(dict(operation=self.derivative_kind, order=order, method=self.method,
            **{name: int(value) for name, value in counts.items()},
            parameter_identity=tuple(str(value) for value in parameters), plan_identity=plan_identity, resource_limits=limits))
