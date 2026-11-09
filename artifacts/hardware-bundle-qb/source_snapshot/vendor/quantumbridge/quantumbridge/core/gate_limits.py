# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Logical construction/expansion budgets, independent of simulator storage."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class GateCircuitLimits:
    max_definition_depth: int = 64
    max_total_operations: int = 100000
    max_gate_parameters: int = 100000
    max_control_qubits: int = 16
    max_decomposition_work: int = 1000000
    max_repeat_count: int = 100000
    max_matrix_dimension: int = 256
    max_custom_definitions: int = 100000
    max_circuit_bits: int = 100000

    def __post_init__(self):
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 1
               for value in vars(self).values()):
            raise ValueError("Gate/circuit limits must be positive integers.")

    def check(self, name, value):
        if value > getattr(self, name):
            raise MemoryError("Gate/circuit construction exceeds " + name + ".")


_LIMITS = ContextVar("native_gate_limits", default=GateCircuitLimits())


def current_gate_limits():
    return _LIMITS.get()


@contextmanager
def gate_circuit_scope(limits):
    if not isinstance(limits, GateCircuitLimits):
        raise TypeError("Expected GateCircuitLimits.")
    parent = current_gate_limits()
    limits = GateCircuitLimits(**{key: min(value, getattr(parent, key)) for key, value in vars(limits).items()})
    token = _LIMITS.set(limits)
    try:
        yield limits
    finally:
        _LIMITS.reset(token)
