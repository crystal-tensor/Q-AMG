# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Explicit budgets for native second-order quantum/classical work."""

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class HigherOrderLimits:
    max_derivative_order: int = 2
    max_input_parameters: int = 128
    max_output_elements: int = 100000
    max_hessian_elements: int = 2000000
    max_shifted_circuits: int = 10000
    max_adjoint_operations: int = 50000000
    max_total_tapes: int = 20000
    max_total_physical_shots: int = 1000000
    max_parameter_graph_nodes: int = 100000
    max_postprocessing_nodes: int = 100000
    max_memory_bytes: int = 64 * 1024 * 1024
    max_mps_tangent_components: int = 256

    def __post_init__(self):
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 1 for value in self.__dict__.values()):
            raise ValueError("Higher-order limits require positive integers.")
        if self.max_derivative_order > 2:
            raise ValueError("Native higher-order tracing currently implements orders at most two.")
        if self.max_mps_tangent_components > 4096:
            raise ValueError("Native MPS tangent components are bounded by 4096.")

    def check(self, name, count):
        if count > getattr(self, name):
            raise MemoryError("Native higher-order execution exceeded " + name)


def resolve_higher_order_limits(value):
    if value is None:
        return HigherOrderLimits()
    if not isinstance(value, HigherOrderLimits):
        raise TypeError("higher_order_limits requires HigherOrderLimits.")
    return value
