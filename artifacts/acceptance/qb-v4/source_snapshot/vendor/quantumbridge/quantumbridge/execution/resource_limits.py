# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Owned numerical-resource limits, not an operating-system RSS sandbox."""

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class UnifiedExecutionLimits:
    max_total_memory_bytes: int = 256 * 1024 * 1024
    max_total_tapes: int = 20000
    max_total_shots: int = 1000000
    max_derivative_work: int = 50000000

    def __post_init__(self):
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0
               for value in self.__dict__.values()):
            raise ValueError("Unified limits require nonnegative integers.")


def resolve_unified_limits(limits):
    if limits is None:
        return UnifiedExecutionLimits()
    if not isinstance(limits, UnifiedExecutionLimits):
        raise TypeError("Unified derivative execution requires UnifiedExecutionLimits.")
    return limits
