# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.

from dataclasses import dataclass
from numbers import Integral, Real
import math


@dataclass(frozen=True)
class CompilationLimits:
    max_dag_nodes: int = 100000
    max_pass_iterations: int = 100
    max_routing_swaps: int = 10000
    max_compilation_depth: int = 100000
    max_control_nesting: int = 32
    max_schedule_makespan: float = 1e9

    def __post_init__(self):
        for key, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError("Compilation limits must be finite and positive.")
            if key != "max_schedule_makespan" and not isinstance(value, Integral):
                raise ValueError("Compilation count limits must be integers.")
