# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class AerExecutionLimits:
    max_shots: int = 100000
    max_qubits_dense: int = 20
    max_state_bytes: int = 256*1024*1024
    max_snapshot_bytes: int = 64*1024*1024
    max_pershot_snapshot_bytes: int = 64*1024*1024
    max_snapshot_count: int = 1024
    max_conditional_groups: int = 1024
    max_saved_state_dimension: int = 1048576
    max_saved_density_dimension: int = 1024
    max_noise_channels: int = 1024
    max_noise_kraus_ops: int = 256
    max_trajectories: int = 100000
    max_dynamic_branches: int = 4096
    max_matrix_dimension: int = 1024

    def __post_init__(self):
        for value in self.__dict__.values():
            if isinstance(value,bool) or not isinstance(value,Integral) or value<1:
                raise ValueError("Aer resource budgets must be positive integers.")

    def check(self,field,value):
        if value>getattr(self,field): raise MemoryError("Native Aer exceeded "+field)
