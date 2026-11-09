# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class QChemLimits:
    max_atoms: int = 8
    max_basis_functions: int = 8
    max_spin_orbitals: int = 16
    max_integral_tensor_bytes: int = 4 * 1024 * 1024
    max_scf_iterations: int = 128
    max_hamiltonian_terms: int = 100000

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in self.__dict__.values()):
            raise ValueError("Chemistry resource limits must be positive integers.")

    def check(self, name, value):
        if value > getattr(self, name):
            raise MemoryError("Native chemistry exceeded " + name)
