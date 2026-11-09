# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_upstream_integration_strategy_v0.1.md.

from __future__ import annotations

from .molecule import Molecule
from .result import DriverResult


class MinimalH2Driver:
    mode = "Native Core"

    def __init__(self, molecule: Molecule):
        self.molecule = molecule

    def run(self) -> DriverResult:
        return DriverResult(
            nuclear_repulsion_energy=0.7151043390810812,
            num_particles=(1, 1),
            num_spatial_orbitals=2,
            num_spin_orbitals=4,
            reference_energy=-1.137,
            metadata={"basis": self.molecule.basis, "molecule": self.molecule.to_xyz()},
            provenance={"mode": "Native Core", "component": "MinimalH2Driver"},
        )


class ExternalExecutableDriver:
    mode = "Upstream Passthrough"

    def __init__(self, name: str):
        self.name = name

    def run(self):
        raise ImportError(f"QuantumBridge {self.name} driver requires an external executable and is adapter-only/planned in P2.")
