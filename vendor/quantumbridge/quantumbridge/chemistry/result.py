# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_upstream_integration_strategy_v0.1.md.

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any


@dataclass
class DriverResult:
    one_body_integrals: Any = None
    two_body_integrals: Any = None
    nuclear_repulsion_energy: float = 0.0
    num_particles: tuple[int, int] | int | None = None
    num_spatial_orbitals: int | None = None
    num_spin_orbitals: int | None = None
    reference_energy: float | None = None
    dipole_data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "one_body_integrals": self.one_body_integrals,
            "two_body_integrals": self.two_body_integrals,
            "nuclear_repulsion_energy": self.nuclear_repulsion_energy,
            "num_particles": self.num_particles,
            "num_spatial_orbitals": self.num_spatial_orbitals,
            "num_spin_orbitals": self.num_spin_orbitals,
            "reference_energy": self.reference_energy,
            "dipole_data": self.dipole_data,
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
        }


@dataclass
class ChemistryResult:
    molecule: Any = None
    basis: str | None = None
    driver: str | None = None
    mapping: str | None = None
    ansatz: str | None = None
    solver: str | None = None
    optimizer: str | None = None
    electronic_energy: float | None = None
    nuclear_repulsion_energy: float | None = None
    total_energy: float | None = None
    reference_energy: float | None = None
    convergence_history: list[dict[str, Any]] = field(default_factory=list)
    optimal_parameters: list[float] | None = None
    num_qubits: int | None = None
    num_particles: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        molecule = self.molecule.to_xyz() if hasattr(self.molecule, "to_xyz") else self.molecule
        return {
            "molecule": molecule,
            "basis": self.basis,
            "driver": self.driver,
            "mapping": self.mapping,
            "ansatz": self.ansatz,
            "solver": self.solver,
            "optimizer": self.optimizer,
            "electronic_energy": self.electronic_energy,
            "nuclear_repulsion_energy": self.nuclear_repulsion_energy,
            "total_energy": self.total_energy,
            "reference_energy": self.reference_energy,
            "convergence_history": list(self.convergence_history),
            "optimal_parameters": self.optimal_parameters,
            "num_qubits": self.num_qubits,
            "num_particles": self.num_particles,
            "metadata": dict(self.metadata),
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
