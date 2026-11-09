# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_upstream_integration_strategy_v0.1.md.

from __future__ import annotations

import numpy as np

from quantumbridge.algorithms.vqe import run_vqe
from quantumbridge.utils.math import Hamiltonian, PauliString

from .qubit_hamiltonian import QubitHamiltonian
from .result import ChemistryResult


def _as_matrix(hamiltonian) -> np.ndarray:
    if isinstance(hamiltonian, QubitHamiltonian):
        return hamiltonian.matrix()
    if isinstance(hamiltonian, Hamiltonian):
        return QubitHamiltonian(hamiltonian).matrix()
    return np.asarray(hamiltonian, dtype=complex)


class ExactDiagonalizationSolver:
    mode = "Native Core"

    def solve(self, hamiltonian, molecule=None, nuclear_repulsion_energy: float = 0.0) -> ChemistryResult:
        matrix = _as_matrix(hamiltonian)
        values = np.linalg.eigvalsh(matrix)
        electronic = float(np.min(np.real(values)))
        return ChemistryResult(
            molecule=molecule,
            solver="ExactDiagonalizationSolver",
            electronic_energy=electronic,
            nuclear_repulsion_energy=float(nuclear_repulsion_energy),
            total_energy=electronic + float(nuclear_repulsion_energy),
            num_qubits=int(np.log2(matrix.shape[0])),
            provenance={"mode": "Native Core", "component": "quantumbridge.chemistry.solvers"},
        )


class NumPyMinimumEigensolver:
    mode = "Native Core"

    def compute_minimum_eigenvalue(self, operator):
        return ExactDiagonalizationSolver().solve(operator)


class VQEChemistrySolver:
    mode = "Native Core"

    def __init__(self, ansatz, initial_parameters, steps: int = 40):
        self.ansatz = ansatz
        self.initial_parameters = initial_parameters
        self.steps = steps

    def solve(self, hamiltonian, molecule=None, nuclear_repulsion_energy: float = 0.0) -> ChemistryResult:
        qb_hamiltonian = hamiltonian.to_quantumbridge() if isinstance(hamiltonian, QubitHamiltonian) else hamiltonian
        result = run_vqe(qb_hamiltonian, self.ansatz, self.initial_parameters, steps=self.steps)
        electronic = float(result.metadata()["final_energy"])
        return ChemistryResult(
            molecule=molecule,
            solver="VQEChemistrySolver",
            electronic_energy=electronic,
            nuclear_repulsion_energy=float(nuclear_repulsion_energy),
            total_energy=electronic + float(nuclear_repulsion_energy),
            convergence_history=result.trace or [],
            optimal_parameters=result.metadata().get("final_parameters"),
            provenance={"mode": "Native Core", "component": "quantumbridge.chemistry.solvers"},
        )


def minimal_h2_qubit_hamiltonian() -> QubitHamiltonian:
    return QubitHamiltonian(
        [
            (-1.052373245772859, PauliString([])),
            (0.39793742484318045, PauliString([(0, "Z")])),
            (-0.39793742484318045, PauliString([(1, "Z")])),
            (-0.01128010425623538, PauliString([(0, "Z"), (1, "Z")])),
            (0.18093119978423156, PauliString([(0, "X"), (1, "X")])),
        ]
    )
