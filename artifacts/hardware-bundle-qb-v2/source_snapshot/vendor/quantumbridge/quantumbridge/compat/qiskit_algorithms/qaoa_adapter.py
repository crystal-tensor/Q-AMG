# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Algorithms was copied.
"""Qiskit Algorithms QAOA executable adapter."""

from quantumbridge.compat.qiskit_algorithms.algorithms_native import (
    build_maxcut_problem_native,
    maxcut_objective_value,
    qaoa_cost_hamiltonian_metadata,
    run_qaoa_native_maxcut,
    run_qaoa_upstream,
    solve_maxcut_bruteforce_native,
)

__all__ = [
    "build_maxcut_problem_native",
    "maxcut_objective_value",
    "qaoa_cost_hamiltonian_metadata",
    "run_qaoa_native_maxcut",
    "run_qaoa_upstream",
    "solve_maxcut_bruteforce_native",
]
