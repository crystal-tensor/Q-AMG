# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.
# Design source: docs/mvp/algorithm_design_v0.1.md and docs/mvp/simulator_design_v0.1.md.

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from quantumbridge.core import Circuit
from quantumbridge.devices import StatevectorDevice
from quantumbridge.results import Result
from quantumbridge.utils.math import Hamiltonian, PauliString


def maxcut_hamiltonian(edges: Iterable[tuple[int, int]]) -> Hamiltonian:
    terms: list[tuple[float, PauliString]] = []
    for a, b in edges:
        if a == b:
            raise ValueError("QuantumBridge QAOA MaxCut edges cannot be self-loops.")
        terms.append((0.5, PauliString([])))
        terms.append((-0.5, PauliString([(a, "Z"), (b, "Z")])))
    return Hamiltonian(terms)


def qaoa_circuit(num_qubits: int, edges: Sequence[tuple[int, int]], depth: int, parameters: Sequence[float]) -> Circuit:
    if len(parameters) != 2 * depth:
        raise ValueError("QuantumBridge QAOA parameters must contain one gamma and one beta per layer.")
    circuit = Circuit(num_qubits)
    for wire in range(num_qubits):
        circuit.h(wire)
    gammas = parameters[:depth]
    betas = parameters[depth:]
    for gamma, beta in zip(gammas, betas):
        for a, b in edges:
            circuit.cx(a, b)
            circuit.rz(-float(gamma), b)
            circuit.cx(a, b)
        for wire in range(num_qubits):
            circuit.rx(2 * float(beta), wire)
    return circuit


def run_qaoa(
    graph_edges: Sequence[tuple[int, int]],
    depth: int,
    initial_parameters: Sequence[float],
    device: Optional[StatevectorDevice] = None,
    steps: int = 40,
    step_size: float = 0.25,
) -> Result:
    """Run a minimal coordinate-search QAOA loop for unweighted MaxCut."""

    if depth <= 0:
        raise ValueError("QuantumBridge QAOA depth must be positive.")
    edges = tuple((int(a), int(b)) for a, b in graph_edges)
    num_qubits = 1 + max(max(edge) for edge in edges)
    device = device or StatevectorDevice()
    hamiltonian = maxcut_hamiltonian(edges)
    params = np.array(initial_parameters, dtype=float)
    if len(params) != 2 * depth:
        raise ValueError("QuantumBridge QAOA initial parameters must match depth.")

    def objective(values: np.ndarray) -> float:
        return device.expectation(qaoa_circuit(num_qubits, edges, depth, values), hamiltonian)

    best_value = float(objective(params))
    best_params = params.copy()
    trace: list[dict] = []
    current_step = float(step_size)
    for step in range(steps):
        improved = False
        for index in range(len(params)):
            for direction in (1.0, -1.0):
                candidate = best_params.copy()
                candidate[index] += direction * current_step
                value = float(objective(candidate))
                if value > best_value:
                    best_value = value
                    best_params = candidate
                    improved = True
        trace.append({"step": step, "value": best_value, "parameters": best_params.tolist(), "step_size": current_step})
        if not improved:
            current_step *= 0.5
        if current_step < 1e-4:
            break

    return Result(
        expectation_data=best_value,
        metadata_data={
            "algorithm": "QAOA",
            "objective": "MaxCut expected cut value",
            "final_value": best_value,
            "final_parameters": best_params.tolist(),
            "depth": depth,
            "graph_edges": list(edges),
            "steps_completed": len(trace),
        },
        trace=trace,
    )
