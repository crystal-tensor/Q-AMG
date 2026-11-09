# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.
# Design source: docs/mvp/algorithm_design_v0.1.md and docs/mvp/gradient_design_v0.1.md.

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from quantumbridge.devices import StatevectorDevice
from quantumbridge.diff import parameter_shift
from quantumbridge.results import Result


def run_vqe(
    hamiltonian,
    ansatz: Callable[[np.ndarray], object],
    initial_parameters: Sequence[float],
    device: Optional[StatevectorDevice] = None,
    steps: int = 60,
    learning_rate: float = 0.2,
    callback=None,
) -> Result:
    """Run a minimal exact-device VQE loop."""

    device = device or StatevectorDevice()
    params = np.array(initial_parameters, dtype=float)
    trace: list[dict] = []

    def objective(values: np.ndarray) -> float:
        return device.expectation(ansatz(values), hamiltonian)

    best_params = params.copy()
    best_energy = float(objective(params))
    for step in range(steps):
        energy = float(objective(params))
        grad = np.asarray(parameter_shift(objective, params), dtype=float)
        trace_entry = {
            "step": step,
            "energy": energy,
            "parameters": params.tolist(),
            "gradient": grad.tolist(),
        }
        trace.append(trace_entry)
        if callback is not None:
            callback(dict(trace_entry))
        if energy < best_energy:
            best_energy = energy
            best_params = params.copy()
        params = params - learning_rate * grad

    final_energy = float(objective(params))
    if final_energy < best_energy:
        best_energy = final_energy
        best_params = params.copy()
    return Result(
        expectation_data=final_energy,
        metadata_data={
            "algorithm": "VQE",
            "final_energy": final_energy,
            "final_parameters": params.tolist(),
            "best_energy": best_energy,
            "best_parameters": best_params.tolist(),
            "steps_completed": steps,
        },
        trace=trace,
    )
