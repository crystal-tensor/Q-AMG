# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Qiskit Experiments was copied.
"""Educational offline T1 experiment workflow."""

from __future__ import annotations

import math
from typing import Iterable

from quantumbridge.schema.experiments_results import T1ExperimentResult

from .calibration_fit_native import deterministic_noise, goodness, linear_space, sse
from .experiment_data_native import NativeExperimentData
from .warnings import experiments_warnings, native_provenance


def generate_t1_synthetic_data(
    time_points: Iterable[float] | None = None,
    t1: float = 50.0,
    amplitude: float = 1.0,
    offset: float = 0.0,
    noise: float = 0.0,
    seed: int | None = None,
) -> NativeExperimentData:
    points = list(time_points) if time_points is not None else linear_space(0.0, 120.0, 41)
    y_values = [
        float(offset + amplitude * math.exp(-point / t1) + deterministic_noise(index, noise, seed))
        for index, point in enumerate(points)
    ]
    return NativeExperimentData(
        "t1",
        "time",
        [float(point) for point in points],
        "population",
        y_values,
        {"t1": float(t1), "amplitude": float(amplitude), "offset": float(offset), "noise": float(noise), "seed": seed},
    )


def fit_t1_decay_native(data: NativeExperimentData | dict) -> dict[str, float]:
    payload = data.to_dict() if hasattr(data, "to_dict") else dict(data)
    x_values = [float(value) for value in payload["x_values"]]
    y_values = [float(value) for value in payload["y_values"]]
    offset = min(y_values)
    amplitude = max(1e-12, max(y_values) - offset)
    candidates = [5.0 + index * 1.0 for index in range(196)]
    best_t1 = min(
        candidates,
        key=lambda candidate: sse(
            y_values,
            [offset + amplitude * math.exp(-x / candidate) for x in x_values],
        ),
    )
    predicted = [offset + amplitude * math.exp(-x / best_t1) for x in x_values]
    metrics = goodness(y_values, predicted)
    return {"t1": float(best_t1), "amplitude": float(amplitude), "offset": float(offset), **metrics}


def run_t1_experiment_native(**kwargs) -> T1ExperimentResult:
    data = generate_t1_synthetic_data(**kwargs)
    fit = fit_t1_decay_native(data)
    return T1ExperimentResult(
        workflow="t1_experiment_native",
        mode="native_minimal",
        capability_level=3,
        native_implementation=True,
        production_ready=False,
        hardware_calibration=False,
        input_summary=data.metadata,
        data=data.to_dict(),
        fit_parameters=fit,
        time_points=data.x_values,
        raw_type="NativeExperimentData",
        metadata={"educational": True, "offline_only": True},
        warnings=experiments_warnings(),
        provenance=native_provenance("t1_experiment_native"),
    )
