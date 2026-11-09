# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Qiskit Experiments was copied.
"""Educational offline Ramsey experiment workflow."""

from __future__ import annotations

import math
from typing import Iterable

from quantumbridge.schema.experiments_results import RamseyExperimentResult

from .calibration_fit_native import deterministic_noise, goodness, linear_space, sse
from .experiment_data_native import NativeExperimentData
from .warnings import experiments_warnings, native_provenance


def generate_ramsey_synthetic_data(
    time_points: Iterable[float] | None = None,
    detuning: float = 0.1,
    t2star: float = 40.0,
    contrast: float = 0.4,
    offset: float = 0.5,
    noise: float = 0.0,
    seed: int | None = None,
) -> NativeExperimentData:
    points = list(time_points) if time_points is not None else linear_space(0.0, 80.0, 41)
    y_values = [
        float(
            offset
            + contrast * math.cos(2.0 * math.pi * detuning * point) * math.exp(-point / t2star)
            + deterministic_noise(index, noise, seed)
        )
        for index, point in enumerate(points)
    ]
    return NativeExperimentData(
        "ramsey",
        "time",
        [float(point) for point in points],
        "population",
        y_values,
        {
            "detuning": float(detuning),
            "t2star": float(t2star),
            "contrast": float(contrast),
            "offset": float(offset),
            "noise": float(noise),
            "seed": seed,
        },
    )


def fit_ramsey_native(data: NativeExperimentData | dict) -> dict[str, float]:
    payload = data.to_dict() if hasattr(data, "to_dict") else dict(data)
    x_values = [float(value) for value in payload["x_values"]]
    y_values = [float(value) for value in payload["y_values"]]
    offset = sum(y_values) / len(y_values)
    contrast = max(1e-12, (max(y_values) - min(y_values)) / 2.0)
    best = None
    for detuning in (0.02 + index * 0.005 for index in range(57)):
        for t2star in (10.0 + index * 2.0 for index in range(46)):
            predicted = [
                offset + contrast * math.cos(2.0 * math.pi * detuning * x) * math.exp(-x / t2star)
                for x in x_values
            ]
            error = sse(y_values, predicted)
            if best is None or error < best[0]:
                best = (error, detuning, t2star, predicted)
    assert best is not None
    _, detuning, t2star, predicted = best
    metrics = goodness(y_values, predicted)
    return {
        "detuning": float(detuning),
        "t2star": float(t2star),
        "contrast": float(contrast),
        "offset": float(offset),
        **metrics,
    }


def run_ramsey_experiment_native(**kwargs) -> RamseyExperimentResult:
    data = generate_ramsey_synthetic_data(**kwargs)
    fit = fit_ramsey_native(data)
    return RamseyExperimentResult(
        workflow="ramsey_experiment_native",
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
        provenance=native_provenance("ramsey_experiment_native"),
    )
