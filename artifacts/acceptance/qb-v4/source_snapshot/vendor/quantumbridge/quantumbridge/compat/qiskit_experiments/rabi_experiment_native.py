# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Qiskit Experiments was copied.
"""Educational offline Rabi experiment workflow."""

from __future__ import annotations

import math
from typing import Iterable

from quantumbridge.schema.experiments_results import RabiExperimentResult

from .calibration_fit_native import deterministic_noise, goodness, linear_space, sse
from .experiment_data_native import NativeExperimentData
from .warnings import experiments_warnings, native_provenance


def generate_rabi_synthetic_data(
    amplitude_points: Iterable[float] | None = None,
    frequency: float = 1.0,
    contrast: float = 0.45,
    offset: float = 0.5,
    noise: float = 0.0,
    seed: int | None = None,
) -> NativeExperimentData:
    points = list(amplitude_points) if amplitude_points is not None else linear_space(0.0, 1.0, 41)
    y_values = [
        float(offset + contrast * math.cos(2.0 * math.pi * frequency * point) + deterministic_noise(index, noise, seed))
        for index, point in enumerate(points)
    ]
    return NativeExperimentData(
        "rabi",
        "amplitude",
        [float(point) for point in points],
        "population",
        y_values,
        {
            "frequency": float(frequency),
            "contrast": float(contrast),
            "offset": float(offset),
            "noise": float(noise),
            "seed": seed,
            "hardware_calibration": False,
        },
    )


def fit_rabi_oscillation_native(data: NativeExperimentData | dict) -> dict[str, float]:
    payload = data.to_dict() if hasattr(data, "to_dict") else dict(data)
    x_values = [float(value) for value in payload["x_values"]]
    y_values = [float(value) for value in payload["y_values"]]
    offset = sum(y_values) / len(y_values)
    contrast = max(1e-12, (max(y_values) - min(y_values)) / 2.0)
    candidates = [0.25 + index * 0.025 for index in range(121)]
    best_frequency = min(
        candidates,
        key=lambda freq: sse(
            y_values,
            [offset + contrast * math.cos(2.0 * math.pi * freq * x) for x in x_values],
        ),
    )
    predicted = [offset + contrast * math.cos(2.0 * math.pi * best_frequency * x) for x in x_values]
    metrics = goodness(y_values, predicted)
    return {
        "frequency": float(best_frequency),
        "contrast": float(contrast),
        "offset": float(offset),
        "pi_amplitude": float(0.5 / best_frequency),
        **metrics,
    }


def run_rabi_experiment_native(**kwargs) -> RabiExperimentResult:
    data = generate_rabi_synthetic_data(**kwargs)
    fit = fit_rabi_oscillation_native(data)
    return RabiExperimentResult(
        workflow="rabi_experiment_native",
        mode="native_minimal",
        capability_level=3,
        native_implementation=True,
        production_ready=False,
        hardware_calibration=False,
        input_summary=data.metadata,
        data=data.to_dict(),
        fit_parameters=fit,
        raw_type="NativeExperimentData",
        metadata={"educational": True, "offline_only": True},
        warnings=experiments_warnings(),
        provenance=native_provenance("rabi_experiment_native"),
    )
