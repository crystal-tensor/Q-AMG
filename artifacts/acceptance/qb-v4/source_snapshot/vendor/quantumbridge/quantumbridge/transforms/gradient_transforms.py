# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from numbers import Real
from typing import Callable, Sequence, Union

import numpy as np


def _as_parameter_array(parameters: Union[Sequence[Real], Real]) -> tuple[np.ndarray, bool]:
    scalar_input = isinstance(parameters, Real)
    return np.array([parameters] if scalar_input else list(parameters), dtype=float), scalar_input


def finite_difference(objective: Callable[[np.ndarray], Real], parameters: Union[Sequence[Real], Real], step: float = 1e-6):
    values, scalar_input = _as_parameter_array(parameters)
    gradient = np.zeros_like(values)
    for index in range(len(values)):
        plus = values.copy()
        minus = values.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (float(objective(plus)) - float(objective(minus))) / (2 * step)
    return float(gradient[0]) if scalar_input else gradient


def spsa_gradient(
    objective: Callable[[np.ndarray], Real],
    parameters: Union[Sequence[Real], Real],
    step: float = 1e-3,
    samples: int = 1,
    seed: int | None = None,
):
    values, scalar_input = _as_parameter_array(parameters)
    if samples <= 0:
        raise ValueError("QuantumBridge SPSA samples must be positive.")
    rng = np.random.default_rng(seed)
    gradient = np.zeros_like(values)
    for _ in range(samples):
        direction = rng.choice([-1.0, 1.0], size=len(values))
        plus = values + step * direction
        minus = values - step * direction
        gradient += ((float(objective(plus)) - float(objective(minus))) / (2 * step)) * direction
    gradient = gradient / samples
    return float(gradient[0]) if scalar_input else gradient


def metric_tensor(
    state_fn: Callable[[np.ndarray], np.ndarray],
    parameters: Union[Sequence[Real], Real],
    step: float = 1e-6,
    regularization: float = 0.0,
) -> np.ndarray:
    values, _ = _as_parameter_array(parameters)
    state = np.asarray(state_fn(values), dtype=complex)
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("QuantumBridge metric tensor requires a non-zero state.")
    state = state / norm
    derivatives = []
    for index in range(len(values)):
        plus = values.copy()
        minus = values.copy()
        plus[index] += step
        minus[index] -= step
        derivatives.append((np.asarray(state_fn(plus), dtype=complex) - np.asarray(state_fn(minus), dtype=complex)) / (2 * step))
    out = np.zeros((len(values), len(values)), dtype=float)
    for row, d_row in enumerate(derivatives):
        row_overlap = np.vdot(state, d_row)
        for col, d_col in enumerate(derivatives):
            col_overlap = np.vdot(state, d_col)
            value = np.vdot(d_row, d_col) - np.conjugate(row_overlap) * col_overlap
            out[row, col] = float(np.real(value))
    if regularization:
        out += float(regularization) * np.eye(len(values))
    return out


def qng_step(
    objective: Callable[[np.ndarray], Real],
    state_fn: Callable[[np.ndarray], np.ndarray],
    parameters: Union[Sequence[Real], Real],
    stepsize: float = 0.1,
    gradient_fn: Callable[[Callable[[np.ndarray], Real], Union[Sequence[Real], Real]], np.ndarray] | None = None,
    regularization: float = 1e-6,
) -> np.ndarray:
    values, _ = _as_parameter_array(parameters)
    if gradient_fn is None:
        gradient = np.asarray(finite_difference(objective, values), dtype=float)
    else:
        gradient = np.asarray(gradient_fn(objective, values), dtype=float)
    metric = metric_tensor(state_fn, values, regularization=regularization)
    natural_gradient = np.linalg.solve(metric, gradient)
    return values - float(stepsize) * natural_gradient
