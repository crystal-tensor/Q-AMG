# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.
# Design source: docs/mvp/gradient_design_v0.1.md.

from __future__ import annotations

from math import pi
from numbers import Real
from typing import Callable, Sequence, Union

import numpy as np


def parameter_shift(objective: Callable[[np.ndarray], Real], parameters: Union[Sequence[Real], Real], shift: float = pi / 2):
    """Differentiate a native QNode, or a known single-frequency scalar function.

    Generic callables must have unit-frequency dependence on each input.
    QNodes use gate-occurrence recipes and the classical chain rule instead.
    """

    if hasattr(objective, "compute_jacobian"):
        return objective.compute_jacobian(parameters, method="parameter-shift")
    if not np.isfinite(shift) or abs(np.sin(shift)) < 1e-12:
        raise ValueError("Parameter shift requires a finite shift with non-zero sine.")

    scalar_input = isinstance(parameters, Real)
    values = np.array([parameters] if scalar_input else list(parameters), dtype=float)
    gradient = np.zeros_like(values)
    for index in range(len(values)):
        plus = values.copy()
        minus = values.copy()
        plus[index] += shift
        minus[index] -= shift
        gradient[index] = (float(objective(plus)) - float(objective(minus))) / (2 * np.sin(shift))
    return float(gradient[0]) if scalar_input else gradient
