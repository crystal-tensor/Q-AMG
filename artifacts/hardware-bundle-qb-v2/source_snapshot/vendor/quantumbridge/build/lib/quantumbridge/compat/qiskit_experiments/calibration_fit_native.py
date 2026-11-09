# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Qiskit Experiments was copied.
"""Educational deterministic fitting helpers for synthetic experiments."""

from __future__ import annotations

import math
from typing import Iterable


def finite_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("fit values must be finite")
    return value


def mean(values: Iterable[float]) -> float:
    data = [float(value) for value in values]
    if not data:
        raise ValueError("cannot average an empty sequence")
    return sum(data) / len(data)


def sse(observed: list[float], predicted: list[float]) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(observed, predicted))


def goodness(observed: list[float], predicted: list[float]) -> dict[str, float]:
    if len(observed) != len(predicted):
        raise ValueError("observed and predicted must have the same length")
    error = sse(observed, predicted)
    baseline = mean(observed)
    total = sum((value - baseline) ** 2 for value in observed)
    r_squared = 1.0 if total == 0 else max(0.0, 1.0 - error / total)
    rmse = math.sqrt(error / len(observed)) if observed else 0.0
    return {"sse": float(error), "rmse": float(rmse), "r_squared": float(r_squared)}


def linear_space(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [float(start)]
    step = (float(stop) - float(start)) / float(count - 1)
    return [float(start) + step * index for index in range(count)]


def deterministic_noise(index: int, scale: float, seed: int | None) -> float:
    if not scale:
        return 0.0
    phase = (seed or 0) * 0.173 + index * 1.618
    return float(scale) * math.sin(phase)
