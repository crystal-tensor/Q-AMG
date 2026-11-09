# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_upstream_integration_strategy_v0.1.md.

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


class GradientDescentOptimizer:
    mode = "Native Core"

    def __init__(self, learning_rate: float = 0.1, steps: int = 100, tolerance: float = 1e-8):
        self.learning_rate = float(learning_rate)
        self.steps = int(steps)
        self.tolerance = float(tolerance)

    def minimize(self, fun: Callable[[np.ndarray], float], x0: Sequence[float], jac: Optional[Callable[[np.ndarray], Sequence[float]]] = None):
        x = np.asarray(x0, dtype=float)
        trace: list[dict[str, Any]] = []
        best_x = x.copy()
        best_fun = float(fun(x))
        for step in range(self.steps):
            value = float(fun(x))
            gradient = np.asarray(jac(x) if jac is not None else _finite_difference_gradient(fun, x), dtype=float)
            trace.append({"step": step, "fun": value, "x": x.tolist(), "gradient": gradient.tolist()})
            if value < best_fun:
                best_fun = value
                best_x = x.copy()
            if float(np.linalg.norm(gradient)) <= self.tolerance:
                return OptimizerResult(
                    x=x,
                    fun=value,
                    nit=step + 1,
                    nfev=len(trace),
                    success=True,
                    message="gradient norm below tolerance",
                    trace=trace,
                )
            x = x - self.learning_rate * gradient
        final_value = float(fun(x))
        if final_value < best_fun:
            best_fun = final_value
            best_x = x.copy()
        return OptimizerResult(
            x=best_x,
            fun=best_fun,
            nit=self.steps,
            nfev=len(trace),
            success=False,
            message="maximum steps reached",
            trace=trace,
        )


@dataclass(frozen=True)
class OptimizerResult:
    x: np.ndarray
    fun: float
    nit: int
    nfev: int
    success: bool
    message: str
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x.tolist(),
            "fun": float(self.fun),
            "nit": int(self.nit),
            "nfev": int(self.nfev),
            "success": bool(self.success),
            "message": self.message,
            "trace": list(self.trace),
        }


def _finite_difference_gradient(fun: Callable[[np.ndarray], float], x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    gradient = np.zeros_like(x, dtype=float)
    for index in range(x.size):
        plus = x.copy()
        minus = x.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        gradient[index] = (float(fun(plus)) - float(fun(minus))) / (2 * epsilon)
    return gradient


__all__ = ["GradientDescentOptimizer", "OptimizerResult"]
