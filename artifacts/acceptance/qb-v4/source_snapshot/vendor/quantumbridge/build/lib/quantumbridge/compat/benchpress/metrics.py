# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Benchpress was copied.
"""Small deterministic benchmark metrics."""

from __future__ import annotations

from typing import Mapping


def l1_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    return float(sum(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys))


def max_probability_delta(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    return float(max((abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys), default=0.0))


def objective_delta(observed: float | None, expected: float | None) -> float | None:
    if observed is None or expected is None:
        return None
    return float(abs(float(observed) - float(expected)))


def energy_delta(observed: float | None, expected: float | None) -> float | None:
    return objective_delta(observed, expected)


def accuracy(predictions: list[int], labels: list[int]) -> float:
    if not labels:
        return 0.0
    return float(sum(int(a == b) for a, b in zip(predictions, labels)) / len(labels))


def pass_fail(value: bool) -> str:
    return "passed" if bool(value) else "failed"


def skipped_reason(reason: str | None) -> dict[str, object]:
    return {"skipped": reason is not None, "skipped_reason": reason}


def unsupported_reason(reason: str | None) -> dict[str, object]:
    return {"unsupported": reason is not None, "unsupported_reason": reason}
