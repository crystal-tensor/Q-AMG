# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MeasurementRequest:
    """Native PennyLane-like measurement request recorded on a QML tape."""

    kind: str
    observable: Any = None
    wires: tuple[int, ...] | None = None
    shots: int | None = None
    seed: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ClassicalShadowResult:
    """Finite-shot classical shadow payload.

    ``bits`` and ``recipes`` follow the common PennyLane layout of shape
    ``(shots, len(wires))``. Recipes use 0=X, 1=Y, 2=Z.
    """

    bits: np.ndarray
    recipes: np.ndarray
    wires: tuple[int, ...]
    metadata: dict[str, Any]

    def __iter__(self):
        yield self.bits
        yield self.recipes

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray]:
        return self.bits, self.recipes


def normalize_wires(wires) -> tuple[int, ...] | None:
    if wires is None:
        return None
    if isinstance(wires, int):
        return (wires,)
    return tuple(int(wire) for wire in wires)
