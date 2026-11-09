# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_execution_plan_v0.1.md.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PassManagerConfig:
    basis_gates: tuple[str, ...] = ("id", "x", "y", "z", "h", "rx", "ry", "rz", "cx", "cz")
    optimization_level: int = 1
    coupling_map: object | None = None
