# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_execution_plan_v0.1.md.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompilerReport:
    input_operations: int
    output_operations: int
    analyses: dict[str, Any] = field(default_factory=dict)
    changed: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_operations": self.input_operations,
            "output_operations": self.output_operations,
            "analyses": dict(self.analyses),
            "changed": self.changed,
            "warnings": list(self.warnings),
        }
