# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_execution_plan_v0.1.md.

from __future__ import annotations


class QASMParseError(ValueError):
    """Syntax or subset error raised by the QuantumBridge QASM parser."""

    def __init__(self, message: str, line: int | None = None, column: int | None = None):
        location = ""
        if line is not None:
            location = f" at line {line}"
            if column is not None:
                location += f", column {column}"
        super().__init__(f"QuantumBridge QASM parse error{location}: {message}")
        self.line = line
        self.column = column
