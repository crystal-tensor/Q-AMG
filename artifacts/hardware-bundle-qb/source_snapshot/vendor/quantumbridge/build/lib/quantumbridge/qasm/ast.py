# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_execution_plan_v0.1.md.

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RegisterDecl:
    kind: str
    name: str
    size: int


@dataclass(frozen=True)
class IndexedRef:
    register: str
    index: int


@dataclass(frozen=True)
class GateCall:
    name: str
    qubits: tuple[IndexedRef, ...]
    params: tuple[float, ...] = ()


@dataclass(frozen=True)
class Measurement:
    qubit: IndexedRef | str
    bit: IndexedRef | str


@dataclass(frozen=True)
class Barrier:
    qubits: tuple[IndexedRef | str, ...]


@dataclass(frozen=True)
class QASMProgram:
    version: str = "2.0"
    include: str | None = None
    registers: tuple[RegisterDecl, ...] = ()
    statements: tuple[GateCall | Measurement | Barrier, ...] = field(default_factory=tuple)
