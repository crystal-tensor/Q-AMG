# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Shared source AST; lowering uses the existing native Circuit and ControlBlock.
from dataclasses import dataclass


@dataclass(frozen=True)
class Expression:
    op: str
    value: object = None
    args: tuple = ()
    line: int = 1
    column: int = 1


@dataclass(frozen=True)
class Statement:
    kind: str
    data: dict
    line: int = 1
    column: int = 1


@dataclass(frozen=True)
class Program:
    version: str
    statements: tuple
