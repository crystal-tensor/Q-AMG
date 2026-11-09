# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Derivative payloads and inspectable execution outcomes."""

from dataclasses import dataclass, field

from .derivative_status import DerivativeExecutionStatus


@dataclass
class DerivativeExecutionResult:
    value: object = None
    derivative: object = None
    status: DerivativeExecutionStatus = DerivativeExecutionStatus.SUCCESS
    receipt: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    error: object = None
    warnings: tuple = ()
    provenance: dict = field(default_factory=lambda: {"engine": "quantumbridge-native", "status_protocol_version": 1})
    jacobian: object = None
    second: object = None

    @property
    def code(self):
        return int(self.status)
