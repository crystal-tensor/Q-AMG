# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.
# Design source: docs/mvp/api_contract_v0.1.md and docs/mvp/simulator_design_v0.1.md.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Measurement:
    """Computational-basis measurement request."""

    wire: int
    bit: int
    kind: str = "computational"
    metadata: dict[str, Any] = field(default_factory=dict)

