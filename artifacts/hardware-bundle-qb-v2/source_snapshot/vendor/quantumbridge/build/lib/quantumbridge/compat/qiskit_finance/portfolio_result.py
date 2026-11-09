# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Finance was copied.
"""Portfolio optimization result helpers for Qiskit Finance compatibility."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PortfolioOptimizationResult:
    """Serializable portfolio optimization result.

    The result can describe either the QuantumBridge-native minimal exact path
    or an optional upstream Qiskit Finance passthrough execution.
    """

    selection: tuple[int, ...]
    objective_value: float
    method: str
    path: str
    probabilities: dict[str, float] = field(default_factory=dict)
    samples: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    expected_returns: tuple[float, ...] = field(default_factory=tuple)
    covariances: tuple[tuple[float, ...], ...] = field(default_factory=tuple)
    risk_factor: float | None = None
    budget: int | None = None
    upstream_package: str = "qiskit-finance"
    upstream_version: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "0.1"

    def validate(self) -> bool:
        if not self.selection:
            raise ValueError("selection must be non-empty")
        if self.budget is not None and sum(self.selection) != self.budget:
            raise ValueError("selection does not satisfy the budget constraint")
        if self.risk_factor is not None and self.risk_factor < 0:
            raise ValueError("risk_factor must be non-negative")
        if self.expected_returns and len(self.expected_returns) != len(self.selection):
            raise ValueError("expected_returns length must match selection length")
        if self.covariances:
            if len(self.covariances) != len(self.selection):
                raise ValueError("covariance row count must match selection length")
            for row in self.covariances:
                if len(row) != len(self.selection):
                    raise ValueError("covariance matrix must be square")
        return True

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["selection"] = list(self.selection)
        payload["samples"] = list(self.samples)
        payload["expected_returns"] = list(self.expected_returns)
        payload["covariances"] = [list(row) for row in self.covariances]
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PortfolioOptimizationResult":
        data = dict(payload)
        data["selection"] = tuple(int(value) for value in data.get("selection", ()))
        data["samples"] = tuple(dict(row) for row in data.get("samples", ()))
        data["expected_returns"] = tuple(float(value) for value in data.get("expected_returns", ()))
        data["covariances"] = tuple(tuple(float(value) for value in row) for row in data.get("covariances", ()))
        data["probabilities"] = {str(key): float(value) for key, value in data.get("probabilities", {}).items()}
        data["warnings"] = tuple(str(value) for value in data.get("warnings", ()))
        data["provenance"] = dict(data.get("provenance", {}))
        data["metadata"] = dict(data.get("metadata", {}))
        return cls(**data)
