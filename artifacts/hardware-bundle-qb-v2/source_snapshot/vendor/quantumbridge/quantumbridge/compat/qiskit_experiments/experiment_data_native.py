# This file is independently implemented for QuantumBridge SDK.
# No source code from IBM, Qiskit, or Qiskit Experiments was copied.
"""Small serializable containers for offline experiment data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class NativeExperimentData:
    experiment_type: str
    x_name: str
    x_values: list[float]
    y_name: str
    y_values: list[float]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_type": self.experiment_type,
            "x_name": self.x_name,
            "x_values": [float(value) for value in self.x_values],
            "y_name": self.y_name,
            "y_values": [float(value) for value in self.y_values],
            "metadata": dict(self.metadata),
        }


def experiment_result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return {"raw_type": type(result).__name__, "repr": repr(result)}
