# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""PennyLane result schema adapter."""

from __future__ import annotations

from typing import Any

from quantumbridge.schema.pennylane_results import (
    ConversionResult,
    DeviceResult,
    GradientResult,
    MeasurementResult,
    PennyLaneResult,
    QChemResult,
    QCutResult,
    QNNResult,
    QNodeResult,
    ResourceResult,
    ShadowResult,
    TemplateResult,
    TransformResult,
)

from .dependency import get_upstream_version
from .warnings import PENNYLANE_ADAPTER_WARNING, get_provenance


def wrap_pennylane_result(obj: Any, metadata: dict | None = None, result_class=PennyLaneResult) -> PennyLaneResult:
    return result_class(
        upstream_package="pennylane",
        upstream_version=get_upstream_version(),
        capability_level=2,
        mode="schema-adapter",
        raw_type=type(obj).__name__,
        data=obj,
        metadata=dict(metadata or {}),
        warnings=[PENNYLANE_ADAPTER_WARNING],
        provenance=get_provenance(mode="result-wrapper").to_dict(),
    )


def wrap_qnode_result(obj: Any, metadata: dict | None = None) -> QNodeResult:
    return wrap_pennylane_result(obj, metadata=metadata, result_class=QNodeResult)


def wrap_measurement_result(obj: Any, metadata: dict | None = None) -> MeasurementResult:
    return wrap_pennylane_result(obj, metadata=metadata, result_class=MeasurementResult)


__all__ = [
    "ConversionResult",
    "DeviceResult",
    "GradientResult",
    "MeasurementResult",
    "PennyLaneResult",
    "QChemResult",
    "QCutResult",
    "QNNResult",
    "QNodeResult",
    "ResourceResult",
    "ShadowResult",
    "TemplateResult",
    "TransformResult",
    "wrap_measurement_result",
    "wrap_pennylane_result",
    "wrap_qnode_result",
]
