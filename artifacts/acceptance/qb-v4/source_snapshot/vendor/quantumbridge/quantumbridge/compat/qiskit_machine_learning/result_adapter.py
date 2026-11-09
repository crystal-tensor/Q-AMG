# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Machine Learning result schema adapter."""

try:
    from quantumbridge.compat.qiskit_machine_learning.dependency import get_version
    upstream_version = get_version()
except ImportError:
    upstream_version = "unknown"

from quantumbridge.schema.ml_results import (
    KernelClassifierResult,
    MachineLearningResult,
    MLComparisonResult,
    QNNClassifierResult,
    QNNForwardResult,
    QuantumKernelResult,
    UpstreamMLResult,
)

from .warnings import QISKIT_ML_WARNING, ml_warnings, upstream_provenance


def wrap_ml_result(obj, result_type: str = "generic", metadata=None):
    """Wrap a generic Qiskit ML result in the legacy QuantumBridge MLResult schema."""
    from quantumbridge.schema import MLResult

    return MLResult(
        ecosystem="qiskit",
        upstream_package="qiskit-machine-learning",
        upstream_version=upstream_version,
        capability_level=2,
        mode="schema-adapter",
        raw_type=result_type,
        data=obj,
        metadata=metadata or {},
        provenance={
            "upstream_package": "qiskit-machine-learning",
            "adapter": "result_adapter",
            "capability_level": 2,
            "source_code_copied": False,
            "official_endorsement": False,
            "cloud_access": False,
            "token_read": False,
        },
        warnings=ml_warnings(QISKIT_ML_WARNING),
    )


def wrap_upstream_result(obj, workflow: str = "upstream_ml_schema_bridge") -> UpstreamMLResult:
    """Wrap an upstream object in the Stage 9E dedicated ML schema."""

    return UpstreamMLResult(
        workflow=workflow,
        mode="upstream_passthrough",
        capability_level=1,
        upstream_package="qiskit-machine-learning",
        upstream_version=upstream_version,
        production_ready=False,
        native_implementation=False,
        raw_type=type(obj).__name__,
        model_summary={"object_repr": repr(obj)[:200]},
        metadata={"legacy_wrapper": False},
        warnings=ml_warnings(QISKIT_ML_WARNING),
        provenance=upstream_provenance(workflow),
    )


__all__ = [
    "KernelClassifierResult",
    "MachineLearningResult",
    "MLComparisonResult",
    "QNNClassifierResult",
    "QNNForwardResult",
    "QuantumKernelResult",
    "UpstreamMLResult",
    "wrap_ml_result",
    "wrap_upstream_result",
]
