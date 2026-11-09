# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Machine Learning was copied.
"""Optional upstream qiskit-machine-learning passthrough helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from quantumbridge.schema.ml_results import UpstreamMLResult

from .warnings import UPSTREAM_ML_WARNING, ml_warnings, upstream_provenance


def dependency_available() -> bool:
    try:
        import_module("qiskit_machine_learning")
    except Exception:
        return False
    return True


def get_upstream_version() -> str | None:
    try:
        module = import_module("qiskit_machine_learning")
    except Exception:
        return None
    return getattr(module, "__version__", None)


def validate_ml_dependencies() -> dict[str, Any]:
    return {
        "upstream_package": "qiskit-machine-learning",
        "available": dependency_available(),
        "upstream_version": get_upstream_version(),
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
        "warnings": ml_warnings(UPSTREAM_ML_WARNING),
    }


def get_upstream_object(module_name: str, object_name: str) -> Any:
    module = import_module(module_name)
    return getattr(module, object_name)


def run_upstream_qnn_if_available() -> UpstreamMLResult:
    return _introspect_first_available(
        "upstream_qnn_passthrough",
        (
            ("qiskit_machine_learning.neural_networks", "EstimatorQNN"),
            ("qiskit_machine_learning.neural_networks", "SamplerQNN"),
        ),
    )


def run_upstream_quantum_kernel_if_available() -> UpstreamMLResult:
    return _introspect_first_available(
        "upstream_quantum_kernel_passthrough",
        (
            ("qiskit_machine_learning.kernels", "FidelityQuantumKernel"),
            ("qiskit_machine_learning.kernels", "TrainableFidelityQuantumKernel"),
        ),
    )


def run_upstream_classifier_if_available() -> UpstreamMLResult:
    return _introspect_first_available(
        "upstream_classifier_passthrough",
        (
            ("qiskit_machine_learning.algorithms", "QSVC"),
            ("qiskit_machine_learning.algorithms", "NeuralNetworkClassifier"),
            ("qiskit_machine_learning.algorithms", "VQC"),
        ),
    )


def wrap_upstream_ml_result(raw: Any, workflow: str = "upstream_ml_passthrough") -> UpstreamMLResult:
    return UpstreamMLResult(
        workflow=workflow,
        mode="upstream_passthrough",
        capability_level=1,
        upstream_package="qiskit-machine-learning",
        upstream_version=get_upstream_version(),
        production_ready=False,
        native_implementation=False,
        raw_type=type(raw).__name__,
        model_summary={"object_repr": repr(raw)[:200]},
        warnings=ml_warnings(UPSTREAM_ML_WARNING),
        provenance=upstream_provenance(workflow),
    )


def _introspect_first_available(workflow: str, candidates: tuple[tuple[str, str], ...]) -> UpstreamMLResult:
    if not dependency_available():
        return _unsupported(workflow, "optional upstream dependency unavailable: qiskit-machine-learning")
    errors: list[str] = []
    for module_name, object_name in candidates:
        try:
            obj = get_upstream_object(module_name, object_name)
        except Exception as exc:  # pragma: no cover - depends on optional package shape
            errors.append(f"{module_name}.{object_name}: {type(exc).__name__}: {exc}")
            continue
        return UpstreamMLResult(
            workflow=workflow,
            mode="upstream_passthrough",
            capability_level=1,
            upstream_package="qiskit-machine-learning",
            upstream_version=get_upstream_version(),
            production_ready=False,
            native_implementation=False,
            raw_type=getattr(obj, "__name__", type(obj).__name__),
            model_summary={
                "module": module_name,
                "object": object_name,
                "instantiated": False,
                "runtime_introspection": True,
            },
            warnings=ml_warnings(UPSTREAM_ML_WARNING),
            provenance=upstream_provenance(workflow),
        )
    return _unsupported(workflow, "; ".join(errors) or "no candidate upstream object available")


def _unsupported(workflow: str, reason: str) -> UpstreamMLResult:
    return UpstreamMLResult(
        workflow=workflow,
        mode="upstream_passthrough",
        capability_level=1,
        upstream_package="qiskit-machine-learning",
        upstream_version=get_upstream_version(),
        production_ready=False,
        native_implementation=False,
        raw_type="UnsupportedUpstreamML",
        warnings=ml_warnings(UPSTREAM_ML_WARNING),
        provenance=upstream_provenance(workflow),
        unsupported_reason=reason,
    )
