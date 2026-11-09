# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Algorithms result schema adapters."""

from __future__ import annotations

from quantumbridge.compat.qiskit_algorithms.grover_adapter import ADAPTER
from quantumbridge.schema.algorithms_results import (
    AlgorithmComparisonResult,
    AlgorithmsResult as AlgorithmsSchemaResult,
    NativeAlgorithmResult,
    UpstreamAlgorithmResult,
)
from quantumbridge.schema import AlgorithmsResult


def wrap_algorithms_result(obj, metadata=None) -> AlgorithmsResult:
    return ADAPTER.make_result(AlgorithmsResult, obj, metadata=metadata)


def wrap_native_algorithm_result(result: AlgorithmsSchemaResult) -> NativeAlgorithmResult:
    """Return a native algorithm schema result preserving semantic fields."""

    if not isinstance(result, AlgorithmsSchemaResult):
        raise TypeError("result must be an AlgorithmsResult schema")
    payload = result.to_dict()
    payload["mode"] = "native_minimal"
    payload["native_implementation"] = True
    payload["production_ready"] = False
    return NativeAlgorithmResult.from_dict(payload)


def wrap_upstream_algorithm_schema_result(result: AlgorithmsSchemaResult) -> UpstreamAlgorithmResult:
    """Return an upstream algorithm schema result preserving semantic fields."""

    if not isinstance(result, AlgorithmsSchemaResult):
        raise TypeError("result must be an AlgorithmsResult schema")
    payload = result.to_dict()
    payload["mode"] = "upstream_passthrough"
    payload["native_implementation"] = False
    payload["production_ready"] = False
    return UpstreamAlgorithmResult.from_dict(payload)


def compare_algorithm_results(
    native_result: AlgorithmsSchemaResult,
    upstream_result: AlgorithmsSchemaResult | None,
) -> AlgorithmComparisonResult:
    """Create a lightweight native/upstream comparison envelope."""

    upstream_available = upstream_result is not None and upstream_result.unsupported_reason is None
    eigenvalue_delta = None
    if upstream_available and native_result.eigenvalue is not None and upstream_result.eigenvalue is not None:
        eigenvalue_delta = abs(float(native_result.eigenvalue) - float(upstream_result.eigenvalue))
    return AlgorithmComparisonResult(
        algorithm=native_result.algorithm,
        mode="comparison",
        capability_level=2,
        problem_type=native_result.problem_type,
        eigenvalue=native_result.eigenvalue,
        optimal_parameters=list(native_result.optimal_parameters),
        bitstring=native_result.bitstring,
        objective_value=native_result.objective_value,
        probabilities=dict(native_result.probabilities),
        input_summary=dict(native_result.input_summary),
        output_summary=dict(native_result.output_summary),
        raw_type="AlgorithmComparisonResult",
        metadata={
            "upstream_available": upstream_available,
            "eigenvalue_delta": eigenvalue_delta,
            "native_mode": native_result.mode,
            "upstream_mode": upstream_result.mode if upstream_result is not None else None,
        },
        warnings=list(native_result.warnings),
        provenance=dict(native_result.provenance),
        production_ready=False,
        native_implementation=True,
    )
