# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Optimization result schema adapters."""

from __future__ import annotations

import warnings

from quantumbridge.compat.qiskit_optimization.quadratic_program_result import NativeOptimizationResult
from quantumbridge.compat.qiskit_optimization.warnings import OPTIMIZATION_ADAPTER_WARNING
from quantumbridge.schema.optimization_results import (
    BruteForceOptimizationResult,
    OptimizationComparisonResult,
    UpstreamOptimizationResult,
)

warnings.warn(
    OPTIMIZATION_ADAPTER_WARNING,
    UserWarning,
    stacklevel=2
)

from quantumbridge.compat.qiskit_optimization.applications_adapter import ADAPTER
from quantumbridge.schema import OptimizationResult


def wrap_optimization_result(obj, metadata=None) -> OptimizationResult:
    """Wrap a Qiskit Optimization result in QuantumBridge OptimizationResult schema."""
    return ADAPTER.make_result(OptimizationResult, obj, metadata=metadata)


def wrap_native_optimization_result(result: NativeOptimizationResult) -> BruteForceOptimizationResult:
    """Wrap a native executable optimization result in the Stage 9B schema."""

    if not isinstance(result, NativeOptimizationResult):
        raise TypeError("result must be a NativeOptimizationResult")
    return BruteForceOptimizationResult(
        mode="native_minimal",
        capability_level=3,
        objective_sense=result.objective_sense,
        objective_value=result.objective_value,
        assignment=dict(result.assignment),
        feasible=result.feasible,
        constraints=[dict(row) for row in result.constraints],
        qubo_metadata=dict(result.qubo_metadata),
        ising_metadata=dict(result.ising_metadata),
        raw_type=type(result).__name__,
        metadata=dict(result.metadata),
        warnings=list(result.warnings),
        provenance=dict(result.provenance),
        unsupported_reason=result.unsupported_reason,
        upstream_package=result.upstream_package,
        upstream_version=result.upstream_version,
        production_ready=False,
        native_implementation=True,
        num_variables=int(result.metadata.get("num_variables", len(result.assignment))),
    )


def wrap_upstream_schema_result(result: NativeOptimizationResult) -> UpstreamOptimizationResult:
    """Wrap an upstream passthrough result in the Stage 9B schema."""

    if not isinstance(result, NativeOptimizationResult):
        raise TypeError("result must be a NativeOptimizationResult")
    return UpstreamOptimizationResult(
        mode="upstream_passthrough",
        capability_level=2,
        objective_sense=result.objective_sense,
        objective_value=result.objective_value,
        assignment=dict(result.assignment),
        feasible=result.feasible,
        constraints=[dict(row) for row in result.constraints],
        qubo_metadata=dict(result.qubo_metadata),
        ising_metadata=dict(result.ising_metadata),
        raw_type=type(result).__name__,
        metadata=dict(result.metadata),
        warnings=list(result.warnings),
        provenance=dict(result.provenance),
        unsupported_reason=result.unsupported_reason,
        upstream_package=result.upstream_package,
        upstream_version=result.upstream_version,
        production_ready=False,
        native_implementation=False,
        num_variables=int(result.metadata.get("num_variables", len(result.assignment))),
    )


def compare_optimization_results(
    native_result: NativeOptimizationResult,
    upstream_result: NativeOptimizationResult | None,
) -> OptimizationComparisonResult:
    """Create a lightweight native/upstream comparison envelope."""

    upstream_available = upstream_result is not None
    agrees = False
    if upstream_result is not None:
        agrees = (
            native_result.feasible == upstream_result.feasible
            and native_result.assignment == upstream_result.assignment
            and native_result.objective_value == upstream_result.objective_value
        )
    return OptimizationComparisonResult(
        mode="comparison",
        capability_level=2,
        objective_sense=native_result.objective_sense,
        objective_value=native_result.objective_value,
        assignment=dict(native_result.assignment),
        feasible=native_result.feasible,
        constraints=[dict(row) for row in native_result.constraints],
        qubo_metadata=dict(native_result.qubo_metadata),
        ising_metadata=dict(native_result.ising_metadata),
        raw_type="NativeOptimizationResult",
        metadata={
            "upstream_available": upstream_available,
            "native_upstream_agree": agrees,
        },
        warnings=list(native_result.warnings),
        provenance=dict(native_result.provenance),
        production_ready=False,
        native_implementation=True,
        num_variables=int(native_result.metadata.get("num_variables", len(native_result.assignment))),
    )
