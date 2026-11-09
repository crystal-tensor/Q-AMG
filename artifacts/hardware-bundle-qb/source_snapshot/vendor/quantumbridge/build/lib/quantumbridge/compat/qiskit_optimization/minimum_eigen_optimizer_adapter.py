# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Optimization was copied.
"""MinimumEigenOptimizer passthrough helpers for optional upstream execution."""

from quantumbridge.compat.qiskit_optimization.optimizer_adapter import ADAPTER
from quantumbridge.compat.qiskit_optimization.quadratic_program_adapter import (
    solve_quadratic_program_exact_upstream,
    validate_optimization_dependencies,
    wrap_upstream_optimization_result,
)

dependency_available = ADAPTER.dependency_available
get_upstream_version = ADAPTER.get_upstream_version
list_public_api_inventory = ADAPTER.list_public_api_inventory
passthrough_class = ADAPTER.passthrough_class
passthrough_function = ADAPTER.passthrough_function
wrap_result = ADAPTER.wrap_result
to_quantumbridge_schema = ADAPTER.to_quantumbridge_schema
provenance_metadata = ADAPTER.provenance_metadata
warn_unsupported = ADAPTER.warn_unsupported

__all__ = [
    "ADAPTER",
    "dependency_available",
    "get_upstream_version",
    "list_public_api_inventory",
    "passthrough_class",
    "passthrough_function",
    "provenance_metadata",
    "solve_quadratic_program_exact_upstream",
    "to_quantumbridge_schema",
    "validate_optimization_dependencies",
    "warn_unsupported",
    "wrap_result",
    "wrap_upstream_optimization_result",
]
