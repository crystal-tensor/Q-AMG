# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Algorithms optional adapter coverage."""

from quantumbridge.compat.qiskit_algorithms.minimum_eigensolver_adapter import ADAPTER as minimum_eigensolver_adapter
from quantumbridge.compat.qiskit_algorithms.amplitude_adapter import ADAPTER as amplitude_adapter
from quantumbridge.compat.qiskit_algorithms.eigensolver_adapter import ADAPTER as eigensolver_adapter
from quantumbridge.compat.qiskit_algorithms.gradient_adapter import ADAPTER as gradient_adapter
from quantumbridge.compat.qiskit_algorithms.grover_adapter import ADAPTER as grover_adapter
from quantumbridge.compat.qiskit_algorithms.optimizer_adapter import ADAPTER as optimizer_adapter
from quantumbridge.compat.qiskit_algorithms.algorithms_native import (
    NativeVQEAnsatz,
    build_grover_oracle_marked_bitstrings,
    build_maxcut_problem_native,
    build_two_qubit_test_hamiltonian,
    build_vqe_ansatz_rx_ry,
    dependency_available as algorithms_dependency_available,
    evaluate_ansatz_statevector,
    expectation_value_statevector,
    get_upstream_version as get_algorithms_upstream_version,
    grover_result_to_dict,
    maxcut_objective_value,
    qaoa_cost_hamiltonian_metadata,
    run_grover_native,
    run_grover_upstream,
    run_qaoa_native_maxcut,
    run_qaoa_upstream,
    run_vqe_native,
    run_vqe_upstream,
    simulate_grover_statevector_native,
    solve_maxcut_bruteforce_native,
    solve_vqe_grid_search_native,
    validate_algorithms_dependencies,
    wrap_upstream_algorithm_result,
)
from quantumbridge.compat.qiskit_algorithms.result_adapter import (
    compare_algorithm_results,
    wrap_native_algorithm_result,
    wrap_upstream_algorithm_schema_result,
)
from quantumbridge.compat.qiskit_algorithms.examples import run_all_native_algorithm_examples
from quantumbridge.compat.qiskit_common import QiskitAdapterFacade

ADAPTER = QiskitAdapterFacade(
    ecosystem="qiskit_algorithms",
    upstream_package="qiskit-algorithms",
    dependency_extra="qiskit-algorithms",
    adapters=(
        minimum_eigensolver_adapter,
        eigensolver_adapter,
        optimizer_adapter,
        gradient_adapter,
        amplitude_adapter,
        grover_adapter,
    ),
    notes="Qiskit Algorithms passthrough/schema bridge; no algorithm parity claim.",
)

capability_level = ADAPTER.capability_level
production_ready = ADAPTER.production_ready
native_implementation = ADAPTER.native_implementation
upstream_required = ADAPTER.upstream_required
dependency_available = ADAPTER.dependency_available
get_upstream_version = ADAPTER.get_upstream_version
get_dependency_report = ADAPTER.get_dependency_report
list_public_api_inventory = ADAPTER.list_public_api_inventory
get_public_object = ADAPTER.get_public_object
passthrough_call = ADAPTER.passthrough_call
passthrough_class = ADAPTER.passthrough_class
wrap_result = ADAPTER.wrap_result
to_quantumbridge_schema = ADAPTER.to_quantumbridge_schema
get_warnings = ADAPTER.get_warnings
get_provenance = ADAPTER.get_provenance
unsupported = ADAPTER.unsupported
validate_environment = ADAPTER.validate_environment

__all__ = [
    "ADAPTER",
    "NativeVQEAnsatz",
    "amplitude_adapter",
    "algorithms_dependency_available",
    "build_grover_oracle_marked_bitstrings",
    "build_maxcut_problem_native",
    "build_two_qubit_test_hamiltonian",
    "build_vqe_ansatz_rx_ry",
    "capability_level",
    "compare_algorithm_results",
    "dependency_available",
    "eigensolver_adapter",
    "evaluate_ansatz_statevector",
    "expectation_value_statevector",
    "get_algorithms_upstream_version",
    "get_dependency_report",
    "get_provenance",
    "get_public_object",
    "get_upstream_version",
    "get_warnings",
    "gradient_adapter",
    "grover_result_to_dict",
    "grover_adapter",
    "list_public_api_inventory",
    "maxcut_objective_value",
    "minimum_eigensolver_adapter",
    "native_implementation",
    "optimizer_adapter",
    "passthrough_call",
    "passthrough_class",
    "production_ready",
    "qaoa_cost_hamiltonian_metadata",
    "run_grover_native",
    "run_grover_upstream",
    "run_qaoa_native_maxcut",
    "run_qaoa_upstream",
    "run_all_native_algorithm_examples",
    "run_vqe_native",
    "run_vqe_upstream",
    "simulate_grover_statevector_native",
    "solve_maxcut_bruteforce_native",
    "solve_vqe_grid_search_native",
    "to_quantumbridge_schema",
    "unsupported",
    "upstream_required",
    "validate_algorithms_dependencies",
    "validate_environment",
    "wrap_native_algorithm_result",
    "wrap_result",
    "wrap_upstream_algorithm_result",
    "wrap_upstream_algorithm_schema_result",
]
