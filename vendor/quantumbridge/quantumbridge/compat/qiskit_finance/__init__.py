# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Finance optional adapter and educational portfolio subset.

Design source: docs/compat/strategy/qiskit_finance_compatibility_strategy.md.
"""

from quantumbridge.compat.qiskit_finance.applications_adapter import ADAPTER as applications_adapter
from quantumbridge.compat.qiskit_finance.circuits_adapter import ADAPTER as circuits_adapter
from quantumbridge.compat.qiskit_finance.data_provider_adapter import ADAPTER as data_provider_adapter
from quantumbridge.compat.qiskit_finance.portfolio_examples import (
    tutorial_synthetic_inputs,
    tutorial_synthetic_portfolio_problem,
)
from quantumbridge.compat.qiskit_finance.portfolio_native import (
    PortfolioProblem,
    build_portfolio_problem_native,
    enumerate_budget_selections,
    portfolio_objective_value,
    solve_portfolio_exact_native,
)
from quantumbridge.compat.qiskit_finance.portfolio_optimization_adapter import (
    compare_native_and_upstream_exact,
    dependency_available as portfolio_dependency_available,
    get_upstream_version as get_portfolio_upstream_version,
    run_portfolio_optimization_exact_upstream,
    run_portfolio_optimization_native,
    run_portfolio_optimization_qaoa_upstream,
    run_portfolio_optimization_sampling_vqe_upstream,
    run_portfolio_optimization_upstream,
    validate_portfolio_dependencies,
)
from quantumbridge.compat.qiskit_finance.portfolio_result import PortfolioOptimizationResult
from quantumbridge.compat.qiskit_finance.uncertainty_adapter import ADAPTER as uncertainty_adapter
from quantumbridge.compat.qiskit_common import QiskitAdapterFacade

ADAPTER = QiskitAdapterFacade(
    ecosystem="qiskit_finance",
    upstream_package="qiskit-finance",
    dependency_extra="qiskit-finance",
    adapters=(applications_adapter, circuits_adapter, data_provider_adapter, uncertainty_adapter),
    notes="Qiskit Finance inventory/schema bridge; no production finance, trading, or advice claim.",
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
    "applications_adapter",
    "PortfolioOptimizationResult",
    "PortfolioProblem",
    "build_portfolio_problem_native",
    "capability_level",
    "circuits_adapter",
    "compare_native_and_upstream_exact",
    "data_provider_adapter",
    "dependency_available",
    "enumerate_budget_selections",
    "get_portfolio_upstream_version",
    "get_dependency_report",
    "get_provenance",
    "get_public_object",
    "get_upstream_version",
    "get_warnings",
    "list_public_api_inventory",
    "native_implementation",
    "passthrough_call",
    "passthrough_class",
    "portfolio_dependency_available",
    "portfolio_objective_value",
    "production_ready",
    "run_portfolio_optimization_exact_upstream",
    "run_portfolio_optimization_native",
    "run_portfolio_optimization_qaoa_upstream",
    "run_portfolio_optimization_sampling_vqe_upstream",
    "run_portfolio_optimization_upstream",
    "solve_portfolio_exact_native",
    "to_quantumbridge_schema",
    "tutorial_synthetic_inputs",
    "tutorial_synthetic_portfolio_problem",
    "uncertainty_adapter",
    "unsupported",
    "upstream_required",
    "validate_environment",
    "validate_portfolio_dependencies",
    "wrap_result",
]
