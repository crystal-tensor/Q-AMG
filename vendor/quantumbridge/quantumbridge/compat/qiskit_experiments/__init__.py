# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Experiments optional adapter scaffold.

Design source: docs/compat/strategy/qiskit_ecosystem_coverage_strategy.md.
"""

from quantumbridge.compat.qiskit_experiments.calibration_adapter import ADAPTER as calibration_adapter
from quantumbridge.compat.qiskit_experiments.experiments_adapter import ADAPTER as experiments_adapter
from quantumbridge.compat.qiskit_experiments.rb_adapter import ADAPTER as rb_adapter
from quantumbridge.compat.qiskit_experiments.tomography_adapter import ADAPTER as tomography_adapter
from quantumbridge.compat.qiskit_common import QiskitAdapterFacade
from quantumbridge.compat.qiskit_experiments.dependency import (
    dependency_available as native_dependency_available,
    get_upstream_version as native_get_upstream_version,
    validate_experiments_dependencies,
)
from quantumbridge.compat.qiskit_experiments.experiments_upstream_adapter import (
    run_upstream_experiments_if_available,
    wrap_upstream_experiments_result,
)
from quantumbridge.compat.qiskit_experiments.rabi_experiment_native import (
    fit_rabi_oscillation_native,
    generate_rabi_synthetic_data,
    run_rabi_experiment_native,
)
from quantumbridge.compat.qiskit_experiments.ramsey_experiment_native import (
    fit_ramsey_native,
    generate_ramsey_synthetic_data,
    run_ramsey_experiment_native,
)
from quantumbridge.compat.qiskit_experiments.t1_experiment_native import (
    fit_t1_decay_native,
    generate_t1_synthetic_data,
    run_t1_experiment_native,
)

ADAPTER = QiskitAdapterFacade(
    ecosystem="qiskit_experiments",
    upstream_package="qiskit-experiments",
    dependency_extra="qiskit-experiments",
    adapters=(experiments_adapter, tomography_adapter, rb_adapter, calibration_adapter),
    advisory=True,
    offline_only=True,
    notes="Qiskit Experiments advisory bridge; no real experiment execution or hardware calibration claim.",
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
    "calibration_adapter",
    "capability_level",
    "dependency_available",
    "experiments_adapter",
    "get_dependency_report",
    "get_provenance",
    "get_public_object",
    "get_upstream_version",
    "get_warnings",
    "fit_rabi_oscillation_native",
    "fit_ramsey_native",
    "fit_t1_decay_native",
    "generate_rabi_synthetic_data",
    "generate_ramsey_synthetic_data",
    "generate_t1_synthetic_data",
    "list_public_api_inventory",
    "native_implementation",
    "native_dependency_available",
    "native_get_upstream_version",
    "passthrough_call",
    "passthrough_class",
    "production_ready",
    "rb_adapter",
    "run_rabi_experiment_native",
    "run_ramsey_experiment_native",
    "run_t1_experiment_native",
    "run_upstream_experiments_if_available",
    "to_quantumbridge_schema",
    "tomography_adapter",
    "unsupported",
    "upstream_required",
    "validate_experiments_dependencies",
    "validate_environment",
    "wrap_upstream_experiments_result",
    "wrap_result",
]
