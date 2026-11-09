# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit IBM Runtime optional adapter scaffold.

Design source: docs/compat/strategy/qiskit_ecosystem_coverage_strategy.md.
"""

from quantumbridge.compat.qiskit_runtime.backend_adapter import ADAPTER as backend_adapter
from quantumbridge.compat.qiskit_runtime.job_adapter import ADAPTER as job_adapter
from quantumbridge.compat.qiskit_runtime.runtime_adapter import ADAPTER as runtime_adapter
from quantumbridge.compat.qiskit_common import QiskitAdapterFacade

ADAPTER = QiskitAdapterFacade(
    ecosystem="qiskit_runtime",
    upstream_package="qiskit-ibm-runtime",
    dependency_extra="qiskit-runtime",
    adapters=(runtime_adapter, backend_adapter, job_adapter),
    advisory=True,
    offline_only=True,
    notes="Qiskit IBM Runtime offline-only bridge; no IBM Cloud, token, credential, or hardware access.",
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
    "backend_adapter",
    "capability_level",
    "dependency_available",
    "get_dependency_report",
    "get_provenance",
    "get_public_object",
    "get_upstream_version",
    "get_warnings",
    "job_adapter",
    "list_public_api_inventory",
    "native_implementation",
    "passthrough_call",
    "passthrough_class",
    "production_ready",
    "runtime_adapter",
    "to_quantumbridge_schema",
    "unsupported",
    "upstream_required",
    "validate_environment",
    "wrap_result",
]
