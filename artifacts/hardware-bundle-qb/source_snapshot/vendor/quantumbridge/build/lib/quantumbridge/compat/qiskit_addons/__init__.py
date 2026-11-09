# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Addons optional adapter scaffold.

Design source: docs/compat/strategy/qiskit_ecosystem_coverage_strategy.md.
"""

from quantumbridge.compat.qiskit_addons.aqc_adapter import ADAPTER as aqc_adapter
from quantumbridge.compat.qiskit_addons.mpf_adapter import ADAPTER as mpf_adapter
from quantumbridge.compat.qiskit_addons.obp_adapter import ADAPTER as obp_adapter
from quantumbridge.compat.qiskit_addons.sqd_adapter import ADAPTER as sqd_adapter
from quantumbridge.compat.qiskit_common import QiskitAdapterFacade

ADAPTER = QiskitAdapterFacade(
    ecosystem="qiskit_addons",
    upstream_package="qiskit-addons",
    dependency_extra="qiskit-addons",
    adapters=(sqd_adapter, mpf_adapter, aqc_adapter, obp_adapter),
    advisory=True,
    notes="Qiskit Addons advisory inventory bridge; optional dependency only.",
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
    "aqc_adapter",
    "capability_level",
    "dependency_available",
    "get_dependency_report",
    "get_provenance",
    "get_public_object",
    "get_upstream_version",
    "get_warnings",
    "list_public_api_inventory",
    "mpf_adapter",
    "native_implementation",
    "obp_adapter",
    "passthrough_call",
    "passthrough_class",
    "production_ready",
    "sqd_adapter",
    "to_quantumbridge_schema",
    "unsupported",
    "upstream_required",
    "validate_environment",
    "wrap_result",
]
