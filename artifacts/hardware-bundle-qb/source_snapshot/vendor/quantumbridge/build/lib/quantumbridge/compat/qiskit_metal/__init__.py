# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Metal optional adapter coverage."""

from quantumbridge.compat.qiskit_metal.design_adapter import ADAPTER as design_adapter
from quantumbridge.compat.qiskit_metal.component_adapter import ADAPTER as component_adapter
from quantumbridge.compat.qiskit_metal.renderer_adapter import ADAPTER as renderer_adapter
from quantumbridge.compat.qiskit_metal.simulation_adapter import ADAPTER as simulation_adapter
from quantumbridge.compat.qiskit_common import QiskitAdapterFacade

ADAPTER = QiskitAdapterFacade(
    ecosystem="qiskit_metal",
    upstream_package="qiskit-metal",
    dependency_extra="qiskit-metal",
    adapters=(design_adapter, component_adapter, renderer_adapter, simulation_adapter),
    advisory=True,
    notes="Qiskit Metal advisory bridge; no chip fabrication, EM solver validation, or production design support.",
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
    "capability_level",
    "component_adapter",
    "dependency_available",
    "design_adapter",
    "get_dependency_report",
    "get_provenance",
    "get_public_object",
    "get_upstream_version",
    "get_warnings",
    "list_public_api_inventory",
    "native_implementation",
    "passthrough_call",
    "passthrough_class",
    "production_ready",
    "renderer_adapter",
    "simulation_adapter",
    "to_quantumbridge_schema",
    "unsupported",
    "upstream_required",
    "validate_environment",
    "wrap_result",
]
