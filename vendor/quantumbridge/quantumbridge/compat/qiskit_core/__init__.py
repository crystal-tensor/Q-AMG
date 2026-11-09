# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit core optional adapter scaffold.

Design source: docs/compat/strategy/qiskit_ecosystem_coverage_strategy.md.
"""

from quantumbridge.compat.qiskit_core.circuit_adapter import ADAPTER as circuit_adapter
from quantumbridge.compat.qiskit_core.primitives_adapter import ADAPTER as primitives_adapter
from quantumbridge.compat.qiskit_core.quantum_info_adapter import ADAPTER as quantum_info_adapter
from quantumbridge.compat.qiskit_core.result_adapter import ADAPTER as result_adapter
from quantumbridge.compat.qiskit_core.transpiler_adapter import ADAPTER as transpiler_adapter
from quantumbridge.compat.qiskit_common import QiskitAdapterFacade

ADAPTER = QiskitAdapterFacade(
    ecosystem="qiskit_core",
    upstream_package="qiskit",
    dependency_extra="qiskit-core",
    adapters=(circuit_adapter, primitives_adapter, quantum_info_adapter, result_adapter, transpiler_adapter),
    notes="Qiskit core circuit, transpiler, primitives, quantum_info, and result bridge.",
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
    "circuit_adapter",
    "capability_level",
    "dependency_available",
    "get_dependency_report",
    "get_provenance",
    "get_public_object",
    "get_upstream_version",
    "get_warnings",
    "list_public_api_inventory",
    "native_implementation",
    "passthrough_call",
    "passthrough_class",
    "primitives_adapter",
    "production_ready",
    "quantum_info_adapter",
    "result_adapter",
    "to_quantumbridge_schema",
    "transpiler_adapter",
    "unsupported",
    "upstream_required",
    "validate_environment",
    "wrap_result",
]
