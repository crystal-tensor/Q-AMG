# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Aer optional adapter scaffold.

Design source: docs/compat/strategy/qiskit_aer_compatibility_strategy.md.
"""

from quantumbridge.compat.qiskit_aer.aer_adapter import ADAPTER as aer_adapter
from quantumbridge.compat.qiskit_aer.aer_upstream_adapter import (
    dependency_available as upstream_dependency_available,
    get_upstream_version as upstream_get_upstream_version,
    run_noisy_upstream_aer,
    run_qasm_upstream_aer,
    run_statevector_upstream_aer,
    validate_aer_dependencies,
    wrap_upstream_aer_result,
)
from quantumbridge.compat.qiskit_aer.circuit_execution_adapter import (
    execute_basic_circuit_native,
    normalize_circuit_to_quantumbridge_ir,
)
from quantumbridge.compat.qiskit_aer.examples import (
    bell_circuit,
    run_noisy_counts_example,
    run_qasm_counts_example,
    run_statevector_example,
)
from quantumbridge.compat.qiskit_aer.noise_adapter import ADAPTER as noise_adapter
from quantumbridge.compat.qiskit_aer.noise_adapter import (
    apply_measurement_bitflip_noise,
    create_bitflip_noise_model,
    create_depolarizing_metadata,
    validate_noise_model,
)
from quantumbridge.compat.qiskit_aer.qasm_simulator_adapter import (
    run_noisy_qasm_simulator_native,
    run_qasm_simulator_native,
)
from quantumbridge.compat.qiskit_aer.result_adapter import wrap_aer_result
from quantumbridge.compat.qiskit_aer.statevector_simulator_adapter import (
    run_statevector_simulator_native,
)
from quantumbridge.compat.qiskit_aer.simulator_native import run_stabilizer_simulator_native
from quantumbridge.compat.qiskit_aer.simulator_native import run_density_matrix_simulator_native
from quantumbridge.compat.qiskit_aer.simulator_native import run_matrix_product_state_simulator_native
from quantumbridge.compat.qiskit_common import QiskitAdapterFacade

ADAPTER = QiskitAdapterFacade(
    ecosystem="qiskit_aer",
    upstream_package="qiskit-aer",
    dependency_extra="qiskit-aer",
    adapters=(aer_adapter, noise_adapter),
    notes="Qiskit Aer simulator and noise inventory bridge; no production simulation parity claim.",
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
    "aer_adapter",
    "apply_measurement_bitflip_noise",
    "bell_circuit",
    "capability_level",
    "create_bitflip_noise_model",
    "create_depolarizing_metadata",
    "dependency_available",
    "execute_basic_circuit_native",
    "get_dependency_report",
    "get_provenance",
    "get_public_object",
    "get_upstream_version",
    "get_warnings",
    "list_public_api_inventory",
    "native_implementation",
    "noise_adapter",
    "normalize_circuit_to_quantumbridge_ir",
    "passthrough_call",
    "passthrough_class",
    "production_ready",
    "run_noisy_qasm_simulator_native",
    "run_noisy_counts_example",
    "run_noisy_upstream_aer",
    "run_qasm_counts_example",
    "run_qasm_simulator_native",
    "run_qasm_upstream_aer",
    "run_statevector_simulator_native",
    "run_stabilizer_simulator_native",
    "run_density_matrix_simulator_native",
    "run_matrix_product_state_simulator_native",
    "run_statevector_example",
    "run_statevector_upstream_aer",
    "to_quantumbridge_schema",
    "unsupported",
    "upstream_dependency_available",
    "upstream_get_upstream_version",
    "upstream_required",
    "validate_aer_dependencies",
    "validate_environment",
    "validate_noise_model",
    "wrap_aer_result",
    "wrap_result",
    "wrap_upstream_aer_result",
]
