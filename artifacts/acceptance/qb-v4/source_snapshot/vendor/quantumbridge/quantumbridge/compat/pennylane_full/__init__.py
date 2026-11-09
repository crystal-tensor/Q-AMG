# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""PennyLane full-ecosystem optional adapter scaffold.

Design source: docs/compat/strategy/pennylane_full_coverage_strategy.md.
"""

from quantumbridge.compat.pennylane_full.device_adapter import ADAPTER as device_adapter
from quantumbridge.compat.pennylane_full import dependency, qiskit_bridge, qos_uqci_bridge
from quantumbridge.compat.pennylane_full.gradients_adapter import ADAPTER as gradients_adapter
from quantumbridge.compat.pennylane_full import inventory, passthrough, registry, warnings
from quantumbridge.compat.pennylane_full.interface_adapter import ADAPTER as interface_adapter
from quantumbridge.compat.pennylane_full.io_adapter import ADAPTER as io_adapter
from quantumbridge.compat.pennylane_full.kernels_adapter import ADAPTER as kernels_adapter
from quantumbridge.compat.pennylane_full.math_adapter import ADAPTER as math_adapter
from quantumbridge.compat.pennylane_full.measurement_adapter import ADAPTER as measurement_adapter
from quantumbridge.compat.pennylane_full.measurements_adapter import list_measurements
from quantumbridge.compat.pennylane_full.observables_adapter import list_observables
from quantumbridge.compat.pennylane_full.operations_adapter import ADAPTER as operations_adapter
from quantumbridge.compat.pennylane_full.plugin_adapter import ADAPTER as plugin_adapter
from quantumbridge.compat.pennylane_full.qaoa_adapter import ADAPTER as qaoa_adapter
from quantumbridge.compat.pennylane_full.qchem_adapter import ADAPTER as qchem_adapter
from quantumbridge.compat.pennylane_full.qcut_adapter import ADAPTER as qcut_adapter
from quantumbridge.compat.pennylane_full.qiskit_bridge import quantumbridge_ir_to_qiskit_circuit
from quantumbridge.compat.pennylane_full.qnn_adapter import ADAPTER as qnn_adapter
from quantumbridge.compat.pennylane_full.qnode_adapter import ADAPTER as qnode_adapter
from quantumbridge.compat.pennylane_full.qos_uqci_bridge import quantumbridge_ir_to_uqci_payload
from quantumbridge.compat.pennylane_full.resource_adapter import ADAPTER as resource_adapter
from quantumbridge.compat.pennylane_full.shadows_adapter import ADAPTER as shadows_adapter
from quantumbridge.compat.pennylane_full.tape_adapter import tape_to_quantumbridge_ir
from quantumbridge.compat.pennylane_full.templates_adapter import ADAPTER as templates_adapter
from quantumbridge.compat.pennylane_full.transforms_adapter import ADAPTER as transforms_adapter

__all__ = [
    "device_adapter",
    "dependency",
    "gradients_adapter",
    "interface_adapter",
    "inventory",
    "io_adapter",
    "kernels_adapter",
    "list_measurements",
    "list_observables",
    "math_adapter",
    "measurement_adapter",
    "operations_adapter",
    "passthrough",
    "plugin_adapter",
    "qaoa_adapter",
    "qchem_adapter",
    "qcut_adapter",
    "qiskit_bridge",
    "qnn_adapter",
    "qnode_adapter",
    "qos_uqci_bridge",
    "quantumbridge_ir_to_qiskit_circuit",
    "quantumbridge_ir_to_uqci_payload",
    "registry",
    "resource_adapter",
    "shadows_adapter",
    "tape_to_quantumbridge_ir",
    "templates_adapter",
    "transforms_adapter",
    "warnings",
]
