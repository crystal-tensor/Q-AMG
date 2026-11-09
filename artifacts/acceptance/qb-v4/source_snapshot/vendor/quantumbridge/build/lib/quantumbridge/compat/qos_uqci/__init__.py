# This file is independently implemented for QuantumBridge SDK.
# No source code from QOS-UQCI was copied.
"""QOS-UQCI clean-room backend compatibility helpers."""

from .calset_adapter import build_calset, validate_calset
from .dependency import dependency_available, get_upstream_version, validate_qos_uqci_dependencies
from .devicespec_adapter import build_devicespec, validate_devicespec
from .examples import qos_uqci_bell_circuit, run_qos_uqci_bell_job_example, run_qos_uqci_mock_runtime_example
from .job_spec import build_qos_uqci_job_spec, validate_qos_uqci_job_spec
from .manifest_adapter import build_manifest, validate_manifest
from .mock_runtime import run_qos_uqci_mock_runtime
from .openqasm_bridge import build_openqasm_compatibility_artifact, validate_openqasm_compatibility_artifact
from .result_adapter import to_quantumbridge_schema, wrap_qos_uqci_result
from .uqci_ir_adapter import quantumbridge_ir_to_uqci_ir, uqci_ir_to_quantumbridge_ir, validate_uqci_ir
from .upstream_adapter import run_upstream_qos_uqci_if_available
from .warnings import QOS_UQCI_COMPAT_WARNING, qos_uqci_warnings

__all__ = [
    "QOS_UQCI_COMPAT_WARNING",
    "build_calset",
    "build_devicespec",
    "build_manifest",
    "build_openqasm_compatibility_artifact",
    "build_qos_uqci_job_spec",
    "dependency_available",
    "get_upstream_version",
    "qos_uqci_bell_circuit",
    "qos_uqci_warnings",
    "quantumbridge_ir_to_uqci_ir",
    "run_qos_uqci_bell_job_example",
    "run_qos_uqci_mock_runtime",
    "run_qos_uqci_mock_runtime_example",
    "run_upstream_qos_uqci_if_available",
    "to_quantumbridge_schema",
    "uqci_ir_to_quantumbridge_ir",
    "validate_calset",
    "validate_devicespec",
    "validate_manifest",
    "validate_openqasm_compatibility_artifact",
    "validate_qos_uqci_dependencies",
    "validate_qos_uqci_job_spec",
    "validate_uqci_ir",
    "wrap_qos_uqci_result",
]
