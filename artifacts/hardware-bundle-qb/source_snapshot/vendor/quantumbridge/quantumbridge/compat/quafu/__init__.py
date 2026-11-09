# This file is independently implemented for QuantumBridge SDK.
# No source code from Quafu or pyquafu was copied.
"""Quafu clean-room backend compatibility helpers."""

from .cloud_backend import QUARK_SHOT_UNIT, QuafuCloudBackend, QuafuCloudJob, QuarkStudioTransport
from .dependency import (
    dependency_available,
    get_upstream_version,
    validate_quafu_dependencies,
    validate_quarkstudio_dependencies,
)
from .examples import quafu_bell_circuit, run_quafu_bell_payload_example, run_quafu_mock_backend_example
from .job_adapter import build_quafu_job_spec, validate_quafu_job_spec
from .mock_backend import run_quafu_mock_backend
from .payload_adapter import quantumbridge_ir_to_quafu_payload, quafu_payload_to_quantumbridge_ir, validate_quafu_payload
from .result_adapter import to_quantumbridge_schema, wrap_quafu_result
from .upstream_adapter import run_upstream_pyquafu_if_available
from .warnings import QUAFU_COMPAT_WARNING, quafu_warnings

__all__ = [
    "QUAFU_COMPAT_WARNING",
    "QUARK_SHOT_UNIT",
    "QuafuCloudBackend",
    "QuafuCloudJob",
    "QuarkStudioTransport",
    "build_quafu_job_spec",
    "dependency_available",
    "get_upstream_version",
    "quafu_bell_circuit",
    "quafu_payload_to_quantumbridge_ir",
    "quafu_warnings",
    "quantumbridge_ir_to_quafu_payload",
    "run_quafu_bell_payload_example",
    "run_quafu_mock_backend",
    "run_quafu_mock_backend_example",
    "run_upstream_pyquafu_if_available",
    "to_quantumbridge_schema",
    "validate_quafu_dependencies",
    "validate_quarkstudio_dependencies",
    "validate_quafu_job_spec",
    "validate_quafu_payload",
    "wrap_quafu_result",
]
