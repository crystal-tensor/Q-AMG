# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Dynamics result schema adapter."""

try:
    from quantumbridge.compat.qiskit_dynamics.dependency import get_version
    upstream_version = get_version()
except ImportError:
    upstream_version = "unknown"


def wrap_dynamics_result(obj, metadata=None):
    """Wrap a Qiskit Dynamics result in QuantumBridge DynamicsResult schema."""
    from quantumbridge.schema.dynamics_results import UpstreamDynamicsResult
    from quantumbridge.compat.qiskit_dynamics.warnings import upstream_provenance, upstream_warnings
    return UpstreamDynamicsResult(
        workflow="qiskit_dynamics_result_wrapper",
        mode="upstream_passthrough",
        upstream_package="qiskit-dynamics",
        upstream_version=upstream_version,
        capability_level=2,
        native_implementation=False,
        production_ready=False,
        hardware_calibration=False,
        raw_type="dynamics",
        data=repr(obj),
        metadata=metadata or {},
        provenance=upstream_provenance("qiskit_dynamics_result_wrapper", upstream_version),
        warnings=upstream_warnings("Wrapped result is not production dynamics evidence."),
    )
