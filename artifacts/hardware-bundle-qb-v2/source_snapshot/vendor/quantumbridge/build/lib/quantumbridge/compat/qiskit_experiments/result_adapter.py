# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Experiments result schema adapter."""

try:
    from quantumbridge.compat.qiskit_experiments.dependency import get_version
    upstream_version = get_version()
except ImportError:
    upstream_version = "unknown"


def wrap_experiment_result(obj, experiment_type: str = "generic", metadata=None):
    """Wrap a Qiskit Experiments result in QuantumBridge ExperimentsResult schema."""
    from quantumbridge.schema.experiments_results import UpstreamExperimentsResult
    from quantumbridge.compat.qiskit_experiments.warnings import upstream_provenance, upstream_warnings
    return UpstreamExperimentsResult(
        workflow="qiskit_experiments_result_wrapper",
        mode="upstream_passthrough",
        upstream_package="qiskit-experiments",
        upstream_version=upstream_version,
        capability_level=2,
        native_implementation=False,
        production_ready=False,
        hardware_calibration=False,
        raw_type=experiment_type,
        data=repr(obj),
        metadata=metadata or {},
        provenance=upstream_provenance("qiskit_experiments_result_wrapper", upstream_version),
        warnings=upstream_warnings("Wrapped result is not hardware calibration evidence."),
    )


# Alias for backward compatibility
wrap_experiments_result = wrap_experiment_result
