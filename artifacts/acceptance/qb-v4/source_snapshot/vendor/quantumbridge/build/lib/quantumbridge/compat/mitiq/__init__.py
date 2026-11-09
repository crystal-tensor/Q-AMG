# This file is independently implemented for QuantumBridge SDK.
# No source code from Mitiq, Qiskit, or Qiskit Aer was copied.
"""Mitiq optional passthrough and educational native error mitigation."""

from .dependency import dependency_available, get_upstream_version, validate_mitiq_dependencies
from .examples import bell_circuit, run_readout_mitigation_example, run_zne_example
from .mitiq_upstream_adapter import (
    run_readout_mitigation_upstream_if_available,
    run_zne_upstream_if_available,
    wrap_upstream_mitiq_result,
)
from .readout_mitigation_native import (
    apply_readout_error_to_counts,
    build_single_qubit_readout_calibration_matrix,
    build_tensor_product_readout_matrix,
    mitigate_readout_counts,
    normalize_counts_to_probabilities,
    run_readout_mitigation_native,
)
from .result_adapter import wrap_mitiq_result
from .warnings import (
    MITIQ_WARNING,
    NATIVE_READOUT_WARNING,
    NATIVE_ZNE_WARNING,
    UPSTREAM_MITIQ_WARNING,
    mitiq_warnings,
)
from .zne_native import (
    compare_noisy_and_mitigated_expectation,
    create_expectation_executor_from_aer_native,
    expectation_from_counts,
    linear_zero_noise_extrapolate,
    polynomial_zero_noise_extrapolate,
    run_noisy_expectation_native,
    run_zne_native,
    scale_noise_model_educational,
    zne_result_to_dict,
)

__all__ = [
    "MITIQ_WARNING",
    "NATIVE_READOUT_WARNING",
    "NATIVE_ZNE_WARNING",
    "UPSTREAM_MITIQ_WARNING",
    "apply_readout_error_to_counts",
    "bell_circuit",
    "build_single_qubit_readout_calibration_matrix",
    "build_tensor_product_readout_matrix",
    "compare_noisy_and_mitigated_expectation",
    "create_expectation_executor_from_aer_native",
    "dependency_available",
    "expectation_from_counts",
    "get_upstream_version",
    "linear_zero_noise_extrapolate",
    "mitigate_readout_counts",
    "mitiq_warnings",
    "normalize_counts_to_probabilities",
    "polynomial_zero_noise_extrapolate",
    "run_noisy_expectation_native",
    "run_readout_mitigation_example",
    "run_readout_mitigation_native",
    "run_readout_mitigation_upstream_if_available",
    "run_zne_example",
    "run_zne_native",
    "run_zne_upstream_if_available",
    "scale_noise_model_educational",
    "validate_mitiq_dependencies",
    "wrap_mitiq_result",
    "wrap_upstream_mitiq_result",
    "zne_result_to_dict",
]
