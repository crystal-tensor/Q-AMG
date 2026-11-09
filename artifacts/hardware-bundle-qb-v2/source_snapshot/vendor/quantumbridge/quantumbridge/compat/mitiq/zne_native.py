# This file is independently implemented for QuantumBridge SDK.
# No source code from Mitiq, Qiskit, or Qiskit Aer was copied.
"""Native educational zero-noise extrapolation workflows."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from quantumbridge.compat.qiskit_aer import (
    create_bitflip_noise_model,
    run_noisy_qasm_simulator_native,
    run_statevector_simulator_native,
)
from quantumbridge.compat.qiskit_aer.circuit_execution_adapter import (
    normalize_circuit_to_quantumbridge_ir,
)
from quantumbridge.schema.error_mitigation_results import (
    ErrorMitigationComparisonResult,
    ZNEResult,
)

from .warnings import NATIVE_ZNE_WARNING, mitiq_warnings, native_provenance


def create_expectation_executor_from_aer_native(
    circuit_or_ir: Any,
    observable: str = "Z0",
    shots: int = 1024,
    seed: int | None = None,
    base_noise: float = 0.03,
) -> Callable[[float], float]:
    """Create a deterministic educational expectation executor."""

    def executor(noise_scale: float = 1.0) -> float:
        result = run_noisy_expectation_native(
            circuit_or_ir,
            observable=observable,
            noise_scale=noise_scale,
            base_noise=base_noise,
            shots=shots,
            seed=seed,
        )
        return result.mitigated_expectation_value or 0.0

    return executor


def run_noisy_expectation_native(
    circuit_or_ir: Any,
    observable: str = "Z0",
    noise_scale: float = 1.0,
    base_noise: float = 0.03,
    shots: int = 1024,
    seed: int | None = None,
) -> ZNEResult:
    """Run one noisy expectation-value sample using the Stage 9F simulator."""

    circuit = normalize_circuit_to_quantumbridge_ir(circuit_or_ir)
    _validate_zne_circuit(circuit.num_qubits)
    model = scale_noise_model_educational(create_bitflip_noise_model(base_noise), noise_scale)
    noisy = run_noisy_qasm_simulator_native(circuit, shots=shots, noise_model=model, seed=seed)
    expectation = expectation_from_counts(noisy.counts, observable, shots=noisy.shots or shots)
    ideal = expectation_from_probabilities(
        run_statevector_simulator_native(circuit).probabilities,
        observable,
    )
    return ZNEResult(
        workflow="noisy_expectation_native",
        mode="native_minimal",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        circuit_summary=_circuit_summary(circuit),
        observable=observable,
        shots=noisy.shots,
        seed=seed,
        noise_model=model,
        noise_scales=[float(noise_scale)],
        noisy_expectation_values=[expectation],
        mitigated_expectation_value=expectation,
        ideal_expectation_value=ideal,
        noisy_counts=noisy.counts,
        raw_type="QuantumBridgeNoisyExpectation",
        metadata=_metadata("noisy_expectation_native"),
        warnings=mitiq_warnings(NATIVE_ZNE_WARNING),
        provenance=native_provenance("noisy_expectation_native"),
    )


def scale_noise_model_educational(
    noise_model: dict[str, Any],
    scale_factor: float,
) -> dict[str, Any]:
    scale = float(scale_factor)
    if scale <= 0:
        raise ValueError("scale_factor must be positive")
    data = dict(noise_model)
    p = min(1.0, max(0.0, float(data.get("p", 0.0)) * scale))
    data["p"] = p
    data["scale_factor"] = scale
    data["educational_only"] = True
    data["production_ready"] = False
    return data


def linear_zero_noise_extrapolate(
    noise_scales: Iterable[float],
    expectation_values: Iterable[float],
) -> float:
    scales = _as_float_array(noise_scales, "noise_scales")
    values = _as_float_array(expectation_values, "expectation_values")
    if scales.size != values.size:
        raise ValueError("noise_scales and expectation_values must have the same length")
    if scales.size < 2:
        raise ValueError("at least two noise points are required")
    slope, intercept = np.polyfit(scales, values, 1)
    return float(intercept)


def polynomial_zero_noise_extrapolate(
    noise_scales: Iterable[float],
    expectation_values: Iterable[float],
    degree: int = 1,
) -> float:
    scales = _as_float_array(noise_scales, "noise_scales")
    values = _as_float_array(expectation_values, "expectation_values")
    degree = int(degree)
    if degree < 1:
        raise ValueError("degree must be at least 1")
    if scales.size != values.size:
        raise ValueError("noise_scales and expectation_values must have the same length")
    if scales.size <= degree:
        raise ValueError("need more points than polynomial degree")
    coefficients = np.polyfit(scales, values, degree)
    return float(np.polyval(coefficients, 0.0))


def run_zne_native(
    circuit_or_ir: Any,
    observable: str = "Z0",
    noise_scales: Iterable[float] = (1.0, 2.0, 3.0),
    base_noise: float = 0.03,
    shots: int = 2048,
    seed: int | None = None,
) -> ZNEResult:
    """Run the Stage 9G educational native ZNE workflow."""

    circuit = normalize_circuit_to_quantumbridge_ir(circuit_or_ir)
    _validate_zne_circuit(circuit.num_qubits)
    scales = [float(value) for value in noise_scales]
    if len(scales) < 2:
        raise ValueError("run_zne_native requires at least two noise scales")
    values: list[float] = []
    count_payloads: list[dict[str, int]] = []
    for index, scale in enumerate(scales):
        local_seed = None if seed is None else int(seed) + index
        sample = run_noisy_expectation_native(
            circuit,
            observable=observable,
            noise_scale=scale,
            base_noise=base_noise,
            shots=shots,
            seed=local_seed,
        )
        values.extend(sample.noisy_expectation_values)
        count_payloads.append(sample.noisy_counts)
    mitigated = linear_zero_noise_extrapolate(scales, values)
    ideal = expectation_from_probabilities(
        run_statevector_simulator_native(circuit).probabilities,
        observable,
    )
    return ZNEResult(
        workflow="zne_native",
        mode="native_minimal",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        circuit_summary=_circuit_summary(circuit),
        observable=observable,
        shots=int(shots),
        seed=seed,
        noise_model=create_bitflip_noise_model(base_noise),
        noise_scales=scales,
        noisy_expectation_values=values,
        mitigated_expectation_value=mitigated,
        ideal_expectation_value=ideal,
        noisy_counts=count_payloads[-1],
        raw_type="QuantumBridgeNativeZNE",
        data={"counts_by_scale": count_payloads},
        metadata=_metadata("zne_native"),
        warnings=mitiq_warnings(NATIVE_ZNE_WARNING),
        provenance=native_provenance("zne_native"),
    )


def compare_noisy_and_mitigated_expectation(
    circuit_or_ir: Any,
    observable: str = "Z0",
    base_noise: float = 0.03,
    shots: int = 2048,
    seed: int | None = None,
) -> ErrorMitigationComparisonResult:
    zne = run_zne_native(
        circuit_or_ir,
        observable=observable,
        base_noise=base_noise,
        shots=shots,
        seed=seed,
    )
    first_noisy = zne.noisy_expectation_values[0]
    return ErrorMitigationComparisonResult(
        workflow="compare_noisy_and_mitigated_expectation",
        mode="comparison",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        circuit_summary=zne.circuit_summary,
        observable=observable,
        shots=shots,
        seed=seed,
        noise_model=zne.noise_model,
        noise_scales=zne.noise_scales,
        noisy_expectation_values=zne.noisy_expectation_values,
        mitigated_expectation_value=zne.mitigated_expectation_value,
        ideal_expectation_value=zne.ideal_expectation_value,
        raw_type="QuantumBridgeErrorMitigationComparison",
        data={
            "first_noisy_expectation": first_noisy,
            "mitigation_delta": (zne.mitigated_expectation_value or 0.0) - first_noisy,
        },
        metadata=_metadata("compare_noisy_and_mitigated_expectation"),
        warnings=zne.warnings,
        provenance=native_provenance("compare_noisy_and_mitigated_expectation"),
    )


def zne_result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    raise TypeError("zne_result_to_dict expects an ErrorMitigationResult")


def expectation_from_counts(
    counts: dict[str, int],
    observable: str = "Z0",
    shots: int | None = None,
) -> float:
    total = int(shots or sum(int(value) for value in counts.values()))
    if total <= 0:
        raise ValueError("counts must contain positive shots")
    return sum(_observable_value(label, observable) * int(count) for label, count in counts.items()) / total


def expectation_from_probabilities(
    probabilities: dict[str, float],
    observable: str = "Z0",
) -> float:
    total = float(sum(probabilities.values()))
    if total <= 0:
        raise ValueError("probabilities must be non-empty")
    return sum(_observable_value(label, observable) * float(prob) for label, prob in probabilities.items()) / total


def _observable_value(bitstring: str, observable: str) -> int:
    label = str(bitstring)
    obs = str(observable).upper()
    if obs == "PARITY":
        indices = range(len(label))
    elif obs == "ZZ":
        if len(label) < 2:
            raise ValueError("ZZ observable requires at least two bits")
        indices = (0, 1)
    elif obs.startswith("Z") and obs[1:].isdigit():
        index = int(obs[1:])
        if index >= len(label):
            raise ValueError(f"observable {observable!r} is outside bitstring {bitstring!r}")
        indices = (index,)
    else:
        raise ValueError("observable must be Z0, Z1, ZZ, or PARITY")
    value = 1
    for index in indices:
        value *= 1 if label[index] == "0" else -1
    return value


def _validate_zne_circuit(num_qubits: int) -> None:
    if not 1 <= int(num_qubits) <= 2:
        raise ValueError("Stage 9G native ZNE supports 1-2 qubit small circuits")


def _circuit_summary(circuit: Any) -> dict[str, Any]:
    return {
        "num_qubits": int(circuit.num_qubits),
        "num_bits": int(circuit.num_bits),
        "operation_count": len(circuit.operations),
        "measurement_count": len(circuit.measurements),
    }


def _metadata(workflow: str) -> dict[str, Any]:
    return {
        "workflow": workflow,
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
        "production_error_mitigation": False,
        "mitiq_parity_claim": False,
    }


def _as_float_array(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray([float(value) for value in values], dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D sequence")
    return array
