# This file is independently implemented for QuantumBridge SDK.
# No source code from Mitiq, Qiskit, or Qiskit Aer was copied.
"""Native educational readout-mitigation workflows."""

from __future__ import annotations

from typing import Any

import numpy as np

from quantumbridge.compat.qiskit_aer import run_qasm_simulator_native
from quantumbridge.compat.qiskit_aer.circuit_execution_adapter import (
    normalize_circuit_to_quantumbridge_ir,
)
from quantumbridge.schema.error_mitigation_results import ReadoutMitigationResult

from .warnings import NATIVE_READOUT_WARNING, mitiq_warnings, native_provenance


def build_single_qubit_readout_calibration_matrix(
    p0to1: float,
    p1to0: float,
) -> list[list[float]]:
    p0to1 = _validate_probability(p0to1, "p0to1")
    p1to0 = _validate_probability(p1to0, "p1to0")
    return [
        [1.0 - p0to1, p1to0],
        [p0to1, 1.0 - p1to0],
    ]


def build_tensor_product_readout_matrix(
    num_qubits: int,
    p0to1: float,
    p1to0: float,
) -> list[list[float]]:
    num_qubits = int(num_qubits)
    if not 1 <= num_qubits <= 3:
        raise ValueError("Stage 9G native readout mitigation supports 1-3 qubits")
    single = np.asarray(
        build_single_qubit_readout_calibration_matrix(p0to1, p1to0), dtype=float
    )
    matrix = single
    for _ in range(num_qubits - 1):
        matrix = np.kron(matrix, single)
    return matrix.tolist()


def apply_readout_error_to_counts(
    counts: dict[str, int],
    p0to1: float,
    p1to0: float,
    shots: int,
    seed: int | None = None,
) -> dict[str, int]:
    p0to1 = _validate_probability(p0to1, "p0to1")
    p1to0 = _validate_probability(p1to0, "p1to0")
    shots = int(shots)
    normalized = {str(key): int(value) for key, value in counts.items() if int(value) > 0}
    if shots <= 0 or sum(normalized.values()) != shots:
        raise ValueError("counts total must match positive shots")
    rng = np.random.default_rng(seed)
    noisy: dict[str, int] = {}
    for bitstring, count in sorted(normalized.items()):
        for _ in range(count):
            bits = []
            for bit in bitstring:
                if bit == "0":
                    bits.append("1" if rng.random() < p0to1 else "0")
                elif bit == "1":
                    bits.append("0" if rng.random() < p1to0 else "1")
                else:
                    raise ValueError("bitstrings must contain only 0 or 1")
            label = "".join(bits)
            noisy[label] = noisy.get(label, 0) + 1
    return noisy


def mitigate_readout_counts(
    counts: dict[str, int],
    calibration_matrix: list[list[float]],
    bitstrings: list[str] | None = None,
) -> dict[str, float]:
    normalized = normalize_counts_to_probabilities(counts)
    labels = bitstrings or _labels_for_counts(counts)
    observed = np.asarray([normalized.get(label, 0.0) for label in labels], dtype=float)
    matrix = np.asarray(calibration_matrix, dtype=float)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("calibration_matrix shape must match bitstrings")
    mitigated = np.linalg.pinv(matrix, rcond=1e-10) @ observed
    mitigated = np.clip(mitigated, 0.0, None)
    total = float(mitigated.sum())
    if total <= 0:
        mitigated = np.full(len(labels), 1.0 / len(labels))
    else:
        mitigated = mitigated / total
    return {label: float(value) for label, value in zip(labels, mitigated)}


def normalize_counts_to_probabilities(counts: dict[str, int]) -> dict[str, float]:
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        raise ValueError("counts must contain positive shots")
    return {str(key): int(value) / total for key, value in counts.items() if int(value) > 0}


def run_readout_mitigation_native(
    circuit_or_ir: Any,
    p0to1: float = 0.05,
    p1to0: float = 0.05,
    shots: int = 1024,
    seed: int | None = None,
) -> ReadoutMitigationResult:
    circuit = normalize_circuit_to_quantumbridge_ir(circuit_or_ir)
    if not 1 <= int(circuit.num_qubits) <= 3:
        raise ValueError("Stage 9G native readout mitigation supports 1-3 qubits")
    raw = run_qasm_simulator_native(circuit, shots=shots, seed=seed)
    num_bits = len(next(iter(raw.counts))) if raw.counts else int(circuit.num_qubits)
    labels = [format(index, f"0{num_bits}b") for index in range(1 << num_bits)]
    matrix = build_tensor_product_readout_matrix(num_bits, p0to1, p1to0)
    noisy_counts = apply_readout_error_to_counts(
        raw.counts, p0to1=p0to1, p1to0=p1to0, shots=raw.shots or shots, seed=seed
    )
    mitigated = mitigate_readout_counts(noisy_counts, matrix, labels)
    return ReadoutMitigationResult(
        workflow="readout_mitigation_native",
        mode="native_minimal",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        circuit_summary={
            "num_qubits": int(circuit.num_qubits),
            "num_bits": int(circuit.num_bits),
            "operation_count": len(circuit.operations),
            "measurement_count": len(circuit.measurements),
        },
        shots=raw.shots,
        seed=seed,
        noise_model={
            "type": "readout_bitflip",
            "p0to1": float(p0to1),
            "p1to0": float(p1to0),
            "educational_only": True,
            "production_ready": False,
        },
        raw_counts=raw.counts,
        noisy_counts=noisy_counts,
        mitigated_probabilities=mitigated,
        calibration_matrix=matrix,
        raw_type="QuantumBridgeReadoutMitigation",
        metadata={
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "hardware_calibration": False,
            "production_error_mitigation": False,
            "mitiq_parity_claim": False,
        },
        warnings=mitiq_warnings(NATIVE_READOUT_WARNING),
        provenance=native_provenance("readout_mitigation_native"),
    )


def _labels_for_counts(counts: dict[str, int]) -> list[str]:
    if not counts:
        raise ValueError("counts must be non-empty")
    width = len(next(iter(counts)))
    return [format(index, f"0{width}b") for index in range(1 << width)]


def _validate_probability(value: float, name: str) -> float:
    item = float(value)
    if not 0.0 <= item <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return item
