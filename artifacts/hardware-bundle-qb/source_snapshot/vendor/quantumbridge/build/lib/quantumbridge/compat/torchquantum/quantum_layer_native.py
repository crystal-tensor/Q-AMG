# This file is independently implemented for QuantumBridge SDK.
# No source code from TorchQuantum or PyTorch was copied.
"""Native TorchQuantum-like quantum layer for tiny educational workflows."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any, Sequence

import numpy as np

from quantumbridge.compat.qiskit_aer.simulator_native import run_statevector_simulator_native
from quantumbridge.core import Circuit
from quantumbridge.schema.torchquantum_results import BatchForwardResult, QuantumLayerResult

from .tensor_adapter import normalize_batch_features, to_numpy_array
from .warnings import NATIVE_LAYER_WARNING, native_provenance, torchquantum_warnings


@dataclass(frozen=True)
class TorchQuantumLikeLayerSpec:
    num_qubits: int = 2
    feature_dim: int = 2
    depth: int = 1

    @property
    def num_weights(self) -> int:
        return 2 * self.num_qubits * self.depth

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "angle_encoder_plus_ry_rz_entangler",
            "num_qubits": self.num_qubits,
            "feature_dim": self.feature_dim,
            "depth": self.depth,
            "num_weights": self.num_weights,
        }


def build_torchquantum_like_feature_encoder(num_qubits: int, feature_dim: int) -> dict[str, Any]:
    _validate_num_qubits(num_qubits)
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive")
    return {
        "type": "deterministic_angle_encoder",
        "num_qubits": int(num_qubits),
        "feature_dim": int(feature_dim),
        "scale": "pi/2",
    }


def build_torchquantum_like_variational_layer(num_qubits: int, depth: int = 1) -> TorchQuantumLikeLayerSpec:
    _validate_num_qubits(num_qubits)
    if depth <= 0:
        raise ValueError("depth must be positive")
    return TorchQuantumLikeLayerSpec(num_qubits=int(num_qubits), feature_dim=int(num_qubits), depth=int(depth))


def quantum_layer_to_quantumbridge_ir(
    features: Sequence[float],
    weights: Sequence[float],
    num_qubits: int = 2,
) -> Circuit:
    _validate_num_qubits(num_qubits)
    feature_array = np.asarray(features, dtype=float).reshape(-1)
    if feature_array.size == 0:
        raise ValueError("features must be non-empty")
    spec = TorchQuantumLikeLayerSpec(num_qubits=int(num_qubits), feature_dim=int(feature_array.size), depth=1)
    weights_array = _normalize_weights(weights, spec.num_weights)
    circuit = Circuit(num_qubits=spec.num_qubits, num_bits=spec.num_qubits, name="torchquantum_like_layer")
    encoded = _encode_features(feature_array, spec.num_qubits)
    for wire, angle in enumerate(encoded):
        circuit.ry(float(angle), wire)
        circuit.rz(float(angle) / 2.0, wire)
    cursor = 0
    for _ in range(spec.depth):
        for wire in range(spec.num_qubits):
            circuit.ry(float(weights_array[cursor]), wire)
            cursor += 1
            circuit.rz(float(weights_array[cursor]), wire)
            cursor += 1
        for wire in range(spec.num_qubits - 1):
            circuit.cx(wire, wire + 1)
    circuit.metadata["torchquantum_like_layer"] = spec.to_dict()
    circuit.metadata["cloud_access"] = False
    circuit.metadata["token_read"] = False
    circuit.metadata["hardware_access"] = False
    return circuit


def quantum_layer_forward_native(
    features: Sequence[float],
    weights: Sequence[float],
    num_qubits: int = 2,
) -> QuantumLayerResult:
    circuit = quantum_layer_to_quantumbridge_ir(features, weights, num_qubits=num_qubits)
    sim_result = run_statevector_simulator_native(circuit)
    probabilities = dict(sim_result.probabilities)
    expectation = _expectation_z_wire0(sim_result.final_statevector)
    probability_one = _probability_wire0_one(probabilities)
    prediction = 1 if probability_one >= 0.5 else 0
    spec = circuit.metadata["torchquantum_like_layer"]
    return QuantumLayerResult(
        workflow="torchquantum_like_layer_forward_native",
        mode="native_minimal",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        tensor_backend="numpy",
        dataset_summary={"single_sample": True, "real_user_data": False, "high_risk_decision_use": False},
        num_qubits=int(num_qubits),
        feature_dim=int(len(np.asarray(features).reshape(-1))),
        batch_size=1,
        weights=[float(value) for value in _normalize_weights(weights, int(spec["num_weights"]))],
        forward_outputs=[float(probability_one)],
        probabilities=[probabilities],
        predictions=[prediction],
        circuit_ir=_circuit_summary(circuit),
        raw_type="QuantumBridgeTorchQuantumLikeLayer",
        metadata={
            "expectation_z_wire0": float(expectation),
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "full_torchquantum_replacement_claim": False,
            "production_qml_claim": False,
            "high_risk_decision_use": False,
        },
        warnings=torchquantum_warnings(NATIVE_LAYER_WARNING),
        provenance=native_provenance("torchquantum_like_layer_forward_native"),
    )


def batch_quantum_layer_forward_native(
    X: Any,
    weights: Sequence[float],
    num_qubits: int = 2,
) -> BatchForwardResult:
    x_array = normalize_batch_features(to_numpy_array(X))
    outputs: list[float] = []
    probabilities: list[dict[str, float]] = []
    predictions: list[int] = []
    first_ir: dict[str, Any] = {}
    for row in x_array:
        result = quantum_layer_forward_native(row, weights, num_qubits=num_qubits)
        outputs.extend(result.forward_outputs)
        probabilities.extend(result.probabilities)
        predictions.extend(result.predictions)
        if not first_ir:
            first_ir = dict(result.circuit_ir)
    return BatchForwardResult(
        workflow="torchquantum_like_batch_forward_native",
        mode="native_minimal",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        tensor_backend="numpy",
        dataset_summary={
            "samples": int(len(x_array)),
            "features": int(x_array.shape[1]),
            "real_user_data": False,
            "high_risk_decision_use": False,
        },
        num_qubits=int(num_qubits),
        feature_dim=int(x_array.shape[1]),
        batch_size=int(len(x_array)),
        weights=[float(value) for value in _normalize_weights(weights, 2 * int(num_qubits))],
        forward_outputs=outputs,
        probabilities=probabilities,
        predictions=predictions,
        circuit_ir=first_ir,
        raw_type="QuantumBridgeTorchQuantumLikeBatchForward",
        metadata={"cloud_access": False, "token_read": False, "hardware_access": False},
        warnings=torchquantum_warnings(NATIVE_LAYER_WARNING),
        provenance=native_provenance("torchquantum_like_batch_forward_native"),
    )


def quantum_layer_result_to_dict(result: QuantumLayerResult | BatchForwardResult) -> dict[str, Any]:
    return result.to_dict()


def _validate_num_qubits(num_qubits: int) -> None:
    if int(num_qubits) not in {1, 2, 3}:
        raise ValueError("Stage 9K native TorchQuantum-like layer supports 1-3 qubits")


def _encode_features(features: np.ndarray, num_qubits: int) -> np.ndarray:
    normalized = features.astype(float)
    max_abs = float(np.max(np.abs(normalized))) if normalized.size else 1.0
    normalized = normalized / max(max_abs, 1.0)
    encoded = np.zeros(num_qubits, dtype=float)
    for wire in range(num_qubits):
        encoded[wire] = normalized[wire % len(normalized)] * (pi / 2.0)
    return encoded


def _normalize_weights(weights: Sequence[float], expected: int) -> np.ndarray:
    array = np.asarray(weights, dtype=float).reshape(-1)
    if array.size == 0:
        array = np.zeros(expected, dtype=float)
    if array.size != expected:
        raise ValueError(f"weights length must be {expected}")
    return array


def _expectation_z_wire0(statevector: Sequence[complex]) -> float:
    value = 0.0
    for index, amplitude in enumerate(statevector):
        probability = float(abs(complex(amplitude)) ** 2)
        value += probability if (index & 1) == 0 else -probability
    return float(value)


def _probability_wire0_one(probabilities: dict[str, float]) -> float:
    total = 0.0
    for bitstring, probability in probabilities.items():
        if bitstring and bitstring[-1] == "1":
            total += float(probability)
    return float(total)


def _circuit_summary(circuit: Circuit) -> dict[str, Any]:
    return {
        "num_qubits": int(circuit.num_qubits),
        "operation_count": len(circuit.operations),
        "operations": [
            {"name": op.name, "targets": list(op.targets), "controls": list(op.controls)}
            for op in circuit.operations
        ],
        "metadata": dict(circuit.metadata),
    }
