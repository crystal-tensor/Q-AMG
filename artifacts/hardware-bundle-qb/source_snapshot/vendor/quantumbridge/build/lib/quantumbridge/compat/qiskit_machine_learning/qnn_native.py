# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Machine Learning was copied.
"""Native educational QNN forward-pass helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from quantumbridge.core import Circuit
from quantumbridge.devices import StatevectorDevice
from quantumbridge.schema.ml_results import QNNForwardResult

from .feature_map_native import build_angle_feature_map, describe_feature_map
from .warnings import NATIVE_QNN_WARNING, ml_warnings, native_provenance


@dataclass(frozen=True)
class NativeQNNAnsatz:
    """Small RX/RY/RZ ansatz descriptor for Stage 9E."""

    num_qubits: int = 2
    depth: int = 1

    @property
    def num_parameters(self) -> int:
        return 2 * self.num_qubits * self.depth

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "feature_encoding_plus_rx_ry_linear_entangler",
            "num_qubits": self.num_qubits,
            "depth": self.depth,
            "num_parameters": self.num_parameters,
        }


def build_qnn_ansatz(num_qubits: int = 2, depth: int = 1) -> NativeQNNAnsatz:
    if num_qubits not in {1, 2}:
        raise ValueError("Stage 9E native QNN supports 1 or 2 qubits")
    if depth <= 0:
        raise ValueError("depth must be positive")
    return NativeQNNAnsatz(num_qubits=int(num_qubits), depth=int(depth))


def qnn_circuit(features: Sequence[float], weights: Sequence[float], ansatz: NativeQNNAnsatz | None = None) -> Circuit:
    ansatz = ansatz or build_qnn_ansatz(num_qubits=len(features), depth=1)
    if len(weights) != ansatz.num_parameters:
        raise ValueError("weights length must match the ansatz parameter count")
    circuit = build_angle_feature_map(features, num_qubits=ansatz.num_qubits)
    cursor = 0
    for _ in range(ansatz.depth):
        for wire in range(ansatz.num_qubits):
            circuit.ry(float(weights[cursor]), wire)
            cursor += 1
            circuit.rx(float(weights[cursor]), wire)
            cursor += 1
        if ansatz.num_qubits == 2:
            circuit.cx(0, 1)
    circuit.metadata["qnn_ansatz"] = ansatz.to_dict()
    return circuit


def qnn_expectation_z(features: Sequence[float], weights: Sequence[float]) -> float:
    """Return expectation of Z on wire 0 for a native QNN circuit."""

    circuit = qnn_circuit(features, weights)
    state = StatevectorDevice().statevector(circuit)
    expectation = 0.0
    for index, amplitude in enumerate(state):
        probability = float(abs(amplitude) ** 2)
        z_value = 1.0 if (index & 1) == 0 else -1.0
        expectation += z_value * probability
    return float(expectation)


def qnn_forward_native(features: Sequence[float], weights: Sequence[float]) -> QNNForwardResult:
    ansatz = build_qnn_ansatz(num_qubits=len(features), depth=1)
    expectation = qnn_expectation_z(features, weights)
    prediction = 1 if expectation >= 0 else 0
    return QNNForwardResult(
        workflow="qnn_forward_native",
        mode="native_minimal",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        dataset_summary={"single_sample": True, "features": len(features), "real_user_data": False},
        feature_map=describe_feature_map(features, num_qubits=ansatz.num_qubits),
        model_summary=ansatz.to_dict(),
        weights=[float(value) for value in weights],
        predictions=[int(prediction)],
        expectation=float(expectation),
        raw_type="NativeQNNForward",
        metadata={
            "observable": "Z(wire=0)",
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "high_risk_decision_use": False,
        },
        warnings=ml_warnings(NATIVE_QNN_WARNING),
        provenance=native_provenance("qnn_forward_native"),
    )
