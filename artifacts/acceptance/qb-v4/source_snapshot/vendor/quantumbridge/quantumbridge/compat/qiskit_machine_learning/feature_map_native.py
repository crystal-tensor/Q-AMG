# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Machine Learning was copied.
"""Native educational feature maps for Qiskit Machine Learning compatibility."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from quantumbridge.core import Circuit
from quantumbridge.devices import StatevectorDevice

from .qml_dataset_native import normalize_features_for_angles
from .warnings import NATIVE_KERNEL_WARNING, ml_warnings, native_provenance


def build_angle_feature_map(features: Sequence[float], num_qubits: int | None = None) -> Circuit:
    """Build a small RX/RY/RZ/CX angle feature map circuit."""

    angles = normalize_features_for_angles(np.asarray(features, dtype=float)).reshape(-1)
    qubits = int(num_qubits or len(angles))
    if qubits not in {1, 2}:
        raise ValueError("Stage 9E native feature maps support 1 or 2 qubits")
    if len(angles) > qubits:
        raise ValueError("feature count cannot exceed num_qubits")
    padded = np.zeros(qubits, dtype=float)
    padded[: len(angles)] = angles
    circuit = Circuit(
        qubits,
        name="qb_angle_feature_map",
        metadata={
            "feature_map": "angle_rx_ry_rz_linear_entangler",
            "normalized_angles": [float(value) for value in padded],
            "warnings": ml_warnings(NATIVE_KERNEL_WARNING),
            "provenance": native_provenance("angle_feature_map"),
        },
    )
    for wire, value in enumerate(padded):
        circuit.ry(float(value), wire)
        circuit.rz(float(value / 2.0), wire)
    if qubits == 2:
        circuit.cx(0, 1)
        circuit.rx(float(padded[0] * padded[1] / np.pi if np.pi else 0.0), 1)
    return circuit


def feature_map_to_quantumbridge_ir(features: Sequence[float]) -> Any:
    return build_angle_feature_map(features).to_ir()


def feature_map_statevector(features: Sequence[float], num_qubits: int | None = None) -> np.ndarray:
    circuit = build_angle_feature_map(features, num_qubits=num_qubits)
    return StatevectorDevice().statevector(circuit)


def describe_feature_map(features: Sequence[float], num_qubits: int | None = None) -> dict[str, Any]:
    circuit = build_angle_feature_map(features, num_qubits=num_qubits)
    return {
        "name": circuit.name,
        "num_qubits": circuit.num_qubits,
        "operations": [
            {
                "name": op.name,
                "targets": list(op.targets),
                "controls": list(op.controls),
                "params": [float(value) for value in op.params],
            }
            for op in circuit.operations
        ],
        "metadata": dict(circuit.metadata),
    }
