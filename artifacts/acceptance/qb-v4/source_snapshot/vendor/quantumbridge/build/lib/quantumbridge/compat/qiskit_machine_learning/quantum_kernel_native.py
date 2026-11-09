# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Machine Learning was copied.
"""Native educational quantum-kernel workflows."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from quantumbridge.information.fidelity import state_fidelity
from quantumbridge.schema.ml_results import QuantumKernelResult

from .feature_map_native import describe_feature_map, feature_map_statevector
from .qml_dataset_native import dataset_summary, make_xor_dataset
from .warnings import NATIVE_KERNEL_WARNING, ml_warnings, native_provenance


def compute_state_fidelity(state_a: Sequence[complex], state_b: Sequence[complex]) -> float:
    return float(state_fidelity(state_a, state_b))


def compute_quantum_kernel_matrix(
    x_left: Sequence[Sequence[float]] | np.ndarray,
    x_right: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> np.ndarray:
    left = np.asarray(x_left, dtype=float)
    right = left if x_right is None else np.asarray(x_right, dtype=float)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("kernel inputs must be matrices")
    if left.shape[1] != right.shape[1]:
        raise ValueError("kernel inputs must have the same feature dimension")
    qubits = left.shape[1]
    left_states = [feature_map_statevector(row, num_qubits=qubits) for row in left]
    right_states = [feature_map_statevector(row, num_qubits=qubits) for row in right]
    matrix = np.zeros((len(left_states), len(right_states)), dtype=float)
    for i, left_state in enumerate(left_states):
        for j, right_state in enumerate(right_states):
            matrix[i, j] = compute_state_fidelity(left_state, right_state)
    return matrix


def quantum_kernel_result_to_dict(result: QuantumKernelResult) -> dict[str, object]:
    return result.to_dict()


def run_quantum_kernel_native(
    x_values: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> QuantumKernelResult:
    if x_values is None:
        x_values, y_values = make_xor_dataset()
    else:
        y_values = np.zeros(len(x_values), dtype=int)
    x_array = np.asarray(x_values, dtype=float)
    matrix = compute_quantum_kernel_matrix(x_array)
    return QuantumKernelResult(
        workflow="quantum_kernel_native",
        mode="native_minimal",
        capability_level=3,
        upstream_package=None,
        production_ready=False,
        native_implementation=True,
        dataset_summary=dataset_summary(x_array, y_values),
        feature_map=describe_feature_map(x_array[0]),
        kernel_matrix=[[float(value) for value in row] for row in matrix.tolist()],
        model_summary={
            "kernel": "state_fidelity",
            "symmetric": bool(np.allclose(matrix, matrix.T, atol=1e-10)),
            "diagonal_near_one": bool(np.allclose(np.diag(matrix), np.ones(len(matrix)), atol=1e-10)),
        },
        raw_type="NativeQuantumKernelMatrix",
        metadata={
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "high_risk_decision_use": False,
        },
        warnings=ml_warnings(NATIVE_KERNEL_WARNING),
        provenance=native_provenance("quantum_kernel_native"),
    )
