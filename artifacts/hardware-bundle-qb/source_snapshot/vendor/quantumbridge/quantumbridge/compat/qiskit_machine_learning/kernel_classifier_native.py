# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Machine Learning was copied.
"""Native educational quantum-kernel classifier workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from quantumbridge.schema.ml_results import KernelClassifierResult

from .qml_dataset_native import dataset_summary, make_toy_binary_classification_dataset, split_toy_dataset
from .quantum_kernel_native import compute_quantum_kernel_matrix
from .warnings import NATIVE_KERNEL_WARNING, ml_warnings, native_provenance


@dataclass(frozen=True)
class NativeKernelClassifierModel:
    """Nearest-neighbor classifier in native quantum-kernel similarity space."""

    x_train: tuple[tuple[float, ...], ...]
    y_train: tuple[int, ...]
    rule: str = "nearest_kernel_similarity"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "train_samples": len(self.x_train),
            "labels": sorted({int(label) for label in self.y_train}),
        }


def train_kernel_classifier_native(
    x_train: Sequence[Sequence[float]] | np.ndarray,
    y_train: Sequence[int] | np.ndarray,
) -> NativeKernelClassifierModel:
    x_array = np.asarray(x_train, dtype=float)
    y_array = np.asarray(y_train, dtype=int)
    if len(x_array) != len(y_array):
        raise ValueError("x_train and y_train must have the same length")
    if len(set(int(value) for value in y_array)) < 2:
        raise ValueError("kernel classifier requires at least two labels")
    return NativeKernelClassifierModel(
        x_train=tuple(tuple(float(value) for value in row) for row in x_array),
        y_train=tuple(int(value) for value in y_array),
    )


def predict_kernel_classifier_native(
    model: NativeKernelClassifierModel,
    x_test: Sequence[Sequence[float]] | np.ndarray,
) -> list[int]:
    x_train = np.asarray(model.x_train, dtype=float)
    x_array = np.asarray(x_test, dtype=float)
    kernel = compute_quantum_kernel_matrix(x_array, x_train)
    predictions: list[int] = []
    for row in kernel:
        best_index = int(np.argmax(row))
        predictions.append(int(model.y_train[best_index]))
    return predictions


def score_kernel_classifier_native(
    model: NativeKernelClassifierModel,
    x_test: Sequence[Sequence[float]] | np.ndarray,
    y_test: Sequence[int] | np.ndarray,
) -> float:
    y_array = np.asarray(y_test, dtype=int)
    predictions = np.asarray(predict_kernel_classifier_native(model, x_test), dtype=int)
    if len(predictions) != len(y_array):
        raise ValueError("prediction and label lengths differ")
    return float(np.mean(predictions == y_array))


def run_kernel_classifier_native() -> KernelClassifierResult:
    x_values, y_values = make_toy_binary_classification_dataset()
    x_train, x_test, y_train, y_test = split_toy_dataset(x_values, y_values, train_size=4)
    model = train_kernel_classifier_native(x_train, y_train)
    predictions = predict_kernel_classifier_native(model, x_test)
    accuracy = score_kernel_classifier_native(model, x_test, y_test)
    train_kernel = compute_quantum_kernel_matrix(x_train)
    return KernelClassifierResult(
        workflow="kernel_classifier_native",
        mode="native_minimal",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        dataset_summary={
            **dataset_summary(x_values, y_values),
            "train_samples": int(len(x_train)),
            "test_samples": int(len(x_test)),
        },
        kernel_matrix=[[float(value) for value in row] for row in train_kernel.tolist()],
        model_summary=model.to_dict(),
        predictions=[int(value) for value in predictions],
        accuracy=float(accuracy),
        raw_type="NativeKernelClassifierModel",
        metadata={
            "test_labels": [int(value) for value in y_test.tolist()],
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "high_risk_decision_use": False,
        },
        warnings=ml_warnings(NATIVE_KERNEL_WARNING),
        provenance=native_provenance("kernel_classifier_native"),
    )
