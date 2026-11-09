# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Machine Learning was copied.
"""Native educational QNN classifier workflows."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import pi
from typing import Any, Sequence

import numpy as np

from quantumbridge.schema.ml_results import QNNClassifierResult

from .qml_dataset_native import dataset_summary, make_toy_binary_classification_dataset, split_toy_dataset
from .qnn_native import build_qnn_ansatz, qnn_expectation_z
from .warnings import NATIVE_QNN_WARNING, ml_warnings, native_provenance


@dataclass(frozen=True)
class NativeQNNClassifierModel:
    weights: tuple[float, ...]
    threshold: float = 0.0
    ansatz: dict[str, Any] | None = None
    training_accuracy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": "expectation_threshold",
            "threshold": float(self.threshold),
            "weights": [float(value) for value in self.weights],
            "training_accuracy": float(self.training_accuracy),
            "ansatz": dict(self.ansatz or {}),
        }


def predict_qnn_classifier_native(
    model: NativeQNNClassifierModel,
    x_test: Sequence[Sequence[float]] | np.ndarray,
) -> list[int]:
    predictions: list[int] = []
    for row in np.asarray(x_test, dtype=float):
        expectation = qnn_expectation_z(row, model.weights)
        predictions.append(1 if expectation >= model.threshold else 0)
    return predictions


def score_qnn_classifier_native(
    model: NativeQNNClassifierModel,
    x_test: Sequence[Sequence[float]] | np.ndarray,
    y_test: Sequence[int] | np.ndarray,
) -> float:
    predictions = np.asarray(predict_qnn_classifier_native(model, x_test), dtype=int)
    labels = np.asarray(y_test, dtype=int)
    if len(predictions) != len(labels):
        raise ValueError("prediction and label lengths differ")
    return float(np.mean(predictions == labels))


def train_qnn_classifier_grid_search_native(
    x_train: Sequence[Sequence[float]] | np.ndarray,
    y_train: Sequence[int] | np.ndarray,
    weight_grid: Sequence[float] | None = None,
) -> NativeQNNClassifierModel:
    x_array = np.asarray(x_train, dtype=float)
    y_array = np.asarray(y_train, dtype=int)
    if len(x_array) != len(y_array):
        raise ValueError("x_train and y_train must have the same length")
    ansatz = build_qnn_ansatz(num_qubits=x_array.shape[1], depth=1)
    grid = tuple(float(value) for value in (weight_grid or (-pi / 2.0, 0.0, pi / 2.0)))
    best_weights: tuple[float, ...] | None = None
    best_accuracy = -1.0
    for weights in product(grid, repeat=ansatz.num_parameters):
        model = NativeQNNClassifierModel(
            weights=tuple(float(value) for value in weights),
            ansatz=ansatz.to_dict(),
        )
        accuracy = score_qnn_classifier_native(model, x_array, y_array)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = model.weights
    assert best_weights is not None
    return NativeQNNClassifierModel(
        weights=best_weights,
        ansatz=ansatz.to_dict(),
        training_accuracy=float(best_accuracy),
    )


def run_qnn_classifier_native() -> QNNClassifierResult:
    x_values, y_values = make_toy_binary_classification_dataset()
    x_train, x_test, y_train, y_test = split_toy_dataset(x_values, y_values, train_size=4)
    model = train_qnn_classifier_grid_search_native(x_train, y_train)
    predictions = predict_qnn_classifier_native(model, x_test)
    accuracy = score_qnn_classifier_native(model, x_test, y_test)
    return QNNClassifierResult(
        workflow="qnn_classifier_native",
        mode="native_minimal",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        dataset_summary={
            **dataset_summary(x_values, y_values),
            "train_samples": int(len(x_train)),
            "test_samples": int(len(x_test)),
        },
        model_summary=model.to_dict(),
        weights=[float(value) for value in model.weights],
        predictions=[int(value) for value in predictions],
        accuracy=float(accuracy),
        raw_type="NativeQNNClassifierModel",
        metadata={
            "test_labels": [int(value) for value in y_test.tolist()],
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "high_risk_decision_use": False,
        },
        warnings=ml_warnings(NATIVE_QNN_WARNING),
        provenance=native_provenance("qnn_classifier_native"),
    )
