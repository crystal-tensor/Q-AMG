# This file is independently implemented for QuantumBridge SDK.
# No source code from TorchQuantum or PyTorch was copied.
"""Deterministic educational training for TorchQuantum-like layers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import pi
from typing import Any, Sequence

import numpy as np

from quantumbridge.schema.torchquantum_results import (
    TorchQuantumClassifierResult,
    TorchQuantumTrainingResult,
)

from .qml_dataset_native import dataset_summary, make_torchquantum_like_toy_dataset
from .quantum_layer_native import batch_quantum_layer_forward_native
from .tensor_adapter import normalize_batch_features, to_numpy_array
from .warnings import (
    NATIVE_LAYER_WARNING,
    NATIVE_TRAINING_WARNING,
    native_provenance,
    torchquantum_warnings,
)


@dataclass(frozen=True)
class TorchQuantumLikeClassifierModel:
    weights: tuple[float, ...]
    threshold: float = 0.5
    num_qubits: int = 2
    training_accuracy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": "probability_wire0_one_threshold",
            "threshold": float(self.threshold),
            "num_qubits": int(self.num_qubits),
            "weights": [float(value) for value in self.weights],
            "training_accuracy": float(self.training_accuracy),
        }


def binary_cross_entropy_like_loss(predictions: Sequence[float], labels: Sequence[int]) -> float:
    preds = np.clip(np.asarray(predictions, dtype=float), 1e-9, 1.0 - 1e-9)
    labs = np.asarray(labels, dtype=float)
    if len(preds) != len(labs):
        raise ValueError("prediction and label lengths differ")
    return float(-np.mean(labs * np.log(preds) + (1.0 - labs) * np.log(1.0 - preds)))


def mse_loss(predictions: Sequence[float], labels: Sequence[int]) -> float:
    preds = np.asarray(predictions, dtype=float)
    labs = np.asarray(labels, dtype=float)
    if len(preds) != len(labs):
        raise ValueError("prediction and label lengths differ")
    return float(np.mean((preds - labs) ** 2))


def grid_search_train_quantum_layer_native(
    X: Any,
    y: Sequence[int] | np.ndarray,
    weight_grid: Sequence[float] | None = None,
    num_qubits: int = 2,
) -> TorchQuantumTrainingResult:
    x_array = normalize_batch_features(to_numpy_array(X))
    y_array = np.asarray(y, dtype=int)
    if len(x_array) != len(y_array):
        raise ValueError("X and y must have the same length")
    grid = tuple(float(value) for value in (weight_grid or (-pi / 4.0, 0.0, pi / 4.0)))
    expected_weights = 2 * int(num_qubits)
    best_weights: tuple[float, ...] | None = None
    best_accuracy = -1.0
    best_loss = float("inf")
    best_outputs: list[float] = []
    best_predictions: list[int] = []
    trace: list[dict[str, Any]] = []
    for index, weights in enumerate(product(grid, repeat=expected_weights)):
        batch = batch_quantum_layer_forward_native(x_array, weights, num_qubits=num_qubits)
        predictions = [1 if value >= 0.5 else 0 for value in batch.forward_outputs]
        accuracy = float(np.mean(np.asarray(predictions, dtype=int) == y_array))
        loss = mse_loss(batch.forward_outputs, y_array)
        trace.append(
            {
                "candidate": int(index),
                "weights": [float(value) for value in weights],
                "accuracy": accuracy,
                "loss": loss,
            }
        )
        if accuracy > best_accuracy or (accuracy == best_accuracy and loss < best_loss):
            best_weights = tuple(float(value) for value in weights)
            best_accuracy = accuracy
            best_loss = loss
            best_outputs = [float(value) for value in batch.forward_outputs]
            best_predictions = predictions
    assert best_weights is not None
    return TorchQuantumTrainingResult(
        workflow="torchquantum_like_grid_search_training_native",
        mode="native_minimal",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        tensor_backend="numpy",
        dataset_summary=dataset_summary(x_array, y_array),
        num_qubits=int(num_qubits),
        feature_dim=int(x_array.shape[1]),
        batch_size=int(len(x_array)),
        weights=[float(value) for value in best_weights],
        forward_outputs=best_outputs,
        predictions=best_predictions,
        accuracy=float(best_accuracy),
        loss=float(best_loss),
        training_trace=trace,
        raw_type="QuantumBridgeTorchQuantumLikeTraining",
        metadata={"cloud_access": False, "token_read": False, "hardware_access": False},
        warnings=torchquantum_warnings(NATIVE_LAYER_WARNING, NATIVE_TRAINING_WARNING),
        provenance=native_provenance("torchquantum_like_grid_search_training_native"),
    )


def predict_quantum_layer_classifier_native(model: TorchQuantumLikeClassifierModel | dict[str, Any], X: Any) -> list[int]:
    if isinstance(model, dict):
        weights = model["weights"]
        threshold = float(model.get("threshold", 0.5))
        num_qubits = int(model.get("num_qubits", 2))
    else:
        weights = model.weights
        threshold = model.threshold
        num_qubits = model.num_qubits
    batch = batch_quantum_layer_forward_native(X, weights, num_qubits=num_qubits)
    return [1 if value >= threshold else 0 for value in batch.forward_outputs]


def score_quantum_layer_classifier_native(
    model: TorchQuantumLikeClassifierModel | dict[str, Any],
    X: Any,
    y: Sequence[int] | np.ndarray,
) -> float:
    predictions = np.asarray(predict_quantum_layer_classifier_native(model, X), dtype=int)
    labels = np.asarray(y, dtype=int)
    if len(predictions) != len(labels):
        raise ValueError("prediction and label lengths differ")
    return float(np.mean(predictions == labels))


def run_torchquantum_like_classifier_native() -> TorchQuantumClassifierResult:
    X, y = make_torchquantum_like_toy_dataset()
    training = grid_search_train_quantum_layer_native(X, y, num_qubits=2)
    model = TorchQuantumLikeClassifierModel(
        weights=tuple(training.weights),
        num_qubits=2,
        training_accuracy=float(training.accuracy or 0.0),
    )
    predictions = predict_quantum_layer_classifier_native(model, X)
    accuracy = score_quantum_layer_classifier_native(model, X, y)
    return TorchQuantumClassifierResult(
        workflow="torchquantum_like_classifier_native",
        mode="native_minimal",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        tensor_backend="numpy",
        dataset_summary=dataset_summary(X, y),
        num_qubits=2,
        feature_dim=int(X.shape[1]),
        batch_size=int(len(X)),
        weights=[float(value) for value in model.weights],
        forward_outputs=[float(value) for value in training.forward_outputs],
        predictions=[int(value) for value in predictions],
        accuracy=float(accuracy),
        loss=float(training.loss or 0.0),
        training_trace=list(training.training_trace),
        raw_type="QuantumBridgeTorchQuantumLikeClassifier",
        metadata={
            "model": model.to_dict(),
            "labels": [int(value) for value in y.tolist()],
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
            "high_risk_decision_use": False,
        },
        warnings=torchquantum_warnings(NATIVE_LAYER_WARNING, NATIVE_TRAINING_WARNING),
        provenance=native_provenance("torchquantum_like_classifier_native"),
    )
