# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Machine Learning was copied.
"""Deterministic toy datasets for educational native QML workflows."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def make_xor_dataset(labels: str = "zero_one") -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic 2D XOR dataset."""

    x_values = np.asarray(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    y_values = np.asarray([0, 1, 1, 0], dtype=int)
    if labels == "minus_one_one":
        y_values = np.where(y_values == 1, 1, -1)
    elif labels != "zero_one":
        raise ValueError("labels must be zero_one or minus_one_one")
    return x_values, y_values


def make_toy_binary_classification_dataset(labels: str = "zero_one") -> tuple[np.ndarray, np.ndarray]:
    """Return a small deterministic linearly separable 2D dataset."""

    x_values = np.asarray(
        [
            [-1.0, -0.8],
            [-0.8, -1.0],
            [-0.6, -0.7],
            [0.6, 0.7],
            [0.8, 1.0],
            [1.0, 0.8],
        ],
        dtype=float,
    )
    y_values = np.asarray([0, 0, 0, 1, 1, 1], dtype=int)
    if labels == "minus_one_one":
        y_values = np.where(y_values == 1, 1, -1)
    elif labels != "zero_one":
        raise ValueError("labels must be zero_one or minus_one_one")
    return x_values, y_values


def normalize_features_for_angles(features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Scale 2D features deterministically into angles in [-pi/2, pi/2]."""

    array = np.asarray(features, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] not in {1, 2}:
        raise ValueError("features must be shaped as n x 1 or n x 2")
    max_abs = float(np.max(np.abs(array))) if array.size else 1.0
    scale = max(max_abs, 1.0)
    return (array / scale) * (np.pi / 2.0)


def split_toy_dataset(
    x_values: Sequence[Sequence[float]] | np.ndarray,
    y_values: Sequence[int] | np.ndarray,
    train_size: float | int = 0.67,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically split a toy dataset without shuffling."""

    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=int)
    if len(x_array) != len(y_array):
        raise ValueError("x_values and y_values must have the same length")
    if isinstance(train_size, float):
        if not 0.0 < train_size < 1.0:
            raise ValueError("float train_size must be between 0 and 1")
        split = int(round(len(x_array) * train_size))
    else:
        split = int(train_size)
    split = max(1, min(split, len(x_array) - 1))
    return x_array[:split], x_array[split:], y_array[:split], y_array[split:]


def dataset_summary(x_values: Sequence[Sequence[float]] | np.ndarray, y_values: Sequence[int] | np.ndarray) -> dict[str, object]:
    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=int)
    labels, counts = np.unique(y_array, return_counts=True)
    return {
        "samples": int(len(x_array)),
        "features": int(x_array.shape[1]) if x_array.ndim == 2 else 0,
        "labels": {str(int(label)): int(count) for label, count in zip(labels, counts)},
        "deterministic": True,
        "real_user_data": False,
        "network_access": False,
        "high_risk_decision_use": False,
    }
