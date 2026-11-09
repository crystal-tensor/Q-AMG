# This file is independently implemented for QuantumBridge SDK.
# No source code from TorchQuantum or PyTorch was copied.
"""Deterministic toy data for TorchQuantum-style educational QML."""

from __future__ import annotations

from typing import Any

import numpy as np

from .tensor_adapter import normalize_batch_features


def make_torchquantum_like_toy_dataset() -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(
        [
            [-1.0, -0.8],
            [-0.7, -0.5],
            [-0.4, -0.9],
            [0.4, 0.7],
            [0.8, 0.5],
            [1.0, 0.9],
        ],
        dtype=float,
    )
    y = np.asarray([0, 0, 0, 1, 1, 1], dtype=int)
    return normalize_batch_features(X), y


def dataset_summary(X: Any, y: Any | None = None) -> dict[str, Any]:
    x_array = np.asarray(X, dtype=float)
    if x_array.ndim == 1:
        x_array = x_array.reshape(1, -1)
    summary: dict[str, Any] = {
        "samples": int(len(x_array)),
        "features": int(x_array.shape[1]) if x_array.ndim == 2 else 0,
        "deterministic": True,
        "real_user_data": False,
        "network_access": False,
        "high_risk_decision_use": False,
    }
    if y is not None:
        y_array = np.asarray(y, dtype=int)
        labels, counts = np.unique(y_array, return_counts=True)
        summary["labels"] = {str(int(label)): int(count) for label, count in zip(labels, counts)}
    return summary
