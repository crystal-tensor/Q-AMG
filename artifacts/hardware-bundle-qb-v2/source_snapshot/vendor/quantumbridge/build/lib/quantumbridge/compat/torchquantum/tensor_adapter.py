# This file is independently implemented for QuantumBridge SDK.
# No source code from TorchQuantum or PyTorch was copied.
"""Small TensorLike and batch helpers for PyTorch-style QML workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class TensorLike:
    data: tuple[tuple[float, ...], ...]
    backend: str = "numpy"

    @property
    def shape(self) -> tuple[int, int]:
        if not self.data:
            return (0, 0)
        return (len(self.data), len(self.data[0]))

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self.data, dtype=float)


def is_torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def is_torch_tensor(obj: Any) -> bool:
    if not is_torch_available():
        return False
    import torch

    return isinstance(obj, torch.Tensor)


def torch_tensor_to_numpy(obj: Any) -> np.ndarray:
    if not is_torch_tensor(obj):
        raise TypeError("expected a torch.Tensor")
    return obj.detach().cpu().numpy().astype(float)


def to_numpy_array(data: Any) -> np.ndarray:
    if is_torch_tensor(data):
        array = torch_tensor_to_numpy(data)
    else:
        array = np.asarray(data, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("tensor-like data must be one- or two-dimensional")
    return array.astype(float)


def to_tensor_like(data: Any) -> TensorLike:
    array = to_numpy_array(data)
    backend = "torch" if is_torch_tensor(data) else "numpy"
    return TensorLike(data=tuple(tuple(float(v) for v in row) for row in array), backend=backend)


def numpy_to_torch_tensor_if_available(array: Any) -> Any:
    np_array = to_numpy_array(array)
    if not is_torch_available():
        return np_array
    import torch

    return torch.tensor(np_array, dtype=torch.float64)


def normalize_batch_features(X: Any) -> np.ndarray:
    array = to_numpy_array(X)
    max_abs = float(np.max(np.abs(array))) if array.size else 1.0
    scale = max(max_abs, 1.0)
    return array / scale


def batch_iter(
    X: Any,
    y: Sequence[int] | np.ndarray | None = None,
    batch_size: int = 2,
) -> Iterator[tuple[np.ndarray, np.ndarray | None]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    x_array = to_numpy_array(X)
    y_array = None if y is None else np.asarray(y, dtype=int)
    if y_array is not None and len(x_array) != len(y_array):
        raise ValueError("X and y must have the same length")
    for start in range(0, len(x_array), batch_size):
        stop = start + batch_size
        yield x_array[start:stop], None if y_array is None else y_array[start:stop]
