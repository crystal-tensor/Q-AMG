# This file is independently implemented for QuantumBridge SDK.
# No source code from TorchQuantum or PyTorch was copied.
"""Optional PyTorch tensor helpers for Stage 9K."""

from __future__ import annotations

from typing import Any, Sequence

from quantumbridge.schema.torchquantum_results import BatchForwardResult

from .quantum_layer_native import batch_quantum_layer_forward_native
from .tensor_adapter import (
    batch_iter,
    is_torch_available,
    is_torch_tensor,
    numpy_to_torch_tensor_if_available,
    to_numpy_array,
    torch_tensor_to_numpy,
)


def torch_forward_if_available(X_torch: Any, weights_torch: Any, num_qubits: int = 2) -> BatchForwardResult:
    if not is_torch_available() or not is_torch_tensor(X_torch):
        return BatchForwardResult(
            workflow="torchquantum_like_torch_forward_optional",
            mode="torch_optional",
            capability_level=1,
            production_ready=False,
            native_implementation=False,
            tensor_backend="torch_unavailable",
            unsupported_reason="torch is not installed or X_torch is not a torch.Tensor",
            warnings=[
                "Optional torch tensor path skipped because torch is unavailable or input is not a torch.Tensor."
            ],
            provenance={"cloud_access": False, "token_read": False, "hardware_access": False},
        )
    X = torch_tensor_to_numpy(X_torch)
    weights = torch_tensor_to_numpy(weights_torch).reshape(-1) if is_torch_tensor(weights_torch) else to_numpy_array(weights_torch).reshape(-1)
    result = batch_quantum_layer_forward_native(X, weights, num_qubits=num_qubits)
    result.mode = "torch_optional"
    result.tensor_backend = "torch"
    result.metadata["torch_optional_path"] = True
    return result


def torch_dataset_to_quantumbridge_batches(X: Any, y: Sequence[int] | None = None, batch_size: int = 2):
    return list(batch_iter(to_numpy_array(X), y=y, batch_size=batch_size))


def torch_result_to_numpy(value: Any):
    return torch_tensor_to_numpy(value) if is_torch_tensor(value) else to_numpy_array(value)


__all__ = [
    "is_torch_available",
    "is_torch_tensor",
    "numpy_to_torch_tensor_if_available",
    "torch_dataset_to_quantumbridge_batches",
    "torch_forward_if_available",
    "torch_result_to_numpy",
]
