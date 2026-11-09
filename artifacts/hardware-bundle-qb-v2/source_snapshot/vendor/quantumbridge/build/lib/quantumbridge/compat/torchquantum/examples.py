# This file is independently implemented for QuantumBridge SDK.
# No source code from TorchQuantum or PyTorch was copied.
"""Convenience examples for Stage 9K TorchQuantum-style workflows."""

from __future__ import annotations

from .qml_dataset_native import make_torchquantum_like_toy_dataset
from .quantum_layer_native import batch_quantum_layer_forward_native, quantum_layer_forward_native
from .torch_adapter import is_torch_available, numpy_to_torch_tensor_if_available, torch_forward_if_available
from .training_native import run_torchquantum_like_classifier_native
from .upstream_adapter import run_upstream_torchquantum_if_available


def run_torchquantum_like_layer_example():
    native = quantum_layer_forward_native([0.25, -0.5], [0.0, 0.0, 0.0, 0.0], num_qubits=2)
    upstream = run_upstream_torchquantum_if_available()
    return {"native": native, "upstream": upstream}


def run_torchquantum_like_batch_forward_example():
    X, _ = make_torchquantum_like_toy_dataset()
    native = batch_quantum_layer_forward_native(X[:3], [0.0, 0.0, 0.0, 0.0], num_qubits=2)
    torch_result = None
    if is_torch_available():
        torch_X = numpy_to_torch_tensor_if_available(X[:3])
        torch_result = torch_forward_if_available(torch_X, [0.0, 0.0, 0.0, 0.0], num_qubits=2)
    upstream = run_upstream_torchquantum_if_available()
    return {"native": native, "torch": torch_result, "upstream": upstream}


def run_torchquantum_like_classifier_example():
    native = run_torchquantum_like_classifier_native()
    upstream = run_upstream_torchquantum_if_available()
    return {"native": native, "upstream": upstream}
