# This file is independently implemented for QuantumBridge SDK.
# No source code from TorchQuantum or PyTorch was copied.
"""Clean-room TorchQuantum-like / PyTorch-style QML compatibility slice."""

from .dependency import dependency_available, get_upstream_version, validate_torchquantum_dependencies
from .examples import (
    run_torchquantum_like_batch_forward_example,
    run_torchquantum_like_classifier_example,
    run_torchquantum_like_layer_example,
)
from .qml_dataset_native import dataset_summary, make_torchquantum_like_toy_dataset
from .quantum_layer_native import (
    batch_quantum_layer_forward_native,
    build_torchquantum_like_feature_encoder,
    build_torchquantum_like_variational_layer,
    quantum_layer_forward_native,
    quantum_layer_result_to_dict,
    quantum_layer_to_quantumbridge_ir,
)
from .result_adapter import to_quantumbridge_schema, wrap_torchquantum_result
from .tensor_adapter import (
    TensorLike,
    batch_iter,
    is_torch_available,
    is_torch_tensor,
    normalize_batch_features,
    numpy_to_torch_tensor_if_available,
    to_numpy_array,
    to_tensor_like,
    torch_tensor_to_numpy,
)
from .torch_adapter import (
    torch_dataset_to_quantumbridge_batches,
    torch_forward_if_available,
    torch_result_to_numpy,
)
from .training_native import (
    TorchQuantumLikeClassifierModel,
    binary_cross_entropy_like_loss,
    grid_search_train_quantum_layer_native,
    mse_loss,
    predict_quantum_layer_classifier_native,
    run_torchquantum_like_classifier_native,
    score_quantum_layer_classifier_native,
)
from .upstream_adapter import run_upstream_torchquantum_if_available, wrap_upstream_torchquantum_result
from .warnings import (
    HIGH_RISK_ML_WARNING,
    NATIVE_LAYER_WARNING,
    NATIVE_TRAINING_WARNING,
    TORCHQUANTUM_COMPAT_WARNING,
    UPSTREAM_TORCHQUANTUM_WARNING,
    native_provenance,
    torchquantum_warnings,
    upstream_provenance,
)

__all__ = [
    "HIGH_RISK_ML_WARNING",
    "NATIVE_LAYER_WARNING",
    "NATIVE_TRAINING_WARNING",
    "TORCHQUANTUM_COMPAT_WARNING",
    "TorchQuantumLikeClassifierModel",
    "TensorLike",
    "UPSTREAM_TORCHQUANTUM_WARNING",
    "batch_iter",
    "batch_quantum_layer_forward_native",
    "binary_cross_entropy_like_loss",
    "build_torchquantum_like_feature_encoder",
    "build_torchquantum_like_variational_layer",
    "dataset_summary",
    "dependency_available",
    "get_upstream_version",
    "grid_search_train_quantum_layer_native",
    "is_torch_available",
    "is_torch_tensor",
    "make_torchquantum_like_toy_dataset",
    "mse_loss",
    "native_provenance",
    "normalize_batch_features",
    "numpy_to_torch_tensor_if_available",
    "predict_quantum_layer_classifier_native",
    "quantum_layer_forward_native",
    "quantum_layer_result_to_dict",
    "quantum_layer_to_quantumbridge_ir",
    "run_torchquantum_like_batch_forward_example",
    "run_torchquantum_like_classifier_example",
    "run_torchquantum_like_classifier_native",
    "run_torchquantum_like_layer_example",
    "run_upstream_torchquantum_if_available",
    "score_quantum_layer_classifier_native",
    "to_numpy_array",
    "to_quantumbridge_schema",
    "to_tensor_like",
    "torch_dataset_to_quantumbridge_batches",
    "torch_forward_if_available",
    "torch_result_to_numpy",
    "torch_tensor_to_numpy",
    "torchquantum_warnings",
    "upstream_provenance",
    "validate_torchquantum_dependencies",
    "wrap_torchquantum_result",
    "wrap_upstream_torchquantum_result",
]
