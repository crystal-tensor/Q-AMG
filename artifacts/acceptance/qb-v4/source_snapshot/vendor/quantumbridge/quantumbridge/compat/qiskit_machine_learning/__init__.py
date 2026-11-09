# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Machine Learning optional adapter scaffold.

Design source: docs/compat/strategy/qiskit_machine_learning_compatibility_strategy.md.
"""

from quantumbridge.compat.qiskit_machine_learning.classifier_adapter import ADAPTER as classifier_adapter
from quantumbridge.compat.qiskit_machine_learning.dataset_adapter import ADAPTER as dataset_adapter
from quantumbridge.compat.qiskit_machine_learning.kernel_adapter import ADAPTER as kernel_adapter
from quantumbridge.compat.qiskit_machine_learning.qnn_adapter import ADAPTER as qnn_adapter
from quantumbridge.compat.qiskit_machine_learning.regressor_adapter import ADAPTER as regressor_adapter
from quantumbridge.compat.qiskit_machine_learning.torch_connector_adapter import ADAPTER as torch_connector_adapter
from quantumbridge.compat.qiskit_common import QiskitAdapterFacade
from quantumbridge.compat.qiskit_machine_learning.examples import (
    run_kernel_classifier_example,
    run_qnn_classifier_example,
    run_quantum_kernel_example,
)
from quantumbridge.compat.qiskit_machine_learning.feature_map_native import (
    build_angle_feature_map,
    describe_feature_map,
    feature_map_statevector,
    feature_map_to_quantumbridge_ir,
)
from quantumbridge.compat.qiskit_machine_learning.kernel_classifier_native import (
    NativeKernelClassifierModel,
    predict_kernel_classifier_native,
    run_kernel_classifier_native,
    score_kernel_classifier_native,
    train_kernel_classifier_native,
)
from quantumbridge.compat.qiskit_machine_learning.qml_dataset_native import (
    dataset_summary,
    make_toy_binary_classification_dataset,
    make_xor_dataset,
    normalize_features_for_angles,
    split_toy_dataset,
)
from quantumbridge.compat.qiskit_machine_learning.qnn_classifier_native import (
    NativeQNNClassifierModel,
    predict_qnn_classifier_native,
    run_qnn_classifier_native,
    score_qnn_classifier_native,
    train_qnn_classifier_grid_search_native,
)
from quantumbridge.compat.qiskit_machine_learning.qnn_native import (
    NativeQNNAnsatz,
    build_qnn_ansatz,
    qnn_expectation_z,
    qnn_forward_native,
)
from quantumbridge.compat.qiskit_machine_learning.quantum_kernel_native import (
    compute_quantum_kernel_matrix,
    compute_state_fidelity,
    quantum_kernel_result_to_dict,
    run_quantum_kernel_native,
)
from quantumbridge.compat.qiskit_machine_learning.upstream_adapter import (
    dependency_available as upstream_dependency_available,
    get_upstream_version as upstream_get_upstream_version,
    run_upstream_classifier_if_available,
    run_upstream_qnn_if_available,
    run_upstream_quantum_kernel_if_available,
    validate_ml_dependencies,
    wrap_upstream_ml_result,
)

ADAPTER = QiskitAdapterFacade(
    ecosystem="qiskit_machine_learning",
    upstream_package="qiskit-machine-learning",
    dependency_extra="qiskit-machine-learning",
    adapters=(qnn_adapter, kernel_adapter, classifier_adapter, regressor_adapter, torch_connector_adapter, dataset_adapter),
    notes="Qiskit Machine Learning bridge; no production ML or training-correctness guarantee.",
)

capability_level = ADAPTER.capability_level
production_ready = ADAPTER.production_ready
native_implementation = ADAPTER.native_implementation
upstream_required = ADAPTER.upstream_required
dependency_available = ADAPTER.dependency_available
get_upstream_version = ADAPTER.get_upstream_version
get_dependency_report = ADAPTER.get_dependency_report
list_public_api_inventory = ADAPTER.list_public_api_inventory
get_public_object = ADAPTER.get_public_object
passthrough_call = ADAPTER.passthrough_call
passthrough_class = ADAPTER.passthrough_class
wrap_result = ADAPTER.wrap_result
to_quantumbridge_schema = ADAPTER.to_quantumbridge_schema
get_warnings = ADAPTER.get_warnings
get_provenance = ADAPTER.get_provenance
unsupported = ADAPTER.unsupported
validate_environment = ADAPTER.validate_environment

__all__ = [
    "ADAPTER",
    "NativeKernelClassifierModel",
    "NativeQNNAnsatz",
    "NativeQNNClassifierModel",
    "capability_level",
    "build_angle_feature_map",
    "build_qnn_ansatz",
    "classifier_adapter",
    "compute_quantum_kernel_matrix",
    "compute_state_fidelity",
    "dataset_summary",
    "dataset_adapter",
    "dependency_available",
    "describe_feature_map",
    "feature_map_statevector",
    "feature_map_to_quantumbridge_ir",
    "get_dependency_report",
    "get_provenance",
    "get_public_object",
    "get_upstream_version",
    "get_warnings",
    "kernel_adapter",
    "list_public_api_inventory",
    "make_toy_binary_classification_dataset",
    "make_xor_dataset",
    "native_implementation",
    "normalize_features_for_angles",
    "passthrough_call",
    "passthrough_class",
    "predict_kernel_classifier_native",
    "predict_qnn_classifier_native",
    "production_ready",
    "qnn_expectation_z",
    "qnn_adapter",
    "qnn_forward_native",
    "quantum_kernel_result_to_dict",
    "regressor_adapter",
    "run_kernel_classifier_example",
    "run_kernel_classifier_native",
    "run_qnn_classifier_example",
    "run_qnn_classifier_native",
    "run_quantum_kernel_example",
    "run_quantum_kernel_native",
    "run_upstream_classifier_if_available",
    "run_upstream_qnn_if_available",
    "run_upstream_quantum_kernel_if_available",
    "score_kernel_classifier_native",
    "score_qnn_classifier_native",
    "split_toy_dataset",
    "train_kernel_classifier_native",
    "train_qnn_classifier_grid_search_native",
    "to_quantumbridge_schema",
    "torch_connector_adapter",
    "unsupported",
    "upstream_dependency_available",
    "upstream_get_upstream_version",
    "upstream_required",
    "validate_environment",
    "validate_ml_dependencies",
    "wrap_result",
    "wrap_upstream_ml_result",
]
