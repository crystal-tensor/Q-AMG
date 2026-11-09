# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Machine Learning was copied.
"""Executable examples for Stage 9E Qiskit Machine Learning compatibility."""

from __future__ import annotations

from .kernel_classifier_native import run_kernel_classifier_native
from .qnn_classifier_native import run_qnn_classifier_native
from .quantum_kernel_native import run_quantum_kernel_native
from .upstream_adapter import (
    run_upstream_classifier_if_available,
    run_upstream_qnn_if_available,
    run_upstream_quantum_kernel_if_available,
)


def run_quantum_kernel_example() -> dict[str, object]:
    native = run_quantum_kernel_native()
    upstream = run_upstream_quantum_kernel_if_available()
    return {"native": native, "upstream": upstream}


def run_kernel_classifier_example() -> dict[str, object]:
    native = run_kernel_classifier_native()
    upstream = run_upstream_classifier_if_available()
    return {"native": native, "upstream": upstream}


def run_qnn_classifier_example() -> dict[str, object]:
    native = run_qnn_classifier_native()
    upstream = run_upstream_qnn_if_available()
    return {"native": native, "upstream": upstream}
