# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Algorithms was copied.
"""Small QuantumBridge-owned examples for Qiskit Algorithms compatibility."""

from quantumbridge.compat.qiskit_algorithms.algorithms_native import (
    run_grover_native,
    run_qaoa_native_maxcut,
    run_vqe_native,
)


def run_all_native_algorithm_examples() -> dict[str, dict]:
    """Run Stage 9C native examples and return serializable payloads."""

    return {
        "vqe": run_vqe_native().to_dict(),
        "qaoa": run_qaoa_native_maxcut(((0, 1), (1, 2), (2, 0)), num_nodes=3).to_dict(),
        "grover": run_grover_native(["11"], num_qubits=2).to_dict(),
    }


__all__ = ["run_all_native_algorithm_examples"]
