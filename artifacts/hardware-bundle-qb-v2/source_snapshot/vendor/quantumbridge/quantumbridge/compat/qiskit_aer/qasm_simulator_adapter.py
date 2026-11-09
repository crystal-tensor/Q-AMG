# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Aer was copied.
"""Qasm-style simulator adapter facade."""

from .simulator_native import run_noisy_qasm_simulator_native, run_qasm_simulator_native

__all__ = ["run_noisy_qasm_simulator_native", "run_qasm_simulator_native"]
