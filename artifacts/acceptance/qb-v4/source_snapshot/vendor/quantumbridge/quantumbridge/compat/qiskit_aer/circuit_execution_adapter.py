# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Aer was copied.
"""Circuit execution helpers for the Qiskit Aer compatibility slice."""

from .simulator_native import execute_basic_circuit_native, normalize_circuit_to_quantumbridge_ir

__all__ = ["execute_basic_circuit_native", "normalize_circuit_to_quantumbridge_ir"]
