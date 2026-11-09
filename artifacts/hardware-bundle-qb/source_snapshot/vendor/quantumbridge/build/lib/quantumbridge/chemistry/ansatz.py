# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_upstream_integration_strategy_v0.1.md.

from quantumbridge.core import Circuit


def hardware_efficient_chemistry_ansatz(num_qubits: int, depth: int = 1):
    def build(parameters):
        circuit = Circuit(num_qubits)
        index = 0
        for _ in range(depth):
            for wire in range(num_qubits):
                circuit.ry(float(parameters[index]), wire)
                index += 1
            for wire in range(num_qubits - 1):
                circuit.cx(wire, wire + 1)
        return circuit

    return build
