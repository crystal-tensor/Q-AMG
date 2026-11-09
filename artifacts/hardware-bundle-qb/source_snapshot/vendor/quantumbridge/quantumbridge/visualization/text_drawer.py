# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.


def draw_text(circuit) -> str:
    lines = [f"q{wire}: " for wire in range(circuit.num_qubits)]
    for op in circuit.operations:
        label = op.name.upper()
        for wire in range(circuit.num_qubits):
            marker = label if wire in op.wires else "--"
            lines[wire] += f"{marker} "
    return "\n".join(lines)

