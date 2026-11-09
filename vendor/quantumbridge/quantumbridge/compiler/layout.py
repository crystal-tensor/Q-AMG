# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.


def identity_layout(num_qubits: int) -> dict[int, int]:
    return {wire: wire for wire in range(num_qubits)}


class Layout:
    def __init__(self, mapping):
        from numbers import Integral
        pairs = dict(enumerate(mapping)) if isinstance(mapping, (list, tuple)) else dict(mapping)
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for pair in pairs.items() for value in pair): raise ValueError("Layout indices must be non-negative integers.")
        self.mapping = {int(logical): int(physical) for logical, physical in pairs.items()}
        if len(set(self.mapping.values())) != len(self.mapping): raise ValueError("Layout physical assignments must be one-to-one.")

    @classmethod
    def trivial(cls, num_qubits: int) -> "Layout":
        return cls(identity_layout(num_qubits))

    def physical(self, logical: int) -> int:
        return self.mapping[int(logical)]

    def to_dict(self) -> dict[int, int]:
        return dict(self.mapping)

    def inverse(self): return {physical: logical for logical, physical in self.mapping.items()}

    def validate(self, num_logical, target):
        if set(self.mapping) != set(range(num_logical)): raise ValueError("Layout must assign every logical qubit exactly once.")
        if target.num_qubits and max(self.mapping.values(), default=-1) >= target.num_qubits: raise ValueError("Layout exceeds target capacity.")
        return True


def select_initial_layout(circuit, target):
    if target.num_qubits and circuit.num_qubits > target.num_qubits: raise ValueError("Circuit exceeds target capacity.")
    if target.coupling_map is None or not target.coupling_map.edges: return Layout.trivial(circuit.num_qubits)
    counts = {wire: 0 for wire in range(circuit.num_qubits)}
    for op in circuit.operations:
        if len(op.wires) > 1:
            for wire in op.wires: counts[wire] += 1
    logical = sorted(counts, key=lambda wire: (-counts[wire], wire))
    physical = sorted(range(target.num_qubits or circuit.num_qubits), key=lambda wire: (-len(target.coupling_map.neighbors(wire)), wire))
    return Layout(dict(zip(logical, physical)))


def apply_layout(circuit, layout, target):
    layout = layout if isinstance(layout, Layout) else Layout(layout)
    layout.validate(circuit.num_qubits, target)
    from quantumbridge.core import Circuit
    topology_width = max((max(edge) + 1 for edge in target.coupling_map.edges), default=0) if target.coupling_map is not None else 0
    width = target.num_qubits or max(max(layout.mapping.values(), default=-1) + 1, topology_width)
    from quantumbridge.core.circuit_metadata import empty_execution_circuit, execution_metadata
    out = empty_execution_circuit(circuit,num_qubits=width)
    execution_metadata(out).pop("classical_scope",None)
    out.global_phase = 0.
    out.compose(circuit, qubits=[layout.physical(index) for index in range(circuit.num_qubits)], clbits=range(circuit.num_bits),inplace=True)
    execution_metadata(out)["initial_layout"] = layout.to_dict()
    return out
