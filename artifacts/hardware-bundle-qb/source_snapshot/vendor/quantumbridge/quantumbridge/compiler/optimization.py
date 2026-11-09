# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.


def operation_count(circuit) -> int:
    return len(circuit.operations)


from quantumbridge.core import Circuit
from .pass_base import PassResult, TransformationPass, copy_terminal_measurements


class RemoveIdentityPass(TransformationPass):
    def run(self, circuit):
        from quantumbridge.core.circuit_metadata import empty_execution_circuit
        out = empty_execution_circuit(circuit)
        removed = 0
        for op in circuit.operations:
            if op.name in {"id", "identity"}:
                removed += 1
                continue
            out.operations.append(op)
        copy_terminal_measurements(circuit, out)
        return PassResult(out, {"removed_identity": removed}, bool(removed))


class CancelAdjacentInversePass(TransformationPass):
    inverse_pairs = {("s", "sdg"), ("sdg", "s"), ("t", "tdg"), ("tdg", "t")}
    self_inverse = {"x", "y", "z", "h", "cx", "cz", "swap"}

    def run(self, circuit):
        from quantumbridge.core.circuit_metadata import empty_execution_circuit
        out = empty_execution_circuit(circuit)
        index = 0
        cancelled = 0
        while index < len(circuit.operations):
            current = circuit.operations[index]
            nxt = circuit.operations[index + 1] if index + 1 < len(circuit.operations) else None
            if nxt is not None and current.targets == nxt.targets and current.controls == nxt.controls:
                if (current.name == nxt.name and current.name in self.self_inverse) or (current.name, nxt.name) in self.inverse_pairs:
                    cancelled += 2
                    index += 2
                    continue
            out.operations.append(current)
            index += 1
        copy_terminal_measurements(circuit, out)
        return PassResult(out, {"cancelled_operations": cancelled}, bool(cancelled))


class TwoQubitGateReductionPass(CancelAdjacentInversePass):
    self_inverse = {"cx", "cz", "swap"}
    inverse_pairs = set()


class CircuitOptimizationPass(TransformationPass):
    def __init__(self, passes=None):
        from .cancellation import MergeAdjacentRotationPass, RemoveZeroRotationPass

        self.passes = list(passes or [RemoveIdentityPass(), RemoveZeroRotationPass(), CancelAdjacentInversePass(), MergeAdjacentRotationPass()])

    def run(self, circuit):
        current = circuit
        analyses = {}
        changed = False
        for opt_pass in self.passes:
            result = opt_pass.run(current)
            current = result.circuit
            analyses.update(result.analyses)
            changed = changed or result.changed
        return PassResult(current, analyses, changed)
