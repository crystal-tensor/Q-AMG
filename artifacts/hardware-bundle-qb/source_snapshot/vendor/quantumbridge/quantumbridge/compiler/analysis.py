# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from .pass_base import AnalysisPass, PassResult


class DepthAnalysisPass(AnalysisPass):
    def run(self, circuit):
        from .dag import circuit_to_dag
        depth = circuit_to_dag(circuit).depth(structural=True)
        return PassResult(circuit, {"depth": None if circuit.has_structured_control_flow else depth,
                                  "structural_depth": depth, "runtime_depth": "path-dependent" if circuit.has_structured_control_flow else depth}, False)


class TwoQubitCountAnalysisPass(AnalysisPass):
    def run(self, circuit):
        count = sum(1 for op in circuit.operations if len(op.wires) == 2)
        return PassResult(circuit, {"two_qubit_gate_count": count}, False)


class GateCountAnalysisPass(AnalysisPass):
    def run(self, circuit):
        counts = {}
        for op in circuit.operations:
            counts[op.name] = counts.get(op.name, 0) + 1
        return PassResult(circuit, {"gate_counts": counts, "operation_count": len(circuit.operations)}, False)
