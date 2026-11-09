# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Native shortest-path routing with explicit permutation tracking.

from copy import deepcopy
from dataclasses import replace

from quantumbridge.core import Circuit, Measurement, Operation
from quantumbridge.core.classical import UnsupportedFeature
from .limits import CompilationLimits
from .pass_base import AnalysisPass, PassResult, TransformationPass
from .traversal import append_item, map_control_bodies


def routing_required(circuit, coupling_map):
    return any(len(op.wires) == 2 and not coupling_map.is_connected(*op.wires) for op in circuit.operations)


class BasicRoutingPass(AnalysisPass):
    handles_dynamic = True
    def __init__(self, coupling_map): self.coupling_map = coupling_map
    def run(self, circuit):
        unsupported = [{"operation": op.name, "wires": op.wires} for op in circuit.operations
                       if len(op.wires) == 2 and not self.coupling_map.is_connected(*op.wires)]
        return PassResult(circuit, {"routing_required": bool(unsupported), "unsupported_interactions": unsupported}, False)


class SwapInsertionPass(TransformationPass):
    handles_dynamic = True
    def __init__(self, coupling_map, restore_layout=True, target=None, limits=None):
        self.coupling_map, self.restore_layout, self.target = coupling_map, restore_layout, target
        self.limits = limits or CompilationLimits()

    def run(self, circuit):
        from quantumbridge.core.circuit_metadata import empty_execution_circuit, execution_metadata
        out = empty_execution_circuit(circuit)
        permutation = dict(enumerate(range(circuit.num_qubits)))
        history, forward, restored, nested_swaps, reversed_cx = [], 0, 0, 0, 0
        def emit_cx(op):
            nonlocal reversed_cx
            a, b = op.wires
            available = self.target.instruction_supported("cx", (a,b)) if self.target is not None else self.coupling_map.has_edge(a,b)
            if available: out.operations.append(op); return
            reverse = self.target.instruction_supported("cx", (b,a)) if self.target is not None else self.coupling_map.has_edge(b,a)
            if not reverse: raise UnsupportedFeature("directed_cx", "Neither CX direction is available on this target edge.")
            reversed_cx += 1
            for wire in (a,b): out.operations.append(Operation("h", (wire,), metadata=deepcopy(op.metadata)))
            out.operations.append(Operation("cx", (a,), (b,), metadata=deepcopy(op.metadata)))
            for wire in (a,b): out.operations.append(Operation("h", (wire,), metadata=deepcopy(op.metadata)))
        def swap(a,b, undo=False):
            nonlocal forward, restored
            if forward + restored + nested_swaps >= self.limits.max_routing_swaps: raise MemoryError("Routing exceeds max_routing_swaps.")
            if not self.coupling_map.is_connected(a,b): raise ValueError("Routing attempted a nonlocal SWAP.")
            if self.target is None or self.target.instruction_supported("swap", (a,b)):
                out.operations.append(Operation("swap", (a,b)))
            else:
                for left, right in ((a,b),(b,a),(a,b)): emit_cx(Operation("cx", (right,), (left,)))
            for logical, physical in permutation.items():
                if physical == a: permutation[logical] = b
                elif physical == b: permutation[logical] = a
            if undo: restored += 1
            else: forward += 1; history.append((a,b))
        def restore():
            for a,b in reversed(history): swap(a,b, True)
            history.clear()
        for item in circuit.iter_instructions():
            if isinstance(item, Measurement):
                if self.restore_layout: restore()
                append_item(out, replace(item, wire=permutation[item.wire])); continue
            if item.metadata.get("control_node") is not None:
                restore()
                def compile_body(body):
                    nonlocal nested_swaps
                    result = SwapInsertionPass(self.coupling_map, True, self.target, self.limits).run(body)
                    nested_swaps += result.analyses["total_routing_swaps"]
                    if forward + restored + nested_swaps > self.limits.max_routing_swaps: raise MemoryError("Nested routing exceeds max_routing_swaps.")
                    return result.circuit
                append_item(out, map_control_bodies(item, compile_body)); continue
            if len(item.wires) > 2 and not item.metadata.get("directive"):
                raise UnsupportedFeature("multi_qubit_routing", "Decompose gates with more than two operands before routing.")
            mapped = tuple(permutation[wire] for wire in item.wires)
            if len(mapped) == 2 and not item.metadata.get("directive") and not self.coupling_map.is_connected(*mapped):
                path = self.coupling_map.shortest_path(*mapped)
                for a,b in zip(path[:-2], path[1:-1]): swap(a,b)
            op = replace(item, targets=tuple(permutation[wire] for wire in item.targets), controls=tuple(permutation[wire] for wire in item.controls), metadata=deepcopy(item.metadata))
            if op.name == "cx": emit_cx(op)
            else: out.operations.append(op)
            if self.restore_layout: restore()
        if self.restore_layout: restore()
        analyses = {"inserted_swaps": forward, "restoration_swaps": restored, "nested_routing_swaps": nested_swaps,
                    "total_routing_swaps": forward + restored + nested_swaps, "reversed_cx": reversed_cx,
                    "final_layout": dict(permutation), "layout_restored": self.restore_layout}
        execution_metadata(out)["routing"] = analyses
        return PassResult(out, analyses, bool(forward or restored or nested_swaps or reversed_cx))
