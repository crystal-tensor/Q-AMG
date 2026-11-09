# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Native dependency-aware static ASAP/ALAP scheduling.

from copy import deepcopy
import math

from quantumbridge.core import Measurement
from quantumbridge.core.classical import UnsupportedFeature
from .dag import circuit_to_dag
from .limits import CompilationLimits
from .pass_base import PassResult, TransformationPass
from .traversal import append_item


class SchedulingPass(TransformationPass):
    handles_dynamic = True

    def __init__(self, target, method="asap", *, limits=None, insert_delays=False):
        if method not in {"asap", "alap"}: raise ValueError("Scheduling method must be asap or alap.")
        self.target, self.method = target, method
        self.limits, self.insert_delays = limits or CompilationLimits(), insert_delays

    def run(self, circuit):
        if circuit.has_structured_control_flow:
            raise UnsupportedFeature("dynamic_global_schedule", "Runtime branches/loops require path-dependent feedback timing; no global schedule is fabricated.")
        target = self.target
        tolerance = 1e-8 if target.dt is not None else 1e-15
        if target.dt is None and max(target.pulse_alignment, target.acquire_alignment, target.granularity) > 1:
            raise ValueError("Timing alignments require an explicit target dt.")
        dag = circuit_to_dag(circuit, self.limits)
        nodes = dag.topological_op_nodes()
        durations, alignments, feedback = {}, {}, {}
        for node in nodes:
            item = node.instruction
            name = node.name
            wires = (item.wire,) if isinstance(item, Measurement) else item.wires
            if name == "delay":
                value = float(item.params[0]); unit = item.metadata.get("unit", "dt")
                if not math.isfinite(value) or value < 0: raise ValueError("Delay duration must be non-negative and finite.")
                if unit == "dt":
                    if target.dt is None: raise ValueError("Delay in dt requires target dt.")
                    duration = value
                elif unit in {"s", "ms", "us", "ns", "ps"}:
                    seconds = value * {"s": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}[unit]
                    duration = seconds if target.dt is None else seconds / target.dt
                else: raise ValueError("Unknown delay unit.")
                if target.dt is not None and not math.isclose(duration / target.granularity, round(duration / target.granularity), abs_tol=1e-8):
                    raise ValueError("Delay violates timing granularity.")
            else: duration = target.duration(name, wires)
            durations[node.node_id] = duration
            alignments[node.node_id] = (target.acquire_alignment if isinstance(item, Measurement) else target.pulse_alignment) if target.dt is not None else None
            conditional = any(key in item.metadata for key in ("condition", "gate_condition", "control_flow_stack"))
            if conditional and target.feedback_latency is None: raise ValueError("Conditional timing requires explicit feedback_latency.")
            feedback[node.node_id] = (target.feedback_latency or 0) / (target.dt or 1) if conditional else 0.
        starts, ends = {}, {}
        for node in nodes:
            key = node.node_id
            ready = max((ends.get(parent, 0.) + (feedback[key] if isinstance(dag.nodes[parent].instruction, Measurement) else 0.) for parent in dag.edges[key]), default=0.)
            align = alignments[key]
            starts[key] = math.ceil((ready - 1e-10) / align) * align if align else ready
            ends[key] = starts[key] + durations[key]
        makespan = max(ends.values(), default=0.)
        if makespan > self.limits.max_schedule_makespan: raise MemoryError("Schedule exceeds max_schedule_makespan.")
        if self.method == "alap":
            followers = {node.node_id: [] for node in nodes}
            for node in nodes:
                for parent in dag.edges[node.node_id]:
                    if parent in followers: followers[parent].append(node.node_id)
            for node in reversed(nodes):
                key = node.node_id
                successors = followers[key]
                latest = min((starts[other] - (feedback[other] if isinstance(node.instruction, Measurement) else 0.) for other in successors), default=makespan)
                raw = latest - durations[key]
                align = alignments[key]
                starts[key] = math.floor((raw + 1e-10) / align) * align if align else raw
                ends[key] = starts[key] + durations[key]
                if starts[key] < -tolerance: raise ValueError("ALAP timing constraints cannot fit the selected makespan.")
        for node in nodes:
            key = node.node_id
            for parent in dag.edges[key]:
                minimum = ends.get(parent, 0.) + (feedback[key] if isinstance(dag.nodes[parent].instruction, Measurement) else 0.)
                if starts[key] < minimum - tolerance: raise ValueError("Schedule violates a quantum/classical dependency.")
        records = [{"node_id": node.node_id, "operation": node.name,
                    "wires": [node.instruction.wire] if isinstance(node.instruction, Measurement) else list(node.instruction.wires),
                    "start": starts[node.node_id], "duration": durations[node.node_id], "end": ends[node.node_id]} for node in nodes]
        schedule = {"method": self.method, "unit": "dt" if target.dt is not None else "s", "dt": target.dt,
                    "makespan": makespan, "operations": records, "semantics": "static dependency schedule"}
        from quantumbridge.core.circuit_metadata import execution_copy, empty_execution_circuit, execution_metadata
        out = execution_copy(circuit)
        if self.insert_delays:
            out = empty_execution_circuit(circuit)
            from quantumbridge.core.bits import copy_bits
            copy_bits(circuit,out)
            available = [0.] * circuit.num_qubits
            for node, record in zip(nodes, records):
                for wire in record["wires"]:
                    gap = record["start"] - available[wire]
                    if gap > tolerance: out.delay(gap, wire, unit=schedule["unit"])
                    available[wire] = record["end"]
                append_item(out, node.instruction)
        execution_metadata(out)["schedule"] = deepcopy(schedule)
        return PassResult(out, {"schedule": schedule, "critical_path_duration": makespan}, True)
