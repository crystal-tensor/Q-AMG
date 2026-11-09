# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Native scope-preserving compiler traversal.

from copy import deepcopy
from dataclasses import replace

from quantumbridge.core import Circuit, Measurement
from quantumbridge.core.control_flow import ControlBlock


def append_item(circuit, item):
    item = deepcopy(item)
    if isinstance(item, Measurement):
        item.metadata["operation_index"] = len(circuit.operations)
        circuit.measurements.append(item)
    else:
        circuit.operations.append(item)


def map_control_bodies(operation, transform):
    node = operation.metadata["control_node"]
    data = deepcopy(node.data)
    for field in ("body", "alternate", "default"):
        if data.get(field) is not None: data[field] = transform(data[field])
    if "cases" in data:
        data["cases"] = tuple((values, transform(body)) for values, body in data["cases"])
    return replace(operation, metadata={**deepcopy(operation.metadata), "control_node": ControlBlock(node.kind, data)})


def transform_segments(circuit, transform):
    """Optimize unconditional segments, never across runtime boundaries."""
    from quantumbridge.core.circuit_metadata import empty_execution_circuit
    out = empty_execution_circuit(circuit)
    segment = circuit.copy_empty_like(vars_mode="drop")
    def flush():
        nonlocal segment
        if segment.operations:
            result = transform(segment)
            out.operations.extend(result.operations)
            out.global_phase = out.global_phase + result.global_phase
            segment = circuit.copy_empty_like(vars_mode="drop")
    for item in circuit.iter_instructions():
        if isinstance(item, Measurement):
            flush(); append_item(out, item)
        elif item.metadata.get("control_node") is not None:
            flush(); append_item(out, map_control_bodies(item, lambda body: transform_segments(body, transform)))
        elif item.name in {"reset", "kraus"} or any(key in item.metadata for key in ("directive", "condition", "gate_condition", "control_flow_stack", "control_flow")):
            flush(); append_item(out, item)
        else:
            segment.operations.append(deepcopy(item))
    flush()
    return out
