# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Recover lexical blocks from the native legacy branch-entry snapshots.
from copy import deepcopy
from dataclasses import replace
from quantumbridge.core import Circuit, Measurement
from quantumbridge.core.operations import Operation,execution_operation
from quantumbridge.core.control_flow import ControlBlock, condition_expression
from quantumbridge.compiler.traversal import append_item, map_control_bodies
from .errors import UnsupportedFeature


def normalize_source_control(circuit,limits,depth=0):
    limits.check("max_nesting_depth",depth,format="qasm")
    from quantumbridge.core.circuit_metadata import empty_execution_circuit
    out = empty_execution_circuit(circuit)
    from quantumbridge.core.bits import copy_bits
    copy_bits(circuit,out)
    items = list(circuit.iter_instructions())
    limits.check("max_instructions",len(items),format="qasm")
    index = 0
    while index < len(items):
        item = items[index]
        if type(item) is Operation:item=execution_operation(item)
        frames = item.metadata.get("control_flow_stack",[])
        if not frames:
            if "control_flow" in item.metadata:
                raise UnsupportedFeature("Malformed legacy branch metadata has no ordered entry frames.",format="qasm")
            if not isinstance(item,Measurement) and item.metadata.get("control_node") is not None:
                item = map_control_bodies(item,lambda body:normalize_source_control(body,limits,depth+1))
            append_item(out,item); index += 1; continue
        frame = frames[0]
        if not frame.get("start") or frame.get("type") != "if_else":
            raise UnsupportedFeature("Legacy conditional block has no valid entry snapshot.",format="qasm")
        branches = {"true":Circuit(circuit.num_qubits,circuit.num_bits),"false":Circuit(circuit.num_qubits,circuit.num_bits)}
        seen_false = False
        end = index
        while end < len(items):
            current = items[end]
            stack = current.metadata.get("control_flow_stack",[])
            if not stack or stack[0].get("block_id") != frame.get("block_id") or end > index and stack[0].get("start"): break
            head = stack[0]
            branch = head.get("branch")
            if branch not in branches or head.get("condition") != frame.get("condition") or branch == "true" and seen_false:
                raise UnsupportedFeature("Inconsistent/interleaved legacy conditional branches.",format="qasm")
            seen_false |= branch == "false"
            metadata = deepcopy(current.metadata)
            metadata.pop("condition",None); metadata.pop("control_flow",None)
            if len(stack) > 1:
                metadata["control_flow_stack"] = deepcopy(stack[1:])
                metadata["control_flow"] = deepcopy(stack[1])
            else: metadata.pop("control_flow_stack",None)
            append_item(branches[branch],replace(current,metadata=metadata))
            end += 1
        body = normalize_source_control(branches["true"],limits,depth+1)
        alternate = normalize_source_control(branches["false"],limits,depth+1) if seen_false else None
        out._control(ControlBlock("if",{"condition":condition_expression(frame["condition"]),"body":body,"alternate":alternate}))
        index = end
    return out
