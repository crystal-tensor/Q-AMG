# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Native identifier scopes, independent of ordered runtime Store nodes."""

from collections.abc import Mapping

from .classical import ClassicalExpression, ClassicalType, Var, Stretch, validate_classical_expr
from .errors import CircuitError
from .gate_limits import current_gate_limits
from .circuit_metadata import execution_metadata


_KEY = "classical_scope"
_MISSING = object()


def entries(circuit):
    from quantumbridge.execution.resource_context import current_resources
    if current_resources() is not None:
        from .instruction_parameters import admit_payloads
        from .classical import UnsupportedFeature
        if type(execution_metadata(circuit)) is not dict: raise UnsupportedFeature("managed_classical_scope_metadata")
    payload = execution_metadata(circuit).get(_KEY)
    result, names, identities = [], set(), set()
    if payload is not None:
        if current_resources() is not None: admit_payloads((payload,))
        if type(payload) is not dict or set(payload) != {"version", "identifiers"} or payload["version"] != 1 or type(payload["identifiers"]) is not list:
            raise CircuitError("Malformed native classical scope.")
        current_gate_limits().check("max_gate_parameters", len(payload["identifiers"]))
        for record in payload["identifiers"]:
            if type(record) is not dict or set(record) != {"node", "scope"} or record["scope"] not in {"local", "input", "capture"}:
                raise CircuitError("Malformed native identifier declaration.")
            node = ClassicalExpression.from_dict(record["node"])
            if type(node) not in {Var, Stretch}: raise CircuitError("Scopes require immutable native identifiers.")
            if node.value[0] in identities or node.name in names: raise CircuitError("Duplicate identifier identity or name.")
            if type(node) is Stretch and record["scope"] == "input": raise CircuitError("Stretch cannot be an input variable.")
            identities.add(node.value[0]); names.add(node.name)
            cache = circuit.__dict__.setdefault("_classical_identifiers", {})
            existing = cache.get(node.value[0])
            if existing is not None and existing != node: raise CircuitError("Cached classical identifier disagrees with its declaration.")
            node = cache.setdefault(node.value[0], node)
            result.append((node, record["scope"], False))
    for op in circuit.operations:
        node = op.metadata.get("control_node")
        if node is None or node.kind != "declare": continue
        target = node.data["target"]
        validate_classical_expr(target)
        if target.op != "var": raise CircuitError("A declaration requires variable storage.")
        if target.value[0] in identities or target.value[1] in names:
            raise CircuitError("Duplicate variable identity or name.")
        identities.add(target.value[0]); names.add(target.value[1])
        result.append((target, "local", True))
    if any(scope == "input" for _,scope,_ in result) and any(scope == "capture" for _,scope,_ in result):
        raise CircuitError("Input declarations and captured identifiers cannot coexist.")
    return result


def set_entries(circuit, records):
    serialized = [{"node": node.to_dict(), "scope": scope} for node,scope,ordered in records if not ordered]
    for record in serialized: record["node"]["value"]=list(record["node"]["value"])
    if serialized: execution_metadata(circuit)[_KEY] = {"version":1, "identifiers":serialized}
    else: execution_metadata(circuit).pop(_KEY, None)
    circuit._classical_identifiers = {node.value[0]:node for node,_,ordered in records if not ordered}


def add_identifier(circuit, node, scope):
    if type(node) not in {Var, Stretch}: raise CircuitError("Expected an immutable native Var or Stretch.")
    records = entries(circuit)
    if any(item.value[0] == node.value[0] or item.value[1] == node.name for item,_,_ in records):
        raise CircuitError("An identifier with this identity or name already exists.")
    if scope == "input" and any(mode == "capture" for _,mode,_ in records) or scope == "capture" and any(mode == "input" for _,mode,_ in records):
        raise CircuitError("Input declarations and captured identifiers cannot coexist.")
    current_gate_limits().check("max_gate_parameters", len(records)+1)
    set_entries(circuit, [*records, (node,scope,False)])
    return node


def get_identifier(circuit, name, default=_MISSING, kind=None):
    for node,_,_ in entries(circuit):
        if node.value[1] == name and (kind is None or node.op == kind): return node
    if default is not _MISSING: return default
    raise KeyError("No matching native identifier named " + repr(name))


def copy_scope(source, target, mode):
    if mode not in {"alike","captures","drop"}: raise ValueError("Unknown vars_mode.")
    if mode == "drop": execution_metadata(target).pop(_KEY,None); return
    records = []
    for node,scope,_ in entries(source):
        if type(node) not in {Var,Stretch}:
            node = Var(node.value[0], node.dtype, name=node.value[1])
        records.append((node,"capture" if mode == "captures" else scope,False))
    set_entries(target, records)


def composition_scope(target, source, var_remap, inline_captures):
    from quantumbridge.execution.resource_context import current_resources
    from .classical import UnsupportedFeature
    if current_resources() is not None and var_remap is not None and type(var_remap) is not dict:
        raise UnsupportedFeature("managed_classical_remap_mapping")
    if var_remap is not None and not isinstance(var_remap, Mapping): raise TypeError("var_remap must be a mapping.")
    mapping = {} if var_remap is None else var_remap
    current_gate_limits().check("max_gate_parameters", len(mapping))
    for key,value in mapping.items():
        from quantumbridge.execution.array_ownership import _exact_type
        if not _exact_type(type(key),(str,Var,Stretch)) or not _exact_type(type(value),(str,Var,Stretch)):
            raise CircuitError("Variable remaps need native identifiers or string names.")
    output, rewritten = entries(target), {}
    known = {node.value[0]: node for node,_,_ in output}
    names = {node.value[1] for node,_,_ in output}
    def replacement_for(node):
        replacement = mapping.get(node, _MISSING) if type(node) in {Var,Stretch} else _MISSING
        if replacement is _MISSING: replacement = mapping.get(node.value[1], _MISSING)
        if replacement is _MISSING: replacement = node
        elif type(replacement) is str:
            replacement = Stretch.new(replacement) if node.op == "stretch" else Var.new(replacement,node.dtype)
        if replacement.op != node.op or replacement.dtype != node.dtype:
            raise CircuitError("Variable remap changes identifier kind or type.")
        return replacement
    for node,scope,ordered in entries(source):
        replacement=replacement_for(node)
        rewritten[node.value[0]] = replacement
        if scope == "capture" and inline_captures:
            if replacement.value[0] not in known:
                raise CircuitError("A capture to inline is absent from the target scope.")
            if known[replacement.value[0]].dtype != replacement.dtype: raise CircuitError("Captured variable type mismatch.")
            continue
        if replacement.value[0] in known or replacement.value[1] in names:
            raise CircuitError("Composed identifier conflicts with the target scope.")
        output.append((replacement,scope,ordered))
        known[replacement.value[0]]=replacement;names.add(replacement.value[1])
    if any(scope == "input" for _,scope,_ in output) and any(scope == "capture" for _,scope,_ in output):
        raise CircuitError("Input declarations and captured identifiers cannot coexist.")
    pending=[source];seen=set()
    while pending:
        body=pending.pop()
        if id(body) in seen:continue
        seen.add(id(body))
        for node,_,_ in entries(body):
            if node.value[0] not in rewritten: rewritten[node.value[0]]=replacement_for(node)
        current_gate_limits().check("max_gate_parameters",len(rewritten))
        for op in body.operations:
            block=op.metadata.get("control_node")
            if block is not None:pending.extend(block.bodies)
    current_gate_limits().check("max_gate_parameters", len(output))
    return output, rewritten


def execution_environment(circuit, environment=None):
    from .classical import ClassicalExecutionError
    result = dict(environment or {})
    for node,scope,ordered in entries(circuit):
        if ordered: continue
        key = node.value[0]
        if scope == "capture":
            if result.get(key) != node.dtype:
                raise ClassicalExecutionError("unbound_capture", "Captured identifier is outside the enclosing lexical scope.")
        elif key in result:
            raise ClassicalExecutionError("duplicate_variable", "Identifier already exists in this lexical scope.")
        else: result[key]=node.dtype
    return result
