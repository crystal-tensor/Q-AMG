# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Separate public user data from native executable circuit configuration.

Legacy fluent Circuit retains its metadata configuration ABI. Public circuits,
and native artifacts derived from them, carry execution fields independently.
"""


def is_separated(circuit):
    return "_execution_metadata" in object.__getattribute__(circuit, "__dict__")


def execution_metadata(circuit):
    fields = object.__getattribute__(circuit, "__dict__")
    return fields["_execution_metadata"] if "_execution_metadata" in fields else fields["metadata"]


def execution_copy(value, *, memo=None, category="native_execution_copy", strict=None):
    """Copy executable graphs, borrowing only isolated nonsemantic user data."""
    from quantumbridge.execution.array_ownership import native_configuration_copy, _defined_type
    from quantumbridge.execution.resource_context import current_resources
    from .gate_limits import current_gate_limits
    from .circuit import is_native_circuit
    memo = {} if memo is None else dict(memo)
    pending, seen = [value], set(memo)
    while pending:
        item = pending.pop()
        if id(item) in seen: continue
        seen.add(id(item))
        current_gate_limits().check("max_decomposition_work", len(seen)+len(pending))
        kind = type(item)
        if kind is dict:
            current_gate_limits().check("max_decomposition_work", len(seen)+len(pending)+2*len(item))
            pending.extend(item.keys()); pending.extend(item.values())
        elif any(kind is allowed for allowed in (list, tuple, set, frozenset)):
            current_gate_limits().check("max_decomposition_work", len(seen)+len(pending)+len(item))
            pending.extend(item)
        elif _defined_type(kind, "quantumbridge."):
            fields = object.__getattribute__(item, "__dict__")
            if is_native_circuit(item) and "_execution_metadata" in fields:
                memo[id(fields["metadata"])] = fields["metadata"]
                pending.extend(child for key, child in fields.items() if key != "metadata")
            else: pending.append(fields)
    return native_configuration_copy(value, memo=memo, category=category,
        strict=current_resources() is not None if strict is None else strict)


def copy_user_metadata(circuit, *, memo=None):
    from quantumbridge.execution.array_ownership import native_configuration_copy
    from quantumbridge.execution.resource_context import current_resources
    return native_configuration_copy(circuit.metadata, memo=memo,
        category="native_user_metadata_copy", strict=current_resources() is not None)


def inherit_execution(source, target, *, memo=None):
    if is_separated(source) or is_separated(target):
        target._execution_metadata = execution_copy(execution_metadata(source), memo=memo)
    return target


def empty_execution_circuit(source, *, num_qubits=None, num_bits=None, name=None):
    """A real empty native store, not an execution proxy or parallel graph."""
    from .circuit import Circuit
    metadata = source.metadata if is_separated(source) else execution_copy(source.metadata)
    target = Circuit(source.num_qubits if num_qubits is None else num_qubits,
        source.num_bits if num_bits is None else num_bits,
        source.name if name is None else name)
    target.metadata = metadata
    inherit_execution(source, target)
    if target.num_qubits != source.num_qubits or target.num_bits != source.num_bits:
        execution_metadata(target).pop("registers",None)
    return target


def serialized_user_metadata(circuit):
    if is_separated(circuit): return circuit.metadata
    return {key: value for key, value in circuit.metadata.items() if key not in {"global_phase", "registers"}}
