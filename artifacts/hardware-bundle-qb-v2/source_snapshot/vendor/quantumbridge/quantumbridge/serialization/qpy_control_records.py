# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Standard QPY13 control records over existing native public owners."""

import struct

from quantumbridge.core import control_flow_operations as control
from quantumbridge.core.bits import Clbit, ClassicalRegister
from quantumbridge.core.circuit import is_native_circuit
from quantumbridge.core.gate_limits import current_gate_limits
from .errors import FormatValidationError, UnsupportedFeature
from .common import BoundedParts


CONTROL_NAMES = {kind.__name__ for kind in control.CONTROL_TYPES+control.STATEMENT_TYPES}
_NAMES = {"IfElseOp": "if_else", "ForLoopOp": "for_loop", "WhileLoopOp": "while_loop",
          "SwitchCaseOp": "switch_case", "BreakLoopOp": "break_loop", "ContinueLoopOp": "continue_loop", "Store": "store"}


def read_resource(reader, payload):
    circuit = reader.circuit_owner
    if circuit is None:
        reader.error("A QPY classical resource needs its enclosing Circuit.")
    if payload.startswith("\0"):
        index = payload[1:]
        if not index or any(char not in "0123456789" for char in index):
            reader.error("Invalid QPY single-bit resource encoding.")
        index = int(index)
        if index >= circuit.num_bits:
            reader.error("QPY condition refers to a foreign classical bit.")
        return circuit.clbits[index]
    matches = [reg for reg in circuit.cregs if reg.name == payload]
    if len(matches) != 1:
        reader.error("QPY condition refers to an absent or ambiguous classical register.")
    return matches[0]


def resource_bytes(circuit, resource, bindings=None):
    bindings = circuit._clbit_indices if bindings is None else bindings
    try:
        if type(resource) is Clbit:
            return b"\0"+str(bindings[resource._key]).encode()
        if type(resource) is ClassicalRegister:
            positions = tuple(bindings[bit._key] for bit in resource)
            matches = [reg for reg in circuit.cregs if reg.name == resource.name
                       and tuple(circuit.find_bit(bit).index for bit in reg) == positions]
            if len(matches) == 1:
                return resource.name.encode()
    except KeyError:
        pass
    raise UnsupportedFeature("A QPY classical resource must belong to its enclosing Circuit/register table.", format="qpy", version=13)


def public_block(circuit):
    from quantumbridge.core.public_circuit import QuantumCircuit
    if type(circuit) is QuantumCircuit:
        return circuit
    from quantumbridge.core.bits import copy_bits
    from quantumbridge.core.circuit_metadata import inherit_execution
    from quantumbridge.core.circuit_data import public_factory_entries, commit_entries
    from quantumbridge.core.classical_scope import entries, set_entries
    result = QuantumCircuit(name=circuit.name, metadata=circuit.metadata)
    copy_bits(circuit, result)
    inherit_execution(circuit, result)
    result.global_phase = circuit.global_phase
    result.operations, result.measurements = list(circuit.operations), list(circuit.measurements)
    # Keep Measurement ordering and native matrix origins; only missing
    # primitive owners are admitted through the existing public factory.
    planned = public_factory_entries(result, circuit)
    commit_entries(result, planned)
    set_entries(result, entries(circuit))
    return result


def decode_operation(reader, name, label, params, qargs, cargs, condition, nctrl, state):
    if nctrl or state:
        reader.error("A classical control operation cannot have coherent control fields.")
    expected = {"IfElseOp": 2, "ForLoopOp": 3, "WhileLoopOp": 1,
                "SwitchCaseOp": 2, "BreakLoopOp": 0, "ContinueLoopOp": 0, "Store": 2}[name]
    if len(params) != expected:
        reader.error("QPY control parameter count differs from its standard record.")
    if (name in {"IfElseOp", "WhileLoopOp"}) != (condition is not None):
        reader.error("QPY control condition field differs from its standard operation.")
    if name == "IfElseOp":
        operation = control.IfElseOp(condition, *params, label)
    elif name == "ForLoopOp":
        operation = control.ForLoopOp(*params, label)
    elif name == "WhileLoopOp":
        operation = control.WhileLoopOp(condition, params[0], label)
    elif name == "SwitchCaseOp":
        operation = control.SwitchCaseOp(*params, label=label)
    elif name == "Store":
        from quantumbridge.core import classical as expr
        target, value = params
        if not isinstance(target, expr.ClassicalExpression) or not isinstance(value, expr.ClassicalExpression):
            reader.error("QPY Store requires two typed classical expressions.")
        if target.op == "cast" and target.dtype == expr.Bool() and target.args[0].op == "bit":
            target = target.args[0]
            value = expr.cast(value, target.dtype)
        operation = control.Store(target, value)
        if label is not None:
            operation.label = label
    else:
        if not reader.loop_depth:
            reader.error("QPY break/continue is outside its enclosing loop.")
        operation = getattr(control, name)(len(qargs), len(cargs), label)
    if (operation.num_qubits, operation.num_clbits) != (len(qargs), len(cargs)):
        reader.error("QPY control invocation width differs from its block resources.")
    return control.control_entry(reader.circuit_owner, operation, qargs, cargs,
        expression_map=dict(enumerate(range(reader.circuit_owner.num_bits))))


def encode_operation(item, circuit, limits, budget, depth, path, loop_depth, ancestors):
    from .qpy import INSTRUCTION, VALUE
    fields = object.__getattribute__(item, "__dict__")
    owner = fields["_control_owner"]
    class_name = type(owner).__name__
    if owner.name != _NAMES[class_name] or owner.metadata:
        raise UnsupportedFeature("A changed control operation cannot masquerade as its standard QPY class.", format="qpy", version=13)
    allowed = {"control_node", "instruction_clbits", "label", "operation_index"}
    if set(item.metadata)-allowed:
        raise UnsupportedFeature("Opaque control invocation metadata has no standard QPY13 field.", format="qpy", version=13)
    condition, resource, expected = 0, b"", 0
    expression = None
    if class_name in {"IfElseOp", "WhileLoopOp"}:
        if type(owner.condition) is tuple:
            resource = resource_bytes(circuit, owner.condition[0], fields["_control_bit_bindings"])
            condition, expected = 1, int(owner.condition[1])
        else:
            from .qpy_classical import expression_bytes
            expression = expression_bytes(owner.condition, circuit, limits, budget, bits=fields["_control_expression_map"])
            condition = 2
    if class_name == "SwitchCaseOp":
        if type(owner.target) in {Clbit, ClassicalRegister}:
            target = (b"R", resource_bytes(circuit, owner.target, fields["_control_bit_bindings"]))
        else:
            from .qpy_classical import expression_bytes
            target = (b"x", expression_bytes(owner.target, circuit, limits, budget, bits=fields["_control_expression_map"]))
        params = [target, encode_value(tuple(owner.cases_specifier()), circuit, limits, budget,
                                              depth, path+"/cases", loop_depth, ancestors)]
    elif class_name == "Store":
        from .qpy_classical import expression_bytes
        params = [(b"x", expression_bytes(value, circuit, limits, budget, bits=fields["_control_expression_map"]))
                  for value in owner.params]
    else:
        params = [encode_value(value, circuit, limits, budget, depth, path+"/"+str(index),
            loop_depth+int(class_name in {"ForLoopOp", "WhileLoopOp"}), ancestors)
            for index, value in enumerate(owner.params)]
    label = (owner.label or "").encode()
    key = class_name.encode()
    clbits = tuple(item.metadata.get("instruction_clbits", ()))
    if (len(item.targets), len(clbits)) != (owner.num_qubits, owner.num_clbits) or item.controls:
        raise FormatValidationError("Control invocation differs from its native operation owner.", format="qpy", version=13)
    parts = BoundedParts(limits, "qpy")
    parts.extend((INSTRUCTION.pack(len(key), len(label), len(params), len(item.targets), len(clbits),
                    condition, len(resource), expected, 0, 0), key, label, resource))
    if expression is not None:
        parts.extend((VALUE.pack(b"x", len(expression)), expression))
    parts.extend(struct.pack("!cI", b"q", wire) for wire in item.targets)
    parts.extend(struct.pack("!cI", b"c", bit) for bit in clbits)
    for kind, payload in params:
        parts.extend((VALUE.pack(kind, len(payload)), payload))
    return b"".join(parts)


def admit_range(value, limits, budget):
    size = len(value)
    total = budget.get("loop_indices", 0)+size
    limits.check("max_gate_expansion", total, format="qpy", version=13)
    current_gate_limits().check("max_decomposition_work", total)
    budget["loop_indices"] = total


def encode_value(value, circuit, limits, budget, depth, path, loop_depth, ancestors):
    from .qpy import _parameter_value, _circuit_bytes, VALUE
    depth += 1
    limits.check("max_nesting_depth", depth, format="qpy", version=13)
    budget["control_values"] = budget.get("control_values", 0)+1
    limits.check("max_parameter_nodes", budget["control_values"], format="qpy", version=13)
    if is_native_circuit(value):
        from .qpy_classical import lexical_variables
        return b"q", _circuit_bytes(value, limits, _budget=budget, _depth=depth, _path=path,
            _loop_depth=loop_depth, _ancestors=ancestors, _enclosing_variables=lexical_variables(circuit))
    if value is None:
        return b"z", b""
    if value is control.CASE_DEFAULT:
        return b"d", b""
    if type(value) is range:
        admit_range(value, limits, budget)
        return b"r", struct.pack("!qqq", value.start, value.stop, value.step)
    if type(value) in (tuple, list):
        limits.check("max_parameter_nodes", len(value), format="qpy", version=13)
        parts = BoundedParts(limits, "qpy")
        parts.append(struct.pack("!Q", len(value)))
        for index, element in enumerate(value):
            kind, payload = encode_value(element, circuit, limits, budget, depth, path+"/"+str(index), loop_depth, ancestors)
            parts.extend((VALUE.pack(kind, len(payload)), payload))
        return b"t", b"".join(parts)
    return _parameter_value(value, limits, budget=budget)
