# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Bounded standard QPY13 classical expressions and lexical variable tables."""

import struct
from uuid import UUID

from quantumbridge.core import classical as expr
from quantumbridge.core.classical_scope import entries, set_entries
from quantumbridge.core.gate_limits import current_gate_limits
from .common import BoundedParts
from .errors import FormatValidationError, UnsupportedFeature


DECLARATION = struct.Struct("!16scH")
BINARY = {1: "and", 2: "or", 3: "xor", 4: "logical_and", 5: "logical_or", 6: "eq", 7: "ne",
          8: "lt", 9: "le", 10: "gt", 11: "ge", 12: "shl", 13: "shr", 14: "add", 15: "sub", 16: "mul"}
UNARY = {1: "invert", 2: "not"}


def read_type(reader):
    kind = reader.take(1)
    if kind == b"b":
        return expr.Bool()
    if kind == b"u":
        width, = reader.unpack("!I")
        reader.bounded("max_clbits", width)
        if not 1 <= width <= 4096:
            reader.error("QPY Uint width exceeds native classical storage types.", UnsupportedFeature)
        return expr.UInt(width)
    reader.error("This classical type is not supported by the QPY13 native executor.", UnsupportedFeature)


def type_bytes(dtype):
    if dtype.kind in {"bool", "bit"}:
        return b"b"
    if dtype.kind == "uint":
        return b"u"+struct.pack("!I", dtype.width)
    raise UnsupportedFeature("QPY13 supports Bool/Uint classical types, not this native identifier type.", format="qpy", version=13)


def read_declarations(reader, circuit, count):
    reader.budget["classical_vars"] = reader.budget.get("classical_vars", 0)+count
    reader.bounded("max_parameter_nodes", reader.budget["classical_vars"])
    current_gate_limits().check("max_gate_parameters", count)
    records, names, identities, variables = [], set(), set(), []
    for _ in range(count):
        identity, usage, length = reader.unpack(DECLARATION)
        dtype, name = read_type(reader), reader.text(length)
        identity = str(UUID(bytes=identity))
        if usage not in {b"I", b"C", b"L"} or identity in identities or name in names:
            reader.error("Duplicate/invalid QPY classical variable declaration.")
        descriptor = (name, dtype.kind, dtype.width)
        previous = reader.variable_identities.get(identity)
        if previous is not None and previous != descriptor:
            reader.error("One QPY variable UUID has conflicting names or types.")
        enclosing = reader.enclosing_variables.get(identity)
        if usage == b"C" and reader.nested_control_block and enclosing is None:
            reader.error("Captured QPY variable is outside its enclosing scope.")
        if usage != b"C" and enclosing is not None:
            reader.error("A QPY local/input variable aliases enclosing storage.")
        if enclosing is not None and (enclosing.name, enclosing.dtype.kind, enclosing.dtype.width) != descriptor:
            reader.error("Captured QPY variable differs from its enclosing declaration.")
        value = enclosing if usage == b"C" and enclosing is not None else expr.Var(identity, dtype, name=name)
        names.add(name)
        identities.add(identity)
        reader.variable_identities[identity] = descriptor
        variables.append(value)
        records.append((value, {b"I": "input", b"C": "capture", b"L": "local"}[usage], False))
    set_entries(circuit, records)
    reader.classical_vars = variables


def lexical_variables(circuit, enclosing=None):
    result = dict(enclosing or {})
    for node, _, _ in entries(circuit):
        result[node.value[0]] = node
    return result


def declaration_bytes(circuit, limits, budget, enclosing=None):
    records = entries(circuit)
    budget["classical_vars"] = budget.get("classical_vars", 0)+len(records)
    limits.check("max_parameter_nodes", budget["classical_vars"], format="qpy", version=13)
    current_gate_limits().check("max_gate_parameters", len(records))
    parts = BoundedParts(limits, "qpy")
    for node, scope, ordered in records:
        if node.op != "var" or ordered:
            raise UnsupportedFeature("QPY13 classical ordered declarations/Stretch records are not implemented.", format="qpy", version=13)
        if scope == "capture" and enclosing is not None:
            previous = enclosing.get(node.value[0])
            if previous is None or previous != node:
                raise FormatValidationError("Captured variable is foreign to its QPY block scope.", format="qpy", version=13)
        name = node.name.encode()
        parts.extend((DECLARATION.pack(UUID(node.value[0]).bytes, {"local": b"L", "input": b"I", "capture": b"C"}[scope], len(name)),
                      type_bytes(node.dtype), name))
    return len(records), b"".join(parts)


def validate_circuit(circuit, limits, loop_depth=0, enclosing=None):
    from quantumbridge.core.control_flow import validate_control_flow, ExecutionLimits
    # A fragment may explicitly capture storage without executing standalone.
    # Only a nested block must bind those captures in its enclosing circuit.
    environment = {key: node.dtype for key, node in (enclosing or {}).items()}
    if enclosing is None:
        environment.update({node.value[0]: node.dtype for node, scope, _ in entries(circuit) if scope == "capture"})
    validate_control_flow(circuit, environment, ExecutionLimits(
        max_loop_iterations=limits.max_gate_expansion, max_control_depth=limits.max_nesting_depth), loop_depth=loop_depth)


def read_expression(reader, depth=0):
    reader.budget["classical_nodes"] = reader.budget.get("classical_nodes", 0)+1
    reader.bounded("max_ast_nodes", reader.budget["classical_nodes"])
    reader.bounded("max_expression_depth", depth)
    kind, dtype = reader.take(1), read_type(reader)
    if kind == b"x":
        source = reader.take(1)
        if source == b"C":
            index, = reader.unpack("!I")
            if reader.circuit_owner is None or index >= reader.circuit_owner.num_bits or dtype != expr.Bool():
                reader.error("Invalid QPY classical bit expression.")
            value = expr.cast(expr.bit(index), expr.Bool())
        elif source == b"R":
            from .qpy_control_records import read_resource
            length, = reader.unpack("!H")
            register = read_resource(reader, reader.text(length))
            from quantumbridge.core.bits import ClassicalRegister
            if type(register) is not ClassicalRegister:
                reader.error("QPY register expression does not reference a register.")
            value = expr.register([reader.circuit_owner.find_bit(bit).index for bit in register])
        elif source == b"U":
            # Both actual supported SDKs encode v13 variable indices as u16.
            index, = reader.unpack("!H")
            if index >= len(reader.classical_vars):
                reader.error("QPY variable expression index is outside its declaration table.")
            value = reader.classical_vars[index]
        else:
            reader.error("Unknown QPY variable expression reference.")
        if value.dtype != dtype:
            reader.error("QPY variable expression type differs from its resource.")
    elif kind == b"v":
        source = reader.take(1)
        if source == b"i":
            size, = reader.unpack("!B")
            value = int.from_bytes(reader.take(size), "big", signed=True)
            if dtype == expr.Bool():
                if value not in {0, 1}:
                    reader.error("QPY Bool literal is not zero or one.")
                value = bool(value)
        elif source == b"b":
            raw = reader.take(1)
            if raw not in {b"\0", b"\1"}:
                reader.error("Invalid QPY boolean literal encoding.")
            value = raw == b"\1"
        else:
            reader.error("Unsupported QPY13 classical literal.", UnsupportedFeature)
        value = expr.literal(value, dtype)
    elif kind == b"c":
        implicit = reader.take(1)
        if implicit not in {b"\0", b"\1"}:
            reader.error("Invalid QPY classical Cast flag.")
        value = expr.cast(read_expression(reader, depth+1), dtype)
    elif kind in {b"u", b"b"}:
        opcode, = reader.unpack("!B")
        table = UNARY if kind == b"u" else BINARY
        if opcode not in table:
            reader.error("Unsupported QPY classical expression opcode.", UnsupportedFeature)
        args = tuple(read_expression(reader, depth+1) for _ in range(1 if kind == b"u" else 2))
        value = expr.ClassicalExpression(table[opcode], dtype, args)
    else:
        reader.error("Unsupported QPY13 classical expression node.", UnsupportedFeature)
    expr.validate_classical_expr(value)
    return value


def expression_bytes(value, circuit, limits, budget, *, bits=None):
    from quantumbridge.core.circuit_data import admit
    admit((value, circuit))
    pending, count = [(value, 0, frozenset())], 0
    while pending:
        node, depth, ancestors = pending.pop()
        limits.check("max_expression_depth", depth, format="qpy", version=13)
        count += 1
        limits.check("max_ast_nodes", count+budget.get("classical_nodes", 0), format="qpy", version=13)
        if type(node) not in {expr.ClassicalExpression, expr.Var} or type(node.args) is not tuple:
            raise UnsupportedFeature("QPY classical expressions require admitted native nodes.", format="qpy", version=13)
        if id(node) in ancestors:
            raise FormatValidationError("Cyclic QPY classical expression.", format="qpy", version=13)
        path = ancestors | {id(node)}
        pending.extend((child, depth+1, path) for child in node.args)
    if bits is not None:
        value = value.remap(bits)
    variables = {node.value[0]: index for index, (node, _, _) in enumerate(entries(circuit))}
    parts = BoundedParts(limits, "qpy")
    active = set()

    def emit(node, depth):
        limits.check("max_expression_depth", depth, format="qpy", version=13)
        budget["classical_nodes"] = budget.get("classical_nodes", 0)+1
        limits.check("max_ast_nodes", budget["classical_nodes"], format="qpy", version=13)
        if id(node) in active:
            raise FormatValidationError("Cyclic QPY classical expression.", format="qpy", version=13)
        active.add(id(node))
        dtype = type_bytes(node.dtype)
        if node.op == "literal":
            number = int(node.value)
            size = max(0, ((number if number >= 0 else ~number).bit_length()+8)//8)
            parts.extend((b"v", dtype, b"i", struct.pack("!B", size), number.to_bytes(size, "big", signed=True)))
        elif node.op == "var":
            if node.value[0] not in variables:
                raise FormatValidationError("QPY expression refers to an undeclared variable.", format="qpy", version=13)
            parts.extend((b"x", dtype, b"U", struct.pack("!H", variables[node.value[0]])))
        elif node.op == "bit":
            circuit._check_bit(node.value)
            parts.extend((b"x", dtype, b"C", struct.pack("!I", node.value)))
        elif node.op == "register":
            from .qpy_control_records import resource_bytes
            register = next((reg for reg in circuit.cregs if tuple(circuit.find_bit(bit).index for bit in reg) == tuple(node.value)), None)
            payload = resource_bytes(circuit, register)
            parts.extend((b"x", dtype, b"R", struct.pack("!H", len(payload)), payload))
        elif node.op == "cast":
            if node.dtype == expr.Bool() and node.args[0].op == "bit":
                emit(node.args[0], depth+1)
            else:
                parts.extend((b"c", dtype, b"\0"))
                emit(node.args[0], depth+1)
        elif node.op in UNARY.values() or node.op in BINARY.values():
            table = UNARY if node.op in UNARY.values() else BINARY
            opcode = next(key for key, op in table.items() if op == node.op)
            parts.extend((b"u" if table is UNARY else b"b", dtype, struct.pack("!B", opcode)))
            for arg in node.args:
                emit(arg, depth+1)
        else:
            raise UnsupportedFeature("This native classical node has no implemented QPY13 record.", format="qpy", version=13)
        active.remove(id(node))

    expr.validate_classical_expr(value)
    emit(value, 0)
    return b"".join(parts)
