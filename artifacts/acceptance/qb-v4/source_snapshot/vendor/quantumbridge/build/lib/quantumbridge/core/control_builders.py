# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Lexical public builders over the existing authoritative native block store."""

from contextvars import ContextVar
from itertools import count, islice
from numbers import Integral

from .bits import ClassicalRegister, Clbit, operand_index
from .classical import ClassicalExpression, bit, register, validate_classical_expr
from .classical_scope import execution_environment
from .control_flow import ControlBlock, condition_expression
from .errors import CircuitError
from .gate_limits import current_gate_limits
from .parameters import Parameter, validate_parameter_names


_frames = ContextVar("quantumbridge_public_control_builders", default=())
_loop_names = count()


class _DefaultCase:
    def __deepcopy__(self, memo):
        return self


CASE_DEFAULT = _DefaultCase()


def lexical_environment(circuit):
    enclosing = next((frame.environment for frame in reversed(_frames.get()) if frame.circuit is circuit), None)
    return execution_environment(circuit, enclosing)


def selector(circuit, value):
    from .circuit_data import admit
    admit((circuit, value))
    if isinstance(value, ClassicalExpression):
        validate_classical_expr(value, lexical_environment(circuit), circuit.num_bits)
        return value
    if type(value) is ClassicalRegister:
        if not any(value is existing for existing in circuit.cregs):
            raise CircuitError("Condition register has not been added to this Circuit.")
        return register([operand_index(circuit, item, False) for item in value])
    if type(value) is Clbit or isinstance(value, Integral):
        return bit(operand_index(circuit, value, False))
    raise CircuitError("A control selector requires a native classical bit, register or expression.")


def condition(circuit, value):
    from .circuit_data import admit
    admit((circuit, value))
    if isinstance(value, ClassicalExpression):
        return condition_expression(selector(circuit, value))
    if not isinstance(value, tuple) or len(value) != 2:
        raise CircuitError("A public condition requires a classical expression or (resource, value).")
    resource, expected = value
    if not isinstance(expected, Integral):
        raise CircuitError("A condition comparison requires an integer value.")
    return condition_expression(selector(circuit, resource).eq(int(expected)))


def _expression_bits(expression):
    if not isinstance(expression, ClassicalExpression):
        return ()
    if expression.op == "bit":
        return (expression.value,)
    if expression.op == "register":
        return tuple(expression.value)
    return tuple(index for child in expression.args for index in _expression_bits(child))


def _resources(bodies, expression=None):
    from .measurements import Measurement
    quantum, classical = {}, dict.fromkeys(_expression_bits(expression))

    def visit(body):
        for item in body.iter_instructions():
            if type(item) is Measurement:
                quantum[item.wire] = None
                classical[item.bit] = None
                continue
            quantum.update(dict.fromkeys(item.wires))
            classical.update(dict.fromkeys(item.metadata.get("instruction_clbits", ())))
            node = item.metadata.get("control_node")
            if node is not None:
                for value in node.data.values():
                    classical.update(dict.fromkeys(_expression_bits(value)))
                for child in node.bodies:
                    visit(child)
            for key in ("condition", "gate_condition"):
                value = item.metadata.get(key)
                if isinstance(value, dict) and "bits" in value:
                    classical.update(dict.fromkeys(value["bits"]))

    for body in bodies:
        visit(body)
    return tuple(quantum), tuple(classical)


def _body_plan(circuit, body, quantum, classical):
    from .circuit import _remap_circuit
    from .bits import install_bits
    planned = _remap_circuit(body, {wire: index for index, wire in enumerate(quantum)},
        {bit: index for index, bit in enumerate(classical)}, len(quantum), len(classical), copy_operation=False)
    qubits, clbits = tuple(circuit.qubits[i] for i in quantum), tuple(circuit.clbits[i] for i in classical)
    qregs = tuple(reg for reg in circuit.qregs if all(bit in qubits for bit in reg))
    cregs = tuple(reg for reg in circuit.cregs if all(bit in clbits for bit in reg))
    install_bits(planned, qubits, clbits, qregs, cregs)
    return planned


def _commit_body(body, planned):
    from .bits import copy_bits
    from .circuit_metadata import inherit_execution
    from .classical_scope import entries, set_entries
    body.operations, body.measurements = planned.operations, planned.measurements
    copy_bits(planned, body)
    inherit_execution(planned, body)
    set_entries(body, entries(planned))


def _public_resource(circuit, resource):
    if isinstance(resource, tuple):
        return (_public_resource(circuit, resource[0]), resource[1])
    return circuit.clbits[int(resource)] if isinstance(resource, Integral) else resource


def _pending_jump(body):
    from .control_flow_operations import operation_owner, ForLoopOp, WhileLoopOp
    for operation in body.operations:
        if object.__getattribute__(operation, "__dict__").get("_builder_jump"):
            return True
        owner = operation_owner(operation)
        if owner is not None and type(owner) not in {ForLoopOp, WhileLoopOp}:
            if any(_pending_jump(block) for block in getattr(owner, "blocks", ())):
                return True
    return False


def _resolve_jumps(body, quantum, classical):
    from .control_flow_operations import (operation_owner, control_entry, BreakLoopOp,
                                        ContinueLoopOp, ForLoopOp, WhileLoopOp)
    from .circuit import _remap_circuit
    for position, operation in enumerate(body.operations):
        fields = object.__getattribute__(operation, "__dict__")
        owner = operation_owner(operation)
        if fields.get("_builder_jump"):
            kind = BreakLoopOp if type(owner) is BreakLoopOp else ContinueLoopOp
            body.operations[position] = control_entry(body, kind(len(quantum), len(classical)), quantum, classical)
            continue
        if owner is None or type(owner) in {ForLoopOp, WhileLoopOp} or not any(
                _pending_jump(block) for block in getattr(owner, "blocks", ())):
            continue
        expanded = [_remap_circuit(block, dict(enumerate(operation.targets)),
            dict(enumerate(operation.metadata["instruction_clbits"])), body.num_qubits,
            body.num_bits, copy_operation=False) for block in owner.blocks]
        for block in expanded:
            _resolve_jumps(block, quantum, classical)
        qargs, cargs = _resources(expanded, operation.metadata["control_node"].data.get("condition",
                                    operation.metadata["control_node"].data.get("selector")))
        plans = [_body_plan(body, block, qargs, cargs) for block in expanded]
        candidate = owner.replace_blocks(plans)
        planned = control_entry(body, candidate, qargs, cargs,
                                expression_map=fields["_control_expression_map"])
        originals = owner.blocks
        for block, plan in zip(originals, plans):
            _commit_body(block, plan)
        object.__setattr__(owner, "num_qubits", candidate.num_qubits)
        object.__setattr__(owner, "num_bits", candidate.num_bits)
        if candidate.name == "if_else":
            owner.params = [originals[0], None if len(originals) == 1 else originals[1]]
        else:
            owner.params = list(originals)
        replacement = object.__getattribute__(planned, "__dict__")
        replacement["_control_owner"] = owner
        replacement.pop("_control_signature", None)
        fields.update(replacement)


def _typed_entry(circuit, node, label, source, indexset):
    from .control_flow_operations import (IfElseOp, ForLoopOp, WhileLoopOp, SwitchCaseOp,
        BreakLoopOp, ContinueLoopOp, Store, control_entry)
    if node.kind == "store":
        return control_entry(circuit, Store(node.data["target"], node.data["value"]), (), ())
    if node.kind in {"break", "continue"}:
        kind = BreakLoopOp if node.kind == "break" else ContinueLoopOp
        entry = control_entry(circuit, kind(0, 0, label), (), ())
        object.__setattr__(entry, "_builder_jump", True)
        return entry
    expression = node.data.get("condition", node.data.get("selector"))
    quantum, classical = _resources(node.bodies, expression)
    if node.kind in {"for", "while"}:
        for body in node.bodies:
            _resolve_jumps(body, quantum, classical)
    plans = [_body_plan(circuit, body, quantum, classical) for body in node.bodies]
    source = _public_resource(circuit, expression if source is None else source)
    if node.kind == "if":
        owner = IfElseOp(source, plans[0], None if len(plans) == 1 else plans[1], label)
    elif node.kind == "for":
        owner = ForLoopOp(node.data["values"] if indexset is None else indexset, node.data["parameter"], plans[0], label)
    elif node.kind == "while":
        owner = WhileLoopOp(source, plans[0], label)
    else:
        groups = [labels for labels, _ in node.data["cases"]]
        if node.data["default"] is not None:
            groups.append((CASE_DEFAULT,))
        owner = SwitchCaseOp(source, list(zip(groups, plans)), label=label)
    # Validate the complete candidate before changing escaped builder handles.
    entry = control_entry(circuit, owner, quantum, classical,
                          expression_map=dict(enumerate(range(circuit.num_bits))))
    for body, planned in zip(node.bodies, plans):
        _commit_body(body, planned)
    if node.kind == "if":
        owner.params = [node.bodies[0], None if len(node.bodies) == 1 else node.bodies[1]]
    elif node.kind == "for":
        owner.params = [owner.params[0], owner.params[1], node.bodies[0]]
    else:
        owner.params = list(node.bodies)
    return entry


def append_block(circuit, node, label=None, *, source=None, indexset=None):
    from .circuit import Circuit
    from .circuit_data import InstructionSet, admit
    from .operations import instruction_definition_work

    admit((circuit, node, label))
    if label is not None and not isinstance(label, str):
        raise TypeError("A control label must be a string or None.")
    work = instruction_definition_work(circuit) + 1
    for body in node.bodies:
        work += instruction_definition_work(body)
    current_gate_limits().check("max_total_operations", work)
    current_gate_limits().check("max_decomposition_work", work)
    validate_parameter_names((*circuit.parameters, *node.parameters))
    before = len(circuit.operations)+len(circuit.measurements)
    if node.kind in {"if", "for", "while", "switch", "break", "continue", "store"}:
        circuit.operations.append(_typed_entry(circuit, node, label, source, indexset))
    else:
        Circuit.append(circuit, "control_flow", (), metadata={"control_node": node, **({"label": label} if label is not None else {})})
    return InstructionSet.bound(handle_owner(circuit), [before])


def loop_values(values):
    from .circuit_data import admit
    admit(values)
    maximum = min(1000000, current_gate_limits().max_decomposition_work)
    if isinstance(values, (tuple, list, range)) and len(values) > maximum:
        raise MemoryError("Loop index set exceeds the construction work limit.")
    result = tuple(islice(iter(values), maximum+1))
    if len(result) > maximum:
        raise MemoryError("Loop index set exceeds the construction work limit.")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in result):
        raise CircuitError("Loop index sets require integer values.")
    return tuple(int(value) for value in result)


class _Scope:
    def __init__(self, circuit, kind, finish, result=None):
        self.circuit, self.kind, self.finish, self.result = circuit, kind, finish, result
        self.entered = False

    def __enter__(self):
        if self.entered:
            raise CircuitError("A control builder context cannot be reused.")
        stack = _frames.get()
        current_gate_limits().check("max_definition_depth", len(stack)+1)
        self.environment = lexical_environment(self.circuit)
        body = self.circuit.copy_empty_like(vars_mode="captures")
        body.global_phase = 0
        self.body = body
        parent = self.circuit
        self.saved = (parent.operations, parent.measurements, parent._execution_metadata,
                      parent.__dict__.get("_classical_identifiers"))
        parent.operations, parent.measurements = body.operations, body.measurements
        parent._execution_metadata = body._execution_metadata
        parent._classical_identifiers = body.__dict__.get("_classical_identifiers", {})
        self.token = _frames.set(stack + (self,))
        self.entered = True
        return self.result

    def __exit__(self, exc_type, exc, traceback):
        parent, body = self.circuit, self.body
        body.operations, body.measurements = parent.operations, parent.measurements
        body._execution_metadata = parent._execution_metadata
        body._classical_identifiers = parent.__dict__.get("_classical_identifiers", {})
        parent.operations, parent.measurements, parent._execution_metadata, identifiers = self.saved
        if identifiers is None:
            parent.__dict__.pop("_classical_identifiers", None)
        else:
            parent._classical_identifiers = identifiers
        _frames.reset(self.token)
        if exc_type is None:
            self.finish(body)
        return False


def handle_owner(circuit):
    """Captured handles keep their actual block store after the scope closes."""
    for frame in reversed(_frames.get()):
        if frame.circuit is circuit:
            frame.body.operations, frame.body.measurements = circuit.operations, circuit.measurements
            frame.body._execution_metadata = circuit._execution_metadata
            return frame.body
    return circuit


def check_bit_mutation(circuit):
    if any(frame.circuit is circuit for frame in _frames.get()):
        raise CircuitError("Circuit bits and registers cannot change inside a control builder.")


def explicit_operands(qubits, clbits):
    if qubits is None or clbits is None:
        raise CircuitError("Explicit control bodies require both quantum and classical operands.")


class _Else:
    def __init__(self, circuit):
        self.circuit, self.operation, self.used = circuit, None, False

    def __enter__(self):
        items = list(self.circuit.iter_instructions())
        if self.used or self.operation is None or not items or items[-1] is not self.operation:
            raise CircuitError("Else must immediately follow its matching completed if context.")
        self.used = True
        self.scope = _Scope(self.circuit, "else", self.finish)
        return self.scope.__enter__()

    def finish(self, body):
        from .operations import instruction_definition_work
        work = instruction_definition_work(self.circuit) + instruction_definition_work(body)
        current_gate_limits().check("max_total_operations", work)
        validate_parameter_names((*self.circuit.parameters, *body.parameters))
        from .control_flow_operations import operation_owner
        from .circuit import _remap_circuit
        previous = self.operation.metadata["control_node"]
        owner = operation_owner(self.operation)
        original = owner.blocks[0]
        expanded = _remap_circuit(original, dict(enumerate(self.operation.targets)),
            dict(enumerate(self.operation.metadata["instruction_clbits"])),
            self.circuit.num_qubits, self.circuit.num_bits, copy_operation=False)
        node = ControlBlock("if", {**previous.data, "body": expanded, "alternate": body})
        planned = _typed_entry(self.circuit, node, owner.label, owner.condition, None)
        candidate = operation_owner(planned)
        _commit_body(original, candidate.blocks[0])
        object.__setattr__(owner, "num_qubits", candidate.num_qubits)
        object.__setattr__(owner, "num_bits", candidate.num_bits)
        owner.params = [original, candidate.blocks[1]]
        fields = object.__getattribute__(planned, "__dict__")
        fields["_control_owner"] = owner
        fields.pop("_control_signature", None)
        object.__getattribute__(self.operation, "__dict__").update(fields)

    def __exit__(self, *error):
        return self.scope.__exit__(*error)


def if_context(circuit, expression, label=None, *, source=None):
    otherwise = _Else(circuit)

    def finish(body):
        append_block(circuit, ControlBlock("if", {"condition": expression, "body": body, "alternate": None}), label, source=source)
        otherwise.operation = circuit.operations[-1]

    return _Scope(circuit, "if", finish, otherwise)


def for_context(circuit, values, parameter, label=None, *, indexset=None):
    automatic = parameter is None
    parameter = Parameter("_loop_i_" + str(next(_loop_names))) if automatic else parameter
    if not isinstance(parameter, Parameter):
        raise CircuitError("A for-loop parameter must be a native Parameter or None.")

    def finish(body):
        chosen = None if automatic and parameter not in body.parameters else parameter
        append_block(circuit, ControlBlock("for", {"values": values, "parameter": chosen,
                     "variable": None, "body": body}), label, indexset=indexset)

    return _Scope(circuit, "for", finish, parameter)


def while_context(circuit, expression, label=None, *, source=None):
    return _Scope(circuit, "while", lambda body: append_block(circuit,
        ControlBlock("while", {"condition": expression, "body": body}), label, source=source))


class _Cases:
    DEFAULT = CASE_DEFAULT

    def __init__(self, circuit, expression, label, source=None):
        self.circuit, self.expression, self.label = circuit, expression, label
        self.source = source
        self.cases, self.default, self.seen = [], None, set()
        self.scope = _Scope(circuit, "switch", self.finish, self)

    def __enter__(self):
        return self.scope.__enter__()

    def __exit__(self, *error):
        return self.scope.__exit__(*error)

    def __call__(self, *values):
        from .circuit_data import admit
        admit(values)
        if not _frames.get() or _frames.get()[-1] is not self.scope:
            raise CircuitError("Switch cases must be directly inside their switch builder.")
        if not values or any(value is CASE_DEFAULT for value in values) and len(values) != 1:
            raise CircuitError("A switch case requires distinct values or the default alone.")
        default = values[0] is CASE_DEFAULT
        if default and self.default is not None:
            raise CircuitError("A switch can only have one default case.")
        if not default:
            if any(not isinstance(value, Integral) or value in self.seen for value in values) or len(set(values)) != len(values):
                raise CircuitError("Switch cases require distinct integer values.")
            for value in values:
                self.expression.dtype.normalize(int(value))

        def finish(body):
            if default:
                self.default = body
            else:
                self.seen.update(values)
                self.cases.append((tuple(int(value) for value in values), body))

        return _Scope(self.circuit, "case", finish)

    def finish(self, loose):
        if loose.operations or loose.measurements or loose.global_phase:
            raise CircuitError("Instructions in a switch builder must be inside a case.")
        append_block(self.circuit, ControlBlock("switch", {"selector": self.expression,
            "cases": tuple(self.cases), "default": self.default}), self.label, source=self.source)


def switch_context(circuit, expression, label=None, *, source=None):
    return _Cases(circuit, expression, label, source)


def jump(circuit, kind):
    if not any(frame.circuit is circuit and frame.kind in {"for", "while"} for frame in _frames.get()):
        raise CircuitError("A loop jump needs an enclosing loop builder.")
    return append_block(circuit, ControlBlock(kind, {}))
