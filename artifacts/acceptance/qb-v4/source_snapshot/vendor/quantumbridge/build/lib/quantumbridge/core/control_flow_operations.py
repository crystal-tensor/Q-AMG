# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Native public control instructions owning their actual executable blocks."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from numbers import Integral

from .bits import ClassicalRegister, Clbit, bounded_items
from .classical import ClassicalExpression
from .control_flow import ControlBlock
from .errors import CircuitError
from .gate_limits import current_gate_limits
from .operations import Instruction
from .parameters import Parameter, parameter_sort_key, validate_parameter_names
from .control_builders import CASE_DEFAULT


_active = ContextVar("quantumbridge_control_owner_graph", default=())


@contextmanager
def _guard(owner, purpose):
    key, previous = (id(owner), purpose), _active.get()
    if key in previous:
        raise CircuitError("Cyclic public control-flow owner graph.")
    current_gate_limits().check("max_definition_depth", len(previous)+1)
    token = _active.set(previous+(key,))
    try:
        yield
    finally:
        _active.reset(token)


def _body(value):
    from .circuit import is_native_circuit
    if not is_native_circuit(value):
        raise CircuitError("A control-flow block must be a native Circuit.")
    return value


def _condition(value):
    from .circuit_data import admit
    admit(value)
    if isinstance(value, ClassicalExpression):
        if value.dtype.kind not in {"bool", "bit"}:
            raise CircuitError("A control condition requires a boolean expression.")
        return value
    if (type(value) is not tuple or len(value) != 2 or
            type(value[0]) not in {Clbit, ClassicalRegister} or not isinstance(value[1], Integral)):
        raise CircuitError("A control condition requires an expression or (Clbit/Register, integer).")
    return value


class ControlFlowOp(Instruction, ABC):
    def __init__(self, name, num_qubits, num_clbits, params, label=None):
        object.__setattr__(self, "_control_params", [])
        Instruction.__init__(self, name, num_qubits, num_clbits, (), label)
        self.params = params

    def __setattr__(self, name, value):
        if name in {"condition", "target"}:
            return object.__setattr__(self, name, value)
        return super().__setattr__(name, value)

    @property
    def params(self):
        return self._control_params

    @params.setter
    def params(self, values):
        from .circuit_data import admit
        admit(values)
        candidate = list(bounded_items(values))
        self._validate_params(candidate)
        object.__setattr__(self, "_control_params", candidate)

    def _validate_widths(self, blocks):
        for body in blocks:
            _body(body)
            if (body.num_qubits, body.num_bits) != (self.num_qubits, self.num_clbits):
                raise CircuitError("Control block width differs from its instruction owner.")

    @abstractmethod
    def _validate_params(self, values):
        raise NotImplementedError

    @property
    @abstractmethod
    def blocks(self):
        raise NotImplementedError

    def validate(self):
        self._validate_params(self.params)

    @property
    def parameters(self):
        with _guard(self, "parameters"):
            self.validate()
            found = set().union(*(set(body.parameters) for body in self.blocks))
            if type(self) is ForLoopOp:
                found.discard(self.params[1])
            validate_parameter_names(found)
            return tuple(sorted(found, key=parameter_sort_key))

    def is_parameterized(self):
        return bool(self.parameters)

    def assign_parameters(self, mapping, *, strict=True):
        from .instruction_parameters import bind_definition
        from .circuit import _parameter_mapping_from_values
        values = _parameter_mapping_from_values(self.parameters, mapping, strict)
        if type(self) is ForLoopOp and self.params[1] is not None:
            values = {key: value for key, value in values.items() if key != self.params[1]}
        blocks = [bind_definition(body, values) for body in self.blocks]
        return self.replace_blocks(blocks)

    @abstractmethod
    def replace_blocks(self, blocks):
        raise NotImplementedError

    def iter_captured_vars(self):
        return iter(dict.fromkeys(variable for body in self.blocks for variable in body.iter_captured_vars()))

    def iter_captured_stretches(self):
        return iter(dict.fromkeys(variable for body in self.blocks for variable in body.iter_captured_stretches()))


class _Conditional:
    @property
    def condition(self):
        return self._condition

    @condition.setter
    def condition(self, value):
        object.__setattr__(self, "_condition", _condition(value))


class IfElseOp(_Conditional, ControlFlowOp):
    def __init__(self, condition, true_body, false_body=None, label=None):
        body = _body(true_body)
        self.condition = condition
        super().__init__("if_else", body.num_qubits, body.num_bits, [body, false_body], label)

    def _validate_params(self, values):
        true_body, false_body = values
        self._validate_widths([true_body] if false_body is None else [true_body, false_body])

    @property
    def blocks(self):
        return tuple(body for body in self.params if body is not None)

    def replace_blocks(self, blocks):
        blocks = bounded_items(blocks)
        if not 1 <= len(blocks) <= 2:
            raise CircuitError("IfElseOp requires one or two replacement blocks.")
        return IfElseOp(self.condition, blocks[0], None if len(blocks) == 1 else blocks[1], self.label)


class ForLoopOp(ControlFlowOp):
    def __init__(self, indexset, loop_parameter, body, label=None):
        from .control_builders import loop_values
        body = _body(body)
        values = loop_values(indexset)
        indexset = indexset if type(indexset) is range else values
        super().__init__("for_loop", body.num_qubits, body.num_bits, [indexset, loop_parameter, body], label)

    def _validate_params(self, values):
        indexset, parameter, body = values
        from .control_builders import loop_values
        loop_values(indexset)
        if parameter is not None and not isinstance(parameter, Parameter):
            raise CircuitError("ForLoopOp loop parameter must be a native Parameter or None.")
        self._validate_widths([body])

    @property
    def blocks(self):
        return (self.params[2],)

    def replace_blocks(self, blocks):
        blocks = bounded_items(blocks)
        if len(blocks) != 1:
            raise CircuitError("ForLoopOp requires one replacement body.")
        return ForLoopOp(self.params[0], self.params[1], blocks[0], self.label)


class WhileLoopOp(_Conditional, ControlFlowOp):
    def __init__(self, condition, body, label=None):
        body = _body(body)
        self.condition = condition
        super().__init__("while_loop", body.num_qubits, body.num_bits, [body], label)

    def _validate_params(self, values):
        if len(values) != 1:
            raise ValueError("WhileLoopOp requires one body parameter.")
        self._validate_widths(values)

    @property
    def blocks(self):
        return tuple(self.params)

    def replace_blocks(self, blocks):
        blocks = bounded_items(blocks)
        if len(blocks) != 1:
            raise CircuitError("WhileLoopOp requires one replacement body.")
        return WhileLoopOp(self.condition, blocks[0], self.label)


class SwitchCaseOp(ControlFlowOp):
    def __init__(self, target, cases, *, label=None):
        from .circuit_data import admit
        admit((target, cases, label))
        self.target = target
        groups, blocks, seen = [], [], set()
        for values, body in bounded_items(cases):
            values = (values,) if isinstance(values, Integral) or values is CASE_DEFAULT else bounded_items(values)
            if not values or any(value is CASE_DEFAULT for value in values) and len(values) != 1:
                raise CircuitError("Default must be the only label in its case.")
            for value in values:
                if value is not CASE_DEFAULT and (not isinstance(value, Integral) or value < 0):
                    raise CircuitError("Switch labels require nonnegative integers.")
                if value in seen:
                    raise CircuitError("Duplicate switch case label.")
                seen.add(value)
            groups.append(values)
            blocks.append(_body(body))
        if not blocks:
            raise CircuitError("SwitchCaseOp requires at least one case.")
        object.__setattr__(self, "_case_groups", tuple(groups))
        super().__init__("switch_case", blocks[0].num_qubits, blocks[0].num_bits, blocks, label)
        self.validate()

    @ControlFlowOp.params.setter
    def params(self, values):
        # The public setter accepts live block replacement; executable use
        # still validates the resulting graph before any shot is admitted.
        from .circuit_data import admit
        admit(values)
        object.__setattr__(self, "_control_params", list(bounded_items(values)))

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        if type(value) not in {Clbit, ClassicalRegister} and not isinstance(value, ClassicalExpression):
            raise CircuitError("A switch target requires a classical resource or expression.")
        object.__setattr__(self, "_target", value)

    def _validate_params(self, values):
        if len(values) != len(self._case_groups):
            raise CircuitError("SwitchCaseOp replacement blocks must match its case groups.")
        self._validate_widths(values)

    @property
    def blocks(self):
        return tuple(self.params)

    def cases(self):
        return {value: block for values, block in self.cases_specifier() for value in values}

    def cases_specifier(self):
        return iter(zip(self._case_groups, self.blocks))

    def replace_blocks(self, blocks):
        blocks = bounded_items(blocks)
        if len(blocks) != len(self._case_groups):
            raise CircuitError("SwitchCaseOp replacement blocks must match its cases.")
        return SwitchCaseOp(self.target, list(zip(self._case_groups, blocks)), label=self.label)


class BreakLoopOp(Instruction):
    def __init__(self, num_qubits, num_clbits, label=None):
        super().__init__("break_loop", num_qubits, num_clbits, (), label)
        object.__setattr__(self, "_jump_params", [])

    @property
    def params(self):
        return self._jump_params

    @params.setter
    def params(self, values):
        from .instruction_parameters import admit_payloads
        admit_payloads(values)
        object.__setattr__(self, "_jump_params", list(bounded_items(values)))


class ContinueLoopOp(Instruction):
    def __init__(self, num_qubits, num_clbits, label=None):
        super().__init__("continue_loop", num_qubits, num_clbits, (), label)
        object.__setattr__(self, "_jump_params", [])

    @property
    def params(self):
        return self._jump_params

    @params.setter
    def params(self, values):
        from .instruction_parameters import admit_payloads
        admit_payloads(values)
        object.__setattr__(self, "_jump_params", list(bounded_items(values)))


class Store(Instruction):
    """Public runtime assignment using the existing native classical store."""

    def __init__(self, lvalue, rvalue):
        object.__setattr__(self, "_store_params", [])
        super().__init__("store", 0, 0, ())
        self.params = [lvalue, rvalue]

    @property
    def params(self):
        return self._store_params

    @params.setter
    def params(self, values):
        from .circuit_data import admit
        admit(values)
        candidate = list(bounded_items(values))
        self._validate(candidate)
        object.__setattr__(self, "_store_params", candidate)

    @staticmethod
    def _validate(values):
        from .classical import validate_classical_expr
        if len(values) != 2:
            raise CircuitError("Store requires a target and a value.")
        target, value = values
        validate_classical_expr(target)
        validate_classical_expr(value)
        if target.op not in {"var", "bit", "register"} or target.dtype != value.dtype:
            raise CircuitError("Store requires a writable target and a matching expression type.")

    @property
    def lvalue(self):
        return self.params[0]

    @property
    def rvalue(self):
        return self.params[1]

    @property
    def parameters(self):
        return ()

    def assign_parameters(self, mapping, *, strict=True):
        from .circuit import _parameter_mapping_from_values
        _parameter_mapping_from_values((), mapping, strict)
        return self.copy()


CONTROL_TYPES = (IfElseOp, ForLoopOp, WhileLoopOp, SwitchCaseOp, BreakLoopOp, ContinueLoopOp)
STATEMENT_TYPES = (Store,)


def is_control_owner(value):
    return type(value) in CONTROL_TYPES+STATEMENT_TYPES


def operation_owner(operation):
    return object.__getattribute__(operation, "__dict__").get("_control_owner")


def _selector(value, bit_bindings, expression_map):
    from .classical import bit, register
    if isinstance(value, ClassicalExpression):
        return value.remap(expression_map)
    bits = (value,) if type(value) is Clbit else tuple(value)
    try:
        indices = [bit_bindings[item._key] for item in bits]
    except KeyError:
        raise CircuitError("Control condition refers to a foreign classical bit.") from None
    return bit(indices[0]) if type(value) is Clbit else register(indices)


def refresh_control(operation):
    """Materialize the existing native execution block from its live owner."""
    fields = object.__getattribute__(operation, "__dict__")
    owner, metadata = fields.get("_control_owner"), fields["metadata"]
    if owner is None:
        return metadata
    from .instruction_parameters import definition_signature
    from .circuit import _remap_circuit
    with _guard(owner, "resolve"):
        if isinstance(owner, ControlFlowOp):
            owner.validate()
            blocks = owner.blocks
            from .operations import instruction_definition_work
            work = 1+sum(instruction_definition_work(body) for body in blocks)
            current_gate_limits().check("max_total_operations", work)
            current_gate_limits().check("max_decomposition_work", work)
            signature = [owner.name, owner.label, [(definition_signature(body), tuple(p.uuid for p in body.parameters)) for body in blocks]]
        elif type(owner) is Store:
            owner._validate(owner.params)
            blocks, signature = (), [owner.name, owner.label, [value.to_dict() for value in owner.params]]
        else:
            blocks, signature = (), [owner.name, owner.label]
            if owner.params:
                raise CircuitError("Loop jump instructions do not take executable parameters.")
        resource = getattr(owner, "condition", getattr(owner, "target", None))
        if isinstance(resource, tuple):
            value, expected = resource
            resource_signature = (tuple(item._key for item in value) if type(value) is ClassicalRegister else value._key, int(expected))
        elif isinstance(resource, ClassicalExpression):
            resource_signature = resource.to_dict()
        elif resource is not None:
            resource_signature = tuple(item._key for item in resource) if type(resource) is ClassicalRegister else resource._key
        else:
            resource_signature = None
        signature.append(resource_signature)
        if type(owner) is ForLoopOp:
            signature.append((tuple(owner.params[0]), None if owner.params[1] is None else owner.params[1].uuid))
        if type(owner) is SwitchCaseOp:
            signature.append(owner._case_groups)
        signature.append((fields["targets"], fields["_control_width"],
                          tuple(metadata.get("instruction_clbits", ())),
                          tuple(fields["_control_bit_bindings"].items()),
                          tuple(fields["_control_expression_map"].items())))
        if fields.get("_control_signature") == signature:
            return metadata
        qmap, cmap = dict(enumerate(fields["targets"])), dict(enumerate(metadata.get("instruction_clbits", ())))
        width, bits = fields["_control_width"]
        if (len(qmap), len(cmap)) != (owner.num_qubits, owner.num_clbits):
            raise CircuitError("Control invocation width differs from its owner.")
        mapped = [_remap_circuit(body, qmap, cmap, width, bits, copy_operation=True) for body in blocks]
        def bind(value):
            return _selector(value, fields["_control_bit_bindings"], fields["_control_expression_map"])
        if type(owner) in (IfElseOp, WhileLoopOp):
            condition = bind(owner.condition) if isinstance(owner.condition, ClassicalExpression) else bind(owner.condition[0]).eq(int(owner.condition[1]))
        if type(owner) is IfElseOp:
            node = ControlBlock("if", {"condition": condition, "body": mapped[0], "alternate": None if len(mapped) == 1 else mapped[1]})
        elif type(owner) is ForLoopOp:
            node = ControlBlock("for", {"values": tuple(owner.params[0]), "parameter": owner.params[1], "variable": None, "body": mapped[0]})
        elif type(owner) is WhileLoopOp:
            node = ControlBlock("while", {"condition": condition, "body": mapped[0]})
        elif type(owner) is SwitchCaseOp:
            cases, default = [], None
            for labels, body in zip(owner._case_groups, mapped):
                if labels[0] is CASE_DEFAULT:
                    default = body
                else:
                    cases.append((labels, body))
            node = ControlBlock("switch", {"selector": bind(owner.target), "cases": tuple(cases), "default": default})
        elif type(owner) is Store:
            node = ControlBlock("store", {"target": bind(owner.lvalue), "value": bind(owner.rvalue)})
        else:
            node = ControlBlock("break" if type(owner) is BreakLoopOp else "continue", {})
        metadata["control_node"] = node
        if owner.label is None:
            metadata.pop("label", None)
        else:
            metadata["label"] = owner.label
        fields["_control_signature"] = signature
    return metadata


def control_entry(circuit, owner, qubits, clbits, *, expression_map=None):
    from .bits import operand_index
    from .circuit_data import admit
    from .operations import Operation
    admit((circuit, owner, qubits, clbits))
    qubits, clbits = tuple(qubits), tuple(clbits)
    if len(set(qubits)) != len(qubits) or len(set(clbits)) != len(clbits):
        raise CircuitError("Control operands must be distinct.")
    for value in qubits:
        circuit._check_wire(value)
    for value in clbits:
        circuit._check_bit(value)
    if (len(qubits), len(clbits)) != (owner.num_qubits, owner.num_clbits):
        raise CircuitError("Control operands differ from its operation widths.")
    bindings = {item._key: operand_index(circuit, item, False) for item in circuit.clbits}
    result = Operation(owner.name, qubits, metadata={"instruction_clbits": clbits})
    for name, value in {"_control_owner": owner, "_control_width": (circuit.num_qubits, circuit.num_bits),
                        "_control_bit_bindings": bindings,
                        "_control_expression_map": (dict(enumerate(range(circuit.num_bits))) if type(owner) is Store
                            else dict(enumerate(clbits))) if expression_map is None else expression_map}.items():
        object.__setattr__(result, name, value)
    refresh_control(result)
    return result


def remap_control_entry(operation, qubits, bits, width, num_bits, *, owner=None,
                        identifiers=None, copy_operation=True, target_circuit=None):
    from .operations import Operation
    fields = object.__getattribute__(operation, "__dict__")
    original = fields["_control_owner"]
    owner = original.copy() if owner is None and copy_operation else original if owner is None else owner
    width = max(qubits.values(), default=-1)+1 if width is None else width
    num_bits = max(bits.values(), default=-1)+1 if num_bits is None else num_bits
    bindings = {key: bits[index] for key, index in fields["_control_bit_bindings"].items() if index in bits}
    expression_map = {key: bits[index] for key, index in fields["_control_expression_map"].items() if index in bits}
    resource = getattr(owner, "condition", getattr(owner, "target", None))
    if target_circuit is not None and resource is not None and not isinstance(resource, ClassicalExpression):
        source = resource[0] if isinstance(resource, tuple) else resource
        if type(source) is Clbit:
            mapped = target_circuit.clbits[bindings[source._key]]
        else:
            members = [target_circuit.clbits[bindings[bit._key]] for bit in source]
            mapped = next((register for register in target_circuit.cregs if tuple(register) == tuple(members)),
                          ClassicalRegister(bits=members, name=source.name))
        if owner is original:
            owner = owner.copy()
        if isinstance(resource, tuple):
            owner.condition = (mapped, resource[1])
        else:
            owner.target = mapped
        for bit in (mapped,) if type(mapped) is Clbit else mapped:
            bindings[bit._key] = target_circuit.find_bit(bit).index
    if identifiers:
        if owner is original:
            owner = owner.copy()
        if isinstance(resource, ClassicalExpression):
            updated = resource.remap(dict(enumerate(range(fields["_control_width"][1]))), identifiers)
            if hasattr(owner, "condition"):
                owner.condition = updated
            else:
                owner.target = updated
        if type(owner) is Store:
            owner.params = [value.remap(dict(enumerate(range(fields["_control_width"][1]))), identifiers)
                            for value in owner.params]
        from .circuit import _remap_circuit
        for body in getattr(owner, "blocks", ()):
            planned = _remap_circuit(body, dict(enumerate(range(body.num_qubits))),
                dict(enumerate(range(body.num_bits))), body.num_qubits, body.num_bits, identifiers=identifiers)
            body.operations, body.measurements = planned.operations, planned.measurements
            body._classical_identifiers = planned.__dict__.get("_classical_identifiers", {})
            from .classical_scope import entries, set_entries
            set_entries(body, entries(planned))
    metadata = {"instruction_clbits": tuple(bits[index] for index in fields["metadata"].get("instruction_clbits", ()))}
    result = Operation(owner.name, tuple(qubits[index] for index in operation.targets), metadata=metadata)
    for key, value in {"_control_owner": owner, "_control_width": (width, num_bits),
                       "_control_bit_bindings": bindings, "_control_expression_map": expression_map}.items():
        object.__setattr__(result, key, value)
    if fields.get("_builder_jump"):
        object.__setattr__(result, "_builder_jump", True)
    refresh_control(result)
    return result
