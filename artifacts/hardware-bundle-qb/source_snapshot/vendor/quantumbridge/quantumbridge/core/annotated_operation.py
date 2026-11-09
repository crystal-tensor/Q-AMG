# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Ordered deferred native operations, resolved by the existing Gate engine."""

from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import islice
from numbers import Real

from .gates import Gate, ControlledGate, _copy, _integer, _validate_primitive_arity, validate_definition
from .operations import Instruction
from .parameters import Parameter, parameter_set, parameter_sort_key, resolve_parameter, _coerce_expression
from .gate_limits import current_gate_limits, gate_circuit_scope

_VALIDATION_PATH = ContextVar("native_annotation_validation_path", default=frozenset())


class Modifier:
    """Marker for the native, explicitly validated modifier value types."""


@dataclass(frozen=True)
class InverseModifier(Modifier):
    pass


@dataclass(frozen=True)
class ControlModifier(Modifier):
    num_ctrl_qubits: int
    ctrl_state: object = None

    def __post_init__(self):
        count = _integer(self.num_ctrl_qubits, "Modifier control width")
        if count < 1: raise ValueError("Modifier control width must be positive.")
        current_gate_limits().check("max_control_qubits", count)
        state = self.ctrl_state
        if isinstance(state, str):
            if len(state) != count or set(state)-{"0", "1"}: raise ValueError("Invalid modifier control bitstring.")
            state = int(state, 2)
        state = (1<<count)-1 if state is None else _integer(state, "Modifier control state")
        if not 0 <= state < 1<<count: raise ValueError("Modifier control state is out of range.")
        object.__setattr__(self, "num_ctrl_qubits", count)
        object.__setattr__(self, "ctrl_state", state)


@dataclass(frozen=True)
class PowerModifier(Modifier):
    power: object

    def __post_init__(self):
        if isinstance(self.power, bool): raise TypeError("A modifier power cannot be a boolean.")
        _coerce_expression(self.power)
        if not parameter_set((self.power,)):
            if not isinstance(resolve_parameter(self.power, {}), Real): raise ValueError("A modifier power must be real.")


def _inspect(operation):
    """Admit a live mutable annotation chain without resolving its operations."""
    limits, chain, seen, controls, work = current_gate_limits(), [], set(), 0, 0
    current = operation
    while type(current) is AnnotatedOperation:
        if id(current) in seen: raise ValueError("Cyclic annotated operation.")
        seen.add(id(current))
        limits.check("max_definition_depth", len(seen))
        if type(current.modifiers) not in (list, tuple): raise TypeError("Modifiers require an explicit native sequence.")
        work += len(current.modifiers)
        limits.check("max_decomposition_work", work)
        for modifier in current.modifiers:
            if type(modifier) is ControlModifier:
                checked = ControlModifier(modifier.num_ctrl_qubits, modifier.ctrl_state)
                controls += checked.num_ctrl_qubits
                limits.check("max_control_qubits", controls)
            elif type(modifier) is PowerModifier: PowerModifier(modifier.power)
            elif type(modifier) is not InverseModifier: raise TypeError("Unknown native modifier type.")
        chain.append(current)
        current = current.base_op
    from .gates import is_native_instruction
    if not is_native_instruction(current):
        raise TypeError("An annotation requires a native Gate, Instruction or annotation base.")
    return current, chain, controls


def _base_gate(base):
    from .circuit import Circuit
    if base.num_bits: raise ValueError("Unitary modifiers cannot act on classical bits.")
    body = base.definition
    if body is None:
        _validate_primitive_arity(base.name, base.params, base.metadata)
        if base.metadata:
            body = Circuit(base.num_qubits).append(base.name, range(base.num_qubits), params=base.params, metadata=base.metadata)
        else:
            from quantumbridge.circuit_library import standard_gate
            body = standard_gate(base.name, *base.params, num_qubits=base.num_qubits)
    validate_definition(body, owner=base)
    from .gates import is_native_gate
    if is_native_gate(base) and base.definition is not None: return base.copy()
    return Gate(base.name, base.num_qubits, base.params, getattr(base, "label", None), definition=body)


def validate_annotated_structure(operation):
    _inspect(operation)
    current_gate_limits().check("max_gate_parameters", len(operation.parameters))


def _annotation_literal_target_width(operation):
    import numpy as np
    from .standard_gates import UnitaryGate
    from .operations import _matrix_owner_family
    from quantumbridge.information.operator import _operator_matrix_origin
    base, chain, controls = _inspect(operation)
    if type(base) is UnitaryGate:
        _operator_matrix_origin(base, from_matrix=True)
        _matrix_owner_family(base, public_unmarked=True)
        if object.__getattribute__(base, "__dict__")["_params"][0].dtype != np.dtype(complex):
            from .classical import UnsupportedFeature
            raise UnsupportedFeature("public_operator_matrix_origin", "Hybrid sources require canonical numeric gate storage.")
    validate_annotated_operation(operation)
    # Restrict this representation to an exact raw source and validated
    # modifier chain. Generic wrappers and source labels are not evidence.
    if type(base) is not UnitaryGate or not controls or not base.num_qubits:return None
    if object.__getattribute__(base, "__dict__").get("_public_matrix_basis") is not None:return None
    return base.num_qubits


def validate_annotated_operation(operation):
    base, chain, _ = _inspect(operation)
    ids = {id(base), *(id(node) for node in chain)}
    previous = _VALIDATION_PATH.get()
    if ids & previous: raise ValueError("Cyclic annotation/Gate definition graph.")
    current_gate_limits().check("max_definition_depth", len(previous)+len(ids))
    token = _VALIDATION_PATH.set(previous | ids)
    try:
        if base.num_bits: raise ValueError("Unitary modifiers cannot act on classical bits.")
        if base.definition is not None: validate_definition(base.definition, owner=base)
        else: _base_gate(base)
        current_gate_limits().check("max_gate_parameters", len(operation.parameters))
    finally: _VALIDATION_PATH.reset(token)


class AnnotatedOperation:
    """An unresolved operation, deliberately neither an Instruction nor a Gate.

    Like the public operation protocol, construction borrows the base and an
    explicit modifier list. Circuit append and copy take owned snapshots.
    """

    def __init__(self, base_op, modifiers):
        self.base_op = base_op
        if any(type(modifiers) is kind for kind in (InverseModifier, ControlModifier, PowerModifier)):
            modifiers = [modifiers]
        elif type(modifiers) not in (list, tuple):
            from quantumbridge.execution.resource_context import current_resources
            if current_resources() is not None: raise TypeError("Managed modifier intake needs a native list or tuple.")
            modifiers = list(islice(iter(modifiers), current_gate_limits().max_decomposition_work+1))
        self.modifiers = modifiers
        validate_annotated_structure(self)

    @property
    def name(self): return "annotated"

    @property
    def num_qubits(self):
        base, _, controls = _inspect(self)
        return base.num_qubits+controls

    @property
    def num_clbits(self): return _inspect(self)[0].num_bits

    @property
    def num_bits(self): return self.num_clbits

    @property
    def params(self):
        _inspect(self)
        return self.base_op.params

    @params.setter
    def params(self, values):
        _inspect(self)
        self.base_op.params = values

    @property
    def parameters(self):
        base, chain, _ = _inspect(self)
        values = [*base.params]
        values.extend(modifier.power for node in chain for modifier in node.modifiers if type(modifier) is PowerModifier)
        return tuple(sorted(parameter_set(values), key=parameter_sort_key))

    def is_parameterized(self): return bool(self.parameters)

    def validate_parameter(self, parameter):
        _inspect(self)
        return self.base_op.validate_parameter(parameter)

    def copy(self):
        validate_annotated_structure(self)
        return _copy(self)

    def assign_parameters(self, mapping, *, strict=True):
        from .circuit import _parameter_mapping_from_values
        from .instruction_parameters import bind_definition
        validate_annotated_structure(self)
        values = _parameter_mapping_from_values(self.parameters, mapping, strict)
        if type(self.base_op) is AnnotatedOperation or isinstance(self.base_op, Gate):
            base = self.base_op.assign_parameters(values, strict=False)
        else:
            source = self.base_op
            metadata = _copy(source.metadata)
            if "global_phase" in metadata:
                metadata["global_phase"] = resolve_parameter(metadata["global_phase"], values, allow_partial=True)
            base = Instruction(source.name, source.num_qubits, source.num_bits,
                tuple(resolve_parameter(value, values, allow_partial=True) for value in source.params),
                source.label, definition=None if source.definition is None else bind_definition(source.definition,values), metadata=metadata)
        modifiers = [PowerModifier(resolve_parameter(item.power, values, allow_partial=True)) if type(item) is PowerModifier else item
                     for item in self.modifiers]
        return AnnotatedOperation(base, modifiers)

    def inverse(self, annotated=True):
        validate_annotated_structure(self)
        return AnnotatedOperation(self.base_op, [*self.modifiers, InverseModifier()])

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None, annotated=None):
        validate_annotated_structure(self)
        return AnnotatedOperation(self.base_op, [*self.modifiers, ControlModifier(num_ctrl_qubits, ctrl_state)])

    def power(self, exponent, annotated=False):
        validate_annotated_structure(self)
        return AnnotatedOperation(self.base_op, [*self.modifiers, PowerModifier(exponent)])

    def to_matrix(self):
        from quantumbridge.information.operator import Operator, _operator_matrix_origin
        target_width = _annotation_literal_target_width(self)
        resolved = resolve_annotated_operation(self)
        basis = "hybrid" if target_width is not None else _operator_matrix_origin(resolved, from_matrix=True)
        operator = Operator(resolved.to_matrix())
        if basis is not None:operator._public_matrix_basis = basis
        if target_width is not None:operator._literal_target_width = target_width
        return operator

    def __eq__(self, other):
        if type(other) is not AnnotatedOperation: return False
        from quantumbridge.ir.qb_ir import _gate_to_ir, _DefinitionCodec
        from quantumbridge.serialization.limits import SerializationLimits
        def canonical(value):
            if type(value) is dict:
                ignored = {"expression", "parameters"} if "parameter_ast" in value else set()
                return tuple((name, canonical(child)) for name, child in sorted(value.items()) if name not in ignored)
            if type(value) in (list, tuple): return tuple(canonical(child) for child in value)
            return value
        def key(value):
            validate_annotated_operation(value)
            base = value.base_op
            content = key(base) if type(base) is AnnotatedOperation else (
                type(base).__name__, canonical(_gate_to_ir(base if isinstance(base, Gate) else _base_gate(base),
                    _DefinitionCodec(SerializationLimits(max_nesting_depth=4*current_gate_limits().max_definition_depth),
                        include_parameter_bindings=False))))
            return content, tuple(value.modifiers)
        return key(self) == key(other)


def resolve_annotated_operation(operation, target=None, limits=None):
    """Resolve a checked ordered chain using existing native Gate operations."""
    with gate_circuit_scope(limits) if limits is not None else nullcontext():
        validate_annotated_operation(operation)
        base, chain, _ = _inspect(operation)
        result = _base_gate(base)
        work = 0
        for node in reversed(chain):
            for modifier in node.modifiers:
                size = validate_definition(result.definition, owner=result) if result.definition is not None else 1
                if type(modifier) is PowerModifier:
                    if parameter_set((modifier.power,)):
                        from .classical import UnsupportedFeature
                        raise UnsupportedFeature("unbound_modifier_power", "Bind a symbolic power before resolution.")
                    power = resolve_parameter(modifier.power, {})
                    prediction = max(1, abs(int(power)))*max(1, size) if isinstance(power, Real) and float(power).is_integer() else size
                    current_gate_limits().check("max_total_operations", prediction)
                else: prediction = size+2*modifier.num_ctrl_qubits+1 if type(modifier) is ControlModifier else size
                work += prediction+1
                current_gate_limits().check("max_decomposition_work", work)
                if type(modifier) is InverseModifier: result = result.inverse()
                elif type(modifier) is ControlModifier: result = result.control(modifier.num_ctrl_qubits, ctrl_state=modifier.ctrl_state)
                elif isinstance(power, Real) and float(power).is_integer():
                    result = result._integer_power(int(power))
                else: result = result.power(power)
        if target is not None:
            from quantumbridge.compiler import transpile
            result = transpile(result.definition, target=target, optimization_level=0).to_gate(_name=result.name)
        return result
