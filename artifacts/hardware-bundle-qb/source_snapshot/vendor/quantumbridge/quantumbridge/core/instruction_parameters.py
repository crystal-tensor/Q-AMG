# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Live checked parameter views and owned, repeatable definition bindings."""

from collections.abc import MutableSequence
from dataclasses import dataclass
from itertools import chain, islice
import operator

import numpy as np

from quantumbridge.execution.array_ownership import _exact_type

from .gate_limits import current_gate_limits
from .parameters import Parameter, ParameterExpression, ParameterVectorElement, parameter_set, validate_parameter_names


def _managed():
    from quantumbridge.execution.resource_context import current_resources
    return current_resources() is not None


def admit_owner(owner):
    if not _managed(): return
    from .operations import Instruction
    from .gates import is_native_instruction
    if not is_native_instruction(owner):
        from .classical import UnsupportedFeature
        raise UnsupportedFeature("managed_parameter_owner")


def admit_payloads(values):
    if not _managed(): return
    from .classical import UnsupportedFeature
    from quantumbridge.execution.array_ownership import _defined_type
    pending, seen, work = [(values, 0)], set(), 0
    while pending:
        value, depth = pending.pop()
        if id(value) in seen: continue
        seen.add(id(value))
        work += 1
        current_gate_limits().check("max_decomposition_work", work)
        current_gate_limits().check("max_definition_depth", depth)
        kind = type(value)
        if _exact_type(kind, (int, float, complex, bool, str, bytes, type(None), Parameter, ParameterVectorElement, ParameterExpression)): continue
        if _exact_type(kind, (list, tuple, dict)):
            count = len(value)*(2 if kind is dict else 1)
            current_gate_limits().check("max_gate_parameters", count)
            current_gate_limits().check("max_decomposition_work", work+len(pending)+count)
            children = chain(value.keys(), value.values()) if kind is dict else value
            pending.extend((child, depth+1) for child in children)
        elif kind is np.ndarray:
            if value.dtype.hasobject: raise UnsupportedFeature("managed_parameter_object_array")
        elif _defined_type(kind, "numpy") and np.generic in type.__dict__["__mro__"].__get__(kind):
            if value.dtype.hasobject: raise UnsupportedFeature("managed_parameter_object_scalar")
        else: raise UnsupportedFeature("managed_parameter_payload", "Foreign parameter callbacks are not admitted in managed updates.")


def parameter_values(values, *, capacity=None):
    capacity = current_gate_limits().max_gate_parameters if capacity is None else capacity
    if type(values) is ParameterList: return values._snapshot(capacity)
    if _managed() and not _exact_type(type(values), (tuple, list, np.ndarray)):
        from .classical import UnsupportedFeature
        raise UnsupportedFeature("managed_gate_parameter_container", "Managed parameters require explicit native containers.")
    if isinstance(values, (tuple, list, np.ndarray)):
        current_gate_limits().check("max_gate_parameters", len(values))
        if len(values) > capacity: raise MemoryError("Parameter update exceeds max_gate_parameters.")
        return tuple(values)
    result = tuple(islice(iter(values), capacity+1))
    if len(result) > capacity: raise MemoryError("Parameter update exceeds max_gate_parameters.")
    return result


class ParameterList(MutableSequence):
    """A live owner-backed view; every mutator validates and commits once.

    Canonical storage stays an immutable tuple. Views are not retained by
    their owner, avoiding a Gate/view cycle and its delayed storage release.
    """

    def __init__(self, owner): self._owner = owner

    def _snapshot(self, capacity=None):
        admit_owner(self._owner)
        values = self._owner._params
        maximum = current_gate_limits().max_gate_parameters if capacity is None else capacity
        if len(values) > maximum: raise MemoryError("Parameter update exceeds max_gate_parameters.")
        return values

    def __len__(self): return len(self._snapshot())
    def __iter__(self): return iter(self._snapshot())
    def __repr__(self): return repr(list(self._snapshot()))
    def __getitem__(self, index):
        result = self._snapshot()[self._index(index)]
        return list(result) if isinstance(index, slice) else result

    def _index(self, value):
        if _managed() and not _exact_type(type(value), (int, bool, slice)):
            from .classical import UnsupportedFeature
            raise UnsupportedFeature("managed_parameter_index")
        if isinstance(value, slice):
            if _managed() and any(not _exact_type(type(x), (int, bool, type(None))) for x in (value.start, value.stop, value.step)):
                from .classical import UnsupportedFeature
                raise UnsupportedFeature("managed_parameter_index")
            return value
        return operator.index(value)

    def __setitem__(self, index, value):
        previous = self._snapshot()
        index = self._index(index)
        if isinstance(index, slice):
            replaced = len(range(*index.indices(len(previous))))
            value = parameter_values(value, capacity=current_gate_limits().max_gate_parameters-len(previous)+replaced)
        result = list(previous)
        result[index] = value
        self._owner.params = result

    def __delitem__(self, index):
        result = list(self._snapshot())
        del result[self._index(index)]
        self._owner.params = result

    def insert(self, index, value):
        previous = self._snapshot()
        current_gate_limits().check("max_gate_parameters", len(previous)+1)
        result = list(previous)
        result.insert(self._index(index), value)
        self._owner.params = result

    def extend(self, values):
        previous = self._snapshot()
        values = parameter_values(values, capacity=current_gate_limits().max_gate_parameters-len(previous))
        self._owner.params = previous+values

    def clear(self): self._owner.params = ()
    def reverse(self): self._owner.params = self._snapshot()[::-1]
    def sort(self, *, key=None, reverse=False):
        if _managed() and (key is not None or not _exact_type(type(reverse), (bool, int))):
            from .classical import UnsupportedFeature
            raise UnsupportedFeature("managed_parameter_sort_callback")
        self._owner.params = sorted(self._snapshot(), key=key, reverse=reverse)

    def pop(self, index=-1):
        result = list(self._snapshot())
        value = result.pop(self._index(index))
        self._owner.params = result
        return value

    def remove(self, value):
        admit_payloads((value,))
        result = list(self._snapshot())
        result.remove(value)
        self._owner.params = result

    def copy(self): return list(self._snapshot())
    def __add__(self, other):
        if not _exact_type(type(other), (list, ParameterList)): return NotImplemented
        previous = self._snapshot()
        values = parameter_values(other, capacity=current_gate_limits().max_gate_parameters-len(previous))
        return list(previous+values)

    def __radd__(self, other):
        if type(other) is not list: return NotImplemented
        previous = self._snapshot()
        values = parameter_values(other, capacity=current_gate_limits().max_gate_parameters-len(previous))
        return list(values+previous)

    def __mul__(self, count):
        count = self._index(count)
        previous = self._snapshot()
        current_gate_limits().check("max_gate_parameters", len(previous)*max(0, count))
        return list(previous)*count

    __rmul__ = __mul__
    def __iadd__(self, values): self.extend(values); return self
    def __imul__(self, count):
        count = self._index(count)
        previous = self._snapshot()
        current_gate_limits().check("max_gate_parameters", len(previous)*max(0, count))
        self._owner.params = previous*count
        return self

    def __eq__(self, other):
        if type(other) is ParameterList: return self._snapshot() == other._snapshot()
        if _exact_type(type(other), (tuple, list)): return self._snapshot() == tuple(other)
        return NotImplemented


def owned_copy(value):
    from quantumbridge.execution.array_ownership import native_configuration_copy
    return native_configuration_copy(value, category="native_parameter_definition_copy", strict=_managed())


def definition_signature(definition):
    from quantumbridge.serialization.limits import SerializationLimits
    from quantumbridge.ir.qb_ir import _DefinitionCodec
    limits = current_gate_limits()
    context = _DefinitionCodec(SerializationLimits(
        max_nesting_depth=4*limits.max_definition_depth,
        max_custom_definitions=limits.max_custom_definitions,
        max_instructions=limits.max_total_operations,
        max_gate_expansion=limits.max_decomposition_work), include_parameter_bindings=False, canonical_parameters=True, include_user_metadata=False)
    return definition.to_ir().to_dict(_context=context)


@dataclass(frozen=True)
class ParameterBindingPlan:
    definition: object
    formals: tuple
    signature: dict


def same_values(left, right):
    if len(left) != len(right): return False
    native = (int, float, complex, bool, str, type(None), Parameter, ParameterExpression)
    return all(a is b or _exact_type(type(a), native) and _exact_type(type(b), native) and a == b for a, b in zip(left, right))


def bind_definition(definition, mapping):
    """Bind an internal definition without public output naming side effects."""
    from .circuit import Circuit,is_native_circuit
    if not is_native_circuit(definition):raise TypeError("An executable definition must be a native Circuit.")
    return Circuit.assign_parameters(definition,mapping,strict=False,_copy_user_metadata=False)


def prepare_definition_update(owner, values):
    """Build the entire next definition and binding plan before owner mutation."""
    definition = owner.definition
    if definition is None: return None, None
    owner._validate_definition(definition, owner._params)
    previous = owner._params
    if len(previous) != len(values): raise ValueError("Defined operation parameter updates require matching arity.")
    plan = getattr(owner, "_parameter_plan", None)
    if same_values(previous, values): return definition, plan
    if plan is not None and definition_signature(definition) != plan.signature:
        # A fresh symbolic definition can be rebound safely. Never infer a
        # parameter from a matching numeric constant in a user-edited body.
        plan = None
    if plan is None:
        if not all(isinstance(value, Parameter) for value in previous):
            raise ValueError("Changed bound definitions need explicit symbolic formals before parameter rebinding.")
        template, formals = owned_copy(definition), previous
    else: template, formals = plan.definition, plan.formals
    mapping = {}
    for parameter, value in zip(formals, values):
        if parameter in mapping and not same_values((mapping[parameter],), (value,)):
            raise ValueError("Repeated formal parameters cannot receive conflicting values.")
        mapping[parameter] = value
    result = bind_definition(template,mapping)
    owner._validate_definition(result, values)
    return result, ParameterBindingPlan(template, formals, definition_signature(result))


def validated_values(owner, values):
    admit_owner(owner)
    values = parameter_values(values)
    admit_payloads(values)
    result = tuple(owner.validate_parameter(value) for value in values)
    known = [value for value in result if _exact_type(type(value), (Parameter, ParameterVectorElement, ParameterExpression))]
    validate_parameter_names(parameter_set(known))
    return result


def inherit_binding_plan(source, result):
    """Parameter assignment preserves the caller's declared formal lineage."""
    if source.definition is None or not source._params: return
    plan = getattr(source, "_parameter_plan", None)
    if plan is not None and definition_signature(source.definition) == plan.signature:
        template, formals = plan.definition, plan.formals
    elif all(isinstance(value, Parameter) for value in source._params):
        template, formals = source.definition, source._params
    else: return
    object.__setattr__(result, "_parameter_plan", ParameterBindingPlan(
        owned_copy(template), formals, definition_signature(result.definition)))


def inherit_control_binding_plan(base, result):
    if not base._params or all(isinstance(value, Parameter) for value in base._params): return
    from .gates import Gate
    plan = getattr(base, "_parameter_plan", None)
    if plan is not None and definition_signature(base.definition) == plan.signature:
        formals = plan.formals
        symbolic = Gate(base.name, base.num_qubits, formals, definition=plan.definition, metadata=base.metadata)
    elif base.definition is None:
        formals = tuple(Parameter("_qb_slot_"+str(index)) for index in range(len(base._params)))
        symbolic = Gate(base.name, base.num_qubits, formals, metadata=base.metadata)
    else: return
    template = symbolic.control(result.num_ctrl_qubits, ctrl_state=result.ctrl_state).definition
    definition = bind_definition(template,dict(zip(formals, result._params)))
    result._validate_definition(definition, result._params)
    object.__setattr__(result, "_definition", definition)
    object.__setattr__(result, "_parameter_plan", ParameterBindingPlan(
        owned_copy(template), formals, definition_signature(result.definition)))


def binding_plan_for_serialization(owner, *, preserve_invalidated=False):
    plan = getattr(owner, "_parameter_plan", None)
    if plan is None: return None
    if definition_signature(owner.definition) == plan.signature: return plan
    if all(isinstance(value, Parameter) for value in owner._params):
        return ParameterBindingPlan(owner.definition, owner._params, definition_signature(owner.definition))
    if preserve_invalidated:return plan
    raise ValueError("A changed bound definition cannot silently retain stale parameter binding lineage.")


def restore_binding_plan(owner, template, formals, *, invalidated=False):
    if len(formals) != len(owner._params) or any(type(value) not in (Parameter, ParameterVectorElement) for value in formals):
        raise ValueError("Malformed native parameter binding formals.")
    owner._validate_definition(template, formals)
    mapping = {}
    for parameter, value in zip(formals, owner._params):
        if parameter in mapping and not same_values((mapping[parameter],), (value,)):
            raise ValueError("Conflicting repeated binding formals.")
        mapping[parameter] = value
    expected = bind_definition(template,mapping)
    signature = definition_signature(owner.definition)
    expected_signature=definition_signature(expected)
    if invalidated:
        if parameter_set(owner._params) or expected_signature==signature:
            raise ValueError("An invalidated binding record must retain a genuinely stale nonsymbolic history.")
        signature=expected_signature
    elif expected_signature != signature:
        raise ValueError("Parameter binding template does not describe the actual executable definition.")
    object.__setattr__(owner, "_parameter_plan", ParameterBindingPlan(template, tuple(formals), signature))
