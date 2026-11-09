# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Preserve the classical graph when compilation splits or merges gate angles.

import operator
from numbers import Complex
from dataclasses import dataclass, replace

from quantumbridge.core import Parameter, ParameterExpression, ParameterVectorElement
from quantumbridge.core.parameters import (
    UNARY_FUNCTIONS, _admit_parameter_values, _coerce_expression, _evaluate_unary,
    _with_derivative_order, _with_parameter_context,
)
from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.compiler.traversal import map_control_bodies


def evaluate_compiled_parameter(value, values):
    """Evaluate native scalar syntax without copying a Circuit or recording it."""
    if isinstance(value, Parameter):
        if value not in values:
            raise UnsupportedFeature("compiled_loop_parameter", "Compiled QNode gradients require bound loop parameters.")
        return values[value]
    if not isinstance(value, ParameterExpression):
        return value
    if value.op == "const":
        result = value.args[0]
    elif value.op == "parameter":
        result = evaluate_compiled_parameter(value.args[0], values)
    elif value.op == "neg":
        result = -evaluate_compiled_parameter(value.args[0], values)
    elif value.op in UNARY_FUNCTIONS | {"abs_gradient", "sign_gradient"}:
        result = _unary(value.op, evaluate_compiled_parameter(value.args[0], values))
    else:
        binary = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv, "**": operator.pow}
        result = binary[value.op](*(evaluate_compiled_parameter(arg, values) for arg in value.args))
    # Symbolic capture transports formal identities; numeric/AD evaluation
    # never adds artificial mathematical dependencies on those inputs.
    retained = set()
    for parameter in value._retained_parameters:
        replacement = evaluate_compiled_parameter(parameter, values)
        if isinstance(replacement, (Parameter, ParameterExpression)):
            retained.update(replacement.parameters)
    if retained:
        result = _with_parameter_context(_coerce_expression(result), retained)
    if isinstance(result, ParameterExpression):
        result = _with_derivative_order(result, value._derivative_order)
    return result


@dataclass(frozen=True)
class FrozenParameterBinding:
    """Immutable native scalar occurrences only; no Circuit or operation store."""

    parameters: tuple
    expressions: tuple
    node_count: int

    @classmethod
    def create(cls, parameters, expressions, *, limits=None):
        from numbers import Real
        import math
        from quantumbridge.core.parameters import validate_parameter_names
        from quantumbridge.core.symbolic_limits import current_symbolic_limits
        from .higher_order_limits import resolve_higher_order_limits
        higher, symbolic = resolve_higher_order_limits(limits), current_symbolic_limits()
        if type(parameters) not in {tuple, list} or type(expressions) not in {tuple, list}:
            raise TypeError("Frozen binding requires bounded parameter and expression sequences.")
        _admit_parameter_values(parameters, expressions)
        parameters = tuple(parameters)
        higher.check("max_input_parameters", len(parameters))
        if not all(type(p) in (Parameter, ParameterVectorElement) for p in parameters) or len(set(parameters)) != len(parameters):
            raise ValueError("Frozen binding requires unique native parameter identities.")
        validate_parameter_names(parameters)
        count, active, used, retained = 0, set(), set(), set()
        smooth = {"sin", "cos", "tan", "exp", "log", "arctan", "conjugate"}
        def freeze(value, depth=0):
            nonlocal count
            count += 1
            higher.check("max_parameter_graph_nodes", count)
            symbolic.check("max_depth", depth+1)
            if isinstance(value, Parameter):
                used.add(value)
                return value
            if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value):
                return value
            if type(value) is not ParameterExpression or type(value.args) is not tuple or id(value) in active:
                raise ValueError("Malformed native parameter expression in frozen binding.")
            if value.op not in smooth | {"const", "parameter", "neg", "+", "-", "*", "/", "**"}:
                raise UnsupportedFeature("frozen_parameter_expression", "The frozen scalar mapping contains an unvalidated operation.")
            active.add(id(value))
            children = tuple(freeze(arg, depth+1) for arg in value.args)
            active.remove(id(value))
            result = ParameterExpression(value.op, children,
                _retained_parameters=value._retained_parameters)
            order = value._derivative_order
            if type(order) is not int or order < 0:
                raise ValueError("Frozen scalar derivative order must be a nonnegative integer.")
            symbolic.check("max_derivative_order",order)
            result = _with_derivative_order(result, order)
            retained.update(result._retained_parameters)
            if value.op == "**":
                exponent = result.args[1]
                if exponent.parameters:
                    raise UnsupportedFeature("frozen_parameter_power", "Frozen scalar powers require a fixed real exponent.")
                symbolic.check("max_power_exponent", abs(exponent.numeric()))
            return result
        frozen = tuple(freeze(value) for value in expressions)
        # Formal-only identities occupy mapping storage, not mathematical nodes.
        count += len(retained-used)
        higher.check("max_parameter_graph_nodes", count)
        if used | retained != set(parameters):
            raise ValueError("Frozen binding has missing or surplus native parameter identities.")
        return cls(parameters, frozen, count)

    @property
    def parameter_uuids(self):
        return tuple(str(p.uuid) for p in self.parameters)

    def evaluate(self, values):
        if type(values) not in {tuple, list} or len(values) != len(self.parameters):
            raise ValueError("Frozen binding input count must match its native UUID order.")
        mapping = dict(zip(self.parameters, values))
        return tuple(evaluate_compiled_parameter(value, mapping) for value in self.expressions)

    def to_record(self):
        from quantumbridge.ir.qb_ir import _param_to_ir
        return dict(parameters=[_param_to_ir(p) for p in self.parameters],
            expressions=[_param_to_ir(value) for value in self.expressions])


def record_compiled_parameters(circuit, original_recorder):
    from .qnode_engine import AngleRecorder
    recorder = AngleRecorder()
    values = dict(zip(original_recorder.parameters, original_recorder.angles))
    def evaluate(value):
        return evaluate_compiled_parameter(value, values)
    def angle(value):
        return recorder(evaluate(value)) if isinstance(value, (Parameter, ParameterExpression)) else value
    from .symbolic_power import lower_recorded_powers
    circuit = lower_recorded_powers(circuit, evaluate, values)
    def walk(body):
        from quantumbridge.core.circuit_metadata import execution_copy
        out = execution_copy(body)
        phase = angle(body.global_phase)
        out.operations = [map_control_bodies(op, walk) if op.metadata.get("control_node") is not None
                          else replace(op, params=tuple(angle(value) for value in op.params)) for op in body.operations]
        # Replace old gate identities before validating the new phase namespace.
        out.global_phase = phase
        return out
    return walk(circuit), recorder


def _unary(name, value):
    if isinstance(value, (Parameter, ParameterExpression)):
        if name in {"abs_gradient", "sign_gradient"}:
            raise UnsupportedFeature("symbolic_nonsmooth_interface")
        return getattr(value, name)()
    if isinstance(value, Complex):
        return _evaluate_unary(name, value)
    from .trace import Dual
    if isinstance(value, Dual):
        if name == "abs_gradient":
            return value.sign()
        if name == "sign_gradient":
            return value.sign()*0
        return abs(value) if name == "abs" else getattr(value, name)()
    module = type(value).__module__
    if module.startswith("torch"):
        import torch as numeric
    elif module.startswith("autograd"):
        import autograd.numpy as numeric
    elif module.startswith(("jax", "jaxlib")):
        import jax.numpy as numeric
    elif module.startswith("tensorflow"):
        import tensorflow.math as numeric
    else:
        raise UnsupportedFeature("symbolic_scalar_interface", "Symbolic function needs a supported scalar/trace interface.")
    if name in {"abs_gradient", "sign_gradient"}:
        raise UnsupportedFeature("symbolic_nonsmooth_interface", "Guarded nonsmooth symbolic derivatives require the native real trace.")
    if name in {"abs", "sign", "arcsin", "arccos"}:
        from quantumbridge.interfaces.symbolic_math import guarded_symbolic_unary
        return guarded_symbolic_unary(value, name, numeric, module)
    alias = {"conjugate": "conj", "arcsin": "asin", "arccos": "acos", "arctan": "atan"}
    if module.startswith(("autograd", "jax", "jaxlib")):
        alias = {"conjugate": "conj"}
    return getattr(numeric, alias.get(name, name))(value)
