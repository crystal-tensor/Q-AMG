# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.
# Design source: docs/mvp/api_contract_v0.1.md and docs/mvp/mvp_scope_v0.1.md.

from __future__ import annotations

from dataclasses import dataclass, field
import cmath
import math
from numbers import Complex, Integral, Real
from typing import Any, Iterator, Mapping, Sequence, Union
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from .symbolic_limits import current_symbolic_limits, charge_symbolic_work


UNARY_FUNCTIONS = frozenset({"sin", "cos", "tan", "exp", "log", "arcsin", "arccos",
                             "arctan", "abs", "sign", "conjugate"})
_GUARDED_DERIVATIVES = frozenset({"abs_gradient", "sign_gradient"})
EXPRESSION_OPS = UNARY_FUNCTIONS | _GUARDED_DERIVATIVES | {"const", "parameter", "neg", "+", "-", "*", "/", "**"}


def _admit_parameter_values(*values):
    from .composition import _admit_graph
    _admit_graph(tuple(value.hex if type(value) is UUID else value for value in values))


def _check_vector_size(size):
    from .gate_limits import current_gate_limits
    current_gate_limits().check("max_gate_parameters",size)
    current_symbolic_limits().check("max_parameters",size)


class _SymbolicMethods:
    def sin(self): return self.expression()._unary("sin")
    def cos(self): return self.expression()._unary("cos")
    def tan(self): return self.expression()._unary("tan")
    def exp(self): return self.expression()._unary("exp")
    def log(self): return self.expression()._unary("log")
    def arcsin(self): return self.expression()._unary("arcsin")
    def arccos(self): return self.expression()._unary("arccos")
    def arctan(self): return self.expression()._unary("arctan")
    def abs(self): return self.expression()._unary("abs")
    def sign(self): return self.expression()._unary("sign")
    def conjugate(self): return self.expression()._unary("conjugate")
    def __abs__(self): return self.abs()
    def gradient(self, parameter): return self.expression().gradient(parameter)
    def assign(self, parameter, value): return self.expression().assign(parameter, value)
    def subs(self, parameter_map, allow_unknown_parameters=False):
        return self.expression().subs(parameter_map, allow_unknown_parameters)
    def numeric(self, strict=True): return self.expression().numeric(strict)
    def is_real(self): return self.expression().is_real()
    def is_symbol(self): return self.expression().is_symbol()
    def sympify(self): return self.expression().sympify()
    def bind_all(self, values): return self.expression().bind_all(values)

    @property
    def num_parameters(self): return len(self.expression().parameters)


@dataclass(frozen=True)
class Parameter(_SymbolicMethods):
    """Named scalar parameter used by QuantumBridge circuits."""

    name: str
    uuid: str | None = field(default=None, repr=False)
    _uuid_explicit: bool = field(default=False, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        from quantumbridge.execution.resource_context import current_resources
        if current_resources() is not None and not any(type(self) is kind for kind in (Parameter,ParameterVectorElement)):
            from .classical import UnsupportedFeature
            raise UnsupportedFeature("managed_parameter_type")
        _admit_parameter_values(self.name,self.uuid)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("QuantumBridge parameter names must be non-empty strings.")
        explicit=self.uuid is not None
        if explicit and not isinstance(self.uuid,(str,UUID)):
            raise TypeError("Parameter uuid must be a UUID or UUID string.")
        object.__setattr__(self,"uuid",str(UUID(str(self.uuid))) if explicit else str(uuid4()))
        object.__setattr__(self,"_uuid_explicit",explicit)

    def __repr__(self) -> str:
        return f"Parameter({self.name!r})"

    def __str__(self) -> str:
        return self.name

    def expression(self) -> "ParameterExpression":
        return ParameterExpression("parameter", (self,))

    @property
    def parameters(self): return frozenset({self})

    def bind(self, mapping, allow_partial=True, *, allow_unknown_parameters=False):
        return self.expression().bind(mapping, allow_partial=allow_partial,
                                      allow_unknown_parameters=allow_unknown_parameters)

    def __neg__(self) -> "ParameterExpression":
        return -self.expression()

    def __add__(self, other: Any) -> "ParameterExpression":
        return self.expression() + other

    def __radd__(self, other: Any) -> "ParameterExpression":
        return _coerce_expression(other) + self.expression()

    def __sub__(self, other: Any) -> "ParameterExpression":
        return self.expression() - other

    def __rsub__(self, other: Any) -> "ParameterExpression":
        return _coerce_expression(other) - self.expression()

    def __mul__(self, other: Any) -> "ParameterExpression":
        return self.expression() * other

    def __rmul__(self, other: Any) -> "ParameterExpression":
        return _coerce_expression(other) * self.expression()

    def __truediv__(self, other: Any) -> "ParameterExpression":
        return self.expression() / other

    def __rtruediv__(self, other: Any) -> "ParameterExpression":
        return _coerce_expression(other) / self.expression()

    def __pow__(self, other: Any) -> "ParameterExpression":
        return self.expression() ** other

    def __rpow__(self, other: Any) -> "ParameterExpression":
        return _coerce_expression(other) ** self.expression()


@dataclass(frozen=True)
class ParameterExpression(_SymbolicMethods):
    """Immutable, bounded native scalar AST with UUID-preserving calculus."""

    op: str
    args: tuple[Any, ...]
    _nodes: int = field(init=False, compare=False, repr=False)
    _depth: int = field(init=False, compare=False, repr=False)
    _parameters: frozenset = field(init=False, compare=False, repr=False)
    _dependencies: frozenset = field(init=False, compare=False, repr=False)
    _retained_parameters: frozenset = field(default_factory=frozenset, compare=False, repr=False, kw_only=True)
    _derivative_order: int = field(default=0, init=False, compare=False, repr=False)

    def __post_init__(self):
        charge_symbolic_work()
        if self.op not in EXPRESSION_OPS or type(self.args) is not tuple:
            raise ValueError("Unknown executable native parameter AST.")
        arity = 1 if self.op in UNARY_FUNCTIONS | _GUARDED_DERIVATIVES | {"const", "parameter", "neg"} else 2
        if len(self.args) != arity:
            raise ValueError("Incorrect native parameter AST arity.")
        limits = current_symbolic_limits()
        if type(self._retained_parameters) is not frozenset:
            raise TypeError("Retained parameter context must be an owned frozenset.")
        _check_vector_size(len(self._retained_parameters))
        if any(type(p) not in (Parameter,ParameterVectorElement) for p in self._retained_parameters):
            raise TypeError("Retained parameter context requires exact native identities.")
        _admit_parameter_values(self._retained_parameters)
        charge_symbolic_work(len(self._retained_parameters))
        if self.op == "const":
            object.__setattr__(self, "args", (_checked_numeric(self.args[0]),))
            nodes, depth, parameters, order = 1, 1, frozenset(), 0
        elif self.op == "parameter":
            if not isinstance(self.args[0], Parameter):
                raise TypeError("A parameter AST leaf requires a native Parameter.")
            nodes, depth, parameters, order = 1, 1, frozenset(self.args), 0
        else:
            if not all(isinstance(arg, ParameterExpression) for arg in self.args):
                raise TypeError("Native parameter AST children must be expressions.")
            nodes = 1 + sum(arg._nodes for arg in self.args)
            depth = 1 + max(arg._depth for arg in self.args)
            limits.check("max_nodes", nodes)
            limits.check("max_depth", depth)
            parameters = frozenset().union(*(arg.parameters for arg in self.args))
            order = max(arg._derivative_order for arg in self.args)
        dependencies = (frozenset() if self.op == "const" else frozenset(self.args) if self.op == "parameter"
                        else frozenset().union(*(arg._dependencies for arg in self.args)))
        parameters = parameters | self._retained_parameters
        limits.check("max_parameters", len(parameters))
        validate_parameter_names(parameters)
        if self._retained_parameters:
            identities = {}
            for parameter in parameters:
                descriptor = (type(parameter),parameter.name)
                if type(parameter) is ParameterVectorElement:
                    owner = parameter.vector
                    if type(owner) is not ParameterVector: raise TypeError("Retained vector context requires a native owner.")
                    descriptor += (owner._root_uuid,len(owner),parameter.index)
                previous = identities.setdefault(parameter.uuid,descriptor)
                if previous != descriptor: raise ValueError("Conflicting retained parameter UUID descriptors.")
        object.__setattr__(self, "_nodes", nodes)
        object.__setattr__(self, "_depth", depth)
        object.__setattr__(self, "_parameters", parameters)
        object.__setattr__(self, "_dependencies", dependencies)
        object.__setattr__(self, "_derivative_order", order)

    def expression(self): return self

    @property
    def parameters(self) -> frozenset[Parameter]:
        return self._parameters

    @property
    def name(self) -> str:
        if self.op == "parameter":
            return self.args[0].name
        return str(self)

    def bind(self, mapping, allow_partial=True, *, allow_unknown_parameters=False):
        mapping = self._mapping(mapping, allow_unknown_parameters, numeric_only=True)
        result = self._replace(mapping)
        if result.parameters:
            if not allow_partial:
                try: return result.numeric(strict=False)
                except TypeError:
                    raise ValueError("QuantumBridge parameters are not fully bound: " + ", ".join(sorted(p.name for p in result.parameters))) from None
            return result
        return result.numeric()

    def bind_all(self, values):
        return self.bind(values, allow_partial=False, allow_unknown_parameters=True)

    def assign(self, parameter, value):
        mapping = self._mapping({parameter: value}, False)
        return self._replace(mapping)

    def subs(self, parameter_map, allow_unknown_parameters=False):
        mapping = self._mapping(parameter_map, allow_unknown_parameters)
        if any(not isinstance(value, (Parameter, ParameterExpression)) for value in mapping.values()):
            raise TypeError("Symbolic subs replacements must be native parameters or expressions; use assign for numbers.")
        return self._replace(mapping)

    def _mapping(self, mapping, allow_unknown, numeric_only=False):
        if not isinstance(mapping, Mapping):
            raise TypeError("Parameter bindings must be a mapping.")
        current_symbolic_limits().check("max_substitutions", len(mapping))
        names = {parameter.name: parameter for parameter in self.parameters}
        result = {}
        for key, value in mapping.items():
            key = names.get(key, key) if isinstance(key, str) else key
            if not isinstance(key, (str, Parameter)):
                raise TypeError("Parameter keys must be native parameters or names.")
            if key not in self.parameters:
                if allow_unknown: continue
                raise ValueError("Unknown parameter binding: " + str(key))
            if key in result:
                raise ValueError("A parameter was bound more than once.")
            if numeric_only:
                value = _checked_numeric(value)
            else:
                _coerce_expression(value)
            result[key] = value
        return result

    def _replace(self, mapping):
        work, memo = _SymbolicWork(), {}
        work.validate(self)
        def visit(node):
            work.charge()
            if id(node) in memo: return memo[id(node)]
            if node.op == "const": result = ParameterExpression("const",node.args) if node._retained_parameters else node
            elif node.op == "parameter":
                # Never revisit a replacement: batch substitution is simultaneous.
                result = _coerce_expression(mapping.get(node.args[0], node.args[0]))
            else:
                children = tuple(visit(arg) for arg in node.args)
                result = ParameterExpression(node.op, children).simplify()
            memo[id(node)] = result
            return result
        parameters = set()
        work.charge(len(self.parameters))
        for parameter in self.parameters:
            parameters.update(_coerce_expression(mapping.get(parameter,parameter)).parameters)
        return _with_derivative_order(_with_parameter_context(visit(self),frozenset(parameters)), self._derivative_order)

    def gradient(self, parameter):
        result = self._gradient_expression(parameter)
        value = _constant_value(result, allow_cancellation=True)
        if value is _NOT_CONSTANT:
            return result
        value = _checked_numeric(value)
        if isinstance(value, Integral):
            return float(value) if -1 <= value <= 1 else int(value)
        if isinstance(value, Real):
            return float(value)
        return float(value.real) if value.imag == 0 else complex(value)

    def _gradient_expression(self, parameter):
        """Same calculus with an AST result for internal repeated derivatives."""
        if not isinstance(parameter, Parameter):
            raise TypeError("Symbolic differentiation requires a native Parameter identity.")
        order = self._derivative_order + 1
        current_symbolic_limits().check("max_derivative_order", order)
        _constant_value(self, allow_cancellation=True)
        result = _differentiate(self, parameter)
        if result._dependencies:
            result = _with_parameter_context(result,self.parameters)
        return _with_derivative_order(result, order)

    def numeric(self, strict=True):
        if type(strict) is not bool: raise TypeError("numeric strict must be boolean.")
        if self.parameters and strict:
            raise TypeError("Unbound parameter expressions are not numeric.")
        result = _constant_value(self, allow_cancellation=not strict)
        if result is _NOT_CONSTANT:
            raise TypeError("Unbound parameter expressions are not numeric.")
        return _checked_numeric(result)

    def is_real(self):
        try: value = self.numeric(strict=False)
        except TypeError: return None
        return isinstance(value, Real) or value.imag == 0

    def is_symbol(self):
        return self.simplify().op == "parameter"

    def sympify(self):
        import sympy
        functions = {name: getattr(sympy, {"arcsin": "asin", "arccos": "acos", "arctan": "atan", "abs": "Abs"}.get(name, name)) for name in UNARY_FUNCTIONS}
        work = _SymbolicWork()
        work.validate(self)
        def visit(node):
            work.charge()
            if node.op == "const": return sympy.sympify(node.args[0])
            if node.op == "parameter": return sympy.Symbol(node.args[0].name)
            args = tuple(visit(arg) for arg in node.args)
            if node.op == "neg": return -args[0]
            if node.op in functions: return functions[node.op](*args)
            if node.op in _GUARDED_DERIVATIVES:
                raise ValueError("Guarded nonsmooth derivatives have no lossless SymPy export.")
            return _evaluate_binary(node.op, *args, check=False)
        return visit(self)

    def evaluate(self, mapping: Mapping[Union[Parameter, str], Complex]) -> Complex:
        return self.bind(mapping, allow_partial=False)

    def simplify(self):
        return _with_derivative_order(_with_parameter_context(self._simplify(),self.parameters), self._derivative_order)

    def _simplify(self):
        if self.op in {"const", "parameter"}:
            return self
        if self.op == "neg":
            value = self.args[0]
            if value.op == "neg": return value.args[0]
            if value.op == "const":
                return ParameterExpression("const", (-value.args[0],))
            return self
        if self.op in UNARY_FUNCTIONS | _GUARDED_DERIVATIVES:
            if self.args[0].op == "const":
                return _coerce_expression(_evaluate_unary(self.op, self.args[0].args[0]))
            return self
        left, right = self.args
        if isinstance(left, ParameterExpression) and isinstance(right, ParameterExpression):
            if left.op == "const" and right.op == "const":
                return ParameterExpression("const", (_evaluate_binary(self.op, left.args[0], right.args[0]),))
        if self.op in {"+", "-"} and _literal(right, 0): return left
        if self.op == "+" and _literal(left, 0): return right
        if self.op in {"*", "/", "**"} and _literal(right, 1): return left
        if self.op == "*" and _literal(left, 1): return right
        if self.op == "*" and ((_literal(left,0) and right.op == "parameter")
                               or (_literal(right,0) and left.op == "parameter")):
            return ParameterExpression("const",(0,))
        return self

    def _unary(self, name):
        return ParameterExpression(name, (self,)).simplify()

    def __repr__(self) -> str:
        return f"ParameterExpression({str(self)!r})"

    def __str__(self) -> str:
        if self.op == "const":
            return repr(self.args[0])
        if self.op == "parameter":
            return self.args[0].name
        if self.op == "neg":
            return f"-({self.args[0]})"
        if self.op in UNARY_FUNCTIONS | _GUARDED_DERIVATIVES:
            return f"{self.op}({self.args[0]})"
        symbol = {"+": "+", "-": "-", "*": "*", "/": "/", "**": "**"}[self.op]
        return f"({self.args[0]} {symbol} {self.args[1]})"

    def __float__(self) -> float:
        return float(self.numeric(strict=False))

    def __complex__(self): return complex(self.numeric(strict=False))

    def __neg__(self) -> "ParameterExpression":
        return ParameterExpression("neg", (self,))

    def __add__(self, other: Any) -> "ParameterExpression":
        return ParameterExpression("+", (self, _coerce_expression(other))).simplify()

    def __radd__(self, other: Any) -> "ParameterExpression":
        return _coerce_expression(other) + self

    def __sub__(self, other: Any) -> "ParameterExpression":
        return ParameterExpression("-", (self, _coerce_expression(other))).simplify()

    def __rsub__(self, other: Any) -> "ParameterExpression":
        return _coerce_expression(other) - self

    def __mul__(self, other: Any) -> "ParameterExpression":
        return ParameterExpression("*", (self, _coerce_expression(other))).simplify()

    def __rmul__(self, other: Any) -> "ParameterExpression":
        return _coerce_expression(other) * self

    def __truediv__(self, other: Any) -> "ParameterExpression":
        return ParameterExpression("/", (self, _coerce_expression(other))).simplify()

    def __rtruediv__(self, other: Any) -> "ParameterExpression":
        return _coerce_expression(other) / self

    def __pow__(self, other: Any) -> "ParameterExpression":
        return ParameterExpression("**", (self, _coerce_expression(other))).simplify()

    def __rpow__(self, other: Any) -> "ParameterExpression":
        return _coerce_expression(other) ** self


class ParameterVector(Sequence[Parameter]):
    """Ordered vector of scalar parameters."""

    def __init__(self, name: str, length: int = 0) -> None:
        _admit_parameter_values(self,name,length)
        if not isinstance(name, str):
            raise TypeError("QuantumBridge parameter vector names must be strings.")
        self._name = name
        self._root_uuid = str(uuid4())
        self._params = []
        self.resize(length)

    def resize(self, length: int) -> None:
        from .bits import integer_index
        _admit_parameter_values(self,length)
        length=integer_index(length)
        if length<0:raise OverflowError("Parameter vector length cannot be negative.")
        _check_vector_size(length)
        if length>len(self._params):
            # Stage new identities before changing the list borrowed by callers.
            added=[ParameterVectorElement(self,index) for index in range(len(self._params),length)]
            self._params.extend(added)
        else:del self._params[length:]

    @property
    def name(self):return self._name

    def __getitem__(self, index):
        _admit_parameter_values(self,index)
        return self._params[index]

    def __iter__(self) -> Iterator[Parameter]:
        _admit_parameter_values(self)
        return iter(self._params)

    def __len__(self) -> int:
        _admit_parameter_values(self)
        return len(self._params)

    def index(self,value,start=0,stop=None):
        _admit_parameter_values(self,value,start,stop)
        return self._params.index(value,start) if stop is None else self._params.index(value,start,stop)

    def __repr__(self) -> str:
        return f"ParameterVector({self.name!r}, {len(self)})"

    @property
    def params(self) -> list[Parameter]:
        _admit_parameter_values(self)
        return self._params


class ParameterVectorElement(Parameter):
    """A scalar identity with stable vector position, including after resize."""

    def __init__(self, vector: ParameterVector, index: int, uuid=None):
        from .bits import integer_index
        from quantumbridge.execution.resource_context import current_resources
        if current_resources() is not None and type(vector) is not ParameterVector:
            from .classical import UnsupportedFeature
            raise UnsupportedFeature("managed_parameter_vector_owner")
        _admit_parameter_values(index,uuid)
        if not isinstance(vector,ParameterVector):raise TypeError("ParameterVectorElement requires a native vector.")
        _admit_parameter_values(vector.name,vector._root_uuid)
        index=integer_index(index)
        if index<0:raise OverflowError("Parameter vector index cannot be negative.")
        _check_vector_size(index+1)
        identity=uuid if uuid is not None else UUID(int=(UUID(vector._root_uuid).int+index)%(1<<128))
        super().__init__(f"{vector.name}[{index}]",str(identity))
        object.__setattr__(self,"_uuid_explicit",uuid is not None)
        object.__setattr__(self,"_vector",vector)
        object.__setattr__(self,"_vector_name",vector.name)
        object.__setattr__(self,"_index",index)

    @property
    def vector(self) -> ParameterVector:
        return self._vector

    @property
    def index(self) -> int:
        return self._index


def parameter_sort_key(parameter):
    if isinstance(parameter,ParameterVectorElement):
        return (parameter._vector_name,1,parameter.index)
    return (parameter.name,0,0)


def validate_parameter_names(parameters):
    names={}
    for parameter in parameters:
        previous=names.setdefault(parameter.name,parameter)
        if previous!=parameter:
            raise ValueError(f"Conflicting identities for parameter name {parameter.name!r}.")


def legacy_parameter(name):
    # Older native IR stored only a name; it cannot recover separate identities.
    parameter=Parameter(name,str(uuid5(NAMESPACE_URL,"quantumbridge:parameter:"+name)))
    object.__setattr__(parameter,"_uuid_explicit",False)
    return parameter


ParameterValue = Union[Real, Parameter, ParameterExpression]


def resolve_parameter(value: ParameterValue, mapping: Mapping[Union[Parameter, str], Real], allow_partial: bool = False):
    if isinstance(value, Parameter):
        if value in mapping:
            replacement = mapping[value]
            return _coerce_expression(replacement) if isinstance(replacement,ParameterExpression) else replacement
        if value.name in mapping:
            replacement = mapping[value.name]
            return _coerce_expression(replacement) if isinstance(replacement,ParameterExpression) else replacement
        if allow_partial:
            return value
        raise ValueError(f"QuantumBridge parameter {value.name!r} is not bound.")
    if isinstance(value, ParameterExpression):
        if not isinstance(mapping, Mapping):
            raise TypeError("Parameter bindings must be a mapping.")
        # A circuit-wide dictionary can be much larger than this expression.
        # Lookup its referenced identities only, retaining legacy string keys.
        local = {}
        for parameter in value.parameters:
            if parameter in mapping or parameter.name in mapping:
                current_symbolic_limits().check("max_substitutions",len(local)+1)
                local[parameter] = mapping[parameter] if parameter in mapping else mapping[parameter.name]
        result = value._replace(local)
        if result.parameters:
            if allow_partial: return result
            try: numeric = result.numeric(strict=False)
            except TypeError: raise ValueError("QuantumBridge parameters are not fully bound.") from None
        else: numeric = result.numeric()
        if not isinstance(numeric, Real):
            if numeric.imag != 0:
                raise ValueError("Physical gate angles and global phase must be real; no imaginary part may be discarded.")
            numeric = numeric.real
        return numeric
    if isinstance(value, Real):
        return _checked_numeric(value)
    raise TypeError("QuantumBridge parameters must be real numbers, Parameter objects, or ParameterExpression objects.")


def parameter_set(values: Sequence[Any]) -> frozenset[Parameter]:
    found: set[Parameter] = set()
    for value in values:
        if isinstance(value, Parameter):
            found.add(value)
        elif isinstance(value, ParameterExpression):
            found.update(value.parameters)
    return frozenset(found)


def _coerce_expression(value: Any) -> ParameterExpression:
    if isinstance(value, ParameterExpression):
        limits = current_symbolic_limits()
        limits.check("max_nodes",value._nodes)
        limits.check("max_depth",value._depth)
        limits.check("max_parameters",len(value.parameters))
        return value
    if isinstance(value, Parameter):
        return value.expression()
    if isinstance(value, Complex):
        return ParameterExpression("const", (_checked_numeric(value),))
    raise TypeError("QuantumBridge parameter expressions require numeric scalars or native parameters.")


class _SymbolicWork:
    def __init__(self):
        self.limits, self.work = current_symbolic_limits(), 0

    def charge(self, amount=1):
        charge_symbolic_work(amount)
        self.work += amount
        self.limits.check("max_expansion_work", self.work)

    def validate(self, expression):
        self.limits.check("max_nodes", expression._nodes)
        self.limits.check("max_depth", expression._depth)
        self.limits.check("max_parameters", len(expression.parameters))


def _checked_numeric(value):
    if not isinstance(value, Complex):
        raise TypeError("Symbolic numeric bindings require numbers, not expressions or foreign objects.")
    limits = current_symbolic_limits()
    if isinstance(value, Integral):
        value = int(value)
        limits.check("max_integer_bits", value.bit_length())
    elif not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ValueError("Symbolic numeric values must be finite.")
    if abs(value) > limits.max_numeric_magnitude:
        raise ValueError("Symbolic numeric value exceeds max_numeric_magnitude.")
    return value


def _literal(value, target):
    return value.op == "const" and value.args[0] == target


def _with_derivative_order(expression, order):
    if expression._derivative_order >= order: return expression
    result = ParameterExpression(expression.op, expression.args,_retained_parameters=expression._retained_parameters)
    object.__setattr__(result, "_derivative_order", order)
    return result


def _with_parameter_context(expression, parameters):
    """Keep observable symbol identities separate from mathematical operands."""
    _check_vector_size(len(parameters))
    retained = frozenset(parameters)-expression.parameters
    if not retained: return expression
    return ParameterExpression(expression.op,expression.args,
                               _retained_parameters=expression._retained_parameters|retained)


_NOT_CONSTANT = object()


def _constant_value(expression, *, allow_cancellation=False):
    work, memo = _SymbolicWork(), {}
    work.validate(expression)
    def visit(node):
        work.charge()
        if id(node) in memo: return memo[id(node)]
        if node.op == "const": result = node.args[0]
        elif node.op == "parameter": result = _NOT_CONSTANT
        elif allow_cancellation and node.op == "*" and any(_literal(arg, 0) for arg in node.args):
            for arg in node.args: visit(arg)
            result = 0
        elif allow_cancellation and node.op == "-" and node.args[0] == node.args[1]:
            visit(node.args[0])
            result = 0
        else:
            args = tuple(visit(arg) for arg in node.args)
            if (allow_cancellation and node.op == "/" and args[0] is not _NOT_CONSTANT
                    and args[0] == 0 and args[1] is _NOT_CONSTANT): result = 0
            elif any(arg is _NOT_CONSTANT for arg in args): result = _NOT_CONSTANT
            elif node.op == "neg": result = -args[0]
            elif node.op in UNARY_FUNCTIONS | _GUARDED_DERIVATIVES:
                result = _evaluate_unary(node.op, args[0])
            else: result = _evaluate_binary(node.op, *args)
        memo[id(node)] = result
        return result
    return visit(expression)


def _evaluate_unary(op, value):
    value = _checked_numeric(value)
    real = isinstance(value, Real) or value.imag == 0
    if real and not isinstance(value, Real): value = value.real
    if op == "conjugate": return value.conjugate()
    if op == "abs": return abs(value)
    if op in {"sign", "abs_gradient", "sign_gradient"}:
        if not real: raise ValueError("Real sign and nonsmooth derivatives do not accept complex inputs.")
        if op != "sign" and value == 0:
            raise ValueError("Absolute value/sign is not differentiable at zero.")
        if op == "sign_gradient": return 0.
        return 1. if value > 0 else -1. if value < 0 else 0.
    if op == "log" and value == 0:
        raise ZeroDivisionError("Logarithm is singular at zero.")
    if op in {"arcsin", "arccos"} and real and not -1 <= value <= 1:
        raise ValueError("Real inverse trigonometric input must be in [-1, 1].")
    module = math if real and not (op == "log" and value < 0) else cmath
    name = {"arcsin": "asin", "arccos": "acos", "arctan": "atan"}.get(op, op)
    return _checked_numeric(getattr(module, name)(value))


def _differentiate(expression, parameter):
    work, memo = _SymbolicWork(), {}
    work.validate(expression)
    zero, one = _coerce_expression(0), _coerce_expression(1)

    def mul(left, right):
        if _literal(left, 0) or _literal(right, 0): return zero
        return left * right

    def visit(node):
        work.charge()
        if id(node) in memo: return memo[id(node)]
        if parameter not in node._dependencies: result = zero
        elif node.op == "parameter": result = one
        elif node.op == "neg": result = -visit(node.args[0])
        elif node.op in UNARY_FUNCTIONS | _GUARDED_DERIVATIVES:
            x = node.args[0]
            dx = visit(x)
            if node.op == "sin": factor = x.cos()
            elif node.op == "cos": factor = -x.sin()
            elif node.op == "tan": factor = 1 / x.cos()**2
            elif node.op == "exp": factor = x.exp()
            elif node.op == "log": factor = 1 / x
            elif node.op == "arcsin": factor = 1 / (1-x*x)**.5
            elif node.op == "arccos": factor = -1 / (1-x*x)**.5
            elif node.op == "arctan": factor = 1 / (1+x*x)
            elif node.op == "abs": factor = x._unary("abs_gradient")
            elif node.op in {"sign", "abs_gradient", "sign_gradient"}: factor = x._unary("sign_gradient")
            else:
                result = dx.conjugate()
                memo[id(node)] = result
                return result
            result = mul(factor, dx)
        else:
            a, b = node.args
            da, db = visit(a), visit(b)
            if node.op == "+": result = da + db
            elif node.op == "-": result = da - db
            elif node.op == "*": result = mul(da, b) + mul(a, db)
            elif node.op == "/": result = (mul(da, b) - mul(a, db)) / b**2
            elif _literal(db, 0):
                if _literal(b, 0): result = zero
                else: result = mul(mul(b, a**(b-1)), da)
            else:
                result = mul(node, mul(db, a.log()) + mul(b/a, da))
        memo[id(node)] = result
        return result
    return visit(expression)


def _evaluate_binary(op: str, left: Real, right: Real, *, check=True) -> Real:
    if check and op == "**":
        limits = current_symbolic_limits()
        limits.check("max_power_exponent", abs(right))
        if isinstance(left, Integral) and isinstance(right, Integral) and right > 0:
            limits.check("max_integer_bits", max(1, int(left).bit_length()) * right)
    if op == "+":
        result = left + right
    elif op == "-": result = left - right
    elif op == "*": result = left * right
    elif op == "/": result = left / right
    elif op == "**": result = left**right
    else: raise ValueError(f"Unknown QuantumBridge parameter expression operator {op!r}.")
    return _checked_numeric(result) if check else result
