# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Independent typed classical execution; no upstream implementation copied.

from dataclasses import dataclass
from numbers import Integral
from uuid import UUID, uuid4
import operator


class UnsupportedFeature(ValueError):
    def __init__(self, feature, message=None):
        self.feature = feature
        self.code = "unsupported_feature"
        super().__init__(message or f"QuantumBridge does not implement {feature}.")

    def to_dict(self):
        return {"code": self.code, "feature": self.feature, "message": str(self)}


class ClassicalExecutionError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)

    def to_dict(self):
        return {"code": self.code, "message": str(self)}


@dataclass(frozen=True, eq=False)
class ClassicalType:
    kind: str
    width: int = 1

    def __eq__(self, other):
        return isinstance(other, ClassicalType) and (self.kind, self.width) == (other.kind, other.width)

    def __hash__(self):
        return hash((self.kind, self.width))

    def __post_init__(self):
        if self.kind not in {"bool", "bit", "uint", "int", "duration"}:
            raise UnsupportedFeature("classical_type", f"Unsupported classical type {self.kind!r}.")
        if isinstance(self.width, bool) or not isinstance(self.width, Integral) or not 1 <= self.width <= 4096:
            raise ValueError("QuantumBridge classical widths must be integers in [1, 4096].")
        if self.kind not in {"uint","int"} and self.width != 1:
            raise ValueError("QuantumBridge Bool/Bit have width one.")

    def normalize(self, value, *, wrap=False):
        if self.kind == "duration":
            raise UnsupportedFeature("classical_duration_execution", "Symbolic duration identifiers need a timing executor, not integer storage.")
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise TypeError("QuantumBridge Bool values require bool, not integers.")
            return value
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("QuantumBridge bit/UInt values require integers, not booleans.")
        lower,upper = (-(1 << (self.width-1)),1 << (self.width-1)) if self.kind == "int" else (0,1 << self.width)
        if not wrap and not lower <= value < upper:
            raise ValueError("QuantumBridge classical value does not fit its declared width.")
        result = int(value) & ((1 << self.width) - 1)
        return result-(1 << self.width) if self.kind == "int" and result >= upper else result

    def to_dict(self):
        return {"kind": self.kind, "width": int(self.width)}


class Bool(ClassicalType):
    def __init__(self):
        super().__init__("bool")


class Bit(ClassicalType):
    def __init__(self):
        super().__init__("bit")


class UInt(ClassicalType):
    def __init__(self, width):
        super().__init__("uint", width)


class Int(ClassicalType):
    def __init__(self,width=32): super().__init__("int",width)


class Duration(ClassicalType):
    def __init__(self): super().__init__("duration")


@dataclass(frozen=True, eq=False)
class ClassicalExpression:
    op: str
    dtype: ClassicalType
    args: tuple = ()
    value: object = None

    def __post_init__(self):
        object.__setattr__(self, "args", tuple(self.args))
        if self.op in {"register", "var"} and isinstance(self.value, list):
            object.__setattr__(self, "value", tuple(self.value))

    def __bool__(self):
        raise TypeError("QuantumBridge symbolic classical expressions require runtime evaluation.")

    def eq(self, other): return binary("eq", self, other)
    def ne(self, other): return binary("ne", self, other)
    def __eq__(self, other): return binary("eq", self, other)
    def __ne__(self, other): return binary("ne", self, other)
    __hash__ = None
    def __lt__(self, other): return binary("lt", self, other)
    def __le__(self, other): return binary("le", self, other)
    def __gt__(self, other): return binary("gt", self, other)
    def __ge__(self, other): return binary("ge", self, other)
    def __and__(self, other): return binary("and", self, other)
    def __or__(self, other): return binary("or", self, other)
    def __xor__(self, other): return binary("xor", self, other)
    def __add__(self, other): return binary("add", self, other)
    def __sub__(self, other): return binary("sub", self, other)
    def __mul__(self, other): return binary("mul", self, other)
    def __lshift__(self, other): return binary("shl", self, other)
    def __rshift__(self, other): return binary("shr", self, other)
    def __invert__(self): return ClassicalExpression("invert", self.dtype, (self,))

    def to_dict(self):
        return {"op": self.op, "type": self.dtype.to_dict(), "args": [arg.to_dict() for arg in self.args], "value": self.value}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data)-{"identifier"} != {"op", "type", "args", "value"}:
            raise ValueError("Malformed QuantumBridge classical expression.")
        if "identifier" in data:
            kind = data["identifier"]
            if kind not in {"var", "stretch"} or kind != data["op"] or data["args"]:
                raise ValueError("Malformed native classical identifier.")
            identity, name = data["value"]
            dtype = ClassicalType(**data["type"])
            if kind == "stretch" and dtype != Duration(): raise TypeError("A Stretch must have Duration type.")
            return Var(identity, dtype, name=name) if kind == "var" else Stretch(identity, name)
        result = cls(data["op"], ClassicalType(**data["type"]), tuple(cls.from_dict(arg) for arg in data["args"]), data["value"])
        validate_classical_expr(result)
        return result

    def remap(self, bits, identifiers=None):
        if self.op in {"var", "stretch"}:
            return (identifiers or {}).get(self.value[0], self)
        value = self.value
        if self.op == "bit": value = bits[value]
        if self.op == "register": value = tuple(bits[bit] for bit in value)
        return ClassicalExpression(self.op, self.dtype, tuple(arg.remap(bits, identifiers) for arg in self.args), value)


class _Identifier:
    @property
    def name(self): return self.value[1]
    @property
    def var(self): return UUID(self.value[0])
    @property
    def type(self): return self.dtype
    def __hash__(self): return hash((type(self), self.value, self.dtype))
    def __eq__(self, other): return type(other) is type(self) and self.value == other.value and self.dtype == other.dtype
    def __ne__(self, other): return not self == other
    def to_dict(self): return {**super().to_dict(), "identifier": self.op}


class Var(_Identifier, ClassicalExpression):
    """Immutable named storage identity; legacy variable() keeps infix syntax."""
    def __init__(self, var, type, *, name):
        if not isinstance(name, str) or not isinstance(type, ClassicalType):
            raise TypeError("Var requires a string name and a native classical type.")
        super().__init__("var", type, value=(str(UUID(str(var))), name))
    @classmethod
    def new(cls, name, type): return cls(uuid4(), type, name=name)


class Stretch(_Identifier, ClassicalExpression):
    """A duration identifier, not an executable integer storage variable."""
    def __init__(self, var, name):
        if not isinstance(name, str): raise TypeError("Stretch requires a string name.")
        super().__init__("stretch", Duration(), value=(str(UUID(str(var))), name))
    @classmethod
    def new(cls, name): return cls(uuid4(), name)


def literal(value, dtype=None):
    dtype = dtype or (Bool() if isinstance(value, bool) else UInt(max(1, int(value).bit_length())) if isinstance(value, Integral) else None)
    if dtype is None:
        raise UnsupportedFeature("classical_literal", "QuantumBridge classical literals support Bool and unsigned integers.")
    return ClassicalExpression("literal", dtype, value=dtype.normalize(value))


def bit(index):
    if isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
        raise ValueError("QuantumBridge classical bit references require non-negative integer positions.")
    return ClassicalExpression("bit", Bit(), value=int(index))


def register(bits):
    selected = tuple(bits)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("QuantumBridge classical registers require distinct bit references.")
    for index in selected: bit(index)
    return ClassicalExpression("register", UInt(len(selected)), value=selected)


def variable(name, dtype):
    if not isinstance(name, str) or not name or not isinstance(dtype, ClassicalType):
        raise TypeError("QuantumBridge variables require a name and a classical type.")
    return ClassicalExpression("var", dtype, value=(uuid4().hex, name))


def lift(value, dtype=None):
    if isinstance(value, ClassicalExpression):
        if dtype is not None and value.dtype != dtype:
            raise TypeError("Use an explicit QuantumBridge classical cast to change types.")
        return value
    return literal(value, dtype)


def cast(value, dtype):
    return ClassicalExpression("cast", dtype, (lift(value),))


def logical_not(value):
    return ClassicalExpression("not", Bool(), (lift(value),))


def logical_and(left, right): return ClassicalExpression("logical_and", Bool(), (lift(left), lift(right)))
def logical_or(left, right): return ClassicalExpression("logical_or", Bool(), (lift(left), lift(right)))


_BINARY = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge,
           "and": operator.and_, "or": operator.or_, "xor": operator.xor, "add": operator.add, "sub": operator.sub,
           "mul": operator.mul, "shl": operator.lshift, "shr": operator.rshift}


def binary(op, left, right):
    if op not in _BINARY:
        raise UnsupportedFeature("classical_operator", f"Unsupported QuantumBridge classical operator {op!r}.")
    left = lift(left)
    right = lift(right, left.dtype) if not isinstance(right, ClassicalExpression) else right
    if left.dtype.kind != right.dtype.kind:
        raise TypeError("QuantumBridge classical operands require matching kinds; use explicit casts.")
    if left.dtype.kind in {"uint","int"} and left.dtype.width != right.dtype.width:
        common = ClassicalType(left.dtype.kind,max(left.dtype.width, right.dtype.width))
        left, right = cast(left, common), cast(right, common)
    if op in {"add", "sub", "mul", "shl", "shr", "lt", "le", "gt", "ge"} and left.dtype.kind not in {"uint","int"}:
        raise TypeError("QuantumBridge arithmetic/ordering requires UInt operands.")
    dtype = Bool() if op in {"eq", "ne", "lt", "le", "gt", "ge"} else left.dtype
    return ClassicalExpression(op, dtype, (left, right))


def validate_classical_expr(expr, environment=None, num_bits=None, depth=0):
    if not isinstance(expr, ClassicalExpression) or not isinstance(expr.dtype, ClassicalType) or depth > 64:
        raise ValueError("Invalid or excessively nested QuantumBridge classical expression.")
    for arg in expr.args:
        validate_classical_expr(arg, environment, num_bits, depth + 1)
    op = expr.op
    if op == "literal":
        expr.dtype.normalize(expr.value)
    elif op == "bit":
        if expr.dtype != Bit() or bit(expr.value).value != expr.value or num_bits is not None and expr.value >= num_bits:
            raise ValueError("QuantumBridge classical reference is outside this circuit.")
    elif op == "register":
        check = register(expr.value)
        if expr.dtype != check.dtype or num_bits is not None and max(expr.value) >= num_bits:
            raise ValueError("Invalid QuantumBridge register reference/type.")
    elif op in {"var", "stretch"}:
        if (not isinstance(expr.value, (tuple, list)) or len(expr.value) != 2
                or not isinstance(expr.value[0], str) or not expr.value[0]
                or not isinstance(expr.value[1], str) or not expr.value[1] and type(expr) not in {Var, Stretch}):
            raise ValueError("Malformed QuantumBridge variable reference.")
        if environment is not None and environment.get(expr.value[0]) != expr.dtype:
            raise ClassicalExecutionError("undeclared_variable", "QuantumBridge variable is outside its lexical scope.")
    elif op in _BINARY:
        if len(expr.args) != 2 or binary(op, *expr.args).dtype != expr.dtype:
            raise TypeError("Invalid QuantumBridge binary expression type/arity.")
    elif op in {"not", "invert", "cast", "logical_and", "logical_or"}:
        arity = 2 if op.startswith("logical_") else 1
        if len(expr.args) != arity or op in {"not", "logical_and", "logical_or"} and expr.dtype != Bool() or op == "invert" and expr.dtype != expr.args[0].dtype:
            raise TypeError("Invalid QuantumBridge unary/logical expression type/arity.")
    else:
        raise UnsupportedFeature("classical_expression", f"Unsupported expression {op!r}.")
    if op in {"literal", "bit", "register", "var", "stretch"} and expr.args:
        raise ValueError("QuantumBridge classical leaves cannot contain child expressions.")
    return True


class ClassicalStore:
    def __init__(self, num_bits=0):
        if isinstance(num_bits, bool) or not isinstance(num_bits, Integral) or num_bits < 0:
            raise ValueError("QuantumBridge classical store needs a non-negative bit count.")
        self.bits = [0] * num_bits
        self.types = {}
        self.values = {}
        self.names = {}

    def copy(self):
        result = ClassicalStore()
        result.bits, result.types, result.values, result.names = self.bits.copy(), self.types.copy(), self.values.copy(), self.names.copy()
        return result

    def declare(self, target, initial=None):
        if target.op != "var" or target.value[0] in self.types:
            raise ClassicalExecutionError("duplicate_variable", "QuantumBridge variable is already declared or not a variable.")
        key, name = target.value
        self.types[key], self.names[key] = target.dtype, name
        if initial is not None:
            classical_assign(self, target, initial)

    def to_dict(self):
        return {"bits": self.bits.copy(), "variables": {key: {"name": self.names[key], "type": dtype.to_dict(),
                "initialized": key in self.values, "value": self.values.get(key)} for key, dtype in self.types.items()}}

    @classmethod
    def from_dict(cls, payload):
        result = cls(len(payload["bits"]))
        result.bits = [Bit().normalize(value) for value in payload["bits"]]
        for key, entry in payload["variables"].items():
            if not isinstance(key, str) or not isinstance(entry["name"], str) or not isinstance(entry["initialized"], bool):
                raise ValueError("Malformed QuantumBridge classical storage snapshot.")
            result.types[key], result.names[key] = ClassicalType(**entry["type"]), entry["name"]
            if entry["initialized"]: result.values[key] = result.types[key].normalize(entry["value"])
        return result


def classical_eval(expr, store):
    validate_classical_expr(expr, store.types, len(store.bits))
    def visit(node):
        op = node.op
        if op == "literal": return node.value
        if op == "bit": return store.bits[node.value]
        if op == "register": return sum(store.bits[bit] << offset for offset, bit in enumerate(node.value))
        if op == "var":
            if node.value[0] not in store.values:
                raise ClassicalExecutionError("uninitialized_variable", "QuantumBridge variable was read before initialization.")
            return store.values[node.value[0]]
        values = [visit(arg) for arg in node.args]
        if op == "cast":
            return bool(values[0]) if node.dtype.kind == "bool" else node.dtype.normalize(int(values[0]), wrap=True)
        if op == "not": return not values[0]
        if op == "logical_and": return bool(values[0]) and bool(values[1])
        if op == "logical_or": return bool(values[0]) or bool(values[1])
        if op == "invert": return not values[0] if node.dtype.kind == "bool" else node.dtype.normalize(~values[0], wrap=True)
        if op in {"shl", "shr"} and values[1] >= node.dtype.width: return 0
        result = _BINARY[op](*values)
        return result if node.dtype.kind == "bool" else node.dtype.normalize(result, wrap=True)
    return visit(expr)


def classical_assign(store, target, value):
    validate_classical_expr(target, store.types, len(store.bits))
    expr = lift(value, target.dtype)
    result = target.dtype.normalize(classical_eval(expr, store))
    if target.op == "var": store.values[target.value[0]] = result
    elif target.op == "bit": store.bits[target.value] = result
    elif target.op == "register":
        for offset, bit in enumerate(target.value): store.bits[bit] = (result >> offset) & 1
    else: raise TypeError("QuantumBridge assignment target must be a variable, bit or register.")
    return result


def serialize_classical_expr(expr):
    validate_classical_expr(expr)
    return expr.to_dict()


def deserialize_classical_expr(data):
    return ClassicalExpression.from_dict(data)
