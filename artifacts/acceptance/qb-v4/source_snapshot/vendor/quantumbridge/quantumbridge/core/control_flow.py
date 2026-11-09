# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Native structured control-flow data and lexical validation.

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
import json

from .classical import ClassicalExpression, Bool, Bit, UnsupportedFeature, lift, register, validate_classical_expr
from .parameters import Parameter, ParameterVectorElement, legacy_parameter


@dataclass(frozen=True)
class ExecutionLimits:
    max_loop_iterations: int = 1000
    max_instruction_executions: int = 100000
    max_control_depth: int = 32
    max_branches: int = 4096
    max_shots: int = 1000000

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError("QuantumBridge execution limits must be positive integers.")
            object.__setattr__(self, name, int(value))


def condition_expression(value):
    if isinstance(value, ClassicalExpression):
        result = value
    elif isinstance(value, bool):
        result = lift(value)
    else:
        if isinstance(value, dict): bits, expected = value["bits"], value["value"]
        else:
            bits, expected = value
            if isinstance(bits, Integral): bits = (int(bits),)
        result = register(bits).eq(int(expected))
    if result.dtype not in (Bool(), Bit()):
        raise TypeError("QuantumBridge control conditions require Bool or Bit expressions.")
    return result


def _encode(value, *, _ir_context=None):
    from .circuit import Circuit
    if isinstance(value, ClassicalExpression): return {"classical_expr": value.to_dict()}
    if isinstance(value, Circuit):
        context = _ir_context
        if id(value) in context.active: raise ValueError("Cyclic control-body IR graph.")
        context.depth += 1
        context.count("max_nesting_depth", context.depth)
        context.active.add(id(value))
        try: return {"circuit": value.to_ir().to_dict(_context=context)}
        finally:
            context.active.remove(id(value))
            context.depth -= 1
    if isinstance(value, Parameter):
        result={"loop_parameter":value.name,"loop_parameter_uuid":value.uuid}
        if isinstance(value,ParameterVectorElement):
            from quantumbridge.ir.qb_ir import _param_to_ir
            result["loop_parameter_vector"]=_param_to_ir(value)["parameter_vector"]
        return result
    if isinstance(value, tuple): return [_encode(item, _ir_context=_ir_context) for item in value]
    if value is None or isinstance(value, (str, bool, int)): return value
    raise TypeError("Unsupported value in QuantumBridge structured control-flow data.")


def _decode(value, *, serialization_limits=None, _parameter_vectors=None, _ir_context=None):
    from quantumbridge.ir.qb_ir import IRProgram
    from quantumbridge.compat.qiskit_adapter import circuit_from_ir
    if isinstance(value, list): return tuple(_decode(item,serialization_limits=serialization_limits,
        _parameter_vectors=_parameter_vectors, _ir_context=_ir_context) for item in value)
    if isinstance(value, dict):
        if set(value) == {"classical_expr"}: return ClassicalExpression.from_dict(value["classical_expr"])
        if set(value) == {"circuit"}:
            _ir_context.depth += 1
            _ir_context.count("max_nesting_depth", _ir_context.depth)
            try: return circuit_from_ir(IRProgram.from_dict(value["circuit"], validate=False,
                serialization_limits=serialization_limits,_parameter_vectors=_parameter_vectors,_context=_ir_context))
            finally: _ir_context.depth -= 1
        if set(value)=={"loop_parameter"}:return legacy_parameter(value["loop_parameter"])
        if set(value)=={"loop_parameter","loop_parameter_uuid"}:return Parameter(value["loop_parameter"],value["loop_parameter_uuid"])
        if set(value)=={"loop_parameter","loop_parameter_uuid","loop_parameter_vector"}:
            from quantumbridge.ir.qb_ir import param_from_ir
            return param_from_ir({"parameter":value["loop_parameter"],"parameter_uuid":value["loop_parameter_uuid"],
                "parameter_vector":value["loop_parameter_vector"]},_vectors=_parameter_vectors,_limits=serialization_limits)
        raise ValueError("Unknown QuantumBridge control-flow data node.")
    if value is None or isinstance(value, (str, bool, int)): return value
    raise ValueError("Malformed QuantumBridge control-flow data.")


_FIELDS = {"if": {"condition", "body", "alternate"}, "for": {"values", "variable", "parameter", "body"},
           "while": {"condition", "body"}, "switch": {"selector", "cases", "default"},
           "declare": {"target", "initial"}, "store": {"target", "value"}, "break": set(), "continue": set()}


@dataclass(frozen=True)
class ControlBlock:
    kind: str
    data: dict

    def __post_init__(self):
        if self.kind not in _FIELDS:
            raise UnsupportedFeature("control_flow_node", f"Unknown QuantumBridge control node {self.kind!r}.")
        if set(self.data) != _FIELDS[self.kind]:
            raise ValueError("Malformed QuantumBridge control-flow fields.")

    @property
    def bodies(self):
        if self.kind in {"if", "for", "while"}:
            return tuple(body for body in (self.data["body"], self.data.get("alternate")) if body is not None)
        if self.kind == "switch":
            return tuple(body for _, body in self.data["cases"]) + (() if self.data["default"] is None else (self.data["default"],))
        return ()

    @property
    def parameters(self):
        result = set().union(*(set(body.parameters) for body in self.bodies))
        if self.kind == "for": result.discard(self.data["parameter"])
        return result

    def bind(self, mapping):
        from .instruction_parameters import bind_definition
        if self.kind == "for" and self.data["parameter"] is not None:
            parameter = self.data["parameter"]
            mapping = {key: value for key, value in mapping.items() if key != parameter and key != parameter.name}
        result = deepcopy(self.data)
        for field in ("body", "alternate", "default"):
            if result.get(field) is not None:
                local = {key: value for key, value in mapping.items() if key in result[field].parameters or isinstance(key, str) and key in {p.name for p in result[field].parameters}}
                result[field] = bind_definition(result[field],local)
        if "cases" in result:
            result["cases"] = tuple((values, bind_definition(body,mapping)) for values, body in result["cases"])
        return ControlBlock(self.kind, result)

    def remap(self, qubits, bits, width, num_bits, *, identifiers=None, copy_operation=True):
        from .circuit import Circuit, _remap_circuit
        def mapped(value):
            if isinstance(value, ClassicalExpression): return value.remap(bits,identifiers)
            if isinstance(value, Circuit): return _remap_circuit(value,qubits,bits,width,num_bits,
                identifiers=identifiers,copy_operation=copy_operation)
            if isinstance(value, tuple): return tuple(mapped(item) for item in value)
            return deepcopy(value)
        return ControlBlock(self.kind, {key: mapped(value) for key, value in self.data.items()})

    def to_dict(self, *, _ir_context=None):
        if _ir_context is None:
            from quantumbridge.ir.qb_ir import _DefinitionCodec
            _ir_context = _DefinitionCodec()
        return {"version": "qb-control-v1", "kind": self.kind, "data": {key: _encode(value, _ir_context=_ir_context) for key, value in self.data.items()}}

    def to_json(self): return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data, *, serialization_limits=None, _parameter_vectors=None, _ir_context=None):
        if _ir_context is None:
            from quantumbridge.ir.qb_ir import _DefinitionCodec
            from quantumbridge.schema.serialization import validate_payload_structure
            _ir_context = _DefinitionCodec(serialization_limits)
            validate_payload_structure(data, limits=_ir_context.limits)
        serialization_limits = _ir_context.limits
        _parameter_vectors = {} if _parameter_vectors is None else _parameter_vectors
        if set(data) != {"version", "kind", "data"} or data["version"] != "qb-control-v1":
            raise ValueError("Malformed or unknown QuantumBridge control-flow schema.")
        return cls(data["kind"], {key: _decode(value,serialization_limits=serialization_limits,
            _parameter_vectors=_parameter_vectors,_ir_context=_ir_context) for key, value in data["data"].items()})

    @classmethod
    def from_json(cls, text): return cls.from_dict(json.loads(text))

    def validate(self, width, num_bits, environment=None, limits=None):
        from .circuit import Circuit
        circuit = Circuit(width, num_bits).append("control_flow", (), metadata={"control_node": self})
        validate_control_flow(circuit, environment, limits)
        return True


def validate_control_flow(circuit, environment=None, limits=None, depth=0, loop_depth=0):
    from .circuit import Circuit
    from .measurements import Measurement
    limits = limits or ExecutionLimits()
    if depth > limits.max_control_depth:
        raise ValueError("QuantumBridge control-flow nesting exceeds max_control_depth.")
    from .classical_scope import execution_environment
    env = execution_environment(circuit,environment)
    for item in circuit.iter_instructions():
        if isinstance(item, Measurement):
            circuit._check_wire(item.wire)
            circuit._check_bit(item.bit)
            continue
        node = item.metadata.get("control_node")
        from quantumbridge.devices.execution import validate_conditions, validate_wires
        validate_conditions(item.metadata, circuit.num_bits)
        validate_wires(item.wires, circuit.num_qubits)
        if node is None:
            if item.name == "control_flow": raise ValueError("Missing executable QuantumBridge control node.")
            if item.name == "reset" and (len(item.targets) != 1 or item.controls): raise ValueError("Reset needs one target and no controls.")
            if item.name == "kraus" and item.controls: raise ValueError("Kraus instructions cannot have coherent controls.")
            continue
        if not isinstance(node, ControlBlock): raise ValueError("Malformed QuantumBridge control node object.")
        data, local = node.data, dict(env)
        for key in ("condition", "selector", "initial", "value"):
            if data.get(key) is not None: validate_classical_expr(data[key], env, circuit.num_bits)
        if "condition" in data: condition_expression(data["condition"])
        if node.kind == "declare":
            target = data["target"]
            validate_classical_expr(target)
            if target.op != "var" or target.value[0] in env:
                raise ValueError("QuantumBridge variable declaration is invalid or duplicated in this scope.")
            env[target.value[0]] = target.dtype
            if data["initial"] is not None and target.dtype != data["initial"].dtype: raise TypeError("Classical initializer type mismatch.")
        if node.kind == "store":
            target = data["target"]
            validate_classical_expr(target, env, circuit.num_bits)
            if target.op not in {"var", "bit", "register"} or target.dtype != data["value"].dtype: raise TypeError("Invalid classical store target or type.")
        if node.kind in {"break", "continue"} and not loop_depth:
            raise ValueError("QuantumBridge break/continue requires an enclosing loop.")
        if node.kind == "for":
            if len(data["values"]) > limits.max_loop_iterations: raise ValueError("QuantumBridge for loop exceeds max_loop_iterations.")
            if any(isinstance(value, bool) or not isinstance(value, Integral) for value in data["values"]): raise TypeError("QuantumBridge for loops require integer values.")
            if data["parameter"] is not None and not isinstance(data["parameter"], Parameter): raise TypeError("Invalid loop parameter.")
            var = data["variable"]
            if var is not None:
                if not isinstance(var, ClassicalExpression) or var.op != "var" or var.dtype.kind not in {"uint","int"}: raise TypeError("QuantumBridge loop variable must be integer storage.")
                local[var.value[0]] = var.dtype
                for value in data["values"]: var.dtype.normalize(value)
        if node.kind == "switch":
            if data["selector"].dtype.kind not in {"uint", "bit", "int"}: raise TypeError("QuantumBridge switch selectors require integer types.")
            seen = set()
            for values, _ in data["cases"]:
                for value in values:
                    data["selector"].dtype.normalize(value)
                    if value in seen: raise ValueError("Duplicate QuantumBridge switch case.")
                    seen.add(value)
        for body in node.bodies:
            if not isinstance(body, Circuit) or body.num_qubits != circuit.num_qubits or body.num_bits != circuit.num_bits:
                raise ValueError("QuantumBridge control-flow bodies must use the enclosing resource map.")
            validate_control_flow(body, local, limits, depth + 1, loop_depth + int(node.kind in {"for", "while"}))
    return env
