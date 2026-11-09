# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.
# Design source: docs/mvp/ir_design_v0.1.md.

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Optional

from quantumbridge.core.parameters import Parameter, ParameterExpression, ParameterVector, ParameterVectorElement, legacy_parameter, EXPRESSION_OPS


def matrix_from_ir(value):
    import numpy as np
    from quantumbridge.execution.resource_context import current_resources
    scope = current_resources()
    if scope is not None:
        from quantumbridge.execution.array_reservations import numeric_array, numeric_shape, numeric_into
        from quantumbridge.execution.array_ownership import retained_array
        if type(value) is dict and set(value) == {"real", "imag"}:
            shape = numeric_shape(value["real"], real_only=True)
            if shape != numeric_shape(value["imag"], real_only=True):
                raise ValueError("QuantumBridge matrix real/imag shapes must match.")
            result = retained_array(scope.ledger, "ir_complex_conversion", shape, complex)
            scope.ledger.unavailable("ir_python_conversion_workspace",
                "IR conversion owns its complex output; Python containers and internal iterator workspace are not measured RSS.")
            numeric_into(result.real, value["real"])
            numeric_into(result.imag, value["imag"])
            return result
        return numeric_array(value, category="ir_numeric_conversion")

    if isinstance(value, dict) and set(value) == {"real", "imag"}:
        real, imag = np.asarray(value["real"]), np.asarray(value["imag"])
        if real.shape != imag.shape:
            raise ValueError("QuantumBridge matrix real/imag shapes must match.")
        return real.astype(float) + 1j * imag.astype(float)
    return np.asarray(value, dtype=complex)


class _DefinitionCodec:
    def __init__(self, limits=None, *, include_parameter_bindings=True, canonical_parameters=False, include_user_metadata=True):
        from quantumbridge.serialization.limits import SerializationLimits
        self.limits = limits or SerializationLimits()
        self.definitions = self.operations = self.depth = 0
        self.parameter_nodes = 0
        self.modifiers = 0
        self.active = set()
        self.include_parameter_bindings = include_parameter_bindings
        self.canonical_parameters = canonical_parameters
        self.include_user_metadata = include_user_metadata

    def count(self, field, value):
        self.limits.check(field, value, format="qb-ir")

    def definition(self):
        self.definitions += 1
        self.count("max_custom_definitions", self.definitions)
        self.count("max_nesting_depth", self.depth+1)


def _gate_to_ir(gate, context):
    from quantumbridge.core.gates import Gate, ControlledGate, validate_definition
    from quantumbridge.serialization.errors import UnsupportedFeature
    from quantumbridge.core.gates import is_native_gate
    from quantumbridge.core.standard_gates import standard_type_name, UnitaryGate, invocation_params
    if not is_native_gate(gate):
        raise UnsupportedFeature("Only native Gate types have definition records.", format="qb-ir")
    if id(gate) in context.active: raise ValueError("Cyclic native Gate definition.")
    context.definition()
    context.active.add(id(gate))
    context.depth += 1
    try:
        if gate.definition is not None: validate_definition(gate.definition, owner=gate)
        result = {"type": "controlled_gate" if isinstance(gate, ControlledGate) else "gate",
                  "name": gate.name, "label": gate.label, "num_qubits": gate.num_qubits,
                  "num_clbits": 0, "params": [_param_to_ir(value, _context=context) for value in invocation_params(gate)],
                  "metadata": IRInstruction(gate.name, (), metadata=gate.metadata).to_dict(_context=context)["metadata"],
                  "definition": None if gate.definition is None else gate.definition.to_ir().to_dict(_context=context)}
        standard = standard_type_name(gate)
        if standard is not None: result["standard_gate"] = standard
        if type(gate) is UnitaryGate:
            result["matrix_parameter"] = _matrix_to_ir(gate.params[0])
        if isinstance(gate, ControlledGate):
            result.update(base_gate=_gate_to_ir(gate.base_gate, context),
                          num_ctrl_qubits=gate.num_ctrl_qubits, ctrl_state=gate.ctrl_state)
            if type(gate) is ControlledGate:result["closed_control_name"]=gate._closed_control_name
        if context.include_parameter_bindings:
            from quantumbridge.core.instruction_parameters import binding_plan_for_serialization,definition_signature
            plan = binding_plan_for_serialization(gate,preserve_invalidated=True)
            if plan is not None:
                context.definition()
                context.depth += 1
                try:
                    result["parameter_binding"] = {"formals": [_param_to_ir(value, _context=context) for value in plan.formals],
                        "definition": plan.definition.to_ir().to_dict(_context=context)}
                    if definition_signature(gate.definition)!=plan.signature:result["parameter_binding"]["invalidated"]=True
                finally: context.depth -= 1
        return result
    finally:
        context.depth -= 1
        context.active.remove(id(gate))


def _gate_from_ir(data, vectors, limits, context):
    from quantumbridge.core.gates import Gate, ControlledGate
    from quantumbridge.compat.qiskit_adapter import circuit_from_ir
    required = {"type", "name", "label", "num_qubits", "num_clbits", "params", "metadata", "definition"}
    if type(data) is not dict or data.get("type") not in {"gate", "controlled_gate"}:
        raise ValueError("Unknown native definition record type.")
    controlled = data["type"] == "controlled_gate"
    if "closed_control_name" in data and (not controlled or "standard_gate" in data):
        raise ValueError("A closed control name requires a generic controlled record.")
    fields = required | ({"base_gate", "num_ctrl_qubits", "ctrl_state"} if controlled else set())
    if (not fields <= set(data) <= fields | {"parameter_binding", "standard_gate", "matrix_parameter", "closed_control_name"} or type(data["num_qubits"]) is not int or data["num_qubits"] < 0
            or type(data["num_clbits"]) is not int or data["num_clbits"] != 0
            or type(data["name"]) is not str or not data["name"]
            or not any(type(data["label"]) is kind for kind in (str,type(None)))
            or type(data["params"]) is not list or type(data["metadata"]) is not dict):
        raise ValueError("Malformed native Gate definition record.")
    context.definition()
    context.count("max_qubits", data["num_qubits"])
    context.count("max_parameter_nodes", len(data["params"]))
    context.depth += 1
    try:
        params = tuple(param_from_ir(value, _vectors=vectors, _limits=limits) for value in data["params"])
        metadata = decode_instruction_metadata(data["metadata"], _vectors=vectors, _limits=limits, _context=context)
        definition = None if data["definition"] is None else circuit_from_ir(IRProgram.from_dict(
            data["definition"], serialization_limits=limits, _parameter_vectors=vectors, _context=context))
        if "matrix_parameter" in data and data.get("standard_gate") != "UnitaryGate":
            raise ValueError("Matrix parameters require an explicit native UnitaryGate type record.")
        if "standard_gate" in data:
            from quantumbridge.core.standard_gates import standard_from_record
            from quantumbridge.core.instruction_parameters import definition_signature
            if data["standard_gate"] == "UnitaryGate":
                from quantumbridge.execution.array_reservations import numeric_shape
                from quantumbridge.core.gate_limits import current_gate_limits
                matrix = data.get("matrix_parameter")
                if type(matrix) is not dict or set(matrix) != {"real", "imag"} or params:
                    raise ValueError("A UnitaryGate record needs exactly one typed matrix parameter.")
                shape = numeric_shape(matrix["real"], real_only=True)
                if len(shape) != 2 or shape != numeric_shape(matrix["imag"], real_only=True):
                    raise ValueError("Unitary matrix parameter real/imag shapes disagree.")
                context.count("max_numeric_elements", shape[0]*shape[1])
                current_gate_limits().check("max_matrix_dimension",max(shape))
                values = (matrix_from_ir(matrix),)
            else: values = params
            gate = standard_from_record(data["standard_gate"], values, data["label"], data.get("ctrl_state"))
            if (isinstance(gate, ControlledGate) != controlled or gate.num_qubits != data["num_qubits"]
                    or metadata or definition is None or definition_signature(gate.definition) != definition_signature(definition)):
                raise ValueError("Standard type record differs from its actual executable definition.")
            if controlled:
                base = _gate_from_ir(data["base_gate"], vectors, limits, context)
                if (gate.num_ctrl_qubits != data["num_ctrl_qubits"] or type(base) is not type(gate.base_gate)
                        or definition_signature(base.definition) != definition_signature(gate.base_gate.definition)):
                    raise ValueError("Standard control type differs from its actual base operation.")
            object.__setattr__(gate, "name", data["name"])
        elif controlled:
            if definition is None: raise ValueError("A controlled definition record must be executable.")
            base = _gate_from_ir(data["base_gate"], vectors, limits, context)
            gate = ControlledGate(base, data["num_ctrl_qubits"], data["ctrl_state"], data["label"])
            from quantumbridge.core.standard_gates import invocation_params
            if gate.num_qubits != data["num_qubits"] or invocation_params(gate) != params or metadata:
                raise ValueError("Controlled Gate base and definition header disagree.")
            if "closed_control_name" in data:
                if type(data["closed_control_name"]) is not str:raise ValueError("Invalid native controlled name.")
                gate.name=data["closed_control_name"]
                if gate.name!=data["name"]:raise ValueError("Controlled display and closed name disagree.")
            else:gate._restore_display_name(data["name"])
            gate.definition = definition
        else: gate = Gate(data["name"], data["num_qubits"], params, data["label"], definition=definition, metadata=metadata)
        if "parameter_binding" in data:
            binding = data["parameter_binding"]
            if (type(binding) is not dict or not {"formals","definition"}<=set(binding)<={"formals","definition","invalidated"}
                    or "invalidated" in binding and binding["invalidated"] is not True
                    or type(binding["formals"]) is not list or definition is None):
                raise ValueError("Malformed native parameter binding record.")
            context.definition()
            context.count("max_parameter_nodes", len(binding["formals"]))
            context.depth += 1
            try:
                formals = tuple(param_from_ir(value, _vectors=vectors, _limits=limits) for value in binding["formals"])
                template = circuit_from_ir(IRProgram.from_dict(binding["definition"], serialization_limits=limits,
                    _parameter_vectors=vectors, _context=context))
            finally: context.depth -= 1
            from quantumbridge.core.instruction_parameters import restore_binding_plan
            restore_binding_plan(gate, template, formals,invalidated=binding.get("invalidated",False))
        return gate
    finally:
        context.depth -= 1


def _instruction_to_ir(instruction, context):
    from quantumbridge.core.operations import Instruction, instruction_definition_work
    if type(instruction) is not Instruction: raise ValueError("Unknown native Instruction type.")
    if id(instruction) in context.active: raise ValueError("Cyclic Instruction record.")
    context.definition()
    context.active.add(id(instruction))
    context.depth += 1
    try:
        if instruction.definition is not None: instruction_definition_work(instruction.definition, owner=instruction)
        result = {"type": "instruction", "name": instruction.name, "label": instruction.label,
            "num_qubits": instruction.num_qubits, "num_clbits": instruction.num_clbits,
            "params": [_param_to_ir(value, _context=context) for value in instruction.params],
            "metadata": IRInstruction(instruction.name, (), metadata=instruction.metadata).to_dict(_context=context)["metadata"],
            "definition": None if instruction.definition is None else instruction.definition.to_ir().to_dict(_context=context)}
        if context.include_parameter_bindings:
            from quantumbridge.core.instruction_parameters import binding_plan_for_serialization,definition_signature
            plan = binding_plan_for_serialization(instruction,preserve_invalidated=True)
            if plan is not None:
                context.definition()
                context.depth += 1
                try:
                    result["parameter_binding"] = {"formals": [_param_to_ir(value, _context=context) for value in plan.formals],
                        "definition": plan.definition.to_ir().to_dict(_context=context)}
                    if definition_signature(instruction.definition)!=plan.signature:result["parameter_binding"]["invalidated"]=True
                finally: context.depth -= 1
        return result
    finally:
        context.depth -= 1
        context.active.remove(id(instruction))


def _instruction_from_ir(data, vectors, limits, context):
    from quantumbridge.core.operations import Instruction
    from quantumbridge.compat.qiskit_adapter import circuit_from_ir
    required = {"type", "name", "label", "num_qubits", "num_clbits", "params", "metadata", "definition"}
    if (type(data) is not dict or not required <= set(data) <= required | {"parameter_binding"}
            or data["type"] != "instruction" or type(data["num_qubits"]) is not int or data["num_qubits"] < 0
            or type(data["num_clbits"]) is not int or data["num_clbits"] < 0
            or type(data["params"]) is not list or type(data["metadata"]) is not dict):
        raise ValueError("Malformed native Instruction record.")
    context.definition()
    context.count("max_qubits", data["num_qubits"])
    context.count("max_clbits", data["num_clbits"])
    context.count("max_parameter_nodes", len(data["params"]))
    context.depth += 1
    try:
        params = tuple(param_from_ir(value, _vectors=vectors, _limits=limits) for value in data["params"])
        metadata = decode_instruction_metadata(data["metadata"], _vectors=vectors, _limits=limits, _context=context)
        definition = None if data["definition"] is None else circuit_from_ir(IRProgram.from_dict(
            data["definition"], serialization_limits=limits, _parameter_vectors=vectors, _context=context))
        result = Instruction(data["name"], data["num_qubits"], data["num_clbits"], params, data["label"],
            definition=definition, metadata=metadata)
        if "parameter_binding" in data:
            binding = data["parameter_binding"]
            if (type(binding) is not dict or not {"formals","definition"}<=set(binding)<={"formals","definition","invalidated"}
                    or "invalidated" in binding and binding["invalidated"] is not True
                    or type(binding["formals"]) is not list):
                raise ValueError("Malformed Instruction binding record.")
            context.definition()
            context.depth += 1
            try:
                formals = tuple(param_from_ir(value, _vectors=vectors, _limits=limits) for value in binding["formals"])
                template = circuit_from_ir(IRProgram.from_dict(binding["definition"], serialization_limits=limits,
                    _parameter_vectors=vectors, _context=context))
            finally: context.depth -= 1
            from quantumbridge.core.instruction_parameters import restore_binding_plan
            restore_binding_plan(result, template, formals,invalidated=binding.get("invalidated",False))
        return result
    finally: context.depth -= 1


def _annotation_to_ir(operation, context):
    from quantumbridge.core.annotated_operation import (
        AnnotatedOperation, ControlModifier, PowerModifier, validate_annotated_structure,
    )
    from quantumbridge.core.gates import Gate
    from quantumbridge.core.operations import Instruction
    validate_annotated_structure(operation)
    if id(operation) in context.active: raise ValueError("Cyclic annotated operation record.")
    context.definition()
    context.active.add(id(operation))
    context.depth += 1
    try:
        base = operation.base_op
        if type(base) is AnnotatedOperation:
            kind, record = "annotated_operation", _annotation_to_ir(base, context)
        else:
            kind = "instruction" if type(base) is Instruction else "gate"
            if kind == "instruction":
                record = _instruction_to_ir(base, context)
            else: record = _gate_to_ir(base, context)
        context.modifiers += len(operation.modifiers)
        context.count("max_gate_expansion", context.modifiers)
        modifiers = []
        for item in operation.modifiers:
            if type(item) is ControlModifier:
                modifiers.append({"type": "control", "num_ctrl_qubits": item.num_ctrl_qubits, "ctrl_state": item.ctrl_state})
            elif type(item) is PowerModifier:
                modifiers.append({"type": "power", "power": _param_to_ir(item.power, _context=context)})
            else: modifiers.append({"type": "inverse"})
        return {"type": "annotated_operation", "base_kind": kind, "base_op": record, "modifiers": modifiers}
    finally:
        context.depth -= 1
        context.active.remove(id(operation))


def _annotation_from_ir(data, vectors, limits, context):
    from quantumbridge.core.annotated_operation import AnnotatedOperation, InverseModifier, ControlModifier, PowerModifier
    from quantumbridge.core.operations import Instruction
    if (type(data) is not dict or set(data) != {"type", "base_kind", "base_op", "modifiers"}
            or data["type"] != "annotated_operation" or type(data["modifiers"]) is not list):
        raise ValueError("Malformed annotated operation record.")
    context.definition()
    context.modifiers += len(data["modifiers"])
    context.count("max_gate_expansion", context.modifiers)
    context.depth += 1
    try:
        if data["base_kind"] == "annotated_operation":
            base = _annotation_from_ir(data["base_op"], vectors, limits, context)
        elif data["base_kind"] in {"gate", "instruction"}:
            if type(data["base_op"]) is not dict: raise ValueError("An annotated base requires an operation record.")
            typed_instruction = data["base_kind"] == "instruction" and data["base_op"].get("type") == "instruction"
            base = _instruction_from_ir(data["base_op"], vectors, limits, context) if typed_instruction else _gate_from_ir(data["base_op"], vectors, limits, context)
            if data["base_kind"] == "instruction" and not typed_instruction:
                if data["base_op"]["type"] != "gate": raise ValueError("An Instruction base cannot hide a ControlledGate record.")
                from quantumbridge.core.instruction_parameters import inherit_binding_plan
                original = base
                base = Instruction(base.name, base.num_qubits, params=base.params, definition=base.definition, metadata=base.metadata)
                inherit_binding_plan(original, base)
        else: raise ValueError("Unknown annotated base operation type.")
        modifiers = []
        for item in data["modifiers"]:
            if type(item) is not dict: raise ValueError("Malformed modifier record.")
            if item.get("type") == "inverse" and set(item) == {"type"}:
                modifier = InverseModifier()
            elif item.get("type") == "control" and set(item) == {"type", "num_ctrl_qubits", "ctrl_state"}:
                modifier = ControlModifier(item["num_ctrl_qubits"], item["ctrl_state"])
            elif item.get("type") == "power" and set(item) == {"type", "power"}:
                modifier = PowerModifier(param_from_ir(item["power"], _vectors=vectors, _limits=limits))
            else: raise ValueError("Unknown or malformed modifier record.")
            modifiers.append(modifier)
        return AnnotatedOperation(base, modifiers)
    finally: context.depth -= 1


def decode_instruction_metadata(metadata, *, _vectors=None, _limits=None, _context=None):
    result = dict(metadata)
    if "global_phase" in result:
        result["global_phase"] = param_from_ir(result["global_phase"], _vectors=_vectors, _limits=_limits)
    if isinstance(result.get("gate"), dict):
        context = _context or _DefinitionCodec(_limits)
        result["gate"] = _gate_from_ir(result["gate"], {} if _vectors is None else _vectors, context.limits, context)
    if isinstance(result.get("annotated_operation"), dict):
        context = _context or _DefinitionCodec(_limits)
        result["annotated_operation"] = _annotation_from_ir(result["annotated_operation"],
            {} if _vectors is None else _vectors, context.limits, context)
    if isinstance(result.get("instruction_object"), dict):
        context = _context or _DefinitionCodec(_limits)
        result["instruction_object"] = _instruction_from_ir(result["instruction_object"],
            {} if _vectors is None else _vectors, context.limits, context)
    if "controlled_base" in result:
        description = result["controlled_base"]
        if type(description) is not dict or set(description) != {"name", "metadata", "controls"} or type(description["controls"]) is not int or description["controls"] < 1:
            raise ValueError("Malformed controlled primitive definition.")
        context = _context or _DefinitionCodec(_limits)
        context.definition()
        context.depth += 1
        try:
            result["controlled_base"] = {**description, "metadata": decode_instruction_metadata(
                description["metadata"], _vectors=_vectors, _limits=context.limits, _context=context)}
        finally: context.depth -= 1
    if "matrix" in result:
        result["matrix"] = matrix_from_ir(result["matrix"])
    if "kraus" in result:
        result["kraus"] = tuple(matrix_from_ir(value) for value in result["kraus"])
    if "control_node" in result and isinstance(result["control_node"], dict):
        from quantumbridge.core.control_flow import ControlBlock
        result["control_node"] = ControlBlock.from_dict(result["control_node"],
            serialization_limits=_limits, _parameter_vectors=_vectors, _ir_context=_context)
    if isinstance(result.get("instruction"),dict) and "params" in result["instruction"]:
        result["instruction"]=dict(result["instruction"])
        result["instruction"]["params"]=[param_from_ir(value,_vectors=_vectors,_limits=_limits) if isinstance(value,dict)
            and ("parameter" in value or "parameter_ast" in value) else value for value in result["instruction"]["params"]]
    return result


def _matrix_to_ir(value):
    matrix = matrix_from_ir(value)
    return {"real": matrix.real.tolist(), "imag": matrix.imag.tolist()}


def _param_to_ir(value: Any, *, _context=None, _depth=0) -> Any:
    if _context is not None:
        _context.parameter_nodes += 1
        _context.count("max_parameter_nodes", _context.parameter_nodes)
        _context.count("max_symbolic_expression_depth", _depth)
    if isinstance(value, Parameter):
        result={"parameter":value.name,"parameter_uuid":value.uuid}
        if not value._uuid_explicit:result["parameter_uuid_explicit"]=False
        if isinstance(value,ParameterVectorElement):
            result["parameter_vector"]={"name":value._vector_name,"root_uuid":value.vector._root_uuid,
                "length":len(value.vector),"index":value.index}
        return result
    if isinstance(value, ParameterExpression):
        result = {
            "expression": str(value),
            "parameters": sorted(parameter.name for parameter in value.parameters),
            "parameter_ast": {"op": value.op, "args": [_param_to_ir(arg, _context=_context, _depth=_depth+1) for arg in value.args]},
        }
        if value._retained_parameters:
            from quantumbridge.core.parameters import parameter_sort_key
            result["parameter_ast"]["parameter_context"] = [_param_to_ir(parameter,_context=_context,_depth=_depth+1)
                for parameter in sorted(value._retained_parameters,key=parameter_sort_key)]
        if _context is not None and _context.canonical_parameters:
            result.pop("expression")
            result.pop("parameters")
        return result
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, complex):
        return {"parameter_complex": [value.real, value.imag]}
    return value


def param_from_ir(value, *, _vectors=None, _limits=None, _depth=0):
    if not isinstance(value, dict): return value
    vectors={} if _vectors is None else _vectors
    from quantumbridge.serialization.limits import SerializationLimits
    limits = _limits or SerializationLimits()
    limits.check("max_symbolic_expression_depth", _depth, format="ir-json")
    count = vectors.get("expression_nodes", 0) + 1
    limits.check("max_parameter_nodes", count, format="ir-json")
    vectors["expression_nodes"] = count
    if "parameter_complex" in value:
        parts = value["parameter_complex"]
        if set(value) != {"parameter_complex"} or type(parts) not in (tuple, list) or len(parts) != 2 or not all(isinstance(part, Real) for part in parts):
            raise ValueError("Malformed complex scalar parameter literal.")
        from quantumbridge.core.parameters import _checked_numeric
        return _checked_numeric(complex(*parts))
    if "parameter" in value:
        if not set(value)<={"parameter","parameter_uuid","parameter_uuid_explicit","parameter_vector"}:
            raise ValueError("Unknown QuantumBridge parameter identity fields.")
        if "parameter_uuid_explicit" in value and type(value["parameter_uuid_explicit"]) is not bool:
            raise ValueError("Parameter identity provenance must be a boolean.")
        name,identity=value["parameter"],value.get("parameter_uuid")
        spec=value.get("parameter_vector")
        if spec is None:
            parameter=legacy_parameter(name) if identity is None else Parameter(name,identity)
        else:
            from uuid import UUID
            from quantumbridge.serialization.limits import SerializationLimits
            if type(spec) is not dict or set(spec)!={"name","root_uuid","length","index"}:
                raise ValueError("Malformed parameter vector identity.")
            if type(spec["length"]) is not int or type(spec["index"]) is not int or min(spec["length"],spec["index"])<0:
                raise ValueError("Parameter vector lengths/indices must be nonnegative integers.")
            if type(spec["name"]) is not str or type(spec["root_uuid"]) is not str or type(identity) is not str:
                raise ValueError("Parameter vector identity fields must be strings.")
            limits=_limits or SerializationLimits()
            limits.check("max_parameter_nodes",spec["length"],format="ir-json")
            root_uuid=str(UUID(spec["root_uuid"]))
            identity=str(UUID(identity))
            index=spec["index"]
            if name!=f"{spec['name']}[{index}]":raise ValueError("Vector parameter name does not match its index.")
            key=(spec["name"],root_uuid,spec["length"])
            if key not in vectors:
                count=vectors.get(None,0)+spec["length"]
                limits.check("max_parameter_nodes",count,format="ir-json")
                vector=ParameterVector(spec["name"])
                vector._root_uuid=root_uuid
                vector.resize(spec["length"])
                vectors[key]=vector
                vectors[None]=count
            vector=vectors[key]
            parameter=vector[index] if index<len(vector) and vector[index].uuid==identity else ParameterVectorElement(vector,index,identity)
        if "parameter_uuid_explicit" in value:
            object.__setattr__(parameter,"_uuid_explicit",value["parameter_uuid_explicit"])
        return parameter
    if "parameter_ast" in value:
        tree = value["parameter_ast"]
        if type(tree) is not dict or not {"op","args"} <= set(tree) <= {"op","args","parameter_context"} or tree["op"] not in EXPRESSION_OPS or type(tree["args"]) not in (tuple, list):
            raise ValueError("Unknown executable QuantumBridge parameter AST.")
        context = tree.get("parameter_context",[])
        if type(context) is not list: raise ValueError("Parameter context must be an explicit identity list.")
        limits.check("max_parameter_nodes",vectors.get("expression_nodes",0)+len(context),format="ir-json")
        retained = tuple(param_from_ir(item,_vectors=vectors,_limits=limits,_depth=_depth+1) for item in context)
        if any(not isinstance(item,Parameter) for item in retained) or len(set(retained)) != len(retained):
            raise ValueError("Parameter context requires distinct native scalar identities.")
        args = tuple(param_from_ir(arg,_vectors=vectors,_limits=limits,_depth=_depth+1) for arg in tree["args"])
        return ParameterExpression(tree["op"], args,_retained_parameters=frozenset(retained))
    raise ValueError("QuantumBridge parameter expressions require an executable AST, not text.")


@dataclass(frozen=True)
class IRInstruction:
    op: str
    targets: tuple[int, ...]
    controls: tuple[int, ...] = ()
    params: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, _context=None) -> dict[str, Any]:
        context = _context or _DefinitionCodec()
        metadata = {key: value for key, value in self.metadata.items() if key not in {"matrix", "kraus"}}
        if "global_phase" in metadata: metadata["global_phase"] = _param_to_ir(metadata["global_phase"], _context=context)
        if "gate" in metadata: metadata["gate"] = _gate_to_ir(metadata["gate"], context)
        if "annotated_operation" in metadata: metadata["annotated_operation"] = _annotation_to_ir(metadata["annotated_operation"], context)
        if "instruction_object" in metadata: metadata["instruction_object"] = _instruction_to_ir(metadata["instruction_object"], context)
        if "controlled_base" in metadata:
            description = metadata["controlled_base"]
            context.definition()
            context.depth += 1
            try:
                metadata["controlled_base"] = {**description, "metadata": IRInstruction(
                    description["name"], (), metadata=description["metadata"]).to_dict(_context=context)["metadata"]}
            finally: context.depth -= 1
        if "matrix" in self.metadata:
            metadata["matrix"] = _matrix_to_ir(self.metadata["matrix"])
        if "kraus" in self.metadata:
            metadata["kraus"] = [_matrix_to_ir(value) for value in self.metadata["kraus"]]
        if "control_node" in self.metadata:
            metadata["control_node"] = self.metadata["control_node"].to_dict(_ir_context=context)
        if isinstance(metadata.get("instruction"),dict) and "params" in metadata["instruction"]:
            metadata["instruction"]=dict(metadata["instruction"])
            metadata["instruction"]["params"]=[_param_to_ir(value) for value in metadata["instruction"]["params"]]
        return {
            "op": self.op,
            "targets": list(self.targets),
            "controls": list(self.controls),
            "params": [_param_to_ir(param, _context=context) for param in self.params],
            "metadata": metadata,
        }


@dataclass(frozen=True)
class IRMeasurement:
    kind: str
    wire: int
    bit: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = {"kind": self.kind, "wires": [self.wire], "bits": [self.bit]}
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


@dataclass(frozen=True)
class IRExecutionState:
    global_phase: Any = 0.
    classical_scope: Optional[dict[str, Any]] = None
    calibrations: Optional[dict[str, Any]] = None
    control_flow: Optional[list[Any]] = None
    initial_layout: Optional[dict[int, int]] = None
    routing: Optional[dict[str, Any]] = None
    schedule: Optional[dict[str, Any]] = None
    transpile: Optional[dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data, context, vectors):
        if type(data) is not dict or set(data)-{"version", *cls.__dataclass_fields__} or type(data.get("version")) is not int or data["version"] != 1:
            raise ValueError("Malformed native IR execution record.")
        fields = {key: value for key, value in data.items() if key not in {"version", "global_phase"}}
        for key, value in fields.items():
            expected = list if key in {"control_flow", "initial_layout"} else dict
            if value is not None and type(value) is not expected: raise ValueError("Malformed native IR execution field.")
        if fields.get("initial_layout") is not None:fields["initial_layout"] = cls.layout_from_ir(fields["initial_layout"])
        if fields.get("routing") is not None:
            fields["routing"] = dict(fields["routing"])
            if "final_layout" in fields["routing"]:fields["routing"]["final_layout"] = cls.layout_from_ir(fields["routing"]["final_layout"])
        return cls(param_from_ir(data.get("global_phase", 0.), _vectors=vectors, _limits=context.limits), **fields)

    @staticmethod
    def layout_from_ir(data):
        if type(data) is not list or any(type(pair) is not list or len(pair) != 2 or any(type(i) is not int or i < 0 for i in pair) for pair in data):
            raise ValueError("Malformed native IR layout record.")
        result = dict(data)
        if len(result) != len(data) or len(set(result.values())) != len(data):raise ValueError("Native IR layout must be injective.")
        return result

    def to_dict(self, context):
        result = {"version": 1, "global_phase": _param_to_ir(self.global_phase, _context=context)}
        for key in self.__dataclass_fields__:
            if key == "global_phase": continue
            value = getattr(self,key)
            if value is None: continue
            if type(value) is not (list if key == "control_flow" else dict):raise ValueError("Malformed native IR execution field.")
            if key == "initial_layout":value = [list(pair) for pair in value.items()]
            elif key == "routing" and "final_layout" in value:
                value = {**value,"final_layout":[list(pair) for pair in value["final_layout"].items()]}
            result[key] = value
        return result

    def configuration(self):
        return {key:getattr(self,key) for key in self.__dataclass_fields__ if getattr(self,key) is not None}


@dataclass(frozen=True)
class IRProgram:
    ir_version: str
    num_qubits: int
    num_bits: int
    instructions: tuple[IRInstruction, ...]
    measurements: tuple[IRMeasurement, ...] = ()
    name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    bit_graph: Optional[dict[str, Any]] = None
    execution: Optional[IRExecutionState] = None

    @classmethod
    def from_dict(cls, data, validate=True, *, serialization_limits=None, _parameter_vectors=None, _context=None):
        context = _context or _DefinitionCodec(serialization_limits)
        serialization_limits = context.limits
        if _context is None:
            from quantumbridge.schema.serialization import validate_payload_structure
            validate_payload_structure(data, limits=serialization_limits)
        if data.get("ir_version") != "qb-ir-v0.1": raise ValueError("Unknown QuantumBridge IR version.")
        context.operations += len(data["instructions"])
        context.count("max_instructions", context.operations)
        registers = data["registers"]
        context.count("max_qubits", registers["quantum"]["size"])
        context.count("max_clbits", registers["classical"]["size"])
        graph=registers.get("bit_graph")
        if type(graph) is dict and type(graph.get("groups")) is list:
            context.count("max_registers",len(graph["groups"]))
        vectors={} if _parameter_vectors is None else _parameter_vectors
        result = cls(data["ir_version"], registers["quantum"]["size"], registers["classical"]["size"],
            tuple(IRInstruction(item["op"], tuple(item["targets"]), tuple(item.get("controls", ())),
                tuple(param_from_ir(value,_vectors=vectors,_limits=serialization_limits) for value in item.get("params", ())),
                decode_instruction_metadata(item.get("metadata", {}),_vectors=vectors,_limits=serialization_limits,_context=context)) for item in data["instructions"]),
            tuple(IRMeasurement(item["kind"], item["wires"][0], item["bits"][0], dict(item.get("metadata", {}))) for item in data.get("measurements", ())), data.get("name"), dict(data.get("metadata", {})),graph,
            None if "execution" not in data else IRExecutionState.from_dict(data["execution"],context,vectors))
        if result.execution is None and "global_phase" in result.metadata:
            result.metadata["global_phase"] = param_from_ir(result.metadata["global_phase"],_vectors=vectors,_limits=serialization_limits)
        if validate: result.validate()
        return result

    @classmethod
    def from_json(cls, text, *, serialization_limits=None):
        from quantumbridge.schema.serialization import from_json
        from quantumbridge.core.symbolic_limits import symbolic_serialization_limits
        serialization_limits = symbolic_serialization_limits(serialization_limits)
        return cls.from_dict(from_json(text, limits=serialization_limits),serialization_limits=serialization_limits)

    def to_json(self, *, serialization_limits=None):
        from quantumbridge.schema.serialization import to_json
        from quantumbridge.core.symbolic_limits import symbolic_serialization_limits
        serialization_limits = symbolic_serialization_limits(serialization_limits)
        return to_json(self.to_dict(serialization_limits=serialization_limits), limits=serialization_limits)

    def validate(self):
        from quantumbridge.compat.qiskit_adapter import circuit_from_ir
        from quantumbridge.core.control_flow import validate_control_flow
        if self.ir_version != "qb-ir-v0.1": raise ValueError("Unknown QuantumBridge IR version.")
        validate_control_flow(circuit_from_ir(self))
        return True

    @classmethod
    def from_circuit(cls, circuit) -> "IRProgram":
        from quantumbridge.core.bits import bit_graph
        from quantumbridge.core.operations import execution_operation
        from quantumbridge.core.circuit_metadata import is_separated, execution_metadata
        execution = None
        if is_separated(circuit):
            configuration = execution_metadata(circuit)
            if set(configuration)-{*IRExecutionState.__dataclass_fields__, "registers"}:
                from quantumbridge.serialization.errors import UnsupportedFeature
                raise UnsupportedFeature("This executable configuration has no typed IR transport record.",format="qb-ir")
            execution = IRExecutionState(**{key:value for key,value in configuration.items() if key != "registers"})
        return cls(
            ir_version="qb-ir-v0.1",
            num_qubits=circuit.num_qubits,
            num_bits=circuit.num_bits,
            instructions=tuple(
                IRInstruction(op.name, op.targets, op.controls, op.params, op.metadata)
                for op in (execution_operation(item,standardize_inverse=True) for item in circuit.operations)
            ),
            measurements=tuple(IRMeasurement(item.kind, item.wire, item.bit, dict(item.metadata)) for item in circuit.measurements),
            name=circuit.name,
            metadata=circuit.metadata if execution is not None else dict(circuit.metadata),
            bit_graph=bit_graph(circuit),
            execution=execution,
        )

    def to_dict(self, *, serialization_limits=None, _context=None) -> dict[str, Any]:
        context = _context or _DefinitionCodec(serialization_limits)
        context.operations += len(self.instructions)
        context.count("max_instructions", context.operations)
        if self.execution is not None and context.include_user_metadata:
            from quantumbridge.serialization.common import checked_json_metadata
            checked_json_metadata(self.metadata,context.limits,format="qb-ir")
        metadata = {} if self.execution is not None and not context.include_user_metadata else dict(self.metadata)
        if self.execution is None and "global_phase" in metadata: metadata["global_phase"] = _param_to_ir(metadata["global_phase"], _context=context)
        return {
            "ir_version": self.ir_version,
            "name": self.name,
            "registers": {
                "quantum": {"size": self.num_qubits},
                "classical": {"size": self.num_bits},
                **({} if self.bit_graph is None else {"bit_graph":self.bit_graph}),
            },
            "instructions": [instruction.to_dict(_context=context) for instruction in self.instructions],
            "measurements": [measurement.to_dict() for measurement in self.measurements],
            "metadata": metadata,
            **({} if self.execution is None else {"execution": self.execution.to_dict(context)}),
        }
