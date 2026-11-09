# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Native unitary definitions, structural expansion and coherent control."""

from numbers import Integral, Real
from contextvars import ContextVar
import fnmatch
import math

import numpy as np

from .operations import Instruction, Operation,execution_operation
from .parameters import Parameter, parameter_set, resolve_parameter, _coerce_expression
from .gate_limits import current_gate_limits

_VALIDATION_BUDGET = ContextVar("native_definition_validation_budget", default=None)


def is_native_gate(value):
    from .standard_gates import STANDARD_GATE_TYPES
    return any(type(value) is kind for kind in (Gate, ControlledGate, *STANDARD_GATE_TYPES))


def is_native_instruction(value):
    from .control_flow_operations import is_control_owner
    return type(value) is Instruction or is_native_gate(value) or is_control_owner(value)


def _copy(value):
    from .circuit_metadata import execution_copy
    return execution_copy(value, category="native_gate_definition_copy")


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(label + " must be an integer, not a boolean.")
    return int(value)


def _validate_primitive_arity(name, params, metadata):
    seen = set()
    while "controlled_base" in metadata:
        if id(metadata) in seen: raise ValueError("Cyclic controlled primitive metadata.")
        seen.add(id(metadata))
        current_gate_limits().check("max_definition_depth", len(seen))
        description = metadata["controlled_base"]
        name, metadata = description["name"], description["metadata"]
    if name == "native_matrix_power":
        from quantumbridge.diff.matrix_power_jet import matrix_power_parameter_count
        if len(params) != matrix_power_parameter_count(metadata):
            raise ValueError("Matrix-power gate parameter count does not match its source.")
        return
    expected = (1 if name in {"rx", "ry", "rz", "phase", "p", "u1", "crx", "cry", "crz", "cp", "cphase", "rxx", "ryy", "rzz", "global_phase", "native_matrix_power"}
                else 2 if name == "u2" else 3 if name in {"u", "u3"} else 0)
    if len(params) != expected: raise ValueError("Primitive gate parameter count does not match its operation.")


def validate_definition(circuit, *, owner=None):
    """Validate a whole definition graph before copying or allocating matrices."""
    budget = _VALIDATION_BUDGET.get()
    if budget is not None: return _validate_definition(circuit, owner=owner, budget=budget)
    budget = {"operations": 0, "definitions": 0, "work": 0}
    token = _VALIDATION_BUDGET.set(budget)
    try: return _validate_definition(circuit, owner=owner, budget=budget)
    finally: _VALIDATION_BUDGET.reset(token)


def _validate_definition(circuit, *, owner, budget):
    from .circuit import Circuit,is_native_circuit
    from quantumbridge.utils.math import gate_matrix, validate_unitary_matrix
    from quantumbridge.execution.resource_context import current_resources
    if not isinstance(circuit, Circuit): raise TypeError("Gate definitions require a native Circuit.")
    if current_resources() is not None and not is_native_circuit(circuit):
        from .classical import UnsupportedFeature
        raise UnsupportedFeature("managed_gate_definition_type", "Managed Gate definitions require the exact native Circuit type.")
    limits = current_gate_limits()
    if owner is not None:
        if circuit.num_qubits != owner.num_qubits:
            raise ValueError("Gate definition width no longer matches its owner.")
        if not set(circuit.parameters) <= set(parameter_set(owner.params)):
            raise ValueError("Gate definition parameter namespace no longer matches its owner.")
    # Keep ephemeral execution owners alive while their definition is pending.
    stack = [(circuit, 0, () if owner is None else (owner,))]
    start = budget["operations"]
    while stack:
        body, depth, ancestors = stack.pop()
        if not isinstance(body, Circuit): raise TypeError("Gate definitions require a native Circuit.")
        if current_resources() is not None and not is_native_circuit(body):
            from .classical import UnsupportedFeature
            raise UnsupportedFeature("managed_gate_definition_type")
        if any(body is ancestor for ancestor in ancestors): raise ValueError("Cyclic gate definition.")
        ancestors = ancestors+(body,)
        limits.check("max_definition_depth", depth)
        budget["definitions"] += 1
        limits.check("max_custom_definitions", budget["definitions"])
        budget["operations"] += len(body.operations)
        limits.check("max_total_operations", budget["operations"])
        if body.num_bits or body.measurements or body.has_dynamic_operations:
            raise ValueError("A Gate definition must be unitary and have no classical bits or measurements.")
        _coerce_expression(body.global_phase)
        if not parameter_set((body.global_phase,)):
            phase = resolve_parameter(body.global_phase, {})
            if not isinstance(phase, Real) or not math.isfinite(phase):
                raise ValueError("A unitary Gate definition requires a finite real global phase.")
        limits.check("max_gate_parameters", len(body.parameters))
        for op in body.operations:
            op=execution_operation(op)
            budget["work"] += 1
            limits.check("max_decomposition_work", budget["work"])
            if op.metadata.get("directive") or op.metadata.get("requires_zero_input"):
                raise ValueError("Directive/state preparation instructions are not unitary Gate definitions.")
            child = op.metadata.get("gate")
            instruction = op.metadata.get("instruction_object")
            scalar_child=(is_native_gate(child) and type(child.num_qubits) is int and child.num_qubits==0
                or type(instruction) is Instruction and type(instruction.num_qubits) is int and instruction.num_qubits==0)
            scalar_matrix=op.name=="unitary" and not op.controls
            if (not op.wires and not (scalar_child or scalar_matrix) or len(set(op.wires)) != len(op.wires)
                    or any(type(w) is not int or not 0 <= w < body.num_qubits for w in op.wires)):
                raise ValueError("Invalid gate definition wire mapping.")
            if instruction is not None:
                if (type(instruction) is not Instruction or child is not None or op.controls
                        or instruction.num_bits or instruction.num_qubits != len(op.wires)
                        or tuple(op.params) != tuple(instruction.params) or instruction.definition is None):
                    raise ValueError("A Gate requires an executable unitary Instruction definition.")
                if any(instruction is ancestor for ancestor in ancestors): raise ValueError("Cyclic instruction definition.")
                stack.append((instruction.definition, depth+1, ancestors+(instruction,)))
                continue
            annotation = op.metadata.get("annotated_operation")
            if annotation is not None:
                from .annotated_operation import validate_annotated_operation
                from .standard_gates import invocation_params
                validate_annotated_operation(annotation)
                if child is not None or annotation.num_qubits != len(op.wires) or op.params != invocation_params(annotation):
                    raise ValueError("Invalid annotated definition invocation.")
                continue
            if child is not None:
                if current_resources() is not None and not is_native_gate(child):
                    from .classical import UnsupportedFeature
                    raise UnsupportedFeature("managed_gate_type")
                if not isinstance(child, Gate) or child.num_qubits != len(op.wires):
                    raise ValueError("Invalid native gate node width or type.")
                if any(child is ancestor for ancestor in ancestors): raise ValueError("Cyclic gate definition.")
                definition = child.definition
                if definition is not None:
                    stack.append((definition, depth+1, ancestors+(child,)))
                    continue
            # Symbolic arguments need not be evaluated to check a primitive's
            # arity/unitarity. Numeric custom matrices still receive validation.
            metadata = child.metadata if child is not None else op.metadata
            name = child.name if child is not None else op.name
            params = child.params if child is not None else op.params
            _validate_primitive_arity(name, params, metadata)
            if name == "native_matrix_power" and parameter_set(params):
                from quantumbridge.diff.matrix_power_jet import _decode
                annotation, _, _ = _decode((0.,)*len(params), metadata)
                if annotation.num_qubits != len(op.wires):
                    raise ValueError("Matrix-power definition width disagrees with its invocation.")
                continue
            checked = []
            for value in params:
                _coerce_expression(value)
                checked.append(0. if parameter_set((value,)) else resolve_parameter(value, {}))
            matrix = gate_matrix(name, tuple(checked), metadata)
            if matrix.shape != (1 << len(op.wires),) * 2 or not validate_unitary_matrix(matrix):
                raise ValueError("Gate definition contains an invalid unitary primitive.")
    return budget["operations"]-start


class Gate(Instruction):
    """A native unitary Instruction with owned, validated circuit structure.

    A checked live parameter view updates owned definitions atomically while
    preserving declared formal lineage across repeated rebinding.
    """

    def __init__(self, name, num_qubits, params=(), label=None, *, definition=None, metadata=None):
        if isinstance(num_qubits, bool): raise ValueError("A gate width cannot be a boolean.")
        if label is not None and (not isinstance(label, str) or not label): raise ValueError("Invalid gate label.")
        object.__setattr__(self, "label", label)
        if not isinstance(num_qubits, int) or num_qubits < 0: raise ValueError("A Gate needs a nonnegative qubit count.")
        Instruction.__init__(self, name, num_qubits, 0, params, label,
            definition=definition, metadata=_copy(dict(metadata or {})))

    def __setattr__(self, name, value):
        if name in {"definition", "params"}:
            return object.__setattr__(self, name, value)
        if name in {"name", "label"}:
            if value is not None and (not isinstance(value, str) or not value): raise ValueError("Invalid gate name or label.")
            if name == "name" and value is None: raise ValueError("A Gate needs a name.")
            return object.__setattr__(self, name, value)
        raise AttributeError("Use definition/params assignment or copy to update a native Gate.")

    @property
    def num_clbits(self): return 0

    @property
    def params(self): return Instruction.params.fget(self)

    @params.setter
    def params(self, values):
        Instruction.params.fset(self, values)

    def _validate_parameter_arity(self, values):
        if self.definition is None:
            from quantumbridge.circuit_library.standard_gates import _ONE_QUBIT, _TWO_QUBIT, _THREE_QUBIT
            if self.name in _ONE_QUBIT | _TWO_QUBIT | _THREE_QUBIT or "controlled_base" in self.metadata:
                _validate_primitive_arity(self.name, values, self.metadata)

    def validate_parameter(self, value):
        from .parameters import ParameterExpression
        if isinstance(value, Parameter): return value
        if isinstance(value, ParameterExpression):
            _coerce_expression(value)
            if value.is_real() is False: raise ValueError("Physical Gate parameters must be real.")
            return value
        if not isinstance(value, Real): raise TypeError("Gate parameters must be real scalars or native expressions.")
        if not isinstance(value, (int, float)):
            value = int(value) if isinstance(value, np.integer) else float(value) if isinstance(value, np.floating) else value
        _coerce_expression(value)
        return value

    def _validate_definition(self, value, params):
        validate_definition(value)
        if value.num_qubits != self.num_qubits or not set(value.parameters) <= set(parameter_set(params)):
            raise ValueError("Gate definition width/parameter namespace differs from its owner.")

    @property
    def definition(self): return self._definition

    @definition.setter
    def definition(self, value):
        if value is not None: self._validate_definition(value, self._params)
        object.__setattr__(self, "_definition", _copy(value))
        object.__setattr__(self, "_parameter_plan", None)

    def copy(self, name=None):
        copied = _copy(self)
        if name is not None:
            object.__setattr__(copied, "_mutable", True)
            copied.name = name
        return copied

    def is_parameterized(self): return bool(parameter_set(self.params))

    def assign_parameters(self, mapping, *, strict=True):
        from .circuit import _parameter_mapping_from_values
        from .instruction_parameters import bind_definition
        ordered = tuple(sorted(parameter_set(self.params), key=lambda p: (p.name, p.uuid)))
        values = _parameter_mapping_from_values(ordered, mapping, strict)
        definition = None if self.definition is None else bind_definition(self.definition,values)
        params = tuple(resolve_parameter(value, values, allow_partial=True) for value in self.params)
        result = Gate(self.name, self.num_qubits, params, self.label, definition=definition, metadata=self.metadata)
        from .instruction_parameters import inherit_binding_plan
        inherit_binding_plan(self, result)
        return result

    def to_matrix(self):
        from quantumbridge.utils.math import gate_matrix
        from quantumbridge.information import Operator
        limits = current_gate_limits()
        if self.num_qubits > limits.max_matrix_dimension.bit_length()-1:
            raise MemoryError("Gate matrix exceeds max_matrix_dimension before allocation.")
        if self.is_parameterized(): raise ValueError("Bind gate parameters before requesting a matrix.")
        if self.definition is not None:
            validate_definition(self.definition, owner=self)
            return Operator.from_circuit(self.definition.lowered()).data
        _validate_primitive_arity(self.name, self.params, self.metadata)
        return _copy(gate_matrix(self.name, self.params, self.metadata))

    def inverse(self, annotated=False):
        if annotated:
            from .annotated_operation import AnnotatedOperation, InverseModifier
            return AnnotatedOperation(self, InverseModifier())
        from .circuit import Circuit
        body = self.definition if self.definition is not None else Circuit(self.num_qubits).append(self.name, range(self.num_qubits), params=self.params, metadata=self.metadata)
        result=body.inverse().to_gate(label=self.label, _name=self.name+"_dg")
        if getattr(self,"_public_matrix_basis",None)=="literal_wrapper":
            object.__setattr__(result,"_public_matrix_basis","literal_wrapper")
        return result

    def repeat(self, n):
        from .circuit import Circuit
        n = _integer(n, "Gate repetition count")
        if n < 1: raise ValueError("Gate.repeat requires a strictly positive count.")
        current_gate_limits().check("max_repeat_count", n)
        count = 1 if self.definition is None else validate_definition(self.definition, owner=self)
        current_gate_limits().check("max_total_operations", n*max(1, count))
        body = Circuit(self.num_qubits)
        for _ in range(n): body.append(self, range(self.num_qubits))
        result=body.to_gate(_name=self.name+"*"+str(n))
        from .standard_gates import UnitaryGate
        if type(self) is UnitaryGate or getattr(self,"_public_matrix_basis",None)=="literal_wrapper":
            object.__setattr__(result,"_public_matrix_basis","literal_wrapper")
        return result

    def power(self, exponent, annotated=False):
        from .classical import UnsupportedFeature
        from .instruction_parameters import admit_owner, admit_payloads
        admit_owner(self)
        admit_payloads((exponent, annotated))
        if annotated:
            from .annotated_operation import AnnotatedOperation, PowerModifier
            return AnnotatedOperation(self, PowerModifier(exponent))
        if isinstance(exponent, bool) or not isinstance(exponent, Real) or not math.isfinite(exponent):
            raise ValueError("Gate powers require a finite real exponent.")
        if float(exponent).is_integer():
            result = self._integer_power(int(exponent))
            if self.is_parameterized(): return result
            from .standard_gates import UnitaryGate
            output=UnitaryGate(result.to_matrix(), label=self.name+"^"+str(exponent))
            if type(self) is not UnitaryGate and getattr(self,"_public_matrix_basis",None)!="literal_wrapper":
                object.__setattr__(output,"_public_matrix_basis","native")
            return output
        if self.is_parameterized(): raise UnsupportedFeature("symbolic_fractional_gate_power")
        from quantumbridge.information import Operator
        powered = Operator(self.to_matrix()).power(exponent, branch_cut_rotation=0, assume_unitary=True).data
        from .standard_gates import UnitaryGate
        output=UnitaryGate(powered, label=self.name+"^"+str(exponent))
        if type(self) is not UnitaryGate and getattr(self,"_public_matrix_basis",None)!="literal_wrapper":
            object.__setattr__(output,"_public_matrix_basis","native")
        return output

    def _integer_power(self, exponent):
        from .circuit import Circuit
        if not exponent: return Circuit(self.num_qubits).to_gate(_name=self.name+"^0")
        base = self.inverse() if exponent < 0 else self
        return base.repeat(abs(exponent))

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None, annotated=False):
        if annotated:
            from .annotated_operation import AnnotatedOperation, ControlModifier
            return AnnotatedOperation(self, ControlModifier(num_ctrl_qubits, ctrl_state))
        return ControlledGate(self, num_ctrl_qubits, ctrl_state, label)


class ControlledGate(Gate):
    @property
    def name(self):
        fields=object.__getattribute__(self,"__dict__")
        if type(self) is not ControlledGate:return fields["name"]
        name=fields["_closed_control_name"]
        count,state=fields.get("num_ctrl_qubits"),fields.get("ctrl_state")
        return name+"_o"+str(state) if count is not None and state!=(1<<count)-1 else name

    @name.setter
    def name(self,value):
        fields=object.__getattribute__(self,"__dict__")
        fields["_closed_control_name" if type(self) is ControlledGate else "name"]=value

    def _restore_display_name(self,value):
        if type(self) is not ControlledGate:self.name=value;return
        suffix="_o"+str(self.ctrl_state)
        if self.ctrl_state!=(1<<self.num_ctrl_qubits)-1 and value.endswith(suffix):value=value[:-len(suffix)]
        self.name=value

    @property
    def _params(self):
        if type(self) is not ControlledGate:return object.__getattribute__(self,"__dict__")["_params"]
        current=self;seen=set()
        while type(current) is ControlledGate:
            if id(current) in seen:raise ValueError("Cyclic controlled base parameter graph.")
            seen.add(id(current));current_gate_limits().check("max_definition_depth",len(seen))
            current=object.__getattribute__(current,"__dict__")["base_gate"]
        return object.__getattribute__(current,"_params")

    @_params.setter
    def _params(self,values):
        if type(self) is ControlledGate:object.__setattr__(self.base_gate,"_params",values)
        else:object.__getattribute__(self,"__dict__")["_params"]=values

    @property
    def params(self): return Instruction.params.fget(self)

    @params.setter
    def params(self, values):
        from .instruction_parameters import validated_values
        values = validated_values(self, values)
        base = getattr(self, "base_gate", None)
        updated = None
        if base is not None:
            from .instruction_parameters import same_values
            from .standard_gates import UnitaryGate
            updated = base.copy() if same_values(tuple(base.params), values) else base.to_mutable()
            if not same_values(tuple(base.params), values): updated.params = values
            if type(base) is UnitaryGate:
                candidate = ControlledGate(updated, self.num_ctrl_qubits, self.ctrl_state, self.label)
                candidate._restore_display_name(self.name)
                object.__getattribute__(self, "__dict__").update(object.__getattribute__(candidate, "__dict__"))
                return
        Gate.params.fset(self, values)
        if updated is not None: object.__setattr__(self, "base_gate", updated)

    def validate_parameter(self, value):
        base = getattr(self, "base_gate", None)
        return Gate.validate_parameter(self, value) if base is None else base.validate_parameter(value)

    def assign_parameters(self, mapping, *, strict=True):
        from .circuit import _parameter_mapping_from_values
        ordered = tuple(sorted(parameter_set(self.params), key=lambda p: (p.name, p.uuid)))
        values = _parameter_mapping_from_values(ordered, mapping, strict)
        result = ControlledGate(self.base_gate.assign_parameters(values, strict=False),
                                self.num_ctrl_qubits, self.ctrl_state, self.label)
        result._restore_display_name(self.name)
        if self.definition is not None:
            from .instruction_parameters import bind_definition
            result.definition = bind_definition(self.definition,values)
        from .instruction_parameters import inherit_binding_plan
        inherit_binding_plan(self, result)
        return result

    def inverse(self, annotated=False):
        from .instruction_parameters import admit_owner,definition_signature
        from .standard_gates import invocation_params
        admit_owner(self)
        if annotated or type(self) is not ControlledGate: return Gate.inverse(self,annotated)
        invocation_params(self)
        validate_definition(self.definition,owner=self)
        canonical=ControlledGate(self.base_gate,self.num_ctrl_qubits,self.ctrl_state,self.label)
        if (self.metadata or definition_signature(self.definition)!=definition_signature(canonical.definition)):
            return Gate.inverse(self)
        return ControlledGate(self.base_gate.inverse(),self.num_ctrl_qubits,self.ctrl_state)

    def __init__(self, base_gate, num_ctrl_qubits=1, ctrl_state=None, label=None):
        from .circuit import Circuit
        n = _integer(num_ctrl_qubits, "Control width")
        if n < 1: raise ValueError("Control width must be positive.")
        current_gate_limits().check("max_control_qubits", n)
        if isinstance(ctrl_state, str):
            if len(ctrl_state) != n or set(ctrl_state)-{"0", "1"}: raise ValueError("Invalid control bitstring.")
            ctrl_state = int(ctrl_state, 2)
        state = (1 << n)-1 if ctrl_state is None else _integer(ctrl_state, "Control state")
        if not 0 <= state < (1 << n): raise ValueError("Control state is outside its register.")
        if not isinstance(base_gate, Gate): raise TypeError("ControlledGate requires a native Gate.")
        from quantumbridge.execution.resource_context import current_resources
        if current_resources() is not None and not is_native_gate(base_gate):
            from .classical import UnsupportedFeature
            raise UnsupportedFeature("managed_gate_type")
        base = base_gate.definition
        if base is None:
            base = Circuit(base_gate.num_qubits).append(base_gate.name, range(base_gate.num_qubits), params=base_gate.params, metadata=base_gate.metadata)
        validate_definition(base)
        base = base.lowered()
        current_gate_limits().check("max_total_operations", len(base.operations)+2*n+1)
        out = Circuit(n+base.num_qubits)
        # ctrl_state uses the public integer convention: bit zero names the
        # first control wire, while native state tensors are wire-major.
        opened = [wire for wire in range(n) if not (state >> wire) & 1]
        for wire in opened: out.x(wire)
        if parameter_set((base.global_phase,)) or base.global_phase != 0:
            _append_controlled(out, "phase", (n-1,), base.global_phase, tuple(range(n-1)), {})
        for op in base.operations:
            _append_controlled(out, op.name, tuple(n+w for w in op.wires), op.params, tuple(range(n)), op.metadata)
        for wire in reversed(opened): out.x(wire)
        name = "c"+base_gate.name if n == 1 else "c"+str(n)+base_gate.name
        if opened and type(self) is not ControlledGate: name += "_o"+str(state)
        object.__setattr__(self, "base_gate", base_gate.copy())
        super().__init__(name, out.num_qubits, base_gate.params, label, definition=out)
        object.__setattr__(self, "num_ctrl_qubits", n)
        object.__setattr__(self, "ctrl_state", state)
        from .instruction_parameters import inherit_control_binding_plan
        inherit_control_binding_plan(base_gate, self)


def _append_controlled(out, name, wires, params, controls, metadata):
    params = params if isinstance(params, tuple) else (params,)
    if not controls:
        out.append(name, wires, params=params, metadata=metadata)
        return
    names = {"x": "cx", "y": "cy", "z": "cz", "h": "ch", "sx": "csx", "sxdg": "csxdg",
             "rx": "crx", "ry": "cry", "rz": "crz", "p": "cp", "phase": "cp", "u1": "cp"}
    if len(controls) == 1 and len(wires) == 1 and name in names:
        out.append(names[name], wires, controls=controls, params=params)
    else:
        width = len(controls)+len(wires)
        if width > current_gate_limits().max_matrix_dimension.bit_length()-1:
            raise MemoryError("Controlled primitive matrix exceeds max_matrix_dimension.")
        out.append("native_controlled", wires, controls=controls, params=params,
                   metadata={"controlled_base": {"name": name, "metadata": _copy(metadata), "controls": len(controls)}})


def decompose(circuit, selected=None, reps=1, *, _work=None, _standard=True, _annotation_resolver=None):
    from itertools import islice
    from .circuit import Circuit, _remap_operation
    from .operations import instruction_definition_work
    from .measurements import Measurement
    from quantumbridge.compiler.traversal import append_item
    reps = _integer(reps, "Decomposition repetitions")
    limits = current_gate_limits()
    limits.check("max_decomposition_work", max(0, reps))
    if selected is None:names=None
    elif isinstance(selected,(str,type)):names=(selected,)
    else:
        if type(selected) in (list,tuple):limits.check("max_decomposition_work",len(selected))
        names=tuple(islice(iter(selected),limits.max_decomposition_work+1))
        limits.check("max_decomposition_work",len(names))
    instruction_definition_work(circuit,_resolve_frames=False)
    def matches(gate):
        return names is None or any(isinstance(gate, value) if isinstance(value, type) else
            any(fnmatch.fnmatchcase(text, value) for text in (gate.name, gate.label) if text is not None) for value in names)
    def definition_for(item):
        item=execution_operation(item)
        gate = item.metadata.get("gate")
        if "instruction_object" in item.metadata:
            from .operations import instruction_definition_work
            instruction = item.metadata["instruction_object"]
            if type(instruction) is not Instruction or gate is not None or "annotated_operation" in item.metadata:
                raise ValueError("Invalid native instruction reference type.")
            if (item.name != instruction.name or item.controls or len(item.targets) != instruction.num_qubits
                    or tuple(item.params) != tuple(instruction.params)
                    or len(item.metadata.get("instruction_clbits", ())) != instruction.num_bits):
                raise ValueError("Instruction reference disagrees with its owner.")
            if instruction.definition is not None: instruction_definition_work(instruction.definition, owner=instruction)
            return instruction
        if "annotated_operation" in item.metadata:
            from .annotated_operation import AnnotatedOperation, resolve_annotated_operation
            from .standard_gates import invocation_params
            annotation = item.metadata["annotated_operation"]
            if (type(annotation) is not AnnotatedOperation or gate is not None or item.name != "annotated"
                    or item.controls or len(item.targets) != annotation.num_qubits or item.params != invocation_params(annotation)):
                raise ValueError("Annotated invocation disagrees with its operation.")
            if names is not None and not any(isinstance(annotation, value) if isinstance(value, type) else
                    fnmatch.fnmatchcase(annotation.name, value) for value in names): return None
            return resolve_annotated_operation(annotation) if _annotation_resolver is None else _annotation_resolver(annotation)
        if gate is None and _standard:
            from quantumbridge.circuit_library.standard_gates import _DEFINITION_GATES, standard_gate_object
            if item.name in _DEFINITION_GATES and not any(key in item.metadata for key in ("controlled_base", "matrix", "directive")):
                candidate = Gate(item.name, len(item.wires), item.params, item.metadata.get("label"))
                if matches(candidate):
                    gate = standard_gate_object(item.name, *item.params, label=candidate.label)
                    if gate.num_qubits != len(item.wires): raise ValueError("Standard definition invocation width differs.")
        return gate
    limits.check("max_total_operations", len(circuit.operations))
    result = _copy(circuit)
    work = [0] if _work is None else _work
    for _ in range(max(0, reps)):
        predicted = 0
        definitions = {}
        for item in result.operations:
            gate = definition_for(item)
            definitions[id(item)] = gate
            selected_gate = "annotated_operation" in item.metadata or gate is not None and matches(gate)
            predicted += len(gate.definition.operations) if gate is not None and gate.definition is not None and selected_gate else 1
        limits.check("max_total_operations", predicted)
        from .circuit_metadata import empty_execution_circuit
        out = empty_execution_circuit(result)
        from .bits import copy_bits
        copy_bits(result, out)
        changed = False
        for item in result.iter_instructions():
            work[0] += 1
            limits.check("max_decomposition_work", work[0])
            gate = None if isinstance(item, Measurement) else definitions[id(item)]
            if (type(gate) is Instruction and matches(gate)
                    and (gate.definition is not None or gate.name in {"measure", "reset", "barrier", "delay"})):
                extra = {key: value for key, value in item.metadata.items() if key not in {"instruction_object", "instruction_clbits", "label"}}
                if gate.definition is not None and any(key in extra for key in ("condition", "gate_condition", "control_flow_stack")) and (
                        parameter_set((gate.definition.global_phase,)) or gate.definition.global_phase != 0):
                    from .classical import UnsupportedFeature
                    raise UnsupportedFeature("conditional_instruction_phase")
                out._append_instruction(gate, item.targets, clbits=item.metadata["instruction_clbits"], metadata=extra)
                changed = True
                continue
            if gate is None or gate.definition is None or "annotated_operation" not in item.metadata and not matches(gate):
                append_item(out, item)
                continue
            validate_definition(gate.definition, owner=gate)
            limits.check("max_total_operations", len(out.operations)+len(gate.definition.operations))
            extra = {key: value for key, value in item.metadata.items() if key not in {"gate", "label", "annotated_operation"}}
            if any(key in extra for key in ("condition", "gate_condition", "control_flow_stack")) and (
                    parameter_set((gate.definition.global_phase,)) or gate.definition.global_phase != 0):
                from .classical import UnsupportedFeature
                raise UnsupportedFeature("conditional_gate_phase", "Conditional definition phase needs a scoped execution node.")
            out.global_phase = out.global_phase + gate.definition.global_phase
            mapping = dict(enumerate(item.wires))
            for child in gate.definition.operations:
                mapped = _remap_operation(child, mapping, {}, out.num_qubits, out.num_bits)
                mapped.metadata.update(_copy(extra))
                out.operations.append(mapped)
            changed = True
        limits.check("max_total_operations", len(out.operations))
        result = out
        if not changed: break
    return result


def lower(circuit, *, _ancestors=(), _work=None, _annotation_resolver=None):
    if id(circuit) in _ancestors: raise ValueError("Cyclic circuit/control definition.")
    current_gate_limits().check("max_definition_depth", len(_ancestors))
    work = [0] if _work is None else _work
    from .control_flow_operations import operation_owner
    if any(getattr(op,"_public_matrix_basis",None)=="little" or operation_owner(op) is not None
           for op in circuit.operations):
        from .circuit_metadata import empty_execution_circuit
        result=empty_execution_circuit(circuit)
        result.operations=[execution_operation(op) for op in circuit.operations]
        result.measurements=_copy(circuit.measurements)
        circuit=result
    if any("control_node" in op.metadata for op in circuit.operations):
        from .control_flow import ControlBlock
        ancestors = (*_ancestors, id(circuit))
        transformed = []
        changed = False
        for op in circuit.operations:
            node = op.metadata.get("control_node")
            if node is None:
                transformed.append(op)
                continue
            replacements = {id(body): lower(body, _ancestors=ancestors, _work=work, _annotation_resolver=_annotation_resolver) for body in node.bodies}
            if all(replacements[id(body)] is body for body in node.bodies): transformed.append(op)
            else:
                data = dict(node.data)
                for field in ("body", "alternate", "default"):
                    if data.get(field) is not None: data[field] = replacements[id(data[field])]
                if "cases" in data:
                    data["cases"] = tuple((values, replacements[id(body)]) for values, body in data["cases"])
                transformed.append(Operation(op.name, op.targets, op.controls, op.params,
                    {**_copy(op.metadata), "control_node": ControlBlock(node.kind, data)}))
                changed = True
        if changed:
            circuit = _copy(circuit)
            circuit.operations = transformed
    if not any(set(op.metadata) & {"gate", "annotated_operation", "instruction_object"} for op in circuit.operations): return circuit
    result = circuit
    limits = current_gate_limits()
    # Definitions are acyclic and depth bounded before entering expansion.
    for op in circuit.operations:
        if "instruction_object" in op.metadata and op.metadata["instruction_object"].definition is not None:
            from .operations import instruction_definition_work
            instruction_definition_work(op.metadata["instruction_object"].definition, owner=op.metadata["instruction_object"])
        if "gate" in op.metadata and op.metadata["gate"].definition is not None:
            validate_definition(op.metadata["gate"].definition, owner=op.metadata["gate"])
    result = decompose(result, reps=limits.max_definition_depth+1, _work=work, _standard=False, _annotation_resolver=_annotation_resolver)
    if any(set(op.metadata) & {"gate", "annotated_operation", "instruction_object"} for op in result.operations):
        from .classical import UnsupportedFeature
        raise UnsupportedFeature("opaque_gate_definition", "An opaque Gate needs an executable definition.")
    if any("control_node" in op.metadata for op in result.operations):
        return lower(result, _ancestors=(*_ancestors, id(circuit)), _work=work, _annotation_resolver=_annotation_resolver)
    return result
