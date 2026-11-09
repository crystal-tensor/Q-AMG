# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.
# Design source: docs/mvp/api_contract_v0.1.md and docs/mvp/simulator_design_v0.1.md.

from __future__ import annotations

from collections.abc import MutableSequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .parameters import ParameterValue,parameter_set

_UNSET = object()


def instruction_definition_work(circuit, *, owner=None, params=None, _resolve_frames=True):
    """Bound a definition graph without requiring its operations to be unitary."""
    from .circuit import Circuit,is_native_circuit
    from .control_flow import ControlBlock
    from .gate_limits import current_gate_limits
    from .instruction_parameters import admit_owner, admit_payloads
    from .gates import is_native_instruction
    from .annotated_operation import _inspect
    limits = current_gate_limits()
    from quantumbridge.execution.resource_context import current_resources
    managed = current_resources() is not None
    stack = [(circuit, 0, () if owner is None else (owner,))]
    total = definitions = 0
    while stack:
        body, depth, ancestors = stack.pop()
        if not is_native_circuit(body): raise TypeError("An instruction definition needs a native Circuit.")
        from .circuit_metadata import execution_metadata
        if managed and (type(body.operations) is not list or type(body.measurements) is not list or type(execution_metadata(body)) is not dict):
            from .classical import UnsupportedFeature
            raise UnsupportedFeature("managed_instruction_definition_container")
        if any(body is ancestor for ancestor in ancestors): raise ValueError("Cyclic instruction definition.")
        limits.check("max_definition_depth", depth)
        definitions += 1
        limits.check("max_custom_definitions", definitions)
        total += len(body.operations)+len(body.measurements)
        limits.check("max_total_operations", total)
        limits.check("max_decomposition_work", total+definitions)
        ancestors = ancestors+(body,)
        for operation in body.operations:
            if managed and (type(operation) is not Operation or type(operation.metadata) is not dict):
                from .classical import UnsupportedFeature
                raise UnsupportedFeature("managed_instruction_definition_operation")
            if _resolve_frames:operation = execution_operation(operation)
            node = operation.metadata.get("control_node")
            if node is not None:
                if type(node) is not ControlBlock: raise TypeError("Unknown native control definition.")
                if managed and type(node.data) is not dict:
                    from .classical import UnsupportedFeature
                    raise UnsupportedFeature("managed_instruction_control_data")
                stack.extend((child, depth+1, ancestors) for child in node.bodies)
            child = operation.metadata.get("gate", operation.metadata.get("instruction_object"))
            annotation = operation.metadata.get("annotated_operation")
            if sum(operation.metadata.get(key) is not None for key in ("gate", "instruction_object", "annotated_operation")) > 1:
                raise ValueError("An invocation cannot have multiple operation owners.")
            if annotation is not None: child = _inspect(annotation)[0]
            if child is not None:
                if not is_native_instruction(child): raise TypeError("Unknown native instruction definition.")
                admit_owner(child)
                if any(child is ancestor for ancestor in ancestors): raise ValueError("Cyclic instruction definition.")
                from .standard_gates import invocation_params
                actual_owner=annotation if annotation is not None else child
                actual_params=invocation_params(actual_owner)
                admit_payloads(actual_params)
                if (operation.controls
                        or len(operation.targets) != actual_owner.num_qubits
                        or (_resolve_frames or not has_public_frame(operation)) and (
                            operation.name != actual_owner.name or tuple(operation.params) != actual_params)):
                    raise ValueError("Instruction definition invocation differs from its operation owner.")
                definition = child.definition
                if definition is not None:
                    if not is_native_circuit(definition):raise TypeError("An instruction definition needs a native Circuit.")
                    if (definition.num_qubits,definition.num_bits)!=(child.num_qubits,child.num_bits):
                        raise ValueError("Instruction definition width differs from its owner.")
                    stack.append((definition, depth+1, ancestors+(child,)))
    if owner is not None:
        from .parameters import parameter_set
        admit_payloads(tuple(owner.params if params is None else params))
        if not set(circuit.parameters) <= set(parameter_set(owner.params if params is None else params)):
            raise ValueError("Instruction definition parameter namespace differs from its owner.")
    return total


@dataclass(init=False)
class Instruction:
    """Qiskit-style instruction object with optional circuit definition."""

    name: str
    num_qubits: int
    num_bits: int = 0
    params: MutableSequence[Any] = ()
    definition: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    label: str | None = None

    def __init__(self, name, num_qubits, num_clbits=_UNSET, params=(), label=None, *, definition=None, metadata=None, num_bits=_UNSET):
        from .instruction_parameters import admit_owner, admit_payloads
        admit_owner(self)
        if num_clbits is not _UNSET and num_bits is not _UNSET:
            raise TypeError("Classical width was provided by both num_clbits and num_bits.")
        num_bits = num_clbits if num_clbits is not _UNSET else 0 if num_bits is _UNSET else num_bits
        admit_payloads((name, num_qubits, num_bits, label))
        if label is not None and not isinstance(label, str): raise TypeError("Instruction label must be a string or None.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "num_qubits", num_qubits)
        object.__setattr__(self, "num_bits", num_bits)
        object.__setattr__(self, "_params", ())
        object.__setattr__(self, "_definition", None)
        object.__setattr__(self, "_parameter_plan", None)
        object.__setattr__(self, "metadata", dict(metadata or {}))
        object.__setattr__(self, "label", label)
        self.__post_init__()
        from .instruction_parameters import validated_values
        object.__setattr__(self, "_params", validated_values(self, params))
        self.definition = definition

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("QuantumBridge instructions require a non-empty name.")
        from .errors import CircuitError
        if not isinstance(self.num_qubits, int) or not isinstance(self.num_bits, int):
            raise CircuitError("Instruction quantum and classical widths must be integers.")
        if self.num_qubits < 0 or self.num_bits < 0:
            raise CircuitError("Instruction quantum and classical widths cannot be negative.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def __setattr__(self, name, value):
        if name in {"params", "definition"}: return object.__setattr__(self, name, value)
        if name in {"name", "label"}:
            if value is None and name == "label": return object.__setattr__(self, name, value)
            if not isinstance(value, str) or name == "name" and not value: raise ValueError("Instruction name/label must be valid strings.")
            return object.__setattr__(self, name, value)
        raise AttributeError("Use params/definition assignment or copy to update a native Instruction.")

    @property
    def base_class(self): return type(self)

    @property
    def mutable(self): return getattr(self, "_mutable", True)

    def to_mutable(self):
        result = self.copy()
        object.__setattr__(result, "_mutable", True)
        return result

    @property
    def num_clbits(self): return self.num_bits

    @property
    def params(self):
        from .instruction_parameters import ParameterList
        return ParameterList(self)

    @params.setter
    def params(self, values):
        from .instruction_parameters import validated_values, prepare_definition_update
        values = validated_values(self, values)
        self._validate_parameter_arity(values)
        definition, plan = prepare_definition_update(self, values)
        metadata = self.metadata
        from .instruction_parameters import same_values
        if definition is not None and "global_phase" in metadata and same_values((metadata["global_phase"],), (self.definition.global_phase,)):
            metadata = {**metadata, "global_phase": definition.global_phase}
        object.__setattr__(self, "_params", values)
        object.__setattr__(self, "_definition", definition)
        object.__setattr__(self, "_parameter_plan", plan)
        object.__setattr__(self, "metadata", metadata)

    def validate_parameter(self, parameter): return parameter

    def _validate_parameter_arity(self, values): pass

    def is_parameterized(self):
        from .parameters import parameter_set
        return bool(parameter_set(self.params))

    def _validate_definition(self, value, params):
        from .circuit import Circuit,is_native_circuit
        from .instruction_parameters import _managed
        if _managed() and not is_native_circuit(value):
            from .classical import UnsupportedFeature
            raise UnsupportedFeature("managed_instruction_definition")
        if not isinstance(value, Circuit): raise TypeError("An Instruction definition requires a native Circuit.")
        if (value.num_qubits, value.num_bits) != (self.num_qubits, self.num_bits):
            raise ValueError("Instruction definition width differs from its owner.")
        instruction_definition_work(value, owner=self, params=params)

    @property
    def definition(self): return self._definition

    @definition.setter
    def definition(self, value):
        if value is not None: self._validate_definition(value, self._params)
        from .instruction_parameters import owned_copy,same_values
        value = owned_copy(value)
        metadata=self.metadata;previous=self._definition
        # Keep the conversion-derived name synchronized, not unrelated user data.
        if (previous is not None and "definition_name" in metadata
                and same_values((metadata["definition_name"],),(previous.name,))):
            metadata=dict(metadata)
            if value is None:metadata.pop("definition_name")
            else:metadata["definition_name"]=value.name
        object.__setattr__(self, "_definition", value)
        object.__setattr__(self, "_parameter_plan", None)
        object.__setattr__(self,"metadata",metadata)

    def copy(self, name: str | None = None) -> "Instruction":
        from quantumbridge.execution.array_ownership import native_configuration_copy
        from quantumbridge.execution.resource_context import current_resources
        copied = native_configuration_copy(self, category="native_instruction_copy",
            strict=current_resources() is not None)
        if name is not None:
            if not isinstance(name, str) or not name: raise ValueError("An Instruction copy needs a nonempty name.")
            object.__setattr__(copied, "name", name)
        return copied

    def assign_parameters(self, mapping, *, strict=True):
        from .circuit import _parameter_mapping_from_values
        from .parameters import parameter_set, parameter_sort_key, resolve_parameter
        from .instruction_parameters import admit_owner, inherit_binding_plan, owned_copy,bind_definition
        admit_owner(self)
        values = _parameter_mapping_from_values(tuple(sorted(parameter_set(self.params), key=parameter_sort_key)), mapping, strict)
        if self.definition is not None: instruction_definition_work(self.definition, owner=self)
        metadata = owned_copy(self.metadata)
        if "global_phase" in metadata:
            metadata["global_phase"] = resolve_parameter(metadata["global_phase"], values, allow_partial=True)
        result = Instruction(self.name, self.num_qubits, self.num_bits,
            tuple(resolve_parameter(value, values, allow_partial=True) for value in self.params), self.label,
            definition=None if self.definition is None else bind_definition(self.definition,values), metadata=metadata)
        inherit_binding_plan(self, result)
        return result

    def inverse(self, annotated=False, *, name=None):
        from .instruction_parameters import admit_owner, admit_payloads
        from .errors import CircuitError
        admit_owner(self)
        admit_payloads((annotated, name))
        if annotated:
            from .annotated_operation import AnnotatedOperation, InverseModifier
            return AnnotatedOperation(self, InverseModifier())
        output_name = name or f"{self.name}_dg"
        if self.definition is not None:
            instruction_definition_work(self.definition, owner=self)
            try:
                body = self.definition.inverse()
                return body.to_instruction(name=output_name) if self.num_bits else body.to_gate(_name=output_name)
            except (TypeError, ValueError) as exc:
                raise CircuitError("Instruction inverse is not implemented for this definition.") from exc
        if "matrix" in self.metadata and not self.num_bits:
            from .annotated_operation import _base_gate
            return _base_gate(self).inverse().copy(name=output_name)
        raise CircuitError(f"inverse() is not implemented for {self.name}.")

    def repeat(self, n):
        from .instruction_parameters import admit_owner, admit_payloads
        from .gate_limits import current_gate_limits
        from .errors import CircuitError
        from .circuit import Circuit
        admit_owner(self)
        admit_payloads((n,))
        count = int(n)
        if count < 1 or count != n: raise CircuitError("Repeat requires a strictly positive integral count.")
        limits = current_gate_limits()
        limits.check("max_repeat_count", count)
        work = 1 if self.definition is None else instruction_definition_work(self.definition, owner=self)
        predicted = count*(1+work if self.definition is not None else 1)
        limits.check("max_total_operations", predicted)
        limits.check("max_decomposition_work", predicted)
        body = Circuit(self.num_qubits, self.num_bits)
        for _ in range(count): body._append_instruction_reference(self, range(self.num_qubits), range(self.num_bits))
        return Instruction(f"{self.name}*{count}", self.num_qubits, self.num_bits,
            tuple(self.params), definition=body)

    def power(self, exponent: int, name: str | None = None) -> "Instruction":
        if not isinstance(exponent, int):
            raise ValueError("QuantumBridge Instruction.power requires an integer exponent.")
        if self.definition is not None:
            return self.definition.power(exponent).to_instruction(name=name or f"{self.name}^{exponent}")
        from quantumbridge.core.circuit import Circuit

        circuit = Circuit(self.num_qubits, self.num_bits)
        unit = self.inverse() if exponent < 0 else self
        for _ in range(abs(exponent)):
            circuit.append(unit, range(self.num_qubits), clbits=range(self.num_bits))
        return circuit.to_instruction(name=name or f"{self.name}^{exponent}")

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: int | None = None,
    ) -> "Instruction":
        if self.definition is not None:
            return self.definition.control(num_ctrl_qubits, label=label or f"c{num_ctrl_qubits}_{self.name}", ctrl_state=ctrl_state).to_instruction(
                name=label or f"c{num_ctrl_qubits}_{self.name}"
            )
        from quantumbridge.core.circuit import Circuit

        circuit = Circuit(self.num_qubits, self.num_bits).append(self, range(self.num_qubits), clbits=range(self.num_bits))
        return circuit.control(num_ctrl_qubits, label=label or f"c{num_ctrl_qubits}_{self.name}", ctrl_state=ctrl_state).to_instruction(
            name=label or f"c{num_ctrl_qubits}_{self.name}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "num_bits": self.num_bits,
            "params": list(self.params),
            "has_definition": self.definition is not None,
            "metadata": deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class Operation:
    """Value object describing a QuantumBridge MVP operation."""

    name: str
    targets: tuple[int, ...]
    controls: tuple[int, ...] = ()
    params: tuple[ParameterValue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __deepcopy__(self, memo):
        fields=object.__getattribute__(self,"__dict__")
        if "_native_inverse_owner" in fields or "_native_inverse_qualification" in fields:
            from quantumbridge.execution.array_ownership import native_object_deepcopy
            return native_object_deepcopy(self,memo)
        copied=object.__new__(type(self))
        memo[id(self)]=copied
        object.__getattribute__(copied,"__dict__").update(deepcopy(fields,memo))
        return copied

    def __getattribute__(self, name):
        if name in {"metadata", "params", "name"}:
            control = object.__getattribute__(self, "__dict__").get("_control_owner")
            if control is not None:
                if name == "metadata":
                    from .control_flow_operations import refresh_control
                    return refresh_control(self)
                if name == "name":
                    return control.name
                return tuple(getattr(control, "parameters", ()))
        if name in {"params", "name"}:
            if has_public_frame(self):
                return object.__getattribute__(self,name)
            owner = object.__getattribute__(self,"__dict__").get("_shared_owner")
            if owner is not None:
                if name == "name": return owner.name
                from .standard_gates import invocation_params
                return invocation_params(owner)
        return object.__getattribute__(self,name)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("QuantumBridge operations require a non-empty name.")
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "controls", tuple(self.controls))
        object.__setattr__(self, "params", tuple(self.params))
        owners = [self.metadata[key] for key in ("gate","instruction_object","annotated_operation") if key in self.metadata]
        if len(owners) == 1:
            from .gates import is_native_instruction
            from .annotated_operation import AnnotatedOperation
            owner = owners[0]
            if is_native_instruction(owner) or type(owner) is AnnotatedOperation:
                from .standard_gates import invocation_params
                from .instruction_parameters import admit_payloads
                values = invocation_params(owner)
                admit_payloads(values)
                if self.name == owner.name and self.params == values:
                    object.__setattr__(self,"_shared_owner",owner)

    @property
    def wires(self) -> tuple[int, ...]:
        return self.controls + self.targets


def has_public_frame(operation):
    return object.__getattribute__(operation,"__dict__").get("_public_frame") is True


def capture_public_frame(operation,name,params):
    """Keep invocation slots in the existing Operation, not another owner/list."""
    from .instruction_parameters import admit_payloads
    admit_payloads((name,params))
    from .gate_limits import current_gate_limits
    current_gate_limits().check("max_gate_parameters",len(params))
    object.__setattr__(operation,"_public_frame",True)
    object.__setattr__(operation,"name",name)
    object.__setattr__(operation,"params",tuple(params))
    return operation


def public_matrix_frame(operation):
    """Follow explicit public matrix provenance through existing owner copies."""
    basis=object.__getattribute__(operation,"__dict__").get("_public_matrix_basis")
    if basis is not None and (type(basis) is not str or basis not in ("little","native")):
        raise ValueError("Unknown local matrix basis provenance.")
    if basis=="native":return False
    owner=operation.metadata.get("gate",operation.metadata.get("annotated_operation"))
    return _matrix_owner_family(owner,public_unmarked=has_public_frame(operation))


def _matrix_owner_family(owner,*,public_unmarked=False):
    from .gates import Gate,ControlledGate
    from .standard_gates import UnitaryGate
    from .annotated_operation import AnnotatedOperation
    from .gate_limits import current_gate_limits
    limits=current_gate_limits()
    memo,active={},set()
    def visit(item,depth,unmarked):
        identity=id(item)
        cache_key=(identity,unmarked)
        if identity in active:raise ValueError("Cyclic public matrix owner.")
        limits.check("max_definition_depth",depth)
        if cache_key in memo:
            value=memo[cache_key]
            limits.check("max_definition_depth",depth+value[3])
            return value
        active.add(identity)
        try:
            if type(item) in (UnitaryGate,Gate):
                fields=object.__getattribute__(item,"__dict__")
                params=fields.get("_params")
                if type(params) is not tuple:raise ValueError("Matrix owner needs a native parameter tuple.")
                limits.check("max_gate_parameters",len(params))
                basis=fields.get("_public_matrix_basis")
                if basis is not None and (type(basis) is not str or basis not in ("little","native","literal_wrapper")):
                    raise ValueError("Unknown matrix owner basis.")
            if type(item) is UnitaryGate:
                if basis not in (None,"little","native"):raise ValueError("Unknown matrix owner basis.")
                from numpy import ndarray
                from quantumbridge.execution.array_reservations import numeric_shape
                if len(params)!=1 or type(params[0]) is not ndarray or type(fields.get("num_qubits")) is not int:
                    raise ValueError("Unitary owner has an invalid parameter/width record.")
                if numeric_shape(params[0])!=(1<<item.num_qubits,)*2:
                    raise ValueError("Unitary owner matrix differs from its native width.")
                limits.check("max_matrix_dimension",1<<item.num_qubits)
                value=(True,basis=="little" or basis is None and unmarked,1,0)
            elif type(item) in (ControlledGate,AnnotatedOperation):
                child=item.base_gate if type(item) is ControlledGate else item.base_op
                known,needed,work,height=visit(child,depth+1,unmarked)
                value=(known,needed,work+1,height+1)
            elif type(item) is Gate and not params:
                body=item.definition
                if body is not None:
                    from .circuit import is_native_circuit
                    if not is_native_circuit(body):raise ValueError("Matrix owner definition must be a native Circuit.")
                    if type(body.operations) not in (list,tuple) or type(body.measurements) not in (list,tuple):
                        raise ValueError("Matrix definition needs native operation containers.")
                if body is None or not body.operations or body.measurements:value=(False,False,0,0)
                else:
                    work,height,needed=len(body.operations),0,False
                    limits.check("max_total_operations",work)
                    known=True
                    child_context=unmarked and basis=="literal_wrapper"
                    for op in body.operations:
                        if type(op) is not Operation or type(op.metadata) is not dict:
                            raise ValueError("Matrix family needs native operation records.")
                        child=op.metadata.get("gate",op.metadata.get("annotated_operation"))
                        child_known,child_needed,child_work,child_height=visit(child,depth+1,child_context)
                        if not child_known:known=False;break
                        # Memoized analysis still charges every expanded use.
                        work+=child_work;height=max(height,child_height+1);needed|=child_needed
                        limits.check("max_total_operations",work)
                        limits.check("max_decomposition_work",work)
                    value=(known,known and needed,work,height)
            else:value=(False,False,0,0)
            limits.check("max_total_operations",value[2])
            limits.check("max_decomposition_work",value[2])
            limits.check("max_custom_definitions",len(memo)+1)
            memo[cache_key]=value
            return value
        finally:active.remove(identity)
    return visit(owner,0,public_unmarked)[1]


def _native_matrix_snapshot(owner,*,public_unmarked=False):
    from .gates import Gate,ControlledGate
    from .standard_gates import UnitaryGate
    from .annotated_operation import AnnotatedOperation
    if type(owner) is UnitaryGate:
        basis=getattr(owner,"_public_matrix_basis",None)
        if basis=="native" or basis is None and not public_unmarked:return owner.copy()
        from quantumbridge.information.operator import _reverse_qubit_order
        # Existing owned-array admission applies to both local representations.
        result=UnitaryGate(_reverse_qubit_order(owner.to_matrix(),owner.num_qubits),label=owner.label)
        object.__setattr__(result,"_public_matrix_basis","native")
        return result
    if type(owner) is ControlledGate:
        result=ControlledGate(_native_matrix_snapshot(owner.base_gate,public_unmarked=public_unmarked),owner.num_ctrl_qubits,
            owner.ctrl_state,owner.label)
        result._restore_display_name(owner.name)
        return result
    if type(owner) is AnnotatedOperation:
        from .instruction_parameters import owned_copy
        return AnnotatedOperation(_native_matrix_snapshot(owner.base_op,public_unmarked=public_unmarked),owned_copy(owner.modifiers))
    if type(owner) is Gate and _matrix_owner_family(owner,public_unmarked=public_unmarked):
        result=owner.copy()
        body=result.definition
        if public_unmarked and getattr(owner,"_public_matrix_basis",None)=="literal_wrapper":
            for op in body.operations:capture_public_frame(op,op.name,op.params)
        body.operations=[execution_operation(op) for op in body.operations]
        return result
    raise ValueError("A public matrix snapshot needs a known native matrix owner.")


def execution_operation(operation,*,standardize_inverse=False,serialization_limits=None):
    """Resolve executable slots without replacing the exposed source owner."""
    fields=object.__getattribute__(operation,"__dict__")
    if fields.get("_control_owner") is not None:
        return Operation(operation.name, operation.targets, operation.controls,
                         operation.params, dict(operation.metadata))
    if "_native_inverse_qualification" in fields or "_native_inverse_owner" in fields:
        from .unitary_inverse import canonical_inverse_matrix,inverse_matrix_owner
        if standardize_inverse:return canonical_inverse_matrix(operation,serialization_limits)
        inverse_matrix_owner(operation)
        return operation
    basis=object.__getattribute__(operation,"__dict__").get("_public_matrix_basis")
    if basis is not None and (type(basis) is not str or basis not in ("little","native")):
        raise ValueError("Unknown local matrix basis provenance.")
    if basis=="native":return operation
    if basis=="little":
        from .standard_gates import UnitaryGate
        from quantumbridge.information.operator import _reverse_qubit_order
        if (operation.controls or operation.params or not set(operation.metadata)<={"matrix","label"}
                or "matrix" not in operation.metadata):
            raise ValueError("A public matrix definition requires a checked local matrix payload.")
        from quantumbridge.execution.array_reservations import numeric_shape
        from .gate_limits import current_gate_limits
        shape=numeric_shape(operation.metadata["matrix"])
        if shape!=(1<<len(operation.targets),)*2:
            raise ValueError("Public matrix definition width differs from its operands.")
        current_gate_limits().check("max_matrix_dimension",shape[0])
        owner=UnitaryGate(operation.metadata["matrix"])
        matrix=_reverse_qubit_order(owner.to_matrix(),owner.num_qubits)
        result=Operation(operation.name,operation.targets,metadata={**operation.metadata,"matrix":matrix})
        object.__setattr__(result,"_public_matrix_basis","native")
        return result
    if not has_public_frame(operation) and not public_matrix_frame(operation):return operation
    from .standard_gates import standard_type_name,standard_from_record,invocation_params
    from .gates import Gate,ControlledGate,is_native_instruction
    from .annotated_operation import AnnotatedOperation,validate_annotated_structure,_inspect
    from .instruction_parameters import same_values,admit_owner,admit_payloads
    key=next((key for key in ("gate","instruction_object","annotated_operation") if key in operation.metadata),"gate")
    owner=operation.metadata.get(key)
    annotation=type(owner) is AnnotatedOperation
    if annotation:validate_annotated_structure(owner)
    else:admit_owner(owner)
    if (not (is_native_instruction(owner) or annotation) or operation.controls
            or len(operation.targets)!=owner.num_qubits
            or len(operation.metadata.get("instruction_clbits",()))!=owner.num_clbits
            or sum(k in operation.metadata for k in ("gate","instruction_object","annotated_operation"))!=1):
        raise ValueError("Public execution frame width/ownership differs from its operation.")
    if public_matrix_frame(operation):
        snapshot=_native_matrix_snapshot(owner,public_unmarked=has_public_frame(operation))
        # The temporary native snapshot is unframed, making resolution idempotent.
        result=Operation(snapshot.name,operation.targets,operation.controls,
            invocation_params(snapshot),{**operation.metadata,key:snapshot})
        object.__setattr__(result,"_public_matrix_basis","native")
        return result
    kind=standard_type_name(owner)
    if kind=="UnitaryGate":return operation
    if kind is not None and kind!="UnitaryGate":
        if operation.name==owner.name and same_values(operation.params,invocation_params(owner)):return operation
        snapshot=standard_from_record(kind,operation.params,label=operation.metadata.get("label"),ctrl_state=getattr(owner,"ctrl_state",None))
    elif type(owner) in (Gate,ControlledGate,Instruction,AnnotatedOperation):
        admit_payloads(invocation_params(owner))
        changed=not same_values(operation.params,invocation_params(owner))
        if changed:
            base=_inspect(owner)[0] if annotation else owner
            if base.definition is not None:
                instruction_definition_work(base.definition,owner=base,
                    params=(*invocation_params(base),*operation.params),_resolve_frames=False)
            snapshot=owner.copy()
            copied_base=_inspect(snapshot)[0] if annotation else snapshot
            if (type(copied_base) is ControlledGate and copied_base.definition is not None
                    and not set(copied_base.definition.parameters)<=set(parameter_set(copied_base.params))):
                copied_base.base_gate.params=operation.params
            snapshot.params=operation.params
        else:snapshot=owner
        if not changed and operation.name==owner.name and operation.metadata.get("label")==getattr(owner,"label",None):return operation
    else:raise ValueError("Invalid native public execution frame owner.")
    metadata={**operation.metadata,key:snapshot}
    if kind is None:
        metadata.pop("label",None)
        if getattr(snapshot,"label",None) is not None:metadata["label"]=snapshot.label
    result=Operation(snapshot.name,operation.targets,operation.controls,invocation_params(snapshot),metadata)
    return capture_public_frame(result,result.name,result.params)
