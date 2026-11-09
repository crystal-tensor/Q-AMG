# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.
# Design source: docs/mvp/api_contract_v0.1.md, docs/mvp/ir_design_v0.1.md,
# and docs/mvp/simulator_design_v0.1.md.

from __future__ import annotations

from collections.abc import Mapping as ABCMapping
from copy import deepcopy
from numbers import Real
from typing import Any, Callable, Mapping, Optional, Sequence, Union
from uuid import uuid4

import numpy as np

from .measurements import Measurement
from .operations import Instruction, Operation
from .parameters import (
    Parameter, ParameterExpression, ParameterValue, ParameterVector, ParameterVectorElement,
    parameter_set, parameter_sort_key, resolve_parameter, validate_parameter_names,
)
from quantumbridge.utils.math import gate_matrix
from .circuit_metadata import execution_metadata, is_separated, inherit_execution, empty_execution_circuit

_MISSING = object()


class Circuit:
    """Ordered QuantumBridge MVP circuit."""

    def __init__(
        self,
        num_qubits: int = 0,
        num_bits: int = 0,
        name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        from .bits import initialize_bits
        initialize_bits(self,num_qubits,num_bits)
        self.name = name
        self.metadata = metadata if is_separated(self) and metadata is not None else dict(metadata or {})
        self.operations: list[Operation] = []
        self.measurements: list[Measurement] = []

    @property
    def qubits(self):return list(self._qubits)
    @property
    def clbits(self):return list(self._clbits)
    @property
    def qregs(self):return list(self._qregs)
    @property
    def cregs(self):return list(self._cregs)
    @property
    def num_clbits(self):return self.num_bits
    @property
    def ancillas(self):
        from .bits import AncillaQubit
        return [bit for bit in self._qubits if type(bit) is AncillaQubit]
    @property
    def num_ancillas(self):return len(self.ancillas)

    def add_bits(self,bits):
        from .bits import add_bits
        return add_bits(self,bits)

    def add_register(self,*registers):
        from .bits import add_register
        return add_register(self,*registers)

    def find_bit(self,bit):
        from .bits import find_bit
        return find_bit(self,bit)

    @property
    def global_phase(self): return execution_metadata(self).get("global_phase", 0.)

    @global_phase.setter
    def global_phase(self, phase):
        if not isinstance(phase, (Real, Parameter)) and not hasattr(phase, "parameters"):
            raise TypeError("Circuit global_phase requires a real or native parameter expression.")
        if isinstance(phase, Real) and not np.isfinite(phase): raise ValueError("Circuit global_phase must be finite.")
        parameters=set(parameter_set((phase,)))
        if parameters:
            for op in self.operations:
                parameters.update(parameter_set(op.params))
                if "control_node" in op.metadata:parameters.update(op.metadata["control_node"].parameters)
            validate_parameter_names(parameters)
        execution_metadata(self)["global_phase"] = phase

    def _check_wire(self, wire: int) -> None:
        if not isinstance(wire, int) or not 0 <= wire < self.num_qubits:
            raise ValueError(f"QuantumBridge wire {wire!r} is outside this circuit.")

    def _check_bit(self, bit: int) -> None:
        if not isinstance(bit, int) or not 0 <= bit < self.num_bits:
            raise ValueError(f"QuantumBridge classical bit {bit!r} is outside this circuit.")

    def append(
        self,
        name: Union[str, Instruction],
        targets: Sequence[int],
        controls: Sequence[int] = (),
        params: Sequence[ParameterValue] = (),
        metadata: Optional[Mapping[str, Any]] = None,
        clbits: Optional[Sequence[int]] = None,
    ) -> "Circuit":
        from .bits import operand_index
        targets=tuple(operand_index(self,wire,True) for wire in targets)
        controls=tuple(operand_index(self,wire,True) for wire in controls)
        if clbits is not None:clbits=tuple(operand_index(self,bit,False) for bit in clbits)
        from .annotated_operation import AnnotatedOperation
        if isinstance(name, AnnotatedOperation):
            return self._append_annotated(name, targets, controls, params, metadata, clbits)
        if isinstance(name, Instruction):
            return self._append_instruction(name, targets=targets, controls=controls, params=params, metadata=metadata, clbits=clbits)
        targets_tuple = tuple(targets)
        controls_tuple = tuple(controls)
        all_wires = controls_tuple + targets_tuple
        for wire in all_wires:
            self._check_wire(wire)
        if len(set(all_wires)) != len(all_wires):
            raise ValueError("QuantumBridge operations require distinct wires.")
        incoming=set(parameter_set(params))
        if metadata and "control_node" in metadata:incoming.update(metadata["control_node"].parameters)
        if incoming:validate_parameter_names((*self.parameters,*incoming))
        self.operations.append(
            Operation(name=name, targets=targets_tuple, controls=controls_tuple, params=tuple(params), metadata=dict(metadata or {}))
        )
        return self

    def _append_instruction(
        self,
        instruction: Instruction,
        targets: Sequence[int],
        controls: Sequence[int] = (),
        params: Sequence[ParameterValue] = (),
        metadata: Optional[Mapping[str, Any]] = None,
        clbits: Optional[Sequence[int]] = None,
    ) -> "Circuit":
        from .gates import Gate
        if isinstance(instruction, Gate):
            return self._append_gate(instruction, targets, controls, params, metadata, clbits)
        from .control_flow_operations import is_control_owner, control_entry
        if is_control_owner(instruction):
            from .circuit_data import validate_entries
            if controls or params or metadata:
                raise ValueError("Bind control owners explicitly; extra invocation modifiers are not supported.")
            bits = tuple(range(instruction.num_clbits)) if clbits is None else clbits
            entry = control_entry(self, instruction.copy(), targets, bits)
            validate_entries(self, list(self.iter_instructions())+[entry])
            self.operations.append(entry)
            return self
        from .instruction_parameters import admit_owner
        from quantumbridge.execution.array_ownership import native_configuration_copy
        from quantumbridge.execution.resource_context import current_resources
        admit_owner(instruction)
        def owned(value):
            return native_configuration_copy(value, category="native_instruction_append",
                strict=current_resources() is not None)
        targets_tuple = tuple(int(wire) for wire in targets)
        controls_tuple = tuple(int(wire) for wire in controls)
        clbits_tuple = tuple(range(instruction.num_bits)) if clbits is None else tuple(int(bit) for bit in clbits)
        if len(targets_tuple) != instruction.num_qubits:
            raise ValueError("QuantumBridge instruction append target count must match instruction.num_qubits.")
        if len(clbits_tuple) != instruction.num_bits:
            raise ValueError("QuantumBridge instruction append clbit count must match instruction.num_bits.")
        for wire in controls_tuple + targets_tuple:
            self._check_wire(wire)
        for bit in clbits_tuple:
            self._check_bit(bit)
        if len(set(controls_tuple + targets_tuple)) != len(controls_tuple + targets_tuple):
            raise ValueError("QuantumBridge instruction append requires distinct qubits.")

        merged_metadata = owned(instruction.metadata)
        if isinstance(merged_metadata.get("control_flow"), list):
            merged_metadata.pop("control_flow")
        merged_metadata.update(dict(metadata or {}))
        effective_params = owned(tuple(params) if params else tuple(instruction.params))
        incoming=instruction.definition.parameters if instruction.definition is not None else parameter_set(effective_params)
        if incoming:validate_parameter_names((*self.parameters,*incoming))
        effective_instruction = (
            instruction
            if not controls_tuple
            else instruction.control(num_ctrl_qubits=len(controls_tuple), ctrl_state=merged_metadata.get("ctrl_state"))
        )
        qubit_map = {source: target for source, target in enumerate((controls_tuple + targets_tuple) if controls_tuple else targets_tuple)}
        clbit_map = {source: target for source, target in enumerate(clbits_tuple)}

        if effective_instruction.definition is not None:
            self.global_phase = self.global_phase + effective_instruction.definition.global_phase
            offset = len(self.operations)
            for op in effective_instruction.definition.operations:
                mapped = _remap_operation(op, qubit_map, clbit_map, self.num_qubits, self.num_bits)
                mapped_metadata = owned(mapped.metadata)
                mapped_metadata.setdefault(
                    "instruction",
                    {
                        "name": instruction.name,
                        "params": list(effective_params),
                        "controlled": bool(controls_tuple),
                    },
                )
                mapped_metadata.update(merged_metadata)
                self.operations.append(Operation(mapped.name, mapped.targets, mapped.controls, mapped.params, mapped_metadata))
            for measurement in effective_instruction.definition.measurements:
                mapped = _remap_measurement(measurement, qubit_map, clbit_map, offset, len(effective_instruction.definition.operations))
                mapped_metadata = {**mapped.metadata, **owned(merged_metadata)}
                self.measurements.append(
                    Measurement(mapped.wire, mapped.bit, mapped.kind, mapped_metadata)
                )
            return self

        if effective_instruction.name == "measure" and instruction.num_qubits == instruction.num_bits == 1 and not effective_params and not controls_tuple:
            self.measure(targets_tuple[0], clbits_tuple[0])
            self.measurements[-1].metadata.update(merged_metadata)
            return self
        self.operations.append(
            Operation(
                name=effective_instruction.name,
                targets=targets_tuple,
                controls=controls_tuple,
                params=effective_params,
                metadata=merged_metadata,
            )
        )
        return self

    def _append_instruction_reference(self, instruction, targets, clbits=(), metadata=None):
        from .instruction_parameters import admit_owner, owned_copy
        from .operations import instruction_definition_work
        from .gate_limits import current_gate_limits
        admit_owner(instruction)
        from .control_flow_operations import is_control_owner, control_entry
        if is_control_owner(instruction):
            from .circuit_data import validate_entries
            if metadata:
                raise ValueError("Control instruction references cannot carry opaque invocation metadata.")
            entry = control_entry(self, instruction, targets, clbits)
            validate_entries(self, list(self.iter_instructions())+[entry])
            self.operations.append(entry)
            return self
        if type(instruction) is not Instruction: raise TypeError("An instruction reference needs a native Instruction.")
        targets, clbits = tuple(targets), tuple(clbits)
        if len(targets) != instruction.num_qubits or len(clbits) != instruction.num_bits:
            raise ValueError("Instruction reference width differs from its owner.")
        if len(set(targets)) != len(targets) or len(set(clbits)) != len(clbits):
            raise ValueError("Instruction reference arguments must be distinct.")
        for wire in targets: self._check_wire(wire)
        for bit in clbits: self._check_bit(bit)
        current_gate_limits().check("max_total_operations", len(self.operations)+len(self.measurements)+1)
        if instruction.definition is not None: instruction_definition_work(instruction.definition, owner=instruction)
        validate_parameter_names((*self.parameters, *parameter_set(instruction.params)))
        extra = owned_copy(dict(metadata or {}))
        if set(extra) & {"gate", "annotated_operation", "instruction_object", "instruction_clbits"}:
            raise ValueError("Instruction reference metadata keys are reserved.")
        owner = instruction.copy()
        extra.update(instruction_object=owner, instruction_clbits=clbits)
        if owner.label is not None: extra.setdefault("label", owner.label)
        self.operations.append(Operation(owner.name, targets, params=tuple(owner.params), metadata=extra))
        return self

    def _append_gate(self, gate, targets, controls, params, metadata, clbits):
        from .gates import Gate, ControlledGate, validate_definition
        from .gate_limits import current_gate_limits
        from quantumbridge.execution.resource_context import current_resources
        from .gates import is_native_gate
        from .standard_gates import invocation_params
        if current_resources() is not None and not is_native_gate(gate):
            from .classical import UnsupportedFeature
            raise UnsupportedFeature("managed_gate_type", "Managed append requires an exact native Gate type.")
        wires, controls = tuple(targets), tuple(controls)
        if len(wires) != gate.num_qubits or clbits is not None and tuple(clbits):
            raise ValueError("Gate append requires its exact quantum width and no classical bits.")
        for wire in controls+wires: self._check_wire(wire)
        if len(set(controls+wires)) != len(controls+wires): raise ValueError("Gate wires must be distinct.")
        if params:
            if len(params) != len(gate.params) or not all(isinstance(p, Parameter) for p in gate.params):
                raise ValueError("Gate append parameter overrides require direct formal Parameters.")
            gate = gate.assign_parameters(dict(zip(gate.params, params)))
        if controls: gate = gate.control(len(controls), ctrl_state=(metadata or {}).get("ctrl_state"))
        current_gate_limits().check("max_total_operations", len(self.operations)+1)
        if gate.definition is not None: validate_definition(gate.definition, owner=gate)
        validate_parameter_names((*self.parameters, *parameter_set(gate.params)))
        extra = deepcopy(dict(metadata or {}))
        if "gate" in extra: raise ValueError("The native gate metadata key is reserved.")
        if gate.label is not None: extra.setdefault("label", gate.label)
        if gate.definition is None:
            from .gates import validate_definition as check
            leaf = Circuit(gate.num_qubits).append(gate.name, range(gate.num_qubits), params=gate.params, metadata=gate.metadata)
            check(leaf)
            extra = {**deepcopy(gate.metadata), **extra}
        else:
            extra["gate"] = gate.copy()
        self.operations.append(Operation(gate.name, controls+wires, params=invocation_params(gate), metadata=extra))
        return self

    def _append_annotated(self, operation, targets, controls, params, metadata, clbits):
        from .annotated_operation import AnnotatedOperation, validate_annotated_structure, _inspect
        from .gate_limits import current_gate_limits
        if type(operation) is not AnnotatedOperation: raise TypeError("Only the native annotation type can be appended.")
        validate_annotated_structure(operation)
        from .operations import instruction_definition_work
        base = _inspect(operation)[0]
        bits=() if clbits is None else tuple(clbits)
        if len(bits)!=operation.num_clbits:raise ValueError("Annotation classical operand count differs from its operation.")
        if base.definition is not None: instruction_definition_work(base.definition, owner=base)
        wires, controls = tuple(targets), tuple(controls)
        if params:raise ValueError("Bind annotated parameters explicitly before append.")
        if len(wires) != operation.num_qubits: raise ValueError("Annotation append width differs from its operation.")
        for wire in controls+wires: self._check_wire(wire)
        for bit in bits:self._check_bit(bit)
        if len(set(controls+wires)) != len(controls+wires): raise ValueError("Annotation wires must be distinct.")
        if len(set(bits))!=len(bits):raise ValueError("Annotation classical operands must be distinct.")
        if controls: operation = operation.control(len(controls), ctrl_state=(metadata or {}).get("ctrl_state"))
        current_gate_limits().check("max_total_operations", len(self.operations)+1)
        validate_parameter_names((*self.parameters, *operation.parameters))
        from .gates import _copy
        extra = _copy(dict(metadata or {}))
        if "gate" in extra or "annotated_operation" in extra: raise ValueError("Native definition metadata keys are reserved.")
        extra["annotated_operation"] = operation.copy()
        if bits:extra["instruction_clbits"]=bits
        from .standard_gates import invocation_params
        self.operations.append(Operation(operation.name, controls+wires, params=invocation_params(operation), metadata=extra))
        return self

    def to_gate(self, parameter_map=None, label=None, *, _name=None):
        from .gates import Gate, validate_definition
        validate_definition(self)
        instruction = self.to_instruction(parameter_map=parameter_map)
        result = Gate(_name or self.name or "circuit", self.num_qubits, instruction.params,
                      label, definition=instruction.definition)
        if instruction._parameter_plan is not None:
            from .instruction_parameters import inherit_binding_plan
            inherit_binding_plan(instruction, result)
        return result

    def decompose(self, gates_to_decompose=None, reps=1):
        from .gates import decompose
        return decompose(self, gates_to_decompose, reps)

    def lowered(self):
        from .gates import lower
        return lower(self)

    def repeat(self, reps):
        import operator
        from .instruction_parameters import admit_payloads
        from .operations import instruction_definition_work
        from .gate_limits import current_gate_limits
        admit_payloads((reps,))
        original_reps = reps
        reps = operator.index(reps)
        limits = current_gate_limits()
        limits.check("max_repeat_count", max(0, reps))
        work = 0 if reps <= 0 else instruction_definition_work(self)
        limits.check("max_total_operations", max(0, reps)*max(1, work))
        out = self.copy_empty_like(name=(self.name or "circuit")+"**"+str(original_reps),vars_mode="drop")
        out.global_phase=0.
        if reps <= 0: return out
        if not self.num_bits and not self.measurements and not self.has_dynamic_operations:
            if any("gate" in op.metadata for op in self.operations):
                from .gates import validate_definition
                validate_definition(self)
            try: gate = self.to_gate()
            except ValueError: gate = None
            if gate is not None:
                for _ in range(reps): out.append(gate, range(self.num_qubits))
                return out
        for _ in range(reps): out.compose(self, inplace=True)
        return out

    def tensor(self, other, inplace=False):
        if not isinstance(other, Circuit): raise TypeError("Circuit.tensor requires a native Circuit.")
        from .gate_limits import current_gate_limits
        current_gate_limits().check("max_total_operations", len(self.operations)+len(other.operations))
        validate_parameter_names((*self.parameters, *other.parameters))
        # Match public qubit order (other's bits first). Native tensors remain
        # wire-major, so the raw native matrix is kron(other, self).
        out = empty_execution_circuit(self,num_qubits=self.num_qubits+other.num_qubits,num_bits=self.num_bits+other.num_bits)
        out.global_phase = 0.
        out.compose(other, range(other.num_qubits), range(other.num_bits), inplace=True)
        out.compose(self, range(other.num_qubits, out.num_qubits), range(other.num_bits, out.num_bits), inplace=True)
        if inplace:
            _replace_self(self, out)
            return None
        return out

    @property
    def calibrations(self) -> dict:
        return execution_metadata(self).setdefault("calibrations", {})

    def x(self, q: int) -> "Circuit":
        return self.append("x", (q,))

    def reset(self, q: int) -> "Circuit":
        return self.append("reset", (q,))

    def kraus(self, operators, wires: Sequence[int]) -> "Circuit":
        from quantumbridge.utils.math import validate_kraus_operators

        selected = tuple(wires)
        if not selected:
            raise ValueError("QuantumBridge circuit channels require at least one wire.")
        matrices = validate_kraus_operators(operators, 1 << len(selected))
        return self.append("kraus", selected, metadata={"kraus": matrices})

    def channel(self, channel, wires: Sequence[int]) -> "Circuit":
        if hasattr(channel, "kraus"):
            return self.kraus(channel.kraus(), wires)
        if hasattr(channel, "to_instruction"):
            return self.append(channel.to_instruction(), wires)
        raise TypeError("QuantumBridge circuit channels require Kraus matrices or a native channel instruction.")

    def id(self, q: int) -> "Circuit":
        return self.append("id", (q,))

    def y(self, q: int) -> "Circuit":
        return self.append("y", (q,))

    def z(self, q: int) -> "Circuit":
        return self.append("z", (q,))

    def h(self, q: int) -> "Circuit":
        return self.append("h", (q,))

    def sx(self, q: int) -> "Circuit":
        return self.append("sx", (q,))

    def sxdg(self, q: int) -> "Circuit":
        return self.append("sxdg", (q,))

    def s(self, q: int) -> "Circuit":
        return self.append("s", (q,))

    def sdg(self, q: int) -> "Circuit":
        return self.append("sdg", (q,))

    def t(self, q: int) -> "Circuit":
        return self.append("t", (q,))

    def tdg(self, q: int) -> "Circuit":
        return self.append("tdg", (q,))

    def rx(self, theta: ParameterValue, q: int) -> "Circuit":
        return self.append("rx", (q,), params=(theta,))

    def ry(self, theta: ParameterValue, q: int) -> "Circuit":
        return self.append("ry", (q,), params=(theta,))

    def rz(self, theta: ParameterValue, q: int) -> "Circuit":
        return self.append("rz", (q,), params=(theta,))

    def phase(self, theta: ParameterValue, q: int) -> "Circuit":
        return self.append("phase", (q,), params=(theta,))

    def p(self, theta: ParameterValue, q: int) -> "Circuit":
        return self.phase(theta, q)

    def u(self, theta: ParameterValue, phi: ParameterValue, lam: ParameterValue, q: int) -> "Circuit":
        return self.append("u", (q,), params=(theta, phi, lam))

    def u1(self, lam: ParameterValue, q: int) -> "Circuit":
        return self.append("u1", (q,), params=(lam,))

    def u2(self, phi: ParameterValue, lam: ParameterValue, q: int) -> "Circuit":
        return self.append("u2", (q,), params=(phi, lam))

    def u3(self, theta: ParameterValue, phi: ParameterValue, lam: ParameterValue, q: int) -> "Circuit":
        return self.append("u3", (q,), params=(theta, phi, lam))

    def cx(self, control: int, target: int) -> "Circuit":
        return self.append("cx", (target,), controls=(control,))

    def cy(self, control: int, target: int) -> "Circuit":
        return self.append("cy", (target,), controls=(control,))

    def ch(self, control: int, target: int) -> "Circuit":
        return self.append("ch", (target,), controls=(control,))

    def csx(self, control: int, target: int) -> "Circuit":
        return self.append("csx", (target,), controls=(control,))

    def csxdg(self, control: int, target: int) -> "Circuit":
        return self.append("csxdg", (target,), controls=(control,))

    def crx(self, theta: ParameterValue, control: int, target: int) -> "Circuit":
        return self.append("crx", (target,), controls=(control,), params=(theta,))

    def cry(self, theta: ParameterValue, control: int, target: int) -> "Circuit":
        return self.append("cry", (target,), controls=(control,), params=(theta,))

    def crz(self, theta: ParameterValue, control: int, target: int) -> "Circuit":
        return self.append("crz", (target,), controls=(control,), params=(theta,))

    def cp(self, theta: ParameterValue, control: int, target: int) -> "Circuit":
        return self.append("cp", (target,), controls=(control,), params=(theta,))

    def cphase(self, theta: ParameterValue, control: int, target: int) -> "Circuit":
        return self.cp(theta, control, target)

    def cz(self, control: int, target: int) -> "Circuit":
        return self.append("cz", (target,), controls=(control,))

    def swap(self, a: int, b: int) -> "Circuit":
        return self.append("swap", (a, b))

    def dcx(self, a: int, b: int) -> "Circuit":
        return self.append("dcx", (a, b))

    def ecr(self, a: int, b: int) -> "Circuit":
        return self.append("ecr", (a, b))

    def rxx(self, theta: ParameterValue, a: int, b: int) -> "Circuit":
        return self.append("rxx", (a, b), params=(theta,))

    def ryy(self, theta: ParameterValue, a: int, b: int) -> "Circuit":
        return self.append("ryy", (a, b), params=(theta,))

    def rzz(self, theta: ParameterValue, a: int, b: int) -> "Circuit":
        return self.append("rzz", (a, b), params=(theta,))

    def iswap(self, a: int, b: int) -> "Circuit":
        return self.append("iswap", (a, b), metadata={"experimental": True})

    def ccx(self, c0: int, c1: int, target: int) -> "Circuit":
        return self.append("ccx", (target,), controls=(c0, c1))

    def cswap(self, control: int, a: int, b: int) -> "Circuit":
        return self.append("cswap", (a, b), controls=(control,))

    def fredkin(self, control: int, a: int, b: int) -> "Circuit":
        return self.cswap(control, a, b)

    def barrier(self, *qubits: int) -> "Circuit":
        wires = tuple(range(self.num_qubits)) if not qubits else tuple(int(wire) for wire in qubits)
        if len(set(wires)) != len(wires):
            raise ValueError("QuantumBridge barrier qubits must be distinct.")
        for wire in wires:
            self._check_wire(wire)
        self.operations.append(Operation("barrier", wires, metadata={"directive": True}))
        return self

    def delay(self, duration: ParameterValue, q: int, unit: str = "dt") -> "Circuit":
        self._check_wire(q)
        self.operations.append(
            Operation(
                "delay",
                (q,),
                params=(duration,),
                metadata={"directive": True, "duration": duration, "unit": str(unit)},
            )
        )
        return self

    def unitary(self, matrix: Any, wires: Sequence[int], name: str = "unitary") -> "Circuit":
        from quantumbridge.execution.resource_context import current_resources
        from quantumbridge.utils.math import validate_unitary_matrix
        if current_resources() is not None:
            from quantumbridge.execution.array_reservations import numeric_array
            arr = numeric_array(matrix, category="circuit_unitary_conversion")
        else:arr = np.asarray(matrix, dtype=complex)
        dim = 1 << len(tuple(wires))
        if arr.shape != (dim, dim):
            raise ValueError("QuantumBridge custom unitary shape must match target wires.")
        if not validate_unitary_matrix(arr, dim, atol=1e-10, rtol=1e-5):
            raise ValueError("QuantumBridge custom unitary matrix must be unitary within tolerance.")
        return self.append(name, tuple(wires), metadata={"matrix": arr})

    def compose(
        self,
        other: "Circuit",
        qubits: Optional[Sequence[int]] = None,
        clbits: Optional[Sequence[int]] = None,
        front: bool = False,
        inplace: bool = False,
        wrap: bool = False,
        *,
        copy: bool = True,
        var_remap=None,
        inline_captures: bool = False,
    ) -> "Circuit | None":
        from .composition import compose
        return compose(self, other, qubits, clbits, front, inplace, wrap,
            copy=copy, var_remap=var_remap, inline_captures=inline_captures)

    @property
    def has_structured_control_flow(self):
        return bool(execution_metadata(self).get("classical_scope")) or any("control_node" in op.metadata for op in self.operations)

    @property
    def has_measurements(self):
        return bool(self.measurements) or any(body.has_measurements for op in self.operations
            if "control_node" in op.metadata for body in op.metadata["control_node"].bodies)

    def _control(self, node, inplace=True):
        target = self if inplace else self.copy()
        target.append("control_flow", (), metadata={"control_node": node})
        return target

    def add_var(self, name, dtype=_MISSING, initial=_MISSING):
        from .classical import variable, lift, ClassicalType, Var
        from .control_flow import ControlBlock
        if dtype is _MISSING:
            if initial is _MISSING: raise TypeError("A public variable declaration needs an initializer.")
            dtype,initial=initial,_MISSING
        if isinstance(dtype, ClassicalType):
            target = variable(name, dtype)
            self._control(ControlBlock("declare", {"target": target, "initial": None if initial is _MISSING or initial is None else lift(initial, dtype)}))
            return target
        if initial is not _MISSING: raise TypeError("Use either a public initializer or explicit native dtype/initial.")
        value = lift(dtype)
        target = Var.new(name, value.dtype) if isinstance(name,str) else name
        if type(target) is not Var: raise TypeError("A named declaration requires Var or a string name.")
        value = lift(dtype, target.dtype)
        from .classical_scope import add_identifier
        from .gate_limits import current_gate_limits
        current_gate_limits().check("max_total_operations",len(self.operations)+len(self.measurements)+1)
        add_identifier(self,target,"local")
        self._control(ControlBlock("store", {"target":target,"value":value}))
        return target

    def add_uninitialized_var(self, var):
        from .classical import Var
        from .classical_scope import add_identifier
        if type(var) is not Var: raise TypeError("Expected a native Var.")
        add_identifier(self,var,"local")

    def add_input(self, name_or_var, type_=None):
        from .classical import Var
        from .classical_scope import add_identifier
        var = Var.new(name_or_var,type_) if isinstance(name_or_var,str) else name_or_var
        if type(var) is not Var or type_ is not None and type_ != var.dtype: raise TypeError("Input type differs from its Var.")
        return add_identifier(self,var,"input")

    def add_capture(self, var):
        from .classical_scope import add_identifier
        add_identifier(self,var,"capture")

    def add_stretch(self, name_or_stretch):
        from .classical import Stretch
        from .classical_scope import add_identifier
        var = Stretch.new(name_or_stretch) if isinstance(name_or_stretch,str) else name_or_stretch
        if type(var) is not Stretch: raise TypeError("Expected a native Stretch or string name.")
        return add_identifier(self,var,"local")

    def get_identifier(self, name, default=_MISSING):
        from .classical_scope import get_identifier
        return get_identifier(self,name) if default is _MISSING else get_identifier(self,name,default)

    def get_var(self, name):
        from .classical_scope import get_identifier
        return get_identifier(self,name,kind="var")

    def get_stretch(self, name):
        from .classical_scope import get_identifier
        return get_identifier(self,name,kind="stretch")

    def has_var(self, name_or_var):
        from .classical_scope import entries
        return any(node.op=="var" and (node.value[1]==name_or_var if isinstance(name_or_var,str)
            else node.value==name_or_var.value) for node,_,_ in entries(self))

    def has_stretch(self, name_or_stretch):
        from .classical_scope import entries
        return any(node.op=="stretch" and (node.value[1]==name_or_stretch if isinstance(name_or_stretch,str)
            else node.value==name_or_stretch.value) for node,_,_ in entries(self))

    def _iter_identifiers(self, kind, scope=None):
        from .classical_scope import entries
        return (node for node,mode,_ in entries(self) if node.op==kind and (scope is None or mode==scope))

    def iter_vars(self): return self._iter_identifiers("var")
    def iter_input_vars(self): return self._iter_identifiers("var","input")
    def iter_captured_vars(self): return self._iter_identifiers("var","capture")
    def iter_declared_vars(self): return self._iter_identifiers("var","local")
    def iter_stretches(self): return self._iter_identifiers("stretch")
    def iter_captured_stretches(self): return self._iter_identifiers("stretch","capture")
    def iter_declared_stretches(self): return self._iter_identifiers("stretch","local")

    @property
    def num_vars(self): return sum(1 for _ in self.iter_vars())
    @property
    def num_stretches(self): return sum(1 for _ in self.iter_stretches())

    def store(self, target, value):
        from .classical import lift
        from .control_flow import ControlBlock
        return self._control(ControlBlock("store", {"target": target, "value": lift(value, target.dtype)}))

    def _mapped_body(self, body, qubits=None, clbits=None):
        return self.copy_empty_like(vars_mode="drop").compose(body, qubits=qubits, clbits=clbits)

    def for_loop(self, values, loop_parameter=None, body=None, *, variable=None, qubits=None, clbits=None, inplace=False):
        from .control_flow import ControlBlock
        if isinstance(loop_parameter, Circuit) and body is None:
            body, loop_parameter = loop_parameter, None
        if isinstance(values, range) and len(values) > 1000000:
            raise ValueError("QuantumBridge loop value sets exceed the construction budget.")
        from itertools import islice
        values = tuple(islice(iter(values), 1000001))
        if len(values) > 1000000:
            raise ValueError("QuantumBridge loop value sets exceed the construction budget.")
        return self._control(ControlBlock("for", {"values": values, "parameter": loop_parameter, "variable": variable,
            "body": self._mapped_body(body, qubits, clbits)}), inplace)

    def while_loop(self, condition, body, qubits=None, clbits=None, inplace=False):
        from .control_flow import ControlBlock, condition_expression
        return self._control(ControlBlock("while", {"condition": condition_expression(condition),
            "body": self._mapped_body(body, qubits, clbits)}), inplace)

    def switch(self, selector, cases, default=None, *, qubits=None, clbits=None, inplace=False):
        from .classical import lift
        from .control_flow import ControlBlock
        entries = cases.items() if isinstance(cases, ABCMapping) else cases
        mapped = tuple(((value,) if isinstance(value, int) else tuple(value), self._mapped_body(body, qubits, clbits)) for value, body in entries)
        return self._control(ControlBlock("switch", {"selector": lift(selector), "cases": mapped,
            "default": None if default is None else self._mapped_body(default, qubits, clbits)}), inplace)

    def break_loop(self):
        from .control_flow import ControlBlock
        return self._control(ControlBlock("break", {}))

    def continue_loop(self):
        from .control_flow import ControlBlock
        return self._control(ControlBlock("continue", {}))

    def if_else(
        self,
        condition: Union[dict[str, Any], tuple[Union[int, Sequence[int]], int]],
        true_body: "Circuit",
        false_body: Optional["Circuit"] = None,
        qubits: Optional[Sequence[int]] = None,
        clbits: Optional[Sequence[int]] = None,
        inplace: bool = False,
    ) -> "Circuit":
        """Append lightweight Qiskit-style conditional circuit blocks.

        QuantumBridge records if/else structure as operation metadata and applies
        the matching classical condition to the branch operations. Dynamic
        runtime branching is intentionally left to backend implementations.
        """

        if not isinstance(true_body, Circuit):
            raise TypeError("QuantumBridge Circuit.if_else true_body must be a Circuit.")
        if false_body is not None and not isinstance(false_body, Circuit):
            raise TypeError("QuantumBridge Circuit.if_else false_body must be a Circuit.")
        from .classical import ClassicalExpression
        from .control_flow import ControlBlock, condition_expression
        if isinstance(condition, (ClassicalExpression, bool)) or true_body.has_structured_control_flow or _has_global_phase(true_body) or false_body is not None and (false_body.has_structured_control_flow or _has_global_phase(false_body)):
            return self._control(ControlBlock("if", {"condition": condition_expression(condition),
                "body": self._mapped_body(true_body, qubits, clbits),
                "alternate": None if false_body is None else self._mapped_body(false_body, qubits, clbits)}), inplace)
        normalized = _normalize_condition(condition)
        for bit in normalized["bits"]:
            self._check_bit(bit)
        target = self if inplace else self.copy()
        block_id = uuid4().hex
        mapped_true = target.copy_empty_like(vars_mode="drop").compose(true_body, qubits=qubits, clbits=clbits)
        true_block = _circuit_with_operation_metadata(
            mapped_true,
            {
                "condition": deepcopy(normalized),
                "control_flow": {"type": "if_else", "branch": "true", "condition": deepcopy(normalized), "block_id": block_id},
            },
        )
        target.compose(true_block, inplace=True)
        if false_body is not None:
            mapped_false = target.copy_empty_like(vars_mode="drop").compose(false_body, qubits=qubits, clbits=clbits)
            false_block = _circuit_with_operation_metadata(
                mapped_false,
                {"control_flow": {"type": "if_else", "branch": "false", "condition": deepcopy(normalized), "block_id": block_id}},
                start=not (true_block.operations or true_block.measurements),
            )
            target.compose(false_block, inplace=True)
        execution_metadata(target).setdefault("control_flow", []).append(
            {
                "type": "if_else",
                "condition": deepcopy(normalized),
                "true_body": true_body.name,
                "false_body": None if false_body is None else false_body.name,
            }
        )
        return target

    def if_test(
        self,
        condition: Union[dict[str, Any], tuple[Union[int, Sequence[int]], int]],
        body: "Circuit",
        qubits: Optional[Sequence[int]] = None,
        clbits: Optional[Sequence[int]] = None,
        inplace: bool = False,
    ) -> "Circuit":
        return self.if_else(condition, body, qubits=qubits, clbits=clbits, inplace=inplace)

    def inverse(self, annotated=False, *, inplace: bool = False) -> "Circuit":
        from .errors import CircuitError
        if self.measurements:
            raise CircuitError("QuantumBridge Circuit.inverse does not invert measured circuits.")
        kind=type(self) if is_native_circuit(self) else Circuit
        target = kind(self.num_qubits, self.num_bits, name=(self.name or "circuit")+"_dg", metadata=self.metadata)
        inherit_execution(self,target)
        if is_separated(self):target.metadata={}
        from .bits import copy_bits
        copy_bits(self,target)
        try: target.operations = [_inverse_operation(op, annotated=annotated) for op in reversed(self.operations)]
        except (TypeError, ValueError) as exc: raise CircuitError("Circuit inverse is not implemented for this operation.") from exc
        target.global_phase = -self.global_phase
        return _replace_self(self, target) if inplace else target

    def power(self, exponent: int, inplace: bool = False) -> "Circuit":
        if not isinstance(exponent, int):
            raise ValueError("QuantumBridge Circuit.power requires an integer exponent.")
        if self.measurements:
            raise ValueError("QuantumBridge Circuit.power does not repeat measured circuits.")
        unit = self.inverse() if exponent < 0 else self
        target = empty_execution_circuit(self)
        from .bits import copy_bits
        copy_bits(self,target)
        target.global_phase = 0.
        for _ in range(abs(exponent)):
            target.compose(unit, inplace=True)
        return _replace_self(self, target) if inplace else target

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[int] = None,
    ) -> "Circuit":
        if not isinstance(num_ctrl_qubits, int) or num_ctrl_qubits <= 0:
            raise ValueError("QuantumBridge Circuit.control requires a positive control-qubit count.")
        if self.measurements:
            raise ValueError("QuantumBridge Circuit.control does not control measured circuits.")
        if self.num_qubits==0:
            controlled=self.to_gate().control(num_ctrl_qubits,label=label,ctrl_state=ctrl_state)
            out=empty_execution_circuit(self,num_qubits=num_ctrl_qubits,name=label or self.name)
            out.global_phase=0.
            return out.append(controlled,range(num_ctrl_qubits))
        if any("instruction_object" in op.metadata for op in self.operations):
            from .instruction_parameters import owned_copy
            controlled = self.to_gate().control(num_ctrl_qubits, label=label, ctrl_state=ctrl_state)
            out = controlled.definition.copy()
            out.name = label or self.name
            if is_separated(self):
                phase=out.global_phase
                out.metadata=self.metadata;inherit_execution(self,out);out.global_phase=phase
                execution_metadata(out).pop("registers",None)
            else:
                metadata = owned_copy(self.metadata)
                metadata.pop("global_phase", None)
                metadata.pop("registers", None)
                out.metadata.update(metadata)
            return out
        ctrl_state = (1 << num_ctrl_qubits) - 1 if ctrl_state is None else int(ctrl_state)
        if not 0 <= ctrl_state < (1 << num_ctrl_qubits):
            raise ValueError("QuantumBridge Circuit.control ctrl_state is outside the control register.")
        out = empty_execution_circuit(self,num_qubits=num_ctrl_qubits+self.num_qubits,name=label or self.name)
        out.global_phase = 0.
        prefix = tuple(range(num_ctrl_qubits))
        if _has_global_phase(self):
            if not isinstance(self.global_phase, Real):
                from .classical import UnsupportedFeature
                raise UnsupportedFeature("symbolic_controlled_global_phase", "Bind the phase before coherently controlling this circuit.")
            controlled_phase = _controlled_matrix(np.exp(1j * self.global_phase) * np.eye(2), num_ctrl_qubits, ctrl_state)
            out.unitary(controlled_phase, prefix + (num_ctrl_qubits,), name="controlled_global_phase")
        for op in self.operations:
            base_wires = op.controls + op.targets
            mapped_wires = prefix + tuple(num_ctrl_qubits + wire for wire in base_wires)
            base = gate_matrix(op.name, op.params, op.metadata)
            controlled = _controlled_matrix(base, num_ctrl_qubits, ctrl_state)
            metadata = {
                "matrix": controlled,
                "controlled_operation": op.name,
                "ctrl_state": ctrl_state,
                "base_wires": base_wires,
            }
            out.unitary(controlled, mapped_wires, name=f"c{num_ctrl_qubits}_{op.name}").operations[-1].metadata.update(metadata)
        return out

    def with_condition(self, classical: Union[int, tuple[int, ...]], value: int, inplace: bool = False) -> "Circuit":
        target = self if inplace else self.copy()
        bits = (classical,) if isinstance(classical, int) else tuple(classical)
        for bit in bits:
            target._check_bit(bit)
        condition = {"bits": list(bits), "value": int(value)}
        target.operations = [_operation_with_metadata(op, {"condition": condition}) for op in target.operations]
        return target

    def c_if(self, classical: Union[int, tuple[int, ...]], value: int) -> "Circuit":
        return self.with_condition(classical, value, inplace=True)

    def add_calibration(
        self,
        name: str,
        qubits: Sequence[int],
        schedule: Any,
        params: Sequence[ParameterValue] = (),
    ) -> "Circuit":
        qubits_tuple = tuple(qubits)
        for wire in qubits_tuple:
            self._check_wire(wire)
        key = _calibration_key(name, qubits_tuple, tuple(params))
        self.calibrations[key] = {
            "name": name,
            "qubits": qubits_tuple,
            "params": tuple(params),
            "schedule": schedule,
        }
        return self

    def get_calibration(self, name: str, qubits: Sequence[int], params: Sequence[ParameterValue] = ()) -> Any:
        key = _calibration_key(name, tuple(qubits), tuple(params))
        return self.calibrations[key]["schedule"]

    def measure(self, q: int, bit: Optional[int] = None) -> "Circuit":
        from .bits import operand_index
        q=operand_index(self,q,True)
        if bit is None:
            bit = len(self.measurements)
        bit=operand_index(self,bit,False)
        self.measurements.append(Measurement(wire=q, bit=bit, metadata={"operation_index": len(self.operations)}))
        return self

    def measure_all(self) -> "Circuit":
        if self.num_bits < self.num_qubits:
            from .bits import Clbit,ClassicalRegister
            from .gate_limits import current_gate_limits
            current_gate_limits().check("max_circuit_bits",2*self.num_qubits)
            if not self.num_bits and "c" not in {reg.name for reg in self._qregs}:
                self.add_register(ClassicalRegister(self.num_qubits,"c"))
            else:self.add_bits([Clbit() for _ in range(self.num_qubits-self.num_bits)])
        for wire in range(self.num_qubits):
            self.measure(wire, wire)
        return self

    def remove_final_measurements(self, inplace: bool = False) -> "Circuit":
        target = self if inplace else self.copy()
        target.measurements = [item for item in target.measurements if item.metadata.get("operation_index", len(target.operations)) < len(target.operations)]
        return target

    def iter_instructions(self):
        """Yield gates and measurements in execution order, including mid-circuit reads."""
        positions = {}
        for measurement in self.measurements:
            position = measurement.metadata.get("operation_index", len(self.operations))
            if not isinstance(position, int) or not 0 <= position <= len(self.operations):
                raise ValueError("QuantumBridge measurement operation_index is outside this circuit.")
            positions.setdefault(position, []).append(measurement)
        for index in range(len(self.operations) + 1):
            yield from positions.get(index, ())
            if index < len(self.operations):
                yield self.operations[index]

    @property
    def has_dynamic_operations(self) -> bool:
        return bool(execution_metadata(self).get("classical_scope")) or any(
            item.metadata.get("operation_index", len(self.operations)) < len(self.operations)
            or any(key in item.metadata for key in ("condition", "control_flow", "control_flow_stack"))
            for item in self.measurements
        ) or any(
            op.name in {"reset", "measure", "kraus", "control_flow"} or "simulator_instruction" in op.metadata
            or any(key in op.metadata for key in ("condition", "control_flow", "control_flow_stack", "control_node"))
            for op in self.operations
        )

    @property
    def has_simulator_instructions(self):
        return any("simulator_instruction" in op.metadata or op.metadata.get("control_node") is not None and any(body.has_simulator_instructions for body in op.metadata["control_node"].bodies) for op in self.operations)

    def _save(self,kind,qubits=None,**options):
        from quantumbridge.simulator.instructions import save
        return save(self,kind,qubits,**options)

    def save_state(self,label="state",**options): return self._save("state",label=label,**options)
    def save_statevector(self,label="statevector",**options): return self._save("statevector",label=label,**options)
    def save_statevector_dict(self,label="statevector",**options): return self._save("statevector_dict",label=label,**options)
    def save_density_matrix(self,qubits=None,label="density_matrix",**options): return self._save("density_matrix",qubits,label=label,**options)
    def save_matrix_product_state(self,label="matrix_product_state",**options): return self._save("matrix_product_state",label=label,**options)
    def save_stabilizer(self,label="stabilizer",**options): return self._save("stabilizer",label=label,**options)
    def save_clifford(self,label="clifford",**options): return self._save("clifford",label=label,**options)
    def save_unitary(self,label="unitary",**options): return self._save("unitary",label=label,**options)
    def save_superop(self,label="superop",**options): return self._save("superop",label=label,**options)
    def save_probabilities(self,qubits=None,label="probabilities",**options): return self._save("probabilities",qubits,label=label,**options)
    def save_probabilities_dict(self,qubits=None,label="probabilities",**options): return self._save("probabilities_dict",qubits,label=label,**options)
    def save_amplitudes(self,indices,label="amplitudes",**options): return self._save("amplitudes",indices=indices,label=label,**options)
    def save_amplitudes_squared(self,indices,label="amplitudes_squared",**options): return self._save("amplitudes_squared",indices=indices,label=label,**options)
    def save_expectation_value(self,observable,qubits=None,label="expectation_value",**options): return self._save("expectation_value",qubits,observable=observable,label=label,**options)
    def save_expectation_value_variance(self,observable,qubits=None,label="expectation_value_variance",**options): return self._save("expectation_value_variance",qubits,observable=observable,label=label,**options)

    def _set(self,kind,value):
        from quantumbridge.simulator.instructions import set_state
        return set_state(self,kind,value)

    def set_statevector(self,state): return self._set("statevector",state)
    def set_density_matrix(self,state): return self._set("density_matrix",state)
    def set_matrix_product_state(self,state): return self._set("matrix_product_state",state)
    def set_stabilizer(self,state): return self._set("stabilizer",state)
    def set_unitary(self,state): return self._set("unitary",state)
    def set_superop(self,state): return self._set("superop",state)

    def copy_empty_like(self, name: Optional[str] = None, *, vars_mode="alike") -> "Circuit":
        from quantumbridge.execution.array_ownership import native_configuration_copy
        from quantumbridge.execution.resource_context import current_resources
        if is_separated(self):
            result=empty_execution_circuit(self,name=name)
            result.metadata=native_configuration_copy(self.metadata,category="native_empty_circuit_metadata",strict=current_resources() is not None)
            execution_metadata(result).pop("classical_scope",None)
            result.global_phase=0.
        else:
            metadata = native_configuration_copy(self.metadata,category="native_empty_circuit_metadata",strict=current_resources() is not None)
            metadata.pop("global_phase", None)
            metadata.pop("classical_scope",None)
            result = Circuit(self.num_qubits, self.num_bits, name if name is not None else self.name, metadata)
        from .bits import copy_bits
        copy_bits(self,result)
        from .classical_scope import copy_scope
        copy_scope(self,result,vars_mode)
        return result

    def size(self, filter_function: Optional[Callable[[Any], bool]] = None) -> int:
        items = list(self.operations) + list(self.measurements)
        if filter_function is None:
            return len(items)
        return sum(1 for item in items if filter_function(item))

    def count_ops(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for op in self.operations:
            counts[op.name] = counts.get(op.name, 0) + 1
        for measurement in self.measurements:
            counts["measure"] = counts.get("measure", 0) + 1
        return counts

    def depth(self, filter_function: Optional[Callable[[Any], bool]] = None) -> int:
        if self.has_structured_control_flow:
            from .classical import UnsupportedFeature
            raise UnsupportedFeature("structured_depth", "QuantumBridge dynamic depth depends on the runtime execution path.")
        qubit_layers = [0] * self.num_qubits
        bit_layers = [0] * self.num_bits
        max_layer = 0
        for op in self.iter_instructions():
            if filter_function is not None and not filter_function(op):
                continue
            if isinstance(op, Measurement):
                layer = 1 + max(qubit_layers[op.wire], bit_layers[op.bit])
                qubit_layers[op.wire] = bit_layers[op.bit] = layer
                max_layer = max(max_layer, layer)
                continue
            if op.metadata.get("directive"):
                continue
            wires = op.controls + op.targets
            conditions = [op.metadata.get("condition"), op.metadata.get("gate_condition")]
            conditions += [frame.get("condition") for frame in op.metadata.get("control_flow_stack", [])]
            bits = {bit for condition in conditions if condition for bit in condition["bits"]}
            dependencies = [qubit_layers[wire] for wire in wires] + [bit_layers[bit] for bit in bits]
            layer = 1 + max(dependencies, default=0)
            for wire in wires:
                qubit_layers[wire] = layer
            max_layer = max(max_layer, layer)
        return max_layer

    def copy(self, name=None) -> "Circuit":
        from quantumbridge.execution.array_ownership import native_configuration_copy
        from quantumbridge.execution.resource_context import current_resources
        result = native_configuration_copy(self, category="native_circuit_array_copy",
            strict=is_separated(self) and current_resources() is not None)
        if name is not None:
            if not isinstance(name,str): raise TypeError("Circuit copy name must be a string.")
            result.name=name
        return result

    def to_instruction(self, name: Optional[str] = None, label: Optional[str] = None, *, parameter_map=None) -> Instruction:
        from .classical_scope import entries
        if entries(self):
            from .classical import UnsupportedFeature
            raise UnsupportedFeature("instruction_with_classical_identifiers", "Inline this circuit; Instruction conversion cannot retain identifier scopes.")
        if isinstance(name,ABCMapping):
            if parameter_map is not None:raise TypeError("Instruction parameter map was provided twice.")
            parameter_map,name=name,None
        from quantumbridge.execution.array_ownership import native_configuration_copy
        from .operations import instruction_definition_work
        from .bits import bit_graph,initialize_bits
        from .public_circuit import QuantumCircuit
        instruction_definition_work(self,_resolve_frames=False)
        bit_graph(self)
        ordered=tuple(self.parameters)
        if parameter_map is not None:
            if not isinstance(parameter_map,ABCMapping):raise TypeError("Instruction parameter_map must be a mapping.")
            mapping=_parameter_mapping_from_values(ordered,parameter_map,strict=True)
            if set(mapping)!=set(ordered):raise ValueError("Instruction parameter_map must map every circuit parameter.")
        from .circuit_metadata import execution_copy
        template=execution_copy(self) if is_separated(self) else self.copy()
        if type(self) is QuantumCircuit:
            # Stored operands are local indices. Install new actual default
            # owners without changing their order or mutating the source graph.
            initialize_bits(template,self.num_qubits,self.num_bits)
            template.metadata={}
        if parameter_map is None:definition,parameters=template,ordered
        else:
            from .instruction_parameters import bind_definition
            definition=bind_definition(template,mapping)
            parameters=tuple(mapping[parameter] for parameter in ordered)
        instruction_name = name or self.name or "circuit"
        metadata = native_configuration_copy(definition.metadata,category="native_instruction_metadata_copy")
        metadata.setdefault("definition_name", self.name)
        result = Instruction(
            name=instruction_name,
            num_qubits=self.num_qubits,
            num_bits=self.num_bits,
            params=parameters,
            definition=definition,
            metadata=metadata,
            label=label,
        )
        if parameter_map is not None:
            from .instruction_parameters import ParameterBindingPlan, owned_copy, definition_signature
            object.__setattr__(result, "_parameter_plan", ParameterBindingPlan(
                owned_copy(template), ordered, definition_signature(result.definition)))
        return result

    @classmethod
    def from_instruction(cls, instruction: Instruction) -> "Circuit":
        from .annotated_operation import AnnotatedOperation
        if type(instruction) is AnnotatedOperation:
            return cls(instruction.num_qubits, instruction.num_clbits).append(instruction, range(instruction.num_qubits))
        if not isinstance(instruction, Instruction):
            raise TypeError("QuantumBridge Circuit.from_instruction expects an Instruction.")
        if instruction.definition is not None:
            from .gates import Gate, validate_definition
            if isinstance(instruction, Gate): validate_definition(instruction.definition, owner=instruction)
            return instruction.definition.copy()
        from quantumbridge.execution.array_ownership import native_configuration_copy
        metadata = native_configuration_copy(instruction.metadata,category="native_instruction_metadata_copy")
        if isinstance(metadata.get("control_flow"),list):metadata.pop("control_flow")
        circuit = cls(instruction.num_qubits, instruction.num_bits, instruction.name, instruction.metadata)
        circuit.append(instruction.name, range(instruction.num_qubits),params=instruction.params,metadata=metadata)
        return circuit

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        from .gate_limits import current_gate_limits
        from .symbolic_limits import current_symbolic_limits
        limits,symbolic=current_gate_limits(),current_symbolic_limits()
        found: set[Parameter] = set()
        def include(values):
            for parameter in values:
                if parameter not in found:
                    limits.check("max_gate_parameters",len(found)+1)
                    symbolic.check("max_parameters",len(found)+1)
                    found.add(parameter)
        include(parameter_set((self.global_phase,)))
        for op in self.operations:
            include(parameter_set(op.params))
            if "control_node" in op.metadata:
                include(op.metadata["control_node"].parameters)
            if "annotated_operation" in op.metadata:
                from .operations import execution_operation
                include(execution_operation(op).metadata["annotated_operation"].parameters)
        validate_parameter_names(found)
        return tuple(sorted(found, key=parameter_sort_key))

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def assign_parameters(
        self,
        parameters: Union[Mapping[Union[Parameter, str], ParameterValue], Sequence[ParameterValue]],
        inplace: bool = False,
        strict: bool = True,
        *,
        _copy_memo=None,
        _copy_user_metadata=True,
    ) -> "Circuit":
        from .bits import bit_graph
        from .gate_limits import current_gate_limits
        if type(self.num_qubits) is int and type(self.num_bits) is int:
            current_gate_limits().check("max_circuit_bits",self.num_qubits+self.num_bits)
        bit_graph(self)
        mapping = _parameter_mapping_from_values(self.parameters, parameters, strict=strict)
        if is_native_circuit(self) and type(self) is not Circuit and not mapping and _copy_user_metadata:
            return None if inplace else self.copy()
        from quantumbridge.execution.array_ownership import native_configuration_copy
        kind=type(self) if is_native_circuit(self) else Circuit
        from .circuit_metadata import copy_user_metadata
        metadata=(self.metadata if is_separated(self) and not _copy_user_metadata else copy_user_metadata(self,memo=_copy_memo))
        bound = kind(self.num_qubits, self.num_bits, name=self.name,metadata=metadata)
        inherit_execution(self,bound,memo=_copy_memo)
        from .bits import copy_bits
        copy_bits(self,bound)
        if "global_phase" in execution_metadata(self):
            bound.global_phase = resolve_parameter(self.global_phase, mapping, allow_partial=True)
        for op in self.operations:
            from .control_flow_operations import operation_owner, remap_control_entry
            control = operation_owner(op)
            if control is not None:
                bound.operations.append(remap_control_entry(op,
                    dict(enumerate(range(self.num_qubits))), dict(enumerate(range(self.num_bits))),
                    self.num_qubits, self.num_bits, owner=control.assign_parameters(mapping, strict=False),
                    target_circuit=bound))
                continue
            from .operations import execution_operation
            op=execution_operation(op,standardize_inverse=True)
            params = tuple(resolve_parameter(param, mapping, allow_partial=True) for param in op.params)
            metadata = native_configuration_copy(op.metadata, memo=_copy_memo, category="native_bound_metadata_copy")
            if "control_node" in metadata:
                metadata["control_node"] = metadata["control_node"].bind(mapping)
            if "gate" in metadata:
                metadata["gate"] = metadata["gate"].assign_parameters(mapping, strict=False)
            if "annotated_operation" in metadata:
                metadata["annotated_operation"] = metadata["annotated_operation"].assign_parameters(mapping, strict=False)
            if "instruction_object" in metadata:
                metadata["instruction_object"] = metadata["instruction_object"].assign_parameters(mapping, strict=False)
            output=Operation(op.name, op.targets, op.controls, params, metadata)
            from .operations import has_public_frame,capture_public_frame
            if has_public_frame(op):capture_public_frame(output,output.name,output.params)
            bound.operations.append(output)
        bound.measurements = list(self.measurements)
        bound.parameters
        if inplace:
            _replace_self(self, bound)
            return None
        return bound

    def bind_parameters(
        self,
        parameters: Union[Mapping[Union[Parameter, str], ParameterValue], Sequence[ParameterValue]],
        inplace: bool = False,
        strict: bool = True,
    ) -> "Circuit":
        bound=self.assign_parameters(parameters,strict=strict)
        if strict and bound.parameters:
            raise ValueError("QuantumBridge parameters are not bound: "+", ".join(parameter.name for parameter in bound.parameters))
        return _replace_self(self,bound) if inplace else bound

    def bind(self, mapping: Mapping[Union[Parameter, str], Real]) -> "Circuit":
        bound=Circuit.assign_parameters(self,mapping,strict=True,_copy_user_metadata=False)
        if bound.parameters:raise ValueError("QuantumBridge parameters are not bound: "+", ".join(parameter.name for parameter in bound.parameters))
        return bound

    def to_ir(self):
        from quantumbridge.ir.qb_ir import IRProgram

        return IRProgram.from_circuit(self)

    def to_openqasm(self) -> str:
        from quantumbridge.qasm import dumps

        return dumps(self)


def is_native_circuit(value):
    if type(value) is Circuit:return True
    from .public_circuit import QuantumCircuit
    return type(value) is QuantumCircuit


def _has_global_phase(circuit):
    return not isinstance(circuit.global_phase, Real) or circuit.global_phase != 0


def _index_map(source_size: int, target_size: int, selected: Optional[Sequence[int]], label: str, *, owner=None) -> dict[int, int]:
    import operator
    from numbers import Integral
    from itertools import islice
    from .errors import CircuitError
    from .gate_limits import current_gate_limits
    from .bits import is_bit,operand_index
    if source_size > target_size:raise CircuitError("Composed circuit is wider than its target.")
    current_gate_limits().check("max_decomposition_work",source_size)
    if selected is None:
        selected_tuple = tuple(range(source_size))
    else:
        if isinstance(selected,slice):selected=range(target_size)[selected]
        elif isinstance(selected,Integral) or is_bit(selected):selected=(selected,)
        try:selected_tuple=tuple(operand_index(owner,item,label=="qubits") if owner is not None and is_bit(item)
            else operator.index(item) for item in islice(iter(selected),source_size+1))
        except TypeError as exc:raise CircuitError("Compose wire mapping needs integer indices, not nested or nonintegral values.") from exc
        selected_tuple=tuple(item+target_size if item<0 else item for item in selected_tuple)
    if len(selected_tuple) != source_size:
        raise CircuitError(f"QuantumBridge compose {label} length must match the composed circuit.")
    if any(item < 0 or item >= target_size for item in selected_tuple):
        raise CircuitError(f"QuantumBridge compose {label} are outside the target circuit.")
    if len(set(selected_tuple)) != len(selected_tuple):
        raise CircuitError(f"QuantumBridge compose {label} must be distinct.")
    return {source: target for source, target in enumerate(selected_tuple)}


def _remap_metadata(metadata, clbit_map, *, copy_operation=True, remap_control=False):
    from quantumbridge.execution.array_ownership import native_configuration_copy
    from quantumbridge.execution.resource_context import current_resources
    shared = {key:value for key,value in metadata.items() if (remap_control and key=="control_node")
        or not copy_operation and key in {"gate","instruction_object","annotated_operation"}}
    result = native_configuration_copy({key:value for key,value in metadata.items() if key not in shared},
        category="native_remap_metadata", strict=current_resources() is not None)
    result.update(shared)
    if clbit_map is not None:
        if "instruction_clbits" in result:
            result["instruction_clbits"] = tuple(clbit_map[bit] for bit in result["instruction_clbits"])
        for key in ("condition", "gate_condition"):
            if result.get(key):
                result[key]["bits"] = [clbit_map[bit] for bit in result[key]["bits"]]
        frames = ([result["control_flow"]] if result.get("control_flow") else []) + result.get("control_flow_stack", [])
        for frame in frames:
            frame["condition"]["bits"] = [clbit_map[bit] for bit in frame["condition"]["bits"]]
    return result


def _remap_measurement(measurement, qubit_map, clbit_map, offset, source_size):
    position=measurement.metadata.get("operation_index",source_size)
    if type(position) is not int or not 0 <= position <= source_size:
        raise ValueError("Measurement operation_index is outside the composed source.")
    metadata = _remap_metadata(measurement.metadata, clbit_map)
    metadata["operation_index"] = offset + position
    return Measurement(qubit_map[measurement.wire], clbit_map[measurement.bit], measurement.kind, metadata)


def _remap_operation(op: Operation, qubit_map: Mapping[int, int], clbit_map=None, target_width=None, target_bits=None, *, identifiers=None, copy_operation=True, target_circuit=None) -> Operation:
    from .control_flow_operations import operation_owner, remap_control_entry
    if operation_owner(op) is not None:
        return remap_control_entry(op, qubit_map, clbit_map, target_width, target_bits,
            identifiers=identifiers, copy_operation=copy_operation, target_circuit=target_circuit)
    if copy_operation or "_native_inverse_qualification" in object.__getattribute__(op,"__dict__"):
        from .operations import execution_operation
        op=execution_operation(op,standardize_inverse=True)
    metadata = _remap_metadata(op.metadata, clbit_map, copy_operation=copy_operation,remap_control=True)
    if "control_node" in metadata:
        metadata["control_node"] = metadata["control_node"].remap(qubit_map, clbit_map,
            target_width or max(qubit_map.values()) + 1,
            target_bits if target_bits is not None else max(clbit_map.values(), default=-1) + 1,
            identifiers=identifiers, copy_operation=copy_operation)
    result = Operation(
        name=op.name,
        targets=tuple(qubit_map[wire] for wire in op.targets),
        controls=tuple(qubit_map[wire] for wire in op.controls),
        params=op.params,
        metadata=metadata,
    )
    from .operations import has_public_frame,capture_public_frame
    if has_public_frame(op):capture_public_frame(result,op.name,op.params)
    basis=getattr(op,"_public_matrix_basis",None)
    if basis is not None:object.__setattr__(result,"_public_matrix_basis",basis)
    return result


def _remap_circuit(circuit, qubits, bits, width, num_bits, *, identifiers=None, copy_operation=True):
    from quantumbridge.execution.array_ownership import native_configuration_copy
    from quantumbridge.execution.resource_context import current_resources
    from .classical_scope import entries, set_entries
    out=empty_execution_circuit(circuit,num_qubits=width,num_bits=num_bits)
    execution_metadata(out).pop("registers",None)
    records=[((identifiers or {}).get(node.value[0],node),scope,ordered) for node,scope,ordered in entries(circuit)]
    out.operations=[_remap_operation(op,qubits,bits,width,num_bits,identifiers=identifiers,copy_operation=copy_operation) for op in circuit.operations]
    out.measurements=[_remap_measurement(item,qubits,bits,0,len(circuit.operations)) for item in circuit.measurements]
    set_entries(out,records)
    entries(out)
    return out


def _operation_with_metadata(op: Operation, metadata: Mapping[str, Any]) -> Operation:
    merged = deepcopy(op.metadata)
    merged.update(metadata)
    return Operation(op.name, op.targets, op.controls, op.params, merged)


def _circuit_with_operation_metadata(circuit: Circuit, metadata: Mapping[str, Any], start: bool = True) -> Circuit:
    out = circuit.copy()
    first = next(iter(out.iter_instructions()), None)

    def merged(item):
        previous = deepcopy(item.metadata)
        result = deepcopy(previous)
        result.update(deepcopy(metadata))
        if previous.get("condition") and not previous.get("control_flow"):
            result["gate_condition"] = previous["condition"]
        frame = deepcopy(metadata["control_flow"])
        frame["start"] = start and item is first
        result["control_flow_stack"] = [frame] + previous.get("control_flow_stack", [])
        return result

    out.operations = [Operation(op.name, op.targets, op.controls, op.params, merged(op)) for op in out.operations]
    out.measurements = [Measurement(item.wire, item.bit, item.kind, merged(item)) for item in out.measurements]
    return out


def _normalize_condition(condition: Union[dict[str, Any], tuple[Union[int, Sequence[int]], int]]) -> dict[str, Any]:
    if isinstance(condition, dict):
        bits = condition.get("bits", condition.get("clbits"))
        value = condition.get("value")
    else:
        bits, value = condition
    bits_tuple = (bits,) if isinstance(bits, int) else tuple(bits)
    if not bits_tuple:
        raise ValueError("QuantumBridge conditional blocks require at least one classical bit.")
    return {"bits": [int(bit) for bit in bits_tuple], "value": int(value)}


def _inverse_operation(op: Operation, annotated=False) -> Operation:
    from .operations import has_public_frame,execution_operation,capture_public_frame,public_matrix_frame
    from .standard_gates import standard_type_name
    if has_public_frame(op) and standard_type_name(op.metadata.get("gate")) is None and not public_matrix_frame(op):
        op=execution_operation(op)
    if "instruction_object" in op.metadata:
        from .annotated_operation import AnnotatedOperation
        from .instruction_parameters import owned_copy
        inverse = op.metadata["instruction_object"].inverse(annotated=annotated)
        if has_public_frame(op) and type(inverse) is AnnotatedOperation:inverse=inverse.copy()
        metadata = owned_copy(op.metadata)
        metadata.pop("instruction_object")
        if type(inverse) is Instruction: metadata["instruction_object"] = inverse
        else:
            if not inverse.num_clbits:metadata.pop("instruction_clbits")
            metadata["annotated_operation" if type(inverse) is AnnotatedOperation else "gate"] = inverse
        result=Operation(inverse.name, op.targets, params=tuple(inverse.params), metadata=metadata)
        if has_public_frame(op):
            capture_public_frame(result,result.name,result.params)
        return result
    if "annotated_operation" in op.metadata:
        annotation = op.metadata["annotated_operation"].inverse()
        from .gates import _copy
        metadata = _copy(op.metadata)
        metadata["annotated_operation"] = annotation.copy()
        from .standard_gates import invocation_params
        result=Operation(annotation.name, op.targets, op.controls, invocation_params(annotation), metadata)
        if has_public_frame(op):capture_public_frame(result,result.name,result.params)
        return result
    if "gate" in op.metadata:
        gate = op.metadata["gate"].inverse(annotated=annotated)
        metadata = deepcopy(op.metadata)
        from .annotated_operation import AnnotatedOperation
        if type(gate) is AnnotatedOperation:
            metadata.pop("gate")
            gate=gate.copy() if has_public_frame(op) else gate
            metadata["annotated_operation"] = gate
        else: metadata["gate"] = gate
        from .standard_gates import invocation_params
        result=Operation(gate.name, op.targets, op.controls, invocation_params(gate), metadata)
        if has_public_frame(op):
            capture_public_frame(result,result.name,result.params)
        return result
    if "controlled_base" in op.metadata:
        metadata = deepcopy(op.metadata)
        description = metadata["controlled_base"]
        description["inverse"] = not description.get("inverse", False)
        return Operation(op.name, op.targets, op.controls, op.params, metadata)
    if op.name in {"id", "i", "x", "y", "z", "h", "cx", "cy", "cz", "ch", "swap", "dcx", "ecr", "ccx", "cswap"}:
        return _operation_with_metadata(op, {})
    inverse_name = {"s": "sdg", "sdg": "s", "sx": "sxdg", "sxdg": "sx", "t": "tdg", "tdg": "t", "csx": "csxdg", "csxdg": "csx"}.get(op.name)
    if inverse_name:
        return Operation(inverse_name, op.targets, op.controls, op.params, deepcopy(op.metadata))
    if op.name in {"rx", "ry", "rz", "phase", "p", "u1", "crx", "cry", "crz", "cp", "cphase", "rxx", "ryy", "rzz"}:
        return Operation(op.name, op.targets, op.controls, (-op.params[0],), deepcopy(op.metadata))
    if op.name in {"u", "u3"}:
        theta, phi, lam = op.params
        return Operation(op.name, op.targets, op.controls, (-theta, -lam, -phi), deepcopy(op.metadata))
    if op.name == "u2":
        phi, lam = op.params
        return Operation("u", op.targets, op.controls, (-np.pi / 2, -lam, -phi), deepcopy(op.metadata))
    matrix = op.metadata.get("matrix")
    if matrix is not None:
        from .unitary_inverse import primitive_matrix_inverse
        qualified=primitive_matrix_inverse(op)
        if qualified is not None: return qualified
        metadata = deepcopy(op.metadata)
        metadata["matrix"] = np.asarray(matrix, dtype=complex).conj().T
        metadata["inverse_of"] = op.name
        return Operation(f"{op.name}_dg", op.targets, op.controls, op.params, metadata)
    raise ValueError(f"QuantumBridge cannot invert operation {op.name!r}.")


def _controlled_matrix(base: np.ndarray, num_ctrl_qubits: int, ctrl_state: int) -> np.ndarray:
    base = np.asarray(base, dtype=complex)
    base_dim = base.shape[0]
    ctrl_dim = 1 << num_ctrl_qubits
    out = np.eye(ctrl_dim * base_dim, dtype=complex)
    start = ctrl_state * base_dim
    out[start : start + base_dim, start : start + base_dim] = base
    return out


def _replace_self(original: Circuit, replacement: Circuit) -> Circuit:
    from .bits import copy_bits
    copy_bits(replacement,original)
    original.num_qubits = replacement.num_qubits
    original.num_bits = replacement.num_bits
    original.name = replacement.name
    original.operations = replacement.operations
    original.measurements = replacement.measurements
    original.metadata = replacement.metadata
    if is_separated(replacement):original._execution_metadata=replacement._execution_metadata
    elif is_separated(original):original._execution_metadata=dict(replacement.metadata)
    original.__dict__.pop("_classical_identifiers",None)
    if "_classical_identifiers" in replacement.__dict__:
        original._classical_identifiers=replacement._classical_identifiers
    return original


def _calibration_key(name: str, qubits: tuple[int, ...], params: tuple[ParameterValue, ...]) -> str:
    param_key = ",".join(repr(param) for param in params)
    return f"{name}|{','.join(str(wire) for wire in qubits)}|{param_key}"


def _parameter_mapping_from_values(
    ordered_parameters: Sequence[Parameter],
    parameters: Union[Mapping[Union[Parameter, str], ParameterValue], Sequence[ParameterValue]],
    strict: bool,
) -> dict[Union[Parameter, str], ParameterValue]:
    from quantumbridge.execution.resource_context import current_resources
    scope=current_resources()
    if scope is not None and type(parameters) not in (dict,list,tuple,np.ndarray):
        from quantumbridge.core.classical import UnsupportedFeature
        raise UnsupportedFeature("managed_parameter_container","Managed binding requires an explicit native mapping or numeric sequence.")
    def sequence(values):
        if scope is not None:
            if type(values) not in (tuple,list,np.ndarray):
                from quantumbridge.core.classical import UnsupportedFeature
                raise UnsupportedFeature("managed_parameter_values","Managed parameter sequences must be explicit numeric containers.")
            if type(values) is np.ndarray:
                from quantumbridge.execution.array_reservations import numeric_shape
                if len(numeric_shape(values,real_only=True))!=1:raise ValueError("Parameter values require a one-dimensional sequence.")
        return tuple(values)
    def checked(value):
        if scope is not None:
            if type(value) not in (Parameter,ParameterVectorElement,ParameterExpression):
                from quantumbridge.execution.array_reservations import numeric_shape
                if numeric_shape(value,real_only=True)!=():raise TypeError("Gate parameter values must be real scalars or native expressions.")
        elif not isinstance(value,(Real,Parameter,ParameterExpression)):
            raise TypeError("Gate parameter values must be real scalars or native expressions.")
        return value
    if isinstance(parameters, ABCMapping):
        known=set(ordered_parameters)
        names={parameter.name:parameter for parameter in ordered_parameters}
        mapping={}
        for key,value in parameters.items():
            if scope is not None and type(key) not in (str,Parameter,ParameterVectorElement,ParameterVector):
                from quantumbridge.core.classical import UnsupportedFeature
                raise UnsupportedFeature("managed_parameter_key","Managed binding keys must be native parameters, vectors or strings.")
            entries=((key,value),)
            if isinstance(key,ParameterVector):
                values=sequence(value)
                if len(values)!=len(key):raise ValueError("ParameterVector binding length must match its length.")
                entries=zip(key,values)
            for parameter,item in entries:
                parameter=names.get(parameter,parameter) if isinstance(parameter,str) else parameter
                if parameter not in known:
                    if strict:raise ValueError(f"QuantumBridge received unknown parameter bindings: {parameter}")
                    continue
                if parameter in mapping:raise ValueError("A parameter was bound more than once.")
                mapping[parameter]=checked(item)
        return mapping
    values = sequence(parameters)
    if len(values) != len(ordered_parameters):
        raise ValueError("QuantumBridge sequence parameter binding length must match circuit.num_parameters.")
    return {parameter:checked(value) for parameter,value in zip(ordered_parameters,values)}
