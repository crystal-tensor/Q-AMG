# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Coherent records and bounded unitary lowering of native measurement feedback."""

from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import islice, product
from numbers import Integral, Real
from math import pi

import numpy as np

from quantumbridge.core import Circuit, Measurement, Operation
from quantumbridge.core.classical import (ClassicalExpression, ClassicalStore,
    UnsupportedFeature, classical_eval, validate_classical_expr)
from quantumbridge.core.control_flow import condition_expression, ExecutionLimits, validate_control_flow
from quantumbridge.core.circuit import _controlled_matrix
from quantumbridge.compiler.decomposition import GateDecompositionPass
from quantumbridge.diff.qnode_engine import map_measurements
from quantumbridge.utils.math import Hamiltonian, PauliString
from .dispatch import transform
from .program import check_batch, output_tree, resolve_limits


@dataclass(frozen=True)
class DeferLimits:
    max_mid_measurements: int = 64
    max_ancillas: int = 64
    max_controls: int = 8
    max_controlled_gate_expansion: int = 20000
    max_transformed_operations: int = 100000
    max_transform_depth: int = 32
    max_matrix_qubits: int = 6

    def __post_init__(self):
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 1
               for value in self.__dict__.values()):
            raise ValueError("Defer limits must be positive integers.")

    def check(self, name, value):
        if value > getattr(self, name):
            raise MemoryError("Native coherent deferral exceeded " + name)


def _unsupported(feature):
    raise UnsupportedFeature("defer_" + feature, "Native coherent deferral does not support " + feature + ".")


def _condition_bits(expression, records, width, limits):
    pending, bits, nodes = [expression], set(), 0
    while pending:
        item = pending.pop()
        nodes += 1
        limits.check("max_controlled_gate_expansion", nodes)
        if item.op not in {"bit", "register", "literal", "eq", "ne", "and", "or", "xor", "invert", "not", "cast", "logical_and", "logical_or"}:
            _unsupported("classical_arithmetic_or_storage")
        if item.op == "bit": bits.add(item.value)
        if item.op == "register": bits.update(item.value)
        pending.extend(item.args)
    validate_classical_expr(expression, num_bits=width)
    if not bits <= records.keys():
        _unsupported("undefined_measurement_value")
    limits.check("max_controls", len(bits))
    return tuple(sorted(bits))


class _Lowering:
    def __init__(self, source, ancillas, limits, transform_limits):
        self.source, self.ancillas = source, iter(ancillas)
        self.limits, self.transform_limits = limits, transform_limits
        metadata = deepcopy(source.metadata)
        metadata.pop("control_flow", None)
        self.out = Circuit(source.num_qubits + len(ancillas), source.num_bits, source.name, metadata)
        self.out.global_phase = 0.
        self.records, self.receipts, self.dependencies = {}, [], []
        self.controlled_work, self.max_depth = 0, 0

    def reserve(self, count, controlled=False):
        self.limits.check("max_transformed_operations", len(self.out.operations) + count)
        self.transform_limits.check("max_operations_per_tape", len(self.out.operations) + count)
        if controlled:
            self.limits.check("max_controlled_gate_expansion", self.controlled_work + count)

    def emit(self, name, wires, params=(), metadata=None, *, controlled=False):
        self.reserve(1, controlled)
        self.out.append(name, wires, params=params, metadata=metadata)
        self.controlled_work += int(controlled)

    def projector_phase(self, controls, angle):
        if isinstance(angle, Real) and angle == 0:
            return
        if not controls:
            self.out.global_phase = self.out.global_phase + angle
            return
        if len(controls) > self.limits.max_controls + 1:
            raise MemoryError("Controlled phase exceeds the coherent control budget.")
        # Expand commuting Z projectors, not a matrix over the full register.
        count = sum(2 * (mask.bit_count() - 1) + 1 for mask in range(1, 1 << len(controls)))
        self.reserve(count, True)
        self.out.global_phase = self.out.global_phase + angle / (1 << len(controls))
        for mask in range(1, 1 << len(controls)):
            wires = [wire for index, wire in enumerate(controls) if mask >> index & 1]
            for wire in wires[:-1]: self.emit("cx", (wire, wires[-1]), controlled=True)
            self.emit("rz", (wires[-1],), (-2 * (-1) ** len(wires) * angle / (1 << len(controls)),), controlled=True)
            for wire in reversed(wires[:-1]): self.emit("cx", (wire, wires[-1]), controlled=True)

    def rotation(self, axis, target, angle, controls):
        if len(controls) == 1:
            self.emit("c" + axis, controls + (target,), (angle,), controlled=True)
            return
        self.limits.check("max_controls", len(controls))
        self.reserve((1 << len(controls)) * (2 * len(controls) + 1) + 4, True)
        if axis == "rx": self.emit("h", (target,), controlled=True)
        if axis == "ry":
            self.emit("sdg", (target,), controlled=True)
            self.emit("h", (target,), controlled=True)
        for mask in range(1 << len(controls)):
            wires = [wire for index, wire in enumerate(controls) if mask >> index & 1]
            for wire in wires: self.emit("cx", (wire, target), controlled=True)
            self.emit("rz", (target,), ((-1) ** len(wires) * angle / (1 << len(controls)),), controlled=True)
            for wire in reversed(wires): self.emit("cx", (wire, target), controlled=True)
        if axis == "ry":
            self.emit("h", (target,), controlled=True)
            self.emit("s", (target,), controlled=True)
        if axis == "rx": self.emit("h", (target,), controlled=True)

    def controlled(self, operation, controls):
        controls = tuple(dict.fromkeys(controls))
        self.limits.check("max_controls", len(controls))
        name, wires, params = operation.name, operation.wires, operation.params
        if set(controls) & set(wires):
            _unsupported("control_target_alias")
        existing = {"cx": ("x", 1), "cy": ("y", 1), "cz": ("z", 1), "ch": ("h", 1),
                    "crx": ("rx", 1), "cry": ("ry", 1), "crz": ("rz", 1), "cp": ("phase", 1),
                    "cphase": ("phase", 1), "csx": ("sx", 1), "csxdg": ("sxdg", 1),
                    "ccx": ("x", 2), "cswap": ("swap", 1)}
        if name in existing:
            base, count = existing[name]
            return self.controlled(Operation(base, wires[count:], params=params, metadata=operation.metadata), controls + wires[:count])
        if name in {"id", "i", "identity"}: return
        if name == "global_phase": return self.projector_phase(controls, params[0])
        if name == "x":
            if len(controls) <= 2:
                self.emit("cx" if len(controls) == 1 else "ccx", controls + wires, controlled=True)
            else:
                self.emit("h", wires, controlled=True)
                self.projector_phase(controls + wires, pi)
                self.emit("h", wires, controlled=True)
            return
        if name in {"y", "z"}:
            self.emit("sdg" if name == "y" else "h", wires, controlled=True)
            self.controlled(Operation("x", wires), controls)
            self.emit("s" if name == "y" else "h", wires, controlled=True)
            return
        if name in {"rx", "ry", "rz"}:
            return self.rotation(name, wires[0], params[0], controls)
        if name in {"phase", "p", "u1", "s", "sdg", "t", "tdg"}:
            angle = params[0] if params else {"s": pi / 2, "sdg": -pi / 2, "t": pi / 4, "tdg": -pi / 4}[name]
            return self.projector_phase(controls + wires, angle)
        if name == "swap":
            first, second = wires
            for left, right in ((first, second), (second, first), (first, second)):
                self.controlled(Operation("x", (right,)), controls + (left,))
            return
        if "matrix" in operation.metadata:
            if params: _unsupported("parameterized_opaque_gate")
            self.limits.check("max_matrix_qubits", len(controls) + len(wires))
            self.reserve(1, True)
            matrix = _controlled_matrix(operation.metadata["matrix"], len(controls), (1 << len(controls)) - 1)
            self.emit("deferred_unitary", controls + wires, metadata={"matrix": matrix}, controlled=True)
            return
        base = Circuit(self.out.num_qubits).append(name, wires, params=params)
        expanded = GateDecompositionPass({"rx", "ry", "rz", "cx"}).run(base,
            max_operations=self.limits.max_controlled_gate_expansion,
            max_expansion_steps=self.limits.max_controlled_gate_expansion,
            max_depth=self.limits.max_transform_depth).circuit
        self.projector_phase(controls, expanded.global_phase)
        for child in expanded.operations: self.controlled(child, controls)

    def conditional_gate(self, operation, active):
        if not active:
            self.emit(operation.name, operation.wires, operation.params, deepcopy(operation.metadata))
            return
        if set(operation.metadata) - {"label", "matrix"}:
            _unsupported("conditional_gate_metadata")
        zeros = tuple(wire for wire, value in active.items() if not value)
        start = len(self.out.operations)
        self.reserve(2 * len(zeros) + 1, True)
        for wire in zeros: self.emit("x", (wire,), controlled=True)
        self.controlled(operation, tuple(active))
        for wire in reversed(zeros): self.emit("x", (wire,), controlled=True)
        reverse = {wire: bit for bit, wire in self.records.items()}
        self.dependencies.append({"operation": operation.name, "source_wires": operation.wires,
            "record_conditions": tuple((reverse[wire], value) for wire, value in active.items()),
            "output_operation_span": (start, len(self.out.operations))})

    def enabled(self, expression, active, value):
        bits = _condition_bits(expression, self.records, self.source.num_bits, self.limits)
        self.limits.check("max_controlled_gate_expansion", 1 << len(bits))
        store = ClassicalStore(self.source.num_bits)
        for assignment in product((0, 1), repeat=len(bits)):
            for bit, outcome in zip(bits, assignment): store.bits[bit] = outcome
            if bool(classical_eval(expression, store)) != value: continue
            joined = dict(active)
            for bit, outcome in zip(bits, assignment):
                wire = self.records[bit]
                if wire in joined and joined[wire] != outcome: break
                joined[wire] = outcome
            else:
                self.limits.check("max_controls", len(joined))
                yield joined

    def walk(self, body, active=None, depth=0):
        active = dict(active or {})
        self.limits.check("max_transform_depth", depth)
        self.max_depth = max(depth, self.max_depth)
        phase = body.global_phase
        zeros = tuple(wire for wire, value in active.items() if not value)
        if not isinstance(phase, Real) or phase != 0:
            for wire in zeros: self.emit("x", (wire,), controlled=True)
            self.projector_phase(tuple(active), phase)
            for wire in reversed(zeros): self.emit("x", (wire,), controlled=True)
        for item in body.iter_instructions():
            if isinstance(item, Measurement):
                if active: _unsupported("measurement_inside_conditional_body")
                if item.bit in self.records: _unsupported("measurement_value_shadowing")
                if any(key in item.metadata for key in ("postselect", "postselection", "condition")):
                    _unsupported("postselection_or_conditional_measurement")
                wire = next(self.ancillas)
                self.records[item.bit] = wire
                identity = item.metadata.get("measurement_id", "m" + str(len(self.receipts)))
                if not isinstance(identity, str) or not identity:
                    raise ValueError("Measurement identity must be a nonempty string.")
                if any(row["measurement_id"] == identity for row in self.receipts):
                    _unsupported("measurement_identity_alias")
                self.emit("cx", (item.wire, wire))
                self.receipts.append({"measurement_id": identity, "source_wire": item.wire,
                    "source_bit": item.bit, "ancilla": wire, "initial_state": 0})
                continue
            node = item.metadata.get("control_node")
            if node is not None:
                if node.kind != "if": _unsupported("dynamic_loops_or_classical_store")
                if set(item.metadata) != {"control_node"}: _unsupported("control_side_effect_metadata")
                for value, branch in ((True, node.data["body"]), (False, node.data["alternate"])):
                    if branch is not None:
                        for joined in self.enabled(node.data["condition"], active, value): self.walk(branch, joined, depth + 1)
                continue
            if item.name in {"reset", "kraus", "measure", "control_flow", "initialize", "set_statevector", "store"}:
                _unsupported("nonunitary_or_classical_instruction")
            if any(isinstance(value, ClassicalExpression) for value in item.params):
                _unsupported("measurement_dependent_gate_parameter")
            if any(key in item.metadata for key in ("simulator_instruction", "postselect", "postselection", "control_flow", "control_flow_stack", "gate_condition")):
                _unsupported("snapshot_noise_or_legacy_control_metadata")
            if "matrix" in item.metadata:
                self.limits.check("max_matrix_qubits", len(item.wires))
                matrix = np.asarray(item.metadata["matrix"], dtype=complex)
                size = 1 << len(item.wires)
                if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)) or not np.allclose(matrix.conj().T @ matrix, np.eye(size), atol=1e-12, rtol=0):
                    _unsupported("nonunitary_matrix")
            if item.metadata.get("requires_zero_input") and active: _unsupported("conditional_state_preparation")
            if item.name in {"delay", "barrier"}:
                if item.name == "delay" or active: _unsupported("conditional_or_timed_directive")
            condition = item.metadata.get("condition")
            clean = replace(item, metadata={key: deepcopy(value) for key, value in item.metadata.items() if key != "condition"})
            if active and set(clean.metadata) - {"label", "matrix"}: _unsupported("conditional_gate_metadata")
            if condition is None:
                self.conditional_gate(clean, active)
            else:
                for joined in self.enabled(condition_expression(condition), active, True): self.conditional_gate(clean, joined)


@transform
def defer_measurements(tape, *, max_ancillas=None, ancilla_wires=None, allow_postselect=False,
                       defer_limits=None, limits=None):
    """Replace supported Z measurements/feedback with a larger coherent circuit.

    Each unconditional measurement owns a fresh, zero-initialized record wire.
    Logical terminal outputs exclude those wires unless records are requested.
    Reset, postselection, noisy devices and runtime loops need other contracts.
    """
    limits = resolve_limits(limits)
    bounded = DeferLimits() if defer_limits is None else defer_limits
    if not isinstance(bounded, DeferLimits): raise TypeError("defer_limits requires DeferLimits.")
    if not isinstance(allow_postselect, bool): raise TypeError("allow_postselect requires bool.")
    if allow_postselect: _unsupported("postselection")
    check_batch((tape,), limits)
    source = tape.circuit
    limits.check("max_postprocessing_nodes", source.num_bits)
    count, pending = 0, [(source, 0)]
    while pending:
        body, depth = pending.pop()
        bounded.check("max_transform_depth", depth)
        if depth and body.measurements: _unsupported("measurement_inside_conditional_body")
        count += len(body.measurements)
        bounded.check("max_mid_measurements", count)
        for item in body.operations:
            node = item.metadata.get("control_node")
            if node is not None:
                if node.kind != "if": _unsupported("dynamic_loops_or_classical_store")
                pending.extend((child, depth + 1) for child in node.bodies)
    validate_control_flow(source, limits=ExecutionLimits(max_control_depth=bounded.max_transform_depth))
    capacity = bounded.max_ancillas if max_ancillas is None else max_ancillas
    if isinstance(capacity, bool) or not isinstance(capacity, Integral) or capacity < 0:
        raise ValueError("max_ancillas must be a nonnegative integer.")
    bounded.check("max_ancillas", count)
    if count > capacity: raise MemoryError("Coherent measurement records exceed max_ancillas.")
    expected = tuple(range(source.num_qubits, source.num_qubits + count))
    ancillas = expected if ancilla_wires is None else tuple(islice(ancilla_wires, count + 1))
    if len(ancillas) != count or any(isinstance(wire, bool) or not isinstance(wire, Integral) for wire in ancillas) or set(ancillas) != set(expected):
        raise ValueError("Ancilla wires must be distinct fresh trailing wires, without aliases or gaps.")
    lowering = _Lowering(source, ancillas, bounded, limits)
    lowering.walk(source)
    receipt = {"native_coherent_deferral": True, "logical_wires": source.num_qubits,
        "transformed_wires": lowering.out.num_qubits, "ancilla_count": count,
        "records": lowering.receipts, "logical_output_wires": tuple(range(source.num_qubits)),
        "control_dependencies": lowering.dependencies,
        "transformed_operations": len(lowering.out.operations), "controlled_gate_expansion": lowering.controlled_work,
        "max_source_control_depth": lowering.max_depth, "limits": dict(bounded.__dict__),
        "semantics": "coherent record purification; logical outputs trace unrequested record wires",
        "warning": "Additional initialized wires are real resources, not zero-overhead hardware execution."}
    if not count and isinstance(source.metadata.get("deferred_measurements"), dict):
        receipt = deepcopy(source.metadata["deferred_measurements"])
    lowering.out.metadata["deferred_measurements"] = receipt

    def terminal(request):
        metadata = dict(request.metadata or {})
        observable = request.observable
        for wire in request.wires or ():
            source._check_wire(wire)
        records = metadata.pop("measurement_records", ())
        if request.kind == "state" and count: _unsupported("full_state_output")
        if request.kind not in {"expval", "var", "probs", "density_matrix", "sample", "counts", "pauli_group", "state"}:
            _unsupported("terminal_measurement")
        if records:
            if request.kind not in {"probs", "sample", "counts"}: _unsupported("record_measurement_kind")
            bits = [record.value for record in records if isinstance(record, ClassicalExpression) and record.op == "bit"]
            if len(bits) != len(records) or len(set(bits)) != len(bits) or not set(bits) <= lowering.records.keys():
                _unsupported("undefined_or_aliased_terminal_records")
            metadata["deferred_record_bits"] = tuple(bits)
            wires = tuple(lowering.records[bit] for bit in bits) + tuple(request.wires or ())
        else:
            wires = request.wires
            if wires is None and request.kind in {"probs", "density_matrix", "sample", "counts"}:
                wires = tuple(range(source.num_qubits))
        if request.kind in {"expval", "var"}:
            if hasattr(observable, "to_quantumbridge"): observable = observable.to_quantumbridge()
            if hasattr(observable, "to_hamiltonian"): observable = observable.to_hamiltonian()
            terms = observable.terms if isinstance(observable, Hamiltonian) else ((1., observable),)
            for _, word in terms:
                if isinstance(word, PauliString):
                    for wire, _ in word.terms: source._check_wire(wire)
            if not isinstance(observable, (PauliString, Hamiltonian)) and count:
                _unsupported("unmapped_dense_observable")
        return replace(request, observable=observable, wires=wires, metadata=metadata or None)

    child = tape.copy()
    child.circuit = lowering.out
    child.output_tree = map_measurements(output_tree(tape), terminal)
    child.measurements = []
    map_measurements(child.output_tree, lambda request: child.measurements.append(request))
    check_batch((child,), limits)

    def process(results):
        if len(results) != 1: raise ValueError("Coherent deferral expects one child result.")
        return results[0]

    def validate_device(device):
        if receipt["ancilla_count"] and (getattr(device, "noise_model", None) is not None or getattr(device, "initial_state", None) is not None):
            _unsupported("noisy_device_or_nonzero_record_initialization")
        maximum = getattr(device, "max_qubits", None)
        if maximum is not None and lowering.out.num_qubits > maximum:
            raise ValueError("Coherent deferral requires additional device wires beyond max_qubits.")
        execution = getattr(device, "execution_limits", None)
        if execution is not None and (len(lowering.out.operations) > execution.max_instruction_executions or lowering.max_depth > execution.max_control_depth):
            raise MemoryError("Coherent deferral exceeds native device execution limits.")

    process.native_validate_device = validate_device
    process.last_metadata = receipt
    return (child,), process
