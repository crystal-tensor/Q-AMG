# Copyright 2026 QuantumBridge Contributors. Licensed under Apache-2.0.
"""Read-only native checks of returned QASM/QLISP and classical bit mapping.

Sparse physical registers are compacted before using the existing simulator.
Only the reported all-zero-input circuit behavior is audited, not an arbitrary
input unitary, timing, pulse calibration or actual physical noise.
"""

from __future__ import annotations

import ast
from dataclasses import replace

import numpy as np

from quantumbridge.core import Circuit
from quantumbridge.core.circuit import _remap_circuit
from quantumbridge.devices import StatevectorDevice
from quantumbridge.serialization import deserialize_circuit
from quantumbridge.serialization.limits import SerializationLimits
from quantumbridge.utils.math import probabilities_from_state


def _simulate(circuit, physical_labels):
    if circuit.has_dynamic_operations:
        raise ValueError("diagnostics require a unitary circuit with terminal measurements")
    bybit = {}
    for measurement in circuit.measurements:
        if measurement.bit in bybit:
            raise ValueError("duplicate measurement bit")
        bybit[measurement.bit] = measurement.wire
    if (not bybit or set(bybit) != set(range(circuit.num_bits))
            or len(set(bybit.values())) != len(bybit)):
        raise ValueError("unique contiguous measurement outputs required")
    wires = [bybit[bit] for bit in range(circuit.num_bits)]
    state = StatevectorDevice(max_qubits=6).statevector(circuit)
    probabilities = probabilities_from_state(state, circuit.num_qubits, wires)
    ordered = state.reshape((2,) * circuit.num_qubits).transpose(wires).reshape(-1) if len(wires) == circuit.num_qubits else None
    return {"active_physical_qubits": physical_labels,
            "physical_measurement_map": {bit: physical_labels[wire] for bit, wire in bybit.items()},
            "probabilities": [probabilities.get(format(i, f"0{circuit.num_bits}b"), 0) for i in range(1 << circuit.num_bits)],
            "ordered_amplitudes": None if ordered is None else [[float(z.real), float(z.imag)] for z in ordered],
            "operations": len(circuit.operations), "classical_order": "c0-first",
            "scope": "native ideal all-zero-input execution of returned source only"}


def compiled_probabilities(source):
    """Parse provider QASM with native limits, then compact active physical wires."""
    limits = SerializationLimits(max_input_bytes=1_000_000, max_instructions=20000,
                                 max_gate_expansion=20000, max_clbits=6)
    # The returned provider dialect uses lowercase u without a qelib1 definition.
    circuit = deserialize_circuit(source, "qasm2", limits=limits, strict=False)
    if circuit.has_dynamic_operations:
        raise ValueError("dynamic physical circuits are not supported by this diagnostic")
    active = sorted({wire for op in circuit.operations if op.name not in {"barrier", "delay"}
                     for wire in op.controls + op.targets}
                    | {measurement.wire for measurement in circuit.measurements})
    if not 1 <= len(active) <= 6:
        raise ValueError("compiled diagnostic permits at most six active qubits")
    # Keep directive positions so terminal measurement operation_index stays valid.
    circuit.operations = [replace(op, targets=tuple(w for w in op.targets if w in active),
                                  controls=tuple(w for w in op.controls if w in active))
                          if op.name in {"barrier", "delay"} else op for op in circuit.operations]
    compact = _remap_circuit(circuit, {physical: dense for dense, physical in enumerate(active)},
                             {i: i for i in range(circuit.num_bits)}, len(active), circuit.num_bits)
    return _simulate(compact, active)


def _wire(value):
    if not isinstance(value, str) or not value.startswith("Q") or not value[1:].isdigit():
        raise ValueError("QLISP wire must be Q followed by a physical index")
    index = int(value[1:])
    if index >= 4096:
        raise ValueError("physical wire exceeds bounded diagnostic register")
    return index


def qlisp_probabilities(source):
    """Independent bounded U/CZ/Barrier/Measure public QLISP protocol translation."""
    if not isinstance(source, str) or len(source.encode()) > 1_000_000:
        raise ValueError("bounded QLISP literal source required")
    try:
        tree = ast.parse(source, mode="eval")
        stack, nodes = [(tree, 0)], 0
        while stack:
            node, depth = stack.pop()
            nodes += 1
            if depth > 32 or nodes > 100000:
                raise ValueError("QLISP literal budget exceeded")
            stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
        items = ast.literal_eval(tree)
    except (SyntaxError, RecursionError, TypeError) as exc:
        raise ValueError("invalid QLISP literal") from exc
    if not isinstance(items, (list, tuple)) or not 1 <= len(items) <= 20000:
        raise ValueError("bounded QLISP instruction sequence required")
    decoded, active, measured, bits = [], set(), False, set()
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("QLISP item must contain operation and wire operands")
        operation, operands = item
        if operation == "Barrier":
            continue
        if isinstance(operation, tuple) and len(operation) == 2 and operation[0] == "Measure":
            bit = operation[1]
            if type(bit) is not int or not 0 <= bit < 6 or bit in bits:
                raise ValueError("QLISP measurement index must be unique and bounded")
            wire = _wire(operands)
            decoded.append(("measure", wire, bit))
            active.add(wire)
            bits.add(bit)
            measured = True
            continue
        if measured:
            raise ValueError("QLISP gates after measurement are not supported")
        if isinstance(operation, tuple) and len(operation) == 4 and operation[0] == "U":
            angles = operation[1:]
            if any(type(angle) not in (int, float) or not np.isfinite(angle) for angle in angles):
                raise ValueError("QLISP U angles must be finite real literals")
            wire = _wire(operands)
            decoded.append(("u", wire, angles))
            active.add(wire)
        elif operation == "CZ" and isinstance(operands, (list, tuple)) and len(operands) == 2:
            wires = tuple(_wire(operand) for operand in operands)
            if wires[0] == wires[1]:
                raise ValueError("QLISP CZ operands must differ")
            decoded.append(("cz", wires, ()))
            active.update(wires)
        else:
            raise ValueError("unsupported QLISP instruction in bounded diagnostic")
    if not 1 <= len(active) <= 6 or not bits or bits != set(range(len(bits))):
        raise ValueError("QLISP active wires and measurement outputs must be bounded")
    physical = sorted(active)
    mapping = {wire: dense for dense, wire in enumerate(physical)}
    circuit = Circuit(len(physical), len(bits))
    for name, wires, params in decoded:
        if name == "measure":
            circuit.measure(mapping[wires], params)
        elif name == "u":
            circuit.append("u", (mapping[wires],), params=params)
        else:
            circuit.cz(mapping[wires[0]], mapping[wires[1]])
    return _simulate(circuit, physical)
