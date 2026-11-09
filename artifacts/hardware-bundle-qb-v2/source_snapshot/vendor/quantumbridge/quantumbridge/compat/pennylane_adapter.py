# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.
#
# Adapter Integration: this module optionally accepts PennyLane objects.
# No PennyLane source code is copied into this file.

from __future__ import annotations

from quantumbridge.utils.math import PauliString
from quantumbridge.core import Circuit
from typing import Optional


def observable_from_pennylane(observable) -> PauliString:
    name = getattr(observable, "name", None)
    wires = list(getattr(observable, "wires", []))
    if len(wires) != 1:
        raise ValueError("QuantumBridge PennyLane observable bridge currently supports one-wire Pauli observables.")
    label_map = {"PauliX": "X", "PauliY": "Y", "PauliZ": "Z", "Identity": "I"}
    if name not in label_map:
        raise ValueError(f"QuantumBridge PennyLane observable bridge does not support {name!r}.")
    return PauliString([(int(wires[0]), label_map[name])])


def circuit_to_pennylane_callable(circuit, observable: Optional[PauliString] = None):
    circuit = circuit.lowered()
    try:
        import pennylane as qml
    except Exception as exc:
        raise ImportError("QuantumBridge PennyLane executable bridge requires the optional pennylane package.") from exc

    dev = qml.device("default.qubit", wires=circuit.num_qubits)

    @qml.qnode(dev)
    def executable():
        for op in circuit.operations:
            if op.name == "x":
                qml.PauliX(wires=op.targets[0])
            elif op.name == "y":
                qml.PauliY(wires=op.targets[0])
            elif op.name == "z":
                qml.PauliZ(wires=op.targets[0])
            elif op.name == "h":
                qml.Hadamard(wires=op.targets[0])
            elif op.name == "rx":
                qml.RX(float(op.params[0]), wires=op.targets[0])
            elif op.name == "ry":
                qml.RY(float(op.params[0]), wires=op.targets[0])
            elif op.name == "rz":
                qml.RZ(float(op.params[0]), wires=op.targets[0])
            elif op.name == "phase":
                qml.PhaseShift(float(op.params[0]), wires=op.targets[0])
            elif op.name == "cx":
                qml.CNOT(wires=[op.controls[0], op.targets[0]])
            elif op.name == "cz":
                qml.CZ(wires=[op.controls[0], op.targets[0]])
            elif op.name == "swap":
                qml.SWAP(wires=[op.targets[0], op.targets[1]])
            else:
                raise ValueError(f"QuantumBridge PennyLane executable bridge does not support operation {op.name!r}.")
        if observable is None:
            return qml.probs(wires=range(circuit.num_qubits))
        if len(observable.terms) == 1 and observable.terms[0][1] == "Z":
            return qml.expval(qml.PauliZ(observable.terms[0][0]))
        raise ValueError("QuantumBridge PennyLane executable bridge currently supports probabilities or one PauliZ expectation.")

    return executable


def circuit_and_observable_from_pennylane_tape(tape):
    max_wire = -1
    for op in tape.operations:
        max_wire = max(max_wire, *(int(wire) for wire in op.wires))
    for measurement in tape.measurements:
        obs = getattr(measurement, "obs", None)
        if obs is not None:
            max_wire = max(max_wire, *(int(wire) for wire in obs.wires))
    circuit = Circuit(max_wire + 1 if max_wire >= 0 else 1)
    for op in tape.operations:
        name = op.name
        wires = [int(wire) for wire in op.wires]
        params = [float(value) for value in getattr(op, "parameters", [])]
        if name == "Hadamard":
            circuit.h(wires[0])
        elif name == "PauliX":
            circuit.x(wires[0])
        elif name == "PauliY":
            circuit.y(wires[0])
        elif name == "PauliZ":
            circuit.z(wires[0])
        elif name == "RX":
            circuit.rx(params[0], wires[0])
        elif name == "RY":
            circuit.ry(params[0], wires[0])
        elif name == "RZ":
            circuit.rz(params[0], wires[0])
        elif name == "PhaseShift":
            circuit.phase(params[0], wires[0])
        elif name == "Rot":
            circuit.rz(params[0], wires[0])
            circuit.ry(params[1], wires[0])
            circuit.rz(params[2], wires[0])
        elif name == "CNOT":
            circuit.cx(wires[0], wires[1])
        elif name == "CZ":
            circuit.cz(wires[0], wires[1])
        elif name == "SWAP":
            circuit.swap(wires[0], wires[1])
        else:
            raise ValueError(f"QuantumBridge PennyLane tape bridge does not support operation {name!r}.")
    observable = None
    if tape.measurements:
        obs = getattr(tape.measurements[0], "obs", None)
        if obs is not None:
            observable = observable_from_pennylane(obs)
    return circuit, observable


def ir_from_pennylane_tape(tape):
    circuit, _ = circuit_and_observable_from_pennylane_tape(tape)
    return circuit.to_ir()


def ir_to_pennylane_callable(program, observable: Optional[PauliString] = None):
    from quantumbridge.compat.qiskit_adapter import circuit_from_ir

    return circuit_to_pennylane_callable(circuit_from_ir(program), observable)


class PennyLaneDeviceAdapter:
    def __init__(self, observable: Optional[PauliString] = None):
        self.observable = observable

    def run(self, circuit):
        executable = circuit_to_pennylane_callable(circuit, self.observable)
        return executable()
