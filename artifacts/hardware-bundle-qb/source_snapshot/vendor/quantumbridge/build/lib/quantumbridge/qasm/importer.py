# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_execution_plan_v0.1.md.

from __future__ import annotations

from quantumbridge.core import Circuit

from .ast import Barrier, GateCall, IndexedRef, Measurement, QASMProgram
from .errors import QASMParseError
from .parser import parse


def _register_maps(program: QASMProgram):
    qregs: dict[str, tuple[int, int]] = {}
    cregs: dict[str, tuple[int, int]] = {}
    for decl in program.registers:
        target = qregs if decl.kind == "qreg" else cregs
        if decl.name in target:
            raise QASMParseError(f"duplicate register {decl.name!r}")
        offset = sum(size for _, size in target.values())
        target[decl.name] = (offset, decl.size)
    return qregs, cregs


def _resolve(indexed: IndexedRef, registers: dict[str, tuple[int, int]], kind: str) -> int:
    if indexed.register not in registers:
        raise QASMParseError(f"unknown {kind} register {indexed.register!r}")
    offset, size = registers[indexed.register]
    if not 0 <= indexed.index < size:
        raise QASMParseError(f"{kind} reference {indexed.register}[{indexed.index}] is out of range")
    return offset + indexed.index


def circuit_from_ast(program: QASMProgram) -> Circuit:
    qregs, cregs = _register_maps(program)
    if not qregs:
        raise QASMParseError("qreg declaration is required")
    circuit = Circuit(sum(size for _, size in qregs.values()), sum(size for _, size in cregs.values()))
    for statement in program.statements:
        if isinstance(statement, GateCall):
            wires = [_resolve(ref, qregs, "quantum") for ref in statement.qubits]
            if statement.name in {"id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "reset"}:
                getattr(circuit, statement.name)(wires[0])
            elif statement.name in {"rx", "ry", "rz", "phase"}:
                getattr(circuit, statement.name)(statement.params[0], wires[0])
            elif statement.name in {"cx", "cz"}:
                getattr(circuit, statement.name)(wires[0], wires[1])
            elif statement.name == "swap":
                circuit.swap(wires[0], wires[1])
            elif statement.name == "ccx":
                circuit.ccx(wires[0], wires[1], wires[2])
            else:
                raise QASMParseError(f"unsupported gate {statement.name!r}")
        elif isinstance(statement, Measurement):
            if isinstance(statement.qubit, str) and isinstance(statement.bit, str):
                if statement.qubit not in qregs or statement.bit not in cregs:
                    raise QASMParseError("whole-register measurement references unknown register")
                qoffset, qsize = qregs[statement.qubit]
                coffset, csize = cregs[statement.bit]
                if qsize != csize:
                    raise QASMParseError("whole-register measurement requires matching register sizes")
                for idx in range(qsize):
                    circuit.measure(qoffset + idx, coffset + idx)
            elif isinstance(statement.qubit, IndexedRef) and isinstance(statement.bit, IndexedRef):
                circuit.measure(_resolve(statement.qubit, qregs, "quantum"), _resolve(statement.bit, cregs, "classical"))
            else:
                raise QASMParseError("measurement must use either two indexed refs or two whole registers")
        elif isinstance(statement, Barrier):
            circuit.barrier(*[_resolve(ref,qregs,"quantum") for ref in statement.qubits])
    return circuit


def loads(text: str) -> Circuit:
    from quantumbridge.serialization import qasm2_loads
    from quantumbridge.serialization.errors import SerializationError
    try: return qasm2_loads(text,strict=False)
    except SerializationError as exc:
        raise QASMParseError(str(exc),exc.line,exc.column) from exc
