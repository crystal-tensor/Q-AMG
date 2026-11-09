# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.
# Design source: docs/mvp/api_contract_v0.1.md and docs/mvp/ir_design_v0.1.md.

from __future__ import annotations

from numbers import Real

from .qb_ir import IRProgram


ONE_QUBIT = {"x", "y", "z", "h", "s", "sdg", "t", "tdg"}
ROTATION = {"rx", "ry", "rz", "phase"}
TWO_QUBIT = {"cx", "cz", "swap", "iswap"}


def _format_param(value) -> str:
    if not isinstance(value, Real):
        raise ValueError("QuantumBridge OpenQASM export requires numeric parameters in MVP.")
    return format(float(value), ".17g")


def export_openqasm(program: IRProgram) -> str:
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{program.num_qubits}];",
    ]
    if program.num_bits:
        lines.append(f"creg c[{program.num_bits}];")
    for instruction in program.instructions:
        op = instruction.op
        if op in ROTATION:
            theta = _format_param(instruction.params[0])
            lines.append(f"{op}({theta}) q[{instruction.targets[0]}];")
        elif op in ONE_QUBIT:
            lines.append(f"{op} q[{instruction.targets[0]}];")
        elif op in TWO_QUBIT:
            if instruction.controls:
                first = instruction.controls[0]
                second = instruction.targets[0]
            else:
                first, second = instruction.targets
            lines.append(f"{op} q[{first}],q[{second}];")
        else:
            raise ValueError(f"QuantumBridge OpenQASM export does not support operation {op!r} in MVP.")
    for measurement in program.measurements:
        lines.append(f"measure q[{measurement.wire}] -> c[{measurement.bit}];")
    return "\n".join(lines) + "\n"
