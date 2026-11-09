# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.
#
# Adapter Integration: this module optionally imports Qiskit at runtime.
# No Qiskit source code is copied into this file.

from __future__ import annotations

from quantumbridge.core import Circuit, Measurement
from quantumbridge.results import Result


SUPPORTED_IMPORT = {"id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "t", "tdg", "rx", "ry", "rz", "p", "phase", "cx", "cnot", "cy", "cz", "swap", "dcx", "ecr", "iswap", "measure", "reset", "barrier", "delay", "if_else", "kraus", "unitary"}


def _require_qiskit():
    try:
        from qiskit import QuantumCircuit
    except Exception as exc:
        raise ImportError("QuantumBridge Qiskit adapter requires the optional qiskit package.") from exc
    return QuantumCircuit


def circuit_from_qiskit(qiskit_circuit) -> Circuit:
    qb = Circuit(qiskit_circuit.num_qubits, qiskit_circuit.num_clbits)
    qb.global_phase = float(qiskit_circuit.global_phase)
    for item in qiskit_circuit.data:
        name = item.operation.name
        if name not in SUPPORTED_IMPORT:
            raise ValueError(f"QuantumBridge Qiskit adapter does not support operation {name!r}.")
        qubits = [qiskit_circuit.find_bit(qubit).index for qubit in item.qubits]
        clbits = [qiskit_circuit.find_bit(clbit).index for clbit in item.clbits]
        params = [] if name in {"if_else", "kraus", "unitary"} else [float(param) for param in getattr(item.operation, "params", [])]
        if name in {"id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "t", "tdg", "reset"}:
            getattr(qb, name)(qubits[0])
        elif name in {"rx", "ry", "rz"}:
            getattr(qb, name)(params[0], qubits[0])
        elif name in {"p", "phase"}:
            qb.phase(params[0], qubits[0])
        elif name in {"cx", "cnot", "cy", "cz"}:
            gate = "cx" if name in {"cx", "cnot"} else "cz"
            getattr(qb, gate)(qubits[0], qubits[1])
        elif name in {"swap", "dcx", "ecr", "iswap"}:
            getattr(qb, name)(qubits[0], qubits[1])
        elif name == "measure":
            qb.measure(qubits[0], clbits[0])
        elif name == "barrier":
            qb.barrier(*qubits)
        elif name == "delay":
            qb.delay(params[0], qubits[0], unit=item.operation.unit)
        elif name == "kraus":
            qb.kraus(item.operation.params, tuple(reversed(qubits)))
        elif name == "unitary":
            qb.unitary(item.operation.to_matrix(), tuple(reversed(qubits)))
        elif name == "if_else":
            classical, value = item.operation.condition
            condition_bits = [qiskit_circuit.find_bit(bit).index for bit in classical] if hasattr(classical, "__iter__") else [qiskit_circuit.find_bit(classical).index]
            blocks = item.operation.blocks
            true_body = circuit_from_qiskit(blocks[0])
            false_body = circuit_from_qiskit(blocks[1]) if len(blocks) > 1 else None
            qb.if_else((condition_bits, int(value)), true_body, false_body, qubits=qubits, clbits=clbits, inplace=True)
    return qb


def ir_from_qiskit(qiskit_circuit):
    return circuit_from_qiskit(qiskit_circuit).to_ir()


def circuit_to_qiskit(circuit: Circuit):
    circuit = circuit.lowered()
    QuantumCircuit = _require_qiskit()
    qc = QuantumCircuit(circuit.num_qubits, circuit.num_bits)
    qc.global_phase = float(circuit.global_phase)
    for op in circuit.iter_instructions():
        if any(key in op.metadata for key in ("condition", "control_flow", "control_flow_stack")):
            raise ValueError("QuantumBridge Qiskit export does not yet reconstruct conditional blocks.")
        if isinstance(op, Measurement):
            qc.measure(op.wire, op.bit)
        elif op.name in {"id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "t", "tdg", "reset"}:
            getattr(qc, op.name)(op.targets[0])
        elif op.name in {"rx", "ry", "rz"}:
            getattr(qc, op.name)(float(op.params[0]), op.targets[0])
        elif op.name == "phase":
            qc.p(float(op.params[0]), op.targets[0])
        elif op.name in {"cx", "cy", "cz"}:
            getattr(qc, op.name)(op.controls[0], op.targets[0])
        elif op.name in {"swap", "dcx", "ecr", "iswap"}:
            getattr(qc, op.name)(op.targets[0], op.targets[1])
        elif op.name == "barrier":
            qc.barrier(*op.targets)
        elif op.name == "delay":
            qc.delay(float(op.params[0]), op.targets[0], unit=op.metadata.get("unit", "dt"))
        elif op.name == "kraus":
            from qiskit.quantum_info import Kraus

            qc.append(Kraus(list(op.metadata["kraus"])).to_instruction(), list(reversed(op.wires)))
        elif "matrix" in op.metadata:
            qc.unitary(op.metadata["matrix"], list(reversed(op.wires)))
        else:
            raise ValueError(f"QuantumBridge Qiskit export does not support operation {op.name!r}.")
    return qc


def circuit_from_ir(program) -> Circuit:
    from quantumbridge.ir.qb_ir import decode_instruction_metadata

    circuit = Circuit(program.num_qubits, program.num_bits, program.name, program.metadata)
    if program.execution is not None:
        circuit._execution_metadata=program.execution.configuration()
    from quantumbridge.core.bits import restore_bit_graph
    restore_bit_graph(circuit,program.bit_graph)
    if program.execution is not None:circuit.global_phase=program.execution.global_phase
    elif "global_phase" in program.metadata: circuit.global_phase = program.metadata["global_phase"]
    for instruction in program.instructions:
        metadata = decode_instruction_metadata(instruction.metadata)
        gate = metadata.pop("gate", None)
        annotation = metadata.pop("annotated_operation", None)
        owner = metadata.pop("instruction_object", None)
        from quantumbridge.core.standard_gates import invocation_params
        if owner is not None:
            from quantumbridge.core.operations import Instruction
            bits = metadata.pop("instruction_clbits", ())
            if (type(owner) is not Instruction or gate is not None or annotation is not None or instruction.controls
                    or instruction.op != owner.name or instruction.params != tuple(owner.params)):
                raise ValueError("IR Instruction reference disagrees with its owner.")
            circuit._append_instruction_reference(owner, instruction.targets, bits, metadata)
            continue
        if annotation is not None:
            from quantumbridge.core.annotated_operation import AnnotatedOperation
            if (type(annotation) is not AnnotatedOperation or gate is not None or instruction.op != annotation.name
                    or annotation.num_qubits != len(instruction.targets) or instruction.controls or invocation_params(annotation) != instruction.params):
                raise ValueError("IR annotated invocation disagrees with its operation record.")
            bits=metadata.pop("instruction_clbits",())
            circuit.append(annotation, instruction.targets, metadata=metadata,clbits=bits)
            continue
        if gate is not None:
            if gate.name != instruction.op or gate.num_qubits != len(instruction.targets)+len(instruction.controls) or invocation_params(gate) != instruction.params or instruction.controls:
                raise ValueError("IR Gate invocation width/parameters disagree with its definition.")
            circuit.append(gate, instruction.targets, metadata=metadata)
        else:
            circuit.append(instruction.op, instruction.targets, controls=instruction.controls,
                           params=instruction.params, metadata=metadata)
    for measurement in program.measurements:
        circuit.measurements.append(Measurement(measurement.wire, measurement.bit, measurement.kind, dict(getattr(measurement, "metadata", {}))))
    from quantumbridge.core.parameter_graph import canonical_parameter_graph
    from quantumbridge.core.parameters import parameter_set
    symbols={}
    for item in program.instructions:
        for symbol in parameter_set(item.params):symbols.setdefault(symbol.name,symbol)
    for symbol in parameter_set((circuit.global_phase,)):symbols.setdefault(symbol.name,symbol)
    return canonical_parameter_graph(circuit,symbols)


def ir_to_qiskit(program):
    return circuit_to_qiskit(circuit_from_ir(program))


def result_from_qiskit_counts(qiskit_result_or_counts) -> Result:
    if isinstance(qiskit_result_or_counts, dict):
        counts = qiskit_result_or_counts
    elif hasattr(qiskit_result_or_counts, "get_counts"):
        counts = qiskit_result_or_counts.get_counts()
    else:
        raise TypeError("QuantumBridge Qiskit result adapter expects a counts mapping or an object with get_counts().")
    counts = {str(key): int(value) for key, value in counts.items()}
    total = sum(counts.values())
    probabilities = {key: value / total for key, value in counts.items()} if total else {}
    return Result(
        counts_data=counts,
        probabilities_data=probabilities,
        metadata_data={"adapter": "qiskit_result_counts", "shots": total},
    )
