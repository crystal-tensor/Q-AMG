# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit, PennyLane, or PennyLane-Qiskit was copied.
"""Clean-room native PennyLane-Qiskit bridge helpers."""

from __future__ import annotations

from typing import Any, Iterable

from quantumbridge.compat.qiskit_adapter import (
    circuit_from_qiskit,
    circuit_to_qiskit,
)
from quantumbridge.compat.qiskit_aer.simulator_native import (
    run_qasm_simulator_native,
    run_statevector_simulator_native,
)
from quantumbridge.core import Circuit
from quantumbridge.ir.qb_ir import IRProgram
from quantumbridge.schema.pennylane_qiskit_bridge_results import (
    BridgeEquivalenceResult,
    BridgeIRResult,
    PennyLaneToQiskitResult,
    QiskitToPennyLaneResult,
)

from .warnings import (
    PENNYLANE_TO_QISKIT_WARNING,
    QISKIT_TO_PENNYLANE_WARNING,
    bridge_warnings,
    native_provenance,
)

SUPPORTED_QISKIT_TO_PENNYLANE = {
    "h": "Hadamard",
    "x": "PauliX",
    "y": "PauliY",
    "z": "PauliZ",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "phase": "PhaseShift",
    "cx": "CNOT",
    "cz": "CZ",
    "swap": "SWAP",
}
SUPPORTED_PENNYLANE_TO_QISKIT = {
    "Hadamard": "h",
    "PauliX": "x",
    "PauliY": "y",
    "PauliZ": "z",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "PhaseShift": "phase",
    "CNOT": "cx",
    "CZ": "cz",
    "SWAP": "swap",
}


def qiskit_circuit_to_bridge_ir(qiskit_circuit: Any) -> IRProgram:
    circuit = circuit_from_qiskit(qiskit_circuit)
    circuit.metadata.update(
        {
            "source_ecosystem": "qiskit",
            "target_ecosystem": "pennylane",
            "bridge": "pennylane_qiskit",
            "stage": "9H",
        }
    )
    return circuit.to_ir()


def qiskit_circuit_to_pennylane_spec(qiskit_circuit: Any) -> dict[str, Any]:
    circuit = circuit_from_qiskit(qiskit_circuit)
    return quantumbridge_circuit_to_pennylane_spec(circuit)


def quantumbridge_circuit_to_pennylane_spec(circuit: Circuit) -> dict[str, Any]:
    operations: list[dict[str, Any]] = []
    warnings: list[str] = []
    for op in circuit.operations:
        operation = SUPPORTED_QISKIT_TO_PENNYLANE.get(op.name)
        if operation is None:
            warnings.append(f"Unsupported QuantumBridge operation for PennyLane spec: {op.name}")
            continue
        wires = list(op.controls + op.targets) if op.controls else list(op.targets)
        operations.append(
            {
                "operation": operation,
                "wires": wires,
                "parameters": [_jsonable_parameter(value) for value in op.params],
                "metadata_only": False,
            }
        )
    measurements = [
        {"measurement": "probs", "wires": list(range(circuit.num_qubits))}
    ]
    return {
        "schema_version": "0.1",
        "ecosystem": "pennylane",
        "source": "quantumbridge_ir",
        "workflow": "qiskit_to_pennylane_spec",
        "num_qubits": circuit.num_qubits,
        "operations": operations,
        "measurements": measurements,
        "warnings": bridge_warnings(QISKIT_TO_PENNYLANE_WARNING, *warnings),
        "provenance": native_provenance("qiskit_to_pennylane_spec"),
        "production_ready": False,
        "full_plugin_parity_claim": False,
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
    }


def pennylane_operations_to_bridge_ir(operations: Iterable[Any], num_qubits: int | None = None) -> IRProgram:
    circuit = _circuit_from_pennylane_operations(operations, num_qubits=num_qubits)
    circuit.metadata.update(
        {
            "source_ecosystem": "pennylane",
            "target_ecosystem": "qiskit",
            "bridge": "pennylane_qiskit",
            "stage": "9H",
        }
    )
    return circuit.to_ir()


def pennylane_tape_to_bridge_ir(tape: Any) -> IRProgram:
    operations = getattr(tape, "operations", None)
    if operations is None:
        operations = tape.get("operations", ()) if isinstance(tape, dict) else ()
    return pennylane_operations_to_bridge_ir(operations)


def pennylane_qnode_metadata_to_bridge_ir(metadata: dict[str, Any]) -> IRProgram:
    return pennylane_operations_to_bridge_ir(
        metadata.get("operations", ()),
        num_qubits=metadata.get("num_qubits"),
    )


def bridge_ir_to_qiskit_circuit(ir_or_dict: Any):
    circuit = _circuit_from_ir_like(ir_or_dict)
    return circuit_to_qiskit(circuit)


def pennylane_operations_to_qiskit_circuit(operations: Iterable[Any], num_qubits: int | None = None):
    return bridge_ir_to_qiskit_circuit(pennylane_operations_to_bridge_ir(operations, num_qubits))


def run_qiskit_to_pennylane_bridge(qiskit_circuit: Any, shots: int = 128, seed: int | None = 7) -> QiskitToPennyLaneResult:
    circuit = circuit_from_qiskit(qiskit_circuit)
    ir = qiskit_circuit_to_bridge_ir(qiskit_circuit)
    spec = quantumbridge_circuit_to_pennylane_spec(circuit)
    state = run_statevector_simulator_native(circuit)
    counts = run_qasm_simulator_native(circuit, shots=shots, seed=seed)
    return QiskitToPennyLaneResult(
        workflow="qiskit_to_pennylane_bridge_native",
        mode="native_bridge",
        source_ecosystem="qiskit",
        target_ecosystem="pennylane",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        circuit_summary=_circuit_summary(circuit),
        operation_count=len(circuit.operations),
        num_qubits=circuit.num_qubits,
        shots=shots,
        seed=seed,
        statevector_probabilities=state.probabilities,
        counts=counts.counts,
        converted_ir=ir.to_dict(),
        converted_target=spec,
        raw_type="PennyLaneSpec",
        metadata=_metadata("qiskit_to_pennylane_bridge_native"),
        warnings=bridge_warnings(QISKIT_TO_PENNYLANE_WARNING),
        provenance=native_provenance("qiskit_to_pennylane_bridge_native"),
    )


def run_pennylane_to_qiskit_bridge(operations_or_tape: Any, shots: int = 128, seed: int | None = 7) -> PennyLaneToQiskitResult:
    if hasattr(operations_or_tape, "operations") or (
        isinstance(operations_or_tape, dict) and "operations" in operations_or_tape
    ):
        ir = pennylane_tape_to_bridge_ir(operations_or_tape)
    else:
        ir = pennylane_operations_to_bridge_ir(operations_or_tape)
    circuit = _circuit_from_ir_like(ir)
    qiskit_circuit = circuit_to_qiskit(circuit)
    state = run_statevector_simulator_native(circuit)
    counts = run_qasm_simulator_native(circuit, shots=shots, seed=seed)
    return PennyLaneToQiskitResult(
        workflow="pennylane_to_qiskit_bridge_native",
        mode="native_bridge",
        source_ecosystem="pennylane",
        target_ecosystem="qiskit",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        circuit_summary=_circuit_summary(circuit),
        operation_count=len(circuit.operations),
        num_qubits=circuit.num_qubits,
        shots=shots,
        seed=seed,
        statevector_probabilities=state.probabilities,
        counts=counts.counts,
        converted_ir=ir.to_dict(),
        converted_target=_qiskit_circuit_summary(qiskit_circuit),
        raw_type="QiskitQuantumCircuit",
        metadata=_metadata("pennylane_to_qiskit_bridge_native"),
        warnings=bridge_warnings(PENNYLANE_TO_QISKIT_WARNING),
        provenance=native_provenance("pennylane_to_qiskit_bridge_native"),
    )


def run_bidirectional_bridge_equivalence(shots: int = 128, seed: int | None = 7, tolerance: float = 1e-8) -> BridgeEquivalenceResult:
    try:
        from qiskit import QuantumCircuit
    except Exception as exc:
        raise ImportError("Stage 9H equivalence proof requires optional qiskit.") from exc

    qiskit_circuit = QuantumCircuit(2, 2)
    qiskit_circuit.h(0)
    qiskit_circuit.cx(0, 1)
    qiskit_circuit.measure(0, 0)
    qiskit_circuit.measure(1, 1)
    q_to_pl = run_qiskit_to_pennylane_bridge(qiskit_circuit, shots=shots, seed=seed)
    pl_ops = [
        {"operation": "Hadamard", "wires": [0]},
        {"operation": "CNOT", "wires": [0, 1]},
    ]
    pl_to_q = run_pennylane_to_qiskit_bridge(pl_ops, shots=shots, seed=seed)
    max_delta = _max_probability_delta(
        q_to_pl.statevector_probabilities,
        pl_to_q.statevector_probabilities,
    )
    status = "equivalent" if max_delta <= tolerance else "different"
    return BridgeEquivalenceResult(
        workflow="pennylane_qiskit_bidirectional_equivalence_native",
        mode="comparison",
        source_ecosystem="qiskit+pennylane",
        target_ecosystem="quantumbridge_ir",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        circuit_summary={"case": "bell_state", "max_probability_delta": max_delta},
        operation_count=2,
        num_qubits=2,
        shots=shots,
        seed=seed,
        statevector_probabilities=q_to_pl.statevector_probabilities,
        counts=q_to_pl.counts,
        converted_ir={"qiskit_to_pennylane": q_to_pl.converted_ir, "pennylane_to_qiskit": pl_to_q.converted_ir},
        converted_target={"qiskit_to_pennylane": q_to_pl.converted_target, "pennylane_to_qiskit": pl_to_q.converted_target},
        equivalence_status=status,
        tolerance=tolerance,
        raw_type="BridgeEquivalence",
        metadata={
            **_metadata("pennylane_qiskit_bidirectional_equivalence_native"),
            "pennylane_statevector_probabilities": pl_to_q.statevector_probabilities,
            "pennylane_counts": pl_to_q.counts,
            "statevector_tolerance": tolerance,
            "counts_comparison": "statistical_support",
        },
        warnings=bridge_warnings(QISKIT_TO_PENNYLANE_WARNING, PENNYLANE_TO_QISKIT_WARNING),
        provenance=native_provenance("pennylane_qiskit_bidirectional_equivalence_native"),
    )


def bridge_ir_summary(ir_or_dict: Any) -> BridgeIRResult:
    circuit = _circuit_from_ir_like(ir_or_dict)
    ir = circuit.to_ir()
    return BridgeIRResult(
        workflow="pennylane_qiskit_bridge_ir_summary",
        mode="native_bridge",
        source_ecosystem="quantumbridge_ir",
        target_ecosystem="pennylane_qiskit_bridge",
        capability_level=2,
        production_ready=False,
        native_implementation=True,
        circuit_summary=_circuit_summary(circuit),
        operation_count=len(circuit.operations),
        num_qubits=circuit.num_qubits,
        converted_ir=ir.to_dict(),
        raw_type="IRProgram",
        metadata=_metadata("pennylane_qiskit_bridge_ir_summary"),
        warnings=bridge_warnings(),
        provenance=native_provenance("pennylane_qiskit_bridge_ir_summary"),
    )


def _circuit_from_pennylane_operations(operations: Iterable[Any], num_qubits: int | None = None) -> Circuit:
    normalized = [_normalize_pennylane_operation(item) for item in operations]
    max_wire = max((wire for item in normalized for wire in item["wires"]), default=0)
    circuit = Circuit(int(num_qubits or (max_wire + 1)))
    for item in normalized:
        op = SUPPORTED_PENNYLANE_TO_QISKIT.get(item["operation"])
        if op is None:
            raise ValueError(f"Unsupported PennyLane operation for Qiskit bridge: {item['operation']}")
        wires = item["wires"]
        params = item["parameters"]
        if op in {"h", "x", "y", "z"}:
            getattr(circuit, op)(wires[0])
        elif op in {"rx", "ry", "rz"}:
            getattr(circuit, op)(float(params[0]), wires[0])
        elif op == "phase":
            circuit.phase(float(params[0]), wires[0])
        elif op in {"cx", "cz"}:
            getattr(circuit, op)(wires[0], wires[1])
        elif op == "swap":
            circuit.swap(wires[0], wires[1])
    return circuit


def _normalize_pennylane_operation(operation: Any) -> dict[str, Any]:
    if isinstance(operation, dict):
        name = operation.get("operation") or operation.get("name") or operation.get("op")
        wires = operation.get("wires") or operation.get("targets") or ()
        params = operation.get("parameters") or operation.get("params") or ()
        return {"operation": str(name), "wires": [int(wire) for wire in wires], "parameters": list(params)}
    name = getattr(operation, "name", None) or type(operation).__name__
    wires = [int(wire) for wire in getattr(operation, "wires", ())]
    params = list(getattr(operation, "parameters", ()))
    return {"operation": str(name), "wires": wires, "parameters": params}


def _circuit_from_ir_like(ir_or_dict: Any) -> Circuit:
    if isinstance(ir_or_dict, Circuit):
        return ir_or_dict.copy()
    if isinstance(ir_or_dict, IRProgram):
        from quantumbridge.compat.qiskit_adapter import circuit_from_ir

        return circuit_from_ir(ir_or_dict)
    if hasattr(ir_or_dict, "instructions") and hasattr(ir_or_dict, "num_qubits"):
        from quantumbridge.compat.qiskit_adapter import circuit_from_ir

        return circuit_from_ir(ir_or_dict)
    if isinstance(ir_or_dict, dict):
        registers = dict(ir_or_dict.get("registers", {}))
        num_qubits = int(registers.get("quantum", {}).get("size", ir_or_dict.get("num_qubits", 1)))
        num_bits = int(registers.get("classical", {}).get("size", ir_or_dict.get("num_bits", 0)))
        circuit = Circuit(num_qubits, num_bits)
        for instruction in ir_or_dict.get("instructions") or ir_or_dict.get("operations", ()):
            op = instruction.get("op") or instruction.get("name")
            targets = [int(value) for value in instruction.get("targets", ())]
            controls = [int(value) for value in instruction.get("controls", ())]
            params = list(instruction.get("params", ()))
            _append_quantumbridge_op(circuit, op, targets, controls, params)
        return circuit
    raise TypeError("expected QuantumBridge Circuit, IRProgram, or IR dictionary")


def _append_quantumbridge_op(circuit: Circuit, op: str, targets: list[int], controls: list[int], params: list[Any]) -> None:
    if op in {"h", "x", "y", "z"}:
        getattr(circuit, op)(targets[0])
    elif op in {"rx", "ry", "rz"}:
        getattr(circuit, op)(float(params[0]), targets[0])
    elif op == "phase":
        circuit.phase(float(params[0]), targets[0])
    elif op in {"cx", "cz"}:
        getattr(circuit, op)(controls[0], targets[0])
    elif op == "swap":
        circuit.swap(targets[0], targets[1])
    else:
        raise ValueError(f"Unsupported QuantumBridge operation for Stage 9H bridge: {op}")


def _qiskit_circuit_summary(circuit: Any) -> dict[str, Any]:
    return {
        "type": type(circuit).__name__,
        "num_qubits": int(getattr(circuit, "num_qubits", 0)),
        "num_clbits": int(getattr(circuit, "num_clbits", 0)),
        "operations": [item.operation.name for item in getattr(circuit, "data", ())],
    }


def _circuit_summary(circuit: Circuit) -> dict[str, Any]:
    return {
        "type": "QuantumBridgeCircuit",
        "num_qubits": circuit.num_qubits,
        "num_bits": circuit.num_bits,
        "operations": [op.name for op in circuit.operations],
        "measurements": [item.to_dict() if hasattr(item, "to_dict") else {"wire": item.wire, "bit": item.bit} for item in circuit.measurements],
    }


def _metadata(workflow: str) -> dict[str, Any]:
    return {
        "workflow": workflow,
        "supported_gate_subset": sorted(SUPPORTED_QISKIT_TO_PENNYLANE),
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
        "production_ready": False,
        "full_qiskit_parity_claim": False,
        "full_pennylane_parity_claim": False,
        "full_plugin_parity_claim": False,
    }


def _max_probability_delta(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return max((abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys), default=0.0)


def _jsonable_parameter(parameter: Any) -> float | str:
    try:
        return float(parameter)
    except (TypeError, ValueError):
        return repr(parameter)
