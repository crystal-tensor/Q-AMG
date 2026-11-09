# This file is independently implemented for QuantumBridge SDK.
# No source code from MQT, IBM, or Qiskit was copied.
"""Clean-room MQT Core-like circuit and IR compatibility helpers."""

from __future__ import annotations

import re
from typing import Any

from quantumbridge.compat.qiskit_aer.simulator_native import normalize_circuit_to_quantumbridge_ir
from quantumbridge.core import Circuit
from quantumbridge.ir.qb_ir import IRProgram
from quantumbridge.schema.mqt_results import MQTCoreLikeCircuitResult

from .warnings import mqt_warnings, native_provenance

SUPPORTED_CORE_OPS = {
    "h",
    "x",
    "y",
    "z",
    "rx",
    "ry",
    "rz",
    "phase",
    "p",
    "cx",
    "cnot",
    "cz",
    "swap",
    "measure",
}


def build_mqt_core_like_circuit_from_quantumbridge_ir(ir: Any) -> MQTCoreLikeCircuitResult:
    data = quantumbridge_ir_to_mqt_core_like_dict(ir)
    return MQTCoreLikeCircuitResult(
        workflow="mqt_core_like_circuit_from_quantumbridge_ir",
        project="mqt-core",
        mode="native_minimal",
        capability_level=3,
        production_ready=False,
        native_implementation=True,
        circuit_summary=_circuit_summary(data),
        num_qubits=data["num_qubits"],
        operations=list(data["operations"]),
        measurements=list(data["measurements"]),
        raw_type="MQTCoreLikeDict",
        metadata={
            "mqt_core_parity_claim": False,
            "cloud_access": False,
            "token_read": False,
            "hardware_access": False,
        },
        warnings=mqt_warnings(),
        provenance=native_provenance("mqt_core_like_circuit_from_quantumbridge_ir", "mqt-core"),
    )


def quantumbridge_ir_to_mqt_core_like_dict(ir: Any) -> dict[str, Any]:
    circuit = normalize_circuit_to_quantumbridge_ir(ir)
    operations = [_operation_to_core_like(op) for op in circuit.operations]
    measurements = [
        {"kind": item.kind, "wire": int(item.wire), "bit": int(item.bit)}
        for item in circuit.measurements
    ]
    payload = {
        "schema_version": "mqt-core-like-v0.1",
        "source": "quantumbridge_clean_room",
        "name": circuit.name,
        "num_qubits": circuit.num_qubits,
        "num_bits": circuit.num_bits,
        "registers": {
            "quantum": {"name": "q", "size": circuit.num_qubits},
            "classical": {"name": "c", "size": circuit.num_bits},
        },
        "operations": operations,
        "measurements": measurements,
        "metadata": {
            **dict(circuit.metadata),
            "copied_upstream_source": False,
            "mqt_core_parity_claim": False,
        },
    }
    validate_mqt_core_like_circuit(payload)
    return payload


def mqt_core_like_dict_to_quantumbridge_ir(data: dict[str, Any]) -> IRProgram:
    payload = validate_mqt_core_like_circuit(data)
    circuit = Circuit(
        int(payload["num_qubits"]),
        int(payload.get("num_bits", 0)),
        payload.get("name"),
        dict(payload.get("metadata", {})),
    )
    for operation in payload["operations"]:
        _append_operation(circuit, operation)
    for measurement in payload.get("measurements", ()):
        circuit.measure(int(measurement["wire"]), int(measurement["bit"]))
    return circuit.to_ir()


def validate_mqt_core_like_circuit(data: dict[str, Any]) -> dict[str, Any]:
    payload = dict(data)
    num_qubits = int(payload.get("num_qubits", 0))
    num_bits = int(payload.get("num_bits", 0))
    if num_qubits <= 0:
        raise ValueError("MQT Core-like circuit requires a positive num_qubits")
    if num_bits < 0:
        raise ValueError("num_bits cannot be negative")
    operations = [dict(item) for item in payload.get("operations", ())]
    measurements = [dict(item) for item in payload.get("measurements", ())]
    for operation in operations:
        name = _normalize_op_name(operation.get("name", operation.get("op")))
        if name not in SUPPORTED_CORE_OPS - {"measure"}:
            raise ValueError(f"unsupported MQT Core-like operation {name!r}")
        targets = [int(value) for value in operation.get("targets", ())]
        controls = [int(value) for value in operation.get("controls", ())]
        for wire in controls + targets:
            if wire < 0 or wire >= num_qubits:
                raise ValueError(f"wire {wire!r} is outside the circuit")
        if len(set(controls + targets)) != len(controls + targets):
            raise ValueError("operation wires must be distinct")
        operation["name"] = name
        operation["targets"] = targets
        operation["controls"] = controls
        operation["params"] = [float(value) for value in operation.get("params", ())]
    for measurement in measurements:
        wire = int(measurement["wire"])
        bit = int(measurement["bit"])
        if not 0 <= wire < num_qubits:
            raise ValueError("measurement wire is outside the circuit")
        if not 0 <= bit < num_bits:
            raise ValueError("measurement bit is outside the circuit")
        measurement["kind"] = str(measurement.get("kind", "measure"))
        measurement["wire"] = wire
        measurement["bit"] = bit
    payload["num_qubits"] = num_qubits
    payload["num_bits"] = num_bits
    payload["operations"] = operations
    payload["measurements"] = measurements
    payload.setdefault("schema_version", "mqt-core-like-v0.1")
    payload.setdefault("metadata", {})
    return payload


def export_mqt_core_like_qasm_subset(data: dict[str, Any]) -> str:
    payload = validate_mqt_core_like_circuit(data)
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{payload['num_qubits']}];",
    ]
    if payload["num_bits"]:
        lines.append(f"creg c[{payload['num_bits']}];")
    for operation in payload["operations"]:
        name = operation["name"]
        targets = operation.get("targets", ())
        controls = operation.get("controls", ())
        params = operation.get("params", ())
        if name in {"rx", "ry", "rz", "phase", "p"}:
            qasm_name = "p" if name == "phase" else name
            lines.append(f"{qasm_name}({float(params[0]):.17g}) q[{targets[0]}];")
        elif name in {"h", "x", "y", "z"}:
            lines.append(f"{name} q[{targets[0]}];")
        elif name in {"cx", "cnot", "cz"}:
            qasm_name = "cx" if name == "cnot" else name
            lines.append(f"{qasm_name} q[{controls[0]}],q[{targets[0]}];")
        elif name == "swap":
            lines.append(f"swap q[{targets[0]}],q[{targets[1]}];")
        else:
            raise ValueError(f"cannot export unsupported operation {name!r}")
    for measurement in payload["measurements"]:
        lines.append(f"measure q[{measurement['wire']}] -> c[{measurement['bit']}];")
    return "\n".join(lines) + "\n"


def import_mqt_core_like_qasm_subset(qasm_text: str) -> dict[str, Any]:
    num_qubits = None
    num_bits = 0
    operations: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    for raw in qasm_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("//") or line.startswith("OPENQASM") or line.startswith("include"):
            continue
        if match := re.fullmatch(r"qreg q\[(\d+)\];", line):
            num_qubits = int(match.group(1))
            continue
        if match := re.fullmatch(r"creg c\[(\d+)\];", line):
            num_bits = int(match.group(1))
            continue
        if match := re.fullmatch(r"(rx|ry|rz|p|phase)\(([^)]+)\) q\[(\d+)\];", line):
            name = "phase" if match.group(1) == "p" else match.group(1)
            operations.append({"name": name, "targets": [int(match.group(3))], "controls": [], "params": [float(match.group(2))]})
            continue
        if match := re.fullmatch(r"(h|x|y|z) q\[(\d+)\];", line):
            operations.append({"name": match.group(1), "targets": [int(match.group(2))], "controls": [], "params": []})
            continue
        if match := re.fullmatch(r"(cx|cnot|cz) q\[(\d+)\],q\[(\d+)\];", line):
            name = "cx" if match.group(1) == "cnot" else match.group(1)
            operations.append({"name": name, "targets": [int(match.group(3))], "controls": [int(match.group(2))], "params": []})
            continue
        if match := re.fullmatch(r"swap q\[(\d+)\],q\[(\d+)\];", line):
            operations.append({"name": "swap", "targets": [int(match.group(1)), int(match.group(2))], "controls": [], "params": []})
            continue
        if match := re.fullmatch(r"measure q\[(\d+)\] -> c\[(\d+)\];", line):
            measurements.append({"kind": "measure", "wire": int(match.group(1)), "bit": int(match.group(2))})
            continue
        raise ValueError(f"unsupported MQT Core-like QASM subset line: {line!r}")
    if num_qubits is None:
        raise ValueError("QASM subset must define qreg q[N]")
    return validate_mqt_core_like_circuit(
        {
            "schema_version": "mqt-core-like-v0.1",
            "source": "qasm_subset",
            "num_qubits": num_qubits,
            "num_bits": num_bits,
            "operations": operations,
            "measurements": measurements,
            "metadata": {"qasm_subset": True, "mqt_core_parity_claim": False},
        }
    )


def _operation_to_core_like(op: Any) -> dict[str, Any]:
    name = _normalize_op_name(op.name)
    if name not in SUPPORTED_CORE_OPS - {"measure"}:
        raise ValueError(f"unsupported MQT Core-like operation {name!r}")
    return {
        "name": "cx" if name == "cnot" else name,
        "targets": [int(value) for value in op.targets],
        "controls": [int(value) for value in op.controls],
        "params": [float(value) for value in op.params],
        "metadata": {key: value for key, value in dict(op.metadata).items() if key != "matrix"},
    }


def _append_operation(circuit: Circuit, operation: dict[str, Any]) -> None:
    name = _normalize_op_name(operation["name"])
    targets = [int(value) for value in operation.get("targets", ())]
    controls = [int(value) for value in operation.get("controls", ())]
    params = [float(value) for value in operation.get("params", ())]
    if name in {"h", "x", "y", "z"}:
        getattr(circuit, name)(targets[0])
    elif name in {"rx", "ry", "rz"}:
        getattr(circuit, name)(params[0], targets[0])
    elif name in {"phase", "p"}:
        circuit.phase(params[0], targets[0])
    elif name in {"cx", "cnot"}:
        circuit.cx(controls[0], targets[0])
    elif name == "cz":
        circuit.cz(controls[0], targets[0])
    elif name == "swap":
        circuit.swap(targets[0], targets[1])
    else:
        raise ValueError(f"unsupported operation {name!r}")


def _normalize_op_name(name: Any) -> str:
    normalized = str(name).lower()
    if normalized == "cnot":
        return "cx"
    if normalized == "p":
        return "phase"
    return normalized


def _circuit_summary(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_qubits": int(data["num_qubits"]),
        "num_bits": int(data.get("num_bits", 0)),
        "operation_count": len(data.get("operations", ())),
        "measurement_count": len(data.get("measurements", ())),
        "supported_gates": sorted(SUPPORTED_CORE_OPS),
    }
