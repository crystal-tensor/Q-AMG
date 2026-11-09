# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""PennyLane operations inventory and passthrough scaffold.

Design source: docs/compat/strategy/pennylane_full_coverage_strategy.md.
"""

from quantumbridge.compat.contracts import CapabilityLevel
from quantumbridge.compat.pennylane_full.passthrough import get_public_object
from quantumbridge.compat.pennylane_full.warnings import adapter_metadata, unsupported
from quantumbridge.ecosystem.registry import EcosystemAdapter

ADAPTER = EcosystemAdapter("pennylane", "pennylane", ("pennylane.ops",), "pennylane-full", "pennylane_operations")

BASIC_OPERATION_NAMES = (
    "H",
    "X",
    "Y",
    "Z",
    "RX",
    "RY",
    "RZ",
    "PhaseShift",
    "CNOT",
    "CZ",
    "SWAP",
    "Hadamard",
    "PauliX",
    "PauliY",
    "PauliZ",
)

_ALIASES = {
    "H": "Hadamard",
    "X": "PauliX",
    "Y": "PauliY",
    "Z": "PauliZ",
    "Phase": "PhaseShift",
    "CX": "CNOT",
}

_IR_GATE_MAP = {
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

_IR_TO_PENNYLANE = {
    "h": "Hadamard",
    "x": "PauliX",
    "y": "PauliY",
    "z": "PauliZ",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "phase": "PhaseShift",
    "p": "PhaseShift",
    "cx": "CNOT",
    "cnot": "CNOT",
    "cz": "CZ",
    "swap": "SWAP",
}

dependency_available = ADAPTER.dependency_available
get_upstream_version = ADAPTER.get_upstream_version
list_public_api_inventory = ADAPTER.list_public_api_inventory
passthrough_class = ADAPTER.passthrough_class
passthrough_function = ADAPTER.passthrough_function
wrap_result = ADAPTER.wrap_result
to_quantumbridge_schema = ADAPTER.to_quantumbridge_schema
provenance_metadata = ADAPTER.provenance_metadata
warn_unsupported = ADAPTER.warn_unsupported


def list_operations() -> list[str]:
    return list(BASIC_OPERATION_NAMES)


def get_operation(name: str):
    return get_public_object("pennylane", _ALIASES.get(name, name))


def describe_operation(name: str) -> dict:
    normalized = _ALIASES.get(name, name)
    supported = normalized in _IR_GATE_MAP
    return {
        "name": name,
        "upstream_name": normalized,
        "module": "pennylane",
        "supported": supported,
        "metadata_only": True,
        "unsupported_reason": None if supported else f"{name!r} is not in the Stage 8C basic gate subset.",
        **adapter_metadata(CapabilityLevel.SCHEMA_ADAPTER if supported else CapabilityLevel.INVENTORY),
    }


def _wire_to_python(wire):
    try:
        return int(wire)
    except (TypeError, ValueError):
        return str(wire)


def _wires_to_list(wires) -> list:
    if wires is None:
        return []
    return [_wire_to_python(wire) for wire in wires]


def _jsonable_parameters(parameters) -> list:
    values = []
    for parameter in parameters or []:
        try:
            values.append(float(parameter))
        except (TypeError, ValueError):
            values.append(repr(parameter))
    return values


def operation_to_metadata(obj) -> dict:
    name = getattr(obj, "name", type(obj).__name__)
    parameters = list(getattr(obj, "parameters", []))
    normalized = _ALIASES.get(name, name)
    supported = normalized in _IR_GATE_MAP
    wires = _wires_to_list(getattr(obj, "wires", []))
    return {
        "name": name,
        "upstream_name": normalized,
        "module": getattr(type(obj), "__module__", "pennylane"),
        "wires": wires,
        "parameters": _jsonable_parameters(parameters),
        "num_wires": len(wires),
        "num_params": len(parameters),
        "trainable": bool(getattr(obj, "grad_method", None)),
        "has_matrix": hasattr(obj, "matrix"),
        "supported": supported,
        "metadata_only": True,
        "unsupported_reason": None if supported else f"PennyLane operation {name!r} is not in the Stage 8C basic gate subset.",
        **adapter_metadata(CapabilityLevel.SCHEMA_ADAPTER if supported else CapabilityLevel.INVENTORY),
    }


def operation_to_quantumbridge_ir_fragment(obj):
    name = getattr(obj, "name", type(obj).__name__)
    normalized = _ALIASES.get(name, name)
    wires = _wires_to_list(getattr(obj, "wires", []))
    params = _jsonable_parameters(getattr(obj, "parameters", []))
    if normalized not in _IR_GATE_MAP:
        return unsupported(f"PennyLane operation {name!r} is not in the Stage 8C basic gate subset.").to_dict()
    op = _IR_GATE_MAP[normalized]
    controls = wires[:1] if op in {"cx", "cz"} else []
    targets = wires[1:] if controls else wires
    return {
        "op": op,
        "name": normalized,
        "targets": targets,
        "controls": controls,
        "wires": wires,
        "params": params,
        "supported": True,
        "warnings": [],
    }


def operation_sequence_to_quantumbridge_ir(ops) -> dict:
    instructions = []
    unsupported_operations = []
    warnings = []
    seen_wires = set()
    for op in ops:
        fragment = operation_to_quantumbridge_ir_fragment(op)
        if fragment.get("supported") is False:
            unsupported_operations.append(fragment)
            warnings.extend(fragment.get("warnings", []))
            continue
        instructions.append(fragment)
        seen_wires.update(fragment.get("wires", []))
    return {
        "schema_version": "0.1",
        "ir_version": "qb-ir-v0.1",
        "ecosystem": "pennylane",
        "wires": sorted(seen_wires, key=str),
        "operations": instructions,
        "instructions": instructions,
        "measurements": [],
        "parameters": [parameter for instruction in instructions for parameter in instruction.get("params", [])],
        "shots": None,
        "unsupported_operations": unsupported_operations,
        "warnings": warnings,
        "provenance": adapter_metadata(CapabilityLevel.SCHEMA_ADAPTER)["provenance"],
    }


def quantumbridge_ir_fragment_to_pennylane_operation(fragment) -> dict:
    payload = fragment.to_dict() if hasattr(fragment, "to_dict") else dict(fragment)
    op_name = str(payload.get("op", "")).lower()
    pennylane_name = _IR_TO_PENNYLANE.get(op_name)
    if not pennylane_name:
        return unsupported(f"QuantumBridge operation {payload.get('op')!r} is not in the Stage 8C PennyLane subset.").to_dict()
    wires = payload.get("wires") or payload.get("controls", []) + payload.get("targets", [])
    return {
        "ecosystem": "pennylane",
        "operation": pennylane_name,
        "wires": wires,
        "parameters": payload.get("params", []),
        "supported": True,
        "metadata_only": True,
        **adapter_metadata(CapabilityLevel.SCHEMA_ADAPTER),
    }
