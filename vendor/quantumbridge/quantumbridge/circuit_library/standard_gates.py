# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

from quantumbridge.core import Circuit


_ONE_QUBIT = {"id", "x", "y", "z", "h", "s", "sdg", "sx", "sxdg", "t", "tdg", "rx", "ry", "rz", "phase", "p", "u", "u1", "u2", "u3"}
_TWO_QUBIT = {
    "cx",
    "cy",
    "cz",
    "ch",
    "csx",
    "csxdg",
    "swap",
    "dcx",
    "ecr",
    "iswap",
    "crx",
    "cry",
    "crz",
    "cp",
    "cphase",
    "rxx",
    "ryy",
    "rzz",
}
_THREE_QUBIT = {"ccx", "cswap", "fredkin"}
_DEFINITION_GATES = {
    "id", "h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg",
    "rx", "ry", "rz", "phase", "p", "u1", "u", "u2", "u3",
    "cx", "cz", "swap", "crx", "cry", "crz", "cp", "cphase", "rxx", "ryy", "rzz",
}


def standard_gate_definition(name, *params):
    """Return a phase-exact native definition using existing compiler recipes.

    Definitions are not canonical public SDK instruction classes. Each call
    owns its circuit; recursive decomposition remains an explicit operation.
    """
    from quantumbridge.core.gates import _validate_primitive_arity, validate_definition
    from quantumbridge.core.gate_limits import current_gate_limits
    from quantumbridge.compiler.decomposition import GateDecompositionPass
    from quantumbridge.core.classical import UnsupportedFeature
    gate = name.lower()
    if gate not in _DEFINITION_GATES: raise UnsupportedFeature("standard_gate_definition", f"No native definition recipe for {name!r}.")
    _validate_primitive_arity(gate, params, {})
    source = standard_gate(gate, *params)
    validate_definition(source)
    if gate == "rz":
        body = Circuit(1).phase(params[0], 0)
        body.global_phase = -params[0]/2
    else:
        limits = current_gate_limits()
        body = GateDecompositionPass((_ONE_QUBIT | _TWO_QUBIT | _THREE_QUBIT)-{gate}).run(
            source, max_operations=limits.max_total_operations,
            max_expansion_steps=limits.max_decomposition_work,
            max_depth=limits.max_definition_depth).circuit
    validate_definition(body)
    return body


def standard_gate_object(name, *params, label=None):
    """Construct a native Gate with an independently owned real definition."""
    from quantumbridge.core.gates import Gate
    from quantumbridge.core import Parameter
    formals = tuple(Parameter("_qb_slot_"+str(index)) for index in range(len(params)))
    body = standard_gate_definition(name, *formals)
    result = Gate(name.lower(), body.num_qubits, formals, label, definition=body)
    if params: result.params = params
    return result


def standard_gate(name: str, *params, num_qubits: int | None = None) -> Circuit:
    """Build a circuit containing a single standard gate operation."""

    gate = name.lower()
    if gate in _ONE_QUBIT:
        circuit = Circuit(num_qubits or 1)
        _append_one_qubit(circuit, gate, params, 0)
        return circuit
    if gate in _TWO_QUBIT:
        circuit = Circuit(num_qubits or 2)
        _append_two_qubit(circuit, gate, params, 0, 1)
        return circuit
    if gate in _THREE_QUBIT:
        circuit = Circuit(num_qubits or 3)
        if gate == "ccx":
            circuit.ccx(0, 1, 2)
        else:
            circuit.cswap(0, 1, 2)
        return circuit
    raise ValueError(f"QuantumBridge standard gate {name!r} is not registered.")


def _append_one_qubit(circuit: Circuit, gate: str, params: tuple, wire: int) -> None:
    if gate in {"rx", "ry", "rz", "phase", "p", "u1"}:
        if not params:
            raise ValueError(f"QuantumBridge {gate} requires one parameter.")
        getattr(circuit, gate)(params[0], wire)
    elif gate == "u2":
        if len(params) != 2:
            raise ValueError("QuantumBridge u2 requires two parameters.")
        circuit.u2(params[0], params[1], wire)
    elif gate in {"u", "u3"}:
        if len(params) != 3:
            raise ValueError(f"QuantumBridge {gate} requires three parameters.")
        getattr(circuit, gate)(params[0], params[1], params[2], wire)
    else:
        getattr(circuit, gate)(wire)


def _append_two_qubit(circuit: Circuit, gate: str, params: tuple, left: int, right: int) -> None:
    if gate in {"cx", "cy", "cz", "ch", "csx", "csxdg"}:
        getattr(circuit, gate)(left, right)
    elif gate in {"swap", "dcx", "ecr", "iswap"}:
        getattr(circuit, gate)(left, right)
    elif gate in {"crx", "cry", "crz", "cp", "cphase"}:
        if not params:
            raise ValueError(f"QuantumBridge {gate} requires one parameter.")
        getattr(circuit, gate)(params[0], left, right)
    elif gate in {"rxx", "ryy", "rzz"}:
        if not params:
            raise ValueError(f"QuantumBridge {gate} requires one parameter.")
        getattr(circuit, gate)(params[0], left, right)
    else:
        raise ValueError(f"QuantumBridge standard gate {gate!r} is not a two-qubit gate.")
