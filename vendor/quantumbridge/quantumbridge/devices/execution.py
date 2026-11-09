# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This implementation is developed for the QuantumBridge native architecture.

from numbers import Integral


def validate_preparation_input(state, wires, width, kind):
    """A unitary amplitude preparation is defined here only on a zero subsystem."""
    import numpy as np
    from quantumbridge.core.classical import UnsupportedFeature
    if kind == "matrix_product_state":
        probability = state.probabilities(wires).get("0" * len(wires), 0.)
    elif kind in {"statevector", "exact_density_matrix"}:
        probabilities = np.abs(state) ** 2 if kind == "statevector" else np.diag(state).real
        index = tuple(0 if wire in wires else slice(None) for wire in range(width))
        probability = np.sum(probabilities.reshape((2,) * width)[index])
    else:
        raise UnsupportedFeature("state_preparation_device", "This device has no verified amplitude-preparation input contract.")
    if not np.isclose(probability, 1., atol=1e-10, rtol=0):
        raise UnsupportedFeature("state_preparation_input", "Native amplitude preparation requires a zero input subsystem; it does not reset an arbitrary input state.")


def needs_dynamic_engine(circuit,noise_model=None):
    return circuit.has_structured_control_flow or getattr(circuit,"has_simulator_instructions",False) or bool(getattr(noise_model,"requires_dynamic_engine",False))


def prepare_program(circuit, parameters=None, max_qubits=None, noise_model=None, *, _leaf_only=False):
    """Validate and order native unitary, CPTP and classical execution events."""
    import numpy as np

    from quantumbridge.core import Circuit, Measurement
    from quantumbridge.utils.math import gate_matrix, validate_kraus_operators, validate_unitary_matrix

    if not isinstance(circuit, Circuit):
        raise TypeError("QuantumBridge simulation requires a native Circuit.")
    bound = (circuit.bind(parameters) if parameters else circuit).lowered()
    if max_qubits is not None and bound.num_qubits > max_qubits:
        raise ValueError("QuantumBridge circuit exceeds this device qubit limit.")
    if not _leaf_only and needs_dynamic_engine(bound,noise_model):
        return bound, [("structured", bound, None, {})]
    noise = []
    if noise_model is not None:
        if not hasattr(noise_model, "channels") or not hasattr(noise_model, "readout_errors"):
            raise TypeError("QuantumBridge execution requires a native NoiseModel.")
        for entry in noise_model.channels:
            operators = validate_kraus_operators(entry["channel"].kraus(), 2)
            for wire in range(bound.num_qubits) if entry["wires"] is None else entry["wires"]:
                validate_wires((wire,), bound.num_qubits)
                noise.append(("channel", operators, (wire,), {}))
        for wire, error in noise_model.readout_errors.items():
            validate_wires((wire,), bound.num_qubits)
            if any(not np.isfinite(p) or not 0 <= p <= 1 for p in (error.p0_to_1, error.p1_to_0)):
                raise ValueError("QuantumBridge readout probabilities must be between zero and one.")
    reset = None
    program, inserted = [], False
    for item in bound.iter_instructions():
        validate_conditions(item.metadata, bound.num_bits)
        if isinstance(item, Measurement):
            bound._check_wire(item.wire)
            bound._check_bit(item.bit)
            if not inserted and item.metadata.get("operation_index", len(bound.operations)) == len(bound.operations):
                program.extend(noise)
                inserted = True
            program.append(("measure", item.wire, item.bit, item.metadata))
            continue
        if item.metadata.get("directive") and item.name in {"barrier", "delay"}:
            continue
        wires = validate_wires(item.wires, bound.num_qubits)
        if item.name == "reset":
            if item.controls or len(item.targets) != 1:
                raise ValueError("QuantumBridge reset requires one target and no controls.")
            if reset is None:reset = validate_kraus_operators(([[1, 0], [0, 0]], [[0, 1], [0, 0]]), 2)
            program.append(("channel", reset, wires, item.metadata))
        elif item.name == "kraus":
            if item.controls:
                raise ValueError("QuantumBridge Kraus instructions do not accept coherent controls.")
            operators = validate_kraus_operators(item.metadata.get("kraus", ()), 1 << len(wires))
            program.append(("channel", operators, wires, item.metadata))
        else:
            matrix = np.asarray(gate_matrix(item.name, item.params, item.metadata), dtype=complex)
            dimension = 1 << len(wires)
            if not validate_unitary_matrix(matrix, dimension):
                raise ValueError("QuantumBridge gate matrices must be finite unitaries matching their wires.")
            program.append(("gate", matrix, wires, item.metadata))
    if not inserted:
        program.extend(noise)
    return bound, program


def validate_wires(wires, width):
    selected = tuple(range(width) if wires is None else wires)
    if len(set(selected)) != len(selected) or any(not isinstance(wire, Integral) or not 0 <= wire < width for wire in selected):
        raise ValueError("QuantumBridge wires must be distinct valid circuit wires.")
    return selected


def validate_conditions(metadata, num_bits):
    frames = metadata.get("control_flow_stack", [])
    if metadata.get("control_flow") and not frames:
        raise ValueError("QuantumBridge control flow requires ordered branch metadata; rebuild this block with Circuit.if_else.")
    conditions = [metadata.get("condition"), metadata.get("gate_condition")]
    for frame in frames:
        if frame.get("type") != "if_else" or frame.get("branch") not in {"true", "false"} or not frame.get("block_id"):
            raise ValueError("QuantumBridge execution supports if/else control-flow blocks only.")
        conditions.append(frame.get("condition"))
    for condition in conditions:
        if condition is None:
            continue
        bits = condition.get("bits", [])
        if not bits or len(set(bits)) != len(bits) or any(not isinstance(bit, Integral) or not 0 <= bit < num_bits for bit in bits):
            raise ValueError("QuantumBridge condition bits are outside the classical register.")
        value = condition.get("value")
        if not isinstance(value, Integral) or not 0 <= value < (1 << len(bits)):
            raise ValueError("QuantumBridge condition value is outside its bit width.")


def condition_enabled(metadata, bits, decisions):
    def matches(condition):
        return sum(bits[bit] << index for index, bit in enumerate(condition["bits"])) == condition["value"]

    frames = metadata.get("control_flow_stack", [])
    for frame in frames:
        key = frame["block_id"]
        # Snapshot at block entry, even when the body overwrites tested bits.
        if frame.get("start") or key not in decisions:
            decisions[key] = matches(frame["condition"])
        if decisions[key] != (frame["branch"] == "true"):
            return False
    condition = metadata.get("gate_condition") if frames else metadata.get("condition")
    return condition is None or matches(condition)


def decision_liveness(program):
    """Find branch snapshots that future events still need before their next entry."""
    live = set()
    result = [set() for _ in program]
    for index in range(len(program) - 1, -1, -1):
        result[index] = live.copy()
        for frame in reversed(program[index][3].get("control_flow_stack", [])):
            if frame.get("start"):
                live.discard(frame["block_id"])
            else:
                live.add(frame["block_id"])
    return result


def readout_probability(noise_model, wire, outcome):
    error = None if noise_model is None else noise_model.readout_for((wire,)) if hasattr(noise_model,"readout_for") else noise_model.readout_errors.get(wire)
    return 0.0 if error is None else error.p1_to_0 if outcome else error.p0_to_1


def apply_readout_distribution(probabilities, noise_model, wires):
    result = dict(probabilities)
    if noise_model is None:
        return result
    for offset, wire in enumerate(wires):
        updated = {}
        for label, weight in result.items():
            probability = readout_probability(noise_model, wire, int(label[offset]))
            flipped = label[:offset] + ("0" if label[offset] == "1" else "1") + label[offset + 1:]
            for key, value in ((label, 1 - probability), (flipped, probability)):
                if value:
                    updated[key] = updated.get(key, 0.0) + weight * value
        result = updated
    return result
