# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Joint terminal classical records and quantum samples on the native engine."""

from itertools import islice

from quantumbridge.core.classical import ClassicalExpression, UnsupportedFeature
from quantumbridge.execution.control_flow_engine import DynamicExecutionEngine
from quantumbridge.devices.execution import readout_probability
from quantumbridge.utils.math import probabilities_from_state


def record_metadata(records):
    if isinstance(records, ClassicalExpression): records = (records,)
    records = tuple(islice(records, 17))
    if len(records) > 16: raise MemoryError("Terminal record selection exceeds 16 bits.")
    if any(not isinstance(record, ClassicalExpression) or record.op != "bit" for record in records):
        raise TypeError("Terminal measurement records require native bit references.")
    if len({record.value for record in records}) != len(records):
        raise ValueError("Terminal measurement records cannot contain aliases.")
    return {"measurement_records": records} if records else None


def execute_records(device, circuit, request):
    records = request.metadata["measurement_records"]
    bits = tuple(record.value for record in records)
    wires = tuple(request.wires or ())
    if any(not 0 <= bit < circuit.num_bits for bit in bits):
        raise ValueError("Terminal measurement record is outside the classical register.")
    if circuit.num_bits > 100000:
        raise MemoryError("Terminal record storage exceeds the native classical-bit budget.")
    if len(bits) + len(wires) > 16:
        raise UnsupportedFeature("record_output_size", "Joint classical/quantum output exceeds 16 bits.")
    engine = DynamicExecutionEngine(device, circuit)
    resources = None
    def capture(state):
        nonlocal resources
        if engine.adapter.kind != "matrix_product_state": return
        current = device._metadata(circuit, [state])
        if resources is None: resources = current
        else:
            for key in ("max_bond_observed", "max_discarded_weight", "max_tensor_bytes", "max_truncations"):
                resources[key] = max(resources[key], current[key])
            resources["max_bond_dimensions_observed"] = [max(a, b) for a, b in zip(resources["max_bond_dimensions_observed"], current["max_bond_dimensions_observed"])]
    labels = [format(index, f"0{len(bits)+len(wires)}b") for index in range(1 << (len(bits)+len(wires)))]
    if request.shots is None:
        if request.kind != "probs": raise ValueError("Record sample/counts need finite shots.")
        result = dict.fromkeys(labels, 0.)
        for state, memory, _, weight in engine.branches():
            capture(state)
            prefix = "".join(str(memory[bit]) for bit in bits)
            if not wires: probabilities = {"": 1.}
            elif engine.adapter.kind == "statevector":
                probabilities = probabilities_from_state(state, circuit.num_qubits, wires)
            elif engine.adapter.kind == "exact_density_matrix":
                probabilities = device._probabilities(state, wires, circuit.num_qubits)
            elif engine.adapter.kind == "matrix_product_state":
                probabilities = state.probabilities(wires)
            else:
                raise UnsupportedFeature("record_device", "Analytic joint records require native statevector, density or MPS.")
            for label, probability in probabilities.items(): result[prefix + label] += weight * probability
        if resources is not None: device.last_execution_metadata = resources
        return result
    memory = []
    for state, classical, rng in engine.trajectories(request.shots, request.seed):
        capture(state)
        label = [str(classical[bit]) for bit in bits]
        for wire in wires:
            _, state, outcome = engine._choose(engine.adapter.project(state, wire), rng)[0]
            p = readout_probability(engine.noise_model, wire, outcome)
            label.append(str(outcome ^ int(p > 0 and rng.random() < p)))
        memory.append("".join(label))
    if resources is not None: device.last_execution_metadata = {**resources, "shots": request.shots, "seed": request.seed}
    if request.kind == "sample": return memory
    counts = {}
    for label in memory: counts[label] = counts.get(label, 0) + 1
    if request.kind == "counts": return counts
    return {label: counts.get(label, 0) / request.shots for label in labels}
