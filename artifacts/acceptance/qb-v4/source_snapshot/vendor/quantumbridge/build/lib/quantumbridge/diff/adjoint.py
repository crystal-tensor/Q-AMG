# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Native inverse-state/reverse-cotangent differentiation; no SDK gradients.

import numpy as np

from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.utils.math import apply_unitary, gate_matrix, pauli_matrix, Hamiltonian, PauliString, observable_matrix


def gate_derivative_spec(operation):
    """Resolve bounded native control descriptors without constructing matrices."""
    from quantumbridge.core.gate_limits import current_gate_limits
    limits = current_gate_limits()
    name, metadata = operation.name, operation.metadata
    controls, inverse, seen = 0, False, set()
    while True:
        if type(name) is not str or type(metadata) is not dict or any(type(key) is not str for key in metadata):
            raise UnsupportedFeature("gate_derivative_description")
        if "controlled_base" not in metadata: break
        if id(metadata) in seen: raise ValueError("Cyclic controlled derivative description.")
        seen.add(id(metadata))
        limits.check("max_definition_depth",len(seen))
        description = metadata["controlled_base"]
        if (type(description) is not dict or any(type(key) is not str for key in description)
                or type(description.get("controls")) is not int or description["controls"] < 1):
            raise ValueError("Malformed controlled derivative description.")
        flag = description.get("inverse",False)
        if type(flag) is not bool: raise ValueError("Controlled derivative inverse must be a boolean.")
        controls += description["controls"]
        limits.check("max_control_qubits",controls)
        inverse ^= flag
        name,metadata = description.get("name"),description.get("metadata")
    base = {"crx":"rx", "cry":"ry", "crz":"rz", "cphase":"phase", "cp":"phase"}.get(name,name)
    controls += int(base != name)
    if base not in {"rx","ry","rz","phase","p","u1","rxx","ryy","rzz","global_phase"}:
        raise UnsupportedFeature("gate_derivative",f"No native derivative recipe for {name}.")
    limits.check("max_control_qubits",controls)
    width = 2 if base in {"rxx","ryy","rzz"} else 1
    if controls+width >= limits.max_matrix_dimension.bit_length():
        raise MemoryError("Controlled derivative exceeds max_matrix_dimension before allocation.")
    return base,controls,inverse


def gate_derivative(operation, parameters, *, parameter_index=0):
    if operation.name == "native_matrix_power":
        from .matrix_power_jet import matrix_power_derivative
        return matrix_power_derivative(parameters, operation.metadata, 1, parameter_index=parameter_index)
    if parameter_index != 0:
        raise UnsupportedFeature("gate_derivative_parameter_index")
    from quantumbridge.utils.math import _gate_array, _gate_identity
    from quantumbridge.execution.resource_context import current_resources
    base,controls,inverse = gate_derivative_spec(operation)
    scope = current_resources()
    if scope is not None:
        from quantumbridge.execution.array_reservations import numeric_shape
        if len(numeric_shape(parameters,real_only=True)) != 1:
            raise ValueError("Derivative parameters must be a one-dimensional numeric sequence.")
        if issubclass(type(parameters),np.ndarray): parameters = np.ndarray.view(parameters,np.ndarray)
    if len(parameters) != 1: raise ValueError("A native derivative recipe requires one angle.")
    width = 4 if base in {"rxx","ryy","rzz"} else 2
    dimension = width*(1<<controls)
    if scope is not None:
        scope.ledger.preflight(memory_bytes=16*(width*width+(dimension*dimension if controls else 0)))
    theta = parameters[0]
    s,c = -.5*np.sin(theta/2),-.5j*np.cos(theta/2)
    if base == "rx": derivative = _gate_array([[s,c],[c,s]])
    elif base == "ry": derivative = _gate_array([[s,-1j*c],[1j*c,s]])
    elif base == "rz": derivative = _gate_array([[-.5j*np.exp(-.5j*theta),0],[0,.5j*np.exp(.5j*theta)]])
    elif base in {"phase","p","u1"}: derivative = _gate_array([[0,0],[0,1j*np.exp(1j*theta)]])
    elif base == "global_phase": derivative = _gate_identity(2,factor=1j*np.exp(1j*theta))
    else:
        derivative = _gate_identity(4,factor=s)
        # Pauli-pair generators have one signed entry per row.
        for row in range(4):
            col = row if base == "rzz" else row^3
            sign = (1 if row in (0,3) else -1) if base == "rzz" else (-1 if row in (0,3) else 1) if base == "ryy" else 1
            derivative[row,col] += sign*c
    if inverse:
        np.conjugate(derivative,out=derivative)
        derivative = derivative.T
    if not controls: return derivative
    controlled = _gate_identity(dimension,factor=0)
    controlled[-width:,-width:] = derivative
    return controlled


def gate_second_derivative(operation, parameters):
    if operation.name == "native_matrix_power":
        from .matrix_power_jet import matrix_power_derivative
        return matrix_power_derivative(parameters, operation.metadata, 2)
    from quantumbridge.execution import mps_arrays as arrays
    derivative = gate_derivative(operation, parameters)
    matrix = gate_matrix(operation.name, parameters, operation.metadata)
    return arrays.contracted(arrays.contracted(derivative, arrays.conjugated(matrix).T), derivative)


def apply_observable(state, observable, width):
    if hasattr(observable, "to_hamiltonian"): observable = observable.to_hamiltonian()
    if isinstance(observable, PauliString):
        result = state
        for wire, label in observable.terms: result = apply_unitary(result, pauli_matrix(label), (wire,), width)
        return result
    if isinstance(observable, Hamiltonian):
        return sum((coefficient * apply_observable(state, pauli, width) for coefficient, pauli in observable.terms), np.zeros_like(state))
    if width > 12: raise UnsupportedFeature("dense_observable", "Large adjoint observables require native Pauli/Hamiltonian form.")
    matrix = observable_matrix(observable, width)
    from quantumbridge.utils.math import validate_observable_matrix
    if not validate_observable_matrix(matrix, width, atol=1e-8, rtol=1e-5): raise ValueError("Adjoint requires a Hermitian observable matching circuit width.")
    return matrix @ state


def adjoint_rows(device, circuit, angle_parameters, values, requests):
    from quantumbridge.devices import StatevectorDevice
    if type(device) is not StatevectorDevice or circuit.has_dynamic_operations or getattr(device,"noise_model",None) is not None:
        raise UnsupportedFeature("adjoint_device", "Native adjoint requires analytic unitary StatevectorDevice execution.")
    mapping = dict(zip(angle_parameters, values))
    bound = circuit.bind(mapping)
    state = device.statevector(bound)
    # The reverse sweep retains a constant number of state vectors per output.
    if device.max_memory_mb is not None and state.nbytes * 8 > device.max_memory_mb * 1024 ** 2:
        raise MemoryError("Adjoint buffers exceed max_memory_mb.")
    rows = []
    for request in requests:
        if request.kind in {"expval", "var"}:
            operated = apply_observable(state, request.observable, circuit.num_qubits)
            if request.kind == "var":
                mean = np.vdot(state, operated).real
                cotangents = [(apply_observable(operated, request.observable, circuit.num_qubits), -2 * mean * operated)]
            else: cotangents = [(operated,)]
        elif request.kind == "probs":
            wires = tuple(range(circuit.num_qubits)) if request.wires is None else request.wires
            if len(wires) > 16: raise UnsupportedFeature("probability_jacobian_size", "Probability Jacobian output exceeds the native budget.")
            labels = np.zeros(state.size, dtype=np.int64)
            for wire in wires: labels = 2 * labels + ((np.arange(state.size) >> (circuit.num_qubits - 1 - wire)) & 1)
            if state.size * (1 << len(wires)) * max(1, len(bound.operations)) > 100000000:
                raise MemoryError("Adjoint probability derivative work exceeds the native contraction budget.")
            cotangents = ((state * (labels == label),) for label in range(1 << len(wires)))
        else: raise UnsupportedFeature("adjoint_measurement", f"No adjoint rule for {request.kind}.")
        for terms in cotangents:
            psi, adjoint, row = state.copy(), sum(terms), np.zeros(len(values))
            for raw, operation in zip(reversed(circuit.operations), reversed(bound.operations)):
                if operation.metadata.get("directive") and operation.name in {"barrier", "delay"}: continue
                matrix = gate_matrix(operation.name, operation.params, operation.metadata)
                before = apply_unitary(psi, matrix.conj().T, operation.wires, circuit.num_qubits)
                for parameter_index, parameter in enumerate(raw.params):
                    if parameter not in angle_parameters: continue
                    if raw.name != "native_matrix_power" and (parameter_index or len(raw.params) != 1):
                        raise UnsupportedFeature("adjoint_gate_parameters")
                    derivative = gate_derivative(raw, operation.params, parameter_index=parameter_index)
                    row[angle_parameters.index(parameter)] += 2 * np.vdot(adjoint, apply_unitary(before, derivative, operation.wires, circuit.num_qubits)).real
                adjoint = apply_unitary(adjoint, matrix.conj().T, operation.wires, circuit.num_qubits)
                psi = before
            rows.append(row)
    return np.asarray(rows)
