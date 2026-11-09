# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Native tangent-over-reverse second derivatives with constant state buffers."""

from quantumbridge.execution.resource_context import copy_execution_payload as deepcopy
from uuid import uuid4

import numpy as np

from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.qml.measurements import MeasurementRequest
from quantumbridge.utils.math import Hamiltonian, PauliString, apply_unitary, gate_matrix, observable_matrix
from .adjoint import apply_observable, gate_derivative, gate_second_derivative
from .processing import result_payload
from .trace import flatten_tree


def _terminals(plan):
    for request in plan.requests:
        if request.kind == "pauli_group":
            for word in request.observable.words:
                yield MeasurementRequest("expval", observable=PauliString(word))
        else:
            yield request


def _observable_check(observable, width):
    if hasattr(observable, "to_hamiltonian"):
        observable = observable.to_hamiltonian()
    terms = observable.terms if isinstance(observable, Hamiltonian) else ((1., observable),)
    if isinstance(observable, (Hamiltonian, PauliString)):
        for coefficient, word in terms:
            value = np.asarray(coefficient)
            if value.shape or value.dtype.kind not in "fiuc" or not np.isfinite(value) or np.imag(value) != 0:
                raise UnsupportedFeature("adjoint_observable", "Adjoint needs fixed finite real Pauli coefficients; transform live coefficients into processing slots.")
            if any(wire < 0 or wire >= width for wire, _ in word.terms):
                raise ValueError("Adjoint observable wires exceed circuit width.")
        return
    if width > 12:
        raise UnsupportedFeature("dense_observable", "Large adjoint observables require native Pauli/Hamiltonian form.")
    matrix = observable_matrix(observable, width)
    from quantumbridge.utils.math import validate_observable_matrix
    if not validate_observable_matrix(matrix, width, atol=1e-8, rtol=1e-5):
        raise ValueError("Adjoint requires a finite Hermitian observable matching circuit width.")


def second_adjoint_work(plan, values, classical, classical_second, direction, limits):
    """Preflight numerical work and combined working buffers before simulation."""
    plan.gradient_capability("adjoint")
    width, size = plan.circuit.num_qubits, classical.shape[1]
    if getattr(plan.node.device, "noise_model", None) is not None:
        raise UnsupportedFeature("adjoint_device", "Second adjoint requires a noiseless analytic StatevectorDevice.")
    if plan.node.device.max_qubits is not None and width > plan.node.device.max_qubits:
        raise ValueError("Second adjoint exceeds this device qubit limit.")
    limits.check("max_derivative_order", 2)
    columns = size if direction is None else 1
    output_size, dense_bytes, observable_work = 0, 0, 0
    for request in _terminals(plan):
        if (request.metadata or {}).get("measurement_records"):
            raise UnsupportedFeature("second_order_measurement_records")
        if request.kind == "probs":
            wires = tuple(range(width)) if request.wires is None else tuple(request.wires)
            if len(set(wires)) != len(wires) or any(wire < 0 or wire >= width for wire in wires):
                raise ValueError("Second adjoint probability wires must be unique and in range.")
            if len(wires) > 16:
                raise UnsupportedFeature("probability_output_size")
            output_size += 1 << len(wires)
            observable_work += (1 << len(wires))*max(1, len(wires))
        elif request.kind in {"expval", "var"}:
            output_size += 1
            if not isinstance(request.observable, (PauliString, Hamiltonian)) and not hasattr(request.observable, "to_hamiltonian"):
                dense_bytes = max(dense_bytes, 16*(1 << (2*width)))
                observable_work += 1 << width
            else:
                observable = request.observable.to_hamiltonian() if hasattr(request.observable, "to_hamiltonian") else request.observable
                words = [word for _, word in observable.terms] if isinstance(observable, Hamiltonian) else [observable]
                observable_work += sum(4*max(1, len(word.terms)) for word in words)
        else:
            raise UnsupportedFeature("second_order_measurement")
    limits.check("max_output_elements", output_size)
    if direction is None:
        limits.check("max_hessian_elements", output_size*size*columns)
    # The state and derivative working arrays share this conservative budget.
    # Owned input arrays and arbitrary user converter internals are not sandboxed.
    derivative_bytes = 8*(4*output_size*(1+size+size*columns)+classical.size+classical_second.size)
    local_dimension = max((1 << len(op.wires) for op in plan.circuit.operations
                           if not op.metadata.get("directive")), default=1)
    buffer_bytes = 20*16*(1 << width)+4*dense_bytes+4*16*local_dimension**2+derivative_bytes
    matrix_functions = [op for op in plan.circuit.operations if op.name == "native_matrix_power"]
    if matrix_functions:
        buffer_bytes += max(16*128*(4*(1 << len(op.wires)))**2 for op in matrix_functions)
    limits.check("max_memory_bytes", buffer_bytes)
    device_limit = plan.node.device.max_memory_mb
    if device_limit is not None and buffer_bytes > device_limit*1024**2:
        raise MemoryError("Second adjoint combined working buffers exceed device max_memory_mb.")
    operations = max(1, columns)*(1 << width)*(max(1, len(plan.circuit.operations))*local_dimension*(3+16*output_size)+8*observable_work)
    matrix_function_work = 0
    if matrix_functions:
        from quantumbridge.core.gate_limits import current_gate_limits
        # Count preflight, forward direction, reverse direction and per-input
        # mixed jets; each evaluator is itself capped by the native work guard.
        calls = sum(len(op.params) for op in matrix_functions)+columns*sum(
            1+output_size*(1+len(op.params)) for op in matrix_functions)
        matrix_function_work = calls*current_gate_limits().max_decomposition_work
        operations += matrix_function_work
    limits.check("max_adjoint_operations", operations)
    occurrences = plan._occurrences()
    mapping = dict(zip(plan.recorder.parameters, values))
    for index, parameter in enumerate(plan.recorder.parameters):
        if parameter in occurrences and (np.any(classical[index]) or np.any(classical_second[index])):
            operation = occurrences[parameter]
            gate_derivative(operation, tuple(mapping.get(item, item) for item in operation.params),
                parameter_index=operation.params.index(parameter))
    bound = plan.circuit.bind(mapping)
    for operation in bound.operations:
        if operation.metadata.get("directive") and operation.name in {"barrier", "delay"}:
            continue
        matrix = gate_matrix(operation.name, operation.params, operation.metadata)
        from quantumbridge.utils.math import validate_unitary_matrix
        if not validate_unitary_matrix(matrix, 1 << len(operation.wires), atol=1e-8, rtol=1e-5):
            raise UnsupportedFeature("adjoint_unitary", "Second adjoint requires finite unitary gate matrices.")
    for request in _terminals(plan):
        if request.kind != "probs":
            _observable_check(request.observable, width)
    work = {"first_shift_count": 0, "second_shift_count": 0, "shift_count": 0,
            "evaluation_count": 1+columns, "tape_count": 1+columns,
            "physical_shots": 0, "sample_elements": 0, "derivative_buffer_bytes": derivative_bytes,
            "combined_working_buffer_bytes": buffer_bytes, "adjoint_operations": operations,
            "matrix_function_derivative_work_bound": matrix_function_work,
            "adjoint_second_sweeps": columns*output_size, "output_size": output_size,
            "linear_size": output_size, "method": "adjoint", "direction_independent_sampling_schedule": False}
    limits.check("max_total_tapes", work["tape_count"])
    return {}, [], classical if direction is None else (classical @ direction).reshape(len(values), 1), work


def _cotangents(request, state, tangent, width):
    if request.kind == "probs":
        wires = tuple(range(width)) if request.wires is None else request.wires
        labels = np.zeros(state.size, dtype=np.int64)
        basis = np.arange(state.size)
        for wire in wires:
            labels = 2*labels+((basis >> (width-1-wire)) & 1)
        for label in range(1 << len(wires)):
            mask = labels == label
            yield state*mask, tangent*mask
        return
    operated = apply_observable(state, request.observable, width)
    tangent_operated = apply_observable(tangent, request.observable, width)
    if request.kind == "var":
        mean = np.vdot(state, operated).real
        mean_tangent = 2*np.vdot(operated, tangent).real
        yield (apply_observable(operated, request.observable, width)-2*mean*operated,
               apply_observable(tangent_operated, request.observable, width)-2*mean*tangent_operated-2*mean_tangent*operated)
    else:
        yield operated, tangent_operated


def execute_second_adjoint(plan, values, classical, classical_second, direction, work):
    base = plan.evaluate(values)
    base_metadata = deepcopy(plan.metadata)
    flat, spec = flatten_tree(result_payload(base))
    if flat.size != work["output_size"]:
        raise ValueError("Second adjoint output changed after preflight.")
    from contextlib import ExitStack
    from quantumbridge.execution.resource_context import current_resources
    resources = current_resources()
    with ExitStack() as stack:
        if resources is not None:
            ledger = resources.ledger
            vector_bytes = 16*(1 << plan.circuit.num_qubits)
            for name, count in (("adjoint_forward_states", 4), ("adjoint_tangent_states", 4),
                                ("adjoint_reverse_states", 8), ("adjoint_temporary_states", 4)):
                stack.enter_context(ledger.reserve(name, memory_bytes=count*vector_bytes))
            other = work["combined_working_buffer_bytes"]-work["derivative_buffer_bytes"]-20*vector_bytes
            stack.enter_context(ledger.reserve("adjoint_local_matrices", memory_bytes=other))
            ledger.consume(tapes=classical.shape[1] if direction is None else 1,
                           category="native_second_adjoint_directions")
        return _execute_second_adjoint(plan, values, classical, classical_second, direction, work,
                                       base, base_metadata, flat, spec)


def _execute_second_adjoint(plan, values, classical, classical_second, direction, work, base, base_metadata, flat, spec):
    width, size = plan.circuit.num_qubits, classical.shape[1]
    columns = size if direction is None else 1
    jacobian = np.zeros((flat.size, size))
    second = np.zeros((flat.size, size, columns))
    bound = plan.circuit.bind(dict(zip(plan.recorder.parameters, values)))
    indices = {parameter: index for index, parameter in enumerate(plan.recorder.parameters)}
    active = {index for index in range(len(values)) if np.any(classical[index]) or np.any(classical_second[index])}
    receipts = [{"evaluation_id": str(uuid4()), "kind": "base", **base_metadata}]
    state = plan.node.device.statevector(bound) if columns else None
    def apply(vector, matrix, operation):
        return apply_unitary(vector, matrix, operation.wires, width)
    for column in range(columns):
        projection = classical[:, column] if direction is None else classical @ direction
        primal, tangent = plan.node.device._initial(width), np.zeros_like(state)
        for raw, operation in zip(plan.circuit.operations, bound.operations):
            if operation.metadata.get("directive") and operation.name in {"barrier", "delay"}:
                continue
            matrix = gate_matrix(operation.name, operation.params, operation.metadata)
            index = indices.get(raw.params[0]) if len(raw.params) == 1 else None
            tangent_next = apply(tangent, matrix, operation)
            if raw.name == "native_matrix_power":
                from .matrix_power_jet import matrix_power_directional
                local_direction = tuple(projection[indices[p]] if indices.get(p) in active else 0. for p in raw.params)
                if any(local_direction):
                    tangent_next += apply(primal, matrix_power_directional(operation.params, raw.metadata, local_direction), operation)
            elif index in active:
                tangent_next += projection[index]*apply(primal, gate_derivative(raw, operation.params), operation)
            primal, tangent = apply(primal, matrix, operation), tangent_next
        phase = np.exp(1j*float(bound.global_phase))
        primal, tangent = phase*primal, phase*tangent
        if not np.allclose(primal, state, atol=1e-12, rtol=0):
            raise RuntimeError("Second adjoint primal differs from actual native device execution.")
        row = 0
        for request in _terminals(plan):
            for adjoint, adjoint_tangent in _cotangents(request, state, tangent, width):
                psi, psi_tangent = state.copy(), tangent.copy()
                for raw, operation in zip(reversed(plan.circuit.operations), reversed(bound.operations)):
                    if operation.metadata.get("directive") and operation.name in {"barrier", "delay"}:
                        continue
                    matrix = gate_matrix(operation.name, operation.params, operation.metadata)
                    inverse = matrix.conj().T
                    before, before_tangent = apply(psi, inverse, operation), apply(psi_tangent, inverse, operation)
                    if raw.name == "native_matrix_power":
                        from .matrix_power_jet import matrix_power_directional, matrix_power_directions
                        local_direction = tuple(projection[indices[p]] if indices.get(p) in active else 0. for p in raw.params)
                        directional = matrix_power_directional(operation.params, raw.metadata, local_direction) if any(local_direction) else None
                        if directional is not None:
                            before_tangent += apply(psi, directional.conj().T, operation)
                        for local, parameter in enumerate(raw.params):
                            index = indices.get(parameter)
                            if index not in active: continue
                            seed = tuple(float(j == local) for j in range(len(raw.params)))
                            _, derivative, _, mixed = matrix_power_directions(operation.params, raw.metadata, seed, local_direction)
                            differentiated = apply(before, derivative, operation)
                            gradient = 2*np.vdot(adjoint, differentiated).real
                            gradient_tangent = 2*(np.vdot(adjoint_tangent, differentiated)
                                + np.vdot(adjoint, apply(before_tangent, derivative, operation))
                                + np.vdot(adjoint, apply(before, mixed, operation))).real
                            if column == 0: jacobian[row] += gradient*classical[index]
                            second[row, :, column] += gradient_tangent*classical[index]+gradient*classical_second[index, :, column]
                        previous_tangent = apply(adjoint_tangent, inverse, operation)
                        if directional is not None:
                            previous_tangent += apply(adjoint, directional.conj().T, operation)
                        adjoint, adjoint_tangent = apply(adjoint, inverse, operation), previous_tangent
                        psi, psi_tangent = before, before_tangent
                        continue
                    index = indices.get(raw.params[0]) if len(raw.params) == 1 else None
                    derivative = gate_derivative(raw, operation.params) if index in active else None
                    if derivative is not None:
                        before_tangent += projection[index]*apply(psi, derivative.conj().T, operation)
                        differentiated = apply(before, derivative, operation)
                        gradient = 2*np.vdot(adjoint, differentiated).real
                        second_gate = (gate_second_derivative(raw, operation.params)
                            if raw.name == "native_matrix_power" else derivative @ inverse @ derivative)
                        gradient_tangent = 2*(np.vdot(adjoint_tangent, differentiated)
                            + np.vdot(adjoint, apply(before_tangent, derivative, operation))
                            + projection[index]*np.vdot(adjoint, apply(before, second_gate, operation))).real
                        if column == 0:
                            jacobian[row] += gradient*classical[index]
                        second[row, :, column] += gradient_tangent*classical[index]+gradient*classical_second[index, :, column]
                    previous_tangent = apply(adjoint_tangent, inverse, operation)
                    if derivative is not None:
                        previous_tangent += projection[index]*apply(adjoint, derivative.conj().T, operation)
                    adjoint, adjoint_tangent = apply(adjoint, inverse, operation), previous_tangent
                    psi, psi_tangent = before, before_tangent
                row += 1
        receipts.append({"evaluation_id": str(uuid4()), "kind": "native_second_adjoint", "column": column,
                         "forward_tangent_sweeps": 1, "reverse_tangent_sweeps": row, "physical_shots": 0})
    if not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(second)):
        raise ValueError("Second adjoint produced a nonfinite derivative.")
    plan.metadata = {**base_metadata, "derivative_order": 2, "gradient_method": "adjoint",
        "second_order_method": "native tangent-over-reverse adjoint", "hvp_streamed": direction is not None,
        "first_shifted_evaluation_count": 0, "second_shifted_evaluation_count": 0,
        "shifted_evaluation_count": 0, "derivative_evaluation_count": columns,
        "adjoint_second_sweeps": work["adjoint_second_sweeps"], "second_order_evaluations": receipts,
        "second_order_work": work, "total_physical_shots": 0, "gradient_physical_shots": 0,
        "gradient_semantics": "analytic", "memory_budget_scope": "combined second-adjoint working state and derivative buffers; not a universal process memory cap"}
    return base, jacobian, second, spec
