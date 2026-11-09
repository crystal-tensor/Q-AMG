# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Forward mixed MPS jets on the existing QNode/Device execution plans."""

from contextlib import ExitStack, nullcontext
from itertools import product
from uuid import uuid4

import numpy as np

from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.core.gate_limits import current_gate_limits
from quantumbridge.execution import mps_arrays as arrays
from quantumbridge.execution.resource_context import current_resources, copy_execution_payload
from quantumbridge.information import MatrixProductState
from quantumbridge.utils.math import Hamiltonian, PauliString, gate_matrix
from .adjoint import gate_derivative, gate_second_derivative
from .higher_order_limits import resolve_higher_order_limits
from .mps_tangent import MPSTangentJet
from .processing import result_payload
from .trace import flatten_tree


METHOD = "mps-tangent"


def _terms(observable, width):
    if hasattr(observable, "to_hamiltonian"):
        observable = observable.to_hamiltonian()
    terms = observable.terms if isinstance(observable, Hamiltonian) else ((1., observable),)
    result = []
    for coefficient, word in terms:
        value = np.asarray(coefficient)
        if value.shape or value.dtype.kind not in "fiuc" or not np.isfinite(value) or value.imag != 0:
            raise UnsupportedFeature("mps_tangent_observable", "MPS jets require fixed finite real observable coefficients.")
        if not isinstance(word, PauliString):
            raise UnsupportedFeature("mps_tangent_observable", "Use native Pauli/Hamiltonian observables for MPS derivatives.")
        label = ["I"]*width
        if len(dict(word.terms)) != len(word.terms):
            raise ValueError("MPS observable wires must be distinct.")
        for wire, char in word.terms:
            if not 0 <= wire < width:
                raise ValueError("MPS observable wires exceed circuit width.")
            label[wire] = char
        result.append((float(value.real), "".join(label)))
    return tuple(result)


def _square(terms):
    current_gate_limits().check("max_decomposition_work", len(terms)**2)
    table = {("X", "Y"): (1j, "Z"), ("Y", "Z"): (1j, "X"), ("Z", "X"): (1j, "Y"),
             ("Y", "X"): (-1j, "Z"), ("Z", "Y"): (-1j, "X"), ("X", "Z"): (-1j, "Y")}
    result = {}
    for (a, first), (b, second) in product(terms, repeat=2):
        coefficient, label = complex(a*b), []
        for x, y in zip(first, second):
            if x == "I":
                label.append(y)
            elif y == "I":
                label.append(x)
            elif x == y:
                label.append("I")
            else:
                phase, char = table[x, y]
                coefficient *= phase
                label.append(char)
        key = "".join(label)
        result[key] = result.get(key, 0j)+coefficient
    if any(abs(value.imag) > 1e-12 for value in result.values()):
        raise ValueError("MPS observable square is not Hermitian.")
    return tuple((float(value.real), key) for key, value in result.items() if value.real != 0)


def _measurements(plan, limits):
    width, rows, work, compiled = plan.circuit.num_qubits, 0, 0, []
    for request in plan.requests:
        if (request.metadata or {}).get("measurement_records"):
            raise UnsupportedFeature("mps_tangent_measurement_records")
        if request.kind in {"expval", "var"}:
            terms = _terms(request.observable, width)
            squared = _square(terms) if request.kind == "var" else ()
            rows += 1
            work += len(terms)+len(squared)
            compiled.append((request.kind, terms, squared))
        elif request.kind == "pauli_group":
            request.observable.validate(width)
            for word in request.observable.words:
                terms = _terms(PauliString(word), width)
                rows += 1
                work += len(terms)
                compiled.append(("expval", terms, ()))
        elif request.kind == "probs":
            wires = tuple(range(width)) if request.wires is None else tuple(request.wires)
            if len(set(wires)) != len(wires) or any(wire < 0 or wire >= width for wire in wires):
                raise ValueError("MPS probability wires must be distinct and valid.")
            if len(wires) > 16:
                raise UnsupportedFeature("probability_output_size")
            rows += 1 << len(wires)
            work += 1 << len(wires)
            compiled.append(("probs", wires, ()))
        else:
            raise UnsupportedFeature("mps_tangent_measurement")
        limits.check("max_output_elements", rows)
    return compiled, rows, work


def _local_seeds(operation, indices, first, other, mixed):
    slots = [indices.get(parameter) for parameter in operation.params]
    return tuple(tuple(0. if index is None else float(seed[index]) for index in slots)
                 for seed in (first, other, mixed))


def _routing_steps(wires, width):
    if len(wires) < 2:
        return 0
    from quantumbridge.information.matrix_product_state import mps_routing
    return 2*len(mps_routing(tuple(int(wire) for wire in wires), width)[1])


def _bond_bound(device, circuit):
    width, cap = circuit.num_qubits, device.state_options["max_bond_dim"]
    initial = device.initial_state
    if initial is not None and type(initial) is not MatrixProductState:
        raise UnsupportedFeature("derivative_mps_dense_initial_state")
    if initial is not None and initial.discarded_weight:
        raise UnsupportedFeature("mps_tangent_truncated_initial_state")
    bonds = [1]*(width-1) if initial is None else list(initial.bond_dimensions)
    if len(bonds) != width-1:
        raise ValueError("MPS derivative initial width does not match the circuit.")
    for operation in circuit.operations:
        if operation.metadata.get("directive") and operation.name in {"barrier", "delay"}:
            continue
        if len(operation.wires) > 1:
            for cut in range(min(operation.wires), max(operation.wires)):
                exponent = min(sum(w <= cut for w in operation.wires), sum(w > cut for w in operation.wires))
                bonds[cut] = min(bonds[cut]*4**exponent, 1 << min(cut+1, width-cut-1))
                if cap is not None:
                    bonds[cut] = min(bonds[cut], cap)
    dimensions = [1, *bonds, 1]
    return max(dimensions), 32*sum(a*b for a, b in zip(dimensions, dimensions[1:]))


def mps_work(plan, values, classical, classical_second, direction, limits, *, order=2):
    """One planning contract used by QNode, Host and Device before simulation."""
    plan.gradient_capability(METHOD)
    limits.check("max_derivative_order", order)
    width, size = plan.circuit.num_qubits, classical.shape[1]
    limits.check("max_input_parameters", size)
    if width < 1:
        raise UnsupportedFeature("mps_tangent_zero_width")
    compiled, output_size, moment_count = _measurements(plan, limits)
    columns = 1 if order == 1 or direction is not None else size
    if order == 2 and direction is None:
        limits.check("max_hessian_elements", output_size*size*columns)
    indices = {parameter: index for index, parameter in enumerate(plan.recorder.parameters)}
    plan._occurrences()
    bond, stored = _bond_bound(plan.node.device, plan.circuit)
    largest, components, operations, function_calls = 2, 1, 0, 0
    sweeps = size*columns
    for column in range(columns):
        other = np.zeros(len(values)) if order == 1 else classical[:, column] if direction is None else classical @ direction
        for row in range(size):
            first = classical[:, row]
            mixed = np.zeros(len(values)) if order == 1 else classical_second[:, row, column]
            counts = (1, 0, 0, 0)
            for operation in plan.circuit.operations:
                if operation.metadata.get("directive") and operation.name in {"barrier", "delay"}:
                    continue
                local = 1 << len(operation.wires)
                current_gate_limits().check("max_matrix_dimension", local)
                largest = max(largest, local)
                left, right, cross = _local_seeds(operation, indices, first, other, mixed)
                a, b = any(left), any(right)
                ab = any(cross) or (a and b)
                p, x, y, xy = counts
                counts = p, x+p*a, y+p*b, xy+y*a+x*b+p*ab
                limits.check("max_mps_tangent_components", max(counts))
                components = max(components, sum(counts))
                routing = _routing_steps(operation.wires, width)
                # Local d-by-d contractions and two-site SWAP decompositions
                # have different dimensions; do not charge a d-by-d SVD for
                # every routed SWAP in a larger controlled gate.
                operations += sum(counts)*(width*bond**3+4*local**2*bond**3+64*routing*bond**3)
                if operation.name == "native_matrix_power" and (a or b or ab):
                    function_calls += 1+int(any(cross))
                limits.check("max_adjoint_operations", operations+2*function_calls*current_gate_limits().max_decomposition_work)
            operations += moment_count*width*bond**3*(counts[0]**2+counts[1]*counts[0]
                +counts[2]*counts[0]+counts[3]*counts[0]+counts[1]*counts[2])*8
    # A shared legacy-named work cap also accounts for these forward sweeps;
    # this field is not a claim that the algorithm is adjoint.
    operations += 2*function_calls*current_gate_limits().max_decomposition_work
    limits.check("max_adjoint_operations", operations)
    derivative_bytes = 8*(4*output_size*(1+size+(size*columns if order == 2 else 0))
                           +classical.size+(0 if classical_second is None else classical_second.size))
    matrix_workspace = 16*128*(4*largest)**2
    external_workspace = 256*(largest*bond)**2
    workspace = stored*(3*components+4)+external_workspace+matrix_workspace+512*output_size
    limits.check("max_memory_bytes", workspace+derivative_bytes)
    device_limit = plan.node.device.state_options["max_memory_mb"]
    if device_limit is not None and workspace+derivative_bytes > device_limit*1024**2:
        raise MemoryError("MPS combined tangent workspace exceeds device max_memory_mb.")
    work = dict(method=METHOD, first_shift_count=0, second_shift_count=0, shift_count=0,
        evaluation_count=1+sweeps, tape_count=1+sweeps, physical_shots=0, sample_elements=0,
        derivative_buffer_bytes=derivative_bytes, combined_working_buffer_bytes=workspace+derivative_bytes,
        adjoint_operations=operations, mps_tangent_operations=operations,
        matrix_function_derivative_work_bound=2*function_calls*current_gate_limits().max_decomposition_work,
        output_size=output_size, linear_size=output_size, direction_count=sweeps,
        input_direction_columns=columns, tangent_component_sum_bound=components,
        max_mps_tangent_components=limits.max_mps_tangent_components,
        mps_bond_bound=bond, mps_external_workspace_bytes=external_workspace,
        local_matrix_dimension=largest, device_path="native_mps_scaled_tangents",
        direction_independent_sampling_schedule=False)
    limits.check("max_total_tapes", work["tape_count"])
    # Only local validation/derivatives are evaluated here, after all expansion
    # and resource checks, never a quantum state or an unbound matrix power.
    mapping = dict(zip(plan.recorder.parameters, values))
    bound = plan.circuit.bind(mapping)
    from quantumbridge.utils.math import validate_unitary_matrix
    for operation in bound.operations:
        if operation.metadata.get("directive") and operation.name in {"barrier", "delay"}:
            continue
        matrix = gate_matrix(operation.name, operation.params, operation.metadata)
        if not validate_unitary_matrix(matrix, 1 << len(operation.wires)):
            raise UnsupportedFeature("mps_tangent_unitary_primal")
    for column in range(columns):
        other = np.zeros(len(values)) if order == 1 else classical[:, column] if direction is None else classical @ direction
        for row in range(size):
            mixed = np.zeros(len(values)) if order == 1 else classical_second[:, row, column]
            for raw, operation in zip(plan.circuit.operations, bound.operations):
                if operation.metadata.get("directive") and operation.name in {"barrier", "delay"}:
                    continue
                seeds = _local_seeds(raw, indices, classical[:, row], other, mixed)
                if any(any(seed) for seed in seeds):
                    _gate_jet(raw, operation, *seeds)
    return work, compiled


def first_mps_work(plan, values, limits=None):
    limits = resolve_higher_order_limits(limits or plan.node.higher_order_limits)
    count = len(values)
    limits.check("max_input_parameters", count)
    limits.check("max_memory_bytes", 8*count*count)
    return mps_work(plan, values, np.eye(count), None, None, limits, order=1)


def _scaled(matrix, factor):
    result = arrays.owned("mps_local_derivative", matrix.shape, complex)
    np.multiply(matrix, factor, out=result)
    return result


def _gate_jet(raw, operation, left, right, mixed):
    a, b, ab = any(left), any(right), any(mixed)
    if raw.name == "native_matrix_power" and (a or b or ab):
        from .matrix_power_jet import matrix_power_directions, matrix_power_directional
        matrix, first, other, second = matrix_power_directions(operation.params, raw.metadata, left, right)
        if ab:
            second += matrix_power_directional(operation.params, raw.metadata, mixed)
        return matrix, first if a else None, other if b else None, second if ab or (a and b) else None
    matrix = gate_matrix(operation.name, operation.params, operation.metadata)
    if not (a or b or ab):
        return matrix, None, None, None
    derivative = gate_derivative(raw, operation.params)
    second = None
    if ab or (a and b):
        second = _scaled(derivative, mixed[0])
        if a and b:
            second += _scaled(gate_second_derivative(raw, operation.params), left[0]*right[0])
    return matrix, _scaled(derivative, left[0]) if a else None, _scaled(derivative, right[0]) if b else None, second


def _moments(jet, terms):
    result = np.zeros(4)
    for coefficient, label in terms:
        result += coefficient*jet.moment(label)
    return result


def _readouts(jet, compiled, output_size):
    result = np.zeros((output_size, 4))
    offset = 0
    for kind, terms, squared in compiled:
        if kind == "probs":
            wires, width = terms, jet.primal.num_qubits
            count = 1 << len(wires)
            part = result[offset:offset+count]
            for mask in range(count):
                label = ["I"]*width
                for bit, wire in enumerate(wires):
                    if mask & (1 << (len(wires)-1-bit)):
                        label[wire] = "Z"
                part[mask] = jet.moment("".join(label))
            stride = 1
            while stride < count:
                for start in range(0, count, 2*stride):
                    first = part[start:start+stride].copy()
                    other = part[start+stride:start+2*stride].copy()
                    part[start:start+stride] = first+other
                    part[start+stride:start+2*stride] = first-other
                stride *= 2
            part /= count
            offset += count
        else:
            moment = _moments(jet, terms)
            if kind == "var":
                square = _moments(jet, squared)
                square -= [moment[0]**2, 2*moment[0]*moment[1], 2*moment[0]*moment[2],
                           2*(moment[1]*moment[2]+moment[0]*moment[3])]
                moment = square
            result[offset] = moment
            offset += 1
    return result


def execute_mps(plan, values, classical, classical_second, direction, work, compiled, *, order=2):
    resources = current_resources()
    precharged = resources is not None and getattr(resources, "derivative_work_precharged", False)
    output_ticket = None
    with ExitStack() as stack:
        if resources is not None:
            ledger = resources.ledger
            if not precharged:
                ledger.preflight(memory_bytes=work["combined_working_buffer_bytes"],
                    tapes=work["tape_count"], derivative_work=work["adjoint_operations"])
                ledger.consume(derivative_work=work["adjoint_operations"], category="native_mps_planned_derivative_work")
                output_ticket = ledger.reserve("mps_retained_derivative_outputs", memory_bytes=work["derivative_buffer_bytes"])
            stack.enter_context(ledger.reserve("mps_tangent_external_workspace", memory_bytes=work["mps_external_workspace_bytes"]))
            stack.enter_context(ledger.reserve("mps_tangent_readout_workspace", memory_bytes=512*work["output_size"]))
            ledger.unavailable("mps_tangent_allocator_workspace",
                "Conservative external workspace allowance, not observed NumPy/LAPACK/Python allocator peak or process RSS.")
        try:
            result = _execute_mps(plan, values, classical, classical_second, direction, work, compiled, order=order)
            if output_ticket is not None:
                from quantumbridge.execution.array_ownership import retain_reservation, native_storage_bytes
                output_ticket.observe(native_storage_bytes(result[:3]))
                retain_reservation(output_ticket, result[:3])
            return result
        except BaseException:
            if output_ticket is not None:
                output_ticket.release()
            raise


def _execute_mps(plan, values, classical, classical_second, direction, work, compiled, *, order):
    base = plan.evaluate(values)
    metadata = copy_execution_payload(plan.metadata)
    flat, spec = flatten_tree(result_payload(base))
    size, columns = classical.shape[1], work["input_direction_columns"]
    matrix = np.zeros((flat.size, size))
    second = None if order == 1 else np.zeros((flat.size, size, columns))
    if flat.size != work["output_size"]:
        raise ValueError("MPS derivative output changed after planning.")
    bound = plan.circuit.bind(dict(zip(plan.recorder.parameters, values)))
    indices = {parameter: index for index, parameter in enumerate(plan.recorder.parameters)}
    resources, receipts = current_resources(), []
    observed = dict(max_bond_observed=1, max_tangent_components=0, max_component_tensor_bytes=0,
                    max_discarded_weight=0., primal_max_difference=0.)
    def observe(jet):
        parts = (jet.primal, jet.left, jet.right, jet.mixed)
        chains = [state for part in parts for state, _ in part._components]
        observed["max_bond_observed"] = max(observed["max_bond_observed"],
            max((chain.peak_bond_dimension for chain in chains), default=1))
        observed["max_tangent_components"] = max(observed["max_tangent_components"], sum(part.num_components for part in parts))
        observed["max_component_tensor_bytes"] = max(observed["max_component_tensor_bytes"], sum(part.tensor_bytes for part in parts))
        observed["max_discarded_weight"] = max(observed["max_discarded_weight"], max((part.discarded_weight for part in parts), default=0.))
    for column in range(columns):
        other = np.zeros(len(values)) if order == 1 else classical[:, column] if direction is None else classical @ direction
        for row in range(size):
            first = classical[:, row]
            mixed = np.zeros(len(values)) if order == 1 else classical_second[:, row, column]
            lease = nullcontext() if resources is None else resources.ledger.reserve("active_mps_tangent_sweep", tapes=1)
            with lease:
                jet = MPSTangentJet.from_state(plan.node.device._initial(plan.circuit.num_qubits),
                    max_components=work["max_mps_tangent_components"])
                observe(jet)
                for raw, operation in zip(plan.circuit.operations, bound.operations):
                    if operation.metadata.get("directive") and operation.name in {"barrier", "delay"}:
                        continue
                    left, right, cross = _local_seeds(raw, indices, first, other, mixed)
                    u, a, b, ab = _gate_jet(raw, operation, left, right, cross)
                    jet = jet.applied(u, operation.wires, left=a, right=b, mixed=ab)
                    observe(jet)
                moments = _readouts(jet, compiled, flat.size)
                difference = float(np.max(np.abs(moments[:, 0]-flat), initial=0))
                observed["primal_max_difference"] = max(observed["primal_max_difference"], difference)
                if not np.allclose(moments[:, 0], flat, atol=8e-12, rtol=0):
                    raise UnsupportedFeature("mps_tangent_primal_truncation", "Ordinary MPS value differs from numerical-rank-preserving derivative primal.")
                if column == 0:
                    matrix[:, row] = moments[:, 1]
                if order == 2:
                    second[:, row, column] = moments[:, 3]
                if resources is not None:
                    resources.ledger.consume(tapes=1, category="native_mps_tangent_sweep")
            receipts.append(dict(evaluation_id=str(uuid4()), kind=METHOD, row=row, column=column, physical_shots=0))
            del jet
    if not np.all(np.isfinite(matrix)) or second is not None and not np.all(np.isfinite(second)):
        raise FloatingPointError("MPS derivatives produced a nonfinite result.")
    plan.metadata = {**metadata, "gradient_method": METHOD, "derivative_order": order,
        "gradient_semantics": "analytic", "gradient_physical_shots": 0, "total_physical_shots": 0,
        "shifted_evaluation_count": 0, "derivative_evaluation_count": len(receipts),
        "gradient_evaluations": receipts, "mps_tangent_work": work, "mps_tangent_resources": observed,
        "memory_budget_scope": "bounded native tangent storage/workspace; not observed process RSS"}
    if order == 2:
        plan.metadata.update(second_order_method="native mixed MPS forward tangents", hvp_streamed=direction is not None,
            first_shifted_evaluation_count=0, second_shifted_evaluation_count=0,
            second_order_evaluations=receipts, second_order_work=work)
    return base, matrix, second, spec


def mps_jacobian(plan, values):
    work, compiled = first_mps_work(plan, values)
    base, matrix, _, spec = execute_mps(plan, values, np.eye(len(values)), None, None, work, compiled, order=1)
    return base, matrix, spec
