# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Exact second parameter-shift rules on the existing native execution plan."""

from quantumbridge.execution.resource_context import copy_execution_payload as deepcopy
from uuid import uuid4

import numpy as np

from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.utils.math import Hamiltonian, PauliString
from .higher_order_limits import resolve_higher_order_limits
from .processing import result_payload
from .trace import flatten_tree


def second_seeds(count, classical, classical_second, direction, limits):
    if classical is None:
        limits.check("max_input_parameters", count)
        limits.check("max_memory_bytes", 8*count*count)
        classical = np.eye(count)
    classical = np.asarray(classical)
    if classical.ndim != 2 or classical.shape[0] != count or classical.dtype.kind not in "fiu" or not np.all(np.isfinite(classical)):
        raise ValueError("Second-order classical Jacobian must be a finite real matrix matching gate occurrences.")
    size = classical.shape[1]
    limits.check("max_input_parameters", size)
    columns = size if direction is None else 1
    if direction is not None:
        direction = np.asarray(direction)
        if direction.shape != (size,) or direction.dtype.kind not in "fiu" or not np.all(np.isfinite(direction)):
            raise ValueError("HVP direction must be a finite real vector matching the differentiated inputs.")
    elements = count*size*columns
    if direction is None:
        limits.check("max_hessian_elements", elements)
    limits.check("max_memory_bytes", 8*(count*size+elements))
    if classical_second is None:
        classical_second = np.zeros((count, size, columns))
    classical_second = np.asarray(classical_second)
    if (classical_second.shape != (count, size, columns) or classical_second.dtype.kind not in "fiu"
            or not np.all(np.isfinite(classical_second))):
        raise ValueError("Second-order classical map must match the derivative direction and input shapes.")
    return classical, classical_second, direction


def second_work(plan, values, classical, classical_second, direction, limits, method=None):
    """Validate all recipes and planned physical work before the first execution."""
    requested = method or plan.node.diff_method
    # Preserve established shift schedules for ordinary gates. General matrix
    # functions need adjoint because a fixed shift recipe does not exist.
    if requested == "best" and any(op.name == "native_matrix_power" for op in plan.circuit.operations):
        requested = plan.gradient_capability("best")
    if requested == "adjoint":
        from .second_adjoint import second_adjoint_work
        return second_adjoint_work(plan, values, classical, classical_second, direction, limits)
    if requested == "mps-tangent":
        from .mps_derivatives import mps_work
        work, compiled = mps_work(plan, values, classical, classical_second, direction, limits)
        return compiled, [], classical if direction is None else (classical @ direction).reshape(len(values), 1), work
    if requested not in {"best", "parameter-shift"}:
        raise UnsupportedFeature("second_order_method", "Native second order supports parameter-shift or analytic statevector adjoint, not finite differences.")
    plan.gradient_capability("parameter-shift")
    limits.check("max_derivative_order", 2)
    fixed_sampling_schedule = direction is not None and (
        any(shots is not None for shots in plan.shots.partitions) or any(request.shots is not None for request in plan.requests))
    output_size, physical_shots, sample_elements = 0, 0, 0
    for request in plan.requests:
        if (request.metadata or {}).get("measurement_records"):
            raise UnsupportedFeature("second_order_measurement_records", "Record-conditioned second derivatives are not yet validated.")
        if request.kind in {"expval", "var"}:
            width = 1
        elif request.kind == "pauli_group":
            width = len(request.observable.words)
        elif request.kind == "probs":
            wires = plan.circuit.num_qubits if request.wires is None else len(request.wires)
            if wires > 16:
                raise UnsupportedFeature("probability_output_size")
            width = 1 << wires
        else:
            raise UnsupportedFeature("second_order_measurement")
        output_size += width*len(plan.shots.partitions)
        for partition in plan.shots.partitions:
            shots = request.shots if request.shots is not None else partition
            if shots is not None:
                if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)) or shots < 1:
                    raise ValueError("Second-order shots must be positive integers.")
                if request.kind == "var":
                    raise UnsupportedFeature("sampled_variance_second_order", "Finite-shot variance Hessians have no verified unbiased estimator contract.")
                if request.seed is not None:
                    raise UnsupportedFeature("local_seed_second_order", "Fixed local measurement seeds cannot provide independent shifted streams.")
                if plan.node.seed is None:
                    raise UnsupportedFeature("unseeded_second_order", "Finite-shot second derivatives require an explicit QNode seed.")
            multiplier = 1
            if request.kind in {"expval", "var"} and isinstance(request.observable, PauliString):
                multiplier = int(any(label != "I" for _, label in request.observable.terms))
            if request.kind == "expval" and isinstance(request.observable, Hamiltonian):
                multiplier = sum(any(label != "I" for _, label in word.terms) for _, word in request.observable.terms)
            physical_shots += (shots or 0)*multiplier
            if request.kind == "pauli_group":
                sample_elements += (shots or 0)*(len(request.observable.basis)+width)+width*width
            elif shots is not None:
                sample_elements += shots*max(1, plan.circuit.num_qubits)
    limits.check("max_output_elements", output_size)
    size = classical.shape[1]
    columns = size if direction is None else 1
    output_elements = output_size*size*columns
    if direction is None:
        limits.check("max_hessian_elements", output_elements)
    linear_size = output_size+sum(request.kind == "var" for request in plan.requests)*len(plan.shots.partitions)
    # Bound derivative arrays and their accumulation temporaries, not device state.
    buffer_bytes = 8*(4*linear_size*(1+size+size*columns)+classical.size+classical_second.size)
    limits.check("max_memory_bytes", buffer_bytes)
    occurrences = plan._occurrences()
    recipes = {}
    for index, parameter in enumerate(plan.recorder.parameters):
        active_dependency = fixed_sampling_schedule and getattr(plan.recorder.angles[index], "depends_on_inputs", False)
        if parameter not in occurrences or not (active_dependency or np.any(classical[index]) or np.any(classical_second[index])):
            continue
        positive = plan.shift_recipe(occurrences[parameter], values[index])
        recipes[index] = tuple((sign*coefficient, sign*shift) for coefficient, shift in positive for sign in (1, -1))
    projections = classical if direction is None else (classical @ direction).reshape(len(values), 1)
    first_count = sum(len(recipe) for recipe in recipes.values())
    limits.check("max_shifted_circuits", first_count)
    pairs, second_count = [], 0
    active = [index for index in recipes if fixed_sampling_schedule or np.any(classical[index])]
    if fixed_sampling_schedule or any(np.any(projections[index]) for index in active):
        for offset, i in enumerate(active):
            for j in active[offset:]:
                if not (fixed_sampling_schedule or np.any(projections[i]) or np.any(projections[j])):
                    continue
                second_count += len(recipes[i])*len(recipes[j])
                limits.check("max_shifted_circuits", first_count+second_count)
                pairs.append((i, j))
    count = 2+first_count+second_count
    work = {"first_shift_count": first_count, "second_shift_count": second_count,
            "shift_count": first_count+second_count, "evaluation_count": count,
            "tape_count": count*len(plan.shots.partitions), "physical_shots": count*physical_shots,
            "sample_elements": count*sample_elements, "derivative_buffer_bytes": buffer_bytes,
            "output_size": output_size, "linear_size": linear_size,
            "direction_independent_sampling_schedule": fixed_sampling_schedule}
    limits.check("max_shifted_circuits", work["shift_count"])
    limits.check("max_total_tapes", work["tape_count"])
    limits.check("max_total_physical_shots", work["physical_shots"])
    limits.check("max_memory_bytes", buffer_bytes+8*sample_elements)
    return recipes, pairs, projections, work


def execution_second_derivatives(plan, values, method=None, *, direction=None, classical=None, classical_second=None, limits=None):
    limits = resolve_higher_order_limits(limits if limits is not None else getattr(plan.node, "higher_order_limits", None))
    values = np.asarray(values)
    if values.shape != (len(plan.recorder.parameters),) or values.dtype.kind not in "fiu" or not np.all(np.isfinite(values)):
        raise ValueError("Second-order gate values must be finite real scalars matching the recorded occurrences.")
    classical, classical_second, direction = second_seeds(len(values), classical, classical_second, direction, limits)
    recipes, pairs, projections, work = second_work(plan, values, classical, classical_second, direction, limits, method)
    if work.get("method") == "adjoint":
        from .second_adjoint import execute_second_adjoint
        return execute_second_adjoint(plan, values, classical, classical_second, direction, work)
    if work.get("method") == "mps-tangent":
        from .mps_derivatives import execute_mps
        return execute_mps(plan, values, classical, classical_second, direction, work, recipes)
    receipts = []
    sequence = np.random.SeedSequence(plan.node.seed, spawn_key=(2,))

    def execute(point, kind, shifts=(), *, linear=True):
        seed = plan.node.seed if kind == "base" else int(sequence.spawn(1)[0].generate_state(1)[0])
        value = plan.evaluate(point, linear=linear, sampling_seed=seed)
        receipts.append({"evaluation_id": str(uuid4()), "kind": kind, "angle_shifts": shifts,
                         "root_seed": seed, **deepcopy(plan.metadata)})
        return value

    base = execute(values, "base", linear=False)
    base_metadata = deepcopy(plan.metadata)
    flat, spec = flatten_tree(result_payload(base))
    linear_base, corrections = execute(values, "linear_moments")
    if flat.size != work["output_size"] or sum(len(part) for part in linear_base) != work["linear_size"]:
        raise ValueError("Second-order execution output shape changed after resource preflight.")
    size, columns = classical.shape[1], projections.shape[1]
    jacobian = np.zeros((work["linear_size"], size))
    second = np.zeros((work["linear_size"], size, columns))
    for index, recipe in recipes.items():
        first = np.zeros(work["linear_size"])
        for coefficient, shift in recipe:
            point = np.array(values, dtype=float, copy=True)
            point[index] += shift
            parts, _ = execute(point, "first_shift", ((index, float(shift)),))
            first += coefficient*np.concatenate(parts)
        jacobian += first[:, None]*classical[index]
        second += first[:, None, None]*classical_second[index]
    for i, j in pairs:
        mixed = np.zeros(work["linear_size"])
        for ci, si in recipes[i]:
            for cj, sj in recipes[j]:
                point = np.array(values, dtype=float, copy=True)
                point[i] += si
                point[j] += sj
                parts, _ = execute(point, "second_shift", ((i, float(si)), (j, float(sj))))
                mixed += ci*cj*np.concatenate(parts)
        seed = np.outer(classical[i], projections[j])
        if i != j:
            seed += np.outer(classical[j], projections[i])
        second += mixed[:, None, None]*seed
    matrices, tensors, offset = [], [], 0
    for length, spans, _ in corrections:
        matrix, tensor = jacobian[offset:offset+length].copy(), second[offset:offset+length].copy()
        for row, mean_index, mean in spans:
            index = offset+length+mean_index
            gradient = jacobian[index]
            projected = gradient if direction is None else np.asarray([gradient @ direction])
            matrix[row] -= 2*mean*gradient
            tensor[row] -= 2*(mean*second[index]+np.outer(gradient, projected))
        matrices.append(matrix)
        tensors.append(tensor)
        offset += length+len(spans)
    jacobian, second = np.concatenate(matrices), np.concatenate(tensors)
    physical = sum(part["physical_shots"] for receipt in receipts for part in receipt.get("partitions", ()))
    if physical != work["physical_shots"] or len(receipts) != work["evaluation_count"]:
        raise RuntimeError("Actual second-order work differs from the validated execution receipt.")
    plan.metadata = {**base_metadata, "derivative_order": 2, "gradient_method": "parameter-shift",
                     "second_order_method": "convolved exact parameter-shift", "hvp_streamed": direction is not None,
                     "first_shifted_evaluation_count": work["first_shift_count"],
                     "second_shifted_evaluation_count": work["second_shift_count"],
                     "shifted_evaluation_count": work["shift_count"], "derivative_evaluation_count": work["shift_count"],
                     "second_order_evaluations": receipts, "second_order_work": work,
                     "total_physical_shots": physical, "gradient_physical_shots": physical,
                     "gradient_semantics": "finite-shot estimate" if physical else "analytic",
                     "memory_budget_scope": "classical trace and derivative buffers; native device state limits apply separately"}
    if plan.resource_history:
        plan.metadata["gradient_resource_evaluations"] = list(plan.resource_history)
    return base, jacobian, second, spec


def transformed_second_work(plan, values, classical, classical_second, direction, limits, method=None):
    """Reusable preflight; no child executes while collecting physical work."""
    from quantumbridge.transforms.program import check_batch
    plan._processing_values(values)
    finite = any(shots is not None for child in plan.child_plans for shots in child.shots.partitions) or any(
        request.shots is not None for child in plan.child_plans for request in child.requests)
    for stage, level in enumerate(plan.prepared.processing_levels):
        for parent, (_, _, processor) in enumerate(level):
            if finite and not getattr(processor, "native_linear_quantum_outputs", False):
                raise UnsupportedFeature("sampled_postprocessing_second_order", "Finite-shot second derivatives require verified linear quantum-output postprocessing.")
            start, stop = plan.processing_parameter_slices[(stage, parent)]
            if (finite and getattr(processor, "native_shot_distribution", None) == "weighted"
                    and (np.any(classical[start:stop]) or np.any(classical_second[start:stop]))):
                raise UnsupportedFeature("parameter_dependent_shot_allocation", "Second derivatives through coefficient-dependent integer shot allocation are not implemented.")
    works = []
    for child, (start, stop) in zip(plan.child_plans, plan.parameter_slices):
        works.append(second_work(child, values[start:stop], classical[start:stop], classical_second[start:stop], direction, limits, method)[3])
    totals = {key: sum(work[key] for work in works) for key in
              ("shift_count", "first_shift_count", "second_shift_count", "evaluation_count", "tape_count", "physical_shots", "sample_elements", "derivative_buffer_bytes")}
    totals["adjoint_operations"] = sum(work.get("adjoint_operations", 0) for work in works)
    limits.check("max_shifted_circuits", totals["shift_count"])
    limits.check("max_total_tapes", totals["tape_count"])
    limits.check("max_total_physical_shots", totals["physical_shots"]+plan.forward_physical_shots)
    limits.check("max_memory_bytes", totals["derivative_buffer_bytes"]+8*classical.size+8*classical_second.size)
    limits.check("max_adjoint_operations", totals["adjoint_operations"])
    if any(work.get("method") in {"adjoint", "mps-tangent"} for work in works):
        limits.check("max_memory_bytes", max(work.get("combined_working_buffer_bytes", 0) for work in works)
                     +totals["derivative_buffer_bytes"]+8*(classical.size+classical_second.size))
    plan.limits.check("max_total_shots", totals["physical_shots"]+plan.forward_physical_shots)
    plan.limits.check("max_sample_values", totals["sample_elements"])
    check_batch(plan.prepared.tapes, plan.limits)
    return totals, finite


def transformed_second_derivatives(plan, values, method=None, *, direction=None, classical=None, classical_second=None, limits=None):
    from .processing import processing_second
    from .second_trace import SecondContext
    from quantumbridge.transforms.program import operation_count
    limits = resolve_higher_order_limits(limits if limits is not None else getattr(plan.node, "higher_order_limits", None))
    values = np.asarray(values)
    if values.shape != (len(plan.recorder.parameters),) or values.dtype.kind not in "fiu" or not np.all(np.isfinite(values)):
        raise ValueError("Transformed second-order values must match the recorded scalar slots.")
    classical, classical_second, direction = second_seeds(len(values), classical, classical_second, direction, limits)
    parameters = plan._processing_values(values)
    totals, finite = transformed_second_work(plan, values, classical, classical_second, direction, limits, method)
    context = SecondContext(classical.shape[1], direction, limits, node_budget="max_postprocessing_nodes")
    results, derivatives, executions = [], [], []
    for index, (child, (start, stop)) in enumerate(zip(plan.child_plans, plan.parameter_slices)):
        base, matrix, second, _ = child.second_derivatives(values[start:stop], method, direction=direction,
            classical=classical[start:stop], classical_second=classical_second[start:stop], limits=limits)
        results.append(base)
        derivatives.append((matrix, second))
        executions.append({"tape_index": index, "execution_id": str(uuid4()), "gate_angle_slice": (start, stop),
                           "operation_count": operation_count(child.circuit, plan.limits), **deepcopy(child.metadata)})
    value = plan._reconstruct(values, results, executions)
    parameter_derivatives = {key: (classical[start:stop], classical_second[start:stop])
                             for key, (start, stop) in plan.processing_parameter_slices.items()}
    raw, matrix, second, spec = processing_second(plan.prepared, tuple(results), derivatives, parameters, parameter_derivatives, context)
    flat, actual_spec = flatten_tree(value)
    if actual_spec != spec or not np.allclose(flat, raw, atol=1e-12, rtol=0):
        raise UnsupportedFeature("postprocessing_second_trace", "Second-order tracing changed the actual forward result or structure.")
    physical = sum(child.metadata["total_physical_shots"] for child in plan.child_plans)
    if physical != totals["physical_shots"]:
        raise RuntimeError("Actual transformed second-order physical work differs from preflight.")
    adjoint = bool(executions) and all(child["gradient_method"] == "adjoint" for child in executions)
    mps = bool(executions) and all(child["gradient_method"] == "mps-tangent" for child in executions)
    mixed = bool(executions) and len({child["gradient_method"] for child in executions}) > 1
    plan.metadata.update(derivative_order=2, gradient_method="classical-only" if not executions else "adjoint" if adjoint else "parameter-shift",
        second_order_method="native classical second chain rule" if not executions else "native tangent-over-reverse adjoint" if adjoint else "convolved exact parameter-shift", hvp_streamed=direction is not None,
        postprocessing_second_chain_rule=True, native_postprocessing_derivative=True,
        first_shifted_evaluation_count=totals["first_shift_count"], second_shifted_evaluation_count=totals["second_shift_count"],
        shifted_evaluation_count=totals["shift_count"],
        derivative_evaluation_count=sum(child.get("derivative_evaluation_count", 0) for child in executions),
        second_order_executions=deepcopy(executions), second_order_work=totals,
        gradient_physical_shots=physical, preceding_forward_physical_shots=plan.forward_physical_shots,
        total_physical_shots=physical+plan.forward_physical_shots,
        gradient_semantics="finite-shot estimate" if finite else "analytic",
        postprocessing_derivative_nodes=context.nodes,
        adjoint_second_sweeps=sum(child.get("adjoint_second_sweeps", 0) for child in executions),
        differentiation_path="actual child second derivatives -> native processor second chain rule -> source classical graph",
        memory_budget_scope="combined second-adjoint child working buffers plus parent derivative arrays" if adjoint else "classical trace and derivative buffers; native device state limits apply separately")
    if mps or mixed:
        plan.metadata.update(gradient_method="mps-tangent" if mps else "mixed-native",
            second_order_method="native mixed MPS forward tangents" if mps else "mixed native child derivative methods")
    return value, matrix, second, spec
