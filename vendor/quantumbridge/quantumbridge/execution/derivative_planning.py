# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Prepare existing native plans from reusable Device derivative requests."""

from copy import copy
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from quantumbridge.core import Circuit
from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.diff.higher_order_limits import resolve_higher_order_limits
from quantumbridge.diff.qnode_engine import AngleRecorder, ExecutionPlan
from quantumbridge.diff.trace import flatten_tree, trace_arguments
from quantumbridge.qml.qnode import QNode
from quantumbridge.qml.tape import Tape
from quantumbridge.utils.math import Hamiltonian, PauliString

from .derivative_request import DerivativeExecutionRequest


@dataclass
class PreparedDerivative:
    request: DerivativeExecutionRequest
    node: QNode
    plan: object
    point: np.ndarray
    classical: np.ndarray
    second: object
    direction: object
    size: int
    layout: object
    single: bool
    output_spec: object
    output_size: int
    work: dict
    buffer_bytes: int
    device_bytes: int


def _direction(request, args, size):
    if request.derivative_kind not in {"hvp", "jvp"}:
        return None
    raw = request.tangent
    selected = (raw,) if len(args) == 1 else tuple(raw) if isinstance(raw, (tuple, list)) else ()
    if len(selected) != len(args) or any(np.shape(x) != np.shape(y) for x, y in zip(selected, args)):
        raise ValueError("Derivative tangent must match each differentiated input shape.")
    arrays = [np.asarray(value) for value in selected]
    if any(value.dtype.kind not in "fiu" or not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("Derivative tangent must be finite and real.")
    return flatten_tree(tuple(arrays))[0]


def device_working_bytes(device, circuit, *, mps_external_only=False):
    """Conservative numerical buffers, not observed BLAS allocator memory."""
    width = circuit.num_qubits
    if getattr(device, "simulation", None) == "matrix_product_state":
        from .resource_context import current_resources
        resources = current_resources()
        if resources is not None:
            resources.ledger.unavailable("mps_svd_allocator_peak",
                "Conservative tensor/contraction/SVD reservation, not observed BLAS/LAPACK allocator peak.")
        bonds = [1]*(max(0, width-1))
        initial = getattr(device, "initial_state", None)
        if initial is not None:
            from quantumbridge.information.matrix_product_state import MatrixProductState
            if not isinstance(initial, MatrixProductState):
                raise UnsupportedFeature("derivative_mps_dense_initial_state")
            # tensors returns owned copies; an estimate must only inspect shapes.
            bonds = list(initial.bond_dimensions)
        cap = device.state_options["max_bond_dim"]
        largest = max(bonds, default=1)
        for operation in circuit.operations:
            if operation.metadata.get("directive") and operation.name in {"barrier", "delay"}:
                continue
            wires = operation.controls+operation.targets
            if len(wires) > 1:
                for cut in range(min(wires), max(wires)):
                    rank = 2 if operation.name in {"cx", "cz"} else 4**min(sum(w <= cut for w in wires), sum(w > cut for w in wires))
                    bonds[cut] = min(bonds[cut]*rank, 1 << min(cut+1, width-cut-1))
                    if cap is not None:
                        bonds[cut] = min(bonds[cut], cap)
                    largest = max(largest, bonds[cut])
        dimensions = [1, *bonds, 1]
        stored = 16*sum(2*a*b for a, b in zip(dimensions, dimensions[1:]))
        # Full estimates remain admission guards. During actual MPS execution,
        # native tensor/output roots have individual leases; keep only the
        # existing conservative external-workspace allowance here.
        return int((0 if mps_external_only else 12*stored)+128*(2*largest)**2)
    if getattr(device, "simulation", None) == "exact_stabilizer":
        raise UnsupportedFeature("tableau_continuous_gradient")
    from quantumbridge.devices.density_matrix_device import DensityMatrixDevice
    local_dimension = max((1 << len(op.wires) for op in circuit.operations if not op.metadata.get("directive")), default=1)
    return 16*(1 << (2*width if isinstance(device, DensityMatrixDevice) else width))*8+64*local_dimension**2


def _measurement_size(child):
    size, shots, samples = 0, 0, 0
    for request in child.requests:
        if request.kind in {"expval", "var"}:
            count = 1
        elif request.kind == "pauli_group":
            count = len(request.observable.words)
        elif request.kind == "probs":
            width = child.circuit.num_qubits if request.wires is None else len(request.wires)
            if width > 16: raise UnsupportedFeature("probability_output_size")
            count = 1 << width
        else:
            raise UnsupportedFeature("derivative_measurement")
        size += count*len(child.shots.partitions)
        for partition in child.shots.partitions:
            n = request.shots if request.shots is not None else partition
            multiplier = 1
            if request.kind in {"expval", "var"} and isinstance(request.observable, PauliString):
                multiplier = int(any(label != "I" for _, label in request.observable.terms))
            elif request.kind == "expval" and isinstance(request.observable, Hamiltonian):
                multiplier = sum(any(label != "I" for _, label in word.terms) for _, word in request.observable.terms)
            shots += (n or 0)*multiplier
            samples = max(samples, 16*(n or 0)*(child.circuit.num_qubits+count)+8*count*count)
    return size, shots, samples


def tensorflow_statevector_work(plan, circuit, inputs, order=2, *, request=None, directions=1, cotangents=1, allow_parameter_shift=False):
    """Structural admission for the bounded TF tensor execution strategy.

    This is a logical workspace/work envelope, not observed TF allocator
    memory or a graph-wide cumulative request ledger.
    """
    from quantumbridge.devices.statevector_device import StatevectorDevice
    from quantumbridge.core.gate_limits import current_gate_limits
    if request is not None:
        if not isinstance(request, DerivativeExecutionRequest):
            raise TypeError("Tensor execution requires the shared DerivativeExecutionRequest.")
        order = request.validate_order()
        if request.method not in {"best", "adjoint"}:
            raise UnsupportedFeature("tensorflow_tensor_method")
        if order > plan.node.max_diff:
            raise UnsupportedFeature("higher_order_derivatives", "The compiled QNode has not approved this derivative order.")
    if type(plan) is not ExecutionPlan or type(plan.node.device) is not StatevectorDevice:
        raise UnsupportedFeature("tensorflow_tensor_device")
    device, width = plan.node.device, circuit.num_qubits
    if not 1 <= width <= 4:
        raise UnsupportedFeature("tensorflow_dense_tensor_width", "Dense TF tensor execution is bounded to one through four qubits; MPS requires a separate admitted contraction.")
    if circuit.has_dynamic_operations or circuit.measurements or getattr(device, "noise_model", None) is not None:
        raise UnsupportedFeature("tensorflow_tensor_dynamic_or_noise")
    if plan.shots.partitions != (None,) or any(request.shots is not None for request in plan.requests):
        raise UnsupportedFeature("tensorflow_tensor_sampling")
    if plan.node.diff_method not in ({"best", "adjoint", "parameter-shift"} if allow_parameter_shift else {"best", "adjoint"}):
        raise UnsupportedFeature("tensorflow_tensor_method")
    if device.max_qubits is not None and width > device.max_qubits:
        raise ValueError("TF tensor execution exceeds this device qubit limit.")
    if any(request.kind != "expval" or request.metadata for request in plan.requests):
        raise UnsupportedFeature("tensorflow_tensor_measurement")
    outputs, shots, _ = _measurement_size(plan)
    limits = resolve_higher_order_limits(plan.node.higher_order_limits)
    limits.check("max_derivative_order", order)
    limits.check("max_input_parameters", inputs)
    limits.check("max_output_elements", outputs)
    # Encoded native syntax bytes conservatively bound source scalar AST
    # nodes. Framework caller/loss graphs are not native parameter graphs.
    source_json = circuit.to_ir().to_json()
    source_bytes = len(source_json.encode("utf-8"))
    limits.check("max_parameter_graph_nodes", source_bytes)
    terms = 0
    for measurement in plan.requests:
        observable = measurement.observable
        if not isinstance(observable, (PauliString, Hamiltonian)):
            raise UnsupportedFeature("tensorflow_tensor_observable")
        entries = observable.terms if isinstance(observable, Hamiltonian) else ((1., observable),)
        for coefficient, word in entries:
            if not isinstance(word, PauliString) or not isinstance(coefficient, (int, float, np.floating)) or not np.isfinite(coefficient):
                raise UnsupportedFeature("tensorflow_tensor_observable_coefficient")
            if any(type(wire) is not int or not 0 <= wire < width for wire, _ in word.terms):
                raise ValueError("TF observable wires exceed the compiled circuit.")
            terms += 1
    limits.check("max_postprocessing_nodes", 1+sum(8+6*len(word.terms)
        for request in plan.requests for _, word in (request.observable.terms
            if isinstance(request.observable, Hamiltonian) else ((1., request.observable),))))
    dimension = 1 << width
    local = max((1 << len(operation.wires) for operation in circuit.operations if not operation.metadata.get("directive")), default=1)
    current_gate_limits().check("max_matrix_dimension", local)
    operations = len(circuit.operations)
    execution_limits = getattr(device, "execution_limits", None)
    if execution_limits is not None:
        from quantumbridge.core.control_flow import ExecutionLimits
        if not isinstance(execution_limits, ExecutionLimits):
            raise TypeError("Native device execution_limits requires ExecutionLimits.")
        if operations > execution_limits.max_instruction_executions:
            raise MemoryError("TF tensor plan exceeds max_instruction_executions.")
    logical_evaluations = 6 if order == 2 else 2 if order == 1 else 1
    if request is None:
        graph_copies = 8+16*max(1, inputs)*max(1, outputs) if order == 2 else 6 if order == 1 else 1
    else:
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 1 for value in (directions, cotangents)):
            raise ValueError("Tensor derivative direction/cotangent counts must be positive integers.")
        kind = request.derivative_kind
        if kind == "gradient" and outputs != 1:
            raise ValueError("A gradient request requires exactly one scalar output.")
        count = 1 if kind == "value" else directions if kind == "jvp" else cotangents if kind in {"gradient", "vjp"} else outputs if kind == "jacobian" else directions*cotangents if kind == "hvp" else inputs*outputs
        logical_evaluations = int(count)
        graph_copies = count*(1 if order == 0 else 6 if order == 1 else 24)
        if kind == "hessian":
            limits.check("max_hessian_elements", outputs*inputs*inputs)
        limits.check("max_total_tapes", count)
    power_workspace = 16*128*(4*local)**2 if any(op.name == "native_matrix_power" for op in circuit.operations) else 0
    result_elements = outputs if request is None or request.derivative_kind == "value" else outputs*inputs*inputs if request.derivative_kind == "hessian" else outputs*inputs if request.derivative_kind == "jacobian" else directions*outputs if request.derivative_kind == "jvp" else directions*cotangents*inputs if request.derivative_kind == "hvp" else cotangents*inputs
    workspace = device_working_bytes(device, circuit)+power_workspace+16*dimension*(1+operations+terms)*graph_copies+8*result_elements
    limits.check("max_memory_bytes", workspace)
    if device.max_memory_mb is not None and workspace > device.max_memory_mb*1024**2:
        raise MemoryError("TF tensor combined workspace exceeds device max_memory_mb.")
    work = max(1, operations+terms)*dimension*local*graph_copies
    if power_workspace:
        # Closed local TF formulas do not execute the generic Schur engine's
        # whole work cap. Bound their decoded expression graph structurally.
        work += graph_copies*sum(16*len(op.metadata["matrix_power_source"]["program"])
            for op in circuit.operations if op.name == "native_matrix_power")
    limits.check("max_adjoint_operations", work)
    limits.check("max_total_tapes", logical_evaluations)
    from .resource_context import current_resources
    resources = current_resources()
    if resources is not None:
        resources.ledger.preflight(memory_bytes=workspace, derivative_work=work, tapes=logical_evaluations, shots=shots)
    from dataclasses import asdict
    import hashlib
    import json
    identity = {"circuit": json.loads(source_json), "method": plan.node.diff_method,
        "max_diff": plan.node.max_diff, "order": order, "inputs": inputs,
        "higher_order_limits": {name: int(value) for name, value in asdict(limits).items()},
        "device_max_qubits": device.max_qubits.item() if isinstance(device.max_qubits, np.generic) else device.max_qubits,
        "device_max_memory_mb": None if device.max_memory_mb is None else float(device.max_memory_mb),
        "observables": [[(float(coefficient), word.terms) for coefficient, word in
            (request.observable.terms if isinstance(request.observable, Hamiltonian)
             else ((1., request.observable),))] for request in plan.requests]}
    identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    contract = None if request is None else dict(request.describe(inputs=inputs, outputs=outputs,
        directions=directions if request.derivative_kind in {"jvp", "hvp"} else inputs if request.derivative_kind == "hessian" else 0,
        cotangents=cotangents if request.derivative_kind in {"gradient", "vjp", "hvp"} else outputs if request.derivative_kind in {"jacobian", "hessian"} else 0,
        parameters=(parameter.uuid for parameter in plan.recorder.parameters), plan_identity=identity))
    return {"workspace_bytes": workspace, "planned_work": work, "maximum_quantum_graph_copies": graph_copies,
        "width": width, "inputs": inputs, "outputs": outputs, "order": order, "physical_shots": 0,
        "compiled_plan_identity": identity,
        "logical_evaluations": logical_evaluations, "construction_work": max(1, work//2+source_bytes),
        "request": contract, "planning_mode": "conservative_max_order_envelope" if request is None else "explicit_derivative_request",
        "cache_policy": "fresh managed QNode plan per call; independent ConcreteFunctions retain their frozen static plan",
        "host_quantum_calls": 0, "device_updates": 0,
        "direction_count": 1, "derivative_execution": "streamed single-direction rule; repeated directions admit separately",
        "parameter_graph_node_bound": source_bytes,
        "budget_semantics": "conservative structural tensor/AD envelope; not observed allocator memory or runtime graph accounting"}


def tensorflow_mps_work(plan, circuit, inputs, order=2, *, request=None,
        directions=1, cotangents=1, allow_parameter_shift=False):
    """Admit exact index-split tensor chains, without SVD or dense state space.

    Bond dimensions here bound the actual algebraic representation, not its
    possibly smaller Schmidt rank. No numerical rank decision or truncation
    can enter a differentiated graph.
    """
    from quantumbridge.devices.matrix_product_state_device import MatrixProductStateDevice
    from quantumbridge.core.gate_limits import current_gate_limits
    from quantumbridge.information.matrix_product_state import mps_routing
    from quantumbridge.diff.mps_derivatives import _terms
    if type(plan) is not ExecutionPlan or type(plan.node.device) is not MatrixProductStateDevice:
        raise UnsupportedFeature("tensorflow_mps_device")
    device, width = plan.node.device, circuit.num_qubits
    if width < 1 or device.initial_state is not None or device.noise_model is not None:
        raise UnsupportedFeature("tensorflow_mps_initial_or_noise")
    if circuit.has_dynamic_operations or circuit.measurements or plan.shots.partitions != (None,):
        raise UnsupportedFeature("tensorflow_mps_dynamic_or_sampling")
    if device.state_options["cutoff"] != 0:
        raise UnsupportedFeature("tensorflow_mps_exact_cutoff", "TF MPS requires explicit cutoff=0; truncation is not differentiated.")
    if device.max_qubits is not None and width > device.max_qubits:
        raise ValueError("TF MPS exceeds device max_qubits.")
    if plan.node.diff_method not in {"best", "mps-tangent"}:
        raise UnsupportedFeature("tensorflow_mps_method")
    if any(r.kind != "expval" or r.metadata or r.shots is not None for r in plan.requests):
        raise UnsupportedFeature("tensorflow_mps_measurement")
    if request is not None:
        if not isinstance(request, DerivativeExecutionRequest):
            raise TypeError("TF MPS requires the shared derivative request.")
        order = request.validate_order()
        if request.method not in {"best", "mps-tangent"}:
            raise UnsupportedFeature("tensorflow_mps_method")
    if order > plan.node.max_diff:
        raise UnsupportedFeature("higher_order_derivatives", "TF MPS request exceeds max_diff.")
    limits = resolve_higher_order_limits(plan.node.higher_order_limits)
    limits.check("max_derivative_order", order)
    limits.check("max_input_parameters", inputs)
    outputs = len(plan.requests)
    limits.check("max_output_elements", outputs)
    terms = sum(len(_terms(r.observable, width)) for r in plan.requests)
    limits.check("max_postprocessing_nodes", 1+terms*width)
    source = circuit.to_ir().to_json()
    source_bytes = len(source.encode("utf-8"))
    limits.check("max_parameter_graph_nodes", source_bytes)
    # Conservative local jet slots: primal, two first directions and mixed.
    # This is not an observed coherent-MPS-sum count or TF allocator reading.
    components = 1 if order == 0 else 2 if order == 1 else 4
    limits.check("max_mps_tangent_components", components)
    bonds, peak, local_workspace, steps, gates, arithmetic = [1]*(width+1), 1, 0, 0, 0, 0
    evolution_bytes, operator_bytes, power_graph_bytes = 0, 0, 0
    cap = device.state_options["max_bond_dim"]
    def split(site):
        nonlocal peak, local_workspace, steps, arithmetic, evolution_bytes
        left, right = bonds[site], bonds[site+2]
        rank = min(2*left, 2*right)
        if cap is not None and rank > cap:
            raise MemoryError("TF MPS exact index-split bond bound exceeds max_bond_dim; no truncation is permitted.")
        peak = max(peak, rank)
        local_workspace = max(local_workspace, 16*(16+12*left*right+rank*rank))
        evolution_bytes += 16*(16+8*left*right+2*left*rank+2*rank*right)
        arithmetic += 64*left*bonds[site+1]*right+64*left*right
        bonds[site+1] = rank
        steps += 1
    for operation in circuit.operations:
        if operation.metadata.get("directive") and operation.name in {"barrier", "delay"}:
            continue
        gates += 1
        if len(operation.wires) > 2:
            raise UnsupportedFeature("tensorflow_mps_local_width", "Exact TF MPS currently admits one/two-site local operators only.")
        local = 1 << len(operation.wires)
        current_gate_limits().check("max_matrix_dimension", local)
        operator_bytes += 16*local*local
        local_workspace = max(local_workspace, 16*8*local*local)
        if len(operation.wires) == 2:
            start, swaps = mps_routing(operation.wires, width)
            for site in (*swaps, start, *reversed(swaps)):
                split(site)
        elif operation.wires:
            site = operation.wires[0]
            arithmetic += 8*bonds[site]*bonds[site+1]
            evolution_bytes += 32*bonds[site]*bonds[site+1]
        else:
            evolution_bytes += 16
        if operation.name == "native_matrix_power":
            local_workspace = max(local_workspace, 16*128*(4*local)**2)
            power_graph_bytes += 16*128*(4*local)**2
            arithmetic += 16*len(operation.metadata["matrix_power_source"]["program"])
    if device.execution_limits is not None:
        from quantumbridge.core.control_flow import ExecutionLimits
        if not isinstance(device.execution_limits, ExecutionLimits):
            raise TypeError("Native MPS execution_limits requires ExecutionLimits.")
        if gates > device.execution_limits.max_instruction_executions:
            raise MemoryError("TF MPS exceeds max_instruction_executions.")
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in (directions, cotangents)):
        raise ValueError("TF MPS direction/cotangent counts require positive integers.")
    directions, cotangents = int(directions), int(cotangents)
    kind = None if request is None else request.derivative_kind
    if kind == "gradient" and outputs != 1:
        raise ValueError("A gradient requires one output.")
    if kind == "hessian":
        limits.check("max_hessian_elements", outputs*inputs*inputs)
    count = 1 if kind in {None, "value"} else directions if kind == "jvp" else cotangents if kind in {"gradient", "vjp"} else outputs if kind == "jacobian" else directions*cotangents if kind == "hvp" else inputs*outputs
    copies = (1 if order == 0 else 6 if order == 1 else 24)*max(1, count)
    # Keep the maximum bond on every site as a conservative retained/AD bound.
    stored = 32*width*peak*peak
    contraction_bytes = terms*(64*width+16*sum(b*b for b in bonds[1:]))
    result_elements = outputs*inputs*inputs if kind == "hessian" else outputs*inputs if kind == "jacobian" else max(inputs, outputs)*max(1, count)
    workspace = copies*(4*stored+local_workspace+64*peak**4+evolution_bytes
        +operator_bytes+power_graph_bytes+contraction_bytes)+8*result_elements
    work = copies*max(1, arithmetic+8*terms*width*peak**3)
    limits.check("max_memory_bytes", workspace)
    limits.check("max_adjoint_operations", work)
    limits.check("max_total_tapes", max(1, count))
    memory_cap = device.state_options["max_memory_mb"]
    if memory_cap is not None and workspace > memory_cap*1024**2:
        raise MemoryError("TF MPS exact tensor workspace exceeds device max_memory_mb.")
    from .resource_context import current_resources
    active = current_resources()
    if active is not None:
        active.ledger.preflight(memory_bytes=workspace, derivative_work=work, tapes=max(1, count))
    import hashlib
    import json
    from dataclasses import asdict
    identity = hashlib.sha256(json.dumps(dict(circuit=json.loads(source),
        state_options=device.state_options, max_qubits=device.max_qubits,
        higher_order_limits={name: int(value) for name, value in asdict(limits).items()}, method=plan.node.diff_method,
        execution_limits=None if device.execution_limits is None else asdict(device.execution_limits),
        max_diff=int(plan.node.max_diff), inputs=inputs,
        observables=[_terms(r.observable, width) for r in plan.requests]), sort_keys=True).encode()).hexdigest()
    contract = None if request is None else dict(request.describe(inputs=inputs, outputs=outputs,
        directions=directions if kind in {"jvp", "hvp"} else inputs if kind == "hessian" else 0,
        cotangents=cotangents if kind in {"gradient", "vjp", "hvp"} else outputs if kind in {"jacobian", "hessian"} else 0,
        parameters=(p.uuid for p in plan.recorder.parameters), plan_identity=identity))
    return dict(workspace_bytes=workspace, planned_work=work, construction_work=work//2+source_bytes,
        maximum_quantum_graph_copies=copies, width=width, physical_tensor_count=width,
        inputs=inputs, outputs=outputs, order=order, physical_shots=0,
        logical_evaluations=max(1, count), compiled_plan_identity=identity, request=contract,
        parameter_graph_node_bound=source_bytes, mps_bond_bound=peak,
        mps_routed_two_site_steps=steps, active_gates=gates, tensor_chain_bytes_bound=stored,
        tensor_evolution_bytes_bound=evolution_bytes, operator_storage_bytes_bound=operator_bytes,
        power_graph_storage_bytes_bound=power_graph_bytes, observable_contraction_bytes_bound=contraction_bytes,
        ad_tangent_component_bound=components, local_workspace_bytes=local_workspace,
        device_path="native_mps_exact_index_split", host_quantum_calls=0,
        truncation=False, normalization="unscaled exact unitary chain with global complex scale",
        budget_semantics="structural tensor/AD envelope; allocator/RSS estimated, external graph executions out of scope")


def tensorflow_execution_capability(plan, *, execution_mode, maximum_order):
    """Read-only contracts for already admitted native TF plan snapshots.

    This uses the native method selector, not ecosystem inventory levels or
    another global registry. A configured envelope is not verification of AD.
    """
    from types import MappingProxyType
    if type(plan) is not ExecutionPlan:
        raise TypeError("Tensor contracts require a native ExecutionPlan.")
    if isinstance(maximum_order, bool) or not isinstance(maximum_order, Integral) or not 0 <= maximum_order <= min(2, plan.node.max_diff):
        raise ValueError("Tensor contract order must match an approved native plan.")
    from quantumbridge.devices.matrix_product_state_device import MatrixProductStateDevice
    mps = type(plan.node.device) is MatrixProductStateDevice
    methods = ("best", "mps-tangent") if mps else ("best", "adjoint")
    if plan.node.diff_method not in methods:
        raise UnsupportedFeature("tensorflow_tensor_method")
    selected = plan.gradient_capability(plan.node.diff_method)
    query = execution_mode == "MANAGED_EXPLICIT_QUERY"
    inline = execution_mode == "TRACE_INLINE_DIFFERENTIABLE"
    experimental = execution_mode == "EXPERIMENTAL_CACHED_CALLABLE"
    if not (query or inline or experimental):
        raise ValueError("Unknown native TensorFlow execution contract.")
    verified = min(int(maximum_order), 1) if experimental else int(maximum_order)
    operations = ["value"]
    if verified >= 1:
        operations.extend(("jacobian", "jvp", "vjp"))
        if len(plan.requests) == 1:
            operations.append("gradient")
    if verified == 2:
        operations.extend(("hvp", "hessian"))
    return MappingProxyType(dict(execution_mode=execution_mode,
        supported_derivative_methods=methods, native_selected_method=selected,
        maximum_configured_quantum_derivative_order=int(maximum_order),
        maximum_verified_quantum_derivative_order=verified,
        supported_operations=tuple(operations), eligible_for_verified_second_order=verified == 2,
        automatic_qnode_selection=False,
        external_mixed_ad_contract="UNSUPPORTED_QUERY_AUTODIFF" if query else "VERIFIED_EAGER_AND_CO_TRACED_LOW_QUBIT" if inline else "UNGUARANTEED_EXTERNAL_AD",
        resource_accounting_scope="STATIC_BOUND_AND_RUNTIME_ASSERTIONS_ONLY" if inline else "CONTROLLED_SYNCHRONOUS_ENTRY",
        graph_cache_contract="EXTERNAL_CACHED_GRAPH_OUT_OF_SCOPE" if inline else "CONTROLLED_CACHED_QUERY_INVOCATIONS" if query else "EXPERIMENTAL_CACHED_CALLABLE",
        mathematical_ownership="SNAPSHOT_MATH", owner_guard="LIVE_GUARD_AT_CONTROLLED_ENTRY",
        numeric_guard="RUNTIME_TF_ASSERTIONS", allocator_memory="ESTIMATED_ONLY",
        implicit_external_ad_nesting="UNGUARANTEED_EXTERNAL_AD",
        unsupported_reason="Cached callable nonlinear RoF remains unresolved; configured order two is diagnostic only." if experimental else "External cached-graph AD and per-execution Python guards are not guaranteed." if inline else "Explicit derivative queries are not composable AD outputs."))


def require_tensorflow_derivative_contract(request, capability, *, external_ad="none"):
    """Check a declared contract; runtime values/budgets still need admission."""
    if not isinstance(request, DerivativeExecutionRequest):
        raise TypeError("A derivative contract requires DerivativeExecutionRequest.")
    order = request.validate_order()
    if request.method not in capability["supported_derivative_methods"]:
        raise UnsupportedFeature("tensorflow_tensor_method")
    if external_ad not in {"none", "eager", "co_traced", "cached_outer_graph"}:
        raise ValueError("Unknown external TensorFlow AD context.")
    if external_ad == "cached_outer_graph":
        raise UnsupportedFeature("tensorflow_external_ad_contract", "UNGUARANTEED_EXTERNAL_AD: differentiation outside an independently cached outer value graph is not verified.")
    if order > capability["maximum_verified_quantum_derivative_order"]:
        raise UnsupportedFeature("tensorflow_derivative_contract", "This execution mode has not verified the requested derivative order; cached callable order two remains experimental.")
    if request.derivative_kind not in capability["supported_operations"]:
        raise UnsupportedFeature("tensorflow_derivative_contract", "The operation is outside this verified scalar/output contract.")
    mode = capability["execution_mode"]
    if mode == "MANAGED_EXPLICIT_QUERY" and external_ad != "none":
        raise UnsupportedFeature("tensorflow_query_autodiff", "Explicit cached queries do not provide composable external AD outputs.")
    if mode == "EXPERIMENTAL_CACHED_CALLABLE" and external_ad == "co_traced":
        raise UnsupportedFeature("tensorflow_controlled_execution", "A controlled cached callable requires a synchronous eager entry.")
    if mode == "TRACE_INLINE_DIFFERENTIABLE" and order and external_ad not in {"eager", "co_traced"}:
        raise UnsupportedFeature("tensorflow_external_ad_contract", "Declare eager or co_traced AD for an inline derivative contract.")
    return capability


def prepare_derivative(device, request):
    if not isinstance(request, DerivativeExecutionRequest):
        raise TypeError("Device derivatives require DerivativeExecutionRequest.")
    request.validate_order()
    method = request.method
    if method == "second_adjoint":
        if request.derivative_order != 2: raise ValueError("second_adjoint requires order two.")
        method = "adjoint"
    if method not in {"best", "adjoint", "parameter-shift", "mps-tangent"}:
        raise UnsupportedFeature("device_derivative_method")
    source = request.circuit_or_tape
    if isinstance(source, QNode):
        node = copy(source)
        args = request.parameters if isinstance(request.parameters, tuple) else (request.parameters,)
    elif isinstance(source, (Tape, Circuit)):
        tape = source.copy() if isinstance(source, Tape) else Tape(source.num_qubits)
        if isinstance(source, Circuit):
            from quantumbridge.core.circuit_metadata import execution_copy
            tape.circuit = execution_copy(source)
            if request.observable is None: raise ValueError("Circuit derivative requests require an observable.")
            tape.set_output(tape.expval(request.observable))
        parameters = tuple(tape.circuit.parameters)
        supplied = request.parameters
        if isinstance(supplied, dict):
            values = [supplied[p] if p in supplied else supplied[p.name] for p in parameters]
        else:
            values = np.asarray(supplied).reshape(-1)
        if len(values) != len(parameters): raise ValueError("Device parameter count does not match native circuit.")
        values = np.asarray(values)
        if values.dtype.kind not in "fiu" or not np.all(np.isfinite(values)):
            raise ValueError("Native circuit parameters must be finite and real.")
        args = (values.astype(float),)
        def program(target, values):
            from quantumbridge.diff.compiled_parameters import record_compiled_parameters
            from quantumbridge.diff.qnode_engine import map_measurements
            original = AngleRecorder()
            original.parameters, original.angles = list(parameters), list(values)
            circuit, recorder = record_compiled_parameters(tape.circuit, original)
            target.circuit = circuit.assign_parameters({p: target.parameter_transform(x) for p, x in zip(recorder.parameters, recorder.angles)})
            tree = tape.output_tree if tape.output_tree is not None else tape.measurements[0] if len(tape.measurements) == 1 else tuple(tape.measurements)
            if not tree: raise ValueError("Tape derivative requests require terminal measurements.")
            map_measurements(tree, lambda item: target.measurements.append(item))
            return tree
        node = QNode(program, tape.circuit.num_qubits, shots=tape.shots, seed=tape.seed, max_diff=2)
    else:
        raise TypeError("Derivative source must be a native Circuit, Tape or QNode.")
    if not args: raise ValueError("Derivative request requires explicit floating-point parameters.")
    arrays = [np.asarray(value) for value in args]
    if any(a.dtype.kind not in "f" or not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("Derivative parameters must be finite real floating-point arrays.")
    node.device, node.interface, node.diff_method, node.last_metadata = device, "numpy", method, {}
    if request.shots is not None: node.shots = request.shots
    if request.seed is not None: node.seed = request.seed
    node.unified_limits = None
    size = sum(a.size for a in arrays)
    direction = _direction(request, args, size)
    limits = resolve_higher_order_limits(node.higher_order_limits)
    argnums = 0 if len(args) == 1 else tuple(range(len(args)))
    order = max(1, request.derivative_order)
    traced, _, layout, single, _ = trace_arguments(args, {}, argnums, order=order,
        direction=direction if order == 2 else None, limits=limits)
    tape, recorder, program = node._construct(traced, {}, prepare_only=True)
    if program is None:
        plan = ExecutionPlan(node, tape, tape.output_tree, recorder)
    else:
        from quantumbridge.transforms.program import TransformedExecutionPlan
        plan = TransformedExecutionPlan(node, tape, recorder, program)
    from quantumbridge.interfaces.native_host import _shape_tree
    from quantumbridge.qml.shot_vector import ShotSpec
    shape = _shape_tree(tape.output_tree, node.num_qubits, ShotSpec.parse(node.shots), limits, "native_device")
    flat, spec = flatten_tree(shape)
    if request.derivative_kind == "gradient" and len(flat) != 1:
        raise ValueError("A gradient request requires exactly one scalar output.")
    if request.derivative_kind == "vjp":
        cotangent, cotangent_spec = flatten_tree(request.cotangent)
        if cotangent_spec != spec or not np.all(np.isfinite(cotangent)):
            raise ValueError("Derivative cotangent must match the output tree and be finite.")
    point = plan.recorder.native_values()
    columns = 1 if direction is not None else size
    from .resource_context import current_resources
    scope = current_resources()
    map_bytes = 8*len(point)*size*(1+(columns if order == 2 else 0))
    if scope is not None: scope.retain("classical_source_maps", map_bytes)
    classical = plan.recorder.classical_jacobian(size)
    second = plan.recorder.classical_second(size, columns) if order == 2 else None
    prepared = finish_derivative(request, node, plan, point, classical, second, direction,
        size, layout, single, spec, len(flat))
    if scope is not None: prepared.buffer_bytes -= map_bytes
    return prepared


def finish_derivative(request, node, plan, point, classical, second, direction, size, layout, single, spec, output_size):
    """Shared resource planning for direct Device and framework source maps."""
    device, method = node.device, node.diff_method
    limits = resolve_higher_order_limits(node.higher_order_limits)
    children = getattr(plan, "child_plans", (plan,))
    if any(child.circuit.has_simulator_instructions for child in children):
        raise UnsupportedFeature("derivative_simulator_instructions",
            "Managed derivative requests do not yet own save/set snapshot result lifetimes; simulator instructions require their dedicated execution API.")
    columns = 1 if direction is not None else size
    workspace = max((device_working_bytes(device, child.circuit) for child in children), default=0)
    if request.derivative_order == 2:
        from quantumbridge.diff.second_order import second_work, transformed_second_work
        work = transformed_second_work(plan, point, classical, second, direction, limits, method)[0] if hasattr(plan, "child_plans") else second_work(plan, point, classical, second, direction, limits, method)[3]
        buffers = work["derivative_buffer_bytes"]+8*(classical.size+second.size)+8*output_size*size*(1+columns)
        if method == "adjoint" or any(child.gradient_capability(method) == "mps-tangent" for child in children):
            # Include C2A local matrices/observables as well as state vectors.
            slices = getattr(plan, "parameter_slices", ((0, len(point)),))
            for child, (start, stop) in zip(children, slices):
                child_work = second_work(child, point[start:stop], classical[start:stop], second[start:stop], direction, limits, method)[3]
                workspace = max(workspace, child_work["combined_working_buffer_bytes"]-child_work["derivative_buffer_bytes"])
    else:
        evaluations = total_shots = samples = tangent_operations = 0
        slices = getattr(plan, "parameter_slices", ((0, len(point)),))
        for child, (start, stop) in zip(children, slices):
            selected = child.gradient_capability(method) if request.derivative_order else None
            n, shots, sample_bytes = _measurement_size(child)
            if selected == "mps-tangent":
                from quantumbridge.diff.mps_derivatives import first_mps_work
                child_work, _ = first_mps_work(child, point[start:stop], limits)
                workspace = max(workspace, child_work["combined_working_buffer_bytes"])
                count = child_work["evaluation_count"]
                evaluations += count
                tangent_operations += child_work["adjoint_operations"]
                continue
            count = 1 if not request.derivative_order else 2 if selected == "adjoint" else 2+sum(
                2*len(child.shift_recipe(op, point[0] if len(point) else 0.)) for op in child._occurrences().values())
            evaluations += count*len(child.shots.partitions)
            total_shots += count*shots
            samples = max(samples, sample_bytes)
        work = {"tape_count": evaluations, "physical_shots": total_shots,
                "evaluation_count": evaluations, "derivative_buffer_bytes": 8*output_size*max(1, size)*4}
        if tangent_operations:
            limits.check("max_adjoint_operations", tangent_operations)
            work["adjoint_operations"] = tangent_operations
        buffers = work["derivative_buffer_bytes"]+8*classical.size
        workspace += samples
    work = dict(work)
    work["derivative_work"] = work.get("adjoint_operations", 0) or max(1, work["evaluation_count"])*sum(max(1, len(child.circuit.operations)) for child in children)
    return PreparedDerivative(request, node, plan, point, classical, second, direction, size,
        layout, single, spec, output_size, work, int(buffers), int(workspace))
