# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Native gate-occurrence derivatives and executable QNode plans.

from dataclasses import replace
from quantumbridge.execution.resource_context import copy_execution_payload as deepcopy
from types import SimpleNamespace
from uuid import uuid4
import warnings
import numpy as np

from quantumbridge.core import Parameter
from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.qml.measurements import MeasurementRequest
from quantumbridge.qml.shot_vector import ShotSpec
from quantumbridge.utils.math import Hamiltonian, PauliString, observable_matrix
from .adjoint import adjoint_rows, gate_derivative
from .trace import Dual, flatten_tree, unflatten_tree


def _physical_measurement_shots(request, shots):
    physical = shots or 0
    if request.kind in {"expval", "var"} and isinstance(request.observable, PauliString):
        if all(label == "I" for _, label in request.observable.terms):
            return 0
    if request.kind == "expval" and isinstance(request.observable, Hamiltonian):
        physical *= sum(any(label != "I" for _, label in word.terms) for _, word in request.observable.terms)
    return physical


class AngleRecorder:
    def __init__(self):
        self.parameters, self.angles = [], []

    def __call__(self, value):
        if len(self.angles) >= 4096: raise MemoryError("Native QNode exceeds the 4096 gate-angle budget.")
        parameter = Parameter(f"__qb_angle_{len(self.angles)}")
        self.parameters.append(parameter)
        self.angles.append(value)
        return parameter

    def native_values(self):
        values = np.asarray([angle.value if isinstance(angle, Dual) else angle for angle in self.angles], dtype=float)
        if values.shape != (len(self.angles),): raise ValueError("Quantum gate angles must be real scalars; parameter broadcasting is not implemented.")
        if not np.all(np.isfinite(values)): raise ValueError("Native QNode angles must be finite real scalars.")
        return values

    def classical_jacobian(self, size):
        result = np.asarray([angle.derivative if isinstance(angle, Dual) else np.zeros(size) for angle in self.angles]).reshape(len(self.angles), size)
        if not np.all(np.isfinite(result)): raise ValueError("Classical preprocessing has a nonfinite derivative.")
        return result

    def classical_second(self, size, columns):
        from .second_trace import SecondDual
        result = np.asarray([angle.second if isinstance(angle, SecondDual) else np.zeros((size, columns))
                             for angle in self.angles]).reshape(len(self.angles), size, columns)
        if not np.all(np.isfinite(result)):
            raise ValueError("Classical preprocessing has a nonfinite second derivative.")
        return result


def measurement_tree(returned, tape):
    def valid(value):
        if isinstance(value, MeasurementRequest): return True
        if isinstance(value, dict): return bool(value) and all(valid(child) for child in value.values())
        return isinstance(value, (tuple, list)) and bool(value) and all(valid(child) for child in value)
    if valid(returned): return returned
    if isinstance(returned, dict): raise UnsupportedFeature("mixed_output_tree", "QNode return dictionaries must contain measurement requests, not discarded constants.")
    if tape.measurements:
        return tape.measurements[0] if len(tape.measurements) == 1 else tuple(tape.measurements)
    return MeasurementRequest(tape.measurement or "expval", observable=tape.observable if tape.observable is not None else returned)


def map_measurements(tree, function):
    if isinstance(tree, MeasurementRequest): return function(tree)
    if isinstance(tree, dict): return {key: map_measurements(child, function) for key, child in tree.items()}
    children = [map_measurements(child, function) for child in tree]
    return tuple(children) if isinstance(tree, tuple) else children


class ExecutionPlan:
    def __init__(self, node, tape, tree, recorder):
        self.node, self.circuit, self.tree, self.recorder = node, tape.circuit, tree, recorder
        self.requests = []
        map_measurements(tree, lambda request: self.requests.append(request))
        self.shots = ShotSpec.parse(node.shots)
        self.legacy_state_queries = any(request.kind in {"state", "density_matrix"} for request in self.requests)
        if self.legacy_state_queries and any(shots is not None for shots in self.shots.partitions):
            warnings.warn("Legacy QNode state queries keep state/expval/var/probs analytic; shots apply only to sampled measurements. Prefer local sample/counts shots with an analytic QNode.", FutureWarning, stacklevel=3)
        from quantumbridge.core.control_flow import ExecutionLimits
        self.limits = getattr(node.device, "execution_limits", None) or ExecutionLimits()
        if sum(shots or 0 for shots in self.shots.partitions) > self.limits.max_shots:
            raise ValueError("QNode shot vector exceeds the total max_shots budget.")
        self.metadata = {}
        self.last_spec = None
        self.resource_history = []

    def _sampling(self, circuit, request, shots, seed):
        device, width = self.node.device, circuit.num_qubits
        observable = request.observable
        if hasattr(observable, "to_hamiltonian"): observable = observable.to_hamiltonian()
        if request.kind == "expval" and isinstance(observable, Hamiltonian):
            return sum(coefficient * self._sampling(circuit, replace(request, observable=pauli), shots, None if seed is None else seed + index)[0]
                       for index, (coefficient, pauli) in enumerate(observable.terms)), None
        if isinstance(observable, PauliString):
            rotated, wires = circuit.copy(), []
            for wire, label in observable.terms:
                if label == "I": continue
                wires.append(wire)
                if label == "X": rotated.h(wire)
                elif label == "Y": rotated.sdg(wire).h(wire)
            if not wires: samples = np.ones(shots)
            else:
                labels = device.sample(rotated, wires=wires, shots=shots, seed=seed)
                samples = np.asarray([(-1.) ** label.count("1") for label in labels])
        else:
            if width > 10: raise UnsupportedFeature("finite_shot_observable", "Large finite-shot observables require Pauli/Hamiltonian form.")
            matrix = observable_matrix(observable, width)
            from quantumbridge.utils.math import validate_observable_matrix
            if not validate_observable_matrix(matrix, width, atol=1e-8, rtol=1e-5): raise ValueError("Sampled observable must be Hermitian.")
            eigenvalues, vectors = np.linalg.eigh(matrix)
            rotated = circuit.copy().unitary(vectors.conj().T, list(range(width)))
            labels = device.sample(rotated, shots=shots, seed=seed)
            samples = eigenvalues[[int(label, 2) for label in labels]]
        return (float(np.mean(samples)) if request.kind == "expval" else float(np.var(samples))), float(np.mean(samples))

    def evaluate(self, values, linear=False, *, sampling_seed=...):
        from quantumbridge.execution.resource_context import current_resources
        resources = current_resources()
        if resources is None:
            return self._evaluate(values, linear, sampling_seed=sampling_seed)
        from quantumbridge.execution.derivative_planning import device_working_bytes
        ledger = resources.ledger
        workspace = device_working_bytes(self.node.device, self.circuit, mps_external_only=True)
        with ledger.reserve("native_device_workspace", memory_bytes=workspace), ledger.reserve("active_native_tape", tapes=1):
            result = self._evaluate(values, linear, sampling_seed=sampling_seed)
            return result

    def _evaluate(self, values, linear=False, *, sampling_seed=...):
        if len(values) != len(self.recorder.parameters): raise ValueError("QNode angle shape mismatch.")
        bound = self.circuit.bind(dict(zip(self.recorder.parameters, values)))
        outputs, partitions, corrections = [], [], []
        root_seed = self.node.seed if sampling_seed is ... else sampling_seed
        for shots, seed in zip(self.shots.partitions, self.shots.seeds(root_seed)):
            statistics, spans, offset, measurement_metadata, joint_metadata, device_resources = [], [], 0, [], [], []
            def execute_measurement(request):
                nonlocal offset
                if getattr(self.node.device,"simulation",None) == "matrix_product_state":
                    self.node.device.last_execution_metadata = {}
                effective_shots = request.shots if request.shots is not None else shots
                if self.legacy_state_queries and request.kind in {"state", "density_matrix", "expval", "var", "probs"}:
                    effective_shots = None
                effective_seed = request.seed if request.seed is not None else seed
                if effective_shots is not None and (isinstance(effective_shots, bool) or not isinstance(effective_shots, (int, np.integer)) or effective_shots < 1):
                    raise ValueError("QNode measurement shots must be positive integers.")
                if effective_shots is not None and effective_shots > self.limits.max_shots:
                    raise ValueError("QNode shots exceed max_shots.")
                physical = _physical_measurement_shots(request, effective_shots)
                measurement_metadata.append({"kind": request.kind, "shots": effective_shots, "seed": effective_seed,"physical_shots":physical})
                if request.kind == "probs" and len(request.wires if request.wires is not None else range(bound.num_qubits)) > 16:
                    raise UnsupportedFeature("probability_output_size", "QNode probability output exceeds the native budget.")
                if request.kind == "pauli_group":
                    from quantumbridge.qml.pauli_group import measure_pauli_group
                    value = measure_pauli_group(self.node.device, bound, request.observable, effective_shots, effective_seed)
                    joint_metadata.append(value.metadata())
                    if value.resources is not None:
                        device_resources.append(deepcopy(value.resources))
                    mean = None
                    if linear:
                        value = value.means
                elif request.kind in {"expval", "var"} and effective_shots is not None:
                    value, mean = self._sampling(bound, request, effective_shots, effective_seed)
                else:
                    request_bound = replace(request, shots=effective_shots, seed=effective_seed)
                    value = self.node._execute_measurement(SimpleNamespace(circuit=bound), request_bound)
                    mean = self.node.device.expectation(bound, request.observable) if request.kind == "var" else None
                if request.kind == "probs" and not (request.metadata or {}).get("measurement_records"):
                    wires = tuple(range(bound.num_qubits)) if request.wires is None else request.wires
                    labels = [format(index, f"0{len(wires)}b") if wires else "" for index in range(1 << len(wires))]
                    if effective_shots is not None:
                        counts = self.node.device.counts(bound, wires=wires, shots=effective_shots, seed=effective_seed)
                        value = {label: counts.get(label, 0) / effective_shots for label in labels}
                    else: value = {label: value.get(label, 0.) for label in labels}
                if request.kind != "pauli_group" and getattr(self.node.device,"simulation",None) == "matrix_product_state" and self.node.device.last_execution_metadata:
                    device_resources.append(deepcopy(self.node.device.last_execution_metadata))
                if linear and request.kind == "var":
                    statistics.append(mean)
                    spans.append((offset, len(statistics) - 1, mean))
                    value += mean ** 2
                try: size = flatten_tree(value)[0].size
                except UnsupportedFeature:
                    size = 0
                offset += size
                return value
            def execute(request):
                from quantumbridge.execution.resource_context import current_resources, measurement_shots
                resources = current_resources()
                if resources is None:
                    return execute_measurement(request)
                effective = request.shots if request.shots is not None else shots
                if self.legacy_state_queries and request.kind in {"state", "density_matrix", "expval", "var", "probs"}:
                    effective = None
                if effective is None and request.kind in {"sample", "counts", "classical_shadow"}:
                    raise UnsupportedFeature("unified_implicit_shots", "Managed sampling requires an explicit shot count.")
                with measurement_shots(resources.ledger, _physical_measurement_shots(request, effective)):
                    return execute_measurement(request)
            output = map_measurements(self.tree, execute)
            if linear:
                vector, spec = flatten_tree(output)
                output = np.concatenate([vector, statistics])
                corrections.append((len(vector), spans, spec))
            outputs.append(output)
            partition = {"shots": shots, "seed": seed, "measurements": measurement_metadata,
                         "simulation": getattr(self.node.device, "simulation", "statevector"),
                         "physical_shots":sum(item["physical_shots"] for item in measurement_metadata)}
            if joint_metadata:
                partition["joint_measurements"] = joint_metadata
            if getattr(self.node.device, "simulation", None) == "matrix_product_state":
                partition["resources"] = {"available":bool(device_resources),"native_measurement_calls":device_resources}
                if device_resources:
                    for key in ("max_bond_observed","max_discarded_weight","max_tensor_bytes","max_truncations"):
                        partition["resources"][key] = max(item[key] for item in device_resources)
                    partition["resources"]["bond_dimensions"] = [max(item["max_bond_dimensions_observed"][index] for item in device_resources) for index in range(bound.num_qubits-1)]
                partition["approximation_warning"] = "MPS truncation may bias values/gradients; discarded weights are not a final infidelity bound."
                self.resource_history.append(partition)
            partitions.append(partition)
            from quantumbridge.execution.resource_context import current_resources
            resources = current_resources()
            if resources is not None:
                resources.ledger.consume(tapes=1, category="completed_partition")
        self.metadata = {"partitions": partitions, "native_execution": True, "legacy_analytic_state_queries": self.legacy_state_queries}
        if linear:
            return outputs, corrections
        result = tuple(outputs) if self.shots.is_vector else outputs[0]
        try: _, self.last_spec = flatten_tree(result)
        except UnsupportedFeature: self.last_spec = None
        return result

    def gradient_capability(self, method):
        pending = [self.circuit]
        while pending:
            body = pending.pop()
            for operation in body.operations:
                if "postselect" in operation.metadata or "postselection" in operation.metadata:
                    raise UnsupportedFeature("postselection_gradient", "Native postselected transformed derivative semantics are not implemented.")
                control = operation.metadata.get("control_node")
                if control is not None:
                    pending.extend(control.bodies)
        if getattr(self.node.device, "simulation", None) == "exact_stabilizer": raise UnsupportedFeature("tableau_continuous_gradient")
        if any(request.kind not in {"expval", "var", "probs", "pauli_group"} for request in self.requests): raise UnsupportedFeature("gradient_measurement")
        if method == "backprop": raise UnsupportedFeature("simulator_backprop", "Framework VJPs are available; differentiable simulator backprop is not implemented.")
        if method not in {"parameter-shift", "finite-diff", "adjoint", "best", "mps-tangent"}: raise UnsupportedFeature("gradient_method", f"Unknown native gradient method {method}.")
        from quantumbridge.devices import StatevectorDevice, MatrixProductStateDevice
        analytic = all(shots is None for shots in self.shots.partitions) and all(request.shots is None for request in self.requests)
        adjoint_ok = type(self.node.device) is StatevectorDevice and not self.circuit.has_dynamic_operations and analytic
        mps_ok = (type(self.node.device) is MatrixProductStateDevice and analytic
            and not self.circuit.has_dynamic_operations and not self.circuit.has_simulator_instructions
            and self.node.device.noise_model is None)
        if method == "best":
            method = "mps-tangent" if mps_ok and any(op.name == "native_matrix_power" for op in self.circuit.operations) else "adjoint" if adjoint_ok else "parameter-shift"
        if method == "adjoint" and not adjoint_ok: raise UnsupportedFeature("adjoint_capability", "Adjoint requires analytic unitary Statevector execution.")
        if method == "mps-tangent" and not mps_ok:
            raise UnsupportedFeature("mps_tangent_capability", "MPS tangents require analytic noiseless unitary MPS execution.")
        return method

    def _occurrences(self):
        found = {}
        def walk(circuit, in_loop=False):
            for operation in circuit.operations:
                node = operation.metadata.get("control_node")
                if node is not None:
                    for body in node.bodies: walk(body, in_loop or node.kind in {"for", "while"})
                for index, parameter in enumerate(operation.params):
                    if isinstance(parameter, Parameter) and parameter in self.recorder.parameters:
                        if in_loop: raise UnsupportedFeature("parameterized_loop_gradient", "A reused loop-body occurrence needs a loop-aware gradient rule.")
                        if parameter in found: raise UnsupportedFeature("reused_gate_occurrence", "Shared inputs must retain separate gate occurrences.")
                        if operation.name != "native_matrix_power" and (index or len(operation.params) != 1):
                            raise UnsupportedFeature("gate_gradient_recipe")
                        found[parameter] = operation
        walk(self.circuit)
        return found

    def shift_recipe(self, operation, value):
        if operation.name == "native_matrix_power":
            raise UnsupportedFeature("matrix_power_parameter_shift",
                "A moving-projector matrix function has no fixed native parameter-shift recipe; use analytic adjoint or MPS tangents.")
        gate_derivative(operation, (value,))
        from .adjoint import gate_derivative_spec
        base,controls,_ = gate_derivative_spec(operation)
        if controls and base in {"rx","ry","rz","rxx","ryy","rzz"}:
            shifts = np.asarray([np.pi / 2, 3 * np.pi / 2])
            frequencies = np.asarray([.5, 1.])
            coefficients = np.linalg.solve(2 * np.sin(frequencies[:, None] * shifts), frequencies)
            return tuple(zip(coefficients, shifts))
        return ((.5, np.pi / 2),)

    def second_derivatives(self, values, method=None, *, direction=None, classical=None, classical_second=None, limits=None):
        from .second_order import execution_second_derivatives
        return execution_second_derivatives(self, values, method, direction=direction, classical=classical,
                                            classical_second=classical_second, limits=limits)

    def jacobian(self, values, method=None, step=1e-6):
        method = self.gradient_capability(method or self.node.diff_method)
        if method == "mps-tangent":
            from .mps_derivatives import mps_jacobian
            return mps_jacobian(self, values)
        base = self.evaluate(values)
        base_metadata = deepcopy(self.metadata)
        receipts = [{"evaluation_id":str(uuid4()),"kind":"base",**deepcopy(self.metadata)}]
        from .processing import result_payload
        flat, spec = flatten_tree(result_payload(base))
        if flat.size * max(1, len(values)) > 2000000: raise MemoryError("QNode Jacobian exceeds the output budget.")
        evaluations = 0
        if method == "adjoint":
            requests = [terminal for request in self.requests for terminal in (
                [MeasurementRequest("expval",observable=PauliString(word)) for word in request.observable.words]
                if request.kind == "pauli_group" else [request])]
            from contextlib import ExitStack
            from quantumbridge.execution.resource_context import current_resources
            resources = current_resources()
            with ExitStack() as stack:
                if resources is not None:
                    from quantumbridge.execution.derivative_planning import device_working_bytes
                    stack.enter_context(resources.ledger.reserve("first_adjoint_workspace",
                        memory_bytes=device_working_bytes(self.node.device, self.circuit)))
                matrix = adjoint_rows(self.node.device, self.circuit, tuple(self.recorder.parameters), values, requests)
                if resources is not None:
                    resources.ledger.consume(tapes=1, category="native_adjoint_sweep")
            evaluations = 1
            receipts.append({"evaluation_id":str(uuid4()),"kind":"native_adjoint_reverse_sweep","physical_shots":0})
        else:
            if not np.isfinite(step) or step <= 0: raise ValueError("Finite-difference step must be positive and finite.")
            occurrences = self._occurrences()
            base_linear, corrections = self.evaluate(values, linear=True)
            receipts.append({"evaluation_id":str(uuid4()),"kind":"linear_moments",**deepcopy(self.metadata)})
            seed_sequence = np.random.SeedSequence(self.node.seed)
            def shifted(point,index,shift,sign):
                seed = int(seed_sequence.spawn(1)[0].generate_state(1)[0])
                result,_ = self.evaluate(point,linear=True,sampling_seed=seed)
                receipts.append({"evaluation_id":str(uuid4()),"kind":"shift" if method == "parameter-shift" else "finite-diff",
                                 "angle_index":index,"shift":float(shift),"sign":sign,"root_seed":seed,**deepcopy(self.metadata)})
                return result
            rows = sum(len(part) for part in base_linear)
            matrix_linear = np.zeros((rows, len(values)))
            for index, parameter in enumerate(self.recorder.parameters):
                operation = occurrences.get(parameter)
                if operation is None: continue
                if method == "finite-diff": recipes = [(1 / (2 * step), step)]
                else:
                    recipes = self.shift_recipe(operation, values[index])
                for coefficient, shift in recipes:
                    plus, minus = np.array(values, copy=True), np.array(values, copy=True)
                    plus[index] += shift; minus[index] -= shift
                    upper = shifted(plus,index,shift,1)
                    lower = shifted(minus,index,shift,-1)
                    matrix_linear[:, index] += coefficient * (np.concatenate(upper) - np.concatenate(lower))
                    evaluations += 2
            matrix, offset = [], 0
            for length, spans, _ in corrections:
                part = matrix_linear[offset:offset + length].copy()
                for row, mean_index, mean in spans: part[row] -= 2 * mean * matrix_linear[offset + length + mean_index]
                matrix.append(part)
                offset += length + len(spans)
            matrix = np.concatenate(matrix, axis=0)
        if matrix.shape != (len(flat), len(values)): raise ValueError("Native Jacobian shape does not match executed outputs.")
        self.metadata = base_metadata
        self.metadata.update(gradient_method=method, shifted_evaluation_count=evaluations if method == "parameter-shift" else 0,
                             derivative_evaluation_count=evaluations, derivative_order=1,
                             gradient_evaluations=receipts,
                             gradient_physical_shots=sum(part["physical_shots"] for receipt in receipts for part in receipt.get("partitions",())),
                             gradient_semantics="finite-shot estimate" if any(shots is not None for shots in self.shots.partitions) or any(request.shots is not None for request in self.requests) else "analytic")
        if self.resource_history: self.metadata["gradient_resource_evaluations"] = list(self.resource_history)
        return base, matrix, spec
