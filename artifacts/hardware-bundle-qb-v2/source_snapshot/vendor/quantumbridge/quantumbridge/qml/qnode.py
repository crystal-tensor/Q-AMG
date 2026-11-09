# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from __future__ import annotations

import numpy as np

from quantumbridge.devices import StatevectorDevice
from quantumbridge.utils.math import Hamiltonian, PauliString
from .measurements import ClassicalShadowResult, MeasurementRequest
from .tape import Tape


class QNode:
    def __init__(self, fn, num_qubits: int, device=None, shots=None, seed: int | None = None,
                 interface="auto", diff_method="best", gradient_kwargs=None, max_diff=1,
                 jax_limits=None, tensorflow_limits=None, higher_order_limits=None, unified_limits=None):
        self.fn = fn
        self.num_qubits = num_qubits
        self.device = device or StatevectorDevice()
        self.shots = shots
        self.seed = seed
        self.interface, self.diff_method = interface, diff_method
        self.gradient_kwargs = dict(gradient_kwargs or {})
        self.max_diff = max_diff
        self.jax_limits = jax_limits
        self.tensorflow_limits = tensorflow_limits
        self.higher_order_limits = higher_order_limits
        self.unified_limits = unified_limits
        self.last_metadata = {}
        self.compiler_options = None
        self.transform_program = None
        from .shot_vector import ShotSpec
        ShotSpec.parse(shots)
        if interface not in {None, "auto", "numpy", "torch", "autograd", "jax", "tf", "tensorflow"}:
            raise ValueError("Unknown QuantumBridge QNode interface.")

    def _construct(self, args, kwargs, *, prepare_only=False, capture_native_parameters=False):
        from quantumbridge.diff.qnode_engine import AngleRecorder, ExecutionPlan, measurement_tree
        if self.transform_program is not None and not prepare_only:
            from quantumbridge.interfaces.qnode_autodiff import _leaves
            from quantumbridge.transforms.program import _reject_unsupported_interface
            _reject_unsupported_interface(_leaves((args, kwargs)))
        recorder = AngleRecorder()
        tape = Tape(self.num_qubits, recorder, shots=self.shots, seed=self.seed)
        if capture_native_parameters and len(args) == 1 and not kwargs and self.compiler_options is None:
            tape._capture_native_parameter_input = args[0]
            tape._capture_native_parameter_limits = self.higher_order_limits
        returned = self.fn(tape, *args, **kwargs)
        if self.compiler_options is not None:
            from quantumbridge.compiler import native_transpile
            from quantumbridge.diff.compiled_parameters import record_compiled_parameters
            from quantumbridge.core.classical import UnsupportedFeature
            compiled = native_transpile(tape.circuit, **self.compiler_options)
            identity = dict(enumerate(range(self.num_qubits)))
            from quantumbridge.core.circuit_metadata import execution_metadata
            if compiled.num_qubits != self.num_qubits or execution_metadata(compiled).get("initial_layout") != identity or execution_metadata(compiled).get("routing", {}).get("final_layout", identity) != identity:
                raise UnsupportedFeature("compiled_qnode_layout", "QNode measurements currently require equal width and restored identity layout.")
            tape.circuit, recorder = record_compiled_parameters(compiled, recorder)
        tree = measurement_tree(returned, tape)
        program = self.transform_program
        if program is None:
            from quantumbridge.diff.scalars import is_live_scalar
            from quantumbridge.diff.qnode_engine import map_measurements
            requests = []
            map_measurements(tree,lambda request:requests.append(request))
            if any(isinstance(request.observable,Hamiltonian) and any(is_live_scalar(coefficient) for coefficient,_ in request.observable.terms) for request in requests):
                from quantumbridge.transforms import TransformProgram, split_to_single_terms
                program = TransformProgram((split_to_single_terms,))
        if prepare_only:
            tape.output_tree = tree
            return tape, recorder, program
        if program is None:
            plan = ExecutionPlan(self, tape, tree, recorder)
        else:
            from quantumbridge.transforms.program import TransformedExecutionPlan
            tape.output_tree = tree
            plan = TransformedExecutionPlan(self, tape, recorder, program)
        plan.input_values = (args, kwargs)
        return plan

    def with_transform(self, transform, *, limits=None):
        """Return an independent QNode using a fresh transformed batch per call."""
        from copy import copy
        from quantumbridge.transforms.program import TransformProgram
        previous = self.transform_program
        prefix = () if previous is None else previous.transforms
        stages = transform.transforms if isinstance(transform, TransformProgram) else (transform,)
        if limits is None:
            limits = getattr(transform, "limits", None) if previous is None else previous.limits
        result = copy(self)
        result.last_metadata = {}
        result.transform_program = TransformProgram(prefix + stages, limits=limits)
        return result

    def compile(self, **options):
        from copy import copy
        result = copy(self)
        result.compiler_options, result.last_metadata = dict(options), {}
        return result

    @classmethod
    def from_circuit(cls,circuit,observable,**options):
        """Execute a native parameterized Circuit; Python callables are not serialized."""
        from quantumbridge.diff.qnode_engine import AngleRecorder
        from quantumbridge.diff.compiled_parameters import record_compiled_parameters
        parameters = circuit.parameters
        def program(tape,*values):
            capture = (len(values) == 1 and hasattr(tape, "_capture_native_parameter_input")
                and values[0] is tape._capture_native_parameter_input
                and not getattr(tape.parameter_transform, "angles", ()))
            if len(values) == 1 and isinstance(values[0],(tuple,list)):
                values = tuple(values[0])
            elif len(values) == 1 and hasattr(values[0],"shape") and len(values[0].shape):
                if type(values[0]).__module__.startswith("tensorflow"):
                    import tensorflow as tf
                    values = tuple(tf.unstack(tf.reshape(values[0], (-1,))))
                else:
                    values = tuple(values[0].reshape(-1))
            if len(values) != len(parameters): raise ValueError("Circuit QNode argument count must match its free parameter count.")
            source = AngleRecorder()
            source.parameters = list(parameters)
            if capture:
                from quantumbridge.core.classical import UnsupportedFeature
                from quantumbridge.diff.compiled_parameters import FrozenParameterBinding
                try:
                    source.angles = list(parameters)
                    prepared, recorded = record_compiled_parameters(circuit, source)
                    binding = FrozenParameterBinding.create(tuple(parameters), tuple(recorded.angles),
                        limits=tape._capture_native_parameter_limits)
                except UnsupportedFeature as error:
                    if error.feature not in {"frozen_parameter_expression", "frozen_parameter_power"}:
                        raise
                    # Optional native capture must not revoke an existing TF
                    # scalar-subset query. Identity/domain/safety errors and
                    # resource failures must not enter this fallback.
                    capture = False
            if capture:
                angles = binding.evaluate(tuple(values))
            else:
                source.angles = list(values)
                prepared, recorded = record_compiled_parameters(circuit, source)
                angles = recorded.angles
            mapping = {parameter:tape.parameter_transform(value) for parameter,value in zip(recorded.parameters,angles)}
            if capture:
                tape._native_parameter_binding = (binding, tuple(tape.parameter_transform.parameters))
            from quantumbridge.core.circuit import Circuit
            tape.circuit = Circuit.assign_parameters(prepared,mapping,_copy_user_metadata=False)
            return tape.expval(observable)
        return cls(program,circuit.num_qubits,**options)

    def __call__(self, *args, **kwargs):
        from quantumbridge.interfaces.qnode_autodiff import execute_interface, infer_interface
        interface = infer_interface(args, kwargs) if self.interface == "auto" else self.interface
        if self.transform_program is not None and interface in {None, "numpy"} and infer_interface(args, kwargs) is not None:
            from quantumbridge.core.classical import UnsupportedFeature
            raise UnsupportedFeature("transform_interface", "Use the framework interface or native gradient/VJP API; explicit NumPy cannot detach live framework inputs.")
        if interface == "jax":
            from quantumbridge.interfaces.jax_autodiff import execute_jax
            return execute_jax(self, args, kwargs)
        if interface in {"tf", "tensorflow"}:
            from quantumbridge.interfaces.tensorflow_autodiff import execute_tensorflow
            return execute_tensorflow(self, args, kwargs)
        if interface in {None, "numpy"}:
            from quantumbridge.execution.qnode_derivatives import manages_native_node, execute_native_qnode
            if manages_native_node(self):
                return execute_native_qnode(self, args, kwargs)[0].value
        plan = self._construct(args, kwargs)
        if interface in {None, "numpy"}:
            result = plan.evaluate(plan.recorder.native_values())
        else: result = execute_interface(plan, interface)
        self.last_metadata = plan.metadata
        return result

    def _differentiate(self, args, kwargs, argnums=0, argnames=(), method=None):
        from quantumbridge.execution.qnode_derivatives import manages_native_node, execute_native_qnode
        if manages_native_node(self):
            result, prepared = execute_native_qnode(self, args, kwargs, order=1,
                argnums=argnums, argnames=argnames, method=method)
            return result.value, result.jacobian, prepared.output_spec, prepared.layout, prepared.single
        from quantumbridge.core.classical import UnsupportedFeature
        from quantumbridge.diff.trace import trace_arguments
        if self.max_diff not in {1, 2}: raise UnsupportedFeature("higher_order_derivatives", "Native QNode implements derivative orders at most two.")
        if isinstance(argnums, str): argnames, argnums = (argnums,), ()
        if argnums == 0 and not args and not argnames:
            import inspect
            name = next(iter(tuple(inspect.signature(self.fn).parameters)[1:]), None)
            if name not in kwargs: raise ValueError("No first trainable QNode argument was supplied.")
            argnames, argnums = (name,), ()
        traced_args, traced_kwargs, layout, single, size = trace_arguments(args, kwargs, argnums, argnames)
        plan = self._construct(traced_args, traced_kwargs)
        base, matrix, spec = plan.jacobian(plan.recorder.native_values(), method, **self.gradient_kwargs)
        classical = plan.recorder.classical_jacobian(size)
        matrix = matrix @ classical
        gate_count = plan.metadata.get("gate_parameter_count",len(plan.recorder.parameters))
        active = np.flatnonzero(np.any(classical != 0,axis=1))
        self.last_metadata = {**plan.metadata, "classical_chain_rule": True, "input_parameter_count": size,
                              "gate_parameter_count": gate_count,"scalar_parameter_count":len(plan.recorder.parameters),
                              "trainable_gate_occurrences":active[active < gate_count].tolist(),
                              "trainable_coefficient_slots":(active[active >= gate_count]-gate_count).tolist()}
        return base, matrix, spec, layout, single

    def jax_with_status(self, *args, **kwargs):
        """Return (value, nondifferentiable status) for checked CPU JAX calls.

        Codes: 0 success, 1 invalid/nonfinite value, 2 unsupported runtime path,
        3 resource budget, 4 other runtime failure. Failed values are NaN.
        Static capability/shape errors still reject before callback creation.
        """
        from quantumbridge.interfaces.jax_autodiff import execute_jax
        return execute_jax(self, args, kwargs, return_status=True)

    def tensorflow_with_status(self, *args, **kwargs):
        """Return (value, status) for checked CPU TF host calls.

        Status: 0 success, 1 invalid value, 2 unsupported path, 3 resource
        limit, 4 unexpected host failure. Failed values/derivatives are NaN.
        Static capability/shape errors reject before callback creation.
        """
        from quantumbridge.interfaces.tensorflow_autodiff import execute_tensorflow
        return execute_tensorflow(self, args, kwargs, return_status=True)

    def compile_tensorflow(self, *, input_signature, max_derivative_order=None, controlled_execution=True):
        """Freeze a native tensor plan with a scoped, synchronous CPU entry.

        Mathematical sources are snapshots. Owner execution/resource settings
        must remain unchanged and the QNode owner must stay alive; compile a
        new handle after reconfiguration.
        A raw internal ConcreteFunction has no Python invocation accounting.
        Cached outer reverse-over-forward remains experimental, not accepted.
        Use handle.execute(request) for explicit queries, or its checked
        differentiable_call for a declared first-order contract. Bare handle
        calls preserve the experimental diagnostic envelope, not verified
        second-order callable support.
        """
        from quantumbridge.interfaces.tensorflow_managed import ManagedTensorFlowQNode
        return ManagedTensorFlowQNode(self, input_signature=input_signature,
            max_derivative_order=max_derivative_order, controlled_execution=controlled_execution)

    def compile_tensorflow_inline(self, *, parameters, input_signature, max_derivative_order=2):
        """Freeze native scalar syntax for explicit trace-inline differentiation.

        Unlike compile_tensorflow, this mode has static bounds and runtime
        numerical guards, not per-execution accounting of external TF graphs.
        Active resource scopes are refused. Python owner checks run when this
        callable is entered, not on direct calls to a user's cached TF graph.
        """
        from quantumbridge.interfaces.tensorflow_inline import TraceInlineTensorFlowQNode
        return TraceInlineTensorFlowQNode(self, parameters=parameters,
            input_signature=input_signature, max_derivative_order=max_derivative_order)

    def jax_hvp_with_status(self, tangent, *args, **kwargs):
        """Return a native directional HVP and runtime status (CPU query API)."""
        from quantumbridge.interfaces.jax_derivative_status import execute_jax_derivative_status
        return execute_jax_derivative_status(self, args, kwargs, kind="hvp", tangent=tangent)

    def jax_hessian_with_status(self, *args, **kwargs):
        """Return a native materialized Hessian and runtime status."""
        from quantumbridge.interfaces.jax_derivative_status import execute_jax_derivative_status
        return execute_jax_derivative_status(self, args, kwargs, kind="hessian")

    def jax_jacobian_with_status(self, *args, **kwargs):
        """Return a native Jacobian and runtime status."""
        from quantumbridge.interfaces.jax_derivative_status import execute_jax_derivative_status
        return execute_jax_derivative_status(self, args, kwargs, kind="jacobian")

    def tensorflow_hvp_with_status(self, tangent, *args, **kwargs):
        """Return a CPU native HVP query and its derivative-specific status."""
        from quantumbridge.interfaces.tensorflow_derivative_status import execute_tensorflow_derivative_status
        return execute_tensorflow_derivative_status(self, args, kwargs, kind="hvp", tangent=tangent)

    def tensorflow_hessian_with_status(self, *args, **kwargs):
        from quantumbridge.interfaces.tensorflow_derivative_status import execute_tensorflow_derivative_status
        return execute_tensorflow_derivative_status(self, args, kwargs, kind="hessian")

    def tensorflow_jacobian_with_status(self, *args, **kwargs):
        from quantumbridge.interfaces.tensorflow_derivative_status import execute_tensorflow_derivative_status
        return execute_tensorflow_derivative_status(self, args, kwargs, kind="jacobian")

    def tensorflow_vjp_with_status(self, cotangent, *args, **kwargs):
        from quantumbridge.interfaces.tensorflow_derivative_status import execute_tensorflow_derivative_status
        return execute_tensorflow_derivative_status(self, args, kwargs, kind="vjp", cotangent=cotangent)

    def tensorflow_value_with_status(self, *args, **kwargs):
        """Explicit value query using the same unified Device request account."""
        from quantumbridge.interfaces.tensorflow_derivative_status import execute_tensorflow_derivative_status
        return execute_tensorflow_derivative_status(self, args, kwargs, kind="value")

    def tensorflow_batch(self, values):
        """Bounded CPU leading-axis mapping for one static tensor argument."""
        from quantumbridge.interfaces.tensorflow_autodiff import execute_tensorflow_batch
        return execute_tensorflow_batch(self, values)

    @staticmethod
    def _input_results(values, layout, single):
        results = [values[indices].reshape(shape) for shape, indices in layout]
        results = [value.item() if not value.shape else value for value in results]
        return results[0] if single else tuple(results)

    def compute_jacobian(self, *args, argnums=0, argnames=(), method=None, **kwargs):
        from quantumbridge.diff.trace import unflatten_tree
        _, matrix, spec, layout, single = self._differentiate(args, kwargs, argnums, argnames, method)
        result = [unflatten_tree(matrix[:, indices], spec, shape) for shape, indices in layout]
        return result[0] if single else tuple(result)

    jacobian = compute_jacobian

    def _differentiate_second(self, args, kwargs, argnums, argnames, method, tangent=None):
        from quantumbridge.execution.qnode_derivatives import manages_native_node, execute_native_qnode
        if manages_native_node(self):
            result, prepared = execute_native_qnode(self, args, kwargs, order=2,
                argnums=argnums, argnames=argnames, method=method, tangent=tangent)
            return result.value, result.jacobian, result.second, prepared.output_spec, prepared.layout, prepared.single
        from quantumbridge.core.classical import UnsupportedFeature
        from quantumbridge.diff.higher_order_limits import resolve_higher_order_limits
        from quantumbridge.diff.trace import trace_arguments
        if self.max_diff != 2:
            raise UnsupportedFeature("higher_order_derivatives", "Second derivatives require max_diff=2; orders above two are not implemented.")
        if isinstance(argnums, str): argnames, argnums = (argnums,), ()
        if argnums == 0 and not args and not argnames:
            import inspect
            name = next(iter(tuple(inspect.signature(self.fn).parameters)[1:]), None)
            if name not in kwargs: raise ValueError("No first trainable QNode argument was supplied.")
            argnames, argnums = (name,), ()
        direction = None
        if tangent is not None:
            positions = (argnums,) if isinstance(argnums, int) else tuple(argnums)
            names = (argnames,) if isinstance(argnames, str) else tuple(argnames)
            selected = [args[index] for index in positions]+[kwargs[name] for name in names]
            single = isinstance(argnums, int) and not argnames or not positions and len(names) == 1
            tangents = [tangent] if single else list(tangent)
            if len(tangents) != len(selected) or any(np.shape(value) != np.shape(source) for value, source in zip(tangents, selected)):
                raise ValueError("HVP tangent shapes must match the selected trainable arguments.")
            if any(np.asarray(value).dtype.kind not in "fiu" or not np.all(np.isfinite(value)) for value in tangents):
                raise ValueError("HVP tangents must contain finite real values.")
            direction = np.concatenate([np.asarray(value, dtype=float).reshape(-1) for value in tangents])
        limits = resolve_higher_order_limits(self.higher_order_limits)
        traced_args, traced_kwargs, layout, single, size = trace_arguments(
            args, kwargs, argnums, argnames, order=2, direction=direction, limits=limits)
        plan = self._construct(traced_args, traced_kwargs)
        columns = size if direction is None else 1
        count = len(plan.recorder.parameters)
        limits.check("max_memory_bytes", 8*count*size*(1+columns))
        base, matrix, second, spec = plan.second_derivatives(
            plan.recorder.native_values(), method, direction=direction,
            classical=plan.recorder.classical_jacobian(size),
            classical_second=plan.recorder.classical_second(size, columns), limits=limits)
        self.last_metadata = {**plan.metadata, "classical_chain_rule": True,
                              "second_classical_chain_rule": True, "input_parameter_count": size,
                              "scalar_parameter_count": count,
                              "gate_parameter_count": plan.metadata.get("gate_parameter_count", count)}
        return base, matrix, second, spec, layout, single

    def execute_and_compute_hessian(self, *args, argnums=0, argnames=(), method=None, **kwargs):
        from quantumbridge.diff.trace import unflatten_tree
        base, _, second, spec, layout, single = self._differentiate_second(args, kwargs, argnums, argnames, method)
        blocks = tuple(tuple(unflatten_tree(second[:, left, right], spec, shape+other_shape)
                             for other_shape, right in layout) for shape, left in layout)
        return base, blocks[0][0] if single else blocks

    def compute_hessian(self, *args, argnums=0, argnames=(), method=None, **kwargs):
        return self.execute_and_compute_hessian(*args, argnums=argnums, argnames=argnames, method=method, **kwargs)[1]

    hessian = compute_hessian

    def execute_and_compute_hvp(self, tangent, *args, argnums=0, argnames=(), method=None, **kwargs):
        from quantumbridge.diff.trace import unflatten_tree
        if tangent is None:
            raise ValueError("HVP requires an explicit real tangent matching the selected inputs.")
        base, _, second, spec, layout, single = self._differentiate_second(args, kwargs, argnums, argnames, method, tangent)
        results = tuple(unflatten_tree(second[:, indices, 0], spec, shape) for shape, indices in layout)
        return base, results[0] if single else results

    def compute_hvp(self, tangent, *args, argnums=0, argnames=(), method=None, **kwargs):
        return self.execute_and_compute_hvp(tangent, *args, argnums=argnums, argnames=argnames, method=method, **kwargs)[1]

    def gradient(self, *args, argnums=0, argnames=(), method=None, **kwargs):
        base, matrix, spec, layout, single = self._differentiate(args, kwargs, argnums, argnames, method)
        if len(matrix) != 1: raise ValueError("QNode.gradient requires a scalar output; use jacobian for multiple outputs.")
        return self._input_results(matrix[0], layout, single)

    def compute_vjp(self, cotangent, *args, argnums=0, argnames=(), method=None, **kwargs):
        return self.execute_and_compute_vjp(cotangent, *args, argnums=argnums, argnames=argnames, method=method, **kwargs)[1]

    def execute_and_compute_vjp(self, cotangent, *args, argnums=0, argnames=(), method=None, **kwargs):
        from quantumbridge.diff.trace import flatten_tree
        base, matrix, spec, layout, single = self._differentiate(args, kwargs, argnums, argnames, method)
        vector, cotangent_spec = flatten_tree(cotangent)
        if spec != cotangent_spec: raise ValueError("Cotangent tree/shape must match the QNode output.")
        return base, self._input_results(vector @ matrix, layout, single)

    def compute_jvp(self, tangent, *args, argnums=0, argnames=(), method=None, **kwargs):
        return self.execute_and_compute_jvp(tangent, *args, argnums=argnums, argnames=argnames, method=method, **kwargs)[1]

    def execute_and_compute_jvp(self, tangent, *args, argnums=0, argnames=(), method=None, **kwargs):
        from quantumbridge.diff.trace import unflatten_tree
        base, matrix, spec, layout, single = self._differentiate(args, kwargs, argnums, argnames, method)
        tangents = [tangent] if single else list(tangent)
        if len(tangents) != len(layout) or any(np.asarray(value).shape != shape for value, (shape, _) in zip(tangents, layout)):
            raise ValueError("Tangent shapes must match the trainable arguments.")
        if any(np.iscomplexobj(value) for value in tangents): raise TypeError("Real inputs require real tangents.")
        vector = np.concatenate([np.asarray(value, dtype=float).reshape(-1) for value in tangents])
        return base, unflatten_tree(matrix @ vector, spec)

    def execute_and_compute_derivatives(self, *args, argnums=0, argnames=(), method=None, **kwargs):
        from quantumbridge.diff.trace import unflatten_tree
        base, matrix, spec, layout, single = self._differentiate(args, kwargs, argnums, argnames, method)
        result = [unflatten_tree(matrix[:, indices], spec, shape) for shape, indices in layout]
        return base, result[0] if single else tuple(result)

    def _execute_measurement(self, tape: Tape, request: MeasurementRequest):
        wires = _validate_wires(request.wires, self.num_qubits)
        if (request.metadata or {}).get("measurement_records"):
            from .measurement_records import execute_records
            return execute_records(self.device, tape.circuit, request)
        if request.kind == "expval":
            return self.device.expectation(tape.circuit, request.observable)
        if request.kind == "var":
            return self.device.variance(tape.circuit, request.observable)
        if request.kind == "probs":
            return self.device.probabilities(tape.circuit, wires=wires)
        if request.kind == "state":
            if hasattr(self.device, "state"):
                return self.device.state(tape.circuit)
            return self.device.statevector(tape.circuit)
        if request.kind == "density_matrix":
            if hasattr(self.device, "reduced_density_matrix"):
                return self.device.reduced_density_matrix(tape.circuit, wires=wires)
            matrix = self.device.density_matrix(tape.circuit)
            return matrix if wires is None else _partial_density_matrix(matrix, self.num_qubits, wires)
        if request.kind == "sample":
            shots = _resolve_shots(request.shots, self.shots, "sample")
            seed = request.seed if request.seed is not None else self.seed
            return self.device.sample(tape.circuit, wires=wires, shots=shots, seed=seed)
        if request.kind == "counts":
            shots = _resolve_shots(request.shots, self.shots, "counts")
            seed = request.seed if request.seed is not None else self.seed
            return self.device.counts(tape.circuit, wires=wires, shots=shots, seed=seed)
        if request.kind == "classical_shadow":
            shots = _resolve_shots(request.shots, self.shots, "classical_shadow")
            seed = request.seed if request.seed is not None else self.seed
            return _classical_shadow(self.device, tape.circuit, self.num_qubits, wires, shots, seed)
        if request.kind == "shadow_expval":
            shots = _resolve_shots(request.shots, self.shots, "shadow_expval")
            seed = request.seed if request.seed is not None else self.seed
            shadow = _classical_shadow(self.device, tape.circuit, self.num_qubits, wires, shots, seed)
            k = None if request.metadata is None else request.metadata.get("k")
            return _shadow_expectation(shadow, request.observable, k=k)
        raise ValueError(f"Unsupported QuantumBridge QML measurement kind: {request.kind!r}")


def _returned_measurements(returned) -> tuple[MeasurementRequest, ...]:
    if isinstance(returned, MeasurementRequest):
        return (returned,)
    if isinstance(returned, (list, tuple)) and all(isinstance(item, MeasurementRequest) for item in returned):
        return tuple(returned)
    return ()


def _validate_wires(wires: tuple[int, ...] | None, num_qubits: int) -> tuple[int, ...] | None:
    if wires is None:
        return None
    selected = tuple(wires)
    if len(set(selected)) != len(selected):
        raise ValueError("QuantumBridge QML measurement wires must be unique.")
    if any(wire < 0 or wire >= num_qubits for wire in selected):
        raise ValueError(f"QuantumBridge QML measurement wires must be in [0, {num_qubits}).")
    return selected


def _resolve_shots(request_shots, node_shots, measurement: str) -> int:
    shots = request_shots if request_shots is not None else node_shots
    if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)) or shots <= 0:
        raise ValueError(f"QuantumBridge QML {measurement} requires a positive integer number of shots.")
    return int(shots)


def _partial_density_matrix(matrix: np.ndarray, num_qubits: int, wires: tuple[int, ...]) -> np.ndarray:
    selected = tuple(wires)
    traced = tuple(wire for wire in range(num_qubits) if wire not in selected)
    size = 1 << len(selected)
    reduced = np.zeros((size, size), dtype=complex)
    for row in range(1 << num_qubits):
        for col in range(1 << num_qubits):
            if all(_bit(row, wire, num_qubits) == _bit(col, wire, num_qubits) for wire in traced):
                reduced[
                    _project_index(row, selected, num_qubits),
                    _project_index(col, selected, num_qubits),
                ] += matrix[row, col]
    return reduced


def _bit(index: int, wire: int, num_qubits: int) -> int:
    return (index >> (num_qubits - 1 - wire)) & 1


def _project_index(index: int, wires: tuple[int, ...], num_qubits: int) -> int:
    out = 0
    for wire in wires:
        out = (out << 1) | _bit(index, wire, num_qubits)
    return out


def _classical_shadow(device, circuit, num_qubits: int, wires: tuple[int, ...] | None, shots: int, seed: int | None):
    selected = tuple(range(num_qubits)) if wires is None else tuple(wires)
    rng = np.random.default_rng(seed)
    recipes = rng.integers(0, 3, size=(shots, len(selected)), dtype=np.int8)
    bits = np.zeros((shots, len(selected)), dtype=np.int8)
    unique_recipes, groups = np.unique(recipes, axis=0, return_inverse=True)
    for group, recipe_row in enumerate(unique_recipes):
        indices = np.flatnonzero(groups == group)
        rotated = circuit.copy()
        for wire, recipe_value in zip(selected, recipe_row):
            recipe = int(recipe_value)
            if recipe == 0:
                rotated.h(wire)
            elif recipe == 1:
                rotated.sdg(wire)
                rotated.h(wire)
        group_seed = int(rng.integers(0, 2**32 - 1))
        samples = device.sample(rotated, wires=selected, shots=len(indices), seed=group_seed)
        for index, bitstring in zip(indices, samples):
            bits[index] = np.fromiter((int(bit) for bit in bitstring), dtype=np.int8, count=len(selected))
    return ClassicalShadowResult(
        bits=bits,
        recipes=recipes,
        wires=selected,
        metadata={
            "measurement": "classical_shadow",
            "shots": shots,
            "seed": seed,
            "recipe_encoding": {"X": 0, "Y": 1, "Z": 2},
        },
    )


def _shadow_expectation(shadow: ClassicalShadowResult, observable, k=None):
    if isinstance(observable, (list, tuple)):
        return np.asarray([_shadow_expectation(shadow, item, k=k) for item in observable], dtype=float)
    per_shot = _shadow_observable_samples(shadow, observable)
    partitions = _resolve_shadow_partitions(k, len(per_shot))
    means = [float(np.mean(partition)) for partition in np.array_split(per_shot, partitions)]
    return float(np.median(means))


def _shadow_observable_samples(shadow: ClassicalShadowResult, observable) -> np.ndarray:
    if hasattr(observable, "to_hamiltonian"):
        observable = observable.to_hamiltonian()
    if isinstance(observable, PauliString):
        return _shadow_pauli_samples(shadow, observable)
    if isinstance(observable, Hamiltonian):
        estimate = np.zeros(len(shadow.bits), dtype=float)
        for coefficient, pauli in observable.terms:
            estimate += coefficient * _shadow_pauli_samples(shadow, pauli)
        return estimate
    raise TypeError("QuantumBridge shadow_expval supports PauliString and Hamiltonian observables.")


def _shadow_pauli_samples(shadow: ClassicalShadowResult, observable: PauliString) -> np.ndarray:
    wire_positions = {wire: offset for offset, wire in enumerate(shadow.wires)}
    seen: set[int] = set()
    estimate = np.ones(len(shadow.bits), dtype=float)
    recipe_for = {"X": 0, "Y": 1, "Z": 2}
    for wire, label in observable.terms:
        if label == "I":
            continue
        if wire in seen:
            raise ValueError("QuantumBridge shadow_expval Pauli terms must use unique wires.")
        seen.add(wire)
        if wire not in wire_positions:
            raise ValueError(f"Observable wire {wire} is not present in the classical shadow wires.")
        offset = wire_positions[wire]
        matches = shadow.recipes[:, offset] == recipe_for[label]
        eigenvalues = 1.0 - 2.0 * shadow.bits[:, offset]
        estimate *= np.where(matches, 3.0 * eigenvalues, 0.0)
    return estimate


def _resolve_shadow_partitions(k, shots: int) -> int:
    if k is None:
        return 1
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k <= 0 or k > shots:
        raise ValueError("QuantumBridge shadow_expval k must be a positive integer no larger than shots.")
    return int(k)
