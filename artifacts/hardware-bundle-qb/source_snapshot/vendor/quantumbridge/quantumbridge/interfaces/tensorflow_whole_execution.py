# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Bounded complete-circuit tensor specialization of existing native plans."""

from contextlib import contextmanager

from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.core.operations import execution_operation
from quantumbridge.core.parameters import resolve_parameter
from quantumbridge.diff.compiled_parameters import evaluate_compiled_parameter, record_compiled_parameters
from quantumbridge.diff.qnode_engine import AngleRecorder, ExecutionPlan
from quantumbridge.execution.derivative_planning import tensorflow_statevector_work
from quantumbridge.execution.resource_context import copy_execution_payload
from quantumbridge.utils.math import Hamiltonian, gate_matrix, validate_unitary_matrix
from .tensorflow_native_execution import TensorFlowLocalOperator, _primitive


class TensorFlowStatevectorPlan:
    """All-or-nothing tensor execution; no dynamic/noise/dense MPS fallback."""

    work_planner = staticmethod(tensorflow_statevector_work)

    def __init__(self, plan, *, derivative_order=2, readonly=False, allow_parameter_shift=False):
        if type(plan) is not ExecutionPlan:
            raise TypeError("TF tensor execution requires the existing native ExecutionPlan.")
        compiled = plan.circuit.lowered()
        self.input_size, self.width = len(plan.recorder.parameters), compiled.num_qubits
        self._parameters, self._circuit = tuple(plan.recorder.parameters), compiled
        self._readonly = readonly
        self._local, self._fixed, self._controlled = {}, {}, {}
        fixed_operations = {}
        moving = {"rx", "ry", "rz", "phase", "p", "u1", "u", "u2", "u3", "global_phase"}
        fixed = {"id", "i", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg", "cx", "cy", "cz", "swap"}
        for index, raw in enumerate(compiled.operations):
            operation = execution_operation(raw)
            wires = operation.wires
            if len(set(wires)) != len(wires) or any(type(wire) is not int or not 0 <= wire < self.width for wire in wires):
                raise ValueError("TF compiled operation has invalid wire operands.")
            if operation.metadata.get("directive") and operation.name in {"barrier", "delay"}:
                continue
            if operation.name == "native_matrix_power":
                self._local[index] = TensorFlowLocalOperator(plan, index, readonly=readonly)
            elif operation.name in {"crx", "cry", "crz", "cp", "cphase"} and not operation.metadata and len(wires) == 2 and len(operation.params) == 1:
                self._controlled[index] = "phase" if operation.name in {"cp", "cphase"} else operation.name[1:]
            elif operation.name in moving and not operation.metadata and (len(wires) == 1 or operation.name == "global_phase" and not wires):
                continue
            elif self._supports_fixed(operation, fixed):
                fixed_operations[index] = operation
            else:
                raise UnsupportedFeature("tensorflow_whole_operation", "The whole circuit contains an unvalidated TF operation or metadata.")
        self.work = self.work_planner(plan, compiled, self.input_size, order=derivative_order,
            allow_parameter_shift=allow_parameter_shift)
        self._requests = tuple(copy_execution_payload(plan.requests))
        for index, operation in fixed_operations.items():
            matrix = gate_matrix(operation.name, (), operation.metadata)
            if not validate_unitary_matrix(matrix, 1 << len(operation.wires)):
                raise ValueError("TF fixed primitive does not match its compiled wire frame.")
            self._fixed[index] = matrix
        self.operation_signatures = tuple((execution_operation(raw).name, execution_operation(raw).wires) for raw in compiled.operations)
        self.has_moving_power = bool(self._local)

    @staticmethod
    def _supports_fixed(operation, fixed):
        return operation.name in fixed and not operation.params and set(operation.metadata) <= {"controlled_base"}

    def _check(self, point):
        import tensorflow as tf
        from .tensorflow_native_execution import _finite_parameters
        if not tf.is_tensor(point) or point.dtype != tf.float64 or point.shape != (self.input_size,):
            raise ValueError("TF whole-plan input must match its static float64 occurrence vector.")
        valid = _finite_parameters()(point)
        return tf.where(valid, point, tf.zeros_like(point))

    def fingerprint(self, *, source_plan=None):
        """Describe the actual frozen math store, not just its parent plan."""
        import hashlib
        import json
        circuit = self._circuit if source_plan is None else source_plan.circuit
        parameters = self._parameters if source_plan is None else source_plan.recorder.parameters
        requests = self._requests if source_plan is None else source_plan.requests
        signature = dict(circuit=json.loads(circuit.to_ir().to_json()),
            parameters=[str(parameter.uuid) for parameter in parameters],
            local=[dict(index=index, body=json.loads(local._body.to_ir().to_json()),
                symbols=[str(parameter.uuid) for parameter in local._symbols],
                bindings=local._bindings, modifiers=local._modifiers, controls=local._controls)
                for index, local in sorted(self._local.items())],
            fixed=[(index, matrix.dtype.str, matrix.shape, matrix.tobytes().hex())
                for index, matrix in sorted(self._fixed.items())],
            controlled=sorted(self._controlled.items()),
            observables=[dict(kind=request.kind, wires=request.wires, shots=request.shots,
                seed=request.seed, metadata=request.metadata,
                terms=[(float(coefficient), word.terms) for coefficient, word in
                    (request.observable.terms if isinstance(request.observable, Hamiltonian)
                     else ((1., request.observable),))]) for request in requests])
        return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()

    def _apply(self, state, matrix, wires):
        import tensorflow as tf
        axes = list(wires)+[wire for wire in range(self.width) if wire not in wires]
        inverse = [axes.index(wire) for wire in range(self.width)]
        tensor = tf.transpose(tf.reshape(state, (2,)*self.width), axes)
        tensor = matrix @ tf.reshape(tensor, (1 << len(wires), -1))
        return tf.reshape(tf.transpose(tf.reshape(tensor, (2,)*self.width), inverse), (1 << self.width,))

    def _evolution(self, point):
        """Local matrices from the same readonly native operation/UUID store."""
        import tensorflow as tf
        from . import tensorflow_native_math as numeric
        point = self._check(point)
        if self._readonly:
            body = self._circuit
            mapping = dict(zip(self._parameters, tf.unstack(point)))
        else:
            recorder = AngleRecorder()
            recorder.parameters, recorder.angles = list(self._parameters), list(tf.unstack(point))
            body, recorded = record_compiled_parameters(self._circuit, recorder)
            mapping = dict(zip(recorded.parameters, recorded.angles))
        if tuple((execution_operation(raw).name, execution_operation(raw).wires) for raw in body.operations) != self.operation_signatures:
            raise ValueError("TF parameter recording changed the compiled operation frame.")
        def angle(value):
            result = evaluate_compiled_parameter(value, mapping) if self._readonly else resolve_parameter(value, mapping)
            return tf.convert_to_tensor(result, tf.float64)
        for index, raw in enumerate(body.operations):
            operation = execution_operation(raw)
            if operation.metadata.get("directive") and operation.name in {"barrier", "delay"}:
                continue
            if index in self._local:
                matrix = self._local[index](point)
            elif index in self._fixed:
                matrix = tf.constant(self._fixed[index], tf.complex128)
            else:
                values = [angle(value) for value in operation.params]
                matrix, scalar_phase = _primitive(numeric, self._controlled.get(index, operation.name), values)
                if not operation.wires:
                    yield (), numeric.phase(scalar_phase)
                    continue
                matrix *= numeric.phase(scalar_phase)
                if index in self._controlled:
                    projector = tf.linalg.diag(tf.constant([0, 1], tf.complex128))
                    matrix = tf.eye(4, dtype=tf.complex128)+numeric.kron(projector, matrix-tf.eye(2, dtype=tf.complex128))
            yield operation.wires, matrix
        yield (), numeric.phase(angle(body.global_phase))

    def state(self, point):
        import tensorflow as tf
        state = tf.concat([tf.ones((1,), tf.complex128), tf.zeros(((1 << self.width)-1,), tf.complex128)], 0)
        for wires, matrix in self._evolution(point):
            state = self._apply(state, matrix, wires) if wires else matrix*state
        from .tensorflow_native_execution import _finite_parameters
        valid = _finite_parameters()(tf.concat([tf.math.real(state), tf.math.imag(state)], 0))
        return tf.where(valid, state, tf.zeros_like(state))

    def values(self, point):
        import tensorflow as tf
        state = self.state(point)
        values = []
        for request in self._requests:
            observable = request.observable
            terms = observable.terms if isinstance(observable, Hamiltonian) else ((1., observable),)
            total = tf.constant(0., tf.float64)
            for coefficient, word in terms:
                acted = state
                for wire, label in word.terms:
                    if label != "I":
                        acted = self._apply(acted, tf.constant(gate_matrix(label.lower()), tf.complex128), (wire,))
                total += float(coefficient)*tf.math.real(tf.reduce_sum(tf.math.conj(state)*acted))
            values.append(total)
        from .tensorflow_native_execution import _finite_parameters
        flat = tf.stack(values)
        valid = _finite_parameters()(flat)
        return tf.where(valid, flat, tf.zeros_like(flat))


def _eager_admission(ledger, work, audit):
    from quantumbridge.execution.resource_context import ResourceScope, current_resources
    # Capture numerical limits separately from mutable user-facing metadata.
    memory, planned = work["workspace_bytes"], work["planned_work"]
    @contextmanager
    def admitted(order):
        import tensorflow as tf
        if not tf.executing_eagerly():
            raise UnsupportedFeature("tensorflow_managed_graph_execution", "A managed tensor derivative cannot be traced or cached outside its eager ledger boundary.")
        if not tf.config.experimental.get_synchronous_execution():
            raise UnsupportedFeature("tensorflow_managed_async_execution", "Managed tensor completion accounting requires synchronous eager execution.")
        active = current_resources()
        account = ledger
        while account is not None and (active is None or account is not active.ledger):
            account = account.parent
        if active is not None and account is None:
            raise UnsupportedFeature("tensorflow_foreign_resource_scope", "A deferred tensor derivative cannot bypass an unrelated active resource scope.")
        repetitions = 2 if order == 2 else 1
        category = "tensorflow_eager_order_"+str(order)
        try:
            with ledger.reserve(category, memory_bytes=memory, derivative_work=planned*repetitions, tapes=repetitions) as ticket:
                ticket.consume(derivative_work=planned*repetitions)
                with ResourceScope(ledger, new_request=True) as scope:
                    scope.tensorflow_tensor_workspace_admitted = memory
                    yield
                ticket.consume(tapes=repetitions)
        finally:
            audit.clear()
            audit.update(ledger.receipt())
    ledger.unavailable("tensorflow_allocator_and_retained_outputs", "Logical conservative workspace reservations do not observe TF allocator/RSS or the lifetime of external TF result tensors.")
    return admitted


def select_tensorflow_statevector(kernel, packed):
    """Keep old verified paths intact unless this complete plan is supported."""
    import numpy as np
    import tensorflow as tf
    from quantumbridge.devices.statevector_device import StatevectorDevice
    from quantumbridge.execution.resource_context import current_resources
    from .tensorflow_native_derivatives import bounded_tensor_function
    if (kernel.node.max_diff != 2 or packed.dtype != tf.float64 or kernel.program is not None
            or type(kernel.node.device) is not StatevectorDevice or not 1 <= kernel.node.num_qubits <= 4):
        return None
    if any(index is not None for references in kernel.coefficients for index in references):
        return None
    if not any(execution_operation(raw).name == "native_matrix_power" for raw in kernel.source.circuit.operations):
        return None
    active = current_resources()
    if active is not None and not tf.executing_eagerly():
        raise UnsupportedFeature("tensorflow_tensor_shared_graph_resources", "TF whole-plan tensors do not yet have a shared request ledger across graph execution; use explicit native derivative requests for managed budgets.")
    plan = kernel.plan(np.ones(kernel.size))
    # Do not move ordinary Pauli circuits off their established shift/adjoint
    # resource contracts merely because the tensor implementation exists.
    if not any(execution_operation(raw).name == "native_matrix_power" for raw in plan.circuit.lowered().operations):
        return None
    try:
        whole = TensorFlowStatevectorPlan(plan)
    except UnsupportedFeature:
        return None
    limits = kernel.node.unified_limits
    ledger, audit, admission = None, {}, None
    if limits is not None:
        from quantumbridge.execution.resource_ledger import UnifiedResourceLedger
        ledger = UnifiedResourceLedger(limits, parent=None if active is None else active.ledger)
    elif active is not None:
        ledger = active.ledger
    if ledger is not None:
        if not tf.executing_eagerly():
            raise UnsupportedFeature("tensorflow_managed_graph_execution", "Managed graph execution cannot be accounted through tracing side effects.")
        ledger.preflight(memory_bytes=whole.work["workspace_bytes"], derivative_work=whole.work["planned_work"], tapes=6)
        admission = _eager_admission(ledger, whole.work, audit)
    return bounded_tensor_function(whole.values, admission), whole.work, audit


def select_tensorflow_eager_vjp(kernel):
    """Whole-plan native tensor VJP when TF traces an eager backward rule.

    Ordinary eager calls keep their existing native algorithms. This boundary
    executes no host callback and never mixes per-operation backends.
    """
    import numpy as np
    import tensorflow as tf
    from quantumbridge.devices.statevector_device import StatevectorDevice
    from quantumbridge.execution.resource_context import current_resources
    if kernel.node.unified_limits is not None or current_resources() is not None:
        raise UnsupportedFeature("tensorflow_unified_graph_resources",
            "QB_UNSUPPORTED[tensorflow_unified_graph_resources]: A traced eager VJP cannot promise per-execution shared-ledger ownership; use explicit managed queries.")
    if (kernel.program is not None or type(kernel.node.device) is not StatevectorDevice
            or not 1 <= kernel.node.num_qubits <= 4
            or kernel.node.diff_method not in {"best", "adjoint", "parameter-shift"}
            or any(index is not None for refs in kernel.coefficients for index in refs)):
        return None
    plan = kernel.plan(np.ones(kernel.size))
    method = plan.gradient_capability(kernel.node.diff_method)
    try:
        whole = TensorFlowStatevectorPlan(plan, derivative_order=kernel.node.max_diff,
            readonly=True, allow_parameter_shift=method == "parameter-shift")
    except UnsupportedFeature:
        return None
    occurrences = plan._occurrences() if method == "parameter-shift" else {}
    recipes = tuple(tuple(plan.shift_recipe(occurrences[p], 1.)) if p in occurrences else ()
        for p in plan.recorder.parameters) if method == "parameter-shift" else ()
    shifted = sum(2*len(terms) for terms in recipes)
    work = dict(whole.work)
    if method == "parameter-shift":
        factor = max(1, shifted)
        for field in ("workspace_bytes", "planned_work", "construction_work", "logical_evaluations", "maximum_quantum_graph_copies"):
            work[field] *= factor
        from quantumbridge.diff.higher_order_limits import resolve_higher_order_limits
        limits = resolve_higher_order_limits(kernel.node.higher_order_limits)
        for limit, field in (("max_memory_bytes", "workspace_bytes"), ("max_adjoint_operations", "planned_work"), ("max_total_tapes", "logical_evaluations")):
            limits.check(limit, work[field])
        cap = kernel.node.device.max_memory_mb
        if cap is not None and work["workspace_bytes"] > cap*1024**2:
            raise MemoryError("Native tensor shift workspace exceeds device max_memory_mb.")

    def jacobian(point):
        if method == "parameter-shift":
            columns = []
            for index, terms in enumerate(recipes):
                column = tf.zeros((kernel.output_size,), tf.float64)
                unit = tf.one_hot(index, whole.input_size, dtype=tf.float64)
                for coefficient, shift in terms:
                    column += coefficient*(whole.values(point+shift*unit)-whole.values(point-shift*unit))
                columns.append(column)
            return tf.stack(columns, axis=1)
        with tf.GradientTape(persistent=True) as reverse:
            reverse.watch(point)
            values = whole.values(point)
            components = tf.unstack(values)
        return tf.stack([reverse.gradient(value, point, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            for value in components])

    def refuse_order(point, message):
        if tf.executing_eagerly():
            raise UnsupportedFeature("higher_order_derivatives", message)
        # TF can build a backward graph just to record a legal lower-order
        # result. Refuse when that unapproved graph executes, not at build.
        assertion = tf.raw_ops.Assert(condition=tf.constant(False),
            data=[tf.constant("QB_UNSUPPORTED[higher_order_derivatives]: "+message)])
        with tf.control_dependencies([assertion]):
            return tf.fill(tf.shape(point), tf.constant(float("nan"), tf.float64))

    @tf.custom_gradient
    def second_point(point):
        def refuse(incoming):
            return refuse_order(point, "Native TensorFlow genuine third-point derivatives are not implemented.")
        return tf.identity(point), refuse

    @tf.custom_gradient
    def quantum_jacobian(point):
        matrix = jacobian(point)
        def backward(incoming):
            if kernel.node.max_diff != 2:
                return refuse_order(point, "Native TensorFlow second derivatives require max_diff=2.")
            guarded = second_point(point)
            with tf.GradientTape() as reverse:
                reverse.watch(guarded)
                fresh = jacobian(guarded)
            return reverse.gradient(fresh, guarded, output_gradients=incoming,
                unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return matrix, backward

    def vjp(point, cotangent, direction=None):
        from .tensorflow_native_execution import _finite_parameters
        cotangent = tf.where(_finite_parameters()(cotangent), cotangent, tf.zeros_like(cotangent))
        if direction is None:
            return tf.linalg.matvec(quantum_jacobian(point), cotangent, transpose_a=True)
        direction = tf.where(_finite_parameters()(direction), direction, tf.zeros_like(direction))
        with tf.GradientTape() as reverse:
            reverse.watch(point)
            matrix = quantum_jacobian(point)
            first = tf.linalg.matvec(matrix, cotangent, transpose_a=True)
        return reverse.gradient(first, point, output_gradients=direction,
            unconnected_gradients=tf.UnconnectedGradients.ZERO), tf.linalg.matvec(matrix, direction)
    from types import MappingProxyType
    return vjp, MappingProxyType(dict(work, gradient_method=method,
        shifted_evaluation_count=shifted,
        execution="native whole-plan TF VJP; no host quantum callback", resource_accounting="OUT_OF_SCOPE"))
