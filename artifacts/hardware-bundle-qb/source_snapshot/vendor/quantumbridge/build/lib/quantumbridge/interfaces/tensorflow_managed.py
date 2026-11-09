# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Scoped synchronous invocations of frozen native TensorFlow quantum plans."""

from copy import copy, deepcopy
from contextlib import contextmanager
from dataclasses import replace
from numbers import Integral, Real
from threading import RLock
from types import MappingProxyType
import hashlib
import json
import weakref

import numpy as np

from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.diff.higher_order_limits import resolve_higher_order_limits
from quantumbridge.diff.trace import unflatten_tree
from quantumbridge.execution.derivative_planning import (require_tensorflow_derivative_contract,
    tensorflow_execution_capability)
from quantumbridge.execution.derivative_request import DerivativeExecutionRequest
from quantumbridge.execution.derivative_result import DerivativeExecutionResult
from quantumbridge.execution.resource_context import ResourceScope, current_resources
from quantumbridge.execution.resource_ledger import UnifiedResourceLedger
from quantumbridge.execution.resource_limits import UnifiedExecutionLimits, resolve_unified_limits
from .native_host import MissingOptionalDependency, NativeHostKernel
from .tensorflow_autodiff import TensorFlowBridgeLimits
from .tensorflow_native_derivatives import bounded_tensor_function
from .tensorflow_native_execution import _finite_parameters
from .tensorflow_whole_execution import TensorFlowStatevectorPlan


def _configuration(node):
    device = node.device
    return (node.diff_method, node.max_diff, node.num_qubits, node.shots, node.seed,
        node.higher_order_limits, node.unified_limits, node.tensorflow_limits,
        deepcopy(node.gradient_kwargs), id(node.fn), id(device),
        getattr(device, "max_qubits", None), getattr(device, "max_memory_mb", None),
        deepcopy(getattr(device, "execution_limits", None)),
        deepcopy(getattr(device, "state_options", None)),
        id(getattr(device, "noise_model", None)), id(getattr(device, "initial_state", None)))


def _admit_legacy_scalar_query(concrete, source, *, ledger, bridge, higher):
    """Validate the original public TF graph, never a GraphDef execution import.

    Native lowering may miss an operation, but that does not attest to the
    safety of another execution route. Only the unchanged captured pure graph
    may retain its previous explicit-query contract.
    """
    import tensorflow as tf
    from quantumbridge.core.symbolic_limits import current_symbolic_limits
    size = source.ByteSize()
    bridge.check("max_scalar_graph_bytes", size)
    operations = concrete.graph.get_operations()
    storage = sum(t.shape.num_elements()*t.dtype.size for op in operations for t in op.outputs)
    captures = tuple(concrete.captured_inputs)
    for tensor in captures:
        if not tensor.shape.is_fully_defined():
            raise UnsupportedFeature("tensorflow_scalar_source_safety", "Legacy captures require fully static shapes.")
        bridge.check("max_scalar_rank", tensor.shape.rank)
        for dim in tensor.shape:
            bridge.check("max_scalar_elements", dim)
        bridge.check("max_scalar_elements", tensor.shape.num_elements())
    capture_storage = sum(t.shape.num_elements()*t.dtype.size for t in captures)
    workspace, work = 3*size+storage+2*capture_storage, len(operations)+(storage+capture_storage)//8
    higher.check("max_memory_bytes", workspace)
    higher.check("max_adjoint_operations", work)
    with ledger.reserve("tensorflow_legacy_scalar_validation", memory_bytes=workspace,
            derivative_work=work) as ticket:
        ticket.consume(derivative_work=work)
        actual = concrete.graph.as_graph_def(add_shapes=True)
        serialized = source.SerializeToString(deterministic=True)
        if actual.SerializeToString(deterministic=True) != serialized:
            raise UnsupportedFeature("tensorflow_scalar_source_identity", "Captured scalar source identity changed.")
        if actual.library.ByteSize() or concrete.variables:
            raise UnsupportedFeature("tensorflow_scalar_source_safety", "Legacy scalar queries require a closed pure captured graph.")
        capture_hashes = []
        for tensor in captures:
            if not tf.is_tensor(tensor) or not hasattr(tensor, "numpy") or not (
                    tensor.dtype.is_floating or tensor.dtype.is_integer or tensor.dtype.is_complex or tensor.dtype == tf.bool):
                raise UnsupportedFeature("tensorflow_scalar_source_safety", "Legacy captures must be immutable eager numeric tensors, not live resources.")
            device = tf.DeviceSpec.from_string(tensor.device)
            if (device.job not in (None, "localhost") or device.replica not in (None, 0)
                    or device.task not in (None, 0) or device.device_type not in (None, "CPU")
                    or device.device_index not in (None, 0)):
                raise UnsupportedFeature("tensorflow_scalar_source_safety", "Legacy captures require local CPU placement.")
            # Read fixed source data only, never a host quantum result or jet.
            data = tensor.numpy()
            if tensor.dtype.is_floating or tensor.dtype.is_complex:
                if not np.isfinite(data).all():
                    raise UnsupportedFeature("tensorflow_scalar_source_domain", "Nonfinite captured constants cannot select a legacy query.")
                current_symbolic_limits().check("max_numeric_magnitude", float(np.max(np.abs(data), initial=0)))
            capture_hashes.append(hashlib.sha256((tensor.dtype.name+str(tuple(tensor.shape))).encode()+data.tobytes()).hexdigest())
        for op in operations:
            attrs = op.node_def.attr
            if (op.op_def.is_stateful or op.control_inputs or op.type in {"PyFunc", "PyFuncStateless", "EagerPyFunc"}
                    or "_gradient_op_type" in attrs
                    or any(a.WhichOneof("value") == "func" or a.WhichOneof("value") == "list" and a.list.func for a in attrs.values())):
                raise UnsupportedFeature("tensorflow_scalar_source_safety", "Opaque, stateful or callback scalar sources cannot select a legacy query.")
            if op.type == "Placeholder" and not any(op.outputs[0] is tensor for tensor in concrete.inputs):
                raise UnsupportedFeature("tensorflow_scalar_source_identity", "An unregistered scalar input cannot select a legacy query.")
            for tensor in op.outputs:
                dims = tuple(tensor.shape)
                bridge.check("max_scalar_rank", len(dims))
                for dim in dims:
                    bridge.check("max_scalar_elements", dim)
                bridge.check("max_scalar_elements", tensor.shape.num_elements())
            device = tf.DeviceSpec.from_string(op.device)
            if (device.job not in (None, "localhost") or device.replica not in (None, 0)
                    or device.task not in (None, 0) or device.device_type not in (None, "CPU")
                    or device.device_index not in (None, 0)):
                raise UnsupportedFeature("tensorflow_scalar_source_safety", "Legacy scalar queries require local CPU placement.")
            if op.type == "Const":
                data = tf.make_ndarray(op.get_attr("value"))
                floating = op.outputs[0].dtype.is_floating or op.outputs[0].dtype.is_complex
                if floating and not np.isfinite(data).all():
                    raise UnsupportedFeature("tensorflow_scalar_source_domain", "Nonfinite source constants cannot select a legacy query.")
                if floating:
                    current_symbolic_limits().check("max_numeric_magnitude", float(np.max(np.abs(data), initial=0)))
    return MappingProxyType(dict(construction_work=work, workspace_bytes=workspace,
        source_sha256=hashlib.sha256(serialized).hexdigest(), native_lowering=False,
        captured_constant_sha256=tuple(capture_hashes),
        execution="previous captured scalar query only; original source independently admitted"))


class ManagedTensorFlowQNode:
    """A mathematical snapshot, not a live Python-circuit or arbitrary TF runner.

    Each public call owns a normal native request ledger (or child of the
    active scope). Deferred eager AD retains that request. Raw internal
    ConcreteFunctions and external loss allocations are outside accounting.
    Cached outer reverse-over-forward AD is experimental and not accepted;
    its formal correctness failures remain tracked separately.
    """

    def __init__(self, node, *, input_signature, max_derivative_order=None, controlled_execution=True):
        try:
            import tensorflow as tf
        except ImportError as error:
            raise MissingOptionalDependency("Managed tensor execution requires optional TensorFlow.") from error
        self._synchronous()
        if set(node.gradient_kwargs)-{"step"}:
            raise UnsupportedFeature("tensorflow_gradient_options")
        step = node.gradient_kwargs.get("step", 1e-6)
        if isinstance(step, bool) or not isinstance(step, Real) or not np.isfinite(step) or step <= 0:
            raise ValueError("Native derivative step must be finite and positive.")
        if controlled_execution is not True:
            raise ValueError("This API requires an explicit controlled execution handle.")
        if not isinstance(input_signature, (tuple, list)) or len(input_signature) != 1:
            raise UnsupportedFeature("tensorflow_managed_signature", "Use one explicit real floating array input signature.")
        spec = input_signature[0]
        if not isinstance(spec, tf.TensorSpec) or spec.dtype != tf.float64 or not spec.shape.is_fully_defined():
            raise ValueError("Managed tensor input signatures require static float64 TensorSpecs.")
        order = node.max_diff if max_derivative_order is None else max_derivative_order
        if isinstance(order, bool) or not isinstance(order, Integral) or not 0 <= order <= min(2, node.max_diff):
            raise UnsupportedFeature("higher_order_derivatives", "A tensor handle must approve an available derivative order from zero through two.")
        self._shape, self._size, self._max_order = tuple(spec.shape), spec.shape.num_elements(), int(order)
        self._limits = resolve_unified_limits(node.unified_limits)
        self._owner, self._owner_configuration = weakref.ref(node), _configuration(node)
        self._lock, self._graphs = RLock(), {}
        self._closed = False
        self._stats = dict(graphs_built=0, attempted_calls=0, successful_calls=0, aborted_calls=0)
        self._last_metadata = {}
        bridge = node.tensorflow_limits or TensorFlowBridgeLimits()
        if not isinstance(bridge, TensorFlowBridgeLimits):
            raise TypeError("tensorflow_limits requires TensorFlowBridgeLimits.")
        bridge.check("max_input_elements", self._size)
        owner = copy(node)
        owner.max_diff = max(1, self._max_order)
        holders = []
        def prepare(point):
            source, recorder, program = owner._construct((tf.reshape(point, self._shape),), {},
                prepare_only=True, capture_native_parameters=True)
            kernel = NativeHostKernel(owner, source, recorder, program, bridge,
                interface="tensorflow", validate_derivatives=False)
            if kernel.program is not None or any(i is not None for refs in kernel.coefficients for i in refs):
                raise UnsupportedFeature("tensorflow_managed_transform_or_coefficients")
            packed = tf.stack([tf.convert_to_tensor(angle, tf.float64) for angle in kernel.angles]) if kernel.angles else tf.zeros((0,), tf.float64)
            captured = getattr(source, "_native_parameter_binding", None)
            binding = None
            partial_capture = False
            if captured is not None:
                from quantumbridge.core import Parameter, ParameterVectorElement
                from quantumbridge.diff.compiled_parameters import FrozenParameterBinding
                if (type(captured) is not tuple or len(captured) != 2
                        or type(captured[0]) is not FrozenParameterBinding
                        or type(captured[1]) is not tuple
                        or any(type(p) not in (Parameter,ParameterVectorElement) for p in captured[1])):
                    raise UnsupportedFeature("tensorflow_native_binding_owner")
                registered = tuple(recorder.parameters)
                if captured[1] != registered[:len(captured[1])]:
                    raise UnsupportedFeature("tensorflow_native_binding_identity")
                validated = FrozenParameterBinding.create(captured[0].parameters,
                    captured[0].expressions, limits=owner.higher_order_limits)
                if len(validated.expressions) != len(captured[1]):
                    raise UnsupportedFeature("tensorflow_native_binding_occurrences")
                partial_capture = len(captured[1]) != len(registered)
                if not partial_capture:
                    positions = {parameter: index for index, parameter in enumerate(registered)}
                    # Readmit the full owned map before projection, including
                    # formal inputs whose constant angles the kernel removed.
                    binding = replace(validated, expressions=tuple(
                        validated.expressions[positions[parameter]] for parameter in kernel.parameters))
            kernel.angles = []
            holders.append((kernel, binding, partial_capture))
            return packed
        with tf.device("/CPU:0"):
            self._pack = tf.function(prepare, input_signature=[tf.TensorSpec((self._size,), tf.float64)],
                autograph=False, jit_compile=False)
            pack_concrete = self._pack.get_concrete_function()
        if len(holders) != 1 or pack_concrete.variables:
            raise UnsupportedFeature("tensorflow_closure_variables", "All live variables must be explicit signature inputs.")
        self._packing_bytes = 0
        self._packing_nodes = len(pack_concrete.graph.get_operations())
        higher = resolve_higher_order_limits(owner.higher_order_limits)
        higher.check("max_parameter_graph_nodes", self._packing_nodes)
        for operation in pack_concrete.graph.get_operations():
            if operation.type in {"PartitionedCall", "StatefulPartitionedCall", "While", "StatelessWhile"} or operation.op_def.is_stateful:
                raise UnsupportedFeature("tensorflow_managed_source_graph", "The captured scalar mapping must be static, acyclic and free of callbacks, stateful operations and nested opaque functions.")
            for tensor in operation.outputs:
                if not tensor.shape.is_fully_defined() or not (tensor.dtype.is_floating or tensor.dtype.is_integer or tensor.dtype.is_complex or tensor.dtype == tf.bool):
                    raise UnsupportedFeature("tensorflow_managed_source_graph", "The captured mapping has an unbounded shape or nonnumeric storage.")
                self._packing_bytes += tensor.shape.num_elements()*tensor.dtype.size
                higher.check("max_memory_bytes", self._packing_bytes)
        self._kernel, self._frozen_binding, partial_capture = holders.pop()
        compile_ledger = self._ledger()
        self._scalar_lowering_work = None
        self._legacy_scalar_work = None
        if self._frozen_binding is None and not pack_concrete.captured_inputs:
            from quantumbridge.core import ParameterVector
            from .tensorflow_scalar_lowering import lower_scalar_graph, ScalarGraphCapabilityMiss
            scalar_source = pack_concrete.graph.as_graph_def(add_shapes=True)
            try:
                self._frozen_binding, self._scalar_lowering_work = lower_scalar_graph(
                    scalar_source,
                    input_name=pack_concrete.inputs[0].name, output_name=pack_concrete.outputs[0].name,
                    parameters=tuple(ParameterVector("managed_input", self._size)),
                    limits=higher, bridge_limits=bridge, ledger=compile_ledger, allow_square=True)
            except ScalarGraphCapabilityMiss as error:
                if partial_capture:
                    raise UnsupportedFeature("tensorflow_scalar_source_proof_unavailable",str(error)) from error
                self._legacy_scalar_work = _admit_legacy_scalar_query(pack_concrete, scalar_source,
                    ledger=compile_ledger, bridge=bridge, higher=higher)
        elif self._frozen_binding is None:
            if partial_capture:
                raise UnsupportedFeature("tensorflow_scalar_source_proof_unavailable",
                    "Partial native capture requires complete admitted scalar source data.")
            self._legacy_scalar_work = _admit_legacy_scalar_query(pack_concrete,
                pack_concrete.graph.as_graph_def(add_shapes=True),
                ledger=compile_ledger, bridge=bridge, higher=higher)
        else:
            from .tensorflow_scalar_lowering import lower_scalar_graph, ScalarGraphCapabilityMiss
            try:
                _,self._scalar_lowering_work = lower_scalar_graph(
                    pack_concrete.graph.as_graph_def(add_shapes=True),
                    input_name=pack_concrete.inputs[0].name,output_name=pack_concrete.outputs[0].name,
                    parameters=self._frozen_binding.parameters,limits=higher,bridge_limits=bridge,
                    ledger=compile_ledger,verify_occurrence_only=self._frozen_binding)
            except ScalarGraphCapabilityMiss as error:
                raise UnsupportedFeature("tensorflow_scalar_source_proof_unavailable",str(error)) from error
        if self._frozen_binding is not None:
            if len(self._frozen_binding.expressions) != self._kernel.size:
                raise UnsupportedFeature("tensorflow_native_binding_occurrences")
            self._packing_nodes = max(self._packing_nodes, self._frozen_binding.node_count)
            self._packing_bytes = max(self._packing_bytes, 128*self._frozen_binding.node_count)
        owner.fn = None
        self._plan = self._kernel.plan(np.ones(self._kernel.size))
        if tuple(self._plan.recorder.parameters) != tuple(self._kernel.parameters):
            raise UnsupportedFeature("tensorflow_native_binding_plan_identity")
        from quantumbridge.devices.matrix_product_state_device import MatrixProductStateDevice
        tensor_plan = TensorFlowStatevectorPlan
        if type(self._plan.node.device) is MatrixProductStateDevice:
            from .tensorflow_mps_execution import TensorFlowMPSPlan
            tensor_plan = TensorFlowMPSPlan
        self._whole = tensor_plan(self._plan, derivative_order=0,
            readonly=self._frozen_binding is not None)
        self._native_identity = self._whole.work["compiled_plan_identity"]
        self._snapshot_signature = self._whole.fingerprint()
        self._source_signature = self._whole.fingerprint(source_plan=self._plan)
        self._binding_signature = self._binding_fingerprint()
        self._packing_function = self._pack
        # Hash the captured scalar source as provenance. The restricted lowerer
        # imports data into native AST only; it never imports an execution graph.
        mapping = pack_concrete.graph.as_graph_def().SerializeToString(deterministic=True)
        captures = () if self._legacy_scalar_work is None else self._legacy_scalar_work["captured_constant_sha256"]
        self._identity = hashlib.sha256(json.dumps(dict(native=self._native_identity,
            snapshot=self._snapshot_signature, source=self._source_signature,
            binding=self._binding_signature, captured_constants=captures), sort_keys=True).encode()+mapping).hexdigest()
        self._outputs, self._spec = self._kernel.output_size, self._kernel.spec
        self._bridge = bridge
        self._execution_capabilities = MappingProxyType({mode: tensorflow_execution_capability(self._plan,
            execution_mode=mode, maximum_order=self._max_order) for mode in
            ("MANAGED_EXPLICIT_QUERY", "EXPERIMENTAL_CACHED_CALLABLE")})
        kinds = ("value", "vjp", "second_vjp")[:self._max_order+1]
        for kind in kinds:
            request = self._request("hvp" if kind == "second_vjp" else kind)
            work = self._operation_work(kind, request)
            with ResourceScope(compile_ledger, new_request=True) as scope:
                self._kernel.admit(scope)
                compile_ledger.preflight(memory_bytes=work["workspace_bytes"],
                    derivative_work=work["construction_work"]+work["planned_work"], tapes=work["logical_evaluations"])
                with compile_ledger.reserve("tensorflow_graph_construction", memory_bytes=work["workspace_bytes"],
                        derivative_work=work["construction_work"]) as ticket:
                    ticket.consume(derivative_work=work["construction_work"])
                    scope.tensorflow_tensor_workspace_admitted = work["workspace_bytes"]
                    self._graphs[(kind, 1, 1)] = self._build(kind, 1, 1)
                    self._stats["graphs_built"] += 1
        self._construction_receipt = compile_ledger.receipt()

    @staticmethod
    def _synchronous():
        import tensorflow as tf
        if not tf.executing_eagerly() or not tf.config.experimental.get_synchronous_execution():
            raise UnsupportedFeature("tensorflow_controlled_execution", "Managed graph invocation requires a synchronous eager CPU boundary.")
        if tf.distribute.has_strategy():
            raise UnsupportedFeature("tensorflow_distributed")

    def _check_owner(self):
        self._synchronous()
        if self._closed:
            raise UnsupportedFeature("tensorflow_compiled_closed", "This controlled graph handle is closed.")
        owner = self._owner()
        if owner is None:
            raise UnsupportedFeature("tensorflow_compiled_owner_expired", "The compiled owner no longer exists; its live resource contract cannot be verified.")
        if _configuration(owner) != self._owner_configuration:
            raise UnsupportedFeature("tensorflow_compiled_owner_changed", "The compiled owner execution/resource configuration changed; compile a new handle.")
        current = self._whole.work_planner(self._plan, self._whole._circuit, self._whole.input_size, order=0)
        if (current["compiled_plan_identity"] != self._native_identity
                or self._whole.fingerprint(source_plan=self._plan) != self._source_signature
                or self._whole.fingerprint() != self._snapshot_signature
                or self._binding_fingerprint() != self._binding_signature
                or self._pack is not self._packing_function):
            raise UnsupportedFeature("tensorflow_compiled_plan_changed", "The snapshot native plan identity changed.")

    def _binding_fingerprint(self):
        return "" if self._frozen_binding is None else json.dumps(self._frozen_binding.to_record(), sort_keys=True)

    def _input(self, point):
        import tensorflow as tf
        value = tf.convert_to_tensor(point, dtype_hint=tf.float64)
        if value.dtype != tf.float64 or tuple(value.shape) != self._shape:
            raise ValueError("Managed tensor inputs must match the approved shape and float64 dtype.")
        if any(label in value.device.upper() for label in ("GPU", "TPU")):
            raise UnsupportedFeature("tensorflow_device", "Managed tensor execution requires CPU inputs.")
        return tf.reshape(value, (self._size,))

    def _ledger(self, limits=None, existing=None):
        active = current_resources()
        if existing is not None:
            account = existing
            while account is not None and (active is None or active.ledger is not account):
                account = account.parent
            if active is not None and account is None:
                raise UnsupportedFeature("tensorflow_foreign_resource_scope", "A deferred request cannot bypass an unrelated active scope.")
            return existing
        requested = self._limits if limits is None else resolve_unified_limits(limits)
        effective = UnifiedExecutionLimits(**{name: min(value, getattr(requested, name)) for name, value in self._limits.__dict__.items()})
        return UnifiedResourceLedger(effective,
            parent=None if active is None else active.ledger)

    def _request(self, kind, *, source=None):
        order = 0 if kind == "value" else 2 if kind in {"hvp", "hessian"} else 1
        return DerivativeExecutionRequest(self if source is None else source, derivative_kind=kind,
            derivative_order=order, method=self._plan.node.diff_method, limits=self._limits)

    def _work(self, request, directions=1, cotangents=1):
        self._check_owner()
        order = request.validate_order()
        if order > self._max_order:
            raise UnsupportedFeature("higher_order_derivatives", "This handle has not approved the requested derivative order.")
        work = self._whole.work_planner(self._plan, self._whole._circuit, self._size,
            request=request, directions=directions, cotangents=cotangents)
        work = self._source_work(work)
        elements = self._outputs*self._size**2 if request.derivative_kind == "hessian" else self._outputs*self._size if request.derivative_kind == "jacobian" else max(self._outputs, self._size)*directions*cotangents
        self._bridge.check("max_jacobian_elements", elements)
        return work

    def _source_work(self, work):
        work["resource_plan_identity"] = work["compiled_plan_identity"]
        work["compiled_plan_identity"] = self._identity
        if work["request"] is not None:
            work["request"]["plan_identity"] = self._identity
        copies = work["maximum_quantum_graph_copies"]
        work["workspace_bytes"] += self._packing_bytes*copies
        work["planned_work"] += self._packing_nodes*copies
        work["construction_work"] += self._packing_nodes*copies
        work["source_mapping_workspace_bytes"] = self._packing_bytes*copies
        if self._scalar_lowering_work is not None:
            work["scalar_source_lowering"] = dict(self._scalar_lowering_work)
        if self._legacy_scalar_work is not None:
            work["scalar_source_legacy_query"] = dict(self._legacy_scalar_work)
        higher = resolve_higher_order_limits(self._plan.node.higher_order_limits)
        higher.check("max_parameter_graph_nodes", self._packing_nodes+work["parameter_graph_node_bound"])
        return self._check_work_limits(work)

    def _check_work_limits(self, work):
        higher = resolve_higher_order_limits(self._plan.node.higher_order_limits)
        higher.check("max_memory_bytes", work["workspace_bytes"])
        higher.check("max_adjoint_operations", work["planned_work"])
        higher.check("max_total_tapes", work["logical_evaluations"])
        device = self._plan.node.device
        cap = device.state_options["max_memory_mb"] if hasattr(device, "state_options") else device.max_memory_mb
        if cap is not None and work["workspace_bytes"] > cap*1024**2:
            raise MemoryError("Managed source mapping and tensor workspace exceed device max_memory_mb.")
        return work

    def _program(self, point):
        import tensorflow as tf
        valid = _finite_parameters()(point)
        point = tf.where(valid, point, tf.zeros_like(point))
        if self._frozen_binding is None:
            packed = self._pack(point)
        else:
            values = self._frozen_binding.evaluate(tuple(tf.unstack(point)))
            packed = tf.stack([tf.convert_to_tensor(value, tf.float64) for value in values]) if values else tf.zeros((0,), tf.float64)
        return self._whole.values(packed)+tf.cast(0., tf.float64)*tf.reduce_sum(point)

    def _operation_work(self, kind, request, directions=1, cotangents=1):
        work = self._work(request, directions, cotangents)
        if kind == "second_vjp":
            work = dict(work, planned_work=work["planned_work"]*2,
                construction_work=work["construction_work"]*2,
                workspace_bytes=work["workspace_bytes"]*2,
                logical_evaluations=work["logical_evaluations"]+directions)
            work = self._check_work_limits(work)
        return work

    def _build(self, kind, directions, cotangents):
        import tensorflow as tf
        shapes = [(self._size,)] if kind == "value" else [(self._size,), (directions, self._size)] if kind == "jvp" else [(self._size,), (cotangents, self._outputs)] if kind in {"vjp", "gradient", "jacobian"} else [(self._size,), (cotangents, self._outputs), (directions, self._size), (self._size,)] if kind == "second_vjp" else [(self._size,), (cotangents, self._outputs), (directions, self._size)]
        sizes = [int(np.prod(shape)) for shape in shapes]
        signature = [tf.TensorSpec((sum(sizes),), tf.float64)]
        value_graph = self._graphs.get(("value", 1, 1))
        first_graph = self._graphs.get(("vjp", 1, 1))
        def program(point):
            return self._program(point) if value_graph is None else value_graph(point)[0]
        def execute(packet):
            x, *extra = [tf.reshape(piece, shape) for piece, shape in zip(tf.split(packet, sizes), shapes)]
            def checked(seed):
                return tf.where(_finite_parameters()(seed), seed, tf.zeros_like(seed))
            if kind == "value":
                return (self._program(x),)
            if kind == "jvp":
                values, products = None, []
                for direction in tf.unstack(checked(extra[0])):
                    with tf.autodiff.ForwardAccumulator(x, direction) as forward:
                        values = program(x)
                    projected = forward.jvp(values)
                    products.append(tf.zeros_like(values) if projected is None else projected)
                return values, tf.stack(products)
            weights = checked(extra[0])
            if kind in {"vjp", "gradient", "jacobian"}:
                products = []
                for weight in tf.unstack(weights):
                    with tf.GradientTape() as reverse:
                        reverse.watch(x)
                        values = program(x)
                    products.append(reverse.gradient(values, x, output_gradients=weight, unconnected_gradients=tf.UnconnectedGradients.ZERO))
                return values, tf.stack(products)
            vectors = checked(extra[1])
            target = extra[2] if kind == "second_vjp" else x
            products = []
            for weight in tf.unstack(weights):
                row = []
                for direction in tf.unstack(vectors):
                    with tf.autodiff.ForwardAccumulator(target, direction) as forward:
                        if first_graph is None:
                            with tf.GradientTape() as reverse:
                                reverse.watch(target)
                                values = program(target)
                            gradient = reverse.gradient(values, target, output_gradients=weight, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                        else:
                            values, gradients = first_graph(tf.concat([target, tf.reshape(weight, (-1,))], 0))
                            gradient = gradients[0]
                    product = forward.jvp(gradient)
                    row.append(tf.zeros_like(target) if product is None else product)
                products.append(tf.stack(row))
            result = tf.stack(products)
            if kind == "second_vjp":
                projected = []
                for direction in tf.unstack(vectors):
                    with tf.autodiff.ForwardAccumulator(x, direction) as forward:
                        values = program(x)
                    product = forward.jvp(values)
                    projected.append(tf.zeros_like(values) if product is None else product)
                return values, result, tf.stack(projected)
            return values, result
        with tf.device("/CPU:0"):
            function = tf.function(execute, input_signature=signature, autograph=False, jit_compile=False)
            concrete = function.get_concrete_function()
        if concrete.variables:
            raise UnsupportedFeature("tensorflow_closure_variables")
        return function

    def _invoke(self, kind, point, ledger, *, cotangent=None, tangent=None, guarded=None, request=None, approved_order=None,
            execution_mode="MANAGED_EXPLICIT_QUERY"):
        import tensorflow as tf
        ledger = self._ledger(existing=ledger)
        directions = 1 if tangent is None else int(tangent.shape[0])
        cotangents = 1 if cotangent is None else int(cotangent.shape[0])
        request = request or self._request("hvp" if kind == "second_vjp" else kind)
        work = self._operation_work(kind, request, directions, cotangents)
        if approved_order is not None:
            envelope = self._source_work(self._whole.work_planner(self._plan, self._whole._circuit,
                self._size, order=approved_order))
            work = dict(work, workspace_bytes=envelope["workspace_bytes"], planned_work=envelope["planned_work"],
                planning_mode="approved_callable_order_envelope", envelope_order=approved_order,
                tape_reservation=envelope["logical_evaluations"])
        key = (kind, directions, cotangents)
        ticket = None
        ledger.unavailable("tensorflow_allocator_and_retained_outputs", "Only conservative invocation workspace is reserved; TF allocator/RSS, retained result tensors and external loss operations are not measured.")
        try:
            with ResourceScope(ledger, new_request=True) as scope:
                self._kernel.admit(scope)
                with self._lock:
                    if self._closed:
                        raise UnsupportedFeature("tensorflow_compiled_closed", "This controlled graph handle is closed.")
                    build = key not in self._graphs
                    amount = work["planned_work"]+(work["construction_work"] if build else 0)
                    ticket = ledger.reserve("tensorflow_managed_invocation", memory_bytes=work["workspace_bytes"],
                        derivative_work=amount, tapes=work.get("tape_reservation", work["logical_evaluations"]))
                    scope.tensorflow_tensor_workspace_admitted = work["workspace_bytes"]
                    if build:
                        ticket.consume(derivative_work=work["construction_work"])
                        self._graphs[key] = self._build(kind, directions, cotangents)
                        self._stats["graphs_built"] += 1
                    concrete = self._graphs[key]
                    ticket.consume(derivative_work=work["planned_work"])
                    self._stats["attempted_calls"] += 1
                args = (point,) if kind == "value" else (point, tangent) if kind == "jvp" else (point, cotangent) if kind in {"gradient", "vjp", "jacobian"} else (point, cotangent, tangent, guarded) if kind == "second_vjp" else (point, cotangent, tangent)
                with tf.device("/CPU:0"):
                    result = self._run_graph(concrete, tf.concat([tf.reshape(arg, (-1,)) for arg in args], 0))
                ticket.consume(tapes=work["logical_evaluations"])
                with self._lock:
                    self._stats["successful_calls"] += 1
                return result, work
        except BaseException:
            with self._lock:
                self._stats["aborted_calls"] += 1
            raise
        finally:
            if ticket is not None:
                ticket.release()
            with self._lock:
                self._last_metadata = self._metadata(work, ledger, execution_mode)

    def _metadata(self, work, ledger, execution_mode="MANAGED_EXPLICIT_QUERY"):
        return dict(controlled_execution="LEDGER_ACCOUNTED", plan_identity=self._identity,
            execution_mode=execution_mode, execution_capability=dict(self._execution_capabilities[execution_mode]),
            ownership="snapshot mathematics; live owner required for resource/execution configuration guards",
            tensor_plan_work=deepcopy(work), receipt=ledger.receipt(),
            request_limits=dict(ledger.limits.__dict__), parent_budget_enforced=ledger.parent is not None,
            physical_shots=0, host_quantum_calls=0, device_updates=0,
            direct_concrete_function="OUT_OF_SCOPE", allocator_memory="ESTIMATED_ONLY", external_loss="OUT_OF_SCOPE",
            cached_outer_reverse_over_forward="EXPERIMENTAL_NOT_ACCEPTED")

    @staticmethod
    def _run_graph(function, packet):
        result = function(packet)
        # Synchronous materialization only confirms completion and finiteness;
        # it never supplies a host-computed quantum result back to TensorFlow.
        for output in result:
            if not np.all(np.isfinite(output.numpy())):
                raise FloatingPointError("Managed tensor output is nonfinite.")
        return result

    def __call__(self, point):
        """Legacy diagnostic callable; arbitrary mixed AD is not verified."""
        return self._callable(point, self._max_order)

    def differentiable_call(self, point, *, derivative_order=1, external_ad="eager", method="best"):
        """Controlled declared first-order AD; not implicit nesting detection."""
        kind = "value" if derivative_order == 0 else "jacobian" if derivative_order == 1 else "hvp"
        request = DerivativeExecutionRequest(self, point, derivative_kind=kind,
            derivative_order=derivative_order, method=method)
        self.require_derivative_contract(request, execution_mode="EXPERIMENTAL_CACHED_CALLABLE", external_ad=external_ad)
        return self._callable(point, derivative_order)

    @property
    def execution_capabilities(self):
        """Static verified contracts, not runtime input or budget admission."""
        return self._execution_capabilities

    def require_derivative_contract(self, request, *, execution_mode="MANAGED_EXPLICIT_QUERY", external_ad="none"):
        self._validate_request(request)
        if execution_mode not in self._execution_capabilities:
            raise UnsupportedFeature("tensorflow_execution_mode", "This handle does not own the requested execution mode.")
        contract = require_tensorflow_derivative_contract(request,
            self._execution_capabilities[execution_mode], external_ad=external_ad)
        if execution_mode == "MANAGED_EXPLICIT_QUERY":
            self._query_seeds(request)
        self._check_owner()
        return contract

    def supports_derivatives(self, request, **options):
        """Check the declared static contract; execution still admits budgets."""
        try:
            self.require_derivative_contract(request, **options)
        except (UnsupportedFeature, ValueError, TypeError, MemoryError):
            return False
        return True

    def _callable(self, point, approved_order):
        import tensorflow as tf
        self._check_owner()
        point = self._input(point)
        ledger = self._ledger()
        if self._frozen_binding is not None:
            def program(x):
                return self._run_graph(lambda p: (self._program(p),), x)[0]
            flat = bounded_tensor_function(program,
                admission=lambda order: self._inline_admission(order, ledger, approved_order),
                max_order=approved_order)(point)
            return unflatten_tree(flat, self._spec, reshape=tf.reshape)
        def primal(x):
            return self._invoke("value", x, ledger, approved_order=approved_order,
                execution_mode="EXPERIMENTAL_CACHED_CALLABLE")[0][0]
        def first(x, cotangent):
            return self._invoke("vjp", x, ledger, cotangent=tf.reshape(cotangent, (1, self._outputs)),
                execution_mode="EXPERIMENTAL_CACHED_CALLABLE")[0][1][0]
        def second(guarded, x, cotangent, direction):
            values, _ = self._invoke("second_vjp", x, ledger, guarded=guarded,
                cotangent=tf.reshape(cotangent, (1, self._outputs)), tangent=tf.reshape(direction, (1, self._size)),
                execution_mode="EXPERIMENTAL_CACHED_CALLABLE")
            return values[1][0, 0], values[2][0]
        flat = bounded_tensor_function(primal, first_rule=first, second_rule=second, max_order=approved_order)(point)
        return unflatten_tree(flat, self._spec, reshape=tf.reshape)

    @contextmanager
    def _inline_admission(self, order, ledger, approved_order):
        import tensorflow as tf
        ledger = self._ledger(existing=ledger)
        kind = ("value", "vjp", "second_vjp")[order]
        request = self._request("hvp" if order == 2 else kind)
        work = self._operation_work(kind, request)
        if order == 0:
            envelope = self._source_work(self._whole.work_planner(self._plan, self._whole._circuit,
                self._size, order=approved_order))
            work = dict(work, workspace_bytes=envelope["workspace_bytes"], planned_work=envelope["planned_work"],
                planning_mode="approved_callable_order_envelope", envelope_order=approved_order,
                tape_reservation=envelope["logical_evaluations"])
        ticket = None
        try:
            with ResourceScope(ledger, new_request=True) as scope:
                self._kernel.admit(scope)
                with self._lock:
                    if self._closed:
                        raise UnsupportedFeature("tensorflow_compiled_closed", "This controlled graph handle is closed.")
                    amount = work["planned_work"]+work["construction_work"]
                    ticket = ledger.reserve("tensorflow_managed_inline_invocation", memory_bytes=work["workspace_bytes"],
                        derivative_work=amount, tapes=work.get("tape_reservation", work["logical_evaluations"]))
                    scope.tensorflow_tensor_workspace_admitted = work["workspace_bytes"]
                    ticket.consume(derivative_work=amount)
                    self._stats["attempted_calls"] += 1
                with tf.device("/CPU:0"):
                    yield
                ticket.consume(tapes=work["logical_evaluations"])
                with self._lock:
                    self._stats["successful_calls"] += 1
        except BaseException:
            with self._lock:
                self._stats["aborted_calls"] += 1
            raise
        finally:
            if ticket is not None:
                ticket.release()
            with self._lock:
                self._last_metadata = dict(self._metadata(work, ledger, "EXPERIMENTAL_CACHED_CALLABLE"),
                    callable_math_boundary="frozen native inline; no inner cached quantum/scalar execution",
                    work_accounting="conservative synchronous invocation and construction bounds")

    def _seed(self, value, width, name, *, input_shaped=False):
        import tensorflow as tf
        if not (tf.is_tensor(value) or isinstance(value, (tf.Variable, np.ndarray, tuple, list, Real))):
            raise TypeError(name+" requires a bounded tensor/array, not an iterator or arbitrary provider.")
        if isinstance(value, (tuple, list)) and len(value) > resolve_higher_order_limits(self._plan.node.higher_order_limits).max_total_tapes:
            raise MemoryError("Tensor derivative seed count exceeds max_total_tapes.")
        result = tf.convert_to_tensor(value, dtype_hint=tf.float64)
        if input_shaped:
            if result.shape == self._shape:
                result = tf.reshape(result, (1, width))
            elif result.shape.rank == len(self._shape)+1 and tuple(result.shape)[1:] == self._shape:
                result = tf.reshape(result, (result.shape[0], width))
        elif result.shape.rank == 0 and width == 1:
            result = tf.reshape(result, (1, 1))
        if result.dtype != tf.float64 or result.shape.rank not in {1, 2} or not result.shape.is_fully_defined() or result.shape[-1] != width:
            raise ValueError(name+" must be a static float64 vector or bounded leading-axis batch.")
        if not result.shape.num_elements():
            raise ValueError(name+" seed batch must be nonempty.")
        return tf.reshape(result, (-1, width))

    def _query_seeds(self, request):
        kind = request.derivative_kind
        for name, required, allowed in (("tangent", kind in {"jvp", "hvp"}, kind in {"jvp", "hvp"}),
                ("cotangent", kind == "vjp", kind in {"vjp", "hvp"})):
            value = getattr(request, name)
            if required and value is None:
                raise ValueError(kind+" requires an explicit "+name+" seed.")
            if not allowed and value is not None:
                raise ValueError(kind+" does not accept a "+name+" seed.")
        tangent = None if request.tangent is None else self._seed(request.tangent, self._size, "tangent", input_shaped=True)
        cotangent = None if request.cotangent is None else self._seed(request.cotangent, self._outputs, "cotangent")
        directions = self._size if kind == "hessian" else int(tangent.shape[0]) if tangent is not None else 1
        cotangents = self._outputs if kind in {"jacobian", "hessian"} or kind == "hvp" and cotangent is None else int(cotangent.shape[0]) if cotangent is not None else 1
        return directions, cotangents, tangent, cotangent

    def estimate(self, request):
        """Inspect request bounds without building or executing another graph."""
        self._validate_request(request)
        kind = request.derivative_kind
        directions, cotangents, _, _ = self._query_seeds(request)
        work = self._work(request, directions, cotangents)
        with self._lock:
            build = (kind, directions, cotangents) not in self._graphs
        return dict(work, first_call_work=work["planned_work"]+(work["construction_work"] if build else 0),
            required_memory_bytes=work["workspace_bytes"]+self._kernel.storage_owner.nbytes)

    def _validate_request(self, request):
        if not isinstance(request, DerivativeExecutionRequest):
            raise TypeError("Managed execution requires DerivativeExecutionRequest.")
        owner = self._owner()
        if request.circuit_or_tape is not self and (owner is None or request.circuit_or_tape is not owner):
            raise ValueError("The derivative request belongs to another compiled owner.")
        if request.shots is not None or request.seed is not None or request.observable is not None:
            raise UnsupportedFeature("tensorflow_managed_request_override", "Compile another snapshot to change measurements or sampling.")
        request.validate_order()

    def execute(self, request):
        """Execute an explicit query, not a composable external AD output."""
        import tensorflow as tf
        self._validate_request(request)
        self._check_owner()
        kind = request.derivative_kind
        directions, cotangents, tangent, cotangent = self._query_seeds(request)
        self._work(request, directions, cotangents)
        point = self._input(request.parameters[0] if isinstance(request.parameters, tuple) and len(request.parameters) == 1 else request.parameters)
        if kind in {"jacobian", "hessian"} or kind == "hvp" and cotangent is None:
            cotangent = tf.eye(self._outputs, dtype=tf.float64)
        elif kind == "gradient":
            cotangent = tf.ones((1, 1), tf.float64)
        if kind == "hessian":
            tangent = tf.eye(self._size, dtype=tf.float64)
        ledger = self._ledger(request.limits)
        raw, work = self._invoke(kind, point, ledger, cotangent=cotangent, tangent=tangent, request=request)
        @tf.custom_gradient
        def query(value):
            def refuse(incoming):
                raise UnsupportedFeature("tensorflow_query_autodiff", "Explicit results are queries. Use checked differentiable_call for declared first-order AD, or the verified inline subset for co-traced second-order AD; cached callable second-order AD is experimental.")
            return tf.identity(value), refuse
        value = unflatten_tree(query(raw[0]), self._spec, reshape=tf.reshape)
        derivative = None
        if kind != "value":
            payload = query(raw[1])
            if kind in {"vjp", "gradient"}:
                derivative = tf.reshape(payload[0], self._shape) if cotangents == 1 else tf.reshape(payload, (cotangents,)+self._shape)
            elif kind == "jacobian":
                derivative = unflatten_tree(payload, self._spec, self._shape, reshape=tf.reshape)
            elif kind == "hessian":
                derivative = unflatten_tree(tf.transpose(payload, (0, 2, 1)), self._spec, self._shape+self._shape, reshape=tf.reshape)
            elif kind == "jvp":
                derivative = unflatten_tree(payload[0], self._spec, reshape=tf.reshape) if directions == 1 else tuple(unflatten_tree(item, self._spec, reshape=tf.reshape) for item in tf.unstack(payload))
            elif request.cotangent is not None:
                derivative = tf.reshape(payload[0, 0], self._shape) if cotangents == directions == 1 else tf.reshape(payload, (cotangents, directions)+self._shape)
            else:
                derivative = unflatten_tree(payload[:, 0], self._spec, self._shape, reshape=tf.reshape) if directions == 1 else tuple(unflatten_tree(payload[:, index], self._spec, self._shape, reshape=tf.reshape) for index in range(directions))
        return DerivativeExecutionResult(value=value, derivative=derivative, receipt=ledger.receipt(),
            metadata={**self._metadata(work, ledger), "request": dict(work["request"]),
                "query_autodiff": "UNSUPPORTED; checked differentiable_call is first-order only; verified inline second-order is a separate contract"})

    @property
    def last_metadata(self):
        with self._lock:
            return deepcopy(self._last_metadata)

    def cache_info(self):
        with self._lock:
            return dict(self._stats, graph_keys=tuple(self._graphs), plan_identity=self._identity, closed=self._closed,
                trace_counts=tuple((key, function.experimental_get_tracing_count()) for key, function in self._graphs.items()),
                packing_trace_count=self._pack.experimental_get_tracing_count(),
                construction_receipt=deepcopy(self._construction_receipt))

    def close(self):
        """Reject new invocations; already admitted synchronous calls may finish."""
        with self._lock:
            self._closed = True
            self._graphs.clear()

    @property
    def concrete_function(self):
        """Raw frozen value graph; direct invocation is OUT_OF_SCOPE."""
        with self._lock:
            if ("value", 1, 1) not in self._graphs:
                raise RuntimeError("First invoke the managed value entry to build its graph.")
            return self._graphs[("value", 1, 1)].get_concrete_function()
