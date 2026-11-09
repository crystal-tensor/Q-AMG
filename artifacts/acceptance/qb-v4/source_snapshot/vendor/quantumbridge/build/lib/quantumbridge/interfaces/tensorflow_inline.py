# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Explicit static-bounded native math emitted into the caller's TF graph."""

from copy import copy, deepcopy
from numbers import Integral
from types import MappingProxyType
import hashlib
import json
import weakref

import numpy as np

from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.diff.compiled_parameters import FrozenParameterBinding
from quantumbridge.diff.higher_order_limits import resolve_higher_order_limits
from quantumbridge.diff.qnode_engine import AngleRecorder, ExecutionPlan
from quantumbridge.diff.trace import flatten_tree, unflatten_tree
from quantumbridge.execution.derivative_planning import (require_tensorflow_derivative_contract,
    tensorflow_execution_capability, tensorflow_statevector_work)
from quantumbridge.execution.derivative_request import DerivativeExecutionRequest
from quantumbridge.execution.persistent_storage import PersistentStoragePool
from quantumbridge.execution.resource_context import current_resources
from quantumbridge.execution.resource_ledger import UnifiedResourceLedger
from quantumbridge.execution.resource_limits import resolve_unified_limits
from .tensorflow_autodiff import TensorFlowBridgeLimits
from .tensorflow_managed import _configuration
from .tensorflow_native_derivatives import bounded_tensor_function
from .tensorflow_native_execution import _finite_parameters
from .tensorflow_whole_execution import TensorFlowStatevectorPlan


class TraceInlineTensorFlowQNode:
    """Snapshot mathematics; no internal cached trainable execution function.

    A user's independently cached outer graph does not reenter Python. Its
    resource accounting and post-trace owner/config checks are OUT_OF_SCOPE.
    Explicit managed cached queries remain a separate execution contract.
    """

    def __init__(self, node, *, parameters, input_signature, max_derivative_order):
        import tensorflow as tf
        self._scope()
        if not tf.executing_eagerly():
            raise UnsupportedFeature("tensorflow_inline_construction", "Compile an inline snapshot at an explicit eager boundary.")
        from quantumbridge.devices.statevector_device import StatevectorDevice
        if type(node.device) is not StatevectorDevice or not 1 <= node.num_qubits <= 4:
            raise UnsupportedFeature("tensorflow_inline_device", "The native dense TF inline specialization requires a one-through-four-qubit StatevectorDevice.")
        if type(parameters) not in {tuple, list}:
            raise TypeError("Inline parameter identities require a bounded sequence.")
        if type(input_signature) not in {tuple, list} or len(input_signature) != 1:
            raise ValueError("Inline execution requires one explicit float64 input signature.")
        spec = input_signature[0]
        if not isinstance(spec, tf.TensorSpec) or spec.dtype != tf.float64 or not spec.shape.is_fully_defined():
            raise ValueError("Inline inputs require a static float64 TensorSpec.")
        if spec.shape.num_elements() != len(parameters):
            raise ValueError("Inline signature size must match the native input UUID order.")
        order = max_derivative_order
        if isinstance(order, bool) or not isinstance(order, Integral) or not 0 <= order <= min(2, node.max_diff):
            raise UnsupportedFeature("higher_order_derivatives", "Inline derivative order must be approved by the QNode.")
        if node.diff_method not in {"best", "adjoint"} or node.gradient_kwargs:
            raise UnsupportedFeature("tensorflow_inline_method", "Inline math requires the native best/adjoint tensor strategy without gradient overrides.")
        bridge = node.tensorflow_limits or TensorFlowBridgeLimits()
        if not isinstance(bridge, TensorFlowBridgeLimits):
            raise TypeError("tensorflow_limits requires TensorFlowBridgeLimits.")
        higher = resolve_higher_order_limits(node.higher_order_limits)
        bridge.check("max_input_elements", len(parameters))
        higher.check("max_input_parameters", len(parameters))
        higher.check("max_memory_bytes", 8*len(parameters))
        FrozenParameterBinding.create(parameters, tuple(parameters), limits=higher)
        limits = resolve_unified_limits(node.unified_limits)
        # Reject impossible budgets before even constructing symbolic source.
        UnifiedResourceLedger(limits).preflight(memory_bytes=8*len(parameters), derivative_work=1, tapes=1)
        self._shape, self._size, self._order = tuple(spec.shape), len(parameters), int(order)
        self._owner, self._config = weakref.ref(node), _configuration(node)
        try:
            source, recorder, program = node._construct((np.array(parameters, dtype=object).reshape(self._shape),), {}, prepare_only=True)
        except (TypeError, AttributeError, ValueError) as error:
            from quantumbridge.core.symbolic_limits import SymbolicResourceLimitError
            if isinstance(error, SymbolicResourceLimitError):
                raise
            raise UnsupportedFeature("tensorflow_inline_symbolic_source", "The source must express its scalar mapping with native parameter syntax, not foreign tensor functions or callbacks.") from error
        if program is not None:
            raise UnsupportedFeature("tensorflow_inline_transforms", "A frozen inline transform binding is not validated.")
        self._binding = FrozenParameterBinding.create(parameters, tuple(recorder.angles), limits=higher)
        self._storage = PersistentStoragePool(limits).freeze((source.circuit, source.output_tree, node.device, self._binding))
        circuit, tree, device, self._binding = self._storage.payload
        owner = copy(node)
        owner.fn, owner.device = None, device
        native = AngleRecorder()
        native.parameters, native.angles = list(recorder.parameters), [1.]*len(recorder.parameters)
        self._plan = ExecutionPlan(owner, source, tree, native)
        self._plan.circuit = circuit
        self._whole = TensorFlowStatevectorPlan(self._plan, derivative_order=order, readonly=True)
        self._spec = flatten_tree(self._output_shape(tree))[1]
        self._work = tensorflow_statevector_work(self._plan, self._whole._circuit, self._size, order=order)
        # Also bound the explicit complex-state view with real/imag outputs.
        outputs = 2*(1 << node.num_qubits)
        higher.check("max_output_elements", outputs)
        copies = self._work["maximum_quantum_graph_copies"]
        factor = max(1, (outputs+self._work["outputs"]-1)//self._work["outputs"])
        self._work["workspace_bytes"] = self._work["workspace_bytes"]*factor+16*self._binding.node_count*copies
        self._work["planned_work"] = self._work["planned_work"]*factor+self._binding.node_count*copies
        higher.check("max_parameter_graph_nodes", self._binding.node_count+self._work["parameter_graph_node_bound"])
        higher.check("max_memory_bytes", self._work["workspace_bytes"])
        higher.check("max_adjoint_operations", self._work["planned_work"])
        if device.max_memory_mb is not None and self._work["workspace_bytes"] > device.max_memory_mb*1024**2:
            raise MemoryError("Inline snapshot exceeds device max_memory_mb.")
        UnifiedResourceLedger(limits).preflight(memory_bytes=self._work["workspace_bytes"]+self._storage.nbytes,
            derivative_work=self._work["planned_work"], tapes=self._work["logical_evaluations"])
        self._work["cache_policy"] = "frozen native scalar/operation snapshot; caller owns external graph caching and AD"
        self._signature = self._fingerprint()
        identity = hashlib.sha256((self._work["compiled_plan_identity"]+self._signature).encode()).hexdigest()
        self._work["resource_plan_identity"] = self._work["compiled_plan_identity"]
        self._work["compiled_plan_identity"] = identity
        self._execution_capabilities = MappingProxyType({"TRACE_INLINE_DIFFERENTIABLE":
            tensorflow_execution_capability(self._plan, execution_mode="TRACE_INLINE_DIFFERENTIABLE", maximum_order=self._order)})
        self._metadata = dict(execution_mode="TF_TRACE_INLINE_DIFFERENTIABLE_LOW_QUBIT_SUBSET",
            controlled_execution="OUT_OF_SCOPE", external_graph_execution="OUT_OF_SCOPE",
            autodiff_scope="eager inline or AD containers traced together with inline operations; mixed AD outside an already cached outer value graph NOT_ACCEPTED",
            owner_checks="Python entry/tracing only; direct cached outer calls do not recheck the owner",
            allocator_memory="ESTIMATED_ONLY", budget=deepcopy(self._work),
            parameter_uuids=self._binding.parameter_uuids, snapshot=True, plan_identity=identity,
            cached_inner_rof="UNRESOLVED; separate managed cached-callable mode")
        self._metadata["execution_capability"] = dict(self._execution_capabilities["TRACE_INLINE_DIFFERENTIABLE"])

    @staticmethod
    def _output_shape(tree):
        from quantumbridge.diff.qnode_engine import map_measurements
        return map_measurements(tree, lambda request: 0.)

    @staticmethod
    def _scope():
        if current_resources() is not None:
            raise UnsupportedFeature("tensorflow_inline_resource_scope", "Trace-inline external graphs cannot guarantee per-execution shared-ledger accounting; use explicit managed queries.")

    def _fingerprint(self):
        return json.dumps(dict(math_store=self._whole.fingerprint(),
            binding=self._binding.to_record()), sort_keys=True)

    def _check_owner(self):
        self._scope()
        owner = self._owner()
        if owner is None or _configuration(owner) != self._config:
            raise UnsupportedFeature("tensorflow_inline_owner", "The inline owner expired or execution/resource configuration changed.")
        if self._fingerprint() != self._signature:
            raise UnsupportedFeature("tensorflow_inline_plan_identity", "The frozen native operation store changed.")

    def _input(self, point):
        import tensorflow as tf
        self._check_owner()
        if tf.distribute.has_strategy():
            raise UnsupportedFeature("tensorflow_distributed")
        value = tf.convert_to_tensor(point, dtype_hint=tf.float64)
        if value.dtype != tf.float64 or tuple(value.shape) != self._shape:
            raise ValueError("Inline input must match the static float64 shape and UUID order.")
        if any(device in value.device.upper() for device in ("GPU", "TPU")):
            raise UnsupportedFeature("tensorflow_device")
        return tf.reshape(value, (self._size,))

    def _pack_values(self, point):
        import tensorflow as tf
        point = tf.where(_finite_parameters()(point), point, tf.zeros_like(point))
        values = self._binding.evaluate(tuple(tf.unstack(point)))
        return tf.stack([tf.convert_to_tensor(value, tf.float64) for value in values]) if values else tf.zeros((0,), tf.float64)

    def __call__(self, point):
        import tensorflow as tf
        point = self._input(point)
        function = bounded_tensor_function(lambda x: self._whole.values(self._pack_values(x)), max_order=self._order)
        return unflatten_tree(function(point), self._spec, reshape=tf.reshape)

    def state(self, point):
        import tensorflow as tf
        point = self._input(point)
        def state_parts(x):
            value = self._whole.state(self._pack_values(x))
            return tf.concat([tf.math.real(value), tf.math.imag(value)], 0)
        parts = bounded_tensor_function(state_parts, max_order=self._order)(point)
        width = 1 << self._whole.width
        return tf.complex(parts[:width], parts[width:])

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    @property
    def execution_capabilities(self):
        """Static verified contracts, not runtime input or budget admission."""
        return self._execution_capabilities

    def require_derivative_contract(self, request, *, execution_mode="TRACE_INLINE_DIFFERENTIABLE", external_ad="none"):
        if not isinstance(request, DerivativeExecutionRequest):
            raise TypeError("A derivative contract requires DerivativeExecutionRequest.")
        if request.circuit_or_tape is not self and request.circuit_or_tape is not self._owner():
            raise ValueError("The derivative contract belongs to another inline owner.")
        if execution_mode not in self._execution_capabilities:
            raise UnsupportedFeature("tensorflow_execution_mode")
        if any(value is not None for value in (request.shots, request.seed, request.observable, request.limits)):
            raise UnsupportedFeature("tensorflow_inline_request_override",
                "Inline plans cannot honor per-request shots, seed, observable or limit overrides; recompile or use managed explicit queries.")
        contract = require_tensorflow_derivative_contract(request,
            self._execution_capabilities[execution_mode], external_ad=external_ad)
        self._check_owner()
        return contract

    def supports_derivatives(self, request, **options):
        """Check the declared static contract; execution still validates values."""
        try:
            self.require_derivative_contract(request, **options)
        except (UnsupportedFeature, ValueError, TypeError, MemoryError):
            return False
        return True
