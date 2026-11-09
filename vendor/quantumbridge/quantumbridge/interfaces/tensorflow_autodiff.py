# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Optional CPU TensorFlow graphs with native quantum first/second VJPs."""

from copy import copy
from dataclasses import dataclass, replace
from numbers import Integral, Real

import numpy as np

from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.diff.trace import unflatten_tree
from .native_host import MissingOptionalDependency, NativeHostKernel
from .qnode_autodiff import _leaves
from .tensorflow_callbacks import ordinary_status_code


@dataclass(frozen=True)
class TensorFlowBridgeLimits:
    max_input_elements: int = 1024
    max_output_elements: int = 100000
    max_jacobian_elements: int = 2000000
    max_parameter_elements: int = 1024
    max_batch_size: int = 128
    max_callback_executions: int = 128
    max_total_shots: int = 1000000
    max_transform_tapes: int = 128
    max_scalar_graph_bytes: int = 1048576
    max_scalar_rank: int = 8
    max_scalar_elements: int = 4096

    def __post_init__(self):
        if any(isinstance(x, bool) or not isinstance(x, Integral) or x < 1 for x in self.__dict__.values()):
            raise ValueError("TensorFlow bridge limits require positive integers.")

    def check(self, name, value):
        if value > getattr(self, name):
            raise MemoryError("Native TensorFlow bridge exceeded " + name)


def execute_tensorflow(node, args, kwargs, *, return_status=False):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise MissingOptionalDependency("Native TensorFlow QNode requires optional tensorflow.") from exc
    managed_limits = getattr(node, "unified_limits", None)
    if managed_limits is not None:
        from quantumbridge.execution.resource_limits import resolve_unified_limits
        resolve_unified_limits(managed_limits)
        if not tf.executing_eagerly() or return_status:
            raise UnsupportedFeature("tensorflow_unified_graph_resources", "Cached TensorFlow graph callbacks do not have a shared per-execution request account; only the admitted eager native tensor domain supports unified_limits.")
    if node.max_diff not in {1, 2}:
        raise UnsupportedFeature("higher_order_derivatives", "Native TensorFlow supports derivative orders at most two.")
    if set(node.gradient_kwargs) - {"step"}:
        raise UnsupportedFeature("tensorflow_gradient_options", "Use diff_method; gradient_kwargs supports only a static step.")
    step = node.gradient_kwargs.get("step", 1e-6)
    if isinstance(step, bool) or not isinstance(step, Real) or not np.isfinite(step) or step <= 0:
        raise ValueError("Native derivative step must be a finite positive real scalar.")
    if tf.distribute.has_strategy():
        raise UnsupportedFeature("tensorflow_distributed", "Distributed native TF execution is not implemented.")
    from quantumbridge.devices import StatevectorDevice, DensityMatrixDevice, MatrixProductStateDevice
    if not isinstance(node.device, (StatevectorDevice, DensityMatrixDevice, MatrixProductStateDevice)):
        raise UnsupportedFeature("tensorflow_device", "TensorFlow requires a native vector/density/MPS device.")
    limits = TensorFlowBridgeLimits() if node.tensorflow_limits is None else node.tensorflow_limits
    if not isinstance(limits, TensorFlowBridgeLimits):
        raise TypeError("tensorflow_limits requires TensorFlowBridgeLimits.")
    tensors = []
    count = 0
    for value in _leaves((args, kwargs)):
        if tf.is_tensor(value) or isinstance(value, tf.Variable):
            if value.shape.rank is None or not value.shape.is_fully_defined():
                raise UnsupportedFeature("tensorflow_dynamic_shape", "Native TF inputs require fully defined static shapes.")
            if any(label in value.device.upper() for label in ("GPU", "TPU")):
                raise UnsupportedFeature("tensorflow_device", "Native TF supports CPU host tensors only.")
            if value.dtype.is_complex or value.dtype.is_floating and value.dtype not in (tf.float32, tf.float64):
                raise UnsupportedFeature("tensorflow_dtype", "Native TF supports real float32/float64 parameters.")
            if not (value.dtype.is_floating or value.dtype.is_integer or value.dtype == tf.bool):
                raise TypeError("Native TensorFlow inputs require numeric tensors.")
            count += value.shape.num_elements()
            if value.dtype.is_floating:
                tensors.append(value)
        elif isinstance(value, (float, int, bool, np.number, np.ndarray)):
            array = np.asarray(value)
            if array.dtype.kind == "c":
                raise UnsupportedFeature("tensorflow_dtype", "Complex inputs are not supported.")
            count += array.size
    limits.check("max_input_elements", count)
    kernels = []
    def prepare(*call_args, **call_kwargs):
        source, recorder, program = node._construct(call_args, call_kwargs, prepare_only=True)
        kernel = NativeHostKernel(node, source, recorder, program, limits, interface="tensorflow")
        live = [value for value in kernel.angles if tf.is_tensor(value) or isinstance(value, tf.Variable)]
        if any(value.shape.rank != 0 or value.dtype not in (tf.float32, tf.float64) for value in live):
            raise TypeError("TensorFlow gate/coefficient parameters require real float32/float64 scalars.")
        if any(any(label in value.device.upper() for label in ("GPU", "TPU")) for value in live):
            raise UnsupportedFeature("tensorflow_device", "Native TF supports CPU host tensors only.")
        dtype = tf.float64 if any(value.dtype == tf.float64 for value in tensors + live) else tf.float32
        def convert(value):
            return tf.cast(value, dtype) if tf.is_tensor(value) or isinstance(value, tf.Variable) else tf.convert_to_tensor(value, dtype=dtype)
        # All Variable reads are explicit packed inputs, outside custom_gradient.
        packed = tf.stack([convert(value) for value in kernel.angles]) if kernel.angles else tf.zeros((0,), dtype)
        for value in _leaves((call_args, call_kwargs)):
            if (tf.is_tensor(value) or isinstance(value, tf.Variable)) and value.dtype.is_floating:
                packed = packed + tf.cast(0, dtype) * tf.reduce_sum(tf.cast(value, dtype))
        kernel.angles = []
        kernels.append(kernel)
        return packed
    with tf.device("/CPU:0"):
        # Trace classical recording even in eager mode: float(x)/x.numpy()
        # cannot silently erase a trainable quantum parameter's graph.
        packed = prepare(*args, **kwargs) if not tf.executing_eagerly() else tf.function(prepare, autograph=False)(*args, **kwargs)
    kernel, dtype = kernels.pop(), packed.dtype
    # The classical recording FuncGraph can outlive its tracing callable.
    # Its closure must not become another persistent owner of quantum kernels.
    kernels.clear()
    numpy_dtype = dtype.as_numpy_dtype
    graph = not tf.executing_eagerly()
    from .tensorflow_managed import _configuration
    owner_configuration = _configuration(node)
    if not return_status:
        from .tensorflow_whole_execution import select_tensorflow_statevector
        specialization = select_tensorflow_statevector(kernel, packed)
        if specialization is None:
            from .tensorflow_mps_execution import select_tensorflow_mps
            specialization = select_tensorflow_mps(kernel, packed)
        if specialization is not None:
            function, work, receipt = specialization
            flat = function(packed)
            is_mps = work.get("device_path") == "native_mps_exact_index_split"
            node.last_metadata = {"interface": "tensorflow", "execution": "native TF MPS tensor chain" if is_mps else "native TF whole-plan tensor graph",
                "native_tensor_execution": True, "requested_gradient_method": node.diff_method,
                "derivative_strategy": "TF reverse VJP and streamed forward-over-reverse HVP",
                "max_diff_policy": "native order "+str(node.max_diff)+"; higher-point derivatives refused",
                "tensor_plan_work": work, "runtime_graph_accounting": "managed eager entries only; direct cached graph calls have no cumulative execution account",
                "tensor_resource_contract": {"static_plan": "GUARANTEED_STATIC_BOUND", "runtime_parameters": "GRAPH_RUNTIME_VALIDATED",
                    "controlled_execution": "LEDGER_ACCOUNTED" if receipt else "OUT_OF_SCOPE",
                    "allocator_memory": "ESTIMATED_ONLY", "external_loss": "OUT_OF_SCOPE"},
                "tensor_resource_receipt": receipt,
                "trace_configuration_only": graph}
            return unflatten_tree(flat, kernel.spec, reshape=tf.reshape)
    if managed_limits is not None:
        raise UnsupportedFeature("tensorflow_unified_graph_resources", "This QNode is outside the admitted eager whole-tensor unified budget domain.")
    if graph:
        from .tensorflow_graph import TensorFlowGraphKernel
        node.last_metadata = {"interface": "tensorflow", "execution": "CPU host callback",
            "trace_configuration_only": True,
            "callback_budget_semantics": "logical forward/backward per QNode call; not global TF scheduling",
            "observed_callback_count": "not reported from tracing",
            "runtime_failure": "explicit tensorflow_with_status or graph assertion"}
        flat, status = TensorFlowGraphKernel(kernel, dtype, return_status=return_status).evaluate(0, (packed,))
        result = unflatten_tree(flat, kernel.spec, reshape=tf.reshape)
        return (result, status) if return_status else result
    forward_metadata = {}
    backward_calls = 0
    observed_shots = 0

    def host(values, derivative, direction=None):
        nonlocal forward_metadata, backward_calls, observed_shots
        if not graph and derivative:
            limits.check("max_callback_executions", backward_calls + 2)
            limits.check("max_total_shots", kernel.forward_shot_bound + (backward_calls + 1) * kernel.derivative_shot_bound)
        point = np.array(values, dtype=float, copy=True)
        if derivative == 2:
            value, matrix, second, metadata = kernel.run_second(point, direction=np.asarray(direction, dtype=float),
                remaining_shots=None if graph else limits.max_total_shots-observed_shots)
        else:
            value, matrix, metadata = kernel.run(point, derivative,
                remaining_shots=None if graph else limits.max_total_shots-observed_shots)
        if not graph:
            shots = metadata.get("total_physical_shots", metadata.get("gradient_physical_shots",
                                metadata.get("forward_physical_shots", sum(part.get("physical_shots", 0) for part in metadata.get("partitions", ())))))
            observed_shots += shots
            if derivative:
                backward_calls += 1
            else:
                forward_metadata = metadata
            node.last_metadata = {**metadata, "native_vjp": bool(derivative),
                                  "classical_chain_rule": "TensorFlow graph", "execution": "CPU host",
                                  "forward_execution": forward_metadata, "backward_calls": backward_calls,
                                  "observed_physical_shots": observed_shots}
        arrays = (value.astype(numpy_dtype), matrix.astype(numpy_dtype)) if derivative else (value.astype(numpy_dtype),)
        if derivative == 2:
            arrays += (second.astype(numpy_dtype),)
        return (*arrays, np.int32(0))

    def checked(values, derivative, direction=None):
        try:
            return host(values, derivative, direction)
        except Exception as error:
            code = ordinary_status_code(error)
        arrays = (np.full((kernel.output_size,), np.nan, numpy_dtype),)
        if derivative:
            arrays += (np.full((kernel.output_size, kernel.size), np.nan, numpy_dtype),)
        if derivative == 2:
            arrays += (np.full((kernel.output_size, kernel.size, 1), np.nan, numpy_dtype),)
        return (*arrays, np.int32(code))

    def evaluate(values, derivative, direction=None):
        limits.check("max_callback_executions", 3 if derivative == 2 else 2 if derivative else 1)
        vector = direction.numpy() if derivative == 2 else None
        arrays = checked(values.numpy(), derivative, vector) if return_status else host(values.numpy(), derivative, vector)
        outputs = [tf.convert_to_tensor(value) for value in arrays]
        outputs[0].set_shape((kernel.output_size,))
        if derivative:
            outputs[1].set_shape((kernel.output_size, kernel.size))
        if derivative == 2:
            outputs[2].set_shape((kernel.output_size, kernel.size, 1))
        outputs[-1].set_shape(())
        if not return_status:
            with tf.control_dependencies([tf.debugging.assert_equal(outputs[-1], 0, message="Native TensorFlow host execution failed; use tensorflow_with_status for code")]):
                outputs = [tf.identity(value) for value in outputs]
        return outputs

    def traced_rule():
        if tf.executing_eagerly() or dtype != tf.float64 or return_status:
            return None
        if _configuration(node) != owner_configuration:
            raise UnsupportedFeature("tensorflow_compiled_owner_changed",
                "QB_UNSUPPORTED[tensorflow_compiled_owner_changed]: The eager forward owner configuration changed before traced differentiation.")
        from .tensorflow_whole_execution import select_tensorflow_eager_vjp
        return select_tensorflow_eager_vjp(kernel)

    def note_traced(decision):
        node.last_metadata = {**node.last_metadata, "traced_eager_vjp": True,
            "traced_vjp_accounting": "OUT_OF_SCOPE", "traced_vjp_tensor_plan": dict(decision),
            "gradient_method": decision["gradient_method"],
            "shifted_evaluation_count": decision["shifted_evaluation_count"],
            "gradient_physical_shots": 0, "tensor_count_scope": "Structural per VJP, not external graph execution counters"}

    @tf.custom_gradient
    def second_order_vjp(values, cotangent, incoming):
        rule = traced_rule()
        if rule is None:
            _, matrix, second, status = evaluate(values, 2, incoming)
            value_gradient = tf.linalg.matvec(second[:, :, 0], cotangent, transpose_a=True)
            cotangent_gradient = tf.linalg.matvec(matrix, incoming)
        else:
            function, decision = rule
            value_gradient, cotangent_gradient = function(values, cotangent, incoming)
            note_traced(decision)
            status = None
        if return_status and status is not None:
            ok = tf.equal(status, 0)
            value_gradient = tf.where(ok, value_gradient, tf.fill(tf.shape(value_gradient), tf.cast(float("nan"), dtype)))
            cotangent_gradient = tf.where(ok, cotangent_gradient, tf.fill(tf.shape(cotangent_gradient), tf.cast(float("nan"), dtype)))
        def reject(*unused):
            raise UnsupportedFeature("higher_order_derivatives", "Native TensorFlow third derivatives are not implemented.")
        return (value_gradient, cotangent_gradient), reject

    @tf.custom_gradient
    def first_order_vjp(values, cotangent):
        rule = traced_rule()
        if rule is None:
            _, matrix, status = evaluate(values, True)
            result = tf.linalg.matvec(matrix, cotangent, transpose_a=True)
        else:
            function, decision = rule
            result, status = function(values, cotangent), None
            note_traced(decision)
        if return_status and status is not None:
            result = tf.where(tf.equal(status, 0), result, tf.fill(tf.shape(result), tf.cast(float("nan"), dtype)))
        def backward(incoming):
            if node.max_diff != 2:
                raise UnsupportedFeature("higher_order_derivatives", "Native TensorFlow second derivatives require max_diff=2.")
            return second_order_vjp(values, cotangent, incoming)
        return result, backward

    @tf.custom_gradient
    def native(values):
        flat, status = evaluate(values, False)
        def backward(cotangent, status_cotangent):
            return first_order_vjp(values, cotangent)
        return (flat, status), backward

    flat, status = native(packed)
    result = unflatten_tree(flat, kernel.spec, reshape=tf.reshape)
    return (result, status) if return_status else result


def execute_tensorflow_batch(node, values):
    """Bounded leading-axis mapping of one real tensor argument, CPU only."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise MissingOptionalDependency("Native TensorFlow QNode requires optional tensorflow.") from exc
    values = tf.convert_to_tensor(values)
    if values.shape.rank is None or values.shape.rank < 1 or not values.shape.is_fully_defined() or values.shape[0] == 0:
        raise UnsupportedFeature("tensorflow_batch_shape", "TF batch inputs need a nonempty static leading axis.")
    limits = TensorFlowBridgeLimits() if node.tensorflow_limits is None else node.tensorflow_limits
    if not isinstance(limits, TensorFlowBridgeLimits):
        raise TypeError("tensorflow_limits requires TensorFlowBridgeLimits.")
    size = values.shape[0]
    limits.check("max_batch_size", size)
    limits.check("max_input_elements", values.shape.num_elements())
    callbacks_per_item = 3 if node.max_diff == 2 else 2
    limits.check("max_callback_executions", callbacks_per_item * size)
    child = copy(node)
    child.tensorflow_limits = replace(limits, max_total_shots=max(1, limits.max_total_shots // size))
    function = tf.function(lambda x: execute_tensorflow(child, (x,), {}), autograph=False)
    concrete = function.get_concrete_function(tf.TensorSpec(values.shape[1:], values.dtype))
    signature = tf.nest.map_structure(lambda x: tf.TensorSpec(x.shape, x.dtype), concrete.structured_outputs)
    limits.check("max_output_elements", sum(x.shape.num_elements() for x in tf.nest.flatten(signature)) * size)
    node.last_metadata = {"interface": "tensorflow", "execution": "bounded CPU host map",
                          "batch_size": size, "trace_configuration_only": True,
                          "observed_callback_count": "not reported from tracing",
                          "logical_forward_backward_callback_bound": callbacks_per_item * size}
    return tf.map_fn(function, values, fn_output_signature=signature, parallel_iterations=1)
