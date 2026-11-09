# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Payload-free TF gradient registration around the existing native kernel."""

import weakref
from dataclasses import dataclass

import numpy as np

from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.execution.array_ownership import retained_array
from quantumbridge.execution.array_reservations import borrow_callback_array, mapped_matmul, scope_array
from quantumbridge.execution.resource_context import ResourceScope
from quantumbridge.execution.resource_ledger import UnifiedResourceLedger
from .tensorflow_callbacks import TensorFlowCallback, _graph_lock, ordinary_status_code


_calls = weakref.WeakValueDictionary()
_registered = False
_rules = ("QuantumBridgeNativeForward", "QuantumBridgeNativeVJP", "QuantumBridgeNativeSecondVJP")


def _gradient(operation, *cotangents):
    with _graph_lock:
        callback = _calls.get(operation.inputs[0].op.get_attr("token"))
    if callback is None:
        raise RuntimeError("The native TensorFlow gradient's owning graph has expired.")
    return callback.gradient(operation, cotangents)


class _GraphCall(TensorFlowCallback):
    def __init__(self, owner, order):
        self.owner, self.order = owner, order

    def __call__(self, *values):
        return self.owner.run(self.order, values)

    def gradient(self, operation, cotangents):
        import tensorflow as tf
        incoming = cotangents[0]
        prefix = [None]*(len(self.owner.shapes(self.order))+1)
        inputs = operation.inputs[len(prefix):]
        if self.order == 2:
            raise UnsupportedFeature("higher_order_derivatives", "Native TensorFlow third derivatives are not implemented.")
        if incoming is None:
            return prefix + [tf.zeros_like(value) for value in inputs]
        if self.order == 0:
            return prefix + [self.owner.evaluate(1, (inputs[0], incoming))[0]]
        if self.owner.kernel.node.max_diff != 2:
            raise UnsupportedFeature("higher_order_derivatives", "Native TensorFlow second derivatives require max_diff=2.")
        values, cotangent = inputs
        result = self.owner.evaluate(2, (values, cotangent, incoming))
        return prefix + result[:2]


@dataclass(frozen=True)
class TensorFlowGraphKernel:
    """Graph-owned immutable description; runtime accounts are callback-local.

    This is not a graph-wide cumulative budget and stores no framework tensor,
    FuncGraph or request ledger. Separate static owners can share one kernel.
    """

    kernel: object
    dtype: object
    return_status: bool = False
    request_limits: object = None

    @property
    def numpy_dtype(self):
        return np.dtype(self.dtype.as_numpy_dtype)

    def shapes(self, order):
        outputs, parameters = self.kernel.output_size, self.kernel.size
        return ((outputs,),) if order == 0 else ((parameters,),) if order == 1 else ((parameters,), (outputs,))

    def run(self, order, values):
        ledger = UnifiedResourceLedger(self.request_limits)
        ledger.unavailable("framework_allocator_memory",
            "Framework graphs, transport failure envelopes and allocator-owned buffers are not measured native ndarray or process RSS memory.")
        try:
            with ResourceScope(ledger, new_request=True) as scope:
                self.kernel.admit(scope)
                shapes = ((self.kernel.size,), (self.kernel.output_size,), (self.kernel.size,))
                arrays = []
                for value, shape in zip(values, shapes):
                    value = borrow_callback_array(value)
                    if value.shape != shape or value.dtype.kind not in "fiu" or not np.all(np.isfinite(value)):
                        raise ValueError("Native TensorFlow callback inputs must have finite real declared shapes.")
                    arrays.append(scope_array(scope, "tensorflow_callback_input", shape, float,
                        lambda value=value: np.array(value, dtype=float, copy=True)))
                point = arrays[0]
                if order == 0:
                    value, _, _ = self.kernel.run(point, False)
                    result = (value,)
                elif order == 1:
                    _, matrix, _ = self.kernel.run(point, True)
                    result = (mapped_matmul(scope, "tensorflow_graph_vjp", matrix.T, arrays[1]),)
                else:
                    _, matrix, second, _ = self.kernel.run_second(point, direction=arrays[2])
                    result = (mapped_matmul(scope, "tensorflow_graph_second_vjp", second[:, :, 0].T, arrays[1]),
                              mapped_matmul(scope, "tensorflow_graph_cotangent_vjp", matrix, arrays[2]))
                converted = tuple(retained_array(ledger, "tensorflow_callback_result", value.shape,
                    self.numpy_dtype, lambda value=value: value.astype(self.numpy_dtype, copy=True)) for value in result)
                return (*converted, np.int32(0))
        except Exception as error:
            from quantumbridge.execution.derivative_status import detach_status_error
            code = ordinary_status_code(error)
            detach_status_error(error)
        # A bounded framework transport envelope is not a successful native
        # result or a proof of allocator/RSS accounting under a rejected cap.
        return (*(np.full(shape, np.nan, self.numpy_dtype) for shape in self.shapes(order)), np.int32(code))

    def evaluate(self, order, inputs):
        import tensorflow as tf
        reference = weakref.ref(self)
        @tf.function(autograph=False)
        def call(*values):
            owner = reference()
            if owner is None:
                raise RuntimeError("The native TensorFlow graph owner has expired.")
            return owner._evaluate(order, values)
        return list(call(*inputs))

    def _evaluate(self, order, inputs):
        import tensorflow as tf
        global _registered
        self.kernel.limits.check("max_callback_executions", order + 1)
        callback = _GraphCall(self, order)
        with _graph_lock:
            if not _registered:
                for name in _rules: tf.RegisterGradient(name)(_gradient)
                _registered = True
        graph = tf.compat.v1.get_default_graph()
        with tf.device("/CPU:0"):
            outputs = callback.numpy_function(inputs, [self.dtype]*len(self.shapes(order))+[tf.int32], stateful=True)
        with _graph_lock:
            _calls[outputs[0].op.get_attr("token")] = callback
        for output, shape in zip(outputs[:-1], self.shapes(order)): output.set_shape(shape)
        outputs[-1].set_shape(())
        with graph.gradient_override_map({"IdentityN": _rules[order]}):
            outputs = tf.identity_n([*outputs, *inputs])[:len(outputs)]
        if not self.return_status:
            with tf.control_dependencies([tf.debugging.assert_equal(outputs[-1], 0,
                    message="Native TensorFlow host execution failed; use tensorflow_with_status for code")]):
                outputs = [tf.identity(value) for value in outputs]
        return outputs
