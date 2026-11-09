# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Keep native callback payloads with their graph, not a callback registry."""

import weakref
from threading import RLock


_graphs = weakref.WeakKeyDictionary()
_graph_lock = RLock()
_status_gradient_registered = False


def ordinary_status_code(error):
    """Keep the ordinary TF callback's established 0-4 error protocol."""
    from quantumbridge.core.classical import UnsupportedFeature
    seen = set()
    while error.__cause__ is not None and id(error) not in seen:
        seen.add(id(error))
        error = error.__cause__
    return (2 if isinstance(error, UnsupportedFeature) else 3 if isinstance(error, MemoryError)
            else 1 if isinstance(error, (ValueError, FloatingPointError)) else 4)


def _reject_status_gradient(operation, *cotangents):
    from quantumbridge.core.classical import UnsupportedFeature
    raise UnsupportedFeature("derivative_status_higher_order", "Differentiate the QNode graph; explicit derivative-query payloads do not supply additional derivative rules.")


def status_numpy_function(callback, inputs, outputs):
    import tensorflow as tf
    if tf.executing_eagerly():
        @tf.custom_gradient
        def checked(*values):
            return callback.numpy_function(values, outputs), lambda *unused: _reject_status_gradient(None)
        return checked(*inputs)
    global _status_gradient_registered
    name = "QuantumBridgeNativeDerivativeStatus"
    with _graph_lock:
        if not _status_gradient_registered:
            tf.RegisterGradient(name)(_reject_status_gradient)
            _status_gradient_registered = True
    # A process-wide rule must not close over this invocation's output tensors
    # (and thereby its FuncGraph). This rejection rule has no request state.
    with tf.compat.v1.get_default_graph().gradient_override_map(
            {"PyFunc": name, "PyFuncStateless": name}):
        return callback.numpy_function(inputs, outputs)


class TensorFlowCallback:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)

    def numpy_function(self, inputs, outputs, **options):
        import tensorflow as tf
        reference = weakref.ref(self)
        def invoke(*args):
            owner = reference()
            if owner is None:
                raise RuntimeError("The native TensorFlow callback's owning graph has expired.")
            return owner(*args)
        if not tf.executing_eagerly():
            # TF control-flow graphs can share collection dictionaries. Bind to
            # this graph's identity instead, without a strong graph key or any
            # callback registration that owns native configuration indefinitely.
            graph = tf.compat.v1.get_default_graph()
            with _graph_lock:
                _graphs.setdefault(graph, []).append(self)
        return tf.numpy_function(invoke, inputs, outputs, **options)
