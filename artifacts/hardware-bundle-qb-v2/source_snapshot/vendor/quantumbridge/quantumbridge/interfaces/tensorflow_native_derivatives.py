# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Streamed native TF first/second rules with an actual third-point boundary."""

from contextlib import nullcontext

from quantumbridge.core.classical import UnsupportedFeature


def bounded_tensor_function(function, admission=None, *, first_rule=None, second_rule=None, max_order=2):
    import tensorflow as tf

    def admitted(order):
        return nullcontext() if admission is None else admission(order)

    def seed(value):
        from .tensorflow_native_execution import _finite_parameters
        valid = _finite_parameters()(value)
        return tf.where(valid, value, tf.zeros_like(value))

    if max_order == 1:
        @tf.custom_gradient
        def first_point(point):
            def refuse(incoming):
                raise UnsupportedFeature("higher_order_derivatives", "Native TensorFlow second-point derivatives require approval for order two.")
            return tf.identity(point), refuse

        @tf.custom_gradient
        def first_only(point):
            with admitted(0):
                value = function(point)
            def backward(cotangent):
                # Guard the quantum point, not transposition of a cotangent.
                # Forward AD of a first derivative rule needs the latter.
                with admitted(1):
                    guarded, cotangent = first_point(point), seed(cotangent)
                    if first_rule is not None:
                        return first_rule(guarded, cotangent)
                    with tf.GradientTape() as tape:
                        tape.watch(guarded)
                        output = function(guarded)
                    return tape.gradient(output, guarded, output_gradients=cotangent,
                        unconnected_gradients=tf.UnconnectedGradients.ZERO)
            return value, backward
        return first_only

    @tf.custom_gradient
    def second_point(point):
        def refuse(incoming):
            raise UnsupportedFeature("higher_order_derivatives", "Native TensorFlow genuine third-point derivatives are not implemented.")
        return tf.identity(point), refuse

    @tf.custom_gradient
    def vjp(point, cotangent):
        if max_order < 1:
            raise UnsupportedFeature("higher_order_derivatives", "This tensor handle approved value execution only.")
        with admitted(1):
            cotangent = seed(cotangent)
            if first_rule is not None:
                result = first_rule(point, cotangent)
            else:
                with tf.GradientTape() as tape:
                    tape.watch(point)
                    values = function(point)
                result = tape.gradient(values, point, output_gradients=cotangent, unconnected_gradients=tf.UnconnectedGradients.ZERO)

        def gradient(direction):
            # The higher-order boundary is on the quantum HVP's point, not
            # its cotangent/direction. Those remain legal first/second edges.
            if max_order < 2:
                raise UnsupportedFeature("higher_order_derivatives", "This tensor handle requires approval for derivative order two.")
            with admitted(2):
                direction = seed(direction)
                guarded = second_point(point)
                if second_rule is not None:
                    return second_rule(guarded, point, cotangent, direction)
                with tf.autodiff.ForwardAccumulator(guarded, direction) as forward:
                    with tf.GradientTape() as reverse:
                        reverse.watch(guarded)
                        value = function(guarded)
                    weighted = reverse.gradient(value, guarded, output_gradients=cotangent, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                second = forward.jvp(weighted)
                if second is None:
                    second = tf.zeros_like(guarded)
                with tf.autodiff.ForwardAccumulator(point, direction) as tangent:
                    value = function(point)
                projected = tangent.jvp(value)
                if projected is None:
                    projected = tf.zeros_like(value)
            return second, projected
        return result, gradient

    @tf.custom_gradient
    def primal(point):
        with admitted(0):
            value = function(point)
        return value, lambda cotangent: vjp(point, cotangent)

    return primal
