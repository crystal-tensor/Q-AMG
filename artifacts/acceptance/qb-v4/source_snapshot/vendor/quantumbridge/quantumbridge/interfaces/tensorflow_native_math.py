# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Verified local TF tensor formulas; not a QNode executor or MPS engine.

This module is optional and imported only when executing a TF local kernel.
The independent no-QuantumBridge probe remains a separate numerical witness.
"""

import math
from numbers import Real

import tensorflow as tf

_EPS = math.ulp(1.)


def complex_real(value):
    return tf.complex(value, tf.zeros_like(value))


def phase(angle):
    return tf.complex(tf.cos(angle), tf.sin(angle))


def rotation(axis, angle):
    c, s = tf.cos(angle/2), tf.sin(angle/2)
    z, c, s = tf.zeros_like(c), complex_real(c), complex_real(s)
    if axis == "x":
        return tf.stack([tf.stack([c, -1j*s]), tf.stack([-1j*s, c])])
    if axis == "y":
        return tf.stack([tf.stack([c, -s]), tf.stack([s, c])])
    return tf.linalg.diag(tf.stack([c-tf.complex(z, tf.math.real(s)), c+tf.complex(z, tf.math.real(s))]))


def kron(a, b):
    return tf.reshape(tf.einsum("ij,kl->ikjl", a, b), (a.shape[0]*b.shape[0], a.shape[1]*b.shape[1]))


@tf.function(autograph=False, jit_compile=False)
def _validated_source(matrix, global_phase, exponent):
    # Only boolean validation decisions cross this boundary. No trainable
    # output is hidden inside a function that also executes assertion ops.
    identity = tf.eye(2, dtype=tf.complex128)
    residual = tf.reduce_max(tf.reduce_sum(tf.abs(tf.linalg.adjoint(matrix) @ matrix-identity), axis=0))
    checks = [tf.debugging.assert_all_finite(tf.math.real(matrix), "Nonfinite local matrix"),
        tf.debugging.assert_all_finite(tf.math.imag(matrix), "Nonfinite local matrix"),
        tf.raw_ops.Assert(condition=residual <= tf.constant(512*2*_EPS, tf.float64),
            data=[tf.constant("Non-unitary SU(2) source"), residual]),
        tf.raw_ops.Assert(condition=tf.abs(tf.linalg.det(matrix)-1) <= tf.constant(512*2*_EPS, tf.float64),
            data=[tf.constant("SU(2) determinant must equal one")])]
    checks.append(tf.debugging.assert_all_finite(global_phase, "Nonfinite local phase"))
    if exponent.is_integer():
        with tf.control_dependencies(checks):
            return tf.constant(False), tf.constant(False)
    center = tf.math.real(tf.linalg.trace(matrix))/2
    negative = center < 0.
    flip = tf.where(negative, tf.constant(-1., tf.float64), tf.constant(1., tf.float64))
    matrix = complex_real(flip)*matrix
    global_phase += tf.where(flip < 0., tf.constant(math.pi, tf.float64), tf.constant(0., tf.float64))
    center = tf.math.real(tf.linalg.trace(matrix))/2
    delta = matrix-complex_real(center)*identity
    squared = tf.reduce_sum(tf.math.real(tf.math.conj(delta)*delta))/2
    small = squared*(1+abs(exponent))**2 < 1e-6
    safe_squared = tf.where(small, tf.constant(1e-6, tf.float64), squared)
    sine = tf.sqrt(safe_squared)
    theta = tf.math.atan2(sine, center)
    def wrap(angle):
        return tf.math.atan2(tf.sin(angle), tf.cos(angle))
    plus, minus, principal = wrap(global_phase+theta), wrap(global_phase-theta), wrap(global_phase)
    resolution = tf.constant(512*2*_EPS, tf.float64)
    gap = 2*sine
    screen = (64*_EPS*4*(1+abs(exponent))**2*abs(math.sin(math.pi*exponent)))/gap**3
    regular_safe = ((tf.abs(tf.abs(plus)-math.pi) > resolution)
        & (tf.abs(tf.abs(minus)-math.pi) > resolution)
        & ((tf.abs(plus-minus) <= math.pi) | (screen <= math.sqrt(_EPS))))
    # asin(sqrt(t))**2 has a regular power series at t=0. Use it only
    # for the branch guard; the returned matrix contains no sqrt(0) graph.
    theta_squared = tf.zeros_like(squared)
    for index in range(1, 7):
        theta_squared += (4**index/(2*index*index*math.comb(2*index, index)))*squared**index
    margin = math.pi-tf.abs(principal)
    small_safe = (margin > resolution) & ((margin-resolution)**2 > theta_squared)
    guard = tf.raw_ops.Assert(condition=tf.where(small, small_safe, regular_safe),
        data=[tf.constant("Fractional SU(2) principal branch/cubic separation screen is unresolved")])
    with tf.control_dependencies([*checks, guard]):
        return tf.identity(negative), tf.identity(small)


def fractional_su2(matrix, exponent, global_phase):
    if matrix.shape != (2, 2) or matrix.dtype != tf.complex128:
        raise ValueError("The native TF local evaluator requires a static complex128 2x2 SU(2) matrix")
    if isinstance(exponent, bool) or not isinstance(exponent, Real):
        raise TypeError("The exponent must be a static real scalar")
    exponent = float(exponent)
    if not math.isfinite(exponent):
        raise ValueError("The exponent must be a finite static real number")
    global_phase = tf.convert_to_tensor(global_phase, tf.float64)
    if global_phase.shape != ():
        raise ValueError("The local global phase must be scalar")
    negative, small = _validated_source(matrix, global_phase, exponent)
    identity = tf.eye(2, dtype=tf.complex128)
    if exponent.is_integer():
        count, base, result = abs(int(exponent)), matrix, identity
        if exponent < 0:
            base = tf.linalg.adjoint(base)
        while count:
            if count & 1:
                result = base @ result
            count >>= 1
            if count:
                base = base @ base
        return phase(exponent*global_phase)*result
    matrix = tf.where(negative, -matrix, matrix)
    global_phase += tf.where(negative, tf.constant(math.pi, tf.float64), tf.constant(0., tf.float64))
    center = tf.math.real(tf.linalg.trace(matrix))/2
    delta = matrix-complex_real(center)*identity
    squared = tf.reduce_sum(tf.math.real(tf.math.conj(delta)*delta))/2
    safe_squared = tf.where(small, tf.constant(1e-6, tf.float64), squared)
    sine = tf.sqrt(safe_squared)
    theta = tf.math.atan2(sine, center)
    def wrap(angle):
        return tf.math.atan2(tf.sin(angle), tf.cos(angle))
    plus, minus, principal = wrap(global_phase+theta), wrap(global_phase-theta), wrap(global_phase)

    a_coefficients, b_coefficients = [1.], [exponent]
    for index in range(4):
        a_coefficients.append(a_coefficients[-1]*((2*index)**2-exponent**2)/((2*index+2)*(2*index+1)))
        b_coefficients.append(b_coefficients[-1]*((2*index+1)**2-exponent**2)/((2*index+3)*(2*index+2)))
    if not all(math.isfinite(value) for value in a_coefficients+b_coefficients):
        raise FloatingPointError("SU(2) derivative series coefficients exceed float64")
    def polynomial(coefficients):
        result = tf.constant(coefficients[-1], tf.float64)
        for value in reversed(coefficients[:-1]):
            result = result*squared+value
        return result
    small_a = phase(exponent*principal)*complex_real(polynomial(a_coefficients))
    small_b = phase(exponent*principal)*complex_real(polynomial(b_coefficients))
    regular_a = (phase(exponent*plus)+phase(exponent*minus))/2
    regular_b = (phase(exponent*plus)-phase(exponent*minus))/tf.complex(tf.zeros_like(sine), 2*sine)
    a = tf.where(small, small_a, regular_a)
    b = tf.where(small, small_b, regular_b)
    return a*identity+b*delta


