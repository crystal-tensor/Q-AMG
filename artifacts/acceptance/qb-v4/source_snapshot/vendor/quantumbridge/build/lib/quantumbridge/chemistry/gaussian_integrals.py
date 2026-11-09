# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Independent normalized s-Gaussian integrals in atomic units."""
from dataclasses import dataclass
from itertools import product
from math import erf, exp, pi, sqrt
from numbers import Real
import numpy as np


def _center(value):
    if np.iscomplexobj(value):
        raise ValueError("Gaussian centers must be real coordinates.")
    result = np.asarray(value, float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("Gaussian center must contain three finite Bohr coordinates.")
    return result


def primitive_normalization(alpha):
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Gaussian exponent must be finite positive.")
    return (2 * float(alpha) / pi) ** .75


def boys0(value):
    if isinstance(value, bool) or not np.isfinite(value) or value < 0:
        raise ValueError("Boys argument must be finite nonnegative.")
    if value < 1e-6:
        return 1 - value / 3 + value ** 2 / 10 - value ** 3 / 42 + value ** 4 / 216
    return .5 * sqrt(pi / value) * erf(sqrt(value))


def _pair(alpha, beta, a, b):
    normalization = primitive_normalization(alpha) * primitive_normalization(beta)
    a, b = _center(a), _center(b)
    p, mu = alpha + beta, alpha * beta / (alpha + beta)
    distance2 = float(np.dot(a - b, a - b))
    return p, mu, distance2, (alpha * a + beta * b) / p, normalization * exp(-mu * distance2)


def primitive_overlap(alpha, beta, a, b):
    p, _, _, _, factor = _pair(alpha, beta, a, b)
    return factor * (pi / p) ** 1.5


def primitive_kinetic(alpha, beta, a, b):
    p, mu, distance2, _, factor = _pair(alpha, beta, a, b)
    return mu * (3 - 2 * mu * distance2) * factor * (pi / p) ** 1.5


def primitive_attraction(alpha, beta, a, b, nucleus, charge=1):
    if not np.isfinite(charge) or charge <= 0:
        raise ValueError("Nuclear charge must be positive finite.")
    p, _, _, center, factor = _pair(alpha, beta, a, b)
    delta = center - _center(nucleus)
    return -charge * factor * 2 * pi / p * boys0(p * float(delta @ delta))


def primitive_repulsion(alpha, beta, gamma, delta, a, b, c, d):
    p, _, _, center_p, factor_p = _pair(alpha, beta, a, b)
    q, _, _, center_q, factor_q = _pair(gamma, delta, c, d)
    offset = center_p - center_q
    return factor_p * factor_q * 2 * pi ** 2.5 / (p * q * sqrt(p + q)) * boys0(p * q / (p + q) * float(offset @ offset))


@dataclass(frozen=True)
class ContractedSGaussian:
    center: tuple
    exponents: tuple
    coefficients: tuple

    def __init__(self, center, exponents, coefficients):
        center = tuple(_center(center))
        exponents, coefficients = tuple(exponents), tuple(coefficients)
        if not exponents or len(exponents) != len(coefficients) or len(exponents) > 16:
            raise ValueError("s contraction requires matching 1-16 primitives.")
        for exponent in exponents:
            primitive_normalization(exponent)
        if any(not isinstance(value, Real) or not np.isfinite(value) for value in coefficients):
            raise ValueError("Contraction coefficients must be finite real values.")
        norm = sum(x * y * primitive_overlap(a, b, center, center) for a, x in zip(exponents, coefficients) for b, y in zip(exponents, coefficients))
        if not np.isfinite(norm) or norm <= 1e-16:
            raise ValueError("Contracted Gaussian must have nonzero finite norm.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "exponents", tuple(map(float, exponents)))
        object.__setattr__(self, "coefficients", tuple(float(v / sqrt(norm)) for v in coefficients))


def _contract(function, basis, *extra):
    value = 0.
    for indices in product(*(range(len(item.exponents)) for item in basis)):
        exponents = [item.exponents[i] for item, i in zip(basis, indices)]
        coefficient = float(np.prod([item.coefficients[i] for item, i in zip(basis, indices)]))
        value += coefficient * function(*exponents, *(item.center for item in basis), *extra)
    return value


def overlap(a, b):
    return _contract(primitive_overlap, (a, b))


def kinetic(a, b):
    return _contract(primitive_kinetic, (a, b))


def nuclear_attraction(a, b, nucleus, charge=1):
    return _contract(primitive_attraction, (a, b), nucleus, charge)


def electron_repulsion(a, b, c, d):
    return _contract(primitive_repulsion, (a, b, c, d))
