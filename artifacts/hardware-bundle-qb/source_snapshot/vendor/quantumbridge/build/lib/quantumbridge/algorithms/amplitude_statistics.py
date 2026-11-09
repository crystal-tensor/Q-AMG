# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Independent Fourier likelihood and bounded canonical-QAE inference, NumPy only."""
from math import asin, isfinite, pi, sin, sqrt
from numbers import Integral, Real
from statistics import NormalDist

import numpy as np

from quantumbridge.core.gate_limits import current_gate_limits
from .phase_estimation import _evaluation_bits


def _phase_probabilities(amplitude, bits, indices=None):
    size = 1 << _evaluation_bits(bits)
    current_gate_limits().check("max_decomposition_work", size)
    if isinstance(amplitude, bool) or not isinstance(amplitude, Real) or not isfinite(amplitude) or not 0 <= amplitude <= 1:
        raise ValueError("Amplitude probability must be finite and in [0, 1].")
    if indices is None:
        indices = np.arange(size)
    else:
        indices = np.asarray(indices)
    if amplitude in (0., 1.):
        return np.asarray(indices == (0 if amplitude == 0 else size//2), dtype=float)
    phase = asin(sqrt(amplitude))/pi
    def kernel(delta):
        delta = (delta+.5) % 1.-.5
        return (np.sinc(size*delta)/np.sinc(delta))**2
    return .5*(kernel(phase-indices/size)+kernel(-phase-indices/size))


class _Likelihood:
    def __init__(self, result):
        self.bits = _evaluation_bits(result.num_evaluation_qubits)
        self.size = 1 << self.bits
        current_gate_limits().check("max_decomposition_work", self.size)
        if isinstance(result.shots, bool) or not isinstance(result.shots, Integral) or result.shots < 1:
            raise ValueError("Likelihood requires a positive integer shot count.")
        entries = result.measurements
        if not entries:
            if not result.samples:
                raise ValueError("Likelihood requires measured weights.")
            entries = {}
            for point, weight in result.samples.items():
                if not isinstance(point, Real) or not isfinite(point) or not 0 <= point <= 1:
                    raise ValueError("Sample grid points must lie in [0, 1].")
                index = int(round(self.size*asin(sqrt(point))/pi))
                entries[index] = entries.get(index, 0.)+weight
            self.folded = True
        else:
            self.folded = False
        current_gate_limits().check("max_decomposition_work", len(entries))
        for index, weight in entries.items():
            if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < self.size:
                raise ValueError("Measurement indices must lie on the evaluation grid.")
            if isinstance(weight, bool) or not isinstance(weight, Real) or not isfinite(weight) or weight < 0:
                raise ValueError("Likelihood requires finite nonnegative measured weights.")
        positive = [(int(index), float(weight)) for index, weight in entries.items() if weight > 0]
        if not positive:
            raise ValueError("Likelihood requires a positive measured weight.")
        self.indices = np.array([index for index, _ in positive], dtype=np.int64)
        self.weights = np.array([weight for _, weight in positive])
        self.shots, self.work = int(result.shots), 0
        grouped = {}
        for index, weight in positive:
            index = min(index, self.size-index)
            grouped[index] = grouped.get(index, 0.)+weight
        self.mode = max(grouped, key=grouped.get)

    def __call__(self, theta):
        self.work += len(self.indices)
        current_gate_limits().check("max_decomposition_work", self.work)
        probabilities = _phase_probabilities(sin(theta)**2, self.bits, self.indices)
        if self.folded:
            probabilities = probabilities*np.where((self.indices == 0) | (self.indices == self.size//2), 1., 2.)
        if np.any(probabilities <= 0):
            return -float("inf")
        return float(self.shots*np.dot(self.weights, np.log(probabilities)))

    @property
    def intervals(self):
        points = [pi*index/self.size for index in range(max(0, self.mode-1), min(self.size//2, self.mode+1)+1)]
        return list(zip(points, points[1:]))


def _maximize(function, lower, upper):
    ratio = (sqrt(5)-1)/2
    left, right = upper-ratio*(upper-lower), lower+ratio*(upper-lower)
    left_value, right_value = function(left), function(right)
    candidates = [(lower, function(lower)), (upper, function(upper))]
    for _ in range(80):
        if upper-lower <= 2e-13:
            break
        if left_value < right_value:
            lower, left, left_value = left, right, right_value
            right = lower+ratio*(upper-lower)
            right_value = function(right)
        else:
            upper, right, right_value = right, left, left_value
            left = upper-ratio*(upper-lower)
            left_value = function(left)
    return max([*candidates, (left, left_value), (right, right_value)], key=lambda item: item[1])


def maximum_likelihood(result):
    likelihood = _Likelihood(result)
    theta, _ = max((_maximize(likelihood, lower, upper) for lower, upper in likelihood.intervals), key=lambda item: item[1])
    return sin(theta)**2


def _crossing(function, lower, upper):
    left = function(lower)
    for _ in range(80):
        middle = (lower+upper)/2
        value = function(middle)
        if upper-lower <= 2e-13:
            return middle
        if (left >= 0) == (value >= 0):
            lower, left = middle, value
        else:
            upper = middle
    return (lower+upper)/2


def confidence_interval(result, alpha=.05, kind="likelihood_ratio"):
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite real number strictly between zero and one.")
    if kind not in {"likelihood_ratio", "fisher", "observed_fisher"}:
        raise NotImplementedError("Unknown amplitude-estimation confidence interval kind.")
    likelihood = _Likelihood(result)
    amplitude = maximum_likelihood(result) if result.mle is None else result.mle
    if not isinstance(amplitude, Real) or not isfinite(amplitude) or not 0 <= amplitude <= 1:
        raise ValueError("MLE must be a finite probability.")
    if alpha/2 == 0:
        raise ValueError("alpha is below the representable normal-tail range.")
    z = -NormalDist().inv_cdf(alpha/2)
    if kind != "likelihood_ratio":
        information = _fisher_information(likelihood, amplitude, kind == "observed_fisher")
        radius = z/sqrt(likelihood.shots*information)
        return amplitude-radius, amplitude+radius
    theta = asin(sqrt(amplitude))
    peak_likelihood = likelihood(theta)
    if not isfinite(peak_likelihood):
        raise ValueError("Supplied MLE has zero likelihood for the measured data.")
    threshold = peak_likelihood-.5*z*z
    cut = lambda value: likelihood(value)-threshold
    lower = upper = theta
    for start, end in likelihood.intervals:
        peak, value = _maximize(likelihood, start, end)
        if value < threshold:
            continue
        left = start if cut(start) >= 0 else _crossing(cut, start, peak)
        right = end if cut(end) >= 0 else _crossing(cut, peak, end)
        lower, upper = min(lower, left), max(upper, right)
    return sin(lower)**2, sin(upper)**2


def _fisher_information(likelihood, amplitude, observed):
    if not 0 < amplitude < 1:
        raise ValueError("Endpoint Fisher information is singular; use the likelihood-ratio interval.")
    current_gate_limits().check("max_decomposition_work", likelihood.size)
    indices = likelihood.indices if observed else np.arange(likelihood.size)
    phase = asin(sqrt(amplitude))/pi
    def sinc_derivative(value):
        value = np.asarray(value, dtype=float)
        small = np.abs(value) < 1e-4
        safe = np.where(small, 1., value)
        ordinary = (pi*safe*np.cos(pi*safe)-np.sin(pi*safe))/(pi*safe**2)
        series = -pi*pi*value/3+pi**4*value**3/30-pi**6*value**5/840
        return np.where(small, series, ordinary)
    def kernel_derivative(delta):
        delta = (delta+.5) % 1.-.5
        denominator, numerator = np.sinc(delta), np.sinc(likelihood.size*delta)
        slope = (likelihood.size*sinc_derivative(likelihood.size*delta)*denominator
                 -numerator*sinc_derivative(delta))/denominator**2
        return 2*numerator/denominator*slope
    probabilities = _phase_probabilities(amplitude, likelihood.bits, indices)
    derivative = (kernel_derivative(phase-indices/likelihood.size)
                  -kernel_derivative(-phase-indices/likelihood.size))/(4*pi*sqrt(amplitude*(1-amplitude)))
    if observed:
        terms = (derivative/probabilities)**2
        information = float(np.dot(likelihood.weights, terms))
    else:
        information = float(np.sum(derivative**2/probabilities))
    if not isfinite(information) or information <= 0:
        raise ValueError("Fisher information is not regular at this estimate; use likelihood_ratio.")
    return information
