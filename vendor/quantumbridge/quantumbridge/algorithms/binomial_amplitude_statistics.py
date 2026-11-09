# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Joint binomial likelihood for amplitude-amplification experiments, NumPy only."""
from fractions import Fraction
from math import isfinite, pi, sin, sqrt
from numbers import Integral, Real
from statistics import NormalDist

import numpy as np

from quantumbridge.core.gate_limits import current_gate_limits
from .amplitude_statistics import _crossing


def evaluation_schedule(value):
    limits = current_gate_limits()
    if isinstance(value, Integral) and not isinstance(value, bool):
        if value < 0:
            raise ValueError("evaluation_schedule must be nonnegative.")
        if value > limits.max_repeat_count.bit_length():
            raise MemoryError("Exponential evaluation schedule exceeds max_repeat_count.")
        value = [0, *(1 << index for index in range(int(value)))]
    if type(value) not in (list, tuple) or not value:
        raise ValueError("evaluation_schedule must be a nonempty list or nonnegative integer.")
    limits.check("max_total_operations", len(value))
    if any(isinstance(power, bool) or not isinstance(power, Integral) for power in value):
        raise TypeError("Grover powers must be integers.")
    if any(power < 0 for power in value):
        raise ValueError("Grover powers must be nonnegative.")
    result = tuple(map(int, value))
    limits.check("max_repeat_count", sum(result))
    return result


def search_boundaries(schedule):
    frequencies = [2*power+1 for power in schedule]
    current_gate_limits().check("max_decomposition_work", sum(frequencies)+len(frequencies))
    # Every sin/cos zero is included. Within each open interval the binomial
    # log likelihood has nonpositive second derivative, so its maximum is global
    # after comparing all interval maxima, not just a sampled search grid.
    points = {Fraction(index, 2*frequency) for frequency in frequencies for index in range(frequency+1)}
    boundaries = [float(point)*pi for point in sorted(points)]
    current_gate_limits().check("max_decomposition_work", (len(boundaries)-1)*84*len(frequencies))
    return boundaries


class BinomialLikelihood:
    def __init__(self, schedule, good, totals):
        self.schedule = evaluation_schedule(schedule)
        if len(good) != len(self.schedule) or len(totals) != len(self.schedule):
            raise ValueError("One success count and total are required per Grover power.")
        for successes, total in zip(good, totals):
            if any(isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) for value in (successes, total)):
                raise ValueError("Binomial counts must be finite real numbers.")
            if total <= 0 or not 0 <= successes <= total:
                raise ValueError("Binomial counts require 0 <= successes <= positive total.")
            if isinstance(total, Integral) and total > 1 << 53:
                raise ValueError("Integer exposure exceeds exact floating-count representation.")
        self.good, self.totals = np.asarray(good, dtype=float), np.asarray(totals, dtype=float)
        self.bad = self.totals-self.good
        self.frequencies = np.asarray([2*power+1 for power in self.schedule])
        self.work = 0

    def terms(self, theta):
        if isinstance(theta, bool) or not isinstance(theta, Real) or not isfinite(theta) or not 0 <= theta <= pi/2:
            raise ValueError("The likelihood angle must be finite and in [0, pi/2].")
        if theta == 0:
            return np.zeros(len(self.schedule)), np.ones(len(self.schedule))
        if theta == pi/2:
            return np.ones(len(self.schedule)), np.zeros(len(self.schedule))
        angles = self.frequencies*theta
        return np.sin(angles)**2, np.cos(angles)**2

    def __call__(self, theta):
        self.work += len(self.schedule)
        current_gate_limits().check("max_decomposition_work", self.work)
        good, bad = self.terms(theta)
        positive, negative = self.good > 0, self.bad > 0
        with np.errstate(divide="ignore"):
            return float(np.dot(self.good[positive], np.log(good[positive]))
                +np.dot(self.bad[negative], np.log(bad[negative])))

    def maxima(self):
        boundaries = search_boundaries(self.schedule)
        return [self.interval_maximum(lower, upper) for lower, upper in zip(boundaries, boundaries[1:])]

    def negative(self, theta):
        return -self(theta)

    def score(self, theta):
        self.work += len(self.schedule)
        current_gate_limits().check("max_decomposition_work", self.work)
        angles = self.frequencies*theta
        positive, negative = self.good > 0, self.bad > 0
        with np.errstate(divide="ignore"):
            return float(2*(np.dot(self.frequencies[positive], self.good[positive]/np.tan(angles[positive]))
                -np.dot(self.frequencies[negative], self.bad[negative]*np.tan(angles[negative]))))

    def interval_maximum(self, lower, upper):
        endpoints = [(lower, self(lower)), (upper, self(upper))]
        for _ in range(80):
            middle = (lower+upper)/2
            derivative = self.score(middle)
            if upper-lower <= 2e-13 or derivative == 0:
                break
            if derivative > 0:
                lower = middle
            else:
                upper = middle
        return max([*endpoints, (middle, self(middle))], key=lambda item: item[1])

    def maximum(self):
        maxima = self.maxima()
        if 0 in self.schedule and np.all(self.good == self.totals):
            return pi/2
        if np.all(self.good == 0):
            return 0.
        peak = max(value for _, value in maxima)
        tolerance = 64*np.finfo(float).eps*(1+abs(peak))
        return min(theta for theta, value in maxima if value >= peak-tolerance)


def result_likelihood(result):
    totals = getattr(result, "total_counts", None)
    if totals is None:
        if result.shots is None:
            raise ValueError("Results require shot totals for each circuit.")
        totals = [result.shots]*len(result.evaluation_schedule)
    return BinomialLikelihood(result.evaluation_schedule, result.good_counts, totals)


def fisher_information(likelihood, theta, observed=False):
    amplitude = sin(theta)**2
    if not 0 < amplitude < 1:
        if observed:
            raise ValueError("Observed information requires a regular interior estimate.")
        return float("inf")
    variance = amplitude*(1-amplitude)
    if not observed:
        return float(np.dot(likelihood.totals, likelihood.frequencies**2)/variance)
    if not isfinite(likelihood(theta)):
        raise ValueError("Estimate has zero likelihood for observed counts.")
    good, bad = likelihood.terms(theta)
    positive, negative = likelihood.good > 0, likelihood.bad > 0
    frequencies = likelihood.frequencies
    curvature = 2*(np.dot(frequencies[positive]**2, likelihood.good[positive]/good[positive])
        +np.dot(frequencies[negative]**2, likelihood.bad[negative]/bad[negative]))
    angles = frequencies*theta
    score = 2*(np.dot(frequencies[positive], likelihood.good[positive]*np.cos(angles[positive])/np.sin(angles[positive]))
        -np.dot(frequencies[negative], likelihood.bad[negative]*np.sin(angles[negative])/np.cos(angles[negative])))
    value = curvature/(4*variance)+score*(1-2*amplitude)/(4*variance**1.5)
    if not isfinite(value) or value <= 0:
        raise ValueError("Observed likelihood curvature is not positive at this estimate.")
    return float(value)


def confidence_interval(result, alpha, kind="fisher"):
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not isfinite(alpha) or not 0 < alpha < 1 or alpha/2 == 0:
        raise ValueError("alpha must be a finite representable confidence tail in (0, 1).")
    aliases = {"fi": "fisher", "oi": "observed_fisher", "lr": "likelihood_ratio"}
    kind = aliases.get(kind, kind)
    if kind not in {"fisher", "observed_fisher", "likelihood_ratio"}:
        raise NotImplementedError("Unknown multi-circuit confidence interval kind.")
    likelihood = result_likelihood(result)
    theta = result.theta
    value = likelihood(theta)
    if not isfinite(value):
        raise ValueError("Estimate has zero likelihood for the measured data.")
    amplitude, z = sin(theta)**2, -NormalDist().inv_cdf(alpha/2)
    if kind != "likelihood_ratio":
        if not 0 < amplitude < 1:
            raise ValueError("Endpoint Wald intervals are singular; use likelihood_ratio.")
        radius = z/sqrt(fisher_information(likelihood, theta, kind == "observed_fisher"))
        return amplitude-radius, amplitude+radius
    threshold = value-.5*z*z
    cut = lambda point: likelihood(point)-threshold
    boundaries, regions = search_boundaries(likelihood.schedule), []
    for lower, upper in zip(boundaries, boundaries[1:]):
        peak, height = likelihood.interval_maximum(lower, upper)
        if height < threshold:
            continue
        left = lower if cut(lower) >= 0 else _crossing(cut, lower, peak)
        right = upper if cut(upper) >= 0 else _crossing(cut, peak, upper)
        regions.append((left, right))
    if not regions:
        raise ValueError("No likelihood confidence region was found.")
    # The public two-number interval is the enclosing hull, even if a custom
    # schedule leaves several disconnected and physically ambiguous regions.
    return sin(min(left for left, _ in regions))**2, sin(max(right for _, right in regions))**2
