# Copyright 2026 QuantumBridge Contributors. Licensed under Apache-2.0.
"""Sampling diagnostics, not a financial or hardware certification policy."""

from __future__ import annotations

from math import log, sqrt
from statistics import NormalDist

import numpy as np

from .execution import probabilities_from_counts
from .kernels import risk_measures


def wilson_interval(successes, trials, *, confidence=0.95):
    """Two-sided Wilson score interval, including zero/all successes."""
    if type(successes) is not int or type(trials) is not int or not 0 <= successes <= trials or trials < 1:
        raise ValueError("integer binomial observations required")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    probability = successes / trials
    denominator = 1 + z * z / trials
    center = (probability + z * z / (2 * trials)) / denominator
    radius = z / denominator * sqrt(probability * (1 - probability) / trials + z * z / (4 * trials * trials))
    return [max(0.0, center - radius), min(1.0, center + radius)]


def distribution_diagnostics(counts, ideal):
    ideal = np.asarray(ideal, dtype=float)
    size = int(ideal.size)
    if (ideal.ndim != 1 or size < 2 or size > 64 or size & (size - 1)
            or not np.all(np.isfinite(ideal)) or np.any(ideal < 0)
            or not np.isclose(ideal.sum(), 1, atol=1e-10, rtol=0)):
        raise ValueError("normalized ideal distribution of 1..6 qubits required")
    normalized, sampled, shots = probabilities_from_counts(counts, size.bit_length() - 1)
    intervals = {format(i, f"0{size.bit_length()-1}b"): wilson_interval(
        normalized.get(format(i, f"0{size.bit_length()-1}b"), 0), shots,
        confidence=1 - 0.05 / size) for i in range(size)}
    return {"total_variation_distance": float(np.abs(ideal - sampled).sum() / 2),
            "simultaneous_probability_intervals": intervals,
            "interval_method": "Bonferroni-adjusted Wilson score; approximate, not exact simultaneous coverage",
            "shots": shots, "status": "INCONCLUSIVE",
            "status_reason": "no owner-approved business tolerance; diagnostics are not a certification"}


def optimization_diagnostics(case, counts):
    _, probabilities, shots = probabilities_from_counts(counts, 4)
    rows = case.intermediate["candidates"]
    feasible = [row for row in rows if row["feasible"]]
    feasible_count = sum(counts.get(row["bitstring"], 0) for row in feasible)
    optimal = [row for row in feasible if np.isclose(row["objective"], case.reference["best_objective"], atol=1e-12, rtol=0)]
    optimal_count = sum(counts.get(row["bitstring"], 0) for row in optimal)
    objective_sum = sum(probabilities[int(row["bitstring"], 2)] * row["objective"] for row in feasible)
    return {"feasible_shots": feasible_count, "feasible_probability": feasible_count / shots,
            "feasible_wilson_95": wilson_interval(feasible_count, shots),
            "unconditional_optimum_probability": optimal_count / shots,
            "unconditional_optimum_wilson_95": wilson_interval(optimal_count, shots),
            "conditional_optimum_probability": optimal_count / feasible_count if feasible_count else None,
            "conditional_optimum_wilson_95": wilson_interval(optimal_count, feasible_count) if feasible_count else None,
            "conditional_mean_objective": objective_sum * shots / feasible_count if feasible_count else None,
            "feasible_states": [{**row, "measured_count": counts.get(row["bitstring"], 0)} for row in feasible]}


def risk_sampling_diagnostics(losses, counts, *, confidence=0.95, bootstrap_samples=1000, seed=20261023):
    """Empirical multinomial bootstrap plus DKW quantile stability diagnostics.

    Unobserved tail states receive zero bootstrap mass. These percentile
    intervals are descriptive, not guaranteed coverage for rare unseen losses.
    DKW uses the full declared loss support and covers the sampled distribution
    under IID assumptions, not hardware bias relative to the intended model.
    """
    if type(bootstrap_samples) is not int or not 100 <= bootstrap_samples <= 10000:
        raise ValueError("bootstrap_samples must be 100..10000")
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1 or len(losses) < 2 or len(losses) > 64 or len(losses) & (len(losses) - 1):
        raise ValueError("power-of-two bounded loss support required")
    _, probabilities, shots = probabilities_from_counts(counts, len(losses).bit_length() - 1)
    point = risk_measures(losses, probabilities, confidence)
    draws = np.random.default_rng(seed).multinomial(shots, probabilities, size=bootstrap_samples)
    fields = ("expected_loss", "VaR", "CVaR_expected_shortfall")
    records = [risk_measures(losses, row / shots, confidence) for row in draws]
    intervals = {field: np.quantile([record[field] for record in records], [0.025, 0.975]).tolist() for field in fields}
    epsilon = sqrt(log(2 / 0.05) / (2 * shots))
    order = np.argsort(losses)
    cdf = np.cumsum(np.array(probabilities)[order])

    def quantile_at(level):
        if level <= 0:
            return float(losses.min())
        if level > 1:
            return float(losses.max())
        return float(losses[order[min(int(np.searchsorted(cdf, level, side="left")), len(losses) - 1)]])

    bounds = [quantile_at(confidence - epsilon), quantile_at(confidence + epsilon)]
    return {"point": point, "empirical_bootstrap_percentile_95": intervals,
            "bootstrap_samples": bootstrap_samples, "seed": seed,
            "dkw_95_cdf_epsilon": epsilon, "dkw_95_var_range": bounds,
            "var_boundary_stable": bounds[0] == bounds[1],
            "expected_tail_shots": shots * (1 - confidence),
            "unseen_loss_states": [i for i, probability in enumerate(probabilities) if probability == 0],
            "limitations": "IID sampling only; empirical bootstrap cannot recover unseen tails or hardware/model bias"}
