# Copyright 2026 QuantumBridge Contributors. Licensed under Apache-2.0.
# Independent implementation from mathematical identities; no SDK source copied.
"""Bounded finance kernels using the existing native Circuit and simulator.

All bitstrings here are q0-first. State loading is exponential in the number of
qubits; these are transparent small examples, not a QRAM or speedup claim.
"""

from __future__ import annotations

from math import atan2, sqrt

import numpy as np

from quantumbridge.core import Circuit
from quantumbridge.devices import StatevectorDevice


def _vector(values, name):
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    data = np.asarray(raw, dtype=float)
    if data.ndim != 1 or not data.size or not np.all(np.isfinite(data)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return data


def _width(size):
    if size < 2 or size > 16 or size & (size - 1):
        raise ValueError("finance examples require 2, 4, 8 or 16 states (1-4 qubits)")
    return size.bit_length() - 1


def labels(width):
    return tuple(format(i, f"0{width}b") for i in range(1 << width))


def _multiplex_ry(circuit, angles, target):
    # Walsh expansion of a prefix-conditioned RY; CX conjugation flips its sign.
    controls = tuple(range(target))
    size = 1 << target
    for mask in range(size):
        coefficient = sum(
            angle * (-1 if bin(prefix & mask).count("1") % 2 else 1)
            for prefix, angle in enumerate(angles)
        ) / size
        if abs(coefficient) < 1e-15:
            continue
        selected = [wire for wire in controls if mask & (1 << (target - 1 - wire))]
        for wire in selected:
            circuit.cx(wire, target)
        circuit.ry(coefficient, target)
        for wire in reversed(selected):
            circuit.cx(wire, target)


def prepare_real_state(amplitudes, *, name="finance-state"):
    """Prepare a normalized signed real vector using only RY and CX gates."""
    data = _vector(amplitudes, "amplitudes")
    width = _width(int(data.size))
    scale = float(np.max(np.abs(data)))
    if scale == 0:
        raise ValueError("amplitudes must not be zero")
    data = data / scale
    data = data / np.linalg.norm(data)
    circuit = Circuit(width, width, name=name)
    for target in range(width):
        blocks = data.reshape(1 << target, -1)
        angles = []
        for block in blocks:
            half = len(block) // 2
            if target == width - 1:
                left, right = float(block[0]), float(block[1])
            else:
                left, right = np.linalg.norm(block[:half]), np.linalg.norm(block[half:])
            angles.append(2 * atan2(right, left) if left or right else 0.0)
        _multiplex_ry(circuit, angles, target)
    return circuit


def probability_circuit(probabilities, *, name="finance-distribution"):
    data = _vector(probabilities, "probabilities")
    _width(int(data.size))
    if np.any(data < 0) or not np.isclose(data.sum(), 1.0, atol=1e-12, rtol=0):
        raise ValueError("probabilities must be nonnegative and sum to one")
    return prepare_real_state(np.sqrt(data), name=name)


def expectation_summary(values, probabilities, shots=None):
    values = _vector(values, "values")
    probs = _vector(probabilities, "probabilities")
    if values.shape != probs.shape or np.any(probs < 0) or not np.isclose(probs.sum(), 1):
        raise ValueError("values and normalized probabilities must match")
    mean = float(values @ probs)
    variance = max(0.0, float((values - mean) ** 2 @ probs))
    output = {"mean": mean, "distribution_std": sqrt(variance)}
    if shots is not None:
        if type(shots) is not int or shots < 2:
            raise ValueError("at least two shots are required for a sampling standard error")
        standard_error = sqrt(variance / (shots - 1))
        output.update(standard_error=standard_error,
                      approximate_95pct_interval=[mean - 1.96 * standard_error, mean + 1.96 * standard_error])
    return output


def risk_measures(losses, probabilities, confidence=0.95):
    """Discrete VaR and expected shortfall, including fractional mass at VaR."""
    values = _vector(losses, "losses")
    probs = _vector(probabilities, "probabilities")
    if values.shape != probs.shape or np.any(probs < 0) or not np.isclose(probs.sum(), 1):
        raise ValueError("losses and normalized probabilities must match")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    order = np.argsort(values)
    cumulative = np.cumsum(probs[order])
    index = min(int(np.searchsorted(cumulative, confidence, side="left")), len(values) - 1)
    var = float(values[order[index]])
    excess = float(np.maximum(values - var, 0) @ probs)
    return {"confidence": float(confidence), "expected_loss": float(values @ probs),
            "VaR": var, "CVaR_expected_shortfall": var + excess / (1 - confidence),
            "cdf": cumulative.tolist(), "sorted_losses": values[order].tolist()}


def qaoa_circuit(costs, gamma, beta):
    """One QAOA layer for a bounded quadratic binary cost truth table."""
    costs = _vector(costs, "costs")
    width = _width(int(costs.size))
    if not np.all(np.isfinite([gamma, beta])):
        raise ValueError("QAOA angles must be finite")
    coefficients = []
    for mask in range(len(costs)):
        coefficient = sum(value * (-1 if bin(index & mask).count("1") % 2 else 1)
                          for index, value in enumerate(costs)) / len(costs)
        if bin(mask).count("1") > 2 and abs(coefficient) > 1e-10:
            raise ValueError("finance QAOA accepts quadratic objectives only")
        if mask and abs(coefficient) > 1e-14:
            coefficients.append((mask, coefficient))
    circuit = Circuit(width, width, name="finance-qaoa-p1")
    for wire in range(width):
        circuit.h(wire)
    for mask, coefficient in coefficients:
        wires = [wire for wire in range(width) if mask & (1 << (width - 1 - wire))]
        if len(wires) == 2:
            circuit.cx(wires[0], wires[1])
        circuit.rz(2 * float(gamma) * coefficient, wires[-1])
        if len(wires) == 2:
            circuit.cx(wires[0], wires[1])
    for wire in range(width):
        circuit.rx(2 * float(beta), wire)
    return circuit


def train_qaoa(costs):
    """Small deterministic local grid + coordinate refinement; no cloud training."""
    costs = _vector(costs, "costs")
    _width(int(costs.size))
    normalized = (costs - costs.min()) / max(float(np.ptp(costs)), 1e-12)
    device = StatevectorDevice()
    trace = []
    best = (float("inf"), 0.0, 0.0)

    def evaluate(gamma, beta):
        circuit = qaoa_circuit(normalized, gamma, beta)
        probs = device.probabilities(circuit)
        value = float(sum(costs[int(key, 2)] * p for key, p in probs.items()))
        return value

    for gamma in np.linspace(0, 4 * np.pi, 17):
        for beta in np.linspace(0, np.pi, 13, endpoint=False):
            value = evaluate(gamma, beta)
            if value < best[0]:
                best = (value, float(gamma), float(beta))
                trace.append({"evaluation": len(trace), "expected_cost": value,
                              "gamma": best[1], "beta": best[2], "phase": "local-grid"})
    step = 0.25
    for iteration in range(8):
        for dg, db in ((step, 0), (-step, 0), (0, step), (0, -step)):
            gamma, beta = best[1] + dg, best[2] + db
            value = evaluate(gamma, beta)
            if value < best[0]:
                best = (value, gamma, beta)
        trace.append({"evaluation": len(trace), "expected_cost": best[0],
                      "gamma": best[1], "beta": best[2], "phase": "local-refine"})
        step *= 0.6
    return qaoa_circuit(normalized, best[1], best[2]), trace
