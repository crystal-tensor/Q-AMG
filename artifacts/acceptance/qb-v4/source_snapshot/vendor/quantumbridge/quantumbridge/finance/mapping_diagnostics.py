# Copyright 2026 QuantumBridge Contributors. Licensed under Apache-2.0.
"""Offline count-order hypotheses and non-transferable noise diagnostics."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from itertools import permutations
from math import factorial, log

import numpy as np

from quantumbridge.serialization import deserialize_circuit

from .execution import probabilities_from_counts


def ordered_mapping(audit):
    raw = audit["native_compiled"]["physical_measurement_map"]
    mapping = {int(key): value for key, value in raw.items()}
    if set(mapping) != set(range(len(mapping))) or len(set(mapping.values())) != len(mapping):
        raise ValueError("conflicting or incomplete compiled measurement map")
    return [mapping[i] for i in range(len(mapping))]


def gate_profile(source):
    circuit = deserialize_circuit(source, "qasm2", strict=False)
    levels, families, count = {}, set(), 0
    for op in circuit.operations:
        wires = op.controls + op.targets
        if op.name == "barrier":
            level = max((levels.get(wire, 0) for wire in wires), default=0)
            levels.update({wire: level for wire in wires})
            continue
        families.add(op.name)
        count += 1
        level = max((levels.get(wire, 0) for wire in wires), default=0) + 1
        levels.update({wire: level for wire in wires})
    return {"gate_families": sorted(families), "unitary_operation_count": count,
            "logical_layer_depth": max(levels.values(), default=0),
            "depth_scope": "unit-duration gate layers, barriers synchronize; excludes terminal measurement and physical timing"}


def count_order_hypotheses(counts, ideal, physical_order):
    """All <=24 candidates; closeness never certifies an unknown noise/bit order."""
    width = len(physical_order)
    if width not in (2, 3, 4) or len(set(physical_order)) != width:
        raise ValueError("unique bounded physical measurement mapping required")
    _, _, shots = probabilities_from_counts(counts, width)
    ideal = np.asarray(ideal, dtype=float)
    if (ideal.shape != (1 << width,) or not np.all(np.isfinite(ideal)) or np.any(ideal < 0)
            or not np.isclose(ideal.sum(), 1, atol=1e-10, rtol=0)):
        raise ValueError("normalized bounded ideal probabilities required")
    sorted_positions = tuple(sorted(range(width), key=lambda i: physical_order[i]))
    candidates = []
    for permutation in permutations(range(width)):
        # permutation[c] is the raw string position hypothesized to carry c.
        decoded = {"".join(label[i] for i in permutation): value for label, value in counts.items()}
        sampled = np.array([decoded.get(format(i, f"0{width}b"), 0) / shots for i in range(1 << width)])
        impossible = any(value > 0 and ideal[int(label, 2)] == 0 for label, value in decoded.items())
        likelihood = None if impossible else sum(value * log(ideal[int(label, 2)])
                                                 for label, value in decoded.items() if value)
        aliases = []
        if permutation == tuple(range(width)):
            aliases.append("identity")
        if permutation == tuple(reversed(range(width))):
            aliases.append("reverse")
        # Hypothesis: raw labels are sorted by increasing physical index.
        if permutation == tuple(sorted_positions.index(i) for i in range(width)):
            aliases.append("ascending_physical_register_HYPOTHESIS")
        marginal = [sum((sampled[i] - ideal[i]) for i in range(1 << width) if format(i, f"0{width}b")[bit] == "1")
                    for bit in range(width)]
        candidates.append({"raw_position_for_each_classical_bit": list(permutation), "aliases": aliases,
                           "decoded_counts": decoded, "ideal_probabilities": ideal.tolist(),
                           "noiseless_multinomial_log_likelihood_up_to_constant": likelihood,
                           "positive_counts_at_zero_ideal_probability": impossible,
                           "TVD": float(abs(sampled - ideal).sum() / 2), "marginal_deviation": marginal,
                           "compiled_map_compatibility": "NOT_EXCLUDED; metadata does not declare raw-label serialization"})
    return {"status": "AMBIGUOUS", "selected_permutation": None, "candidates": candidates,
            "score_scope": "diagnostic comparison to noiseless ideal, unknown hardware noise; no hypothesis is certified by best fit"}


def compare_tasks(old, probe):
    left, right = ordered_mapping(old), ordered_mapping(probe)
    same_set, same_order = set(left) == set(right), left == right
    old_id, new_id = old["fields"]["calibration_id"]["value"], probe["fields"]["calibration_id"]["value"]
    same_calibration = old_id is not None and old_id == new_id
    try:
        gap = (datetime.fromisoformat(probe["fields"]["created"]["value"]) -
               datetime.fromisoformat(old["fields"]["finished"]["value"])).total_seconds()
    except (TypeError, ValueError):
        gap = None
    classification = "NOT_TRANSFERABLE" if not same_set else "PARTIALLY_COMPARABLE"
    if same_set and same_order and same_calibration and gap is not None and old["submitted_options"] == probe["submitted_options"]:
        classification = "DIRECTLY_COMPARABLE"
    if gap is None and same_set:
        classification = "UNKNOWN"
    return {"historical_task_id": old["task_id"], "diagnostic_task_id": probe["task_id"],
            "physical_set_matches": same_set, "ordered_measurement_mapping_matches": same_order,
            "measurement_connections_identical": same_order, "finish_to_probe_creation_seconds": gap,
            "calibration_id_available_and_equal": same_calibration,
            "old_gate_profile": old["gate_profile"], "probe_gate_profile": probe["gate_profile"],
            "same_gate_profile": old["gate_profile"] == probe["gate_profile"],
            "same_submitted_source": old["source_sha256"] == probe["source_sha256"],
            "old_readout_correction": old["fields"]["corrected"], "probe_readout_correction": probe["fields"]["corrected"],
            "classification": classification, "error_model_transfer_proven": False,
            "qualification": "layout comparison only; drift, X/preparation errors, pulse/calibration and count serialization are not identified"}


def joint_error_bootstrap(counts, prepared, *, draws=2000, seed=20261023):
    width = len(prepared)
    if width not in (3, 4) or set(prepared) not in ({"0"}, {"1"}):
        raise ValueError("bounded uniform prepared bitstring required")
    if type(draws) is not int or not 100 <= draws <= 10000:
        raise ValueError("bootstrap draws must be 100..10000")
    _, p, shots = probabilities_from_counts(counts, width)
    labels = [format(i, f"0{width}b") for i in range(1 << width)]
    errors = np.array([[int(a != b) for a, b in zip(label, prepared)] for label in labels])
    samples = np.random.default_rng(seed).multinomial(shots, p, size=draws) / shots
    pairs = []
    for i in range(width):
        for j in range(i + 1, width):
            both = errors[:, i] * errors[:, j]
            empirical = float(np.dot(p, both))
            independent = float(np.dot(p, errors[:, i]) * np.dot(p, errors[:, j]))
            covariance = samples @ both - (samples @ errors[:, i]) * (samples @ errors[:, j])
            pairs.append({"raw_positions": [i, j], "joint_error": empirical,
                          "independent_marginals_prediction": independent,
                          "excess_joint_error": empirical - independent,
                          "empirical_bootstrap_covariance_percentile_95": np.quantile(covariance, [.025, .975]).tolist()})
    return {"prepared": prepared, "joint_error_patterns": {"".join(map(str, row)): counts.get(label, 0)
                                                            for label, row in zip(labels, errors)},
            "pairs": pairs, "bootstrap_draws": draws, "seed": seed,
            "scope": "IID empirical multinomial bootstrap, cannot identify hidden drift/readout vs preparation or recover unobserved outcomes"}


def experiment_identifiability(patterns):
    if not patterns or len({len(pattern) for pattern in patterns}) != 1:
        raise ValueError("same-width basis patterns required")
    width = len(patterns[0])
    if width not in (3, 4) or any(set(pattern) - {"0", "1"} for pattern in patterns):
        raise ValueError("bounded binary patterns required")
    observations = Counter(tuple("".join(pattern[i] for i in permutation) for pattern in patterns)
                           for permutation in permutations(range(width)))
    return {"patterns_q0_first": patterns, "tasks": len(patterns), "shots_per_task": 1024,
            "total_shots": 1024 * len(patterns), "permutations": factorial(width),
            "distinct_noiseless_observations": len(observations), "worst_indistinguishable_permutations": max(observations.values()),
            "condition": "same fixed mapping across tasks, declared raw label semantics, noiseless preparations; noise can destroy identifiability",
            "hardware_authorization": "NOT_AUTHORIZED_BY_THIS_OFFLINE_PLAN", "execute": False}
