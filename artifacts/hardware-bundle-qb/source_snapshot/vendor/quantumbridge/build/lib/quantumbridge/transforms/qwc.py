# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Bounded native QWC grouping, shot allocation and covariance reconstruction."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from quantumbridge.core.classical import UnsupportedFeature
from quantumbridge.diff.qnode_engine import map_measurements
from quantumbridge.diff.scalars import CoefficientSlot, CoefficientTable
from quantumbridge.qml.measurements import MeasurementRequest
from quantumbridge.qml.pauli_group import PauliGroup, PauliGroupResult
from quantumbridge.qml.shot_vector import ShotSpec
from quantumbridge.utils.math import Hamiltonian, PauliString
from .dispatch import transform
from .program import check_batch, operation_count, output_tree, resolve_limits


def allocate_shots(spec, weights, policy):
    """Floor proportional shares, then give one remainder shot in group order."""
    if not weights:
        return ()
    if policy is None or spec.partitions == (None,):
        return tuple(spec.partitions for _ in weights)
    if policy == "uniform":
        weights = (1.,) * len(weights)
    total = sum(weights)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Shot weights must have a positive finite total.")
    allocated = [[] for _ in weights]
    for shots in spec.partitions:
        if shots < len(weights):
            raise ValueError("Shot budget is insufficient for every nonempty measurement group.")
        shares = [int(np.floor(shots * (weight / total))) for weight in weights]
        remainder = shots - sum(shares)
        if not 0 <= remainder <= len(shares):
            raise ValueError("Shot allocation rounding failed its conservation check.")
        for index in range(remainder):
            shares[index] += 1
        if min(shares) < 1:
            raise ValueError("Shot budget is insufficient for a positive allocation to every effective group.")
        for group, count in zip(allocated, shares):
            group.append(count)
    return tuple(tuple(group) for group in allocated)


@dataclass(frozen=True)
class ShotAllocation:
    parent: ShotSpec
    children: tuple
    weights: tuple
    policy: object
    words: tuple


def validate_shot_allocation(parent, children):
    """Permit a changed shot shape only with a complete native sibling receipt."""
    if not children or any(len(child.measurements) != 1 or child.measurements[0].kind != "pauli_group" for child in children):
        return False
    groups = [child.measurements[0].observable for child in children]
    if not all(isinstance(group, PauliGroup) for group in groups):
        return False
    receipt = groups[0].allocation
    if not isinstance(receipt, ShotAllocation) or receipt.parent != ShotSpec.parse(parent.shots):
        return False
    if receipt.children != allocate_shots(receipt.parent, receipt.weights, receipt.policy) or len(receipt.children) != len(children):
        return False
    return all(group.allocation == receipt and group.group_index == index and group.words == receipt.words[index]
               and ShotSpec.parse(child.shots) == ShotSpec(receipt.children[index], receipt.parent.is_vector)
               for index, (group, child) in enumerate(zip(groups, children)))


@transform
def split_non_commuting(tape, grouping_strategy="default", shot_dist=None, seed=None, *, limits=None):
    """Group native Pauli expvals/variances into real shared local-basis samples.

    Default is QWC for this all-Pauli subset. Wires requires disjoint supports.
    Distributed shot vectors conserve each partition independently. Zero-weight
    groups are elided for weighted allocation; undersampled nonzero groups reject
    instead of silently dropping contributions. Seed is unused by deterministic
    allocation; sampling follows the Tape/QNode seed.
    """
    if grouping_strategy not in {"default", "qwc", "wires", None}:
        raise ValueError("Unknown native Pauli grouping strategy.")
    if not isinstance(shot_dist, (str, type(None))) or shot_dist not in {None, "uniform", "weighted"}:
        raise UnsupportedFeature("shot_distribution", "Only None/uniform/weighted deterministic shot allocation is implemented.")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (Integral, np.random.Generator)) or isinstance(seed, Integral) and seed < 0):
        raise ValueError("Shot allocation seed must be a nonnegative integer or Generator.")
    limits = resolve_limits(limits)
    check_batch((tape,), limits, shot_budget=False)
    tree, requests = output_tree(tape), []
    map_measurements(tree, lambda request: requests.append(request))
    words, word_indices, rows, offsets, kinds = [], {}, [], [], []
    coefficients, recipes = CoefficientTable(), []
    work = 0
    for request in requests:
        if request.kind not in {"expval", "var"}:
            raise UnsupportedFeature("qwc_measurement", "Native QWC grouping currently requires Pauli expval/var terminal measurements.")
        if request.shots is not None or request.seed is not None or request.wires is not None or request.metadata:
            raise UnsupportedFeature("qwc_local_measurement_options", "A shared group requires Tape-level shots/seed and ordinary Pauli measurement options.")
        observable = request.observable
        if hasattr(observable, "to_quantumbridge"):
            observable = observable.to_quantumbridge()
        elif hasattr(observable, "to_hamiltonian"):
            observable = observable.to_hamiltonian()
        if request.kind == "var" and isinstance(observable, Hamiltonian):
            raise UnsupportedFeature("qwc_variance_sum", "Variance of a sum is not linear; use expval uncertainty statistics for grouped Hamiltonians.")
        terms = observable.terms if isinstance(observable, Hamiltonian) else ((1., observable),)
        offset, row = 0., {}
        work += len(terms)
        limits.check("max_expansion_steps", work)
        for coefficient, pauli in terms:
            reference = coefficients.add(coefficient)
            coefficient = coefficients.resolve(reference)
            if not isinstance(pauli, PauliString):
                raise UnsupportedFeature("qwc_observable", "Native QWC grouping requires PauliString/Hamiltonian observables.")
            occupied = [wire for wire, _ in pauli.terms]
            if len(set(occupied)) != len(occupied) or any(not 0 <= wire < tape.circuit.num_qubits for wire in occupied):
                raise ValueError("QWC words need distinct in-register wires, including identity wires.")
            word = tuple(sorted((wire, label) for wire, label in pauli.terms if label != "I"))
            if not word:
                offset += coefficient
                recipes.append((len(rows),None,reference))
                continue
            if word not in word_indices:
                limits.check("max_postprocessing_nodes", len(words) + 1)
                word_indices[word] = len(words)
                words.append(word)
            index = word_indices[word]
            row[index] = row.get(index, 0.) + coefficient
            recipes.append((len(rows),index,reference))
        if not np.isfinite(offset) or not all(np.isfinite(value) for value in row.values()):
            raise ValueError("Combined QWC coefficients must remain finite.")
        rows.append(row)
        offsets.append(offset)
        kinds.append(request.kind)
    limits.check("max_postprocessing_nodes", len(rows) * max(1, len(words)))
    strategy = "qwc" if grouping_strategy == "default" else grouping_strategy
    groups, bases = [], []
    for index, word in enumerate(words):
        chosen = None
        for group, basis in enumerate(bases):
            work += len(word)
            limits.check("max_expansion_steps", work)
            if strategy is not None and all(wire not in basis or strategy == "qwc" and basis[wire] == label for wire, label in word):
                chosen = group
                break
        if chosen is None:
            limits.check("max_tapes", len(groups) + 1)
            chosen = len(groups)
            groups.append([])
            bases.append({})
        groups[chosen].append(index)
        bases[chosen].update(word)
    weights = [sum(abs(row.get(index, 0.)) for row in rows for index in group) for group in groups]
    if not all(np.isfinite(weight) for weight in weights):
        raise ValueError("QWC group weights must remain finite.")
    if shot_dist == "weighted":
        live_words = {index for _,index,reference in recipes if isinstance(reference,CoefficientSlot)}
        if any(weight == 0 and any(index in live_words for index in group) for group,weight in zip(groups,weights)):
            raise UnsupportedFeature("weighted_trainable_zero", "A zero-weight trainable group still needs its means for coefficient derivatives; use undistributed or uniform shots.")
        kept = [index for index, weight in enumerate(weights) if weight > 0]
        groups, bases, weights = ([items[index] for index in kept] for items in (groups, bases, weights))
    spec = ShotSpec.parse(tape.shots)
    physical_shots = sum(shots or 0 for shots in spec.partitions) * (len(groups) if shot_dist is None else int(bool(groups)))
    limits.check("max_total_shots", physical_shots)
    limits.check("max_total_operations", operation_count(tape.circuit, limits) * len(groups))
    allocation = allocate_shots(spec, weights, shot_dist)
    receipt = ShotAllocation(spec, allocation, tuple(weights), shot_dist,
                             tuple(tuple(words[index] for index in group) for group in groups))
    children, descriptions = [], []
    for number, (indices, basis) in enumerate(zip(groups, bases)):
        group_coefficients = tuple(tuple(row.get(index, 0.) for index in indices) for row in rows)
        group = PauliGroup(receipt.words[number], tuple(sorted(basis.items())), tuple(indices), group_coefficients,
                           receipt, number, limits.max_sample_values)
        child = tape.copy()
        terminal = MeasurementRequest("pauli_group", observable=group)
        child.measurements, child.output_tree = [terminal], terminal
        child.measurement, child.observable = terminal.kind, group
        child.shots = ShotSpec(allocation[number], spec.is_vector)
        children.append(child)
        descriptions.append({"group_index": number, "source_term_indices": tuple(indices), "words": group.words,
                             "basis": group.basis, "source_coefficients": group_coefficients,
                             "allocated_shots": allocation[number], "weight": weights[number]})
    check_batch(children, limits)

    def one(results, differentiating=False, parameters=None):
        if any(not isinstance(value, PauliGroupResult) or value.group != child.observable for value, child in zip(results, children)):
            raise TypeError("QWC reconstruction requires the matching native joint measurement results.")
        restored, statistics = [], []
        bound_rows, bound_offsets = [dict() for _ in rows], [0.]*len(rows)
        for source,index,reference in recipes:
            value = coefficients.resolve(reference,parameters)
            if index is None:
                bound_offsets[source] += value
            else:
                bound_rows[source][index] = bound_rows[source].get(index,0.) + value
        for position, (offset, kind) in enumerate(zip(bound_offsets, kinds)):
            mean, empirical, variance_of_mean, uncertain = offset, 0., 0., False
            for result in results:
                row_coefficients = np.asarray([bound_rows[position].get(index,0.) for index in result.group.indices])
                contribution = sum(coefficient*mean for coefficient,mean in zip(row_coefficients,result.means))
                mean += contribution if differentiating else float(contribution)
                if differentiating:
                    continue
                empirical += float(row_coefficients @ result.covariance @ row_coefficients)
                if result.shots is not None and np.any(row_coefficients):
                    if result.sample_covariance is None:
                        uncertain = True
                    else:
                        variance_of_mean += float(row_coefficients @ result.sample_covariance @ row_coefficients) / result.shots
            value = mean if kind == "expval" else (1. - mean * mean if differentiating else max(0., 1. - mean * mean)) if rows[position] else 0.
            restored.append(value)
            statistics.append({"source_measurement_index": position, "kind": kind, "expectation": mean,
                               "variance": max(0., empirical),
                               "variance_semantics": "sum of within-group observable variances; not the full quantum variance across incompatible groups",
                               "standard_error": None if uncertain or kind == "var" else float(np.sqrt(max(0., variance_of_mean))),
                               "within_group_covariance_included": True})
        values = iter(restored)
        return map_measurements(tree, lambda _: next(values)), statistics

    def process(results, *, parameters=None, differentiating=False):
        if len(results) != len(children):
            raise ValueError("QWC result count must match the actual child Tape count.")
        if spec.is_vector:
            if any(not isinstance(result, tuple) or len(result) != len(spec.partitions) for result in results):
                raise ValueError("QWC shot partition count mismatch.")
            processed = [one(tuple(result[partition] for result in results),differentiating,parameters) for partition in range(len(spec.partitions))]
            value = tuple(item[0] for item in processed)
        else:
            processed = [one(results,differentiating,parameters)]
            value = processed[0][0]
        if differentiating:
            return value
        process.last_metadata = {"grouping_strategy": grouping_strategy, "effective_strategy": strategy,
                                 "shot_distribution": shot_dist, "parent_shots": spec.partitions,
                                 "groups": descriptions, "partitions": [
                                     {"partition_index": index, "source_measurements": stats,
                                      "total_physical_shots": sum(counts[index] or 0 for counts in allocation),
                                      "semantics": "analytic" if spec.partitions[index] is None else "finite-shot shared-group estimate"}
                                     for index, (_, stats) in enumerate(processed)],
                                 "zero_weight_policy": "weighted zero-contribution groups omitted; other zero terms retained",
                                 "allocation_rule": "floor proportional shares; one remainder shot to each earliest group; reject any undersampled effective group; preserve vector partitions"}
        return value

    process.native_evaluate = process
    process.native_variance = "var" in kinds
    process.native_parameters = tuple(coefficients.parameters)
    process.native_linear_quantum_outputs = "var" not in kinds
    process.native_shot_distribution = shot_dist
    return tuple(children), process
