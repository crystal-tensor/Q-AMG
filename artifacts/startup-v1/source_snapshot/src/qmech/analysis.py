"""Independent, failure-inclusive reanalysis of immutable benchmark records.

These are descriptive intervals over this synthetic corpus, clustered by instance.
No statistical calculation establishes that a provider actually used a QPU.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from itertools import combinations, product
from pathlib import Path

import numpy as np

from .benchmark import CLASSICAL_METHODS, dump
from .problem import build_problem
from .schema import DesignSpec


def wilson(successes: int, trials: int) -> list[float] | None:
    """Wilson interval for independent *instances*, never for correlated seeds."""
    if not trials:
        return None
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1 + z*z/trials
    center = (rate + z*z/(2*trials))/denominator
    half = z*math.sqrt(rate*(1-rate)/trials + z*z/(4*trials*trials))/denominator
    return [max(0., center-half), min(1., center+half)]


def mean_interval(values: list[float], rng, repeats: int) -> dict:
    if not values:
        return {"mean": None, "instance_bootstrap_ci95": None, "instances": 0}
    array = np.asarray(values, dtype=float)
    boot = rng.choice(array, (repeats, len(array)), replace=True).mean(axis=1)
    return {"mean": float(array.mean()),
            "instance_bootstrap_ci95": list(map(float, np.quantile(boot, [.025, .975]))),
            "instances": len(array)}


def speed_interval(log_ratios: list[float], rng, repeats: int) -> dict:
    result = mean_interval(log_ratios, rng, repeats)
    return {"geomean_ratio": math.exp(result["mean"]),
            "instance_bootstrap_ci95": [math.exp(x) for x in result["instance_bootstrap_ci95"]],
            "paired_instances": len(log_ratios), "excluded_instances": 0}


def reanalyze(records: list[dict], manifest: dict, corpus: dict,
              bootstrap_repeats: int = 4000, bootstrap_seed: int = 20261109) -> dict:
    """Retain every registered slot, including missing/error/over-budget results.

    Failure charge is a reporting policy, NOT a measured failure latency, a
    survival estimate, or an estimate of repeated-until-success time-to-solution.
    """
    if bootstrap_repeats < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    instances, methods = manifest["instances"], manifest["methods"]
    seeds = list(range(1000, 1000 + manifest["seeds"]))
    if not instances or not methods or not seeds:
        raise ValueError("Empty registered cohort")
    expected = set(product(instances, methods, seeds))
    by_key = {}
    for row in records:
        key = (row["instance"], row["method"], row["seed"])
        if key not in expected or key in by_key:
            raise ValueError(f"Unexpected or duplicate benchmark key: {key}")
        by_key[key] = row
    cases = {case["id"]: case for case in corpus["instances"]}
    issues, evaluated = [], []
    timeout = float(manifest["budget"]["timeout_s_per_method"])
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Invalid registered timeout")
    for instance in instances:
        problem = build_problem(DesignSpec.model_validate(cases[instance]["spec"]))
        # Independent reference: original categorical expressions, no solver helper.
        optimum = min(problem.cost(s) for s in problem.assignments() if not problem.violations(s))
        threshold = optimum + manifest["target"]["relative_gap"]*max(1, abs(optimum))
        for method, seed in product(methods, seeds):
            row = by_key.get((instance, method, seed), {})
            missing = not row
            if missing:
                issues.append(f"Missing registered result: {instance}/{method}/{seed}")
            if row and (abs(row["optimum"]-optimum) > 1e-8 or
                        abs(row["threshold"]-threshold) > 1e-8):
                raise ValueError(f"Original-problem reference disagrees: {instance}/{method}/{seed}")
            bits = row.get("bits")
            feasible = bool(bits and problem.feasible(bits))
            gap = ((problem.cost(problem.selection(bits))-optimum)/max(1, abs(optimum))
                   if feasible else None)
            if row.get("relative_gap") is not None and (
                gap is None or abs(row["relative_gap"]-gap) > 1e-8
            ):
                raise ValueError("Stored relative gap disagrees with original problem")
            hit = row.get("time_to_target_s")
            amortized = float(row.get("amortized_training_s", 0))
            allowance = timeout + amortized
            wall = float(row.get("wall_s", 0))
            if not math.isfinite(wall) or wall < 0 or not math.isfinite(amortized) or amortized < 0:
                raise ValueError("Invalid measured or training time")
            if hit is not None and (not math.isfinite(hit) or hit <= 0):
                raise ValueError("Invalid recorded target time")
            target_ok = feasible and gap <= manifest["target"]["relative_gap"]+1e-9
            success = bool(hit is not None and target_ok and hit <= allowance and
                           row.get("status") not in {"error", "timeout"})
            if hit is not None and not target_ok:
                issues.append(f"Recorded hit fails independent predicate: {instance}/{method}/{seed}")
            if row.get("repair", "none") != "none":
                raise ValueError("This analysis is registered for no-repair experiments only")
            counts = Counter()
            for batch in row.get("trace", []):
                for state, count in batch.get("counts", {}).items():
                    if type(count) is not int or count <= 0:
                        raise ValueError("Invalid trace counts")
                    problem.selection(state)  # Also checks the original structure bit width.
                    counts[state] += count
            total = sum(counts.values())
            raw_feasible = sum(count for state, count in counts.items() if problem.feasible(state))
            geom_valid = row.get("geometry", {}).get("valid") is True
            end = row.get("generation_to_validated_geometry_s")
            evaluated.append({
                "instance": instance, "family": problem.spec.family,
                "variables": problem.encode().n, "method": method, "seed": seed,
                "missing": missing, "success": success, "feasible": feasible,
                "relative_gap": gap, "bits": bits, "geometry_valid": geom_valid,
                "final_target_valid": success and (geom_valid or not manifest["geometry"]),
                "recorded_time_to_target_s": hit,
                "budget_charged_time_s": hit if success else max(allowance, wall+amortized),
                "end_to_end_charged_s": (float(end)+amortized if success and geom_valid and end is not None
                                          else max(allowance, wall+amortized, float(end or 0)+amortized)),
                "observed_draws": total, "observed_feasible_draws": raw_feasible,
                "unique_observed_feasible_candidates": sum(problem.feasible(state) for state in counts),
                "execution_source": row.get("execution_source", "unavailable"),
                "resources": {key: row.get("resources", {}).get(key) for key in
                              ["shots", "evaluations", "backend_calls", "two_qubit_gates"]},
            })
    rng = np.random.default_rng(bootstrap_seed)

    def describe(rows):
        groups = [[r for r in rows if r["instance"] == name]
                  for name in sorted({r["instance"] for r in rows})]
        diversity = []
        for group in groups:
            finals = [r["bits"] for r in group if r["final_target_valid"]]
            distances = [sum(a != b for a, b in zip(left, right, strict=True))/len(left)
                         for left, right in combinations(finals, 2)]
            diversity.append({"instance": group[0]["instance"], "registered_runs": len(group),
                              "successful_final_runs": len(finals), "distinct_final_targets": len(set(finals)),
                              "pairwise_normalized_hamming_successes": (float(np.mean(distances))
                                                                         if distances else None)})
        all_pass = sum(all(r["success"] for r in group) for group in groups)
        raw_draws = sum(r["observed_draws"] for r in rows)
        raw_valid = sum(r["observed_feasible_draws"] for r in rows)
        gap_means = [float(np.mean(gaps)) for group in groups
                     if (gaps := [r["relative_gap"] for r in group if r["relative_gap"] is not None])]
        return {
            "registered_runs": len(rows), "successes": sum(r["success"] for r in rows),
            "success_rate": sum(r["success"] for r in rows)/len(rows),
            "failures_including_missing": sum(not r["success"] for r in rows),
            "instances_all_seeds_successful": {"passed": all_pass, "total": len(groups),
                                               "rate": all_pass/len(groups), "wilson_ci95": wilson(all_pass, len(groups))},
            "budget_charged_time_s": mean_interval([float(np.mean([r["budget_charged_time_s"] for r in g]))
                                                     for g in groups], rng, bootstrap_repeats),
            "end_to_end_charged_s": (mean_interval([float(np.mean([r["end_to_end_charged_s"] for r in g]))
                                                        for g in groups], rng, bootstrap_repeats)
                                        if manifest["geometry"] else None),
            "relative_gap_feasible_returns_only": mean_interval(gap_means, rng, bootstrap_repeats),
            "final_model_valid_runs": sum(r["geometry_valid"] for r in rows) if manifest["geometry"] else None,
            "raw_feasible_draws": raw_valid, "raw_draws": raw_draws,
            "raw_feasible_rate_draw_weighted": raw_valid/raw_draws if raw_draws else None,
            "repair": "none; post-repair rate is identical to raw rate",
            "distinct_final_targets_per_instance": mean_interval(
                [d["distinct_final_targets"] for d in diversity], rng, bootstrap_repeats),
            "diversity_by_instance": diversity,
            "observed_candidate_pool_per_run": float(np.mean([r["unique_observed_feasible_candidates"] for r in rows])),
            "resource_means_per_run": {key: (float(np.mean(values)) if values else None) for key in
                                       ["shots", "evaluations", "backend_calls", "two_qubit_gates"]
                                       for values in [[r["resources"][key] for r in rows if r["resources"][key] is not None]]},
            "sources": dict(Counter(r["execution_source"] for r in rows)),
        }

    descriptions = {method: describe([r for r in evaluated if r["method"] == method]) for method in methods}
    strata = {}
    for family, variables in sorted({(r["family"], r["variables"]) for r in evaluated}):
        strata[f"{family}:{variables}"] = {
            method: describe([r for r in evaluated if r["method"] == method and
                              r["family"] == family and r["variables"] == variables]) for method in methods}
    speedups = {}
    classical = [m for m in methods if m in CLASSICAL_METHODS]
    for method in methods:
        if not method.startswith("qaoa") or not classical:
            continue
        metrics = {}
        for metric in ["budget_charged_time_s", "end_to_end_charged_s"]:
            if metric.startswith("end_to_end") and not manifest["geometry"]:
                continue
            paired = {}
            for instance in instances:
                paired[instance] = {m: float(np.mean([r[metric] for r in evaluated
                    if r["instance"] == instance and r["method"] == m])) for m in [method, *classical]}
            ratios = {baseline: [math.log(p[baseline]/p[method]) for p in paired.values()]
                      for baseline in classical}
            ratios["fastest_observed_classical_instancewise"] = [
                math.log(min(p[m] for m in classical)/p[method]) for p in paired.values()]
            metrics[metric] = {baseline: speed_interval(values, rng, bootstrap_repeats)
                               for baseline, values in ratios.items()}
        speedups[method] = metrics
    return {
        "schema_version": 1, "registered_slots": len(expected), "observed_rows": len(records),
        "missing_rows": len(expected)-len(records), "issues": issues,
        "review_status": "ready_within_reviewed_scope" if not issues else "needs_revision",
        "methods": descriptions, "strata": strata, "speedups": speedups,
        "policy": {
            "failure_charge": "max(registered timeout + amortized training, observed wall + amortized training); not a measured TTT",
            "success_interval": "Wilson interval for independent instances passing ALL registered seeds; run rate is descriptive",
            "other_intervals": f"{bootstrap_repeats} instance-cluster bootstrap resamples; seeds stay together; seed={bootstrap_seed}",
            "population": "Registered synthetic held-out corpus; not a claim about general industrial designs",
            "diversity": "Distinct successful final structures within each identical input across seeds; pair distances conditional on success",
            "candidate_pool": "Observed solver visits/quantum batches with unequal draw counts and early stopping; not state entropy or equal-budget diversity",
            "gap": "Conditional on feasible returns; missing/infeasible quality is not silently imputed as zero",
            "baseline_selection": "Instance-wise fastest observed baseline is descriptive and selection-optimistic; individual preregistered baselines also reported",
            "interval_limit": "Bootstrap intervals with identical observations can be degenerate; they do not prove population certainty",
        },
        "hardware_validated": False, "quantum_acceleration_target_met": False,
        "hardware_note": "Execution source labels are evidence claims, not independent physical-device verification",
    }


def analyze_experiment(directory: Path, corpus_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Choose a new analysis output; source experiments are immutable")
    raw, manifest_path = directory/"raw.jsonl", directory/"manifest.json"
    records = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    result = reanalyze(records, json.loads(manifest_path.read_text()), json.loads(corpus_path.read_text()))
    result["evidence"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [raw, manifest_path, corpus_path, Path(__file__)]}
    dump(output, result)
    return result
