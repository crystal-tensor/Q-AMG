"""Budgeted block-coordinate quantum sampling, preserving the full original problem.

Unchanged context is explicit; no variable is discarded. This is a heuristic,
not a guarantee of a global optimum or a quantum speedup.
"""
from __future__ import annotations

import time

import numpy as np

from .problem import StructureProblem
from .quantum import Candidate, PlanResult, QuantumStructurePrior
from .schema import PairRule, UnaryRule


def condition_problem(problem: StructureProblem, active: list[int], fixed: tuple[int, ...]):
    mapping = {old: new for new, old in enumerate(active)}
    groups = [problem.groups[i].model_copy(deep=True) for i in active]
    sub = StructureProblem(problem.spec, groups)
    offset = sum(g.options[fixed[i]].cost for i, g in enumerate(problem.groups) if i not in mapping)
    for rule in problem.unary:
        if rule.group in mapping:
            sub.unary.append(UnaryRule(group=mapping[rule.group], option=rule.option, reason=rule.reason))
        elif fixed[rule.group] == rule.option:
            raise ValueError("Cannot condition on an infeasible fixed option")
    for pair in problem.pairs:
        (a, ka), (b, kb) = pair.left, pair.right
        if a in mapping and b in mapping:
            sub.pairs.append(PairRule(left=(mapping[a], ka), right=(mapping[b], kb),
                                      cost=pair.cost, forbidden=pair.forbidden, reason=pair.reason))
        elif a not in mapping and b not in mapping:
            if fixed[a] == ka and fixed[b] == kb:
                if pair.forbidden:
                    raise ValueError("Cannot condition on an infeasible fixed pair")
                offset += pair.cost
        else:
            inside, option, outside, outside_option = (a, ka, b, kb) if a in mapping else (b, kb, a, ka)
            if fixed[outside] == outside_option:
                if pair.forbidden:
                    sub.unary.append(UnaryRule(group=mapping[inside], option=option,
                                                reason=f"conditioned:{pair.reason}"))
                else:
                    groups[mapping[inside]].options[option].cost += pair.cost
    return sub, offset


def plan_decomposed(backend, problem, settings, seed, target=None) -> PlanResult:
    start = time.perf_counter()
    cap = min(settings.max_qubits, backend.capabilities().max_qubits)
    per_group = 1 if settings.encoding == "compact_binary" else 2
    block_size = cap//per_group
    if block_size < 1:
        raise ValueError("Quantum capacity cannot encode even one choice group")
    rng = np.random.default_rng(seed)
    state = None
    # A random feasible CONTEXT, not an optimized answer. It is not returned as a quantum solution.
    initialization_checks = 0
    for _ in range(256):
        initialization_checks += 1
        trial = tuple(int(rng.integers(len(g.options))) for g in problem.groups)
        if not problem.violations(trial):
            state = trial
            break
    if state is None:
        raise ValueError("Could not initialize a feasible context within the explicit rejection budget")
    initialization_s = time.perf_counter()-start
    initial_state = state
    qubo = problem.encode()
    trace, candidates, covered = [], {}, set()
    resources = {"decomposed": True, "original_qubo_variables": qubo.n,
                 "encoding": settings.encoding, "logical_qubits": 0,
                 "compiled_physical_qubits": 0, "ancillas": 0, "shots": 0,
                 "backend_calls": 0, "two_qubit_gates": 0,
                 "initialization": "bounded_random_feasible_context_not_optimized",
                 "initialization_checks": initialization_checks, "initial_context": initial_state,
                 "block_size_groups": block_size, "passes_requested": settings.decomposition_passes}
    hit = None
    conditioning_s = 0.
    for sweep in range(settings.decomposition_passes):
        for first in range(0, len(problem.groups), block_size):
            active = list(range(first, min(first+block_size, len(problem.groups))))
            before = time.perf_counter()
            sub, constant = condition_problem(problem, active, state)
            conditioning_s += time.perf_counter()-before
            remaining = settings.timeout_s-(time.perf_counter()-start)
            if remaining <= 0:
                raise TimeoutError("Decomposition exhausted the total planning budget")
            local_settings = settings.model_copy(update={"timeout_s": remaining})
            old_state = state
            result = QuantumStructurePrior(backend).plan(
                sub, local_settings, int(rng.integers(0, 2**31-1)))
            if result.best is None:
                raise ValueError("No measured feasible block candidate; no classical fallback")
            chosen = list(state)
            for group, choice in zip(active, sub.selection(result.best.bits), strict=True):
                chosen[group] = choice
            state = tuple(chosen)
            if problem.violations(state):
                raise RuntimeError("Conditioning/decoding broke an original constraint")
            covered.update(active)
            batch_trace = []
            for batch in result.trace:
                expanded = {}
                for bits, count in batch["counts"].items():
                    s = sub.selection(bits)
                    if s is None:
                        continue
                    global_state = list(old_state)
                    for group, choice in zip(active, s, strict=True):
                        global_state[group] = choice
                    global_bits = problem.bits(global_state)
                    expanded[global_bits] = expanded.get(global_bits, 0)+count
                batch_trace.append({**batch, "counts": expanded,
                                    "local_structure_counts": batch["counts"],
                                    "conditioned_groups": active, "fixed_before": old_state})
            elapsed = time.perf_counter()-start
            entry = {"pass": sweep, "active_groups": active, "fixed_before": old_state,
                     "selected_after": state, "fixed_objective_offset": constant,
                     "subproblem": sub.as_dict(), "subplan": result.as_dict(),
                     "circuit_qasm3": result.circuit_qasm3,
                     "counts": {problem.bits(state): result.best.count},
                     "raw_measurements": batch_trace,
                     "note": "Composite structure from conditioned block measurements, not one simultaneous full-register measurement"}
            trace.append(entry)
            for key in ["logical_qubits", "compiled_physical_qubits"]:
                resources[key] = max(resources[key], result.resources[key])
            for key in ["shots", "backend_calls", "two_qubit_gates"]:
                resources[key] += result.resources[key]
            if len(covered) == len(problem.groups):
                bits = problem.bits(state)
                candidates[bits] = Candidate(bits, qubo.energy(bits), True, result.best.count,
                                              result.best.first_job_id, elapsed)
                if target and target(bits):
                    hit = elapsed
                    break
        if hit is not None:
            break
    resources["groups_covered_by_quantum_updates"] = sorted(covered)
    resources["capabilities"] = result.resources["capabilities"]
    resources["subproblem_count"] = len(trace)
    return PlanResult(method="decomposed_qaoa", execution_source=backend.capabilities().execution_source,
                      candidates=sorted(candidates.values(), key=lambda c: c.energy),
                      timings={"wall_s": time.perf_counter()-start,
                               "context_initialization_s": initialization_s,
                               "conditioning_s": conditioning_s,
                               "subproblem_planning_s": sum(t["subplan"]["timings"]["wall_s"] for t in trace)},
                      trace=trace, resources=resources, target_hit_s=hit,
                      status="target_reached" if hit else "completed")
