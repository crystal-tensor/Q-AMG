"""Strong classical references on both the original problem and its QUBO.

No classical solver is called by QuantumStructurePrior. These are independent
baselines and an explicitly selected classical operating mode.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable

import numpy as np
from ortools.sat.python import cp_model

from .problem import StructureProblem
from .quantum import Candidate, PlanResult


def solve_classical(problem: StructureProblem, method: str = "cpsat", seed: int = 42,
                    evaluations: int = 2560, target: Callable[[str], bool] | None = None,
                    timeout_s: float = 120) -> PlanResult:
    start = time.perf_counter()
    qubo = problem.encode()
    encoded = time.perf_counter()
    candidates: dict[str, Candidate] = {}
    hit = None
    visits = 0
    status = "completed"

    def observe(bits: str, count: int = 1):
        nonlocal hit, visits
        energy = qubo.energy(bits)
        feasible = problem.feasible(bits)
        visits += count
        if bits in candidates:
            candidates[bits].count += count
        else:
            candidates[bits] = Candidate(bits, energy, feasible, count, f"classical-{method}",
                                         time.perf_counter() - start)
        if hit is None and target and target(bits):
            hit = time.perf_counter() - start
        return energy

    if method == "exact":
        for assignment in problem.assignments():
            observe(problem.bits(assignment))
            if hit is not None:
                break
            if time.perf_counter() - start >= timeout_s:
                status = "timeout"
                break
    elif method in {"cpsat", "cpsat_qubo"}:
        model = cp_model.CpModel()
        x = [model.new_bool_var(f"x{i}") for i in range(qubo.n)]
        if method == "cpsat":
            for ids in problem.variables:
                model.add_exactly_one(x[i] for i in ids)
            objective = []
            for group, ids in zip(problem.groups, problem.variables, strict=True):
                objective.extend(o.cost*x[i] for o, i in zip(group.options, ids, strict=True))
            for rule in problem.unary:
                model.add(x[problem.variables[rule.group][rule.option]] == 0)
            for k, rule in enumerate(problem.pairs):
                a, b = (problem.variables[rule.left[0]][rule.left[1]],
                        problem.variables[rule.right[0]][rule.right[1]])
                if rule.forbidden:
                    model.add(x[a]+x[b] <= 1)
                elif rule.cost:
                    y = model.new_bool_var(f"pair{k}")
                    model.add_multiplication_equality(y, [x[a], x[b]])
                    objective.append(rule.cost*y)
            model.minimize(sum(objective))
        else:
            objective = [int(c)*x[i] for i, c in enumerate(qubo.linear)]
            for k, ((i, j), coefficient) in enumerate(qubo.quadratic.items()):
                y = model.new_bool_var(f"product{k}")
                model.add_multiplication_equality(y, [x[i], x[j]])
                objective.append(int(coefficient)*y)
            model.minimize(sum(objective)+int(qubo.offset))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(.001, timeout_s-(time.perf_counter()-start))
        solver.parameters.random_seed = seed
        solver.parameters.num_search_workers = 2

        class Visitor(cp_model.CpSolverSolutionCallback):
            def on_solution_callback(self):
                observe("".join(str(self.value(v)) for v in x))
                if hit is not None:
                    self.stop_search()

        solved = solver.solve(model, Visitor())
        if solved in [cp_model.OPTIMAL, cp_model.FEASIBLE] and not candidates:
            observe("".join(str(solver.value(v)) for v in x))
        status = solver.status_name(solved).lower()
    elif method in {"anneal", "factorized"}:
        rng = np.random.default_rng(seed)
        state = tuple(int(rng.integers(len(g.options))) for g in problem.groups)
        energy = observe(problem.bits(state))
        k = 0
        while visits < evaluations:
            if hit is not None:
                break
            if time.perf_counter()-start >= timeout_s:
                status = "timeout"
                break
            if method == "factorized":
                proposed = tuple(int(rng.integers(len(g.options))) for g in problem.groups)
            else:
                proposed = list(state)
                index = int(rng.integers(len(state)))
                proposed[index] = (proposed[index]+1) % len(problem.groups[index].options)
                proposed = tuple(proposed)
            new_energy = observe(problem.bits(proposed))
            # Restarts are explicit and included; no hidden exact-solver warm start.
            temperature = max(.2, 12*(1-k/max(1, evaluations)))
            if new_energy <= energy or rng.random() < math.exp(min(0, (energy-new_energy)/temperature)):
                state, energy = proposed, new_energy
            if method == "anneal" and (k+1) % 128 == 0 and visits < evaluations:
                state = tuple(int(rng.integers(len(g.options))) for g in problem.groups)
                energy = observe(problem.bits(state))
            k += 1
    else:
        raise ValueError(f"Unknown classical method: {method}")
    elapsed = time.perf_counter()-start
    counts = Counter({c.bits: c.count for c in candidates.values()})
    total = sum(counts.values())
    return PlanResult(
        method=method, execution_source="classical_cpu",
        candidates=sorted(candidates.values(), key=lambda c: (not c.feasible, c.energy, -c.count)),
        timings={"encoding_s": encoded-start, "wall_s": elapsed},
        target_hit_s=hit, status="target_reached" if hit is not None else status,
        trace=[{"counts": dict(counts), "evaluations": visits,
                "raw_feasible_rate": sum(c.count for c in candidates.values() if c.feasible)/max(1, total)}],
        resources={"evaluations": visits, "search_workers": 2 if method.startswith("cpsat") else 1,
                   "representation": "qubo" if method == "cpsat_qubo" else "original_categorical",
                   "repair": "none"})

