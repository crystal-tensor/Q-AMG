# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Small real-data CQM portfolio and an explicit gate-model comparison.

ExactCQMSolver is classical enumeration, not D-Wave hardware. The gate-model
encoding uses a full diagonal feasibility penalty, not an unsupported CQM-to-QUBO
claim. Four assets keep both exhaustive checks and state preparation bounded.
"""
from dataclasses import dataclass
from itertools import product
from math import pi

import numpy as np

from quantumbridge.core import Circuit
from quantumbridge.devices import StatevectorDevice
from .market_data import load_stock_history


@dataclass
class PortfolioProblem:
    cqm: object
    history: object
    budget: float
    risk_aversion: float
    holdings: int
    tech_cap: int
    fee: float
    means: np.ndarray
    covariance: np.ndarray
    candidates: list


def build_portfolio_cqm(*, budget=220., risk_aversion=2., holdings=2,
                        tech_cap=1, fee=0.10):
    """One source-price unit per chosen stock, with remaining cash earning zero.

The returned genuine dimod CQM can also be supplied to an owner's separately
configured sampler. This module never constructs cloud clients or reads keys.
"""
    import dimod

    if (not np.all(np.isfinite([budget, risk_aversion, fee])) or budget <= 0
            or risk_aversion < 0 or fee < 0):
        raise ValueError("Positive finite budget and nonnegative risk/fee required")
    if type(holdings) is not int or not 1 <= holdings <= 4:
        raise ValueError("holdings must be an integer in 1..4")
    if type(tech_cap) is not int or not 0 <= tech_cap <= 3:
        raise ValueError("tech_cap must be an integer in 0..3")
    history = load_stock_history()
    returns = history.returns()
    means, covariance = returns.mean(axis=0)*252, np.cov(returns.T, ddof=1)*252
    prices = history.prices[-1]
    weights = prices/budget
    x = [dimod.Binary(s) for s in history.symbols]
    objective = sum(risk_aversion*covariance[i, j]*weights[i]*weights[j]*x[i]*x[j]
                    for i in range(4) for j in range(4))
    objective += sum((-means[i]*weights[i]+fee/budget)*x[i] for i in range(4))
    cqm = dimod.ConstrainedQuadraticModel()
    cqm.set_objective(objective)
    cqm.add_constraint(sum((prices[i]+fee)*x[i] for i in range(4)) <= budget,
                       label="cash_budget")
    cqm.add_constraint(sum(x) == holdings, label="holdings")
    cqm.add_constraint(x[0]+x[1]+x[3] <= tech_cap, label="technology_cap")
    candidates = []
    for bits in product((0, 1), repeat=4):
        a = np.array(bits)
        w = weights*a
        cost = float((prices+fee)@a)
        feasible = bool(cost <= budget+1e-9 and sum(bits) == holdings
                        and bits[0]+bits[1]+bits[3] <= tech_cap)
        energy = float(risk_aversion*w@covariance@w-means@w+fee*a.sum()/budget)
        candidates.append(dict(bits="".join(map(str, bits)), cost=cost, feasible=feasible,
                               objective=energy, expected_return=float(means@w),
                               volatility=float(np.sqrt(max(0., w@covariance@w)))))
    return PortfolioProblem(cqm, history, float(budget), float(risk_aversion),
                            holdings, tech_cap, float(fee), means, covariance, candidates)


def diagonal_qaoa(costs, gamma, beta):
    """Exact Walsh parity synthesis of a four-bit diagonal Hamiltonian, p=1."""
    costs = np.asarray(costs, dtype=float)
    if costs.shape != (16,) or not np.all(np.isfinite(costs)):
        raise ValueError("16 finite costs are required")
    if not np.all(np.isfinite([gamma, beta])):
        raise ValueError("finite angles required")
    circuit = Circuit(4, name="real-stock-diagonal-qaoa")
    for wire in range(4):
        circuit.h(wire)
    for mask in range(16):
        coefficient = sum(v*(-1)**(bin(i & mask).count("1"))
                          for i, v in enumerate(costs))/16
        if mask == 0:
            circuit.global_phase = -float(gamma)*coefficient
            continue
        wires = [w for w in range(4) if mask & (1 << (3-w))]
        if abs(coefficient) < 1e-14:
            continue
        for wire in wires[:-1]:
            circuit.cx(wire, wires[-1])
        circuit.rz(2*float(gamma)*coefficient, wires[-1])
        for wire in reversed(wires[:-1]):
            circuit.cx(wire, wires[-1])
    for wire in range(4):
        circuit.rx(2*float(beta), wire)
    return circuit


def solve_portfolio(problem=None, *, shots=1024, seed=20261102, grid=13):
    """Cross-check real CQM output, then actually sample a native QAOA circuit."""
    import dimod

    if type(shots) is not int or not 32 <= shots <= 8192:
        raise ValueError("shots must be in 32..8192")
    if type(grid) is not int or not 3 <= grid <= 25:
        raise ValueError("grid must be in 3..25")
    problem = build_portfolio_cqm() if problem is None else problem
    samples = dimod.ExactCQMSolver().sample_cqm(problem.cqm)
    checked = {}
    for row in samples.data(fields=["sample", "energy", "is_feasible"]):
        key = "".join(str(int(row.sample[s])) for s in problem.history.symbols)
        checked[key] = (float(row.energy), bool(row.is_feasible))
    for row in problem.candidates:
        energy, allowed = checked[row["bits"]]
        # Independent summation orders need roundoff allowance, not a financial tolerance.
        if not np.isclose(energy, row["objective"], atol=1e-12,
                          rtol=64*np.finfo(float).eps) or allowed != row["feasible"]:
            raise RuntimeError("CQM disagrees with independent financial constraints")
    feasible = [row for row in problem.candidates if row["feasible"]]
    output = dict(backend="dimod.ExactCQMSolver (classical CPU)",
                  dimod_version=dimod.__version__, hardware_jobs=0,
                  source=problem.history.provenance, candidates=problem.candidates,
                  constraints={"budget_usd": problem.budget, "holdings": problem.holdings,
                               "technology_cap": problem.tech_cap, "fee_usd_per_trade": problem.fee},
                  training_dates=[problem.history.dates[0], problem.history.dates[-1]],
                  annual_means=problem.means.tolist(), annual_covariance=problem.covariance.tolist())
    if not feasible:
        return dict(output, status="infeasible", optimum=None, qaoa=None)
    best = min(feasible, key=lambda row: row["objective"])
    energy = np.array([row["objective"] for row in problem.candidates])
    allowed = np.array([row["feasible"] for row in problem.candidates])
    penalty = float(np.ptp(energy)+1.)
    penalized = energy+penalty*(~allowed)
    scaled = (penalized-penalized.min())/np.ptp(penalized)
    device = StatevectorDevice()
    trace, chosen = [], None
    for gamma in np.linspace(0, 2*pi, grid):
        for beta in np.linspace(0, pi, grid):
            circuit = diagonal_qaoa(scaled, gamma, beta)
            probs = np.abs(device.statevector(circuit))**2
            score = float(probs@scaled)
            if chosen is None or score < chosen[0]:
                trace.append(score)
                chosen = (score, float(gamma), float(beta), circuit, probs)
    score, gamma, beta, circuit, probs = chosen
    sampled = device.counts(circuit, shots=shots, seed=seed)
    counts = {row["bits"]: sampled.get(row["bits"], 0) for row in problem.candidates}
    observed = [row for row in feasible if counts[row["bits"]]]
    winner = min(observed, key=lambda row: row["objective"]) if observed else None
    holdout = load_stock_history("2016-01-01", "2016-03-01")
    bits = np.array(list(map(int, best["bits"])))
    replay = ((holdout.prices-problem.history.prices[-1])@bits-problem.fee*bits.sum())/problem.budget
    equal_weight = (holdout.prices/problem.history.prices[-1]-1).mean(axis=1)
    return dict(output, status="solved", optimum=best, qaoa={
        "backend": "QuantumBridge.StatevectorDevice", "qubits": 4, "shots": shots,
        "gamma": gamma, "beta": beta, "counts": counts, "ideal_probabilities": probs.tolist(),
        "feasible_probability": float(probs[allowed].sum()),
        "observed_feasible_rate": sum(counts[r["bits"]] for r in feasible)/shots,
        "best_observed": winner, "observed_gap": None if winner is None else winner["objective"]-best["objective"],
        "trace": trace, "penalty": penalty, "circuit": circuit,
        "encoding": "Full diagonal feasibility penalty, including higher-order terms; not direct CQM/QUBO execution",
    }, holdout={"dates": list(holdout.dates), "portfolio_return": replay.tolist(),
                "equal_weight_return": equal_weight.tolist(),
                "note": "2016 holdout never used for fitting; source-price P&L, cash zero yield; excludes exit fees and unverified corporate actions"})
