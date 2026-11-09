# Copyright 2026 QuantumBridge Contributors. Licensed under Apache-2.0.
# Independent mathematical examples; no upstream SDK implementation is used.
"""Five synthetic finance case studies with explicit classical oracles."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt

import numpy as np

from quantumbridge.compat.qiskit_finance.portfolio_native import (
    build_portfolio_problem_native, portfolio_objective_value, solve_portfolio_exact_native,
)
from quantumbridge.core import Circuit

from .kernels import (
    expectation_summary, labels, prepare_real_state, probability_circuit,
    risk_measures, train_qaoa,
)


@dataclass
class FinanceCase:
    key: str
    title: str
    experiments: dict
    inputs: dict
    intermediate: dict
    reference: dict

    def analyze(self, results):
        """Compute finance quantities from measured frequencies, never ideal fallback."""
        if set(results) != set(self.experiments):
            raise ValueError("results must contain every experiment, with no extras")
        for name, result in results.items():
            probabilities = np.asarray(result["probabilities"], dtype=float)
            if (probabilities.shape != (1 << self.experiments[name].num_qubits,)
                    or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0)
                    or not np.isclose(probabilities.sum(), 1, atol=1e-10, rtol=0)):
                raise ValueError("result probabilities must be finite, normalized and match circuit width")
            if type(result["shots"]) is not int or result["shots"] < 2:
                raise ValueError("result shots must be an integer >=2")
        if self.key in {"allocation", "constraints"}:
            result = results["qaoa"]
            probabilities = np.array(result["probabilities"])
            rows = self.intermediate["candidates"]
            feasible = [row for row in rows if row["feasible"]]
            observed = [row for row in feasible if probabilities[int(row["bitstring"], 2)] > 0]
            best = min(observed, key=lambda row: row["objective"]) if observed else None
            optimal = self.reference["best_objective"]
            return {
                "feasible_probability": sum(probabilities[int(row["bitstring"], 2)] for row in feasible),
                "optimal_probability": sum(probabilities[int(row["bitstring"], 2)] for row in feasible
                                           if np.isclose(row["objective"], optimal, atol=1e-12, rtol=0)),
                "expected_penalized_cost": float(probabilities @ np.array([row["penalized_cost"] for row in rows])),
                "best_observed_feasible": best,
                "best_observed_gap": None if best is None else best["objective"] - optimal,
                "decision_rule": "best feasible sampled candidate; not necessarily the most frequent string",
            }
        if self.key == "risk":
            return {name: risk_measures(self.inputs["losses"], result["probabilities"], self.inputs["confidence"])
                    for name, result in results.items()}
        if self.key == "derivatives":
            result = results["risk_neutral_paths"]
            probs, shots = result["probabilities"], result["shots"]
            inputs = self.inputs
            discount = exp(-inputs["rate"] * inputs["maturity"])
            call = expectation_summary(self.intermediate["call_payoff"], probs, shots)
            delta = expectation_summary(self.intermediate["delta_payoff"], probs, shots)
            put = expectation_summary(self.intermediate["put_payoff"], probs, shots)
            price = discount * call["mean"]
            hedge_delta = discount * delta["mean"]
            terminal = np.array(self.intermediate["terminal_prices"])
            payoff = np.array(self.intermediate["call_payoff"])
            unhedged = price / discount - payoff
            hedged = -payoff + hedge_delta * terminal + (price - hedge_delta * inputs["spot"]) / discount
            return {
                "call_price": price, "put_price": discount * put["mean"], "delta": hedge_delta,
                "call_price_standard_error": discount * call["standard_error"],
                "call_approximate_95pct_interval": [discount * bound for bound in call["approximate_95pct_interval"]],
                "delta_standard_error": discount * delta["standard_error"],
                "unhedged_terminal_pnl": unhedged.tolist(), "hedged_terminal_pnl": hedged.tolist(),
                "unhedged_pnl_std": expectation_summary(unhedged, probs)["distribution_std"],
                "hedged_pnl_std": expectation_summary(hedged, probs)["distribution_std"],
                "put_call_parity_residual": price - discount * put["mean"] - inputs["spot"] + inputs["strike"] * discount,
                "model": "3-step risk-neutral binomial; static delta hedge; not a continuous dynamic hedge",
            }
        correlations = {}
        for name in ("positive_correlation", "negative_correlation"):
            result = results[name]
            p0 = sum(p for label, p in zip(labels(4), result["probabilities"]) if label[0] == "0")
            correlations[name] = {
                "correlation": 2 * p0 - 1,
                "sampling_standard_error": 2 * sqrt(max(p0 * (1 - p0), 0) / (result["shots"] - 1)),
            }
        result = results["time_sampling"]
        moments = expectation_summary(self.inputs["asset_a_returns"], result["probabilities"], result["shots"])
        return {**correlations, "daily_return_mean": moments["mean"],
                "daily_volatility": moments["distribution_std"],
                "annualized_volatility_252": moments["distribution_std"] * sqrt(252),
                "volatility_definition": "population historical volatility, ddof=0; not a forecast"}


def _allocation_case(constrained=False):
    mu = [0.17, 0.14, 0.11, 0.09]
    covariance = [[0.12, 0.03, 0.02, 0.01], [0.03, 0.10, 0.01, 0.02],
                  [0.02, 0.01, 0.08, 0.01], [0.01, 0.02, 0.01, 0.06]]
    risk, penalty = 0.65, 1.6
    previous = np.array([1, 0, 1, 0])
    candidates = []
    for label in labels(4):
        bits = np.array(list(map(int, label)))
        objective = portfolio_objective_value(bits, mu, covariance, risk)
        violations = {"holdings": int(bits.sum()) - 2}
        if constrained:
            objective += 0.03 * float(np.abs(bits - previous).sum())
            violations.update(sector=int(bits[0] + bits[1]) - 1,
                              capital=int(np.array([2, 2, 1, 1]) @ bits) - 3)
        candidates.append({"bitstring": label, "selection": bits.tolist(),
                           "objective": objective, "feasible": all(v == 0 for v in violations.values()),
                           "violations": violations,
                           "penalized_cost": objective + penalty * sum(v * v for v in violations.values())})
    feasible = sorted((row for row in candidates if row["feasible"]), key=lambda row: row["objective"])
    all_costs = [row["penalized_cost"] for row in candidates]
    circuit, trace = train_qaoa(all_costs)
    key = "constraints" if constrained else "allocation"
    reference = {"best_selection": feasible[0]["bitstring"], "best_objective": feasible[0]["objective"],
                 "feasible_count": len(feasible),
                 "penalized_global_minimum_is_feasible": min(candidates, key=lambda row: row["penalized_cost"])["feasible"]}
    if not constrained:
        problem = build_portfolio_problem_native(mu, covariance, risk, 2)
        reference["existing_native_exact_solver"] = list(solve_portfolio_exact_native(problem).selection)
    return FinanceCase(key, "金融约束优化" if constrained else "量化投资与资产配置", {"qaoa": circuit},
                       {"asset_names": ["Synthetic A", "Synthetic B", "Synthetic C", "Synthetic D"],
                        "expected_returns": mu, "covariance": covariance, "risk_factor": risk,
                        "holdings": 2, "penalty": penalty, "previous_selection": previous.tolist(),
                        "transaction_cost_per_changed_bit": 0.03 if constrained else 0,
                        "data_source": "fixed synthetic annualized inputs, not observed market quotes"},
                       {"candidates": candidates, "local_optimizer_trace": trace}, reference)


def _risk_case():
    losses = [0, 1, 2, 4, 6, 8, 12, 20]
    probabilities = {"base": [0.52, 0.20, 0.12, 0.07, 0.04, 0.025, 0.015, 0.01],
                     "stress": [0.40, 0.18, 0.14, 0.10, 0.07, 0.05, 0.035, 0.025]}
    return FinanceCase("risk", "风险计量",
                       {name: probability_circuit(probs, name=f"risk-{name}") for name, probs in probabilities.items()},
                       {"losses": losses, "probabilities": probabilities, "confidence": 0.95,
                        "loss_unit": "synthetic monetary units", "measure": "physical scenario probabilities"},
                       {"loss_table": [{"state": state, "loss": loss, "base": probabilities["base"][i],
                                        "stress": probabilities["stress"][i]}
                                       for i, (state, loss) in enumerate(zip(labels(3), losses))]},
                       {name: risk_measures(losses, probs) for name, probs in probabilities.items()})


def _derivatives_case():
    spot, strike, rate, volatility, maturity, steps, bump = 100.0, 100.0, 0.03, 0.20, 1.0, 3, 1.0
    up = exp(volatility * sqrt(maturity / steps))
    down = 1 / up
    probability_up = (exp(rate * maturity / steps) - down) / (up - down)
    factors, probabilities = [], []
    for state in labels(steps):
        count_up = state.count("1")
        factors.append(up ** count_up * down ** (steps - count_up))
        probabilities.append(probability_up ** count_up * (1 - probability_up) ** (steps - count_up))
    factors, probabilities = np.array(factors), np.array(probabilities)
    terminal = spot * factors
    payoff = np.maximum(terminal - strike, 0)
    delta_payoff = (np.maximum((spot + bump) * factors - strike, 0)
                    - np.maximum((spot - bump) * factors - strike, 0)) / (2 * bump)
    discount = exp(-rate * maturity)
    def normal_cdf(z):
        return (1 + erf(z / sqrt(2))) / 2
    d1 = (log(spot / strike) + (rate + 0.5 * volatility ** 2) * maturity) / (volatility * sqrt(maturity))
    bs_call = spot * normal_cdf(d1) - strike * discount * normal_cdf(d1 - volatility * sqrt(maturity))
    return FinanceCase("derivatives", "衍生品定价与对冲",
                       {"risk_neutral_paths": probability_circuit(probabilities, name="risk-neutral-binomial")},
                       {"spot": spot, "strike": strike, "rate": rate, "volatility": volatility,
                        "maturity": maturity, "steps": steps, "spot_bump": bump,
                        "risk_neutral_up_probability": probability_up, "up": up, "down": down,
                        "probabilities": probabilities.tolist(), "measure": "risk-neutral Q, not physical P"},
                       {"terminal_prices": terminal.tolist(), "call_payoff": payoff.tolist(),
                        "put_payoff": np.maximum(strike - terminal, 0).tolist(), "delta_payoff": delta_payoff.tolist()},
                       {"call_price": float(discount * (payoff @ probabilities)),
                        "delta": float(discount * (delta_payoff @ probabilities)), "black_scholes_call": bs_call,
                        "discretization_note": "binomial-vs-Black-Scholes difference is model error, not shot error"})


def _correlation_case():
    asset_a = np.array([0.01, -0.02, 0.015, 0.005, -0.01, 0.025, -0.005, -0.015])
    noise = np.array([0.004, -0.001, -0.003, 0.002, -0.004, 0.001, 0.003, -0.002])
    asset_b, asset_c = 0.6 * asset_a + noise, -0.8 * asset_a + noise
    centered = asset_a - asset_a.mean()
    normalized = centered / np.linalg.norm(centered)
    experiments, reference = {}, {}
    for name, values in (("positive_correlation", asset_b), ("negative_correlation", asset_c)):
        other = values - values.mean()
        other /= np.linalg.norm(other)
        joint = np.concatenate((normalized, other)) / sqrt(2)
        circuit = prepare_real_state(joint, name=name)
        circuit.h(0)
        experiments[name] = circuit
        reference[name] = float(normalized @ other)
    uniform = Circuit(3, 3, name="uniform-historical-time-index")
    for wire in range(3):
        uniform.h(wire)
    experiments["time_sampling"] = uniform
    reference.update(daily_return_mean=float(asset_a.mean()), daily_volatility=float(asset_a.std()),
                     annualized_volatility_252=float(asset_a.std() * sqrt(252)))
    return FinanceCase("correlation", "相关性与市场波动", experiments,
                       {"asset_a_returns": asset_a.tolist(), "asset_b_returns": asset_b.tolist(),
                        "asset_c_returns": asset_c.tolist(), "data_source": "8 synthetic daily returns; no live data"},
                       {"centered_asset_a": centered.tolist(), "normalized_asset_a": normalized.tolist(),
                        "interference_identity": "P(ancilla=0)=(1+Pearson correlation)/2",
                        "sample_size": 8}, reference)


def build_finance_cases():
    """Construct the five bounded case studies without credentials or network."""
    return {case.key: case for case in (_allocation_case(), _risk_case(), _derivatives_case(),
                                        _allocation_case(True), _correlation_case())}
