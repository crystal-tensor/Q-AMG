# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Finance was copied.
"""Native educational portfolio optimization helpers.

This module implements a small, deterministic mean-variance portfolio
optimization subset. It is intended for compatibility examples and tests, not
for production finance, investment advice, or market-data workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

from quantumbridge.compat.qiskit_finance.portfolio_result import PortfolioOptimizationResult

PORTFOLIO_WARNING = (
    "This is an educational portfolio-optimization compatibility path. "
    "It is not production finance, investment advice, pricing support, "
    "or a complete Qiskit Finance replacement."
)


@dataclass(frozen=True)
class PortfolioProblem:
    """Validated binary mean-variance portfolio problem."""

    expected_returns: tuple[float, ...]
    covariances: tuple[tuple[float, ...], ...]
    risk_factor: float
    budget: int

    @property
    def num_assets(self) -> int:
        return len(self.expected_returns)

    def validate(self) -> bool:
        if not self.expected_returns:
            raise ValueError("expected_returns must be non-empty")
        if len(self.covariances) != self.num_assets:
            raise ValueError("covariances row count must match expected_returns")
        for row in self.covariances:
            if len(row) != self.num_assets:
                raise ValueError("covariances must be a square matrix")
        if self.risk_factor < 0:
            raise ValueError("risk_factor must be non-negative")
        if not 0 <= self.budget <= self.num_assets:
            raise ValueError("budget must be between 0 and the number of assets")
        return True

    def to_inputs(self) -> dict[str, object]:
        self.validate()
        return {
            "expected_returns": self.expected_returns,
            "covariances": self.covariances,
            "risk_factor": self.risk_factor,
            "budget": self.budget,
        }


def build_portfolio_problem_native(
    expected_returns: Sequence[float],
    covariances: Sequence[Sequence[float]],
    risk_factor: float,
    budget: int,
) -> PortfolioProblem:
    """Build a validated native mean-variance portfolio problem."""

    problem = PortfolioProblem(
        expected_returns=_float_tuple(expected_returns, "expected_returns"),
        covariances=_covariance_tuple(covariances),
        risk_factor=float(risk_factor),
        budget=int(budget),
    )
    problem.validate()
    return problem


def portfolio_objective_value(
    selection: Sequence[int],
    expected_returns: Sequence[float],
    covariances: Sequence[Sequence[float]],
    risk_factor: float,
) -> float:
    """Evaluate q * x^T Sigma x - mu^T x for a binary selection."""

    x = tuple(int(value) for value in selection)
    returns = _float_tuple(expected_returns, "expected_returns")
    covariance = _covariance_tuple(covariances)
    if len(x) != len(returns):
        raise ValueError("selection length must match expected_returns")
    if len(covariance) != len(x):
        raise ValueError("covariances row count must match selection")
    risk = float(risk_factor)
    if risk < 0:
        raise ValueError("risk_factor must be non-negative")
    variance = 0.0
    for i, left in enumerate(x):
        for j, right in enumerate(x):
            variance += left * covariance[i][j] * right
    expected_return = sum(bit * mu for bit, mu in zip(x, returns))
    return risk * variance - expected_return


def enumerate_budget_selections(num_assets: int, budget: int) -> tuple[tuple[int, ...], ...]:
    """Enumerate all binary selections satisfying a fixed budget."""

    count = int(num_assets)
    selected = int(budget)
    if count < 0:
        raise ValueError("num_assets must be non-negative")
    if not 0 <= selected <= count:
        raise ValueError("budget must be between 0 and num_assets")
    rows = []
    for indices in combinations(range(count), selected):
        row = [0] * count
        for index in indices:
            row[index] = 1
        rows.append(tuple(row))
    return tuple(rows)


def solve_portfolio_exact_native(
    problem: PortfolioProblem | None = None,
    *,
    expected_returns: Sequence[float] | None = None,
    covariances: Sequence[Sequence[float]] | None = None,
    risk_factor: float | None = None,
    budget: int | None = None,
    metadata: dict[str, object] | None = None,
) -> PortfolioOptimizationResult:
    """Solve the native educational portfolio subset by exact enumeration."""

    resolved = _resolve_problem(problem, expected_returns, covariances, risk_factor, budget)
    samples = []
    for selection in enumerate_budget_selections(resolved.num_assets, resolved.budget):
        value = portfolio_objective_value(
            selection,
            resolved.expected_returns,
            resolved.covariances,
            resolved.risk_factor,
        )
        bitstring = selection_to_bitstring(selection)
        samples.append(
            {
                "selection": list(selection),
                "bitstring": bitstring,
                "objective_value": value,
                "feasible": True,
            }
        )
    samples.sort(key=lambda row: (row["objective_value"], row["bitstring"]))
    best = samples[0]
    selection = tuple(int(value) for value in best["selection"])
    result = PortfolioOptimizationResult(
        selection=selection,
        objective_value=float(best["objective_value"]),
        method="exact-enumeration",
        path="quantumbridge-native",
        probabilities={best["bitstring"]: 1.0},
        samples=tuple(samples),
        expected_returns=resolved.expected_returns,
        covariances=resolved.covariances,
        risk_factor=resolved.risk_factor,
        budget=resolved.budget,
        upstream_version=None,
        warnings=(PORTFOLIO_WARNING,),
        provenance={
            "adapter_package": "qiskit-finance",
            "official_endorsement": False,
            "native_subset": "binary mean-variance exact enumeration",
            "copied_upstream_source": False,
        },
        metadata={
            "objective": "minimize risk_factor * x^T Sigma x - expected_returns^T x",
            "constraint": "sum(selection) == budget",
            "input_source": "deterministic synthetic data unless explicitly provided",
            **dict(metadata or {}),
        },
    )
    result.validate()
    return result


def selection_to_bitstring(selection: Iterable[int]) -> str:
    """Return a stable left-to-right bitstring for a portfolio selection."""

    return "".join(str(int(value)) for value in selection)


def _resolve_problem(
    problem: PortfolioProblem | None,
    expected_returns: Sequence[float] | None,
    covariances: Sequence[Sequence[float]] | None,
    risk_factor: float | None,
    budget: int | None,
) -> PortfolioProblem:
    if problem is not None:
        problem.validate()
        return problem
    missing = [
        name
        for name, value in (
            ("expected_returns", expected_returns),
            ("covariances", covariances),
            ("risk_factor", risk_factor),
            ("budget", budget),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"missing portfolio inputs: {', '.join(missing)}")
    return build_portfolio_problem_native(
        expected_returns=expected_returns or (),
        covariances=covariances or (),
        risk_factor=float(risk_factor),
        budget=int(budget),
    )


def _float_tuple(values: Sequence[float], name: str) -> tuple[float, ...]:
    try:
        row = tuple(float(value) for value in values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of numbers") from exc
    if not row:
        raise ValueError(f"{name} must be non-empty")
    return row


def _covariance_tuple(values: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    try:
        rows = tuple(tuple(float(value) for value in row) for row in values)
    except TypeError as exc:
        raise ValueError("covariances must be a matrix of numbers") from exc
    if not rows:
        raise ValueError("covariances must be non-empty")
    width = len(rows[0])
    if width == 0:
        raise ValueError("covariance rows must be non-empty")
    for row in rows:
        if len(row) != width:
            raise ValueError("covariance rows must have equal length")
    return rows
