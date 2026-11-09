# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Finance was copied.
"""Executable portfolio optimization adapter for Qiskit Finance compatibility."""

from __future__ import annotations

from importlib import metadata, util
from typing import Any, Sequence

from quantumbridge.compat.qiskit_finance.portfolio_examples import tutorial_synthetic_portfolio_problem
from quantumbridge.compat.qiskit_finance.portfolio_native import (
    PORTFOLIO_WARNING,
    PortfolioProblem,
    build_portfolio_problem_native,
    portfolio_objective_value,
    selection_to_bitstring,
    solve_portfolio_exact_native,
)
from quantumbridge.compat.qiskit_finance.portfolio_result import PortfolioOptimizationResult

UPSTREAM_WARNING = (
    "This path calls installed upstream Qiskit Finance and related optional packages. "
    "QuantumBridge records provenance but does not claim ownership, parity, "
    "or production finance support."
)

_PACKAGES = {
    "qiskit_finance": "qiskit-finance",
    "qiskit_optimization": "qiskit-optimization",
    "qiskit_algorithms": "qiskit-algorithms",
    "qiskit_aer": "qiskit-aer",
}


def dependency_available() -> bool:
    """Return whether the core upstream portfolio stack is importable."""

    report = validate_portfolio_dependencies()
    return bool(report["ready_for_upstream_exact"])


def get_upstream_version(package: str = "qiskit-finance") -> str | None:
    """Return an installed upstream package version when available."""

    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def validate_portfolio_dependencies(*, require_aer: bool = False) -> dict[str, Any]:
    """Report optional upstream portfolio dependency availability."""

    modules = {}
    versions = {}
    for module_name, package_name in _PACKAGES.items():
        modules[module_name] = util.find_spec(module_name) is not None
        versions[package_name] = get_upstream_version(package_name)
    exact_modules = ("qiskit_finance", "qiskit_optimization", "qiskit_algorithms")
    exact_ready = all(modules[name] for name in exact_modules)
    sampling_ready = exact_ready and (modules["qiskit_aer"] or not require_aer)
    missing = [name for name in exact_modules if not modules[name]]
    if require_aer and not modules["qiskit_aer"]:
        missing.append("qiskit_aer")
    return {
        "modules": modules,
        "versions": versions,
        "ready_for_upstream_exact": exact_ready,
        "ready_for_upstream_sampling": sampling_ready,
        "missing": tuple(missing),
        "network_required": False,
        "cloud_required": False,
        "credentials_required": False,
        "advisory": True,
        "production_ready": False,
    }


def run_portfolio_optimization_native(
    problem: PortfolioProblem | None = None,
    *,
    expected_returns: Sequence[float] | None = None,
    covariances: Sequence[Sequence[float]] | None = None,
    risk_factor: float | None = None,
    budget: int | None = None,
    metadata: dict[str, object] | None = None,
) -> PortfolioOptimizationResult:
    """Run QuantumBridge's native educational exact portfolio subset."""

    if problem is None and all(value is None for value in (expected_returns, covariances, risk_factor, budget)):
        problem = tutorial_synthetic_portfolio_problem()
        metadata = {"example": "tutorial-synthetic-four-asset", **dict(metadata or {})}
    return solve_portfolio_exact_native(
        problem,
        expected_returns=expected_returns,
        covariances=covariances,
        risk_factor=risk_factor,
        budget=budget,
        metadata=metadata,
    )


def run_portfolio_optimization_upstream(
    problem: PortfolioProblem | None = None,
    *,
    method: str = "exact",
    expected_returns: Sequence[float] | None = None,
    covariances: Sequence[Sequence[float]] | None = None,
    risk_factor: float | None = None,
    budget: int | None = None,
    **kwargs: Any,
) -> PortfolioOptimizationResult:
    """Run an installed upstream Qiskit Finance portfolio path."""

    normalized = method.lower().replace("-", "_")
    if normalized in {"exact", "numpy", "numpy_minimum_eigensolver"}:
        return run_portfolio_optimization_exact_upstream(
            problem,
            expected_returns=expected_returns,
            covariances=covariances,
            risk_factor=risk_factor,
            budget=budget,
        )
    if normalized in {"sampling_vqe", "vqe"}:
        return run_portfolio_optimization_sampling_vqe_upstream(
            problem,
            expected_returns=expected_returns,
            covariances=covariances,
            risk_factor=risk_factor,
            budget=budget,
            **kwargs,
        )
    if normalized == "qaoa":
        return run_portfolio_optimization_qaoa_upstream(
            problem,
            expected_returns=expected_returns,
            covariances=covariances,
            risk_factor=risk_factor,
            budget=budget,
            **kwargs,
        )
    raise ValueError(f"unsupported upstream portfolio method: {method}")


def run_portfolio_optimization_exact_upstream(
    problem: PortfolioProblem | None = None,
    *,
    expected_returns: Sequence[float] | None = None,
    covariances: Sequence[Sequence[float]] | None = None,
    risk_factor: float | None = None,
    budget: int | None = None,
) -> PortfolioOptimizationResult:
    """Run the upstream exact solver when optional dependencies are installed."""

    resolved = _resolve_problem(problem, expected_returns, covariances, risk_factor, budget)
    _require_upstream_exact()
    portfolio_class = _import_portfolio_optimization()
    from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    quadratic_program = portfolio_class(
        expected_returns=list(resolved.expected_returns),
        covariances=[list(row) for row in resolved.covariances],
        risk_factor=resolved.risk_factor,
        budget=resolved.budget,
    ).to_quadratic_program()
    optimizer = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    upstream_result = optimizer.solve(quadratic_program)
    return _wrap_upstream_result(upstream_result, resolved, method="upstream-exact")


def run_portfolio_optimization_sampling_vqe_upstream(
    problem: PortfolioProblem | None = None,
    *,
    expected_returns: Sequence[float] | None = None,
    covariances: Sequence[Sequence[float]] | None = None,
    risk_factor: float | None = None,
    budget: int | None = None,
    reps: int = 2,
    maxiter: int = 100,
) -> PortfolioOptimizationResult:
    """Run upstream SamplingVQE portfolio optimization when installed."""

    resolved = _resolve_problem(problem, expected_returns, covariances, risk_factor, budget)
    _require_upstream_exact()
    portfolio_class = _import_portfolio_optimization()
    sampler = _make_sampler()
    from qiskit.circuit.library import TwoLocal
    from qiskit_algorithms import SamplingVQE
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    quadratic_program = portfolio_class(
        expected_returns=list(resolved.expected_returns),
        covariances=[list(row) for row in resolved.covariances],
        risk_factor=resolved.risk_factor,
        budget=resolved.budget,
    ).to_quadratic_program()
    ansatz = TwoLocal(resolved.num_assets, "ry", "cz", reps=int(reps), entanglement="full")
    optimizer = MinimumEigenOptimizer(SamplingVQE(sampler, ansatz, COBYLA(maxiter=int(maxiter))))
    upstream_result = optimizer.solve(quadratic_program)
    return _wrap_upstream_result(upstream_result, resolved, method="upstream-sampling-vqe")


def run_portfolio_optimization_qaoa_upstream(
    problem: PortfolioProblem | None = None,
    *,
    expected_returns: Sequence[float] | None = None,
    covariances: Sequence[Sequence[float]] | None = None,
    risk_factor: float | None = None,
    budget: int | None = None,
    reps: int = 2,
    maxiter: int = 100,
) -> PortfolioOptimizationResult:
    """Run upstream QAOA portfolio optimization when installed."""

    resolved = _resolve_problem(problem, expected_returns, covariances, risk_factor, budget)
    _require_upstream_exact()
    portfolio_class = _import_portfolio_optimization()
    sampler = _make_sampler()
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_optimization.algorithms import MinimumEigenOptimizer

    quadratic_program = portfolio_class(
        expected_returns=list(resolved.expected_returns),
        covariances=[list(row) for row in resolved.covariances],
        risk_factor=resolved.risk_factor,
        budget=resolved.budget,
    ).to_quadratic_program()
    optimizer = MinimumEigenOptimizer(QAOA(sampler, COBYLA(maxiter=int(maxiter)), reps=int(reps)))
    upstream_result = optimizer.solve(quadratic_program)
    return _wrap_upstream_result(upstream_result, resolved, method="upstream-qaoa")


def compare_native_and_upstream_exact(
    problem: PortfolioProblem | None = None,
) -> dict[str, PortfolioOptimizationResult | bool]:
    """Run native and upstream exact paths and compare their selections."""

    resolved = problem or tutorial_synthetic_portfolio_problem()
    native = run_portfolio_optimization_native(resolved)
    upstream = run_portfolio_optimization_exact_upstream(resolved)
    return {
        "native": native,
        "upstream": upstream,
        "same_selection": native.selection == upstream.selection,
        "same_objective": abs(native.objective_value - upstream.objective_value) < 1e-8,
    }


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
    if expected_returns is None:
        return tutorial_synthetic_portfolio_problem()
    if covariances is None or risk_factor is None or budget is None:
        raise ValueError("covariances, risk_factor, and budget are required with expected_returns")
    return build_portfolio_problem_native(expected_returns, covariances, risk_factor, budget)


def _require_upstream_exact() -> None:
    report = validate_portfolio_dependencies()
    if not report["ready_for_upstream_exact"]:
        missing = ", ".join(report["missing"]) or "unknown"
        raise ImportError(
            "Qiskit Finance upstream portfolio optimization requires optional packages: "
            f"{missing}. Install the qiskit-finance, qiskit-optimization, and qiskit-algorithms extras."
        )


def _import_portfolio_optimization():
    try:
        from qiskit_finance.applications.optimization import PortfolioOptimization
    except ImportError:
        from qiskit_finance.applications import PortfolioOptimization
    return PortfolioOptimization


def _make_sampler():
    try:
        from qiskit.primitives import Sampler

        return Sampler()
    except (ImportError, AttributeError):
        from qiskit.primitives import StatevectorSampler

        return StatevectorSampler()


def _wrap_upstream_result(
    upstream_result: Any,
    problem: PortfolioProblem,
    *,
    method: str,
) -> PortfolioOptimizationResult:
    selection = tuple(int(round(float(value))) for value in getattr(upstream_result, "x", ()))
    if not selection:
        raise ValueError("upstream optimizer result did not expose a portfolio selection")
    objective = getattr(upstream_result, "fval", None)
    if objective is None:
        objective = portfolio_objective_value(
            selection,
            problem.expected_returns,
            problem.covariances,
            problem.risk_factor,
        )
    probabilities = _extract_probabilities(upstream_result, selection)
    result = PortfolioOptimizationResult(
        selection=selection,
        objective_value=float(objective),
        method=method,
        path="upstream-passthrough",
        probabilities=probabilities,
        samples=_extract_samples(upstream_result, problem),
        expected_returns=problem.expected_returns,
        covariances=problem.covariances,
        risk_factor=problem.risk_factor,
        budget=problem.budget,
        upstream_version=get_upstream_version("qiskit-finance"),
        warnings=(PORTFOLIO_WARNING, UPSTREAM_WARNING),
        provenance={
            "adapter_package": "qiskit-finance",
            "official_endorsement": False,
            "native_subset": None,
            "upstream_passthrough": True,
            "copied_upstream_source": False,
            "cloud_access": False,
            "credentials_read": False,
        },
        metadata={
            "objective": "minimize risk_factor * x^T Sigma x - expected_returns^T x",
            "constraint": "sum(selection) == budget",
            "source": "installed upstream optional dependencies",
        },
    )
    result.validate()
    return result


def _extract_probabilities(upstream_result: Any, selection: tuple[int, ...]) -> dict[str, float]:
    bitstring = selection_to_bitstring(selection)
    samples = getattr(upstream_result, "samples", None)
    if not samples:
        return {bitstring: 1.0}
    probabilities: dict[str, float] = {}
    for sample in samples:
        x = getattr(sample, "x", None)
        probability = getattr(sample, "probability", None)
        if x is None or probability is None:
            continue
        probabilities[selection_to_bitstring(int(round(float(value))) for value in x)] = float(probability)
    return probabilities or {bitstring: 1.0}


def _extract_samples(upstream_result: Any, problem: PortfolioProblem) -> tuple[dict[str, Any], ...]:
    rows = []
    for sample in getattr(upstream_result, "samples", ()) or ():
        x = getattr(sample, "x", None)
        if x is None:
            continue
        selection = tuple(int(round(float(value))) for value in x)
        rows.append(
            {
                "selection": list(selection),
                "bitstring": selection_to_bitstring(selection),
                "objective_value": float(
                    getattr(
                        sample,
                        "fval",
                        portfolio_objective_value(
                            selection,
                            problem.expected_returns,
                            problem.covariances,
                            problem.risk_factor,
                        ),
                    )
                ),
                "probability": float(getattr(sample, "probability", 0.0)),
                "feasible": bool(sum(selection) == problem.budget),
            }
        )
    rows.sort(key=lambda row: (not row["feasible"], row["objective_value"], row["bitstring"]))
    return tuple(rows)
