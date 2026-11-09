# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit Finance was copied.
"""Deterministic portfolio optimization examples."""

from __future__ import annotations

from quantumbridge.compat.qiskit_finance.portfolio_native import PortfolioProblem, build_portfolio_problem_native


def tutorial_synthetic_inputs() -> dict[str, object]:
    """Return deterministic 4-asset mean-variance inputs.

    The shape mirrors the common four-asset portfolio-optimization tutorial
    pattern while using synthetic constants owned by QuantumBridge.
    """

    return {
        "expected_returns": (0.014, 0.0008, 0.0001, 0.015),
        "covariances": (
            (0.0010, 0.0002, 0.0001, 0.0001),
            (0.0002, 0.0015, 0.0002, 0.0001),
            (0.0001, 0.0002, 0.0011, 0.0003),
            (0.0001, 0.0001, 0.0003, 0.0012),
        ),
        "risk_factor": 0.5,
        "budget": 2,
    }


def tutorial_synthetic_portfolio_problem() -> PortfolioProblem:
    """Build the deterministic educational four-asset portfolio problem."""

    inputs = tutorial_synthetic_inputs()
    return build_portfolio_problem_native(
        expected_returns=inputs["expected_returns"],
        covariances=inputs["covariances"],
        risk_factor=inputs["risk_factor"],
        budget=inputs["budget"],
    )
