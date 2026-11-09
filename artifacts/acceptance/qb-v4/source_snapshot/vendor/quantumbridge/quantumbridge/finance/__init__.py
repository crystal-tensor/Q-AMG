# Copyright 2026 QuantumBridge Contributors. Licensed under Apache-2.0.
"""Small, independently implemented finance research workflows, not trading advice."""

from .cases import build_finance_cases
from .execution import FinanceCloudSession, run_local
from .kernels import prepare_real_state, probability_circuit, risk_measures
from .market_data import load_stock_history
from .cqm_portfolio import build_portfolio_cqm, solve_portfolio
from .quantum_monte_carlo import estimate_expectation, pennylane_qmc, pricing_case, risk_case
from .hybrid_volatility import hybrid_volatility_case

__all__ = [
    "build_finance_cases", "FinanceCloudSession", "run_local",
    "prepare_real_state", "probability_circuit", "risk_measures",
    "load_stock_history", "build_portfolio_cqm", "solve_portfolio",
    "estimate_expectation", "pennylane_qmc", "pricing_case", "risk_case",
    "hybrid_volatility_case",
]
