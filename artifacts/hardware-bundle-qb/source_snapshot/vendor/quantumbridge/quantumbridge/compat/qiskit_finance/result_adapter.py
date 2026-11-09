# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Qiskit Finance result schema adapter."""

import warnings

warnings.warn(
    "This adapter is a SCAFFOLD adapter. "
    "Full Qiskit Finance implementation is not yet available. "
    "This adapter provides result schema wrappers and an educational native "
    "portfolio optimization subset. "
    "Level 0: inventory only; Level 1: passthrough; Level 2: schema wrapper. "
    "Stage 9A portfolio support is not production finance or full parity.",
    UserWarning,
    stacklevel=2
)

from quantumbridge.compat.qiskit_finance.applications_adapter import ADAPTER
from quantumbridge.compat.qiskit_finance.portfolio_result import PortfolioOptimizationResult
from quantumbridge.schema import FinanceResult


def wrap_finance_result(obj, metadata=None) -> FinanceResult:
    """Wrap a Qiskit Finance result in QuantumBridge FinanceResult schema."""
    return ADAPTER.make_result(FinanceResult, obj, metadata=metadata)


def wrap_portfolio_optimization_result(result: PortfolioOptimizationResult, metadata=None) -> FinanceResult:
    """Wrap a portfolio optimization result in QuantumBridge FinanceResult schema."""

    if not isinstance(result, PortfolioOptimizationResult):
        raise TypeError("result must be a PortfolioOptimizationResult")
    combined_metadata = {
        "adapter": "qiskit_finance.portfolio_optimization",
        "path": result.path,
        "method": result.method,
        **dict(metadata or {}),
    }
    return wrap_finance_result(result.to_dict(), metadata=combined_metadata)
