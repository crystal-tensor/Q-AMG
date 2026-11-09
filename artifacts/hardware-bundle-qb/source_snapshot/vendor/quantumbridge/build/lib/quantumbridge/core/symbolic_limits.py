# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Context-local limits applied before symbolic expansion and numeric powers."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
import math
import sys
from numbers import Integral, Real


class SymbolicResourceLimitError(ValueError):
    """A symbolic operation would exceed its configured construction budget."""


@dataclass(frozen=True)
class SymbolicExpressionLimits:
    max_nodes: int = 100000
    max_depth: int = 128
    max_parameters: int = 100000
    max_substitutions: int = 100000
    max_expansion_work: int = 1000000
    max_derivative_order: int = 16
    max_numeric_magnitude: float = sys.float_info.max
    max_integer_bits: int = 4096
    max_power_exponent: int = 4096
    max_serialized_bytes: int = 16777216

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == "max_numeric_magnitude":
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                    raise ValueError("Symbolic numeric magnitude must be positive and finite.")
            elif isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError("Symbolic limits must be positive integers.")

    def check(self, name, value):
        if value > getattr(self, name):
            raise SymbolicResourceLimitError(f"Symbolic expression exceeds {name}.")


_LIMITS = ContextVar("quantumbridge_symbolic_limits", default=SymbolicExpressionLimits())
_ACCOUNTS = ContextVar("quantumbridge_symbolic_work_accounts", default=())


def current_symbolic_limits():
    return _LIMITS.get()


def charge_symbolic_work(amount=1):
    for limits, account in _ACCOUNTS.get():
        account[0] += amount
        limits.check("max_expansion_work", account[0])


def symbolic_serialization_limits(limits=None):
    from quantumbridge.serialization.limits import SerializationLimits
    limits = limits or SerializationLimits()
    maximum = current_symbolic_limits().max_serialized_bytes
    return replace(limits, max_input_bytes=min(limits.max_input_bytes, maximum))


@contextmanager
def symbolic_expression_scope(limits):
    if not isinstance(limits, SymbolicExpressionLimits):
        raise TypeError("Expected SymbolicExpressionLimits.")
    accounts = _ACCOUNTS.get()
    if accounts:
        previous = current_symbolic_limits()
        limits = SymbolicExpressionLimits(**{name: min(value, getattr(previous, name))
                                             for name, value in vars(limits).items()})
    token = _LIMITS.set(limits)
    account_token = _ACCOUNTS.set(accounts+((limits, [0]),))
    try:
        yield limits
    finally:
        _ACCOUNTS.reset(account_token)
        _LIMITS.reset(token)
