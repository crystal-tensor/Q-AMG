# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Request-local lifetimes; nested plans inherit one resource account."""

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
from numbers import Integral
from copy import deepcopy
from inspect import signature

import numpy as np


_current = ContextVar("quantumbridge_derivative_resources", default=None)
_sample_ticket = ContextVar("quantumbridge_native_sample_ticket", default=None)


def current_resources():
    return _current.get()


def numerical_payload_bytes(value):
    """Logical array/scalar/text bytes, excluding Python container overhead."""
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, np.generic):
        return value.dtype.itemsize
    if isinstance(value, (float, int, complex, bool)):
        return 16 if isinstance(value, complex) else 8
    if isinstance(value, (str, bytes)):
        return len(value.encode("utf-8") if isinstance(value, str) else value)
    if isinstance(value, dict):
        return sum(numerical_payload_bytes(k)+numerical_payload_bytes(v) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return sum(numerical_payload_bytes(item) for item in value)
    return 0


def copy_execution_payload(value):
    """Reserve retained receipt payload before copying, never claim object RSS."""
    resources = current_resources()
    if resources is not None:
        ticket = resources.retain("retained_receipt_payload", numerical_payload_bytes(value))
        resources.ledger.unavailable("python_container_allocator_memory",
            "Receipt accounting covers logical numerical/text payload, not Python objects, container headers or allocator peak.")
    result = deepcopy(value)
    if resources is not None:
        ticket.observe(numerical_payload_bytes(result))
    return result


@contextmanager
def measurement_shots(ledger, shots):
    """Reserve one measurement's sampling before dispatch, then release unused work."""
    with ledger.reserve("measurement_shots", shots=shots) as ticket:
        token = _sample_ticket.set(ticket)
        try:
            yield ticket
        finally:
            _sample_ticket.reset(token)


def record_completed_native_shot():
    """Record one explicitly completed trajectory, not unknown partial work."""
    ticket = _sample_ticket.get()
    if ticket is not None: ticket.consume(shots=1)


def record_completed_native_shots(shots):
    """A completed categorical buffer remains spent if later formatting fails."""
    ticket = _sample_ticket.get()
    if ticket is not None:ticket.consume(shots=shots)


def result_shots(result):
    return sum((result.counts_data or {}).values())


def meter_native_samples(function=None, *, result_count=len):
    """Charge completed buffers before any downstream statistics can fail."""
    if function is None:
        return lambda function: meter_native_samples(function, result_count=result_count)
    parameters = signature(function)
    @wraps(function)
    def sampled(self, *args, **kwargs):
        resources = current_resources()
        if resources is None:
            return function(self, *args, **kwargs)
        bound = parameters.bind(self, *args, **kwargs)
        bound.apply_defaults()
        shots = bound.arguments["shots"]
        if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 0:
            raise ValueError("Native sampling shots must be nonnegative integers.")
        def execute(ticket):
            if shots > ticket.amounts["shots"]:
                from .derivative_status import ResourceLimitError
                raise ResourceLimitError("shots", shots, ticket.amounts["shots"],
                    stage="sample_dispatch", receipt=resources.ledger.receipt())
            before = ticket.amounts["shots"]
            try:
                result = function(self, *args, **kwargs)
            except Exception:
                resources.ledger.unavailable("failed_native_sample_completion",
                    "Explicitly completed trajectories remain consumed; additional internal partial work in the failed trajectory/buffer is unavailable, not zero.")
                raise
            count = result_count(result)
            completed = before-ticket.amounts["shots"]
            if isinstance(count, bool) or not isinstance(count, Integral) or not completed <= count <= shots:
                raise RuntimeError("Native sample completion differs from its declared trajectory/buffer count.")
            ticket.consume(shots=count-completed)
            return result
        ticket = _sample_ticket.get()
        if ticket is not None:
            return execute(ticket)
        with measurement_shots(resources.ledger, shots) as ticket:
            return execute(ticket)
    return sampled


class ResourceScope:
    def __init__(self, ledger, *, new_request=False):
        self.ledger, self.stack = ledger, ExitStack()
        self.new_request = new_request

    def __enter__(self):
        parent = _current.get()
        self.request_token = parent.request_token if (not self.new_request and parent is not None
            and parent.ledger is self.ledger) else object()
        self.token = _current.set(self)
        return self

    def __exit__(self, *exception):
        try:
            self.stack.__exit__(*exception)
        finally:
            _current.reset(self.token)

    def retain(self, category, memory_bytes):
        return self.stack.enter_context(self.ledger.reserve(category, memory_bytes=memory_bytes))
