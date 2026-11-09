# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
"""Stable, framework-independent derivative outcomes (protocol version 1)."""

from enum import IntEnum


class DerivativeExecutionStatus(IntEnum):
    SUCCESS = 0
    INVALID_INPUT = 1
    UNSUPPORTED = 2
    RESOURCE_LIMIT = 3
    NATIVE_RUNTIME_FAILURE = 4
    NUMERICAL_FAILURE = 5


class DerivativeExecutionError(RuntimeError):
    def __init__(self, status, message, *, stage="execution", derivative_kind=None,
                 device_method=None, failed_operation=None, receipt=None):
        self.status = DerivativeExecutionStatus(status)
        self.code = int(self.status)
        self.stage = stage
        self.derivative_kind = derivative_kind
        self.device_method = device_method
        self.failed_operation = failed_operation
        self.receipt = {} if receipt is None else receipt
        super().__init__(message)

    def to_dict(self):
        return {"protocol_version": 1, "code": self.code, "status": self.status.name,
                "stage": self.stage, "derivative_kind": self.derivative_kind,
                "device_method": self.device_method, "failed_operation": self.failed_operation,
                "message": str(self), "receipt": self.receipt}


class ResourceLimitError(DerivativeExecutionError, MemoryError):
    def __init__(self, category, estimated, limit, *, actual=0, stage="preflight", receipt=None):
        self.category, self.estimated, self.actual, self.limit = category, estimated, actual, limit
        super().__init__(DerivativeExecutionStatus.RESOURCE_LIMIT,
            f"Unified {category} budget exceeded: requested total {estimated}, limit {limit}.",
            stage=stage, receipt=receipt)

    def to_dict(self):
        return {**super().to_dict(), "category": self.category, "estimated": self.estimated,
                "actual": self.actual, "limit": self.limit}


def classify_derivative_error(error, **context):
    """Do not include arbitrary exception messages or input data in graph receipts."""
    from quantumbridge.core.classical import UnsupportedFeature
    from quantumbridge.serialization.errors import ResourceLimitError as SerializationResourceLimit
    while (type(error).__module__ == "quantumbridge.transforms.program"
           and type(error).__name__ == "TransformError" and error.__cause__ is not None):
        error = error.__cause__
    if isinstance(error, DerivativeExecutionError):
        for name, value in context.items():
            if value is not None:
                setattr(error, name, value)
        return error
    if isinstance(error, UnsupportedFeature):
        status, message = DerivativeExecutionStatus.UNSUPPORTED, "Unsupported native derivative path."
        context.setdefault("failed_operation", error.feature)
    elif isinstance(error, (MemoryError, SerializationResourceLimit)):
        status, message = DerivativeExecutionStatus.RESOURCE_LIMIT, "Native derivative resource limit exceeded."
    elif isinstance(error, FloatingPointError):
        status, message = DerivativeExecutionStatus.NUMERICAL_FAILURE, "Native derivative produced a nonfinite numerical result."
    elif isinstance(error, (ValueError, TypeError, IndexError, KeyError)) and context.get("stage", "validation") in {"validation", "preflight"}:
        status, message = DerivativeExecutionStatus.INVALID_INPUT, "Invalid native derivative input or shape."
    else:
        status, message = DerivativeExecutionStatus.NATIVE_RUNTIME_FAILURE, "Native derivative execution failed."
    return DerivativeExecutionError(status, message, **context)


def detach_status_error(error):
    """Status payloads carry safe fields, not traceback-owned native buffers.

    The exception-raising API keeps ordinary tracebacks. A returned status is
    deliberately not a deferred traceback/debugger retaining failed execution.
    """
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen: continue
        seen.add(id(current))
        pending.extend(item for item in (current.__cause__, current.__context__) if item is not None)
        current.__traceback__ = current.__cause__ = current.__context__ = None
