# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_execution_plan_v0.1.md.

from __future__ import annotations

import json
from math import prod
from typing import Any


def _limits(limits):
    from quantumbridge.serialization.limits import SerializationLimits
    if limits is None:return SerializationLimits()
    if not isinstance(limits, SerializationLimits):raise TypeError("JSON limits require SerializationLimits.")
    return limits


def _note_python_heap():
    from quantumbridge.execution.resource_context import current_resources
    scope = current_resources()
    if scope is not None:
        scope.ledger.unavailable("json_python_allocator_memory",
            "JSON byte/node/depth/element limits bound logical transport and structure, not decoder/list/string headers, allocator peak or total process RSS.")


def validate_payload_structure(payload, *, limits=None):
    """Bound decoded structure before numerical codecs allocate arrays."""
    from quantumbridge.serialization.errors import FormatValidationError
    limits = _limits(limits)
    _note_python_heap()
    pending, ancestors, nodes, numbers, text_bytes = [(payload, 0, False)], set(), 0, 0, 0
    while pending:
        value, depth, leaving = pending.pop()
        if leaving:
            ancestors.remove(id(value));continue
        nodes += 1
        limits.check("max_ast_nodes", nodes, format="json")
        limits.check("max_json_depth", depth, format="json")
        kind = type(value)
        if kind is str:
            limits.check("max_input_bytes", len(value), format="json")
            text_bytes += len(value.encode("utf-8"))
            limits.check("max_input_bytes", text_bytes, format="json")
        elif kind is dict or kind is list or kind is tuple:
            if id(value) in ancestors:raise FormatValidationError("Cyclic JSON data.", format="json")
            ancestors.add(id(value));pending.append((value, depth, True))
            limits.check("max_ast_nodes", nodes+len(value), format="json")
            if kind is dict:
                if any(not any(type(key) is allowed for allowed in (str, bool, int, float, type(None))) for key in value):
                    raise FormatValidationError("Unsupported JSON key type.", format="json")
                if type(value.get("saved_snapshots")) is dict:
                    limits.check("max_result_labels", len(value["saved_snapshots"]), format="json")
                if type(value.get("results")) is list:
                    limits.check("max_circuits", len(value["results"]), format="json")
                for key, child in value.items():
                    pending.append((key, depth+1, False));pending.append((child, depth+1, False))
            else:
                pending.extend((child, depth+1, False) for child in value)
        elif kind is type(None):continue
        elif any(kind is allowed for allowed in (bool, int, float, complex)):
            numbers += 1
        else:
            from quantumbridge.execution.array_reservations import numeric_shape
            shape = numeric_shape(value)
            numbers += prod(shape) if shape else 1
        limits.check("max_numeric_elements", numbers, format="json")
    return payload


def check_numeric_export(arrays, *, limits=None):
    """Logical exported scalar bytes, explicitly not Python heap or wire size."""
    import numpy as np
    limits, count = _limits(limits), 0
    for array in arrays:
        value = np.ndarray.view(array, np.ndarray)
        count += value.size*(2 if value.dtype.kind == "c" else 1)
        limits.check("max_numeric_elements", count, format="json")
    _note_python_heap()
    return 8*count


def to_json(payload: dict[str, Any], *, indent: int | None = None, limits=None, allow_nan=True) -> str:
    limits = _limits(limits)
    validate_payload_structure(payload, limits=limits)
    parts, size = [], 0
    for part in json.JSONEncoder(sort_keys=True, indent=indent, allow_nan=allow_nan).iterencode(payload):
        size += len(part.encode("utf-8"))
        limits.check("max_input_bytes", size, format="json")
        parts.append(part)
    return "".join(parts)


def from_json(text: str, *, limits=None) -> dict[str, Any]:
    limits = _limits(limits)
    if type(text) is not str and type(text) is not bytes and type(text) is not bytearray:
        raise TypeError("JSON input requires text or encoded bytes.")
    limits.check("max_input_bytes", len(text), format="json")
    limits.check("max_input_bytes", len(text.encode("utf-8")) if type(text) is str else len(text), format="json")
    _note_python_heap()
    try:payload = json.loads(text)
    except RecursionError as error:
        from quantumbridge.serialization.errors import ResourceLimitError
        raise ResourceLimitError("JSON decoder nesting exceeded its supported bound.", format="json") from error
    return validate_payload_structure(payload, limits=limits)
