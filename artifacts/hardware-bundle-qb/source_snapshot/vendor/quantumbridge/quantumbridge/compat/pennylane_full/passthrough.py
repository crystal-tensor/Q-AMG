# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Generic PennyLane passthrough helpers."""

from __future__ import annotations

import inspect
from importlib import import_module
from typing import Any

from quantumbridge.compat.contracts import CapabilityLevel, UnsupportedCapability

from .dependency import DEPENDENCY_EXTRA, dependency_available
from .result_adapter import wrap_pennylane_result
from .warnings import unsupported


def get_public_object(module_name: str, object_name: str) -> Any | UnsupportedCapability:
    if not dependency_available():
        return unsupported("Optional PennyLane dependency is not installed.", CapabilityLevel.INVENTORY)
    try:
        module = import_module(module_name)
        if object_name.startswith("_") or not hasattr(module, object_name):
            return unsupported(f"{object_name!r} is not a public object in {module_name!r}.", CapabilityLevel.INVENTORY)
        return getattr(module, object_name)
    except Exception as exc:
        return unsupported(f"Could not resolve {module_name}.{object_name}: {type(exc).__name__}: {exc}")


def passthrough_call(module_name: str, object_name: str, *args: Any, **kwargs: Any):
    obj = get_public_object(module_name, object_name)
    if isinstance(obj, UnsupportedCapability):
        return obj
    if not callable(obj) or inspect.isclass(obj):
        return unsupported(f"{module_name}.{object_name} is not a function-like callable.", CapabilityLevel.INVENTORY)
    return wrap_pennylane_result(obj(*args, **kwargs), metadata={"module": module_name, "object": object_name})


def passthrough_class(module_name: str, class_name: str):
    obj = get_public_object(module_name, class_name)
    if isinstance(obj, UnsupportedCapability):
        return obj
    if not inspect.isclass(obj):
        return unsupported(f"{module_name}.{class_name} is not a class.", CapabilityLevel.INVENTORY)
    return obj


def safe_describe_object(module_name: str, object_name: str) -> dict:
    obj = get_public_object(module_name, object_name)
    if isinstance(obj, UnsupportedCapability):
        return obj.to_dict()
    return {
        "module": module_name,
        "name": object_name,
        "object_type": "class" if inspect.isclass(obj) else "callable" if callable(obj) else type(obj).__name__,
        "callable": callable(obj),
        "dependency_extra": DEPENDENCY_EXTRA,
        "doc_available": bool(getattr(obj, "__doc__", None)),
    }
