# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Dependency discovery for optional ecosystem adapters.

Design source: docs/compat/full_ecosystem_coverage_policy_v0.1.md.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from types import ModuleType


def dependency_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def import_optional(module_name: str, extra_name: str) -> ModuleType:
    if not dependency_available(module_name):
        raise ImportError(
            "QuantumBridge optional ecosystem adapter requires "
            f"the upstream package for module {module_name!r}. "
            f"Install the QuantumBridge extra {extra_name!r} in a separate, "
            "compatible environment if needed."
        )
    return importlib.import_module(module_name)


def get_upstream_version(module_name: str, distribution_name: str | None = None) -> str | None:
    names = [name for name in (distribution_name, module_name.split(".")[0]) if name]
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    if not dependency_available(module_name):
        return None
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else None
