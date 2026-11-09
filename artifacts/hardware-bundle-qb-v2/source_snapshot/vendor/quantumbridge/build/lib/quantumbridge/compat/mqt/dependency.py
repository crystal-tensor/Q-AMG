# This file is independently implemented for QuantumBridge SDK.
# No source code from MQT, IBM, or Qiskit was copied.
"""Optional MQT dependency detection helpers."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import Any

from .warnings import UPSTREAM_MQT_WARNING, mqt_warnings

MQT_PACKAGES = {
    "mqt-core": ("mqt.core", "mqt"),
    "mqt-ddsim": ("mqt.ddsim", "ddsim"),
    "mqt-qmap": ("mqt.qmap", "qmap"),
}


def dependency_available(package_name: str = "mqt") -> bool:
    module_name = _module_for_package(package_name)
    try:
        return find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError):
        return False


def get_upstream_version(package_name: str = "mqt") -> str | None:
    module_name = _module_for_package(package_name)
    try:
        module = import_module(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def validate_mqt_dependencies() -> dict[str, Any]:
    packages = {}
    for package_name, (module_name, _) in MQT_PACKAGES.items():
        try:
            available = find_spec(module_name) is not None
        except (ImportError, ModuleNotFoundError):
            available = False
        packages[package_name] = {
            "module": module_name,
            "available": available,
            "upstream_version": get_upstream_version(package_name) if available else None,
        }
    return {
        "ecosystem": "mqt",
        "packages": packages,
        "any_available": any(item["available"] for item in packages.values()),
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
        "warnings": mqt_warnings(UPSTREAM_MQT_WARNING),
    }


def _module_for_package(package_name: str) -> str:
    normalized = str(package_name).replace("_", "-").lower()
    if normalized in MQT_PACKAGES:
        return MQT_PACKAGES[normalized][0]
    if normalized in {"mqt", "mqt-core-like"}:
        return "mqt"
    return str(package_name).replace("-", "_")
