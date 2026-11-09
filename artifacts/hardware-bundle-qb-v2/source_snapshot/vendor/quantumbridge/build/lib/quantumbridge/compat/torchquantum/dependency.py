# This file is independently implemented for QuantumBridge SDK.
# No source code from TorchQuantum or PyTorch was copied.
"""Optional dependency checks for TorchQuantum-style compatibility."""

from __future__ import annotations

from importlib import import_module, metadata, util
from typing import Any


def dependency_available(package_name: str = "torchquantum") -> bool:
    try:
        return util.find_spec(package_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def get_upstream_version(package_name: str = "torchquantum") -> str | None:
    if not dependency_available(package_name):
        return None
    for dist_name in (package_name, package_name.replace("_", "-")):
        try:
            return metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            continue
    module = import_module(package_name)
    return getattr(module, "__version__", None)


def validate_torchquantum_dependencies() -> dict[str, Any]:
    available = dependency_available("torchquantum")
    return {
        "package": "torchquantum",
        "available": available,
        "version": get_upstream_version("torchquantum") if available else None,
        "required_by_default": False,
        "cloud_access": False,
        "token_read": False,
        "hardware_access": False,
    }
